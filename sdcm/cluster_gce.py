# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright (c) 2020 ScyllaDB
import json
import logging
import os
import time
from collections.abc import Callable
from functools import cached_property, cache
from typing import Dict, ParamSpec, TypeVar

import tenacity
from libcloud.common.google import GoogleBaseError, ResourceNotFoundError, InvalidRequestError
from libcloud.compute.base import NodeDriver

from sdcm import cluster
from sdcm.sct_events import Severity
from sdcm.sct_events.gce_events import GceInstanceEvent
from sdcm.utils.gce_utils import GceLoggingClient
from sdcm.provision.helpers.cloud_init import get_cloud_init_config
from sdcm.keystore import pub_key_from_private_key_file
from sdcm.provision import GCEProvisioner
from sdcm.provision.gce.data_disks import ScratchDisk, PersistentStandardDisk, GCE_DISK_TYPES
from sdcm.provision.provisioner import InstanceDefinition, VmArch, DataDisk, PricingModel, VmInstance, ProvisionError
from sdcm.sct_events.system import SpotTerminationEvent
from sdcm.utils.common import list_instances_gce, gce_meta_to_dict
from sdcm.utils.decorators import retrying
from sdcm.wait import exponential_retry

SPOT_TERMINATION_CHECK_DELAY = 5 * 60

LOGGER = logging.getLogger(__name__)

P = ParamSpec("P")  # pylint: disable=invalid-name
R = TypeVar("R")  # pylint: disable=invalid-name


class CreateGCENodeError(Exception):
    pass


class GCENode(cluster.BaseNode):

    """
    Wraps GCE instances, so that we can also control the instance through SSH.
    """

    log = LOGGER

    def __init__(self, vm_instance: VmInstance, gce_service, credentials, parent_cluster,  # pylint: disable=too-many-arguments
                 node_prefix='node', node_index=1, gce_image_username='root',
                 base_logdir=None, dc_idx=0):
        name = f"{node_prefix}-{dc_idx}-{node_index}".lower()
        self.node_index = node_index
        self._instance = vm_instance
        self._gce_service = gce_service
        self._gce_logging_client = GceLoggingClient(name, self._instance.node.extra["zone"].name)
        self._last_logs_fetch_time = 0.0
        ssh_login_info = {'hostname': None,
                          'user': gce_image_username,
                          'key_file': credentials.key_file,
                          'extra_ssh_options': '-tt'}
        super().__init__(name=name,
                         parent_cluster=parent_cluster,
                         ssh_login_info=ssh_login_info,
                         base_logdir=base_logdir,
                         node_prefix=node_prefix,
                         dc_idx=dc_idx)

    @staticmethod
    def is_gce() -> bool:
        return True

    def init(self):
        self._wait_public_ip()

        # sleep 10 seconds for waiting users are added to system
        # related issue: https://github.com/scylladb/scylla-cluster-tests/issues/1121
        time.sleep(10)

        super().init()

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**super().tags,
                "NodeIndex": str(self.node_index), }

    def _set_keep_alive(self) -> bool:
        return self._instance_wait_safe(self._instance.add_tags, {"keep": "alive"}) and \
            super()._set_keep_alive()

    def _instance_wait_safe(self, instance_method: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return exponential_retry(func=lambda: instance_method(*args, **kwargs), logger=self.log)
        except tenacity.RetryError:
            raise cluster.NodeError(
                f"Timeout while running '{instance_method.__name__}' method on GCE instance '{self._instance.name}'"
            ) from None

    def _refresh_instance_state(self):
        node_name = self._instance.name
        node = self._instance_wait_safe(self._gce_service.ex_get_node, node_name)
        self._instance.private_ip_address = node.private_ips
        self._instance.public_ip_address = node.public_ips
        return self._instance.public_ip_address, self._instance.private_ip_address

    @property
    def region(self):
        return self._instance.region

    def set_hostname(self):
        self.log.debug("Hostname for node %s left as is", self.name)

    @property
    def is_spot(self):
        return self._instance.pricing_model == PricingModel.SPOT

    def check_spot_termination(self):
        """Check if a spot instance termination was initiated by the cloud.

        There are few different methods how to detect this event in GCE:

            https://cloud.google.com/compute/docs/instances/create-start-preemptible-instance#detecting_if_an_instance_was_preempted

        We moved from internal metadata because sometimes it was missing the spot. Moved to GCE logging along with providing more
        important notifications about node state.
        """
        since = self._last_logs_fetch_time
        self._last_logs_fetch_time = time.time()
        try:
            for entry in self._gce_logging_client.get_system_events(from_=since, until=self._last_logs_fetch_time):
                match entry['protoPayload']['methodName']:
                    case "compute.instances.preempted":
                        self.log.warning('Got spot termination notification from GCE')
                        SpotTerminationEvent(node=self, message='Instance was preempted.').publish()
                    case "compute.instances.automaticRestart" | "compute.instances.hostError":
                        GceInstanceEvent(entry).publish()
                    case _:
                        GceInstanceEvent(entry, severity=Severity.WARNING).publish()
        except Exception as details:  # pylint: disable=broad-except
            self.log.warning('Error during getting spot termination notification %s', details)
            self._last_logs_fetch_time = since
        return SPOT_TERMINATION_CHECK_DELAY

    def restart(self):
        # When using local_ssd disks in GCE, there is no option to Stop and Start an instance.
        # So, for now we will keep restart the same as hard reboot.
        self._instance_wait_safe(self._instance.reboot)

    def hard_reboot(self):
        self._instance_wait_safe(self._instance.reboot)

    def _safe_destroy(self):
        self._instance.terminate()

    def destroy(self):
        self.stop_task_threads()
        self.wait_till_tasks_threads_are_stopped()
        self._instance_wait_safe(self._safe_destroy())
        super().destroy()

    def get_console_output(self):
        # TODO adding console output from instance on GCE
        self.log.warning('Method is not implemented for GCENode')
        return NotImplemented

    def get_console_screenshot(self):
        # TODO adding console output from instance on GCE
        self.log.warning('Method is not implemented for GCENode')
        return NotImplemented

    @cache
    def _get_ipv6_ip_address(self):
        self.log.warning('On GCE, VPC networks only support IPv4 unicast traffic. '
                         'They do not support IPv6 traffic within the network.')
        return NotImplemented

    @property
    def image(self):
        return self._instance.image


class GCECluster(cluster.BaseCluster):  # pylint: disable=too-many-instance-attributes,abstract-method

    """
    Cluster of Node objects, started on GCE (Google Compute Engine).
    """

    # pylint: disable=too-many-arguments
    def __init__(self, gce_image, gce_image_type, gce_image_size, gce_network, services: list[NodeDriver],
                 credentials, provisioners: list[GCEProvisioner], cluster_uuid=None, gce_instance_type='n1-standard-1',
                 gce_region_names=None, gce_n_local_ssd=1, gce_image_username='root', cluster_prefix='cluster',
                 node_prefix='node', n_nodes=3, add_disks=None, params=None, node_type=None,
                 service_accounts=None, add_nodes=True):

        # pylint: disable=too-many-locals
        self._gce_image = gce_image
        self._gce_image_type = gce_image_type
        self._gce_image_size = gce_image_size
        self._gce_network = gce_network
        self._gce_services = services
        self._credentials = credentials
        self._gce_instance_type = gce_instance_type
        self._gce_image_username = gce_image_username
        self._gce_region_names = gce_region_names
        self._gce_n_local_ssd = int(gce_n_local_ssd) if gce_n_local_ssd else 0
        self._add_disks = add_disks
        self._service_accounts = service_accounts
        # the full node prefix will contain unique uuid, so use this for search of existing nodes
        self._node_prefix = node_prefix
        self.provisioners: list[GCEProvisioner] = provisioners
        super().__init__(cluster_uuid=cluster_uuid,
                         cluster_prefix=cluster_prefix,
                         node_prefix=node_prefix,
                         n_nodes=n_nodes,
                         params=params,
                         # services=services,
                         region_names=gce_region_names,
                         node_type=node_type,
                         add_nodes=add_nodes)
        self.log.debug("GCECluster constructor")

    def __str__(self):
        identifier = 'GCE Cluster %s | ' % self.name
        identifier += 'Image: %s | ' % os.path.basename(self._gce_image)
        identifier += 'Root Disk: %s %s GB | ' % (self._gce_image_type, self._gce_image_size)
        if self._gce_n_local_ssd:
            identifier += 'Local SSD: %s | ' % self._gce_n_local_ssd
        if self._add_disks:
            for disk_type, disk_size in self._add_disks.items():
                if int(disk_size):
                    identifier += '%s: %s | ' % (disk_type, disk_size)
        identifier += 'Type: %s' % self._gce_instance_type
        return identifier

    def _get_provisioner_for_dc(self, dc_idx: int) -> GCEProvisioner:
        gce_service = self._gce_services[dc_idx]
        found_provisioners = [item for item in self.provisioners if gce_service.region.name in item.region]
        if len(found_provisioners) == 1:
            provisioner = found_provisioners[0]
            return provisioner
        else:
            raise ProvisionError(f"Did not find a provisioner for the given region:"
                                 f"\ngce service region name: {gce_service.region.name}\n"
                                 f"provisioners list: {[self.provisioners]}")

    def _prepare_user_data(self):
        if not self.params.get("use_preinstalled_scylla"):
            return get_cloud_init_config()
        else:
            return json.dumps(dict(scylla_yaml=dict(cluster_name=self.name),
                                   start_scylla_on_first_boot=False,
                                   raid_level=self.params.get("raid_level")))

    def _create_instance(self, node_index, dc_idx, pricing_model: PricingModel) -> VmInstance:
        # if size of disk is larget than 80G, then
        # change the timeout of job completion to default * 3.

        gce_job_default_timeout = None
        if self._gce_image_size and int(self._gce_image_size) > 80:
            gce_job_default_timeout = self._gce_services[dc_idx].connection.timeout
            self._gce_services[dc_idx].connection.timeout = gce_job_default_timeout * 3
            self.log.info("Job complete timeout is set to %ss" %
                          self._gce_services[dc_idx].connection.timeout)

        data_disks = []

        for _ in range(self._gce_n_local_ssd):
            data_disks.append(ScratchDisk(type=GCE_DISK_TYPES.SCRATCH))

        for add_disk_type, add_disk_size in self._add_disks.items():
            if add_disk_size > 0:
                data_disks.append(PersistentStandardDisk(type=add_disk_type, size=add_disk_size))
            else:
                self.log.debug("Got additional disk with disk_size <= 0. Skipping adding the disk.")

        instance_definition = self.create_instance_definition(data_disks, dc_idx, node_index)

        pricing_model = PricingModel.SPOT

        instance = self._create_node_with_retries(dc_idx=dc_idx,
                                                  instance_definition=instance_definition,
                                                  pricing_model=pricing_model)

        self.log.info('Created %s instance %s', pricing_model.value, instance)
        try:
            instance.add_tags(instance_definition.tags)
        except InvalidRequestError as exc:
            self.log.warning(f"Unable to set tags as labels due to {exc}")

        if gce_job_default_timeout:
            self.log.info('Restore default job timeout %s' % gce_job_default_timeout)
            self._gce_services[dc_idx].connection.timeout = gce_job_default_timeout
        return instance

    def create_instance_definition(self, data_disks: list[DataDisk], dc_idx, node_index):
        instance_definition = InstanceDefinition(
            instance_index=node_index,
            name=f"{self.node_prefix}-{dc_idx}-{node_index}".lower(),
            image_id=self._gce_image,
            type=self._gce_instance_type,
            user_name=self.params.get("gce_image_username"),
            # TODO: can we switch to SSHKey wrapped in NodeAuthSSHKey
            ssh_key=pub_key_from_private_key_file(self.params.get("user_credentials_path")),
            tags=dict((k.lower(), v.lower().replace('.', '-')) for k, v in self.tags.items()),
            arch=VmArch.X86,  # TODO: need to switch between architectures here
            root_disk_size=self._gce_image_size,
            data_disks=data_disks,
            user_data=self._prepare_user_data(),  # TODO: add user data object list for GCE
            startup_script=self.test_config.get_startup_script()
        )

        LOGGER.info("Created instance definition:\n%s", instance_definition)
        return instance_definition

    @retrying(n=3,
              sleep_time=900,
              allowed_exceptions=(GoogleBaseError, ResourceNotFoundError),
              message="Retrying to create a GCE node...",
              raise_on_exceeded=True)
    def _create_node_with_retries(self,
                                  dc_idx: int,
                                  instance_definition: InstanceDefinition,
                                  pricing_model: PricingModel) -> VmInstance:
        provisioner = self._get_provisioner_for_dc(dc_idx=dc_idx)

        try:
            instance = None
            instance = provisioner.get_or_create_instance(
                definition=instance_definition,
                pricing_model=pricing_model
            )
            return instance
        except GoogleBaseError as gbe:
            if not pricing_model == pricing_model.SPOT:
                raise

            #  attempt to destroy if node did not start due to preemption
            if "Instance failed to start due to preemption" in str(gbe):
                try:
                    provisioner.terminate_instance(name=instance_definition.name)
                except ResourceNotFoundError:
                    LOGGER.warning("Attempted to destroy node: {name: %s}, but could not find it in region %s. "
                                   "Possibly the node was destroyed already.",
                                   instance_definition.name, provisioner.region)

            # retry with ON_DEMAND if SPOT failed
            pricing_model = PricingModel.ON_DEMAND

            raise gbe

    def _create_instances(self, count, dc_idx=0) -> list[VmInstance]:
        pricing_model = PricingModel.SPOT if self.instance_provision == PricingModel.SPOT \
            else PricingModel.ON_DEMAND  # TODO: generalize for all pricing models
        instances = []
        for node_index in range(self._node_index + 1, self._node_index + count + 1):
            instances.append(self._create_instance(node_index=node_index, dc_idx=dc_idx, pricing_model=pricing_model))
        return instances

    def _destroy_instance(self, name: str, dc_idx: int) -> bool:
        provisioner = self._get_provisioner_for_dc(dc_idx)
        return provisioner.terminate_instance(name=name)

    def _get_instances_by_prefix(self, dc_idx: int = 0) -> list[VmInstance]:
        provisioner = self._get_provisioner_for_dc(dc_idx)
        instances_list = provisioner.list_instances()
        # instances_by_zone = self._gce_services[dc_idx].list_nodes(ex_zone=self._gce_region_names[dc_idx])
        return instances_list

    def _get_instances_by_name(self, name: str, dc_idx: int = 0) -> VmInstance:
        # instances_by_zone = self._gce_services[dc_idx].list_nodes(ex_zone=self._gce_region_names[dc_idx])
        provisioner = self._get_provisioner_for_dc(dc_idx)
        instances_list = provisioner.list_instances()
        found = [instance for instance in instances_list if instance.name == name]
        return found[0] if found else None

    def _get_instances(self, dc_idx):
        # TODO: rework to be VmInstance-compatible
        test_id = self.test_config.test_id()
        if not test_id:
            raise ValueError("test_id should be configured for using reuse_cluster")
        instances_by_nodetype = list_instances_gce(tags_dict={'TestId': test_id, 'NodeType': self.node_type})
        instances_by_zone = self._get_instances_by_prefix(dc_idx)
        instances = []
        attr_name = 'public_ips' if self._node_public_ips else 'private_ips'
        for node_zone in instances_by_zone:
            # Filter nodes by zone and by ip addresses
            if not getattr(node_zone, attr_name):
                continue
            for node_nodetype in instances_by_nodetype:
                if node_zone.uuid == node_nodetype.uuid:
                    instances.append(node_zone)

        def sort_by_index(node):
            metadata = gce_meta_to_dict(node.extra['metadata'])
            return metadata.get('NodeIndex', 0)

        instances = sorted(instances, key=sort_by_index)
        return instances

    def _create_node(self, instance: VmInstance, node_index, dc_idx):
        try:
            node = GCENode(vm_instance=instance,
                           gce_service=self._gce_services[dc_idx],
                           credentials=self._credentials[0],
                           parent_cluster=self,
                           gce_image_username=self._gce_image_username,
                           node_prefix=self.node_prefix,
                           node_index=node_index,
                           base_logdir=self.logdir,
                           dc_idx=dc_idx)
            node.init()
            return node
        except Exception as ex:
            raise CreateGCENodeError('Failed to create node: %s' % ex) from ex

    # pylint: disable=too-many-arguments
    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False):
        if count <= 0:
            return []
        self.log.info("Adding nodes to cluster")
        nodes = []
        if self.test_config.REUSE_CLUSTER:
            instances = self._get_instances(dc_idx)
            if not instances:
                raise RuntimeError("No nodes found for testId %s " % (self.test_config.test_id(),))
        else:
            instances = self._create_instances(count, dc_idx)

        self.log.debug('instances: %s', instances)
        if instances:
            self.log.debug('GCE instance extra info: %s', instances[0].node.extra)
        for node_index, instance in enumerate(instances, start=self._node_index + 1):
            node = self._create_node(instance, node_index, dc_idx)
            nodes.append(node)
            self.nodes.append(node)
            self.log.info("Added node: %s", node.name)
            node.enable_auto_bootstrap = enable_auto_bootstrap

        self._node_index += count
        self.log.info('added nodes: %s', nodes)
        return nodes


class ScyllaGCECluster(cluster.BaseScyllaCluster, GCECluster):

    def __init__(self, gce_image, gce_image_type, gce_image_size, gce_network, services, credentials,  # pylint: disable=too-many-arguments
                 provisioners: list[GCEProvisioner], gce_instance_type='n1-standard-1', gce_n_local_ssd=1,
                 gce_image_username='centos',
                 user_prefix=None, n_nodes=3, add_disks=None, params=None, gce_datacenter=None, service_accounts=None):
        # pylint: disable=too-many-locals
        # We have to pass the cluster name in advance in user_data
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'db-cluster')
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'db-node')
        super().__init__(
            gce_image=gce_image,
            gce_image_type=gce_image_type,
            gce_image_size=gce_image_size,
            gce_n_local_ssd=gce_n_local_ssd,
            gce_network=gce_network,
            gce_instance_type=gce_instance_type,
            gce_image_username=gce_image_username,
            services=services,
            provisioners=provisioners,
            credentials=credentials,
            cluster_prefix=cluster_prefix,
            node_prefix=node_prefix,
            n_nodes=n_nodes,
            add_disks=add_disks,
            params=params,
            gce_region_names=gce_datacenter,
            node_type='scylla-db',
            service_accounts=service_accounts,
        )
        self.version = '2.1'

    @staticmethod
    def _wait_for_preinstalled_scylla(node):
        node.wait_for_machine_image_configured()


class LoaderSetGCE(cluster.BaseLoaderSet, GCECluster):

    def __init__(self, gce_image, gce_image_type, gce_image_size, gce_network, service, credentials,  # pylint: disable=too-many-arguments
                 provisioners: list[GCEProvisioner], gce_instance_type='n1-standard-1', gce_n_local_ssd=1,
                 gce_image_username='centos',
                 user_prefix=None, n_nodes=10, add_disks=None, params=None, gce_datacenter=None):
        # pylint: disable=too-many-locals
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'loader-node')
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'loader-set')
        cluster.BaseLoaderSet.__init__(self,
                                       params=params)
        GCECluster.__init__(self,
                            gce_image=gce_image,
                            gce_network=gce_network,
                            gce_image_type=gce_image_type,
                            gce_image_size=gce_image_size,
                            gce_n_local_ssd=gce_n_local_ssd,
                            gce_instance_type=gce_instance_type,
                            gce_image_username=gce_image_username,
                            services=service,
                            provisioners=provisioners,
                            credentials=credentials,
                            cluster_prefix=cluster_prefix,
                            node_prefix=node_prefix,
                            n_nodes=n_nodes,
                            add_disks=add_disks,
                            params=params,
                            node_type='loader',
                            gce_region_names=gce_datacenter
                            )


class MonitorSetGCE(cluster.BaseMonitorSet, GCECluster):

    def __init__(self, gce_image, gce_image_type, gce_image_size, gce_network, service, credentials,  # pylint: disable=too-many-arguments
                 provisioners: list[GCEProvisioner], gce_instance_type='n1-standard-1', gce_n_local_ssd=1,
                 gce_image_username='centos', user_prefix=None, n_nodes=1,
                 targets=None, add_disks=None, params=None, gce_datacenter=None,
                 add_nodes=True, monitor_id=None):
        # pylint: disable=too-many-locals
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-node')
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-set')

        targets = targets if targets else {}
        cluster.BaseMonitorSet.__init__(
            self, targets=targets, params=params, monitor_id=monitor_id,
        )
        GCECluster.__init__(self,
                            gce_image=gce_image,
                            gce_image_type=gce_image_type,
                            gce_image_size=gce_image_size,
                            gce_n_local_ssd=gce_n_local_ssd,
                            gce_network=gce_network,
                            gce_instance_type=gce_instance_type,
                            gce_image_username=gce_image_username,
                            services=service,
                            provisioners=provisioners,
                            credentials=credentials,
                            cluster_prefix=cluster_prefix,
                            node_prefix=node_prefix,
                            n_nodes=n_nodes,
                            add_disks=add_disks,
                            params=params,
                            node_type='monitor',
                            gce_region_names=gce_datacenter,
                            add_nodes=add_nodes,
                            )
