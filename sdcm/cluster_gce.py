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

import os
import re
import time
import logging
from typing import Dict, Any, Optional, ParamSpec, TypeVar
from textwrap import dedent
from functools import cached_property, cache
from collections.abc import Callable

import tenacity
import google.api_core.exceptions
from google.cloud import compute_v1

from sdcm import cluster
from sdcm.provision.network_configuration import ssh_connection_ip_type
from sdcm.provision.helpers.cloud_init import wait_cloud_init_completes
from sdcm.sct_events import Severity
from sdcm.sct_events.gce_events import GceInstanceEvent
from sdcm.utils.gce_utils import (
    GceLoggingClient,
    create_instance,
    disk_from_image,
    get_gce_compute_disks_client,
    wait_for_extended_operation,
    gce_private_addresses,
    gce_public_addresses,
    random_zone,
    gce_set_labels,
)
from sdcm.wait import exponential_retry
from sdcm.keystore import pub_key_from_private_key_file
from sdcm.sct_events.system import SpotTerminationEvent
from sdcm.utils.common import list_instances_gce, gce_meta_to_dict
from sdcm.utils.decorators import retrying


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

    def __init__(self, gce_instance: compute_v1.Instance,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 gce_service: compute_v1.InstancesClient,
                 credentials,
                 parent_cluster,
                 node_prefix: str = 'node', node_index: int = 1,
                 gce_image_username: str = 'root',
                 base_logdir: str = None,
                 dc_idx: int = 0,
                 rack: int = 0,
                 gce_project: str = None,
                 shard_num: Optional[int] = None):
        name = f"{node_prefix}-{dc_idx}-{node_index}".lower()
        self.node_index = node_index
        self.project = gce_project
        self._instance = gce_instance
        self._gce_service = gce_service
        self._gce_logging_client = GceLoggingClient(instance_name=name, zone=self.zone)
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
                         dc_idx=dc_idx,
                         rack=rack,
                         shard_num=shard_num)

    def refresh_network_interfaces_info(self):
        pass

    @staticmethod
    def is_gce() -> bool:
        return True

    def init(self):
        self._wait_public_ip()

        # sleep 10 seconds for waiting users are added to system
        # related issue: https://github.com/scylladb/scylla-cluster-tests/issues/1121
        time.sleep(10)

        super().init()

    def wait_for_cloud_init(self):
        if self.remoter.sudo("bash -c 'command -v cloud-init'", ignore_status=True).ok:
            wait_cloud_init_completes(self.remoter, self)

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**super().tags,
                "NodeIndex": str(self.node_index), }

    def _set_keep_alive(self) -> bool:
        return gce_set_labels(instances_client=self._gce_service,
                              instance=self._instance,
                              new_labels={"keep": "alive"},
                              project=self.project,
                              zone=self.zone) and \
            super()._set_keep_alive()

    def _instance_wait_safe(self, instance_method: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return exponential_retry(func=lambda: instance_method(*args, **kwargs), logger=self.log)
        except tenacity.RetryError:
            raise cluster.NodeError(
                f"Timeout while running '{instance_method.__name__}' method on GCE instance '{self._instance.id}'"
            ) from None

    def _refresh_instance_state(self):
        instance = self._instance_wait_safe(self._gce_service.get,
                                            project=self.project,
                                            zone=self.zone,
                                            instance=self._instance.name)
        self._instance = instance

        ip_tuple = (gce_public_addresses(instance), gce_private_addresses(instance))
        return ip_tuple

    @property
    def network_interfaces(self):
        pass

    @property
    def vm_region(self):
        return self._instance.zone.split('/')[-1][:-2]

    @property
    def zone(self):
        return self._instance.zone.split('/')[-1]

    def set_hostname(self):
        self.log.debug("Hostname for node %s left as is", self.name)

    @property
    def is_spot(self):
        return self._instance.scheduling.preemptible

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
        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.warning('Error during getting spot termination notification %s', details)
            self._last_logs_fetch_time = since
        return SPOT_TERMINATION_CHECK_DELAY

    def restart(self):
        # When using local_ssd disks in GCE, there is no option to Stop and Start an instance.
        # So, for now we will keep restart the same as hard reboot.
        self._instance_wait_safe(self._gce_service.reset, instance=self._instance.name,
                                 project=self.project, zone=self.zone)

    def hard_reboot(self):
        self._instance_wait_safe(self._gce_service.reset, instance=self._instance.name,
                                 project=self.project, zone=self.zone)

    def _safe_destroy(self):
        try:
            operation = self._gce_service.delete(instance=self._instance.name,
                                                 project=self.project, zone=self.zone)
            wait_for_extended_operation(operation, "Wait for instance deletion")
        except google.api_core.exceptions.NotFound:
            self.log.exception("Instance doesn't exist, skip destroy")

    def destroy(self):
        self.stop_task_threads()
        self.wait_till_tasks_threads_are_stopped()
        self._instance_wait_safe(self._safe_destroy)
        super().destroy()

    def get_console_output(self):
        # TODO adding console output from instance on GCE
        self.log.warning('Method is not implemented for GCENode')
        return ''

    def get_console_screenshot(self):
        # TODO adding console output from instance on GCE
        self.log.warning('Method is not implemented for GCENode')
        return b''

    @cache
    def _get_ipv6_ip_address(self):
        self.log.warning('On GCE, VPC networks only support IPv4 unicast traffic. '
                         'They do not support IPv6 traffic within the network.')
        return ""

    @cached_property
    def image(self):
        disk_client, _ = get_gce_compute_disks_client()
        disk = disk_client.get(disk=self._instance.disks[0].source.split(
            '/')[-1], project=self.project, zone=self.zone)
        return disk.source_image


class GCECluster(cluster.BaseCluster):  # pylint: disable=too-many-instance-attributes,abstract-method

    """
    Cluster of Node objects, started on GCE (Google Compute Engine).
    """
    _gce_service: compute_v1.InstancesClient

    def __init__(self, gce_image, gce_image_type, gce_image_size, gce_network, gce_service, credentials,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 cluster_uuid=None, gce_instance_type='n2-standard-1', gce_region_names=None,
                 gce_n_local_ssd=1, gce_image_username='root', cluster_prefix='cluster',
                 node_prefix='node', n_nodes=3, add_disks=None, params=None, node_type=None,
                 service_accounts=None, add_nodes=True, nodes_smp=[]):

        # pylint: disable=too-many-locals
        self._gce_image = gce_image
        self._gce_image_type = gce_image_type
        self._gce_image_size = gce_image_size
        self._gce_network = gce_network
        self._gce_service, info = gce_service
        self.project = info['project_id']
        self._credentials = credentials
        self._gce_instance_type = gce_instance_type
        self._gce_image_username = gce_image_username
        self._gce_zone_names: list[str] = [f'{region}-{random_zone(region)}' for region in gce_region_names]
        self._gce_n_local_ssd = int(gce_n_local_ssd) if gce_n_local_ssd else 0
        self._add_disks = add_disks
        self._service_accounts = service_accounts
        # the full node prefix will contain unique uuid, so use this for search of existing nodes
        self._node_prefix = node_prefix.lower()
        super().__init__(cluster_uuid=cluster_uuid,
                         cluster_prefix=cluster_prefix,
                         node_prefix=node_prefix,
                         n_nodes=n_nodes,
                         params=params,
                         region_names=gce_region_names,
                         node_type=node_type,
                         add_nodes=add_nodes,
                         nodes_smp=nodes_smp)
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

    def _get_disk_url(self, disk_type='pd-standard', dc_idx=0):
        return "projects/%s/zones/%s/diskTypes/%s" % (self.project, self._gce_zone_names[dc_idx], disk_type)

    def _get_root_disk_struct(self, name, disk_type='pd-standard', dc_idx=0):
        device_name = '%s-root-%s' % (name, disk_type)
        return disk_from_image(disk_type=self._get_disk_url(disk_type, dc_idx=dc_idx),
                               disk_size_gb=self._gce_image_size,
                               boot=True,
                               source_image=self._gce_image,
                               auto_delete=True,
                               type_="PERSISTENT",
                               device_name=device_name,
                               )

    def _get_local_ssd_disk_struct(self, name, index, interface='NVME', dc_idx=0):
        device_name = '%s-data-local-ssd-%s' % (name, index)
        return disk_from_image(disk_type=self._get_disk_url('local-ssd', dc_idx=dc_idx),
                               boot=False,
                               auto_delete=True,
                               type_="SCRATCH",
                               device_name=device_name,
                               interface=interface,
                               )

    def _get_persistent_disk_struct(self, name, disk_size, disk_type='pd-ssd', dc_idx=0):
        device_name = '%s-data-%s' % (name, disk_type)
        return disk_from_image(disk_type=self._get_disk_url(disk_type, dc_idx=dc_idx),
                               disk_size_gb=disk_size,
                               boot=False,
                               auto_delete=True,
                               type_="PERSISTENT",
                               device_name=device_name,
                               )

    def _get_disks_struct(self, name, dc_idx):

        gce_disk_struct = [self._get_root_disk_struct(name=name,
                                                      disk_type=self._gce_image_type,
                                                      dc_idx=dc_idx)]
        for i in range(self._gce_n_local_ssd):
            gce_disk_struct.append(self._get_local_ssd_disk_struct(name=name, index=i, dc_idx=dc_idx))
        if self._add_disks:
            for disk_type, disk_size in self._add_disks.items():
                disk_size = int(disk_size)  # noqa: PLW2901
                if disk_size:
                    gce_disk_struct.append(self._get_persistent_disk_struct(name=name, disk_size=disk_size,
                                                                            disk_type=disk_type, dc_idx=dc_idx))
        self.log.debug(gce_disk_struct)
        return gce_disk_struct

    def _create_instance(self, node_index, dc_idx, spot=False, enable_auto_bootstrap=False, instance_type=None):  # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        def set_tags_as_labels(_instance: compute_v1.Instance):
            self.log.debug(f"Expected tags are {self.tags}")
            # https://cloud.google.com/resource-manager/docs/tags/tags-creating-and-managing#adding_tag_values
            regex_key = re.compile(r'[^a-z0-9-_]')

            def to_short_name(value):
                return regex_key.sub('_', value.lower())[:60]

            normalized_tags = {to_short_name(k): to_short_name(v) for k, v in self.tags.items()}
            self.log.debug(f"normalized tags are {normalized_tags}")

            gce_set_labels(instances_client=self._gce_service,
                           instance=_instance,
                           new_labels=normalized_tags,
                           project=self.project,
                           zone=self._gce_zone_names[dc_idx])

        name = f"{self.node_prefix}-{dc_idx}-{node_index}".lower()
        gce_disk_struct = self._get_disks_struct(name=name, dc_idx=dc_idx)
        # Name must start with a lowercase letter followed by up to 63
        # lowercase letters, numbers, or hyphens, and cannot end with a hyphen
        assert len(name) <= 63, "Max length of instance name is 63"
        startup_script = ""

        if self.params.get("scylla_linux_distro") in ("ubuntu-bionic", "ubuntu-xenial", "ubuntu-focal",):
            # we need to disable sshguard to prevent blocking connections from the builder
            startup_script += dedent("""
                sudo systemctl disable sshguard
                sudo systemctl stop sshguard
            """)
        username = self._gce_image_username
        public_key, key_type = pub_key_from_private_key_file(self.params.get("user_credentials_path"))
        zone = self._gce_zone_names[dc_idx]
        network_tags = ["sct-network-only"]
        if self.params.get('intra_node_comm_public') or ssh_connection_ip_type(self.params) == 'public':
            network_tags.append("sct-allow-public")
        create_node_params = dict(
            project_id=self.project, zone=zone,
            machine_type=instance_type or self._gce_instance_type,
            instance_name=name,
            network_name=self._gce_network,
            disks=gce_disk_struct,
            external_access=True,
            metadata={
                **self.tags,
                "Name": name,
                "NodeIndex": node_index,
                "startup-script": startup_script,
                "user-data": self.prepare_user_data(enable_auto_bootstrap),
                "block-project-ssh-keys": "true",
                "ssh-keys": f"{username}:{key_type} {public_key} {username}",
            },
            spot=spot,
            service_accounts=self._service_accounts,
            network_tags=network_tags
        )
        instance = self._create_node_with_retries(name=name,
                                                  dc_idx=dc_idx,
                                                  create_node_params=create_node_params)

        self.log.debug('Created %s instance %s', 'spot' if spot else 'on-demand', instance)
        try:
            set_tags_as_labels(instance)
        except google.api_core.exceptions.InvalidArgument as exc:
            self.log.warning(f"Unable to set tags as labels due to {exc}")

        return instance

    @retrying(n=3,
              sleep_time=900,
              allowed_exceptions=(google.api_core.exceptions.GoogleAPIError, google.api_core.exceptions.NotFound),
              message="Retrying to create a GCE node...",
              raise_on_exceeded=True)
    def _create_node_with_retries(self,
                                  name: str,
                                  dc_idx: int,
                                  create_node_params: dict[str, Any]) -> compute_v1.Instance:
        try:
            return create_instance(**create_node_params)
        except google.api_core.exceptions.GoogleAPIError as gbe:
            #  attempt to destroy if node did not start due to preemption or quota exceeded
            if "Instance failed to start due to preemption" in str(gbe) or "Quota exceeded" in str(gbe):
                try:
                    self._destroy_instance(name=name, dc_idx=dc_idx)
                except google.api_core.exceptions.NotFound:
                    LOGGER.warning("Attempted to destroy node: {name: %s, idx:%s}, but could not find it. "
                                   "Possibly the node was destroyed already.", name, dc_idx)

            # on any case of failure, fall back to on-demand instance
            create_node_params['spot'] = False

            raise gbe

    def _create_instances(self, count, dc_idx=0, enable_auto_bootstrap=False, instance_type=None):
        spot = 'spot' in self.instance_provision
        instances = []
        for node_index in range(self._node_index + 1, self._node_index + count + 1):
            instances.append(self._create_instance(node_index, dc_idx, spot,
                             enable_auto_bootstrap, instance_type=instance_type))
        return instances

    def _destroy_instance(self, name: str, dc_idx: int):
        target_node = self._get_instances_by_name(dc_idx=dc_idx, name=name)
        operation = self._gce_service.delete(instance=target_node,
                                             project=self.project,
                                             zone=self._gce_zone_names[dc_idx])
        wait_for_extended_operation(operation, "Wait for instance deletion")

    def _get_instances_by_prefix(self, dc_idx: int = 0):
        instances_by_zone = self._gce_service.list(project=self.project, zone=self._gce_zone_names[dc_idx])
        return [node for node in instances_by_zone if node.name.startswith(self._node_prefix)]

    def _get_instances_by_name(self, name: str, dc_idx: int = 0):
        instances_by_zone = self._gce_service.list(project=self.project, zone=self._gce_zone_names[dc_idx])
        found = [node for node in instances_by_zone if node.name == name]
        return found[0] if found else None

    def _get_instances(self, dc_idx: int) -> list[compute_v1.Instance]:
        test_id = self.test_config.test_id()
        if not test_id:
            raise ValueError("test_id should be configured for using reuse_cluster")

        instances_by_tags = list_instances_gce(tags_dict={'TestId': test_id, 'NodeType': self.node_type})
        region = self._gce_zone_names[dc_idx][:-2]
        ip_addresses = gce_public_addresses if self._node_public_ips else gce_private_addresses

        # Filter instances by ip addresses and region
        instances = [
            instance for instance in instances_by_tags
            if ip_addresses(instance) and region in instance.zone
        ]

        def sort_by_index(instance):
            metadata = gce_meta_to_dict(instance.metadata)
            return metadata.get('NodeIndex', 0)

        instances = sorted(instances, key=sort_by_index)
        return instances

    def _create_node(self, instance, node_index, dc_idx, rack):
        try:
            node = GCENode(gce_instance=instance,
                           gce_service=self._gce_service,
                           credentials=self._credentials[0],
                           gce_project=self.project,
                           parent_cluster=self,
                           gce_image_username=self._gce_image_username,
                           node_prefix=self.node_prefix,
                           node_index=node_index,
                           base_logdir=self.logdir,
                           dc_idx=dc_idx,
                           rack=rack,
                           shard_num=self.get_node_shard_num(node_index))
            node.init()
            return node
        except Exception as ex:  # noqa: BLE001
            raise CreateGCENodeError('Failed to create node: %s' % ex) from ex

    # pylint: disable=too-many-arguments
    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False, instance_type=None):
        if count <= 0:
            return []
        self.log.info("Adding nodes to cluster")
        nodes = []
        instance_dc = 0 if self.params.get("simulated_regions") else dc_idx
        if self.test_config.REUSE_CLUSTER:
            instances = self._get_instances(instance_dc)
            if not instances:
                raise RuntimeError("No nodes found for testId %s " % (self.test_config.test_id(),))
        else:
            instances = self._create_instances(
                count, instance_dc, enable_auto_bootstrap, instance_type=instance_type)

        self.log.debug('instances: %s', instances)
        if instances:
            self.log.debug('GCE instance extra info: %s', instances[0])
        for node_index, instance in enumerate(instances, start=self._node_index + 1):
            # in case rack is not specified, spread nodes to different racks
            node_rack = node_index % self.racks_count if rack is None else rack
            node = self._create_node(instance, node_index, dc_idx, rack=node_rack)
            nodes.append(node)
            self.nodes.append(node)
            self.log.info("Added node: %s", node.name)
            node.enable_auto_bootstrap = enable_auto_bootstrap

        self._node_index += count
        self.log.info('added nodes: %s', nodes)
        return nodes


class ScyllaGCECluster(cluster.BaseScyllaCluster, GCECluster):

    def __init__(self, gce_image, gce_image_type, gce_image_size, gce_network, gce_service, credentials,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 gce_instance_type='n2-standard-1', gce_n_local_ssd=1,
                 gce_image_username='centos',
                 user_prefix=None, n_nodes=3, add_disks=None, params=None, gce_datacenter=None, service_accounts=None, nodes_smp=[]):
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
            gce_service=gce_service,
            credentials=credentials,
            cluster_prefix=cluster_prefix,
            node_prefix=node_prefix,
            n_nodes=n_nodes,
            add_disks=add_disks,
            params=params,
            gce_region_names=gce_datacenter,
            node_type='scylla-db',
            service_accounts=service_accounts,
            nodes_smp=nodes_smp,
        )
        self.version = '2.1'

    @staticmethod
    def _wait_for_preinstalled_scylla(node):
        node.wait_for_machine_image_configured()

    def _reuse_cluster_setup(self, node: GCENode) -> None:
        node.run_startup_script()


class LoaderSetGCE(cluster.BaseLoaderSet, GCECluster):

    def __init__(self, gce_image, gce_image_type, gce_image_size, gce_network, gce_service, credentials,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 gce_instance_type='n2-standard-1', gce_n_local_ssd=1,
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
                            gce_service=gce_service,
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

    def __init__(self, gce_image, gce_image_type, gce_image_size, gce_network, gce_service, credentials,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 gce_instance_type='n2-standard-1', gce_n_local_ssd=1,
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
                            gce_service=gce_service,
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
