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
# Copyright (c) 2022 ScyllaDB
import json
import logging
from functools import cached_property
from typing import Dict, List

from sdcm import cluster
from sdcm.provision.azure.provisioner import AzureProvisioner
from sdcm.provision.provisioner import PricingModel, VmInstance
from sdcm.sct_events.system import SpotTerminationEvent
from sdcm.sct_provision import region_definition_builder
from sdcm.sct_provision.instances_provider import provision_instances_with_fallback
from sdcm.utils.decorators import retrying

LOGGER = logging.getLogger(__name__)
SPOT_TERMINATION_CHECK_DELAY = 15


class CreateAzureNodeError(Exception):
    pass


class AzureNode(cluster.BaseNode):

    """
    Wraps Azure instances, so that we can also control the instance through SSH.
    """

    log = LOGGER

    def __init__(self, azure_instance: VmInstance,  # pylint: disable=too-many-arguments
                 credentials, parent_cluster,
                 node_prefix='node', node_index=1,
                 base_logdir=None, dc_idx=0, rack=0, shard_num=None):
        self.node_index = node_index
        self.dc_idx = dc_idx
        self.parent_cluster = parent_cluster
        self._instance = azure_instance
        name = f"{node_prefix}-{self.region}-{node_index}".lower()
        self.last_event_document_incarnation = -1
        ssh_login_info = {'hostname': None,
                          'user': azure_instance.user_name,
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

    def init(self) -> None:
        super().init()
        # disable auditd service
        self.remoter.sudo("systemctl stop auditd", ignore_status=True)
        self.remoter.sudo("systemctl disable auditd", ignore_status=True)
        self.remoter.sudo("systemctl mask auditd", ignore_status=True)
        self.remoter.sudo("systemctl daemon-reload", ignore_status=True)

    def wait_for_cloud_init(self):
        pass  # azure for it, on resources creation

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**super().tags,
                "NodeIndex": str(self.node_index), }

    @property
    def network_interfaces(self):
        pass

    def refresh_network_interfaces_info(self):
        pass

    @retrying(n=6, sleep_time=1)
    def _set_keep_alive(self) -> bool:
        self._instance.add_tags({"keep": "alive"})
        return super()._set_keep_alive()

    def _refresh_instance_state(self):
        ip_tuple = ([self._instance.public_ip_address], [self._instance.private_ip_address])
        return ip_tuple

    @property
    def vm_region(self):
        return self._instance.region

    def set_hostname(self):
        self.log.debug("Hostname for node %s left as is", self.name)

    @property
    def is_spot(self):
        return self._instance.pricing_model.is_spot()

    def check_spot_termination(self):
        """Check if a spot instance termination was initiated by the cloud.

        Returns number of seconds to wait before next check.
        """
        try:
            self.wait_ssh_up(verbose=False)
            result = self.remoter.run(
                'curl http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01 -H "Metadata: true"', verbose=False)
            status = json.loads(result.stdout.strip())
            if status["DocumentIncarnation"] == self.last_event_document_incarnation:
                # each change in status["Events"] increments "DocumentIncarnation", return if there was no change.
                return SPOT_TERMINATION_CHECK_DELAY
            for event in status["Events"]:
                self.last_event_document_incarnation = status["DocumentIncarnation"]
                if event["EventType"] == "Preempt":
                    message = f"Got spot termination event for node: {event['Resources']}. VM eviction time is {event['NotBefore']}."
                    SpotTerminationEvent(node=self, message=message).publish()
                else:
                    # other EventType's that can be triggered by Azure's maintenance: "Reboot" | "Redeploy" | "Freeze" | "Terminate"
                    self.log.warning(f"Unhandled Azure scheduled event: {event}")
        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.warning('Error during getting Azure scheduled events: %s', details)
            return 0
        return SPOT_TERMINATION_CHECK_DELAY

    def restart(self):
        # When using NVMe disks in Azure, there is no option to Stop and Start an instance.
        # So, for now we will keep restart the same as hard reboot.
        self._instance.reboot(wait=True, hard=False)

    def hard_reboot(self):
        self._instance.reboot(wait=True, hard=True)

    def destroy(self):
        self.stop_task_threads()
        self.wait_till_tasks_threads_are_stopped()
        self._instance.terminate(wait=True)
        super().destroy()

    def get_console_output(self):
        # TODO adding console output from instance on Azure
        self.log.warning('Method is not implemented for AzureNode')
        return ''

    def get_console_screenshot(self):
        # TODO adding console output from instance on Azure
        self.log.warning('Method is not implemented for AzureNode')
        return b''

    def _get_ipv6_ip_address(self):
        # todo: fix it
        return ""

    @property
    def image(self):
        return self._instance.image

    def _get_public_ip_address(self) -> str | None:
        return self._instance.public_ip_address

    def _get_private_ip_address(self) -> str | None:
        return self._instance.private_ip_address

    def configure_remote_logging(self) -> None:
        """Remote logging configured upon vm provisioning using UserDataObject"""
        return


class AzureCluster(cluster.BaseCluster):   # pylint: disable=too-many-instance-attributes
    def __init__(self, image_id, root_disk_size,  # pylint: disable=too-many-arguments, too-many-locals  # noqa: PLR0913
                 provisioners: List[AzureProvisioner], credentials,
                 cluster_uuid=None, instance_type='Standard_L8s_v3', region_names=None,
                 user_name='root', cluster_prefix='cluster',
                 node_prefix='node', n_nodes=3, params=None, node_type=None, nodes_smp=[]):
        self.provisioners: List[AzureProvisioner] = provisioners
        self._image_id = image_id
        self._root_disk_size = root_disk_size
        self._credentials = credentials
        self._instance_type = instance_type
        self._user_name = user_name
        self._azure_region_names = region_names
        self._node_prefix = node_prefix
        self._definition_builder = region_definition_builder.get_builder(params, test_config=self.test_config)
        super().__init__(cluster_uuid=cluster_uuid,
                         cluster_prefix=cluster_prefix,
                         node_prefix=node_prefix,
                         n_nodes=n_nodes,
                         params=params,
                         region_names=region_names,
                         node_type=node_type,
                         nodes_smp=nodes_smp)
        self.log.debug("AzureCluster constructor")

    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False, instance_type=None):  # pylint: disable=too-many-arguments
        self.log.info("Adding nodes to cluster")
        nodes = []

        instance_dc = 0 if self.params.get("simulated_regions") else dc_idx
        instances = self._create_instances(count, instance_dc, instance_type=instance_type)

        self.log.debug('instances: %s', instances)
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

    def _create_node(self, instance, node_index, dc_idx, rack):
        try:
            node = AzureNode(azure_instance=instance,
                             credentials=self._credentials[0],
                             parent_cluster=self,
                             node_prefix=self.node_prefix,
                             node_index=node_index,
                             base_logdir=self.logdir,
                             dc_idx=dc_idx,
                             rack=rack,
                             shard_num=self.get_node_shard_num(self, node_index))
            node.init()
            return node
        except Exception as ex:  # noqa: BLE001
            raise CreateAzureNodeError('Failed to create node: %s' % ex) from ex

    def _create_instances(self, count, dc_idx=0, instance_type=None) -> List[VmInstance]:
        region = self._definition_builder.regions[dc_idx]
        assert region, "no region provided, please add `azure_region_name` param"
        pricing_model = PricingModel.SPOT if 'spot' in self.instance_provision else PricingModel.ON_DEMAND
        definitions = []
        for node_index in range(self._node_index + 1, self._node_index + count + 1):
            definitions.append(
                self._definition_builder.build_instance_definition(
                    region=region, node_type=self.node_type, index=node_index, instance_type=instance_type)
            )
        return provision_instances_with_fallback(self.provisioners[dc_idx], definitions=definitions, pricing_model=pricing_model,
                                                 fallback_on_demand=self.params.get("instance_provision_fallback_on_demand"))

    def get_node_ips_param(self, public_ip=True):
        # todo lukasz: why gce cluster didn't have to implement this?
        raise NotImplementedError("get_node_ips_param should not run")

    def node_setup(self, node, verbose=False, timeout=3600):
        # todo lukasz: why gce cluster didn't have to implement this?
        raise NotImplementedError("node_setup should not run")

    def node_startup(self, node, verbose=False, timeout=3600):
        raise NotImplementedError("'node_startup' should not run")

    def wait_for_init(self):
        # todo lukasz: why gce cluster didn't have to implement this?
        raise NotImplementedError("wait_for_init should not run")


class ScyllaAzureCluster(cluster.BaseScyllaCluster, AzureCluster):

    def __init__(self, image_id, root_disk_size,  # pylint: disable=too-many-arguments
                 provisioners: List[AzureProvisioner], credentials,
                 instance_type='Standard_L8s_v3',
                 user_name='ubuntu',
                 user_prefix=None, n_nodes=3, params=None, region_names=None, nodes_smp=[]):
        # pylint: disable=too-many-locals
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'db-cluster')
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'db-node')
        super().__init__(
            image_id=image_id,
            root_disk_size=root_disk_size,
            instance_type=instance_type,
            user_name=user_name,
            provisioners=provisioners,
            credentials=credentials,
            cluster_prefix=cluster_prefix,
            node_prefix=node_prefix,
            n_nodes=n_nodes,
            params=params,
            region_names=region_names,
            node_type='scylla-db',
            nodes_smp=nodes_smp
        )
        self.version = '2.1'

    @staticmethod
    def _wait_for_preinstalled_scylla(node):
        node.wait_for_machine_image_configured()

    def _reuse_cluster_setup(self, node: AzureNode) -> None:
        node.run_startup_script()


class LoaderSetAzure(cluster.BaseLoaderSet, AzureCluster):

    def __init__(self, image_id, root_disk_size, provisioners, credentials,  # pylint: disable=too-many-arguments
                 instance_type='Standard_D2_v4',
                 user_name='centos',
                 user_prefix=None, n_nodes=1, params=None, region_names=None):
        # pylint: disable=too-many-locals
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'loader-node')
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'loader-set')
        cluster.BaseLoaderSet.__init__(self, params=params)
        AzureCluster.__init__(self,
                              image_id=image_id,
                              root_disk_size=root_disk_size,
                              instance_type=instance_type,
                              user_name=user_name,
                              provisioners=provisioners,
                              credentials=credentials,
                              cluster_prefix=cluster_prefix,
                              node_prefix=node_prefix,
                              n_nodes=n_nodes,
                              params=params,
                              node_type='loader',
                              region_names=region_names
                              )


class MonitorSetAzure(cluster.BaseMonitorSet, AzureCluster):

    def __init__(self, image_id, root_disk_size, provisioners, credentials,  # pylint: disable=too-many-arguments
                 instance_type='Standard_D2_v4',
                 user_name='centos', user_prefix=None, n_nodes=1,
                 targets=None, params=None, region_names=None):
        # pylint: disable=too-many-locals
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-node')
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-set')

        targets = targets if targets else {}
        cluster.BaseMonitorSet.__init__(self,
                                        targets=targets,
                                        params=params)
        AzureCluster.__init__(self,
                              image_id=image_id,
                              root_disk_size=root_disk_size,
                              instance_type=instance_type,
                              user_name=user_name,
                              provisioners=provisioners,
                              credentials=credentials,
                              cluster_prefix=cluster_prefix,
                              node_prefix=node_prefix,
                              n_nodes=n_nodes,
                              params=params,
                              node_type='monitor',
                              region_names=region_names
                              )
