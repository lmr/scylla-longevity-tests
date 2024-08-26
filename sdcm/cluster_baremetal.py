import logging
from typing import Optional, TypedDict

from invoke import UnexpectedExit

from sdcm import cluster

LOGGER = logging.getLogger(__name__)

BASE_NAME = 'db-node'
LOADER_NAME = 'loader-node'
MONITOR_NAME = 'monitor-node'


class NodeInfo(TypedDict):
    public_ip: str
    private_ip: str


class NodeCredentialInformation(TypedDict):
    username: str
    node_list: list[NodeInfo]


class BareMetalCredentials(TypedDict):
    db_nodes: NodeCredentialInformation
    loader_nodes: NodeCredentialInformation
    monitor_nodes: NodeCredentialInformation


class NodeIpsNotConfiguredError(Exception):
    pass


class PhysicalMachineNode(cluster.BaseNode):
    log = LOGGER

    # pylint: disable=too-many-arguments
    def __init__(self, name, parent_cluster: 'PhysicalMachineCluster',
                 public_ip, private_ip, credentials, base_logdir=None, node_prefix=None, shard_num=None):
        ssh_login_info = {'hostname': None,
                          'user': getattr(parent_cluster, "ssh_username", credentials.name),
                          'key_file': credentials.key_file}
        self._public_ip = public_ip
        self._private_ip = private_ip
        super().__init__(name=name,
                         parent_cluster=parent_cluster,
                         base_logdir=base_logdir,
                         ssh_login_info=ssh_login_info,
                         node_prefix=node_prefix,
                         shard_num=shard_num)

    def init(self):
        super().init()
        self.set_hostname()
        self.run_startup_script()

    def wait_for_cloud_init(self):
        pass

    def _get_public_ip_address(self) -> Optional[str]:
        return self._public_ip

    @property
    def vm_region(self):
        return "baremetal"

    @property
    def region(self):
        return "baremetal"

    def scylla_setup(self, disks, devname: str):
        try:
            super().scylla_setup(disks, devname)
        except UnexpectedExit as exc:
            # Covering for case when scylla-setup script has already been run. If the command is scylla_setup
            # and there's "already" in the stdout of that command, we can skip this method safely as nics and disks
            # were already configured on this node.
            if "scylla_setup" in exc.result.command and "already" in exc.streams_for_display()[0].lower():
                return
            raise exc

    def _get_private_ip_address(self) -> Optional[str]:
        return self._private_ip

    def set_hostname(self):
        # disabling for now, since doesn't working with Fabric from within docker, and not needed for scylla-cloud,
        # since not using hostname anywhere
        # self.remoter.run('sudo hostnamectl set-hostname {}'.format(self.name))
        pass

    def reboot(self, hard=True, verify_ssh=True):
        raise NotImplementedError("reboot not implemented")

    def restart(self):
        self.remoter.run('sudo reboot -h now', ignore_status=True)

    def destroy(self):
        self.stop_task_threads()  # For future implementation of destroy
        self.wait_till_tasks_threads_are_stopped()
        super().destroy()


class PhysicalMachineCluster(cluster.BaseCluster):  # pylint: disable=abstract-method
    def __init__(self, **kwargs):
        self.nodes = []
        self.credentials = kwargs.pop('credentials')
        n_nodes = kwargs.get('n_nodes')
        self._node_public_ips = kwargs.pop('public_ips', None) or []
        self._node_private_ips = kwargs.pop('private_ips', None) or []
        node_cnt = n_nodes[0] if isinstance(n_nodes, list) else n_nodes
        if len(self._node_public_ips) < node_cnt or len(self._node_private_ips) < node_cnt:
            raise NodeIpsNotConfiguredError('Physical hosts IPs are not configured!')
        super().__init__(nodes_smp=kwargs.pop('nodes_smp') or [], **kwargs)

    @property
    def ssh_username(self) -> str:
        # pylint: disable=no-member
        return self._ssh_username

    def _create_node(self, name, public_ip, private_ip, shard_num):
        node = PhysicalMachineNode(name,
                                   parent_cluster=self,
                                   public_ip=public_ip,
                                   private_ip=private_ip,
                                   credentials=self.credentials[0],
                                   base_logdir=self.logdir,
                                   node_prefix=self.node_prefix,
                                   shard_num=shard_num)
        node.init()
        return node

    # pylint: disable=unused-argument,too-many-arguments
    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False, instance_type=None):
        assert instance_type is None, "baremetal can provision diffrent types"
        for node_index in range(count):
            node_name = '%s-%s' % (self.node_prefix, node_index)
            self.nodes.append(self._create_node(node_name,
                                                self._node_public_ips[node_index],
                                                self._node_private_ips[node_index],
                                                self.get_node_shard_num(node_index)))


class ScyllaPhysicalCluster(cluster.BaseScyllaCluster, PhysicalMachineCluster):

    def __init__(self, **kwargs):
        user_prefix = kwargs.pop('user_prefix')
        if username := kwargs.pop('ssh_username', None):
            self._ssh_username = username
        kwargs.update(dict(
            cluster_prefix=cluster.prepend_user_prefix(user_prefix, 'db-cluster'),
            node_prefix=cluster.prepend_user_prefix(user_prefix, 'db-node'),
            node_type='scylla-db'
        ))
        super().__init__(**kwargs)


class LoaderSetPhysical(cluster.BaseLoaderSet, PhysicalMachineCluster):  # pylint: disable=abstract-method

    def __init__(self, **kwargs):
        user_prefix = kwargs.pop('user_prefix')
        if username := kwargs.pop('ssh_username', None):
            self._ssh_username = username
        kwargs.update(dict(
            cluster_prefix=cluster.prepend_user_prefix(user_prefix, 'loader-set'),
            node_prefix=cluster.prepend_user_prefix(user_prefix, 'loader-node'),
            node_type='loader'
        ))
        cluster.BaseLoaderSet.__init__(self, kwargs["params"])
        PhysicalMachineCluster.__init__(self, **kwargs)

    @classmethod
    def _get_node_ips_param(cls, ip_type='public'):
        return cluster.BaseLoaderSet.get_node_ips_param(ip_type)


class MonitorSetPhysical(cluster.BaseMonitorSet, PhysicalMachineCluster):

    def __init__(self, **kwargs):
        user_prefix = kwargs.pop('user_prefix')
        if username := kwargs.pop('ssh_username', None):
            self._ssh_username = username
        kwargs.update(dict(
            cluster_prefix=cluster.prepend_user_prefix(user_prefix, 'monitor-set'),
            node_prefix=cluster.prepend_user_prefix(user_prefix, 'monitor-node'),
            node_type='monitor'
        ))
        cluster.BaseMonitorSet.__init__(self,
                                        targets=kwargs["targets"],
                                        params=kwargs["params"])
        kwargs.pop("targets")
        PhysicalMachineCluster.__init__(self, **kwargs)
