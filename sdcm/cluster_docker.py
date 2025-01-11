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

# pylint: disable=too-many-arguments; looks like we need to increase DESIGN.max_args to 10 in our pylintrc
# pylint: disable=invalid-overridden-method; pylint doesn't know that cached_property is property
import os
import re
import logging
from typing import Optional, Union, Dict
from functools import cached_property

from sdcm import cluster
from sdcm.remote import LOCALRUNNER
from sdcm.utils.docker_utils import get_docker_bridge_gateway, Container, ContainerManager, DockerException
from sdcm.utils.health_checker import check_nodes_status
from sdcm.utils.net import get_my_public_ip

DEFAULT_SCYLLA_DB_IMAGE = "scylladb/scylla-nightly"
DEFAULT_SCYLLA_DB_IMAGE_TAG = "latest"
AIO_MAX_NR_RECOMMENDED_VALUE = 1048576

LOGGER = logging.getLogger(__name__)


class ScyllaDockerRequirementError(cluster.ScyllaRequirementError, DockerException):
    pass


class NodeContainerMixin:
    @cached_property
    def node_container_image_tag(self):
        return self.parent_cluster.node_container_image_tag

    def node_container_image_dockerfile_args(self):
        return dict(path=self.parent_cluster.node_container_context_path)

    def node_container_image_build_args(self):
        return dict(buildargs={"SOURCE_IMAGE": self.parent_cluster.source_image, },
                    labels=self.parent_cluster.tags)

    def node_container_run_args(self, seed_ip):
        volumes = {
            '/var/run/docker.sock': {"bind": '/var/run/docker.sock', "mode": "rw"},
        }

        smp = 1
        if self.node_type == 'db':
            scylla_args = self.parent_cluster.params.get('append_scylla_args')
            smp_match = re.search(r'--smp\s(\d+)', scylla_args)
            smp = int(smp_match.group(1)) if smp_match else 1

        return dict(name=self.name,
                    image=self.node_container_image_tag,
                    command=f'--seeds="{seed_ip}"' if seed_ip else None,
                    volumes=volumes,
                    network=self.parent_cluster.params.get('docker_network'),
                    nano_cpus=smp*10**9)  # Same as `docker run --cpus=N ...' CLI command.


class DockerNode(cluster.BaseNode, NodeContainerMixin):  # pylint: disable=abstract-method
    def __init__(self,  # pylint: disable=too-many-arguments
                 parent_cluster: "DockerCluster",
                 container: Optional[Container] = None,
                 node_prefix: str = "node",
                 base_logdir: Optional[str] = None,
                 ssh_login_info: Optional[dict] = None,
                 node_index: int = 1,
                 shard_num: Optional[int] = None) -> None:
        super().__init__(name=f"{node_prefix}-{node_index}",
                         parent_cluster=parent_cluster,
                         ssh_login_info=ssh_login_info,
                         base_logdir=base_logdir,
                         node_prefix=node_prefix,
                         shard_num=shard_num)
        self.node_index = node_index

        if container is not None:
            assert int(container.labels["NodeIndex"]) == node_index, "Container labeled with wrong index."
            self._containers["node"] = container

    def wait_for_cloud_init(self):
        pass

    @property
    def network_interfaces(self):
        pass
        # TODO: it did not work, we can it check later
        # return [NetworkInterface(ipv4_public_address=self._get_public_ip_address(),
        #                          ipv6_public_addresses='',
        #                          ipv4_private_addresses=[self._get_private_ip_address()],
        #                          ipv6_private_address='',
        #                          dns_private_name='',
        #                          dns_public_name='',
        #                          device_index=0
        #                          )]

    def is_docker(self):
        return True

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**super().tags,
                "NodeIndex": str(self.node_index), }

    def _get_public_ip_address(self) -> Optional[str]:
        return ContainerManager.get_ip_address(self, "node")

    def _get_private_ip_address(self) -> Optional[str]:
        return self.public_ip_address

    def _refresh_instance_state(self):
        public_ipv4_addresses = [self._get_public_ip_address()]
        private_ipv4_addresses = [self._get_private_ip_address()]
        return public_ipv4_addresses, private_ipv4_addresses

    def _get_ipv6_ip_address(self):
        self.log.warning("We don't support IPv6 for Docker backend")
        return ""

    @cached_property
    def host_public_ip_address(self) -> str:
        return get_my_public_ip()

    def is_running(self):
        return ContainerManager.is_running(self, "node")

    def restart(self):
        ContainerManager.get_container(self, "node").restart()

    def get_service_status(self, service_name: str, timeout: int = 500, ignore_status=False):
        return self.remoter.sudo('sh -c "{0} || {0}.service"'.format(f"supervisorctl status {service_name}"),
                                 timeout=timeout, ignore_status=ignore_status)

    def start_scylla_server(self, verify_up=True, verify_down=False, timeout=300, verify_up_timeout=None):
        verify_up_timeout = verify_up_timeout or self.verify_up_timeout
        if verify_down:
            self.wait_db_down(timeout=timeout)
        self.remoter.sudo('sh -c "{0} || {0}-server"'.format("supervisorctl start scylla"),
                          timeout=timeout)
        if verify_up:
            self.wait_db_up(timeout=verify_up_timeout)

        # Need to start the scylla-housekeeping service manually because of autostart of this service is disabled
        # for the docker backend. See, for example, docker/scylla-sct/ubuntu/Dockerfile
        self.start_scylla_housekeeping_service(timeout=timeout)

    @cluster.log_run_info
    def start_scylla(self, verify_up=True, verify_down=False, timeout=300):
        self.start_scylla_server(verify_up=verify_up, verify_down=verify_down, timeout=timeout)

    def stop_scylla_server(self, verify_up=False, verify_down=True, timeout=300, ignore_status=False):
        if verify_up:
            self.wait_db_up(timeout=timeout)
        self.remoter.sudo('sh -c "{0} || {0}-server"'.format("supervisorctl stop scylla"),
                          timeout=timeout)
        if verify_down:
            self.wait_db_down(timeout=timeout)

        # Need to start the scylla-housekeeping service manually because of autostart of this service is disabled
        # for the docker backend. See, for example, docker/scylla-sct/ubuntu/Dockerfile
        self.stop_scylla_housekeeping_service(timeout=timeout)

    def stop_scylla_housekeeping_service(self, timeout=300):
        self.remoter.sudo('sh -c "{0} || {0}-server"'.format("supervisorctl stop scylla-housekeeping"),
                          timeout=timeout)

    def start_scylla_housekeeping_service(self, timeout=300):
        self.remoter.sudo('sh -c "{0} || {0}-server"'.format("supervisorctl start scylla-housekeeping"),
                          timeout=timeout)

    @cluster.log_run_info
    def stop_scylla(self, verify_up=False, verify_down=True, timeout=300):
        self.stop_scylla_server(verify_up=verify_up, verify_down=verify_down, timeout=timeout)

    def restart_scylla_server(self, verify_up_before=False, verify_up_after=True, timeout=1800, verify_up_timeout=None):
        verify_up_timeout = verify_up_timeout or self.verify_up_timeout

        if verify_up_before:
            self.wait_db_up(timeout=verify_up_timeout)

        # Need to restart the scylla-housekeeping service manually because of autostart of this service is disabled
        # for the docker backend. See, for example, docker/scylla-sct/ubuntu/Dockerfile
        self.stop_scylla_housekeeping_service(timeout=timeout)
        self.remoter.sudo('sh -c "{0} || {0}-server"'.format("supervisorctl restart scylla"), timeout=timeout)
        if verify_up_after:
            self.wait_db_up(timeout=verify_up_timeout)
        self.start_scylla_housekeeping_service(timeout=timeout)

    @cluster.log_run_info
    def restart_scylla(self, verify_up_before=False, verify_up_after=True, timeout=1800) -> None:
        self.restart_scylla_server(verify_up_before=verify_up_before, verify_up_after=verify_up_after, timeout=timeout)

    @property
    def image(self) -> str:
        return self.parent_cluster.source_image

    @property
    def vm_region(self):
        return "docker"

    @property
    def region(self):
        return "docker"


class DockerCluster(cluster.BaseCluster):  # pylint: disable=abstract-method
    node_container_user = "scylla-test"

    def __init__(self,
                 docker_image: str = DEFAULT_SCYLLA_DB_IMAGE,
                 docker_image_tag: str = DEFAULT_SCYLLA_DB_IMAGE_TAG,
                 node_key_file: Optional[str] = None,
                 cluster_prefix: str = "cluster",
                 node_prefix: str = "node",
                 node_type: Optional[str] = None,
                 n_nodes: Union[list, int] = 3,
                 params: dict = None,
                 nodes_smp: list[int] = []) -> None:
        self.source_image = f"{docker_image}:{docker_image_tag}"
        self.node_container_image_tag = f"scylla-sct:{node_type}-{str(self.test_config.test_id())[:8]}"
        self.node_container_key_file = node_key_file

        super().__init__(cluster_prefix=cluster_prefix,
                         node_prefix=node_prefix,
                         n_nodes=n_nodes,
                         params=params,
                         region_names=["localhost-dc", ],  # Multi DC is not supported currently.
                         node_type=node_type,
                         nodes_smp=nodes_smp)

    @property
    def node_container_context_path(self):
        # scylla_linux_distro can be: centos or ubuntu-focal, hence we need to split it
        return os.path.join(os.path.dirname(__file__), '../docker/scylla-sct',
                            self.params.get("scylla_linux_distro").split('-')[0])

    def _create_node(self, node_index, container=None):
        node = DockerNode(parent_cluster=self,
                          container=container,
                          ssh_login_info=dict(hostname=None,
                                              user=self.node_container_user,
                                              key_file=self.node_container_key_file),
                          base_logdir=self.logdir,
                          node_prefix=self.node_prefix,
                          node_index=node_index,
                          shard_num=self.get_node_shard_num(node_index))

        if container is None:
            ContainerManager.build_container_image(node, "node")

        ContainerManager.run_container(node, "node", seed_ip=self.nodes[0].public_ip_address if node_index else None)
        ContainerManager.wait_for_status(node, "node", status="running")
        ContainerManager.ssh_copy_id(node, "node", self.node_container_user, self.node_container_key_file)

        node.init()

        return node

    def _get_new_node_indexes(self, count):
        """Return `count' indexes which will fill gaps or/and continue existent indexes in `self.nodes' list."""

        return sorted(set(range(len(self.nodes) + count)) - set(node.node_index for node in self.nodes))

    def _create_nodes(self, count, enable_auto_bootstrap=False):
        new_nodes = []
        for node_index in self._get_new_node_indexes(count):
            node = self._create_node(node_index)
            node.enable_auto_bootstrap = enable_auto_bootstrap
            self.nodes.append(node)
            new_nodes.append(node)
        return new_nodes

    def _get_nodes(self):
        containers = ContainerManager.get_containers_by_prefix(self.node_prefix)
        for node_index, container in sorted(((int(c.labels["NodeIndex"]), c) for c in containers), key=lambda x: x[0]):
            LOGGER.debug("Found container %s with name `%s' and index=%d", container, container.name, node_index)
            node = self._create_node(node_index, container)
            self.nodes.append(node)
        return self.nodes

    def add_nodes(self, count, ec2_user_data="", dc_idx=0, rack=0, enable_auto_bootstrap=False, instance_type=None):
        assert instance_type is None, "docker can't provision different instance types"
        return self._get_nodes() if self.test_config.REUSE_CLUSTER else self._create_nodes(count, enable_auto_bootstrap)


class ScyllaDockerCluster(cluster.BaseScyllaCluster, DockerCluster):  # pylint: disable=abstract-method
    def __init__(self,
                 docker_image: str = DEFAULT_SCYLLA_DB_IMAGE,
                 docker_image_tag: str = DEFAULT_SCYLLA_DB_IMAGE_TAG,
                 node_key_file: Optional[str] = None,
                 user_prefix: Optional[str] = None,
                 n_nodes: Union[list, str] = 3,
                 params: dict = None,
                 nodes_smp: list[int] = []) -> None:
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'db-cluster')
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'db-node')
        super().__init__(docker_image=docker_image,
                         docker_image_tag=docker_image_tag,
                         node_key_file=node_key_file,
                         cluster_prefix=cluster_prefix,
                         node_prefix=node_prefix,
                         node_type="scylla-db",
                         n_nodes=n_nodes,
                         params=params,
                         nodes_smp=nodes_smp)

    def node_setup(self, node, verbose=False, timeout=3600):
        node.wait_ssh_up(verbose=verbose)

        node.is_scylla_installed(raise_if_not_installed=True)

        self.check_aio_max_nr(node)

        if self.test_config.BACKTRACE_DECODING:
            node.install_scylla_debuginfo()

        node.config_setup(append_scylla_args=self.get_scylla_args())

        node.stop_scylla_server(verify_down=False)
        node.remoter.sudo('rm -Rf /var/lib/scylla/data/*')  # Clear data folder to drop wrong cluster name data.

    def node_startup(self, node, verbose=False, timeout=3600):
        node.start_scylla_server(verify_up=False)

        node.wait_db_up(verbose=verbose, timeout=timeout)
        for event in check_nodes_status(nodes_status=node.get_nodes_status(),
                                        current_node=node,
                                        removed_nodes_list=self.dead_nodes_ip_address_list):
            event.publish()
        self.clean_replacement_node_options(node)

    @staticmethod
    def check_aio_max_nr(node: DockerNode, recommended_value: int = AIO_MAX_NR_RECOMMENDED_VALUE):
        """Verify that sysctl key `fs.aio-max-nr' set to recommended value.

        See https://github.com/scylladb/scylla/issues/5638 for details.
        """
        aio_max_nr = int(node.remoter.run("cat /proc/sys/fs/aio-max-nr").stdout)
        if aio_max_nr < recommended_value:
            raise ScyllaDockerRequirementError(
                f"{node}: value of sysctl key `fs.aio-max-nr' ({aio_max_nr}) "
                f"is less than recommended value ({recommended_value})")

    @cluster.wait_for_init_wrap
    def wait_for_init(self, node_list=None, verbose=False, timeout=None, check_node_health=True):
        node_list = node_list if node_list else self.nodes
        self.wait_for_nodes_up_and_normal(nodes=node_list)

    def get_scylla_args(self):
        # pylint: disable=no-member
        append_scylla_args = self.params.get('append_scylla_args_oracle') if self.name.find('oracle') > 0 else \
            self.params.get('append_scylla_args')
        return re.sub(r'--blocked-reactor-notify-ms[ ]+[0-9]+', '', append_scylla_args)


class LoaderSetDocker(cluster.BaseLoaderSet, DockerCluster):
    def __init__(self,
                 docker_image: str = DEFAULT_SCYLLA_DB_IMAGE,
                 docker_image_tag: str = DEFAULT_SCYLLA_DB_IMAGE_TAG,
                 node_key_file: Optional[str] = None,
                 user_prefix: Optional[str] = None,
                 n_nodes: Union[list, str] = 3,
                 params: dict = None) -> None:
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'loader-node')
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'loader-set')

        cluster.BaseLoaderSet.__init__(self,
                                       params=params)
        DockerCluster.__init__(self,
                               docker_image=docker_image,
                               docker_image_tag=docker_image_tag,
                               node_key_file=node_key_file,
                               cluster_prefix=cluster_prefix,
                               node_prefix=node_prefix,
                               node_type="loader",
                               n_nodes=n_nodes,
                               params=params)

    def node_setup(self, node: DockerNode, verbose=False, **kwargs):
        node.wait_ssh_up(verbose=verbose)
        node.remoter.sudo("apt update", verbose=True, ignore_status=True)
        node.remoter.sudo("apt install -y openjdk-8-jre", verbose=True, ignore_status=True)
        node.remoter.sudo("ln -sf /usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/bin/java* /etc/alternatives/java",
                          verbose=True, ignore_status=True)

        if self.params.get('client_encrypt'):
            node.config_client_encrypt()


class DockerMonitoringNode(cluster.BaseNode):  # pylint: disable=abstract-method,too-many-instance-attributes
    log = LOGGER

    def __init__(self,
                 parent_cluster: "MonitorSetDocker",
                 node_prefix: str = "monitor-node",
                 base_logdir: Optional[str] = None,
                 node_index: int = 1,
                 ssh_login_info: Optional[dict] = None) -> None:
        super().__init__(name=f"{node_prefix}-{node_index}",
                         parent_cluster=parent_cluster,
                         base_logdir=base_logdir,
                         node_prefix=node_prefix,
                         ssh_login_info=ssh_login_info)
        self.node_index = node_index

    def wait_for_cloud_init(self):
        pass

    @staticmethod
    def is_docker() -> bool:
        return True

    @cached_property
    def tags(self) -> dict[str, str]:
        return {**super().tags, "NodeIndex": str(self.node_index), }

    def _init_remoter(self, ssh_login_info):  # pylint: disable=no-self-use
        self.remoter = LOCALRUNNER

    def _init_port_mapping(self):  # pylint: disable=no-self-use
        pass

    def wait_ssh_up(self, verbose=True, timeout=500):
        pass

    def update_repo_cache(self):
        pass

    def _refresh_instance_state(self):
        return ["127.0.0.1"], ["127.0.0.1"]

    @cached_property
    def grafana_address(self):
        # Grafana starts in port mapping mode under Docker and we can't use 127.0.0.1 because it'll be used by
        # RemoteWebDriver from RemoteWebDriverContainer.  We can use a gateway from a bridge network instead,
        # because port mapping works on the gateway address too.
        return get_docker_bridge_gateway(self.remoter)

    def start_journal_thread(self):
        # Hasn't it's own system logs, since running on the host
        pass

    def disable_daily_triggered_services(self):
        pass


class MonitorSetDocker(cluster.BaseMonitorSet, DockerCluster):  # pylint: disable=abstract-method
    def __init__(self,
                 targets: dict,
                 user_prefix: Optional[str] = None,
                 n_nodes: Union[list, int] = 3,
                 params: dict = None) -> None:
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-node')
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-set')

        cluster.BaseMonitorSet.__init__(self,
                                        targets=targets,
                                        params=params)
        DockerCluster.__init__(self,
                               cluster_prefix=cluster_prefix,
                               node_prefix=node_prefix,
                               node_type="monitor",
                               n_nodes=n_nodes,
                               params=params)

    def _create_node(self, node_index, container=None):
        node = DockerMonitoringNode(parent_cluster=self,
                                    base_logdir=self.logdir,
                                    node_prefix=self.node_prefix,
                                    node_index=node_index,
                                    ssh_login_info=dict(hostname=None,
                                                        user=self.node_container_user,
                                                        key_file=self.node_container_key_file)
                                    )
        node.init()
        return node

    def add_nodes(self, count, ec2_user_data="", dc_idx=0, rack=0, enable_auto_bootstrap=False, instance_type=None):
        assert instance_type is None, "docker can provision different instance types"
        return self._create_nodes(count, enable_auto_bootstrap)

    @staticmethod
    def install_scylla_monitoring_prereqs(node):  # pylint: disable=invalid-name
        pass  # since running local, don't install anything, just the monitor

    def get_backtraces(self):
        pass

    def destroy(self):
        for node in self.nodes:
            try:
                self.stop_scylla_monitoring(node)
                self.log.error("Stopping scylla monitoring succeeded")
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                self.log.error(f"Stopping scylla monitoring failed with {str(exc)}")
            try:
                node.remoter.sudo(f"rm -rf '{self.monitor_install_path_base}'")
                self.log.error("Cleaning up scylla monitoring succeeded")
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                self.log.error(f"Cleaning up scylla monitoring failed with {str(exc)}")
            node.destroy()
