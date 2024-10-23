import contextlib
import logging

from typing import Optional

from sdcm.cluster import BaseNode

LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def block_scylla_ports(target_node: "BaseNode", ports: list[int] | None = None):
    ports = ports or [7001, 7000, 9042, 9142, 19042, 19142]
    target_node.install_package("iptables")
    target_node.log.debug("Block connections %s", target_node.name)
    for port in ports:
        target_node.remoter.sudo(f"iptables -A INPUT -p tcp --dport {port} -j DROP")
        target_node.remoter.sudo(f"iptables -A OUTPUT -p tcp --dport {port} -j DROP")
    yield
    target_node.log.debug("Remove all iptable rules %s", target_node.name)
    for port in ports:
        target_node.remoter.sudo(f"iptables -D INPUT -p tcp --dport {port} -j DROP")
        target_node.remoter.sudo(f"iptables -D OUTPUT -p tcp --dport {port} -j DROP")


@contextlib.contextmanager
def stop_scylla_with_sigstop(target_node: "BaseNode"):
    target_node.log.debug("Send signal SIGSTOP to scylla process on node %s", target_node.name)
    target_node.remoter.sudo("pkill --signal SIGSTOP -e scylla", timeout=60)
    yield
    target_node.log.debug("Send signal SIGCONT to scylla process on node %s", target_node.name)
    target_node.remoter.sudo(cmd="pkill --signal SIGCONT -e scylla", timeout=60)


def is_node_removed_from_cluster(removed_node: BaseNode, verification_node: BaseNode) -> bool:
    LOGGER.debug("Verification node %s", verification_node.name)
    cluster_status: Optional[dict] = removed_node.parent_cluster.get_nodetool_status(
        verification_node=verification_node)
    if not cluster_status:
        return False
    result = []
    for dc in cluster_status:
        result.append(removed_node.ip_address not in cluster_status[dc].keys())
    return all(result)


def is_node_seen_as_down(down_node: BaseNode, verification_node: BaseNode) -> bool:
    LOGGER.debug("Verification node %s", verification_node.name)
    return down_node not in down_node.parent_cluster.get_nodes_up_and_normal(verification_node)
