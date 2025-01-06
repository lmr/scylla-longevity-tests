from sdcm.target_node_lock import lock_node
# this file contains thread safe operations on cluster that requires lock node by nemesis before call


def safe_cluster_start_stop(nemesis_label, cluster, nodes=None, random_order=False, lock_timeout=3000):
    # try to stop start cluster within 3000sec lock timeout
    nodes_to_restart = (nodes or cluster.nodes)[:]
    for node in nodes_to_restart:
        with lock_node(node, nemesis_label, timeout=lock_timeout):
            cluster.restart_scylla(nodes=[node], random_order=random_order)


def safe_node_restart(nemesis_label, node, verify_up_before=False, verify_up_after=True, restart_timeout=1800, lock_timeout=3000):
    # try to restart node within 3000sec lock timeout
    with lock_node(node, nemesis_label, timeout=lock_timeout):
        node.restart_scylla(verify_up_before, verify_up_after, restart_timeout)
