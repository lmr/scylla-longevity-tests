import logging
from longevity_test import LongevityTest
from sdcm import prometheus
from sdcm.sct_events.system import InfoEvent
from sdcm.utils import loader_utils

LOGGER = logging.getLogger(__name__)


class FullClusterReplacement(LongevityTest, loader_utils.LoaderUtilsMixin):
    """
    The purpose of these tests is replacement of all nodes in the ScyllaDB.
    Cluster Replacement QA task: https://github.com/scylladb/qa-tasks/issues/1164
    Cluster replacement test plan: https://docs.google.com/document/d/1aKCTs6SiH9ORDYvkAQ8upOpVwZqaxsW9VfbJqKG63g0/edit
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_srv = prometheus.nemesis_metrics_obj()
        self.stress_queue = []

    def start_load(self):
        keyspace_num = self.params.get('keyspace_num')
        stress_cmd = self.params.get('stress_cmd')
        # Start the load
        self.run_pre_create_schema()
        self.run_prepare_write_cmd()
        # self.run_post_prepare_cql_cmds() Uncomment when MV solved: Ref: https://github.com/scylladb/scylladb/issues/15717
        self.assemble_and_run_all_stress_cmd(self.stress_queue, stress_cmd, keyspace_num)

    def verify_stress_threads_after_node_replacement(self):
        for stress in self.stress_queue:
            self.verify_stress_thread(cs_thread_pool=stress)

    def add_new_node_then_decommission_one_node(self, node_to_be_removed, dc_idx=0, new_instance_type=None):
        """
        This method use to replaces a node in a cluster by 'Add a new node to the cluster and then decommission the old node' procedure # pylint: disable=line-too-long
        It adds a new node to the cluster, waits for initialization, decommissions the specified node,
        and reconfigures the monitoring system.
        The procedure from: https://opensource.docs.scylladb.com/stable/operating-scylla/procedures/cluster-management/replace-running-node.html#add-a-new-node-to-the-cluster-and-then-decommission-the-old-node # pylint: disable=line-too-long
        """
        self.metrics_srv.event_start('add_node')
        new_node = \
            self.db_cluster.add_nodes(count=1, dc_idx=dc_idx, enable_auto_bootstrap=True, instance_type=new_instance_type)[
                0]
        self.db_cluster.wait_for_init(node_list=[new_node], timeout=3600, check_node_health=False)
        self.metrics_srv.event_stop('add_node')
        self.monitors.reconfigure_scylla_monitoring()

        # Decommission node to be replaced
        InfoEvent(message='Decommission node: {}'.format(node_to_be_removed)).publish()
        self.db_cluster.decommission(node_to_be_removed)
        for node in self.db_cluster.nodes:
            node.run_nodetool(sub_cmd='cleanup')

    def test_rolling_full_cluster_replacement_add_then_decommission_nodes(self):
        InfoEvent(message="Starting cluster replacement test by add then decommission nodes...").publish()
        self.start_load()
        new_instance_type = self.db_cluster.params.get('new_instance_type')
        # Run node replacement procedure for all nodes and verify data
        for node in self.db_cluster.nodes:
            self.add_new_node_then_decommission_one_node(node_to_be_removed=node, new_instance_type=new_instance_type)
            self.verify_stress_threads_after_node_replacement()

        InfoEvent(message="Finished cluster replacement test by add then decommission nodes...").publish()

    def test_multidc_rolling_full_cluster_replacement_add_then_decommission_nodes(self):
        new_instance_type = self.params.get('new_instance_type')

        InfoEvent(message="Starting multi-dc cluster replacement test by add then decommission nodes "
                          "with repair and major compaction in the end...").publish()
        self.start_load()

        # Run node replacement procedure for all nodes and verify data
        number_of_nodes_in_each_dc = int(len(self.db_cluster.nodes)/2)
        for i, node in enumerate(self.db_cluster.nodes):
            if i >= number_of_nodes_in_each_dc:
                dc_idx = 0
            else:
                dc_idx = 1
            self.add_new_node_then_decommission_one_node(
                node_to_be_removed=node, dc_idx=dc_idx, new_instance_type=new_instance_type)
            self.verify_stress_threads_after_node_replacement()

        # Full Repair and major compaction
        # This is not required for the replacement procedure, but can be a part of the test to check repair and
        # compaction on a replaced cluster.
        for node in self.db_cluster.nodes:
            InfoEvent(message=f"Starting a repair on node {node.name}")
            node.run_nodetool(sub_cmd="repair -pr", publish_event=True)
            node.run_nodetool(sub_cmd="compact", publish_event=True)
            InfoEvent(message=f"Waiting for compactions to finish on {node.name}")
            self.wait_no_compactions_running()

        InfoEvent(message="Finished multi-dc cluster replacement test by add then decommission nodes "
                          "with repair and major compaction in the end...").publish()
