from sdcm.tester import ClusterTester
from sdcm.utils.decorators import measure_time


class MixedShardRepair(ClusterTester):

    @measure_time
    def _run_repair(self, node):
        node.run_nodetool(sub_cmd='repair')

    @measure_time
    def _run_bootstrap(self):
        new_node = self.db_cluster.add_nodes(count=1)[0]
        self.db_cluster.wait_for_init(node_list=[new_node])
        return new_node

    @measure_time
    def _run_rebuild(self, node):
        node.run_nodetool(sub_cmd='rebuild')

    @measure_time
    def _run_decommission(self, node):
        node.run_nodetool(sub_cmd='decommission')

    def _repair(self, node):
        self.log.info('Running nodetool repair on {}'.format(node.name))
        repair_time = self._run_repair(node=node)[0]  # pylint: disable=unsubscriptable-object
        self.log.info('Repair time on node: {} is: {}'.format(node.name, repair_time))

    def _bootstrap(self):
        self.log.info('Bootstrapping a new node...')
        bootstrap_time, new_node = self._run_bootstrap()
        self.monitors.reconfigure_scylla_monitoring()
        self.log.info('Bootstrap time of node: {} is: {}'.format(new_node.name, bootstrap_time))
        return new_node

    def _rebuild(self, node):
        self.log.info('Rebuilding a node: {}...'.format(node.name))
        rebuild_time = self._run_rebuild(node=node)[0]  # pylint: disable=unsubscriptable-object
        self.log.info('Rebuild time of node: {} is: {}'.format(node.name, rebuild_time))

    def _decommission(self, node):
        self.log.info('Decommission of a node: {}...'.format(node.name))
        decommission_time = self._run_decommission(node=node)[0]  # pylint: disable=unsubscriptable-object
        self.log.info('Decommission time of node: {} is: {}'.format(node.name, decommission_time))

    def test_mix_shard_ops(self):
        self.populate_data_parallel(size_in_gb=1024, replication_factor=3, blocking=True)
        node1 = self.db_cluster.nodes[0]
        node2 = self.db_cluster.nodes[1]
        node3 = self._bootstrap()
        self._repair(node1)
        self._rebuild(node2)
        self._decommission(node3)
