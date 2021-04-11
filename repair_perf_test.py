#!/usr/bin/env python

import random
import datetime

from sdcm.tester import ClusterTester
from sdcm.db_stats import PrometheusDBStats


class RepairPerfTest(ClusterTester):
    """
    Test performance of Repair related operations.
    """

    def test_bootstrap(self):
        """
        Init a basic cluster with 3 db nodes, and add a new node
        """
        stress_cmd = self.params.get('prepare_write_cmd')
        stress = self.run_stress_thread(stress_cmd=stress_cmd)
        self.verify_stress_thread(cs_thread_pool=stress)

        stress_cmd = self.params.get('stress_cmd')
        stress = self.run_stress_thread(stress_cmd=stress_cmd)

        self.log.info("Adding new node to cluster...")
        start = datetime.datetime.now()
        new_node = self.db_cluster.add_nodes(count=1, enable_auto_bootstrap=True)[0]
        self.monitors.reconfigure_scylla_monitoring()
        self.db_cluster.wait_for_init(node_list=[new_node], timeout=10800)
        self.db_cluster.wait_for_nodes_up_and_normal(nodes=[new_node])
        bootstrap_end = datetime.datetime.now()

        self.verify_stress_thread(cs_thread_pool=stress)
        read_end = datetime.datetime.now()
        bootstrap_duration = (bootstrap_end - start).total_seconds()
        read_workload_time = (read_end - start).total_seconds()
        self.log.info('bootstrap: {}, read workload:{}'.format(
            bootstrap_duration, read_workload_time))

        prometheus_stats = PrometheusDBStats(host=self.monitors.nodes[0].public_ip_address)
        read_latency_99 = prometheus_stats.get_latency_read_99(start_time=start, end_time=read_end)
        stats = {'bootstrap_duration': bootstrap_duration, 'read_workload_time': read_workload_time,
                 'read_latency_99': read_latency_99}
        self.log.info(stats)
        self.update_test_details(extra_stats=stats)

    def test_decommission(self):

        stress_cmd = self.params.get('prepare_write_cmd')
        stress = self.run_stress_thread(stress_cmd=stress_cmd)
        self.verify_stress_thread(cs_thread_pool=stress)

        stress_cmd = self.params.get('stress_cmd')
        stress = self.run_stress_thread(stress_cmd=stress_cmd)

        non_seed_nodes = [node for node in self.db_cluster.nodes if not node.is_seed and not node.running_nemesis]
        target_node = random.choice(non_seed_nodes)

        self.log.info("Decommission the one node from cluster...")

        start = datetime.datetime.now()
        target_node.run_nodetool("decommission")
        decommission_end = datetime.datetime.now()

        self.db_cluster.terminate_node(target_node)
        self.monitors.reconfigure_scylla_monitoring()

        self.verify_stress_thread(cs_thread_pool=stress)
        read_end = datetime.datetime.now()

        decommission_time = (decommission_end - start).total_seconds()
        read_workload_time = (read_end - start).total_seconds()

        prometheus_stats = PrometheusDBStats(host=self.monitors.nodes[0].public_ip_address)
        read_latency_99 = prometheus_stats.get_latency_read_99(start_time=start, end_time=read_end)
        stats = {'decommission_time': decommission_time, 'read_workload_time': read_workload_time,
                 'read_latency_99': read_latency_99}
        self.log.info(stats)
        self.update_test_details(extra_stats=stats)
