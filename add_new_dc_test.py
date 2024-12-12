import warnings
from typing import Tuple, List

from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement
import pytest  # pylint: disable=no-name-in-module

from longevity_test import LongevityTest
from sdcm.cluster import BaseNode, NodeSetupFailed, NodeSetupTimeout
from sdcm.stress_thread import CassandraStressThread
from sdcm.utils.common import skip_optional_stage
from sdcm.utils.decorators import optional_stage, skip_on_capacity_issues
from sdcm.utils.replication_strategy_utils import NetworkTopologyReplicationStrategy
from sdcm.exceptions import ReadBarrierErrorException
from sdcm.utils.adaptive_timeouts import Operations, adaptive_timeout

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)


class TestAddNewDc(LongevityTest):
    """Test for procedure:
     https://docs.scylladb.com/operating-scylla/procedures/cluster-management/add-dc-to-existing-dc/
     """

    def test_add_new_dc(self) -> None:  # pylint: disable=too-many-locals

        self.log.info("Starting add new DC test...")
        assert self.params.get('n_db_nodes').endswith(" 0"), "n_db_nodes must be a list and last dc must equal 0"
        system_keyspaces = ["system_distributed", "system_traces"]
        # auth-v2 is used when consistent topology is enabled
        if not self.db_cluster.nodes[0].raft.is_consistent_topology_changes_enabled:
            system_keyspaces.insert(0, "system_auth")

        # reconfigure system keyspaces to use NetworkTopologyStrategy
        status = self.db_cluster.get_nodetool_status()
        self.reconfigure_keyspaces_to_use_network_topology_strategy(
            keyspaces=system_keyspaces,
            replication_factors={dc: len(status[dc].keys()) for dc in status}
        )
        self.prewrite_db_with_data()
        if not skip_optional_stage('main_load'):
            read_thread, write_thread = self.start_stress_during_adding_new_dc()
        # no need to change network topology
        new_node = self.add_node_in_new_dc()

        self.querying_new_node_should_return_no_data(new_node)  # verify issue #8354

        status = self.db_cluster.get_nodetool_status()
        self.reconfigure_keyspaces_to_use_network_topology_strategy(
            keyspaces=system_keyspaces + ["keyspace1"],
            replication_factors={dc: len(status[dc].keys()) for dc in status}
        )

        self.log.info("Running rebuild on each node in new DC")
        new_node.run_nodetool(sub_cmd=f"rebuild -- {list(status.keys())[0]}", publish_event=True)

        self.log.info("Running repair on all nodes")
        for node in self.db_cluster.nodes:
            node.run_nodetool(sub_cmd="repair -pr", publish_event=True)

        if not skip_optional_stage('main_load'):
            # wait for stress to complete
            self.verify_stress_thread(cs_thread_pool=read_thread)
            self.verify_stress_thread(cs_thread_pool=write_thread)

        self.verify_data_can_be_read_from_new_dc(new_node)
        self.log.info("Test completed.")

    def test_add_new_dc_with_zero_nodes(self):
        self.log.info("Starting add new DC with zero nodes test...")
        assert self.params.get('n_db_nodes').endswith(" 0"), "n_db_nodes must be a list and last dc must equal 0"
        system_keyspaces = ["system_distributed", "system_traces"]
        # auth-v2 is used when consistent topology is enabled
        if not self.db_cluster.nodes[0].raft.is_consistent_topology_changes_enabled:
            system_keyspaces.insert(0, "system_auth")

        self.prewrite_db_with_data()

        status = self.db_cluster.get_nodetool_status()

        self.reconfigure_keyspaces_to_use_network_topology_strategy(
            keyspaces=system_keyspaces + ["keyspace1"],
            replication_factors={dc: len(status[dc].keys()) for dc in status}
        )

        self.log.info("Running repair on all nodes")
        for node in self.db_cluster.nodes:
            node.run_nodetool(sub_cmd="repair -pr", publish_event=True)

        # Stop all nodes in 1st dc and check that raft quorum is lost

        status = self.db_cluster.get_nodetool_status()
        nodes_to_region = self.db_cluster.nodes_by_region(nodes=self.db_cluster.data_nodes)
        regions = list(nodes_to_region.keys())
        target_dc_name = regions[0]
        alive_dc_name = regions[1]
        for node in nodes_to_region[target_dc_name]:
            node.stop_scylla()

        node = nodes_to_region[alive_dc_name][0]
        with pytest.raises(ReadBarrierErrorException):
            node.raft.call_read_barrier()

        # Start all nodes in 1st dc
        for node in nodes_to_region[target_dc_name]:
            node.start_scylla()

        self.db_cluster.wait_all_nodes_un()

        # Add new dc with zero node only
        new_node = self.add_zero_node_in_new_dc()

        status = self.db_cluster.get_nodetool_status()
        node_host_ids = []
        node_for_termination = []

        nodes_to_region = self.db_cluster.nodes_by_region(nodes=self.db_cluster.data_nodes)
        regions = list(nodes_to_region.keys())
        target_dc_name = regions[0]
        for node in nodes_to_region[target_dc_name]:
            node_host_ids.append(node.host_id)
            node_for_termination.append(node)
            node.stop_scylla()

        # check that raft quorum is not lost
        new_node.raft.call_read_barrier()
        # restore dc1

        def remove_one_replace_other_nodes_in_DC1():
            new_node.run_nodetool(
                sub_cmd=f"removenode {node_host_ids[0]} --ignore-dead-nodes {','.join(node_host_ids[1:])}")

            self.replace_cluster_node(new_node,
                                      node_host_ids[1],
                                      nodes_to_region[target_dc_name][-1].dc_idx,
                                      dead_node_hostids=node_host_ids[2])

            self.replace_cluster_node(new_node,
                                      node_host_ids[2],
                                      nodes_to_region[target_dc_name][-1].dc_idx)

            # bootstrap new node in 1st dc
            new_data_node = self.add_node_in_new_dc(nodes_to_region[target_dc_name][-1].dc_idx, 3)
            for node in node_for_termination:
                self.db_cluster.terminate_node(node)

            self.db_cluster.wait_all_nodes_un()
            status = self.db_cluster.get_nodetool_status()
            self.log.info("Running rebuild  in restored DC")
            new_data_node.run_nodetool(sub_cmd=f"rebuild -- {list(status.keys())[-1]}", publish_event=True)

            self.log.info("Running repair on all nodes")
            for node in self.db_cluster.nodes:
                node.run_nodetool(sub_cmd="repair -pr", publish_event=True)

            self.verify_data_can_be_read_from_new_dc(new_data_node)
            self.log.info("Test completed.")

        def remove_all_add_new_in_DC1():
            # remove all nodes from DC1
            while node_host_ids:
                remove_host_id = node_host_ids.pop(0)
                if node_host_ids:
                    dead_nodes_param = f" --ignore-dead-nodes {','.join(node_host_ids)}"
                else:
                    dead_nodes_param = ""

                new_node.run_nodetool(
                    sub_cmd=f"removenode {remove_host_id}{dead_nodes_param}")

            # bootstrap new node in 1st dc
            new_data_node1 = self.add_node_in_new_dc(nodes_to_region[target_dc_name][-1].dc_idx, 1)
            new_data_node2 = self.add_node_in_new_dc(nodes_to_region[target_dc_name][-1].dc_idx, 2)
            new_data_node3 = self.add_node_in_new_dc(nodes_to_region[target_dc_name][-1].dc_idx, 3)

            for node in node_for_termination:
                self.db_cluster.terminate_node(node)

            self.db_cluster.wait_all_nodes_un()
            status = self.db_cluster.get_nodetool_status()
            self.log.info("Running rebuild  in restored DC")
            new_data_node1.run_nodetool(sub_cmd=f"rebuild -- {list(status.keys())[-1]}", publish_event=True)
            new_data_node2.run_nodetool(sub_cmd=f"rebuild -- {list(status.keys())[-1]}", publish_event=True)
            new_data_node3.run_nodetool(sub_cmd=f"rebuild -- {list(status.keys())[-1]}", publish_event=True)

        def replace_all_nodes_in_DC1():
            self.replace_cluster_node(new_node,
                                      node_host_ids[0],
                                      nodes_to_region[target_dc_name][-1].dc_idx,
                                      dead_node_hostids=",".join(node_host_ids[1:]))

            self.replace_cluster_node(new_node,
                                      node_host_ids[1],
                                      nodes_to_region[target_dc_name][-1].dc_idx,
                                      dead_node_hostids=node_host_ids[2])

            self.replace_cluster_node(new_node,
                                      node_host_ids[2],
                                      nodes_to_region[target_dc_name][-1].dc_idx)

            # bootstrap new node in 1st dc
            new_data_node = self.add_node_in_new_dc(nodes_to_region[target_dc_name][-1].dc_idx, 3)
            for node in node_for_termination:
                self.db_cluster.terminate_node(node)

            self.db_cluster.wait_all_nodes_un()
            status = self.db_cluster.get_nodetool_status()
            self.log.info("Running rebuild  in restored DC")
            new_data_node.run_nodetool(sub_cmd=f"rebuild -- {list(status.keys())[-1]}", publish_event=True)

            self.log.info("Running repair on all nodes")
            for node in self.db_cluster.nodes:
                node.run_nodetool(sub_cmd="repair -pr", publish_event=True)

            self.verify_data_can_be_read_from_new_dc(new_data_node)
            self.log.info("Test completed.")

        # remove_all_add_new_in_DC1()
        replace_all_nodes_in_DC1()

    def reconfigure_keyspaces_to_use_network_topology_strategy(self, keyspaces: List[str], replication_factors: dict[str, int]) -> None:
        node = self.db_cluster.nodes[0]
        self.log.info("Reconfiguring keyspace Replication Strategy")
        network_topology_strategy = NetworkTopologyReplicationStrategy(
            **replication_factors)
        for keyspace in keyspaces:
            cql = f"ALTER KEYSPACE {keyspace} WITH replication = {network_topology_strategy}"
            node.run_cqlsh(cql)
        self.log.info("Replication Strategies for {} reconfigured".format(keyspaces))

    @optional_stage('prepare_write')
    def prewrite_db_with_data(self) -> None:
        self.log.info("Prewriting database...")
        stress_cmd = self.params.get('prepare_write_cmd')
        pre_thread = self.run_stress_thread(stress_cmd=stress_cmd, stats_aggregate_cmds=False, round_robin=False)
        self.verify_stress_thread(cs_thread_pool=pre_thread)
        self.log.info("Database pre write completed")

    def start_stress_during_adding_new_dc(self) -> Tuple[CassandraStressThread, CassandraStressThread]:
        self.log.info("Running stress during adding new DC")
        stress_cmds = self.params.get('stress_cmd')
        read_thread = self.run_stress_thread(stress_cmd=stress_cmds[0], stats_aggregate_cmds=False, round_robin=False)
        write_thread = self.run_stress_thread(stress_cmd=stress_cmds[1], stats_aggregate_cmds=False, round_robin=False)
        self.log.info("Stress during adding DC started")
        return read_thread, write_thread

    def add_node_in_new_dc(self, dc_idx: int = 0, num_of_dc: int = 2) -> BaseNode:
        self.log.info("Adding new node")
        new_node = self.db_cluster.add_nodes(1, dc_idx=dc_idx, enable_auto_bootstrap=True)[0]  # add node
        self.db_cluster.wait_for_init(node_list=[new_node], timeout=900,
                                      check_node_health=False)
        self.db_cluster.wait_for_nodes_up_and_normal(nodes=[new_node])
        self.monitors.reconfigure_scylla_monitoring()

        status = self.db_cluster.get_nodetool_status()
        assert len(status.keys()) == num_of_dc, f"new datacenter was not registered. Cluster status: {status}"
        self.log.info("New DC to cluster has been added")
        return new_node

    def add_zero_node_in_new_dc(self) -> BaseNode:
        if not self.params.get("use_zero_nodes"):
            raise Exception("Zero node support should be enabled")
        self.log.info("Adding new node")
        new_node = self.db_cluster.add_nodes(1, dc_idx=2, enable_auto_bootstrap=True, is_zero_node=True)[0]  # add node
        self.db_cluster.wait_for_init(node_list=[new_node], timeout=900,
                                      check_node_health=True)
        self.db_cluster.wait_for_nodes_up_and_normal(nodes=[new_node])
        self.monitors.reconfigure_scylla_monitoring()

        status = self.db_cluster.get_nodetool_status()
        assert len(status.keys()) == 3, f"new datacenter was not registered. Cluster status: {status}"
        self.log.info("New DC to cluster has been added")
        return new_node

    @optional_stage('post_test_load')
    def verify_data_can_be_read_from_new_dc(self, new_node: BaseNode) -> None:
        self.log.info("Verifying if data has been transferred successfully to the new DC")
        stress_cmd = self.params.get('verify_data_after_entire_test') + f" -node {new_node.ip_address}"
        end_stress = self.run_stress_thread(stress_cmd=stress_cmd, stats_aggregate_cmds=False, round_robin=False)
        self.verify_stress_thread(cs_thread_pool=end_stress)

    def querying_new_node_should_return_no_data(self, new_node: BaseNode) -> None:
        self.log.info("Verifying if querying new node with RF=0 returns no data and does not crash the node. #8354")
        with self.db_cluster.cql_connection_exclusive(new_node) as session:
            statement = SimpleStatement(
                "select * from keyspace1.standard1 limit 1;", consistency_level=ConsistencyLevel.LOCAL_QUORUM,
                fetch_size=10)
            data = session.execute(statement).one()
            assert not data, f"no data should be returned when querying with CL=LOCAL_QUORUM and RF=0. {data}"

    def replace_cluster_node(self, verification_node: BaseNode,
                             host_id: str | None = None,
                             dc_idx: int = 0,
                             dead_node_hostids: str = "",
                             timeout: int | float = 3600 * 8) -> BaseNode:
        """When old_node_ip or host_id are not None then replacement node procedure is initiated"""
        self.log.info("Adding new node to cluster...")
        new_node: BaseNode = skip_on_capacity_issues(self.db_cluster.add_nodes)(
            count=1, dc_idx=dc_idx, enable_auto_bootstrap=True)[0]
        self.monitors.reconfigure_scylla_monitoring()
        with new_node.remote_scylla_yaml() as scylla_yaml:
            scylla_yaml.ignore_dead_nodes_for_replace = dead_node_hostids
        # since we need this logic before starting a node, and in `use_preinstalled_scylla: false` case
        # scylla is not yet installed or target node was terminated, we should use an alive node without nemesis for version,
        # it should be up and with scylla executable available

        new_node.replacement_host_id = host_id

        try:
            with adaptive_timeout(Operations.NEW_NODE, node=verification_node, timeout=timeout):
                self.db_cluster.wait_for_init(node_list=[new_node], timeout=timeout, check_node_health=False)
            self.db_cluster.clean_replacement_node_options(new_node)
            self.db_cluster.set_seeds()
            self.db_cluster.update_seed_provider()
        except (NodeSetupFailed, NodeSetupTimeout):
            self.log.warning("TestConfig of the '%s' failed, removing it from list of nodes" % new_node)
            self.db_cluster.nodes.remove(new_node)
            self.log.warning("Node will not be terminated. Please terminate manually!!!")
            raise

        self.db_cluster.wait_for_nodes_up_and_normal(nodes=[new_node])
        new_node.wait_node_fully_start()
        with new_node.remote_scylla_yaml() as scylla_yaml:
            scylla_yaml.ignore_dead_nodes_for_replace = ""

        return new_node
