#!/usr/bin/env python

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
# Copyright (c) 2019 ScyllaDB

import random
import string
import time

from sdcm.tester import ClusterTester

# performance_regression_row_level_repair_test.py
from sdcm.utils import measure_time, retrying


class PerformanceRegressionRowLevelRepairTest(ClusterTester):
    """
    Test Scylla performance regression with cassandra-stress.

    """

    KEYSPACE_NAME = "ks"
    TABLE_NAME = "cf"
    INT_COLUMNS = 20  # TODO: 99
    PARTITIONS = 30  # TODO: 100
    BIG_PARTITION_IDX = PARTITIONS + 1
    BIG_PARTITION_ROWS = 150  # TODO: 100000
    ROWS_IN_PARTITION = 15  # TODO: 30

    def __init__(self, *args, **kwargs):
        super(PerformanceRegressionRowLevelRepairTest, self).__init__(*args, **kwargs)

    # Util functions ===============================================================================================

    @measure_time
    def _run_repair(self, node):
        self.log.info('Running nodetool repair on {}'.format(node.name))
        result = node.run_nodetool(sub_cmd='repair')
        return result

    def _pre_create_schema_large_scale(self, keyspace_num=1, in_memory=False, scylla_encryption_options=None):
        """
        For cases we are testing many keyspaces and tables, It's a possibility that we will do it better and faster than
        cassandra-stress.
        """

        self.log.debug('Pre Creating Schema for c-s with {} keyspaces'.format(keyspace_num))
        for i in xrange(1, keyspace_num + 1):
            keyspace_name = 'keyspace{}'.format(i)
            self.create_keyspace(keyspace_name=keyspace_name, replication_factor=3)
            self.log.debug('{} Created'.format(keyspace_name))
            table_name = "{}.standard1".format(keyspace_name)
            self.create_table(name=table_name, key_type='blob', read_repair=0.0, compact_storage=True,
                              columns={'"C0"': 'blob'},
                              in_memory=in_memory, scylla_encryption_options=scylla_encryption_options)

    def preload_data(self):
        # if test require a pre-population of data
        prepare_write_cmd = self.params.get('prepare_write_cmd')
        if prepare_write_cmd:
            self.create_test_stats(sub_type='write-prepare')
            stress_queue = list()
            params = {'prefix': 'preload-'}
            # Check if the prepare_cmd is a list of commands
            if not isinstance(prepare_write_cmd, basestring) and len(prepare_write_cmd) > 1:
                # Check if it should be round_robin across loaders
                if self.params.get('round_robin', default='').lower() == 'true':
                    self.log.debug('Populating data using round_robin')
                    params.update({'stress_num': 1, 'round_robin': True})

                for stress_cmd in prepare_write_cmd:
                    params.update({'stress_cmd': stress_cmd})

                    # Run all stress commands
                    params.update(dict(stats_aggregate_cmds=False))
                    self.log.debug('RUNNING stress cmd: {}'.format(stress_cmd))
                    stress_queue.append(self.run_stress_thread(**params))

            # One stress cmd command
            else:
                stress_queue.append(self.run_stress_thread(stress_cmd=prepare_write_cmd, stress_num=1,
                                                           prefix='preload-', stats_aggregate_cmds=False))

            for stress in stress_queue:
                self.get_stress_results(queue=stress, store_results=False)

            self.update_test_details()
        else:
            self.log.warning("No prepare command defined in YAML!")

    # Util functions ===============================================================================================

    # row-level-repair Test - measuring node-repair-time
    def test_row_level_repair(self):
        """
        Test steps:

        1. TODO: docstring
        """


        # Util functions ===============================================================================================

        def _get_cql_session(node=None):
            node = node or self.db_cluster.nodes[0]
            return self.cql_connection_patient(node).session

        def _get_cql_session_and_use_keyspace(node=None, keyspace=self.KEYSPACE_NAME):
            session = _get_cql_session(node=node)
            session.execute("USE {}".format(keyspace))
            return session

        def _create_update_command(column_expr, pk, ck, table_name=self.TABLE_NAME):
            cql_update_cmd = 'update {table_name} set {column_expr} where pk={pk} and ck={ck}'.format(**locals())
            self.log.debug("Generated CQL update command of: {}".format(cql_update_cmd))
            return cql_update_cmd

        def _update_table(table_name=self.TABLE_NAME, keyspace=self.KEYSPACE_NAME):
            self.log.debug('Update table')
            session = _get_cql_session_and_use_keyspace(keyspace=keyspace)

            num_of_updates = 2  # TODO: 50
            num_of_total_updates = num_of_updates * 2  # updating both a big partition and the largest partition.
            stmts = []
            self.log.debug(
                "Going to generate {} CQL updates, {} for big partition and for largest partition each".format(
                    num_of_total_updates, num_of_updates))
            for _ in range(num_of_updates):
                # Update/delete int columns to a random big partition
                column = random.randint(1, self.INT_COLUMNS - 1)
                column_name = 'c{}'.format(column)
                new_value = random.choice(['NULL', random.randint(0, 500000)])
                column_expr = '{} = {}'.format(column_name, new_value)
                stmts.append(_create_update_command(column_expr=column_expr,
                                                    pk=random.randint(1, self.PARTITIONS),
                                                    ck=random.randint(1, self.BIG_PARTITION_ROWS),
                                                    table_name=table_name))

                # Update/delete row inside the largest partition
                stmts.append(_create_update_command(column_expr=column_expr,
                                                    pk=self.BIG_PARTITION_IDX, ck=random.randint(1, self.BIG_PARTITION_ROWS),
                                                    table_name=table_name))

            for stmt in stmts:
                session.execute(stmt)

        def _pre_create_schema(table_name=self.TABLE_NAME, keyspace=self.KEYSPACE_NAME, int_columns=self.INT_COLUMNS):
            self.log.debug('Create schema')
            session = _get_cql_session()

            stmt = "CREATE KEYSPACE IF NOT EXISTS {}".format(
                keyspace) + " WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3}"
            session.execute(stmt)
            session.execute("USE {}".format(keyspace))
            stmt = 'create table {} (pk int, ck int, {}, clist list<int>, cset set<text>, cmap map<int, text>, ' \
                   'PRIMARY KEY(pk, ck))'.format(table_name, ', '.join('c%d int' % i for i in xrange(1, int_columns)))
            session.execute(stmt)

        def _pre_fill_schema(table_name=self.TABLE_NAME, partitions=self.PARTITIONS, rows_in_partition=self.ROWS_IN_PARTITION):

            self.log.debug('Prefill schema')
            session = _get_cql_session_and_use_keyspace()

            # Prefill
            self.log.info('Create {} partitions with {} rows'.format(partitions, rows_in_partition))
            for i in xrange(1, partitions + 1):
                for k in xrange(1, rows_in_partition + 1):
                    str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
                    stmt = 'insert into {table_name} (pk, ck, {columns}, clist, cset, cmap) values ({ilist}, {klist}, {int_values}, ' \
                           '[{ilist}, {klist}], ' \
                           '{open}{set_value}{close}, {map_value})'.format(table_name=table_name,
                                                                           columns=', '.join(
                                                                               'c%d' % l for l in
                                                                               xrange(1, self.INT_COLUMNS)),
                                                                           int_values=', '.join(
                                                                               '%d' % l for l in
                                                                               xrange(1, self.INT_COLUMNS)),
                                                                           ilist=i, klist=k, open='{\'',
                                                                           set_value=str, close='\'}',
                                                                           map_value='{%d: \'%s\'}' % (k, str)
                                                                           )
                    session.execute(stmt)

            # Pre-fill the largest partition
            big_partition = partitions + 1
            big_partition_rows = self.BIG_PARTITION_ROWS  # TODO: 100000
            total_rows = partitions * rows_in_partition + big_partition_rows
            self.log.info('Create partition where pk = {} with {} rows'.format(big_partition, big_partition_rows))
            for k in xrange(1, big_partition_rows + 1):
                str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
                stmt = 'insert into {table_name} (pk, ck, {columns}, clist, cset, cmap) values ({ilist}, {klist}, {int_values}, ' \
                       '[{ilist}, {klist}], ' \
                       '{open}{set_value}{close}, {map_value})'.format(table_name=table_name,
                                                                       columns=', '.join(
                                                                           'c%d' % l for l in xrange(1, self.INT_COLUMNS)),
                                                                       int_values=', '.join(
                                                                           '%d' % l for l in xrange(1, self.INT_COLUMNS)),
                                                                       ilist=big_partition, klist=k, open='{\'',
                                                                       set_value=str, close='\'}',
                                                                       map_value='{%d: \'%s\'}' % (k, str)
                                                                       )
                session.execute(stmt)

        # Util functions ===============================================================================================

        dict_specific_tested_stats = {'repair_runtime': -1}
        self.create_test_stats(specific_tested_stats=dict_specific_tested_stats)

        _pre_create_schema()
        _pre_fill_schema()

        self.log.info('Starting c-s/s-b write workload')
        prepare_write_cmd = self.params.get('prepare_write_cmd')
        prepare_cmd_queue = self.run_stress_thread(stress_cmd=prepare_write_cmd, duration=5)

        node2 = self.db_cluster.nodes[1]
        self.log.info('Stopping node-2 ({}) before updating cluster data'.format(node2.name))
        node2.stop_scylla_server()

        self.log.info('Updating cluster data when node2 ({}) is down'.format(node2.name))
        _update_table()

        self.log.info('Starting node-2 ({}) after updated cluster data'.format(node2.name))
        node2.start_scylla_server()

        self.log.info('Run Repair on node: {}'.format(node2.name))
        repair_time, res = self._run_repair(node=node2)
        self.log.info('Repair time on node: {} is: {}'.format(node2.name, repair_time))

        dict_specific_tested_stats['repair_runtime'] = repair_time
        self.update_test_details(scylla_conf=True, dict_specific_tested_stats=dict_specific_tested_stats)

        self.check_specified_stats_regression(dict_specific_tested_stats=dict_specific_tested_stats)

    # Global Util functions ===============================================================================================

    def get_used_capacity(self, node):
        # (sum(node_filesystem_size{mountpoint="/var/lib/scylla"})-sum(node_filesystem_avail{mountpoint="/var/lib/scylla"}))
        filesystem_capacity_query = 'sum(node_filesystem_size{{mountpoint="{0.scylla_dir}", ' \
                                    'instance=~"{1.private_ip_address}"}})'.format(self, node)

        self.log.debug("filesystem_capacity_query: {}".format(filesystem_capacity_query))

        fs_size_res = self.prometheusDB.query(query=filesystem_capacity_query, start=time.time(), end=time.time())
        kb_size = 2 ** 10
        mb_size = kb_size * 1024
        gb_size = mb_size * 1024
        fs_size_gb = int(fs_size_res[0]["values"][0][1]) / gb_size
        self.log.debug("fs_cap_res: {}".format(fs_size_res))
        used_capacity_query = '(sum(node_filesystem_size{{mountpoint="{0.scylla_dir}", ' \
                              'instance=~"{1.private_ip_address}"}})-sum(node_filesystem_avail{{mountpoint="{0.scylla_dir}", ' \
                              'instance=~"{1.private_ip_address}"}}))'.format(self, node)

        self.log.debug("used_capacity_query: {}".format(used_capacity_query))

        used_cap_res = self.prometheusDB.query(query=used_capacity_query, start=time.time(), end=time.time())
        self.log.debug("used_cap_res: {}".format(used_cap_res))

        assert used_cap_res, "No results from Prometheus"
        used_size_mb = float(used_cap_res[0]["values"][0][1]) / float(mb_size)
        used_size_gb = float(used_size_mb/1024)
        self.log.debug("The used filesystem capacity on node {} is: {} MB/ {} GB".format(node.public_ip_address , used_size_mb, used_size_gb))
        return used_size_mb

    @retrying(n=40, sleep_time=60, allowed_exceptions=(AssertionError,))
    def _wait_no_compactions_running(self):
        q = "sum(scylla_compaction_manager_compactions{})"
        now = time.time()
        results = self.prometheusDB.query(query=q, start=now - 60, end=now)
        self.log.debug("scylla_compaction_manager_compactions: %s" % results)
        # if all are zeros the result will be False, otherwise there are still compactions
        if results:
            assert any([float(v[1]) for v in results[0]["values"]]) is False, \
                "Waiting until all compactions settle down"

    def _disable_hinted_handoff(self):

        yaml_file = "/etc/scylla/scylla.yaml"
        for node in self.db_cluster.nodes: # disable hinted handoff
            res = node.remoter.run('sudo echo hinted_handoff_enabled: false >> {}'.format(yaml_file))
            self.log.debug("Scylla YAML configuration read from: {} {} is:".format(node.public_ip_address, yaml_file))
            res = node.remoter.run('sudo cat {}'.format(yaml_file))
            node.stop_scylla_server()
            node.start_scylla_server()

    def _stop_all_nodes_except_for(self, node):
        self.log.debug("Stopping all nodes except for: {}".format(node.name))

        for c_node in [n for n in self.db_cluster.nodes if n != node]:
            self.log.debug("Stopping node: {}".format(c_node.name))
            c_node.stop_scylla_server()

    def _start_all_nodes_except_for(self, node):
        self.log.debug("Starting all nodes except for: {}".format(node.name))
        node_list = [n for n in self.db_cluster.nodes if n != node]

        # Start down seed nodes first, if exists
        for c_node in [n for n in node_list if n.is_seed]:
            self.log.debug("Starting seed node: {}".format(c_node.name))
            c_node.start_scylla_server()

        for c_node in [n for n in node_list if not n.is_seed]:
            self.log.debug("Starting non-seed node: {}".format(c_node.name))
            c_node.start_scylla_server()

        node.wait_db_up()

    def _print_nodes_used_capacity(self):
        for node in self.db_cluster.nodes:
            used_capacity = self.get_used_capacity(node=node)
            self.log.debug("Node {} ({}) used capacity is: {}".format(node.name, node.private_ip_address, used_capacity))
    # Global Util functions ===============================================================================================


    def test_row_level_repair_single_node_diff(self):
        """
        Start 3 nodes, create keyspace with rf = 3, disable hinted hand off
        requires: export SCT_HINTED_HANDOFF_DISABLED=true
        :return:
        """
        dict_specific_tested_stats = {'repair_runtime': -1}
        self.create_test_stats(specific_tested_stats=dict_specific_tested_stats)

        self._pre_create_schema_large_scale()
        node1, node2, node3 = self.db_cluster.nodes

        for node in self.db_cluster.nodes:
            used_capacity = self.get_used_capacity(node=node)
            self.log.debug("Node {} initial used capacity is: {}".format(node.public_ip_address, used_capacity))

        self._disable_hinted_handoff()

        self.log.info('Stopping node-3 ({}) before updating cluster data'.format(node3.name))
        node3.stop_scylla_server()
        self.log.info('Updating cluster data when node3 ({}) is down'.format(node3.name))
        self.log.info('Starting c-s/s-b write workload')
        self.preload_data()
        self.wait_no_compactions_running()

        self.log.info('Starting node-3 ({}) after updated cluster data'.format(node3.name))
        node3.start_scylla_server()

        for node in self.db_cluster.nodes:
            used_capacity = self.get_used_capacity(node=node)
            self.log.debug("Node {} used capacity after pre-load data is: {}".format(node.public_ip_address, used_capacity))

        self.log.info('Run Repair on node: {} , 0% synced'.format(node3.name))
        repair_time, res = self._run_repair(node=node3)
        self.log.info('Repair (0% synced) time on node: {} is: {}'.format(node3.name, repair_time))

        dict_specific_tested_stats['repair_runtime_all_diff'] = repair_time
        # self.update_test_details(scylla_conf=True, dict_specific_tested_stats=dict_specific_tested_stats)

        self.wait_no_compactions_running()

        self.log.info('Run Repair on node: {} , 100% synced'.format(node3.name))
        repair_time, res = self._run_repair(node=node3)
        self.log.info('Repair (100% synced) time on node: {} is: {}'.format(node3.name, repair_time))

        dict_specific_tested_stats['repair_runtime_no_diff'] = repair_time
        self.update_test_details(scylla_conf=True, dict_specific_tested_stats=dict_specific_tested_stats)

        self.check_specified_stats_regression(dict_specific_tested_stats=dict_specific_tested_stats)

    def test_row_level_repair_3_nodes_small_diff(self):
        """
        Start 3 nodes, create keyspace with rf = 3, disable hinted hand off
        requires: export SCT_HINTED_HANDOFF_DISABLED=true
        :return:
        """

        base_distinct_write_cmd = "cassandra-stress write no-warmup cl=ONE n=1000000 -schema 'replication(factor=3)' -port jmx=6868 -mode cql3 native -rate threads=200 -col 'size=FIXED(1024) n=FIXED(1)'"
        sequence_current_index = 1000000000
        sequence_range = 1000000
        dict_specific_tested_stats = {'repair_runtime': -1}
        self.create_test_stats(specific_tested_stats=dict_specific_tested_stats)

        self._pre_create_schema_large_scale()
        node1, node2, node3 = self.db_cluster.nodes
        self._disable_hinted_handoff()
        self._print_nodes_used_capacity()
        for node in [node1, node2, node3]:
            self.log.info('Stopping all other nodes before updating {}'.format(node.name))
            self._stop_all_nodes_except_for(node=node)
            self.log.info('Updating cluster data only for {}'.format(node.name))
            distinct_write_cmd = "{} -pop seq={}..{} -node {}".format(base_distinct_write_cmd, sequence_current_index+1, sequence_current_index+sequence_range, node.private_ip_address)
            prepare_cmd_queue = self.run_stress(stress_cmd=distinct_write_cmd)
            self._start_all_nodes_except_for(node=node)
            sequence_current_index += sequence_range

        time.sleep(60) # wait for capacity metrics to be fully updated for all nodes.
        self.log.debug("Nodes distinct used capacity is")
        self._print_nodes_used_capacity()

        self.log.info('Starting c-s/s-b write workload')
        self.preload_data()
        self._wait_no_compactions_running()

        self.log.debug("Nodes total used capacity before starting repair is:")
        self._print_nodes_used_capacity()

        self.log.info('Run Repair on node: {} , 99.8% synced'.format(node3.name))
        repair_time, res = self._run_repair(node=node3)

        for node in self.db_cluster.nodes:
            used_capacity = self.get_used_capacity(node=node)
            self.log.debug("Node {} total used capacity after repair complete is: {}".format(node.public_ip_address, used_capacity))

        self.log.info('Repair (99.8% synced) time on node: {} is: {}'.format(node3.name, repair_time))

        dict_specific_tested_stats['repair_runtime_small_diff'] = repair_time
        # self.update_test_details(scylla_conf=True, dict_specific_tested_stats=dict_specific_tested_stats)

        self.update_test_details(scylla_conf=True, dict_specific_tested_stats=dict_specific_tested_stats)

        self.check_specified_stats_regression(dict_specific_tested_stats=dict_specific_tested_stats)