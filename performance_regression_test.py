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
# Copyright (c) 2016 ScyllaDB


import os
import time

from enum import Enum

import yaml
from cassandra.query import SimpleStatement  # pylint: disable=no-name-in-module

from upgrade_test import UpgradeTest
from sdcm.tester import ClusterTester, teardown_on_exception
from sdcm.sct_events import Severity
from sdcm.sct_events.filters import EventsSeverityChangerFilter
from sdcm.sct_events.loaders import CassandraStressEvent
from sdcm.sct_events.system import HWPerforanceEvent, InfoEvent
from sdcm.utils.decorators import log_run_info, latency_calculator_decorator, optional_stage
from sdcm.utils.csrangehistogram import CSHistogramTagTypes
from sdcm.utils.nemesis_utils.indexes import wait_for_view_to_be_built

KB = 1024


class PerformanceTestWorkload(Enum):
    WRITE = "write"
    READ = "read"
    MIXED = "mixed"


class PerformanceTestType(Enum):
    THROUGHPUT = "throughput"
    LATENCY = "latency"


class PerformanceRegressionTest(ClusterTester):  # pylint: disable=too-many-public-methods

    """
    Test Scylla performance regression with cassandra-stress.
    """

    str_pattern = '%8s%16s%10s%14s%16s%12s%12s%14s%16s%16s'
    ops_threshold_prc = 200
    start_ops = 10_000
    throttle_step = 10_000
    max_ops = 200_000

    def __init__(self, *args):
        # need to remove the email_data.json file, as in the builders, it will accumulate and it will send multiple
        # emails for each test. When we move to use SCT Runners, it won't be necessary.
        self._clean_email_data()
        super().__init__(*args)

    @teardown_on_exception
    @log_run_info
    def setUp(self):
        if es_index := self.params.get("custom_es_index"):
            self._test_index = es_index

        super().setUp()
        if self.params.get("run_db_node_benchmarks"):
            self.log.info("Validate node benchmarks results")
            compare_results = self.db_cluster.get_node_benchmarks_results() or {}
            ready_nodes = []
            for node, results in compare_results.items():
                for item, result in results.items():
                    ready_nodes.append(result["is_within_margin"])
                    if not result["is_within_margin"]:
                        self.log.error("HW performance test on node %s has bad results for %s : %s", node, item, result)
            if not all(ready_nodes):
                err_msg = f"DB Cluster doesn't have equal hw performance result {compare_results}"
                self.log.debug(err_msg)
                if self.params.get("stop_on_hw_perf_failure"):
                    HWPerforanceEvent(message=err_msg, severity=Severity.CRITICAL).publish()
                else:
                    HWPerforanceEvent(message=err_msg, severity=Severity.WARNING).publish()
            else:
                self.log.debug("DB cluster passed hardware performance test")
                HWPerforanceEvent(message="DB cluster passed hardware performance test",
                                  severity=Severity.NORMAL).publish()

    # Helpers

    def display_single_result(self, result):
        self.log.info(self.str_pattern, result['op rate'],
                      result['partition rate'],
                      result['row rate'],
                      result['latency mean'],
                      result['latency median'],
                      result['latency 95th percentile'],
                      result['latency 99th percentile'],
                      result['latency 99.9th percentile'],
                      result['total partitions'],
                      result['total errors'])

    def get_test_xml(self, result, test_name=''):
        test_content = """
  <test name="%s: (%s) Loader%s CPU%s Keyspace%s" executed="yes">
    <description>"%s test, ami_id: %s, scylla version:
    %s", hardware: %s</description>
    <targets>
      <target threaded="yes">target-ami_id-%s</target>
      <target threaded="yes">target-version-%s</target>
    </targets>
    <platform name="AWS platform">
      <hardware>%s</hardware>
    </platform>

    <result>
      <success passed="yes" state="1"/>
      <performance unit="kbs" mesure="%s" isRelevant="true" />
      <metrics>
        <op-rate unit="op/s" mesure="%s" isRelevant="true" />
        <partition-rate unit="pk/s" mesure="%s" isRelevant="true" />
        <row-rate unit="row/s" mesure="%s" isRelevant="true" />
        <latency-mean unit="mean" mesure="%s" isRelevant="true" />
        <latency-median unit="med" mesure="%s" isRelevant="true" />
        <l-95th-pct unit=".95" mesure="%s" isRelevant="true" />
        <l-99th-pct unit=".99" mesure="%s" isRelevant="true" />
        <l-99.9th-pct unit=".999" mesure="%s" isRelevant="true" />
        <total_partitions unit="total_partitions" mesure="%s" isRelevant="true" />
        <total_errors unit="total_errors" mesure="%s" isRelevant="true" />
      </metrics>
    </result>
  </test>
""" % (test_name, result['loader_idx'],
            result['loader_idx'],
            result['cpu_idx'],
            result['keyspace_idx'],
            test_name,
            self.params.get('ami_id_db_scylla'),
            self.params.get('ami_id_db_scylla_desc'),
            self.params.get('instance_type_db'),
            self.params.get('ami_id_db_scylla'),
            self.params.get('ami_id_db_scylla_desc'),
            self.params.get('instance_type_db'),
            result['op rate'],
            result['op rate'],
            result['partition rate'],
            result['row rate'],
            result['latency mean'],
            result['latency median'],
            result['latency 95th percentile'],
            result['latency 99th percentile'],
            result['latency 99.9th percentile'],
            result['total partitions'],
            result['total errors'])

        return test_content

    def display_results(self, results, test_name=''):
        self.log.info(self.str_pattern, 'op-rate', 'partition-rate',
                      'row-rate', 'latency-mean',
                      'latency-median', 'l-94th-pct',
                      'l-99th-pct', 'l-99.9th-pct',
                      'total-partitions', 'total-err')

        test_xml = ""
        try:
            for single_result in results:
                self.display_single_result(single_result)
                test_xml += self.get_test_xml(single_result, test_name=test_name)

            with open(os.path.join(self.logdir, 'jenkins_perf_PerfPublisher.xml'), 'w', encoding="utf-8") as pref_file:
                content = """<report name="%s report" categ="none">%s</report>""" % (test_name, test_xml)
                pref_file.write(content)
        except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.debug('Failed to display results: {0}'.format(results))
            self.log.debug('Exception: {0}'.format(ex))

    def _workload(self, stress_cmd, stress_num, test_name, sub_type=None, keyspace_num=1, prefix='', debug_message='',  # pylint: disable=too-many-arguments
                  save_stats=True):
        if debug_message:
            self.log.debug(debug_message)

        if save_stats:
            if not self.exists():
                self.create_test_stats(sub_type=sub_type)
        stress_queue = self.run_stress_thread(stress_cmd=stress_cmd, stress_num=stress_num, keyspace_num=keyspace_num,
                                              prefix=prefix, stats_aggregate_cmds=False)
        results = self.get_stress_results(queue=stress_queue, store_results=True)
        if save_stats:
            self.update_test_details(scylla_conf=True)
            self.display_results(results, test_name=test_name)
            self.check_regression()
            total_ops = self._get_total_ops()
            self.log.debug('Total ops: {}'.format(total_ops))
            return total_ops
        return None

    def _get_total_ops(self):
        return self._stats['results']['stats_total']['op rate']

    @staticmethod
    def _clean_email_data():
        email_data_path = 'email_data.json'
        with open(email_data_path, 'w', encoding="utf-8"):
            pass

    def _stop_load_when_nemesis_threads_end(self):
        for nemesis_thread in self.db_cluster.nemesis_threads:
            nemesis_thread.join()
        with EventsSeverityChangerFilter(new_severity=Severity.NORMAL,  # killing stress creates Critical error
                                         event_class=CassandraStressEvent,
                                         extra_time_to_expiration=60):
            self.loaders.kill_stress_thread()

    def stop_load_during_nemesis(self):
        with EventsSeverityChangerFilter(new_severity=Severity.NORMAL,  # killing stress creates Critical error
                                         event_class=CassandraStressEvent,
                                         extra_time_to_expiration=60):
            self.loaders.kill_stress_thread()

    @optional_stage('perf_preload_data')
    def preload_data(self, compaction_strategy=None):
        # if test require a pre-population of data
        prepare_write_cmd = self.params.get('prepare_write_cmd')
        if prepare_write_cmd:
            # create new document in ES with doc_id = test_id + timestamp
            # allow to correctly save results for future compare
            self.create_test_stats(sub_type='write-prepare', doc_id_with_timestamp=True)
            stress_queue = []
            params = {'prefix': 'preload-'}
            # Check if the prepare_cmd is a list of commands
            if isinstance(prepare_write_cmd, list):
                if len(prepare_write_cmd) == 1:
                    prepare_write_cmd = prepare_write_cmd[0]
            if isinstance(prepare_write_cmd, list):
                # Check if it should be round_robin across loaders
                if self.params.get('round_robin'):
                    self.log.debug('Populating data using round_robin')
                    params.update({'stress_num': 1, 'round_robin': True})
                if compaction_strategy:
                    self.log.debug('Next compaction strategy will be used %s', compaction_strategy)
                    params['compaction_strategy'] = compaction_strategy

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

        if post_prepare_cql_cmds := self.params.get('post_prepare_cql_cmds'):
            self.log.debug("Execute post prepare queries: %s", post_prepare_cql_cmds)
            self._run_cql_commands(post_prepare_cql_cmds)

    def _run_cql_commands(self, cmds, node=None):
        node = node if node else self.db_cluster.nodes[0]

        if not isinstance(cmds, list):
            cmds = [cmds]

        for cmd in cmds:
            # pylint: disable=no-member
            with self.db_cluster.cql_connection_patient(node) as session:
                session.execute(cmd)

    def run_read_workload(self, nemesis=False):
        base_cmd_r = self.params.get('stress_cmd_r')
        # create new document in ES with doc_id = test_id + timestamp
        # allow to correctly save results for future compare
        self.create_test_stats(sub_type='read', doc_id_with_timestamp=True)
        stress_queue = self.run_stress_thread(stress_cmd=base_cmd_r, stress_num=1, stats_aggregate_cmds=False)
        if nemesis:
            interval = self.params.get('nemesis_interval')
            time.sleep(interval)  # Sleeping one interval before starting the nemesis
            self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(), tester_obj=self)
            self.db_cluster.start_nemesis(interval=interval)
        results = self.get_stress_results(queue=stress_queue)

        self.build_histogram(PerformanceTestWorkload.READ, PerformanceTestType.LATENCY)
        self.update_test_details()
        self.display_results(results, test_name='test_latency' if not nemesis else 'test_latency_with_nemesis')
        self.check_regression()

    def run_write_workload(self, nemesis=False):
        base_cmd_w = self.params.get('stress_cmd_w')
        # create new document in ES with doc_id = test_id + timestamp
        # allow to correctly save results for future compare
        self.create_test_stats(sub_type='write', doc_id_with_timestamp=True)
        stress_queue = self.run_stress_thread(stress_cmd=base_cmd_w, stress_num=1, stats_aggregate_cmds=False)
        if nemesis:
            self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(), tester_obj=self)
            self.db_cluster.start_nemesis(interval=self.params.get('nemesis_interval'))
        results = self.get_stress_results(queue=stress_queue)
        self.build_histogram(PerformanceTestWorkload.WRITE, PerformanceTestType.LATENCY)
        self.update_test_details()
        self.display_results(results, test_name='test_latency')
        self.check_regression()

    def run_mixed_workload(self, nemesis=False):
        base_cmd_m = self.params.get('stress_cmd_m')
        # create new document in ES with doc_id = test_id + timestamp
        # allow to correctly save results for future compare
        self.create_test_stats(sub_type='mixed', doc_id_with_timestamp=True)
        stress_queue = self.run_stress_thread(stress_cmd=base_cmd_m, stress_num=1, stats_aggregate_cmds=False)
        if nemesis:
            self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(), tester_obj=self)
            self.db_cluster.start_nemesis(interval=self.params.get('nemesis_interval'))
        results = self.get_stress_results(queue=stress_queue)
        self.build_histogram(PerformanceTestWorkload.MIXED, PerformanceTestType.LATENCY)
        self.update_test_details(scylla_conf=True)
        self.display_results(results, test_name='test_latency')
        self.check_regression()

    def run_workload(self, stress_cmd, nemesis=False, sub_type=None):
        # create new document in ES with doc_id = test_id
        # allow to correctly save results for future compare
        self.stress_cmd = stress_cmd
        test_index = f'latency-during-ops-{sub_type}'
        self.create_test_stats(sub_type=sub_type, append_sub_test_to_name=False, test_index=test_index)
        stress_queue = self.run_stress_thread(stress_cmd=stress_cmd, stress_num=1, stats_aggregate_cmds=False)
        if nemesis:
            interval = self.params.get('nemesis_interval')
            time.sleep(interval * 60)  # Sleeping one interval (in minutes) before starting the nemesis
            self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(), tester_obj=self)
            self.db_cluster.start_nemesis(interval=interval, cycles_count=1)
            self._stop_load_when_nemesis_threads_end()
        results = self.get_stress_results(queue=stress_queue)
        self.update_test_details(scrap_metrics_step=60)
        self.display_results(results, test_name='test_latency' if not nemesis else 'test_latency_with_nemesis')
        check_latency = self.check_regression if not nemesis else self.check_latency_during_ops
        check_latency()

    def prepare_mv(self, on_populated=False):
        with self.db_cluster.cql_connection_patient_exclusive(self.db_cluster.nodes[0]) as session:

            ks_name = 'keyspace1'
            base_table_name = 'standard1'
            if not on_populated:
                # Truncate base table before materialized view creation
                self.log.debug('Truncate base table: {0}.{1}'.format(ks_name, base_table_name))
                self.truncate_cf(ks_name, base_table_name, session)

            # Create materialized view
            view_name = base_table_name + '_mv'
            self.log.debug('Create materialized view: {0}.{1}'.format(ks_name, view_name))
            self.create_materialized_view(ks_name, base_table_name, view_name, ['"C0"'], ['key'], session,
                                          mv_columns=['"C0"', 'key'])

            # Wait for the materialized view is built
            self._wait_for_view(self.db_cluster, session, ks_name, view_name)

    def _write_with_mv(self, on_populated):
        """
        Test steps:

        1. Run a write workload
        2. Create materialized view
        3. Run a write workload
        """
        test_name = 'test_write_with_mv_{}populated'.format('' if on_populated else 'not_')
        base_cmd_w = self.params.get('stress_cmd_w')

        # Run a write workload without MV
        ops_without_mv = self._workload(stress_cmd=base_cmd_w, stress_num=2, sub_type='write_without_mv',
                                        test_name=test_name, keyspace_num=1,
                                        debug_message='First write cassandra-stress command: {}'.format(base_cmd_w))

        # Create MV
        self.prepare_mv(on_populated=on_populated)

        # Start cassandra-stress writes again now with MV
        ops_with_mv = self._workload(stress_cmd=base_cmd_w, stress_num=2, sub_type='write_with_mv',
                                     test_name=test_name, keyspace_num=1,
                                     debug_message='Second write cassandra-stress command: {}'.format(base_cmd_w))

        self.assert_mv_performance(ops_without_mv, ops_with_mv,
                                   'Throughput of run with materialized view is more than {} times lower then '
                                   'throughput of run without materialized view'.format(self.ops_threshold_prc/100))

    def _read_with_mv(self, on_populated):
        """
        Test steps:

        1. Run a write workload as a preparation
        2. Run a read workload
        3. Create MV
        4. Run a read workload again
        """
        test_name = 'test_read_with_mv_{}populated'.format('' if on_populated else 'not_')
        base_cmd_p = self.params.get('prepare_write_cmd')
        base_cmd_w = self.params.get('stress_cmd_w')
        base_cmd_r = self.params.get('stress_cmd_r')

        self.create_test_stats()
        # prepare schema and data before read
        self._workload(stress_cmd=base_cmd_p, stress_num=2, test_name=test_name, prefix='preload-', keyspace_num=1,
                       debug_message='Prepare the test, run cassandra-stress command: {}'.format(base_cmd_p),
                       save_stats=False)

        # run a read workload
        ops_without_mv = self._workload(stress_cmd=base_cmd_r, stress_num=2, sub_type='read_without_mv',
                                        test_name=test_name, keyspace_num=1,
                                        debug_message='First read cassandra-stress command: {}'.format(base_cmd_r))

        self.prepare_mv(on_populated=on_populated)

        # If the MV was created on the empty base table, populate it before reads
        if not on_populated:
            self._workload(stress_cmd=base_cmd_w, stress_num=2, test_name=test_name, prefix='preload-', keyspace_num=1,
                           debug_message='Prepare test before second cassandra-stress command: {}'.format(base_cmd_w),
                           save_stats=False)

        # run a read workload
        ops_with_mv = self._workload(stress_cmd=base_cmd_r, stress_num=2, sub_type='read_with_mv',
                                     test_name=test_name, keyspace_num=1,
                                     debug_message='Second read cassandra-stress command: {}'.format(base_cmd_r))

        self.assert_mv_performance(ops_without_mv, ops_with_mv,
                                   'Throughput of run with materialized view is more than {} times lower then '
                                   'throughput of run without materialized view'.format(self.ops_threshold_prc/100))

    def _mixed_with_mv(self, on_populated):
        """
        Test steps:

        1. Run a write workload as a preparation
        2. Run a mixed workload
        """
        test_name = 'test_mixed_with_mv_{}populated'.format('' if on_populated else 'not_')
        base_cmd_p = self.params.get('prepare_write_cmd')
        base_cmd_m = self.params.get('stress_cmd_m')

        self.create_test_stats()
        # run a write workload as a preparation
        self._workload(stress_cmd=base_cmd_p, stress_num=2, test_name=test_name, keyspace_num=1, prefix='preload-',
                       debug_message='Prepare the test, run cassandra-stress command: {}'.format(base_cmd_p),
                       save_stats=False)

        # run a mixed workload without MV
        ops_without_mv = self._workload(stress_cmd=base_cmd_m, stress_num=2, sub_type='mixed_without_mv',
                                        test_name=test_name, keyspace_num=1,
                                        debug_message='First mixed cassandra-stress command: {}'.format(base_cmd_m))

        self.prepare_mv(on_populated=on_populated)

        # run a mixed workload with MV
        ops_with_mv = self._workload(stress_cmd=base_cmd_p, stress_num=2, sub_type='mixed_with_mv',
                                     test_name=test_name, keyspace_num=1,
                                     debug_message='Second start of mixed cassandra-stress command: {}'.format(
                                         base_cmd_p))

        self.assert_mv_performance(ops_without_mv, ops_with_mv,
                                   'Throughput of stress run with materialized view is more than {} times lower then '
                                   'throughput of stress run without materialized view'.format(
                                       self.ops_threshold_prc / 100))

    def assert_mv_performance(self, ops_without_mv, ops_with_mv, failure_message):
        self.log.debug('Performance results. Ops without MV: {0}; Ops with MV: {1}'.format(ops_without_mv, ops_with_mv))
        self.assertLessEqual(ops_without_mv, (ops_with_mv * self.ops_threshold_prc) / 100, failure_message)

    def _scylla_bench_prepare_table(self):
        node = self.db_cluster.nodes[0]
        with self.db_cluster.cql_connection_patient(node) as session:
            session.execute("""
                CREATE KEYSPACE scylla_bench WITH replication = {'class': 'NetworkTopologyStrategy', 'replication_factor': 3}
                AND durable_writes = true;
            """)
            session.execute("""
                CREATE TABLE scylla_bench.test (
                    pk bigint,
                    ck bigint,
                    v blob,
                    PRIMARY KEY (pk, ck)
                ) WITH CLUSTERING ORDER BY (ck ASC)
                    AND compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_size': '60',
                    'compaction_window_unit': 'MINUTES'}
                    AND bloom_filter_fp_chance = 0.01
                    AND caching = {'keys': 'ALL', 'rows_per_partition': 'ALL'}
                    AND comment = ''
                    AND compression = {}
                    AND crc_check_chance = 1.0
                    AND dclocal_read_repair_chance = 0.1
                    AND default_time_to_live = 86400
                    AND gc_grace_seconds = 0
                    AND max_index_interval = 2048
                    AND memtable_flush_period_in_ms = 0
                    AND min_index_interval = 128
                    AND read_repair_chance = 0.0
                    AND speculative_retry = 'NONE';
            """)

    # Base Tests
    def test_write(self):
        """
        Test steps:

        1. Run a write workload
        """
        # run a write workload
        base_cmd_w = self.params.get('stress_cmd_w')
        stress_multiplier = self.params.get('stress_multiplier')
        if stress_multiplier_w := self.params.get("stress_multiplier_w"):
            stress_multiplier = stress_multiplier_w
        # create new document in ES with doc_id = test_id + timestamp
        # allow to correctly save results for future compare
        self.create_test_stats(doc_id_with_timestamp=True)
        self.run_fstrim_on_all_db_nodes()

        # run a workload
        stress_queue = self.run_stress_thread(
            stress_cmd=base_cmd_w, stress_num=stress_multiplier, stats_aggregate_cmds=False)
        results = self.get_stress_results(queue=stress_queue)

        self.build_histogram(PerformanceTestWorkload.WRITE, PerformanceTestType.THROUGHPUT)
        self.update_test_details(scylla_conf=True)
        self.display_results(results, test_name='test_write')
        self.check_regression()

    def test_read(self):
        """
        Test steps:

        1. Run a write workload as a preparation
        2. Run a read workload
        """

        base_cmd_r = self.params.get('stress_cmd_r')
        stress_multiplier = self.params.get('stress_multiplier')
        if stress_multiplier_r := self.params.get("stress_multiplier_r"):
            stress_multiplier = stress_multiplier_r
        self.run_fstrim_on_all_db_nodes()
        # run a write workload
        self.preload_data()

        # create new document in ES with doc_id = test_id + timestamp
        # allow to correctly save results for future compare
        self.create_test_stats(doc_id_with_timestamp=True)
        # wait compactions will be finished
        self.wait_no_compactions_running(n=240, sleep_time=180)
        self.run_fstrim_on_all_db_nodes()
        # run a read workload
        stress_queue = self.run_stress_thread(
            stress_cmd=base_cmd_r, stress_num=stress_multiplier, stats_aggregate_cmds=False)
        results = self.get_stress_results(queue=stress_queue)

        self.build_histogram(PerformanceTestWorkload.READ, PerformanceTestType.THROUGHPUT)
        self.update_test_details(scylla_conf=True)
        self.display_results(results, test_name='test_read')
        self.check_regression()

    def test_mixed(self):
        """
        Test steps:

        1. Run a write workload as a preparation
        2. Run a mixed workload
        """

        base_cmd_m = self.params.get('stress_cmd_m')
        stress_multiplier = self.params.get('stress_multiplier')
        if stress_multiplier_m := self.params.get("stress_multiplier_m"):
            stress_multiplier = stress_multiplier_m
        self.run_fstrim_on_all_db_nodes()
        # run a write workload as a preparation
        self.preload_data()
        # run a mixed workload
        # create new document in ES with doc_id = test_id + timestamp
        # allow to correctly save results for future compare
        self.create_test_stats(doc_id_with_timestamp=True)
        # wait compactions will be finished
        self.wait_no_compactions_running(n=240, sleep_time=180)
        self.run_fstrim_on_all_db_nodes()
        stress_queue = self.run_stress_thread(
            stress_cmd=base_cmd_m, stress_num=stress_multiplier, stats_aggregate_cmds=False)
        results = self.get_stress_results(queue=stress_queue)

        self.build_histogram(PerformanceTestWorkload.MIXED, PerformanceTestType.THROUGHPUT)
        self.update_test_details(scylla_conf=True)
        self.display_results(results, test_name='test_mixed')
        self.check_regression()

    def test_latency(self):
        """
        Test steps:

        1. Prepare cluster with data (reach steady_stet of compactions and ~x10 capacity than RAM.
        with round_robin and list of stress_cmd - the data will load several times faster.
        2. Run WRITE workload with gauss population.
        """
        self.run_fstrim_on_all_db_nodes()
        self.preload_data()
        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()
        self.run_read_workload()
        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()
        self.run_write_workload()
        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()
        self.run_mixed_workload()

    def test_latency_read_with_nemesis(self):
        self.run_fstrim_on_all_db_nodes()
        self.preload_data()
        self.wait_no_compactions_running(n=160)
        self.run_fstrim_on_all_db_nodes()
        self.run_workload(stress_cmd=self.params.get('stress_cmd_r'), nemesis=True, sub_type='read')

    def test_latency_write_with_nemesis(self):
        self.run_fstrim_on_all_db_nodes()
        self.preload_data()
        self.wait_no_compactions_running(n=160)
        self.run_fstrim_on_all_db_nodes()
        self.run_workload(stress_cmd=self.params.get('stress_cmd_w'), nemesis=True, sub_type='write')

    def test_latency_mixed_with_nemesis(self):
        self.run_fstrim_on_all_db_nodes()
        self.preload_data()
        self.wait_no_compactions_running(n=160)
        self.run_fstrim_on_all_db_nodes()
        self.run_workload(stress_cmd=self.params.get('stress_cmd_m'), nemesis=True, sub_type='mixed')

    # MV Tests
    def test_mv_write(self):
        """
        Test steps:

        1. Run WRITE workload on base table without materialized view
        2. Run WRITE workload with materialized view when view is on partition key is the same host as partition key
        3. Drop MV
        4. Run WRITE workload with materialized view when view is on clustering key is the same host as partition key
        5. Drop MV
        """
        def run_workload(stress_cmd, user_profile):
            self.log.debug('Run stress test with user profile {}'.format(user_profile))
            assert os.path.exists(user_profile), 'File not found: {}'.format(user_profile)
            self.log.debug('Stress cmd: {}'.format(stress_cmd))
            stress_queue = self.run_stress_thread(stress_cmd=stress_cmd, stress_num=1, profile=user_profile,
                                                  stats_aggregate_cmds=False)
            results = self.get_stress_results(queue=stress_queue)
            self.update_test_details(scylla_conf=True)
            self.display_results(results, test_name=test_name)
            self.check_regression()
            self.log.debug('Finish stress test with user profile {}'.format(user_profile))

        def get_mv_name(user_profile):

            # Get materialized view name from user profile

            with open(user_profile, encoding="utf-8") as fobj:
                user_profile_yaml = yaml.safe_load(fobj)
            mv_name = ''

            for k in user_profile_yaml:
                if isinstance(k, tuple) and k[0] == 'extra_definitions':
                    mv_name = k[1][0].split(' AS')[0].split(' ')[-1]
                    break

            if not mv_name:
                assert False, 'Failed to recognoze materialized view name from {0}: {1}'.format(
                    user_profile, user_profile_yaml)

            return mv_name

        def drop_mv(mv_name):
            # drop MV
            self.log.debug('Start dropping materialized view {}'.format(mv_name))
            query = 'drop materialized view {}'.format(mv_name)

            try:
                with self.db_cluster.cql_connection_patient_exclusive(self.db_cluster.nodes[0], connect_timeout=300) as session:
                    self.log.debug('Run query: {}'.format(query))
                    session.execute(SimpleStatement(query), timeout=300)
                    session.execute(query)
            except Exception as ex:
                self.log.debug('Failed to drop materialized view using query {0}. Error: {1}'.format(query, str(ex)))
                raise

            self.log.debug('Finish dropping materialized view {}'.format(mv_name))

        test_name = 'test_mv_write'
        duration = self.params.get('test_duration')
        self.log.debug('Start materialized views performance test. Test duration {} minutes'.format(duration))
        self.create_test_stats()
        cmd_no_mv = self.params.get('stress_cmd_no_mv')
        cmd_no_mv_profile = self.params.get('stress_cmd_no_mv_profile')

        # Run WRITE workload without materialized view
        run_workload(cmd_no_mv, cmd_no_mv_profile)

        # Run WRITE workload with materialized view
        mv_commands = self.params.get("stress_cmd_mv")
        # mv_commands structure (created in correctly parses yaml):
        #   [
        #    [('cmd', <cassandra-stress command line>), ('profile', <profile file name with path>)],
        #    [('cmd', <cassandra-stress command line>), ('profile', <profile file name with path>)]
        #   ]
        for cmd in mv_commands:
            cmd_mv, cmd_mv_profile = cmd[0][1], cmd[1][1]
            run_workload(cmd_mv, cmd_mv_profile)
            drop_mv(get_mv_name(cmd_mv_profile))
            time.sleep(60)

    def test_mv_write_populated(self):
        self._write_with_mv(on_populated=True)

    def test_mv_write_not_populated(self):
        self._write_with_mv(on_populated=False)

    def test_mv_read_populated(self):
        self._read_with_mv(on_populated=True)

    def test_mv_read_not_populated(self):
        self._read_with_mv(on_populated=False)

    def test_mv_mixed_populated(self):
        self._mixed_with_mv(on_populated=True)

    def test_mv_mixed_not_populated(self):
        self._mixed_with_mv(on_populated=False)

    # Counter Tests
    def test_uniform_counter_update_bench(self):  # pylint: disable=invalid-name
        """
        Test steps:

        1. Run workload: -workload uniform -mode counter_update -duration 30m
        """
        base_cmd_r = ("scylla-bench -workload uniform -mode counter_update -duration 30m "
                      "-partition-count 50000000 -clustering-row-count 1 -connection-count "
                      "32 -concurrency 512 -replication-factor 3")

        self.create_test_stats()
        stress_queue = self.run_stress_thread_bench(stress_cmd=base_cmd_r, stats_aggregate_cmds=False)
        results = self.get_stress_results_bench(queue=stress_queue)

        self.update_test_details(scylla_conf=True)
        self.display_results(results, test_name='test_read_bench')
        self.check_regression()

    # Large Partition Tests
    def test_timeseries_bench(self):
        """
        Timeseries write/read workload
        """
        cmd_w = ("scylla-bench -workload=timeseries -mode=write -replication-factor=3 "
                 "-partition-count=5000 -clustering-row-count=1000000 -clustering-row-size=200 "
                 "-concurrency=48 -max-rate=150000 -rows-per-request=5000")

        self.create_test_stats(sub_type='write')
        self._scylla_bench_prepare_table()
        self.run_stress_thread_bench(stress_cmd=cmd_w, stats_aggregate_cmds=False)
        start_timestamp = int(time.time())
        self.db_cluster.wait_total_space_used_per_node(700 * KB * KB * KB, 'scylla_bench.test')  # 700GB

        cmd_r = ("scylla-bench -workload=timeseries -mode=read -partition-count=5000 -concurrency=1 "
                 "-replication-factor=3 -write-rate=30 -clustering-row-count=1000000 -clustering-row-size=200 "
                 "-rows-per-request=1000000 -no-lower-bound -start-timestamp=%s -duration=60m" % start_timestamp)

        self.create_test_stats(sub_type='read')
        stress_queue = self.run_stress_thread_bench(stress_cmd=cmd_r, stats_aggregate_cmds=False)
        results = self.get_stress_results_bench(queue=stress_queue)
        self.update_test_details()
        self.display_results(results, test_name='test_timeseries_read_bench')
        self.check_regression()
        self.kill_stress_thread()

    def build_histogram(self, workload: PerformanceTestWorkload, test_type: PerformanceTestType):
        if not self.params["use_hdr_cs_histogram"]:
            return

        start_time = self.get_test_start_time() or self.start_time
        end_time = time.time()

        if test_type == PerformanceTestType.THROUGHPUT:
            tag_type = CSHistogramTagTypes.THROUGHPUT
        else:
            tag_type = CSHistogramTagTypes.LATENCY
        histogram_total_data = self.get_cs_range_histogram(stress_operation=workload.value,
                                                           start_time=start_time,
                                                           end_time=end_time,
                                                           tag_type=tag_type)
        self.update_hdrhistograms(histogram_name="test_histogram",
                                  histogram_data=histogram_total_data)

        histogram_data_by_interval = self.get_cs_range_histogram_by_interval(stress_operation=workload.value,
                                                                             start_time=start_time,
                                                                             end_time=end_time,
                                                                             tag_type=tag_type)

        self.update_hdrhistograms(histogram_name='test_histogram_by_interval',
                                  histogram_data=histogram_data_by_interval)


class PerformanceRegressionUpgradeTest(PerformanceRegressionTest, UpgradeTest):  # pylint: disable=too-many-ancestors
    def get_email_data(self):  # pylint: disable=no-self-use
        return PerformanceRegressionTest.get_email_data(self)

    @latency_calculator_decorator(legend="Upgrade Node")
    def upgrade_node(self, node):  # pylint: disable=arguments-differ
        InfoEvent(message='Upgrade Node %s begin' % node.name).publish()
        self._upgrade_node(node)
        InfoEvent(message='Upgrade Node %s ended' % node.name).publish()

    def _stop_stress_when_finished(self):  # pylint: disable=no-self-use
        with EventsSeverityChangerFilter(new_severity=Severity.NORMAL,  # killing stress creates Critical error
                                         event_class=CassandraStressEvent,
                                         extra_time_to_expiration=60):
            self.loaders.kill_stress_thread()

    @optional_stage('perf_steady_state_calc')
    @latency_calculator_decorator
    def steady_state_latency(self):  # pylint: disable=no-self-use
        sleep_time = self.db_cluster.params.get('nemesis_interval') * 60
        InfoEvent(message='Starting Steady State calculation for %ss' % sleep_time).publish()
        time.sleep(sleep_time)
        InfoEvent(message='Ended Steady State calculation. Took %ss' % sleep_time).publish()

    @latency_calculator_decorator
    def post_upgrades_steady_state(self):
        sleep_time = self.db_cluster.params.get('nemesis_interval') * 60
        InfoEvent(message='Starting Post-Upgrade Steady State calculation for %ss' % sleep_time).publish()
        time.sleep(sleep_time)
        InfoEvent(message='Ended Post-Upgrade Steady State calculation. Took %ss' % sleep_time).publish()

    def run_workload_and_upgrade(self, stress_cmd, sub_type=None):
        # next 3 lines, is a workaround to have it working inside `latency_calculator_decorator`
        self.cluster = self.db_cluster  # pylint: disable=attribute-defined-outside-init
        self.tester = self  # pylint: disable=attribute-defined-outside-init
        self.monitoring_set = self.monitors  # pylint: disable=attribute-defined-outside-init

        if sub_type is None:
            sub_type = 'read' if ' read ' in stress_cmd else 'write' if ' write ' in stress_cmd else 'mixed'
        test_index = f'latency-during-upgrade-{sub_type}'
        self.create_test_stats(sub_type=sub_type, append_sub_test_to_name=False, test_index=test_index)
        stress_queue = self.run_stress_thread(stress_cmd=stress_cmd, stress_num=1, stats_aggregate_cmds=False)
        time.sleep(60)  # postpone measure steady state latency to skip c-s start period when latency is high
        self.steady_state_latency()
        versions_list = []

        def _get_version_and_build_id_from_node(node):
            version = node.remoter.run('scylla --version')
            build_id = node.remoter.run('scylla --build-id')
            return version.stdout.strip(), build_id.stdout.strip()

        for node in self.db_cluster.nodes:
            base_version, base_build_id = _get_version_and_build_id_from_node(node)
            self.upgrade_node(node)
            target_version, target_build_id = _get_version_and_build_id_from_node(node)
            versions_list.append({'base_version': base_version,
                                  'base_build_id': base_build_id,
                                  'target_version': target_version,
                                  'target_build_id': target_build_id,
                                  'node_name': node.name
                                  })
            time.sleep(120)  # sleeping 2 min to give time for cache to re-heat
        self.post_upgrades_steady_state()

        # TODO: check if all `base_version` and all `target_version` are the same
        self.update({'base_target_versions': versions_list})
        self._stop_stress_when_finished()
        results = self.get_stress_results(queue=stress_queue)
        self.update_test_details(scrap_metrics_step=60)
        self.display_results(results, test_name='test_latency_with_upgrade')
        self.update_test_details(scrap_metrics_step=60)
        self.display_results(results, test_name='test_latency_during_upgrade')
        self.check_latency_during_ops()

    def _prepare_latency_with_upgrade(self):
        self.run_fstrim_on_all_db_nodes()
        self.preload_data()
        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()

    def test_latency_read_with_upgrade(self):
        self._prepare_latency_with_upgrade()
        self.run_workload_and_upgrade(stress_cmd=self.params.get('stress_cmd_r'))

    def test_latency_write_with_upgrade(self):
        self._prepare_latency_with_upgrade()
        self.run_workload_and_upgrade(stress_cmd=self.params.get('stress_cmd_w'))

    def test_latency_mixed_with_upgrade(self):
        self._prepare_latency_with_upgrade()
        self.run_workload_and_upgrade(stress_cmd=self.params.get('stress_cmd_m'))


class PerformanceRegressionMaterializedViewLatencyTest(PerformanceRegressionTest):
    """
    the idea is to reproduce the hardest scenario for MV
    based on internal doc "Consistency problems in materialized views"

    modifying a column that is a regular column in the base table,
    but in the materialized view is one of the primary key columns.
    Other types of materialized view updates are easier to handle,
    once we figure out how to do the hardest case correctly, all of the other cases will be solved as well.

    currently this problem is not solved.
    The test is just reproducer of this problem and should not be used in regular runs

    test steps:
    1 - 3 node cluster with 2 tables
    2 - do special prepare CMD for table 1, and use table 2 as for latency PERF TEST (prepare_write_cmd)
    3 - start read workload for table 2 - measure latency for table 2 (10min) (stress_cmd_r)
    4 - do a special rewrite workload for table 1 to measure latency for table 2 (while changing for table 1 applying )(stress_cmd_no_mv)
    5 - create MV, and wait for MV to sync - measure latency for table 2 (while MV is syncing )
    6- do special rewrite workload for table 1 again - measure latency for table 2 (while changing for table 1 applying ) (stress_cmd_mv)
    """

    def test_read_mv_latency(self):
        self.run_fstrim_on_all_db_nodes()
        self.preload_data()  # prepare_write_cmd
        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()

        self.create_test_stats(sub_type="read", append_sub_test_to_name=False, test_index="mv-overloading-latency-read")
        self.run_stress_thread(stress_cmd=self.params.get('stress_cmd_r'), stress_num=1,
                               stats_aggregate_cmds=False)

        self.steady_state_read_workload_latency()  # stress_cmd_r
        self.do_rewrite_workload()  # stress_cmd_no_mv + #stress_cmd_r
        self.wait_mv_sync()  # stress_cmd_r
        self.do_rewrite_workload_with_mv()  # stress_cmd_mv + #stress_cmd_r
        self.loaders.kill_stress_thread()
        self.check_latency_during_ops()

    @latency_calculator_decorator
    def steady_state_read_workload_latency(self):
        InfoEvent(message='start_read_workload_latency begin').publish()
        time.sleep(15*60)
        InfoEvent(message='start_read_workload_latency ended').publish()

    @latency_calculator_decorator
    def do_rewrite_workload(self):
        base_cmd = self.params.get('stress_cmd_no_mv')
        stress_queue = self.run_stress_thread(stress_cmd=base_cmd, stress_num=1, stats_aggregate_cmds=False)
        results = self.get_stress_results(queue=stress_queue, store_results=False)
        self.display_results(results, test_name='do_rewrite_workload')

    @latency_calculator_decorator
    def wait_mv_sync(self):
        node1 = self.db_cluster.nodes[0]
        node1.run_cqlsh(
            "CREATE TABLE IF NOT EXISTS scylla_bench.test (pk bigint,ck bigint,v blob,PRIMARY KEY(pk, ck)) WITH compression = { }")
        node1.run_cqlsh("CREATE MATERIALIZED VIEW IF NOT EXISTS scylla_bench.view_test AS SELECT * FROM scylla_bench.test where v IS NOT NULL AND ck IS NOT NULL AND pk IS NOT NULL PRIMARY KEY (v, pk, ck)")
        wait_for_view_to_be_built(node1, 'scylla_bench', 'view_test', timeout=1000)

    @latency_calculator_decorator
    def do_rewrite_workload_with_mv(self):
        base_cmd = self.params.get('stress_cmd_mv')
        stress_queue = self.run_stress_thread(stress_cmd=base_cmd, stress_num=1, stats_aggregate_cmds=False)
        results = self.get_stress_results(queue=stress_queue, store_results=False)
        self.display_results(results, test_name='do_rewrite_workload_with_mv')
