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

import time
import contextlib

from sdcm.utils import alternator
from sdcm.utils.decorators import latency_calculator_decorator
from sdcm.sct_events import Severity
from sdcm.sct_events.system import InfoEvent
from sdcm.sct_events.filters import EventsSeverityChangerFilter
from sdcm.sct_events.loaders import YcsbStressEvent
from sdcm.sct_events.group_common_events import ignore_operation_errors, ignore_alternator_client_errors

from performance_regression_test import PerformanceRegressionTest
from upgrade_test import UpgradeTest


class PerformanceRegressionAlternatorTest(PerformanceRegressionTest):
    def __init__(self, *args):
        super().__init__(*args)

        # suppress YCSB client error and timeout to warnings for all the test in this class
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(ignore_alternator_client_errors())
        self.stack.enter_context(ignore_operation_errors())

    def _workload(self, stress_cmd, stress_num=1, test_name=None, sub_type=None, keyspace_num=1, prefix='', debug_message='',  # pylint: disable=too-many-arguments,arguments-differ
                  save_stats=True, is_alternator=True, nemesis=False):
        if not is_alternator:
            stress_cmd = stress_cmd.replace('dynamodb', 'cassandra-cql')

        if debug_message:
            self.log.debug(debug_message)

        if save_stats:
            self.create_test_stats(test_name=test_name, sub_type=sub_type,
                                   doc_id_with_timestamp=True, append_sub_test_to_name=False)
        if nemesis:
            interval = self.params.get('nemesis_interval')
            time.sleep(interval * 60)  # Sleeping one interval (in minutes) before starting the nemesis
            self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(), tester_obj=self)
            self.db_cluster.start_nemesis(interval=interval, cycles_count=1)
            self._stop_load_when_nemesis_threads_end()

        stress_queue = self.run_stress_thread(stress_cmd=stress_cmd, stress_num=stress_num, keyspace_num=keyspace_num,
                                              prefix=prefix, stats_aggregate_cmds=False)
        self.get_stress_results(queue=stress_queue, store_results=True)
        if save_stats:
            self.update_test_details(scylla_conf=True, alternator=is_alternator)

    def create_alternator_table(self, schema, alternator_write_isolation):
        node = self.db_cluster.nodes[0]

        if self.params.get("alternator_enforce_authorization"):
            self.log.info("Create the role for Alternator authorization")
            with self.db_cluster.cql_connection_patient(node) as session:
                session.execute(
                    "CREATE ROLE %s WITH PASSWORD = %s AND login = true AND superuser = true",
                    (
                        self.params.get("alternator_access_key_id"),
                        self.params.get("alternator_secret_access_key"),
                    )
                )

        # drop tables
        table_name = alternator.consts.TABLE_NAME
        if self.alternator.is_table_exists(node=node, table_name=table_name):
            self.alternator.delete_table(node=node, table_name=table_name)
        # create new tables
        self.log.info("Going to create alternator tables")
        self.alternator.create_table(node=node, schema=schema, isolation=alternator_write_isolation)

        self.run_fstrim_on_all_db_nodes()
        self.wait_no_compactions_running()

    def create_cql_ks_and_table(self, field_number):
        node = self.db_cluster.nodes[0]
        with self.db_cluster.cql_connection_patient(node) as session:
            session.execute(
                """create keyspace ycsb WITH REPLICATION = {'class' : 'NetworkTopologyStrategy', 'replication_factor': 3 };""")
            fields = ', '.join([f"field{i} varchar" for i in range(field_number)])
            session.execute(f"""CREATE TABLE ycsb.usertable (
                                y_id varchar primary key,
                                {fields});""")

    def preload_data(self, compaction_strategy=None):
        # if test require a pre-population of data
        prepare_write_cmd = self.params.get('prepare_write_cmd')
        if prepare_write_cmd:
            # create new document in ES with doc_id = test_id + timestamp
            # allow to correctly save results for future compare
            self.create_test_stats(sub_type='write-prepare', doc_id_with_timestamp=True)
            stress_queue = []
            params = {'prefix': 'preload-'}
            for stress_type in ['dynamodb', 'cassandra-cql']:
                # Check if the prepare_cmd is a list of commands
                if not isinstance(prepare_write_cmd, str) and len(prepare_write_cmd) > 1:
                    # Check if it should be round_robin across loaders
                    if self.params.get('round_robin'):
                        self.log.debug('Populating data using round_robin')
                        params.update({'stress_num': 1, 'round_robin': True})

                    for stress_cmd in prepare_write_cmd:
                        params.update({'stress_cmd': stress_cmd.replace('dynamodb', stress_type)})

                        # Run all stress commands
                        params.update(dict(stats_aggregate_cmds=False))
                        self.log.debug('RUNNING stress cmd: {}'.format(stress_cmd.replace('dynamodb', stress_type)))
                        stress_queue.append(self.run_stress_thread(**params))

                # One stress cmd command
                else:
                    stress_queue.append(self.run_stress_thread(stress_cmd=prepare_write_cmd.replace('dynamodb', stress_type), stress_num=1,
                                                               prefix='preload-', stats_aggregate_cmds=False))

            for stress in stress_queue:
                self.get_stress_results(queue=stress, store_results=False)

            self.update_test_details()
        else:
            self.log.warning("No prepare command defined in YAML!")

    def test_write(self):
        """
        Test steps:

        1. Run a write workload with cql
        2. Run a write workload without lwt
        3. Run a write workload with lwt enabled
        """
        # run a write workload
        base_cmd_w = self.params.get('stress_cmd_w')
        stress_multiplier = self.params.get('stress_multiplier')

        self.create_cql_ks_and_table(field_number=10)

        self._workload(sub_type='cql', stress_cmd=base_cmd_w,
                       stress_num=stress_multiplier, keyspace_num=1, is_alternator=False)

        schema = self.params.get("dynamodb_primarykey_type")
        # run a workload without lwt as baseline
        self.create_alternator_table(
            schema=schema, alternator_write_isolation=alternator.enums.WriteIsolation.FORBID_RMW)

        self._workload(sub_type='without-lwt', stress_cmd=base_cmd_w, stress_num=stress_multiplier, keyspace_num=1)

        # run a workload with lwt
        self.create_alternator_table(
            schema=schema, alternator_write_isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT)
        self._workload(sub_type='with-lwt', stress_cmd=base_cmd_w, stress_num=stress_multiplier, keyspace_num=1)

        self.check_regression_with_baseline('cql')

    def test_read(self):
        """
        Test steps:

        1. Run a write workload as a preparation
        2. Run a read workload with cql
        3. Run a read workload without lwt
        4. Run a read workload with lwt enabled
        """
        node = self.db_cluster.nodes[0]

        base_cmd_r = self.params.get('stress_cmd_r')
        stress_multiplier = self.params.get('stress_multiplier')
        self.run_fstrim_on_all_db_nodes()
        # run a prepare write workload
        self.create_cql_ks_and_table(field_number=10)

        self.create_alternator_table(schema=self.params.get("dynamodb_primarykey_type"),
                                     alternator_write_isolation=alternator.enums.WriteIsolation.FORBID_RMW)

        self.preload_data()

        self._workload(sub_type='cql', stress_cmd=base_cmd_r,
                       stress_num=stress_multiplier, keyspace_num=1, is_alternator=False)

        # run a workload without lwt as baseline
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.FORBID_RMW)
        self._workload(sub_type='without-lwt', stress_cmd=base_cmd_r, stress_num=stress_multiplier, keyspace_num=1)

        # run a workload with lwt
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT)
        self._workload(sub_type='with-lwt', stress_cmd=base_cmd_r, stress_num=stress_multiplier, keyspace_num=1)

        self.check_regression_with_baseline('cql')

    def test_mixed(self):
        """
        Test steps:

        1. Run a write workload as a preparation
        2. Run a mixed workload with cql
        3. Run a mixed workload without lwt
        4. Run a mixed workload with lwt
        """
        node = self.db_cluster.nodes[0]

        base_cmd_m = self.params.get('stress_cmd_m')
        stress_multiplier = self.params.get('stress_multiplier')
        self.run_fstrim_on_all_db_nodes()

        self.create_cql_ks_and_table(field_number=10)
        self.create_alternator_table(schema=self.params.get("dynamodb_primarykey_type"),
                                     alternator_write_isolation=alternator.enums.WriteIsolation.FORBID_RMW)

        # run a write workload as a preparation
        self.preload_data()

        self._workload(sub_type='cql', stress_cmd=base_cmd_m,
                       stress_num=stress_multiplier, keyspace_num=1, is_alternator=False)

        # run a mixed workload
        # run a workload without lwt as baseline
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.FORBID_RMW)
        self._workload(sub_type='without-lwt', stress_cmd=base_cmd_m, stress_num=stress_multiplier, keyspace_num=1)

        # run a workload with lwt
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT)
        self._workload(sub_type='with-lwt', stress_cmd=base_cmd_m, stress_num=stress_multiplier, keyspace_num=1)

        self.check_regression_with_baseline('cql')

    def test_latency(self):
        """
        Test steps:

         1. Prepare cluster with data (reach steady_state of compactions and ~x10 capacity than RAM.)
            with round_robin and list of stress_cmd - the data will load several times faster.
         2. Run READ workload with cql.
         3. Run READ workload without lwt.
         4. Run READ workload with lwt.
         5. Run WRITE workload with cql.
         6. Run WRITE workload without lwt.
         7. Run WRITE workload with lwt.
         8. Run MIXED workload with cql.
         9. Run MIXED workload without lwt.
        10. Run MIXED workload with lwt.
        """
        node = self.db_cluster.nodes[0]

        self.create_alternator_table(schema=self.params.get("dynamodb_primarykey_type"),
                                     alternator_write_isolation=alternator.enums.WriteIsolation.FORBID_RMW)
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.FORBID_RMW)

        self.create_cql_ks_and_table(field_number=10)
        self.run_fstrim_on_all_db_nodes()
        self.preload_data()

        base_cmd_w = self.params.get('stress_cmd_w')
        base_cmd_r = self.params.get('stress_cmd_r')
        base_cmd_m = self.params.get('stress_cmd_m')

        stress_multiplier = 2
        self.wait_no_compactions_running(n=120)

        self.run_fstrim_on_all_db_nodes()
        self._workload(
            test_name=self.id() + '_read', sub_type='cql', stress_cmd=base_cmd_r, stress_num=stress_multiplier,
            keyspace_num=1, is_alternator=False)

        # run a workload without lwt as baseline
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.FORBID_RMW)
        self._workload(
            test_name=self.id() + '_read', sub_type='without-lwt', stress_cmd=base_cmd_r, stress_num=stress_multiplier,
            keyspace_num=1)

        self.wait_no_compactions_running()
        # run a workload with lwt
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT)
        self._workload(
            test_name=self.id() + '_read', sub_type='with-lwt', stress_cmd=base_cmd_r, stress_num=stress_multiplier,
            keyspace_num=1)
        self.check_regression_with_baseline('cql')

        stress_multiplier = 1
        self.run_fstrim_on_all_db_nodes()

        self.wait_no_compactions_running()
        self._workload(
            test_name=self.id() + '_write', sub_type='cql', stress_cmd=base_cmd_w + " -target 10000",
            stress_num=stress_multiplier, keyspace_num=1, is_alternator=False)

        self.wait_no_compactions_running()
        # run a workload without lwt as baseline
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.FORBID_RMW)
        self._workload(
            test_name=self.id() + '_write', sub_type='without-lwt', stress_cmd=base_cmd_w + " -target 10000",
            stress_num=stress_multiplier, keyspace_num=1)

        self.wait_no_compactions_running(n=120)
        # run a workload with lwt
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT)
        self._workload(
            test_name=self.id() + '_write', sub_type='with-lwt', stress_cmd=base_cmd_w + " -target 3000",
            stress_num=stress_multiplier, keyspace_num=1)
        self.check_regression_with_baseline('cql')

        stress_multiplier = 1
        self.wait_no_compactions_running(n=120)
        self.run_fstrim_on_all_db_nodes()

        self._workload(
            test_name=self.id() + '_mixed', sub_type='cql', stress_cmd=base_cmd_m + " -target 10000",
            stress_num=stress_multiplier, keyspace_num=1, is_alternator=False)

        self.wait_no_compactions_running()
        # run a workload without lwt as baseline
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.FORBID_RMW)
        self._workload(test_name=self.id() + '_mixed', sub_type='without-lwt',
                       stress_cmd=base_cmd_m + " -target 10000", stress_num=stress_multiplier, keyspace_num=1)

        self.wait_no_compactions_running()
        # run a workload with lwt
        self.alternator.set_write_isolation(node=node, isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT)
        self._workload(test_name=self.id() + '_mixed', sub_type='with-lwt',
                       stress_cmd=base_cmd_m + " -target 5000", stress_num=stress_multiplier, keyspace_num=1)

        self.check_regression_with_baseline('cql')

    def test_latency_write_with_nemesis(self):
        self.create_alternator_table(
            schema=self.params.get("dynamodb_primarykey_type"),
            alternator_write_isolation=alternator.enums.WriteIsolation.FORBID_RMW,
        )
        self.create_cql_ks_and_table(field_number=10)

        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()
        self.alternator.set_write_isolation(
            node=self.db_cluster.nodes[0],
            isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT,
        )
        self._workload(
            test_name=self.id() + "_write",
            sub_type="with-lwt",
            stress_cmd=self.params.get("stress_cmd_w") + " -target 3000",
            nemesis=True,
        )
        self.check_latency_during_ops()

    def test_latency_read_with_nemesis(self):
        self.create_alternator_table(
            schema=self.params.get("dynamodb_primarykey_type"),
            alternator_write_isolation=alternator.enums.WriteIsolation.FORBID_RMW,
        )
        self.create_cql_ks_and_table(field_number=10)

        # Run a write workload as a preparation.
        self.preload_data()

        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()
        self.alternator.set_write_isolation(
            node=self.db_cluster.nodes[0],
            isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT,
        )
        self._workload(
            test_name=self.id() + "_read",
            sub_type="with-lwt",
            stress_cmd=self.params.get("stress_cmd_r") + " -target 3000",
            nemesis=True,
        )
        self.check_latency_during_ops()

    def test_latency_mixed_with_nemesis(self):
        self.create_alternator_table(
            schema=self.params.get("dynamodb_primarykey_type"),
            alternator_write_isolation=alternator.enums.WriteIsolation.FORBID_RMW,
        )
        self.create_cql_ks_and_table(field_number=10)

        # Run a write workload as a preparation.
        self.preload_data()

        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()
        self.alternator.set_write_isolation(
            node=self.db_cluster.nodes[0],
            isolation=alternator.enums.WriteIsolation.ALWAYS_USE_LWT,
        )
        self._workload(
            test_name=self.id() + "_mixed",
            sub_type="with-lwt",
            stress_cmd=self.params.get("stress_cmd_m") + " -target 3000",
            nemesis=True,
        )
        self.check_latency_during_ops()


class PerformanceRegressionAlternatorUpgradeTest(PerformanceRegressionAlternatorTest, UpgradeTest):  # pylint: disable=too-many-ancestors
    def get_email_data(self) -> dict:
        return PerformanceRegressionAlternatorTest.get_email_data(self)

    @latency_calculator_decorator(legend="Upgrade Node")
    def upgrade_node(self, node) -> None:  # pylint: disable=arguments-differ
        InfoEvent(message=f"Upgrade Node {node.name} begin").publish()
        self._upgrade_node(node)
        InfoEvent(message=f"Upgrade Node {node.name} ended").publish()

    def _stop_stress_when_finished(self) -> None:  # pylint: disable=no-self-use
        with EventsSeverityChangerFilter(new_severity=Severity.NORMAL,  # killing stress creates Critical error
                                         event_class=YcsbStressEvent,
                                         extra_time_to_expiration=60):
            self.loaders.kill_stress_thread()

    @latency_calculator_decorator
    def steady_state_latency(self) -> None:  # pylint: disable=no-self-use
        sleep_time = self.db_cluster.params.get("nemesis_interval") * 60
        InfoEvent(message=f"Starting Steady State calculation for {sleep_time}s").publish()
        time.sleep(sleep_time)
        InfoEvent(message=f"Ended Steady State calculation. Took {sleep_time}s").publish()

    @latency_calculator_decorator
    def post_upgrades_steady_state(self) -> None:
        sleep_time = self.db_cluster.params.get("nemesis_interval") * 60
        InfoEvent(message=f"Starting Post-Upgrade Steady State calculation for {sleep_time}s").publish()
        time.sleep(sleep_time)
        InfoEvent(message=f"Ended Post-Upgrade Steady State calculation. Took {sleep_time}s").publish()

    def run_workload_and_upgrade(self, stress_cmd: str, sub_type: str) -> None:
        # next 3 lines, is a workaround to have it working inside `latency_calculator_decorator`
        self.cluster = self.db_cluster  # pylint: disable=attribute-defined-outside-init
        self.tester = self  # pylint: disable=attribute-defined-outside-init
        self.monitoring_set = self.monitors  # pylint: disable=attribute-defined-outside-init

        test_index = f"alternator-latency-during-upgrade-{sub_type}"
        self.create_test_stats(
            sub_type=sub_type,
            append_sub_test_to_name=False,
            test_index=test_index,
        )
        stress_queue = self.run_stress_thread(
            stress_cmd=stress_cmd,
            stats_aggregate_cmds=False,
        )
        time.sleep(60)  # postpone measure steady state latency to skip start period when latency is high
        self.steady_state_latency()
        versions_list = []

        def _get_version_and_build_id_from_node(scylla_node) -> tuple[str, str]:
            version = scylla_node.remoter.run("scylla --version")
            build_id = scylla_node.remoter.run("scylla --build-id")
            return version.stdout.strip(), build_id.stdout.strip()

        for node in self.db_cluster.nodes:
            base_version, base_build_id = _get_version_and_build_id_from_node(node)
            self.upgrade_node(node)
            target_version, target_build_id = _get_version_and_build_id_from_node(node)
            versions_list.append({
                "base_version": base_version,
                "base_build_id": base_build_id,
                "target_version": target_version,
                "target_build_id": target_build_id,
                "node_name": node.name,
            })
            time.sleep(120)  # sleeping 2 min to give time for cache to re-heat
        self.post_upgrades_steady_state()

        # TODO: check if all `base_version` and all `target_version` are the same
        self.update({"base_target_versions": versions_list})
        self._stop_stress_when_finished()
        results = self.get_stress_results(queue=stress_queue)
        self.update_test_details(scrap_metrics_step=60, alternator=True)
        self.display_results(results, test_name="test_alternator_latency_with_upgrade")
        self.update_test_details(scrap_metrics_step=60, alternator=True)
        self.display_results(results, test_name="test_alternator_latency_during_upgrade")
        self.check_latency_during_ops()

    def _prepare_latency_with_upgrade(self) -> None:
        self.run_fstrim_on_all_db_nodes()
        self.create_alternator_table(
            schema=self.params.get("dynamodb_primarykey_type"),
            alternator_write_isolation=alternator.enums.WriteIsolation.FORBID_RMW,
        )
        self.create_cql_ks_and_table(field_number=10)
        self.preload_data()
        self.wait_no_compactions_running()
        self.run_fstrim_on_all_db_nodes()

    def test_latency_write_with_upgrade(self) -> None:
        self._prepare_latency_with_upgrade()
        self.run_workload_and_upgrade(
            stress_cmd=self.params.get("stress_cmd_w") + " -target 3000",
            sub_type="write",
        )

    def test_latency_read_with_upgrade(self) -> None:
        self._prepare_latency_with_upgrade()
        self.run_workload_and_upgrade(
            stress_cmd=self.params.get("stress_cmd_r") + " -target 3000",
            sub_type="read",
        )

    def test_latency_mixed_with_upgrade(self) -> None:
        self._prepare_latency_with_upgrade()
        self.run_workload_and_upgrade(
            stress_cmd=self.params.get("stress_cmd_m") + " -target 3000",
            sub_type="mixed",
        )
