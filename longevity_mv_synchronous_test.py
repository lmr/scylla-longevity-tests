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

# This is stress longevity test that runs light weight transactions in parallel with different node operations:
# disruptive and not disruptive
#
# After the test is finished will be performed the data validation.

from unittest.mock import MagicMock

from longevity_test import LongevityTest
from sdcm.sct_events import Severity
from sdcm.sct_events.database import DatabaseLogEvent
from sdcm.sct_events.filters import DbEventsFilter
from sdcm.sct_events.health import DataValidatorEvent
from sdcm.sct_events.system import InfoEvent
from sdcm.utils.common import keyspace_min_max_tokens, get_view_name_from_user_profile, get_keyspace_from_user_profile
from sdcm.utils.data_validator import LongevityDataValidator
from sdcm.sct_events.group_common_events import ignore_mutation_write_errors


class MvSynchronousLongevityTest(LongevityTest):
    BASE_TABLE_PARTITION_KEYS = ['domain', 'mv_pk']

    def __init__(self, *args):
        super().__init__(*args)
        self.data_validator = None
        self.synchronous_updates_keyspace = get_keyspace_from_user_profile(params=self.params,
                                                                           stress_cmds_part="prepare_write_cmd",
                                                                           search_for_user_profile="_synchronous_updates.")
        self.synch_view_name = get_view_name_from_user_profile(params=self.params,
                                                               stress_cmds_part="prepare_write_cmd",
                                                               search_for_user_profile="_synchronous_updates.",
                                                               view_name_substr="_synch_test")
        self.asynch_view_name = get_view_name_from_user_profile(params=self.params,
                                                                stress_cmds_part="prepare_write_cmd",
                                                                search_for_user_profile="_synchronous_updates.",
                                                                view_name_substr="_asynch_test")

    def wait_for_views_build_for_synch_mode_test(self):
        if not (self.synch_view_name and self.asynch_view_name):
            InfoEvent(message="One or both 'synch_view_name' and 'asynch_view_name' is not found. "
                              "Validation of 'synchronous_updates' mode is not performed")
            return

        with self.db_cluster.cql_connection_patient(self.db_cluster.nodes[0]) as session:
            for view_name in [self.synch_view_name, self.asynch_view_name]:
                self.log.debug("Start waiting for %s view", view_name)
                self._wait_for_view(scylla_cluster=self.db_cluster,
                                    session=session,
                                    key_space=self.synchronous_updates_keyspace,
                                    view=view_name,
                                    wait_attempts=3600)
                self.log.debug("Finish waiting for %s view", view_name)

    def wait_for_views_build_before_test_start(self):
        """
        Part of the views has to be built before test start. So we need to wait for build is completed.
        We do need to do it in the parallel as these views build is running in the parallel.
        So actually we need to wait for the biggest view. It is views for testing of 'synchronous_updates' mode
        """

        # Wait for views for testing of 'synchronous_updates' mode
        self.wait_for_views_build_for_synch_mode_test()

        # Wait for view build for update and delete tests
        views = [view[0] for view in self.data_validator.list_of_view_names_for_update_test()]
        self.log.info("views: %s", views)
        self.log.info("views plus: %s", views + [self.data_validator.view_name_for_deletion_data])
        with self.db_cluster.cql_connection_patient(self.db_cluster.nodes[0]) as session:
            for view_name in views + [self.data_validator.view_name_for_deletion_data]:
                self.log.debug("Start waiting for %s view", view_name)
                self._wait_for_view(scylla_cluster=self.db_cluster,
                                    session=session,
                                    key_space=self.data_validator.keyspace_name,
                                    view=view_name,
                                    wait_attempts=3600)
                self.log.debug("Finish waiting for %s view", view_name)

    def run_prepare_write_cmd(self):
        with ignore_mutation_write_errors():
            super().run_prepare_write_cmd()

        # Stop nemesis. Prefer all nodes will be run before collect data for validation
        # Increase timeout to wait for nemesis finish
        if self.db_cluster.nemesis_threads:
            self.db_cluster.stop_nemesis(timeout=300)

        self.create_main_mv()

        if self.db_cluster.nemesis_count > 1:
            self.data_validator = MagicMock()
            DataValidatorEvent.DataValidator(severity=Severity.WARNING,
                                             message="Test runs with parallel nemesis. Data validator is disabled."
                                             ).publish()
        else:
            self.data_validator = LongevityDataValidator(longevity_self_object=self,
                                                         user_profile_name='lwt',
                                                         base_table_partition_keys=self.BASE_TABLE_PARTITION_KEYS,
                                                         stress_cmds_part="stress_read_cmd")

        self.wait_for_views_build_before_test_start()

        self.data_validator.copy_immutable_expected_data()
        self.data_validator.copy_updated_expected_data()
        self.data_validator.save_count_rows_for_deletion()

        # Run nemesis during stress as it was stopped before copy expected data
        if self.params.get('nemesis_during_prepare'):
            self.db_cluster.start_nemesis()

    def test_mv_synch_longevity(self):
        with DbEventsFilter(db_event=DatabaseLogEvent.WARNING, line="perf_event_open"):
            self.test_custom_time()

            # Stop nemesis. Prefer all nodes will be run before collect data for validation
            # Increase timeout to wait for nemesis finish
            if self.db_cluster.nemesis_threads:
                self.db_cluster.stop_nemesis(timeout=300)
            self.validate_data()

    def validate_data(self):
        node = self.db_cluster.nodes[0]
        with self.db_cluster.cql_connection_patient(node, keyspace=self.data_validator.keyspace_name) as session:
            self.data_validator.validate_range_not_expected_to_change(session=session)
            self.data_validator.validate_range_expected_to_change(session=session)
            self.data_validator.validate_deleted_rows(session=session)

    def create_main_mv(self):
        """
        Create materialized view for keyspace1.standard1 base table
        """
        with self.db_cluster.cql_connection_patient(self.db_cluster.nodes[0]) as session:
            min_token, max_token = keyspace_min_max_tokens(node=self.db_cluster.nodes[0], keyspace="keyspace1")
            self.log.debug("Partition ranges. Min: %s, max: %s", min_token, max_token)
            if min_token is not None and max_token is not None:
                token_range = max_token - min_token
                # Create MV for part of the base table data
                view_token_min = int(min_token + (token_range / 4))
                view_token_max = int(max_token - (token_range / 4))
                where = f"WHERE token(key) > {view_token_min} and token(key) < {view_token_max}"
            else:
                where = "WHERE key is not NULL"

            main_mv_query = (f"CREATE MATERIALIZED VIEW keyspace1.main_mv_standard1 AS "
                             f"SELECT * FROM keyspace1.standard1 "
                             f"{where} PRIMARY KEY (key) WITH synchronous_updates = true")
            self.log.debug("Create mv_standard1 materialized view with query: %s", main_mv_query)
            session.execute(main_mv_query)

            # We do not need to wait for view build here. Let it run as it
