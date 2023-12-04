#!/usr/bin/env python
import logging
import random
import time
import traceback
from functools import cached_property
from threading import Thread, Event

from cassandra import ConsistencyLevel

from sdcm.sct_events import Severity
from sdcm.sct_events.database import DataOperationsEvent, DataOperationsCaseEvent
from sdcm.sct_events.decorators import raise_event_on_failure
from sdcm.sct_events.system import InfoEvent
from sdcm.utils.database_query_utils import fetch_all_rows
from sdcm.utils.user_profile import get_profile_content, get_table_by_stress_cmd, get_keyspace_by_stress_cmd


class ResurrectionDataFound(Exception):
    pass


class NoDataFound(Exception):
    pass


class SchemaHelpers:
    """
    This class keeps test schema helpers, getter for names of keyspace, table, view
    """
    DEFAULT_STRESS_WRITE_COMMAND = ("cassandra-stress user profile=/tmp/data_operation_default.yaml n=100000 ops'(insert=10)' cl=QUORUM "
                                    "-mode native cql3 -rate threads=30")

    def __init__(self, tester, stress_cmd: str = DEFAULT_STRESS_WRITE_COMMAND):
        self.main_stress_cmd = stress_cmd
        _, self.main_profile_content = get_profile_content(stress_cmd=stress_cmd)
        self.tester = tester
        self.cluster = tester.db_cluster

    @staticmethod
    def session_execute(session, query: str, consistency_level: int = ConsistencyLevel.QUORUM, timeout: int = 120):
        session.default_consistency_level = consistency_level
        return session.execute(query, timeout=timeout)

    @cached_property
    def keyspace_name(self):
        return get_keyspace_by_stress_cmd(stress_cmd=self.main_stress_cmd, profile_content=self.main_profile_content)

    @cached_property
    def table_name(self):
        return get_table_by_stress_cmd(stress_cmd=self.main_stress_cmd, profile_content=self.main_profile_content)

    @cached_property
    def view_names(self):
        with self.cluster.cql_connection_patient(self.cluster.nodes[0]) as session:
            query = (f"select view_name from system_schema.views where keyspace_name='{self.keyspace_name}' and "
                     f"base_table_name = '{self.table_name}' ALLOW FILTERING")
            result = list(self.session_execute(session=session, query=query))

            return [row.view_name for row in result] if result else None

    @cached_property
    def partition_keys(self):
        with self.cluster.cql_connection_patient(self.cluster.nodes[0]) as session:
            session.default_consistency_level = ConsistencyLevel.QUORUM
            query = (f"select column_name from system_schema.columns where keyspace_name='{self.keyspace_name}' "
                     f"and table_name='{self.table_name}' and kind = 'partition_key' ALLOW FILTERING")
            result = list(self.session_execute(session=session, query=query))

            return [row.column_name for row in result] if result else None

    @cached_property
    def clustering_key(self):
        with self.cluster.cql_connection_patient(self.cluster.nodes[0]) as session:
            session.default_consistency_level = ConsistencyLevel.QUORUM
            query = (f"select column_name from system_schema.columns where keyspace_name='{self.keyspace_name}' "
                     f"and table_name='{self.table_name}' and kind = 'clustering' ALLOW FILTERING")
            result = list(self.session_execute(session=session, query=query))

            return [row.column_name for row in result][0] if result else None

    def create_default_keyspace_query(self, replication_factor):
        self.tester.create_keyspace(keyspace_name=self.keyspace_name,
                                    replication_factor=replication_factor)


class DataOperations:
    """
    This class presents functions that perform data changes, for example: delete data from partition
    """

    def __init__(self, schema_helpers: SchemaHelpers, interval: int):
        self.schema_helpers = schema_helpers
        self.log = logging.getLogger(self.__class__.__name__)
        self.interval = interval

    def get_unique_partition_values(self, partitions_amount: int):
        if not self.schema_helpers.partition_keys:
            raise ValueError(f"Failed to get partition keys. Maybe the table "
                             f"{self.schema_helpers.keyspace_name}.{self.schema_helpers.table_name} is empty")

        pks_str = ", ".join(self.schema_helpers.partition_keys)
        with self.schema_helpers.cluster.cql_connection_patient(self.schema_helpers.cluster.nodes[0],
                                                                keyspace=self.schema_helpers.keyspace_name) as session:
            session.default_consistency_level = ConsistencyLevel.QUORUM
            query = f"select distinct {pks_str} from {self.schema_helpers.table_name}"
            unique_partitions = list(self.schema_helpers.session_execute(session=session, query=query))
            self.log.debug("Partitions in the %s table: %s", self.schema_helpers.table_name, len(unique_partitions))

        if not unique_partitions:
            raise NoDataFound(
                f"Not found data in the {self.schema_helpers.keyspace_name}.{self.schema_helpers.table_name}")

        return [unique_partitions[random.randint(0, len(unique_partitions))] for _ in range(partitions_amount)]

    def get_max_clustering_value_in_partition(self, where_condition: list) -> dict:
        max_clustering_value = {}
        if not (clustering_key := self.schema_helpers.clustering_key):
            InfoEvent("There is no clustering key in the table. The maximum clustering value in partition can not be found",
                      severity=Severity.NORMAL).publish()
            return max_clustering_value

        with self.schema_helpers.cluster.cql_connection_patient(self.schema_helpers.cluster.nodes[0],
                                                                keyspace=self.schema_helpers.keyspace_name) as session:
            session.default_consistency_level = ConsistencyLevel.QUORUM
            cmd = f"select {clustering_key} as clustering_key from {self.schema_helpers.table_name} " \
                  f"where %s order by {clustering_key} desc limit 1"
            for where_clause in where_condition:
                result = list(session.execute(cmd % where_clause, timeout=300))
                if result:
                    max_clustering_value[where_clause] = {"ck_name": clustering_key, "value": result[0].clustering_key}

            if not result:
                raise NoDataFound(
                    f"Not found data for partition in the {self.schema_helpers.keyspace_name}.{self.schema_helpers.table_name}")

        return max_clustering_value

    @staticmethod
    def build_where_condition(columns: list, values: list):
        where_condition = []
        for row_data in values:
            # The WHERE condition is built for numeric column type
            where_condition.append(" and ".join(f"{one[0]} = {one[1]}" for one in zip(columns, row_data)))

        return where_condition

    def delete_partition(self, where_condition_to_delete):
        self.log.debug("Delete partitions: %s", where_condition_to_delete)
        with self.schema_helpers.cluster.cql_connection_patient(self.schema_helpers.cluster.nodes[0],
                                                                keyspace=self.schema_helpers.keyspace_name) as session:
            for row in where_condition_to_delete:
                query = f"delete from {self.schema_helpers.table_name} where {row}"
                self.log.debug("Delete query: %s", query)
                self.schema_helpers.session_execute(session=session, query=query, timeout=300)
        self.log.debug("Partitions '%s' have been deleted", where_condition_to_delete)

    def query_one_partition(self, where_condition_to_delete, session):
        # TODO: add ALLOW FILTERING when need only?
        query = f"select count(*) from %s where {where_condition_to_delete} ALLOW FILTERING"
        # Search for resurrection rows in the table
        rows = fetch_all_rows(session=session, default_fetch_size=1000, statement=query %
                              self.schema_helpers.table_name, timeout=240)
        self.log.debug("Table %s has rows when filtered by '%s': %s",
                       self.schema_helpers.table_name, where_condition_to_delete, rows)
        if rows and rows[0].count > 0:
            raise ResurrectionDataFound(f"Found resurrection data in the table {self.schema_helpers.table_name}. "
                                        f"Partition: %s", where_condition_to_delete)

        for view in self.schema_helpers.view_names:
            # Search for resurrection rows in the view
            rows = fetch_all_rows(session=session, default_fetch_size=1000, statement=query % view, timeout=240)
            self.log.debug("View %s has rows when filtered by '%s': %s", view, where_condition_to_delete, rows)
            if rows and rows[0].count > 0:
                raise ResurrectionDataFound(
                    f"Found resurrection data in the view {view}. Partition: %s", where_condition_to_delete)

    def validate_deleted_rows_not_resurrected(self, where_condition_to_delete):
        with self.schema_helpers.cluster.cql_connection_patient(self.schema_helpers.cluster.nodes[0],
                                                                keyspace=self.schema_helpers.keyspace_name) as session:
            for row in where_condition_to_delete:
                self.query_one_partition(where_condition_to_delete=row, session=session)

    def count_rows(self, entity: str, limit: int = 1000):
        try:
            with self.schema_helpers.tester.db_cluster.cql_connection_patient(self.schema_helpers.tester.db_cluster.nodes[0],
                                                                              keyspace=self.schema_helpers.keyspace_name) as session:
                # Check if there are enough rows in the table or need to write more data
                query = f"select count(*) as cnt from {entity} LIMIT {limit}"
                self.log.debug("Run select query: %s", query)
                result = self.schema_helpers.session_execute(session=session, query=query, timeout=300)
                self.log.debug(
                    "Rows in the %s.%s: %s", self.schema_helpers.keyspace_name, self.schema_helpers.table_name, result.one().cnt)
                return result.one().cnt
        except Exception as err:  # pylint: disable=broad-except
            self.log.error("Failed query: %s. Error: %s", query, err)
            return None

    def delete_full_partition(self, partitions_amount_to_delete: int = 3):
        with DataOperationsCaseEvent(case="DeleteFullPartition") as case_event:
            try:
                partition_values = self.get_unique_partition_values(partitions_amount=partitions_amount_to_delete)
                where_condition = self.build_where_condition(columns=self.schema_helpers.partition_keys,
                                                             values=partition_values)
                self.log.debug("Where condition for delete: %s", where_condition)
                self.delete_partition(where_condition_to_delete=where_condition)
                time.sleep(self.interval)
                self.validate_deleted_rows_not_resurrected(where_condition_to_delete=where_condition)
            except Exception as error:  # pylint: disable=broad-except
                self.log.error(traceback.format_exc())
                case_event.severity = Severity.CRITICAL if isinstance(error, ResurrectionDataFound) else Severity.ERROR
                case_event.message = str(error)

    def delete_half_partition(self, partitions_amount_to_delete: int = 1):
        with DataOperationsCaseEvent(case="DeleteHalfPartition") as case_event:
            try:
                partition_values = self.get_unique_partition_values(partitions_amount=partitions_amount_to_delete)
                where_condition = self.build_where_condition(columns=self.schema_helpers.partition_keys,
                                                             values=partition_values)
                max_clustering_values = self.get_max_clustering_value_in_partition(where_condition=where_condition)
                if not max_clustering_values:
                    return

                where_condition_to_delete = []
                for pk_clause, clustering_value in max_clustering_values.items():
                    where_condition_to_delete.append(f" {pk_clause} and "
                                                     f"{clustering_value['ck_name']} > {int(clustering_value['value'] / 2)}")

                self.log.debug("Where condition for delete: %s", where_condition_to_delete)

                self.delete_partition(where_condition_to_delete=where_condition_to_delete)
                time.sleep(self.interval)
                self.validate_deleted_rows_not_resurrected(where_condition_to_delete=where_condition_to_delete)
                return
            except Exception as error:  # pylint: disable=broad-except
                self.log.error(traceback.format_exc())
                case_event.severity = Severity.CRITICAL if isinstance(error, ResurrectionDataFound) else Severity.ERROR
                case_event.message = str(error)
                return


# pylint: disable=too-many-instance-attributes
class DataOperationThread(Thread):
    """
    Run thread with data manipulations during entire test. Sleep between operations.
    For now one operation is supported: delete full partition
    """
    DEFAULT_INTERVAL_BETWEEN_OPERATIONS_SEC = 900

    def __init__(self, tester, thread_name: str):
        self.tester = tester
        self.interval = self.DEFAULT_INTERVAL_BETWEEN_OPERATIONS_SEC
        self.schema_helpers = SchemaHelpers(tester=self.tester)
        self.data_operations = DataOperations(schema_helpers=self.schema_helpers, interval=self.interval)
        # parameter "data_operation_ks_class" defined as:
        #  - {'replication_factor': 3}
        #  OR
        #  - {'replication_factor': ['eu-westscylla_node_west': 3, 'us-west-2scylla_node_west': 2]}
        ks_class = self.tester.params.get("data_operation_ks_class")
        self.schema_helpers.create_default_keyspace_query(replication_factor=ks_class["replication_factor"])
        self.duration_sec = self.tester.params.get('test_duration') * 60
        self.log = logging.getLogger(self.__class__.__name__)
        self.scenarios = [self.data_operations.delete_full_partition, self.data_operations.delete_half_partition]
        # TODO: decrease gc_grace_seconds?
        self.tester.run_stress(stress_cmd=self.schema_helpers.DEFAULT_STRESS_WRITE_COMMAND)
        super().__init__(daemon=True)
        self.name = f"{self.__class__.__name__}_{thread_name}"
        self.termination_event = Event()

    @raise_event_on_failure
    def run(self):
        end_time = time.time() + self.duration_sec

        with DataOperationsEvent() as do_event:
            try:

                while time.time() < end_time and not self.termination_event.is_set():
                    rows_number = self.data_operations.count_rows(entity=self.schema_helpers.table_name)
                    if not rows_number or rows_number < 1000:
                        # Run pre-fill data
                        self.tester.run_stress(stress_cmd=self.schema_helpers.DEFAULT_STRESS_WRITE_COMMAND)

                    time.sleep(self.interval)
                    # TODO: save all removed data and validate few times during the run
                    self.scenarios[random.randint(0, len(self.scenarios) - 1)]()

            except Exception as error:  # pylint: disable=broad-except
                self.log.error(traceback.format_exc())
                do_event.severity = Severity.CRITICAL if isinstance(error, ResurrectionDataFound) else Severity.ERROR
                do_event.message = str(error)

    def stop(self):
        self.termination_event.set()
