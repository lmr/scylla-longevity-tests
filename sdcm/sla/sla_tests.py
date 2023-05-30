#!/usr/bin/env python
import logging
import random
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from sdcm.sct_events import Severity
from sdcm.sct_events.system import TestStepEvent
from sdcm.sla.libs.sla_utils import SlaUtils
from sdcm.utils.common import ParallelObject
from sdcm.utils.loader_utils import DEFAULT_USER, DEFAULT_USER_PASSWORD, SERVICE_LEVEL_NAME_TEMPLATE, \
    STRESS_ROLE_NAME_TEMPLATE, STRESS_ROLE_PASSWORD_TEMPLATE, MAX_ALLOWED_SERVICE_LEVELS
from test_lib.sla import create_sla_auth, ServiceLevel, Role

LOGGER = logging.getLogger(__name__)


class Steps(SlaUtils):
    @staticmethod
    def unique_subsrtr_for_name():
        return str(uuid.uuid1()).split("-", maxsplit=1)[0]

    # pylint: disable=too-many-arguments
    def run_stress_and_validate_scheduler_runtime_during_load(self, tester, read_cmds, prometheus_stats, read_roles,
                                                              stress_queue, sleep=600):
        # Wait for server levels will be propagated to all nodes
        time.sleep(15)
        # pylint: disable=not-context-manager
        with TestStepEvent(step="Run stress command and validate scheduler runtime during load") as wp_event:
            try:
                start_time = time.time() + 60
                # pylint: disable=protected-access
                tester._run_all_stress_cmds(stress_queue, params={'stress_cmd': read_cmds, 'round_robin': True})
                time.sleep(sleep)
                end_time = time.time()

                self.validate_scheduler_runtime(start_time=start_time,
                                                end_time=end_time,
                                                read_users=read_roles,
                                                prometheus_stats=prometheus_stats,
                                                db_cluster=tester.db_cluster,
                                                possible_issue={'less resources': 'scylla-enterprise#2717'}
                                                )
                return None
            except Exception as details:  # pylint: disable=broad-except
                wp_event.add_error([str(details)])
                wp_event.full_traceback = traceback.format_exc()
                wp_event.severity = Severity.ERROR
                return wp_event

    # pylint: disable=too-many-arguments
    def alter_sl_and_validate_scheduler_runtime(self, tester, service_level, new_shares, read_roles, prometheus_stats,
                                                sleep=600):
        # pylint: disable=not-context-manager
        with TestStepEvent(step=f"Alter shares from {service_level.shares} to {new_shares} Service "
                                f"Level {service_level.name} and validate scheduler runtime "
                                f"during load") as wp_event:
            try:
                service_level.alter(new_shares=new_shares)
                start_time = time.time() + 60
                time.sleep(sleep)
                end_time = time.time()
                self.validate_scheduler_runtime(start_time=start_time,
                                                end_time=end_time,
                                                read_users=read_roles,
                                                prometheus_stats=prometheus_stats,
                                                db_cluster=tester.db_cluster,
                                                possible_issue={'less resources': "scylla-enterprise#949"})
                return None
            except Exception as details:  # pylint: disable=broad-except
                wp_event.add_error([str(details)])
                wp_event.full_traceback = traceback.format_exc()
                wp_event.severity = Severity.ERROR
                return wp_event

    # pylint: disable=too-many-arguments
    @staticmethod
    def detach_service_level_and_run_load(sl_for_detach, role_with_sl_to_detach, sleep=600):
        # pylint: disable=not-context-manager
        with TestStepEvent(step=f"Detach service level {sl_for_detach.name} with {sl_for_detach.shares} shares from "
                                f"{role_with_sl_to_detach.name}.") as wp_event:
            try:
                role_with_sl_to_detach.detach_service_level()
                time.sleep(sleep)
                return None
            except Exception as details:  # pylint: disable=broad-except
                wp_event.add_error([str(details)])
                wp_event.full_traceback = traceback.format_exc()
                wp_event.severity = Severity.ERROR
                return wp_event

    # pylint: disable=too-many-arguments
    @staticmethod
    def drop_service_level_and_run_load(sl_for_drop, role_with_sl_to_drop, sleep=600):
        # pylint: disable=not-context-manager
        with TestStepEvent(step=f"Drop service level {sl_for_drop.name} with {role_with_sl_to_drop.name}.") as wp_event:
            try:
                sl_for_drop.drop()
                role_with_sl_to_drop.reset_service_level()
                time.sleep(sleep)
                return None
            except Exception as details:  # pylint: disable=broad-except
                wp_event.add_error([str(details)])
                wp_event.full_traceback = traceback.format_exc()
                wp_event.severity = Severity.ERROR
                return wp_event

    # pylint: disable=too-many-arguments
    def attach_sl_and_validate_scheduler_runtime(self, tester, new_service_level, role_for_attach,
                                                 read_roles, prometheus_stats, sleep=600,
                                                 # restart_scylla parameter is temporary - wWorkaround for issue
                                                 # https://github.com/scylladb/scylla-enterprise/issues/2572
                                                 restart_scylla=False):
        # pylint: disable=not-context-manager
        with TestStepEvent(step=f"Attach service level {new_service_level.name} with "
                                f"{new_service_level.shares} shares to {role_for_attach.name}. "
                                f"Validate scheduler runtime during load") as wp_event:
            try:
                role_for_attach.attach_service_level(new_service_level)

                # Workaround for issue https://github.com/scylladb/scylla-enterprise/issues/2572
                if restart_scylla:
                    time.sleep(30)
                    nodes_for_restart = [node for node in tester.db_cluster.nodes if node.jmx_up() and node.db_up()
                                         and not node.running_nemesis]
                    tester.db_cluster.restart_binary_protocol(nodes=nodes_for_restart)
                # End workaround

                start_time = time.time() + 60
                time.sleep(sleep)
                end_time = time.time()
                self.validate_scheduler_runtime(start_time=start_time,
                                                end_time=end_time,
                                                read_users=read_roles,
                                                prometheus_stats=prometheus_stats,
                                                db_cluster=tester.db_cluster,
                                                possible_issue={'less resources':
                                                                'scylla-enterprise#2572 or scylla-enterprise#2717'}
                                                )
                return None
            except Exception as details:  # pylint: disable=broad-except
                wp_event.add_error([str(details)])
                wp_event.full_traceback = traceback.format_exc()
                wp_event.severity = Severity.ERROR
                return wp_event

    def chaos_scenario(self, tester, session, num_of_partitions, stress_duration, column_definition):
        load_func_args = {'tester': tester, 'session': session, 'num_of_partitions': num_of_partitions,
                          'stress_duration': stress_duration, 'scenario_type': "_load",
                          "column_definition": column_definition}
        load_same_name_func_args = load_func_args.copy()
        load_same_name_func_args.update({"service_level_same_name": True})
        no_load_func_args = {'session': session, 'scenario_type': "_load"}
        threads = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            threads.append(executor.submit(self.load_with_role_sl, **load_func_args))
            threads.append(executor.submit(self.load_with_granted_role, **load_func_args))
            threads.append(executor.submit(self.load_with_role_default_sl, **load_func_args))
            threads.append(executor.submit(self.load_with_role_sl, **load_same_name_func_args))
            threads.append(executor.submit(self.no_load_create_alter_drop_service_level, **no_load_func_args))
            threads.append(executor.submit(self.no_load_create_alter_drop_service_level, **no_load_func_args))

            for thread in threads:
                # Run all threads 30 minutes
                thread.result(timeout=1860)

    def no_load_create_alter_drop_service_level(self, session, scenario_type, timeout=1800):
        with TestStepEvent(step="Create role with attached Service Level, alter Service Level and drop. "
                                "Load is not run") as wp_event:
            try:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    role = Role(session=session,
                                name=STRESS_ROLE_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()),
                                password=STRESS_ROLE_PASSWORD_TEMPLATE % self.unique_subsrtr_for_name(),
                                login=True, superuser=True).create()
                    shares = random.randint(400, 600)
                    role.attach_service_level(
                        ServiceLevel(session=session,
                                     name=SERVICE_LEVEL_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()),
                                     shares=shares).create())
                    role.attached_service_level.alter(new_shares=random.randint(0, 1000))
            except Exception as details:  # pylint: disable=broad-except
                if "no more scheduling groups exist" not in str(details):
                    wp_event.add_error([str(details)])
                    wp_event.full_traceback = traceback.format_exc()
                    wp_event.severity = Severity.ERROR
            finally:
                service_level = role.attached_service_level if role else None
                for entity in [service_level, role]:
                    if entity:
                        entity.drop()

    def load_with_role_sl(self, tester, session, num_of_partitions, stress_duration, scenario_type, timeout=1800,
                          service_level_same_name=False, column_definition=None):
        service_level_name = None
        if service_level_same_name:
            service_level_name = SERVICE_LEVEL_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name())

        with TestStepEvent(step="Create role with attached Service Level and run load.") as wp_event:
            try:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    role = Role(session=session,
                                name=STRESS_ROLE_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()),
                                password=STRESS_ROLE_PASSWORD_TEMPLATE % self.unique_subsrtr_for_name(),
                                login=True, superuser=True).create()
                    shares = random.randint(400, 600)
                    service_level_name = (service_level_name or
                                          SERVICE_LEVEL_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()))
                    role.attach_service_level(
                        ServiceLevel(session=session,
                                     name=service_level_name,
                                     shares=shares).create())
                    self.run_load_and_sl_changes(tester=tester, session=session, role=role,
                                                 stress_duration=stress_duration, num_of_partitions=num_of_partitions,
                                                 scenario_type=scenario_type, column_definition=column_definition)
            except Exception as details:  # pylint: disable=broad-except
                if "no more scheduling groups exist" not in str(details):
                    wp_event.add_error([str(details)])
                    wp_event.full_traceback = traceback.format_exc()
                    wp_event.severity = Severity.ERROR

    def load_with_granted_role(self, tester, session, num_of_partitions, stress_duration, scenario_type, timeout=1800,
                               column_definition=None):
        with TestStepEvent(
                step="Create role with attached Service Level, grant it to another role and run load.") as wp_event:
            try:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    role = Role(session=session,
                                name=STRESS_ROLE_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()),
                                password=STRESS_ROLE_PASSWORD_TEMPLATE % self.unique_subsrtr_for_name(),
                                login=True, superuser=True).create()
                    granted_role = self.create_role_and_grant(session=session, scenario_type=scenario_type, role=role)
                    shares = random.randint(400, 600)
                    role.attach_service_level(
                        ServiceLevel(session=session,
                                     name=SERVICE_LEVEL_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()),
                                     shares=shares).create())
                    self.run_load_and_sl_changes(tester=tester, session=session, role=role,
                                                 stress_duration=stress_duration, num_of_partitions=num_of_partitions,
                                                 scenario_type=scenario_type, granted_role=granted_role,
                                                 column_definition=column_definition)
            except Exception as details:  # pylint: disable=broad-except
                if "no more scheduling groups exist" not in str(details):
                    wp_event.add_error([str(details)])
                    wp_event.full_traceback = traceback.format_exc()
                    wp_event.severity = Severity.ERROR

    def load_with_role_default_sl(self, tester, session, num_of_partitions, stress_duration, scenario_type,
                                  timeout=1800, column_definition=None):
        with TestStepEvent(step="Create role with no attached Service Level and run load.") as wp_event:
            try:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    role = Role(session=session,
                                name=STRESS_ROLE_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()),
                                password=STRESS_ROLE_PASSWORD_TEMPLATE % self.unique_subsrtr_for_name(),
                                login=True, superuser=True).create()
                    self.run_load_and_sl_changes(tester=tester, session=session, role=role,
                                                 stress_duration=stress_duration, num_of_partitions=num_of_partitions,
                                                 scenario_type=scenario_type, column_definition=column_definition)
            except Exception as details:  # pylint: disable=broad-except
                if "no more scheduling groups exist" not in str(details):
                    wp_event.add_error([str(details)])
                    wp_event.full_traceback = traceback.format_exc()
                    wp_event.severity = Severity.ERROR

    def run_load_and_sl_changes(self, tester, session, role, stress_duration, num_of_partitions, scenario_type,
                                granted_role=None, column_definition=None):
        stress_queue = []

        try:
            load_cmds = self.define_read_cassandra_stress_command(role=role,
                                                                  load_type=self.MIXED_LOAD,
                                                                  c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                  threads=10,
                                                                  stress_duration_min=stress_duration,
                                                                  num_of_partitions=num_of_partitions,
                                                                  max_rows_for_read=num_of_partitions,
                                                                  kwargs=column_definition)

            # pylint: disable=protected-access
            stress_queue = tester._run_all_stress_cmds(
                stress_queue, params={'stress_cmd': load_cmds, 'round_robin': True})
            role.validate_role_service_level_attributes_against_db()

            if role.attached_service_level:
                # Alter attached service level in parallel and validate shares
                self.run_action_few_times_in_parallel_and_validate_shares(
                    action=role.attached_service_level.alter,
                    objects=[(random.randint(0, 350)) for _ in range(3)],
                    workers=3,
                    role=role,
                    timeout=90)

            # Attach another service level in parallel and validate shares
            self.run_action_few_times_in_parallel_and_validate_shares(
                action=self.add_new_service_level_and_attach,
                objects=[(session, scenario_type, role) for _ in range(3)],
                workers=3,
                role=role,
                timeout=90)

            time.sleep(stress_duration * 1.5)
            for stress_thread in stress_queue:
                tester.verify_stress_thread(stress_thread)
        finally:
            service_level, granted_service_level = None, None
            if role:
                service_level = role.attached_service_level
            if granted_role:
                granted_service_level = granted_role.attached_service_level

            for entity in [service_level, granted_service_level, granted_role, role]:
                if entity:
                    entity.drop()

    def add_new_service_level_and_attach(self, session, scenario_type, role):
        service_level = (self.create_service_level(session, scenario_type) or
                         self.get_service_level(session, scenario_type))
        role.attach_service_level(service_level)

    @staticmethod
    def run_action_few_times_in_parallel_and_validate_shares(action, objects, workers, role, timeout):
        alter_sl = ParallelObject(timeout=timeout, objects=objects, num_workers=workers)
        alter_sl.run(func=action, unpack_objects=True, ignore_exceptions=False)
        time.sleep(60)
        role.validate_role_service_level_attributes_against_db()

    @staticmethod
    def get_service_level(session, scenario_type=None):
        all_service_levels = ServiceLevel(session=session, name="temp").list_all_service_levels()
        if scenario_type:
            all_service_levels = [sl for sl in all_service_levels if scenario_type in sl.name]

        if all_service_levels:
            return random.choice(all_service_levels)
        else:
            return None

    def create_service_level(self, session, scenario_type):
        # If maximum allowed amount of service levels (7) exist already - choose one service level for the action
        all_service_levels = ServiceLevel(session=session, name="temp").list_all_service_levels()
        if len(all_service_levels) == MAX_ALLOWED_SERVICE_LEVELS:
            return None

        return ServiceLevel(session=session,
                            name=SERVICE_LEVEL_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()),
                            shares=random.randint(0, 1000)).create()

    def create_role_and_grant(self, session, scenario_type, role, attach_random_service_level=True,
                              attach_new_service_level=False):
        if attach_random_service_level and attach_new_service_level:
            raise ValueError("One option should be selected: attach_random_service_level or attach_new_service_level."
                             " Both options are True. Please, fix test and rerun")

        role_for_grant = Role(session=session,
                              name=STRESS_ROLE_NAME_TEMPLATE % (scenario_type, self.unique_subsrtr_for_name()),
                              password=STRESS_ROLE_PASSWORD_TEMPLATE % scenario_type, login=True).create()

        service_level = None
        if attach_random_service_level:
            service_level = self.get_service_level(session=session)

        if attach_new_service_level:
            service_level = self.create_service_level(session=session, scenario_type=scenario_type)

        if service_level:
            role_for_grant.attach_service_level(service_level)

        role_for_grant.grant_me_to(role)
        return role_for_grant


class SlaTests(Steps):
    STRESS_READ_CMD = 'cassandra-stress read cl=ALL duration={duration} -mode cql3 native user={user} ' \
                      'password={password} -rate threads={threads} -pop {pop}'

    # @staticmethod
    # def unique_subsrtr_for_name():
    #     return str(uuid.uuid1()).split("-", maxsplit=1)[0]

    @staticmethod
    def _create_sla_auth(session, shares: int, index: str, superuser: bool = True) -> Role:
        role = None
        try:
            role = create_sla_auth(session=session, shares=shares, index=index, superuser=superuser)
            return role
        except Exception:  # pylint: disable=broad-except
            if role and role.attached_service_level:
                role.attached_service_level.drop()
            if role:
                role.drop()
            raise

    @staticmethod
    def verify_stress_threads(tester, stress_queue):
        for stress in stress_queue:
            try:
                tester.verify_stress_thread(cs_thread_pool=stress)
            except Exception as error:  # pylint: disable=broad-except
                LOGGER.error("Stress verifying failed. Error: %s", error)

    @staticmethod
    def refresh_role_in_list(role_to_refresh, read_roles):
        for i, role in enumerate(read_roles):
            if role["role"] == role_to_refresh:
                read_roles[i]["service_level"] = role_to_refresh.attached_service_level

    # pylint: disable=too-many-locals
    def test_increase_shares_by_attach_another_sl_during_load(self, tester, prometheus_stats, num_of_partitions,
                                                              cassandra_stress_column_definition=None):
        low_share = 20
        high_share = 500
        error_events = []
        stress_queue = []
        auth_entity_name_index = self.unique_subsrtr_for_name()

        with tester.db_cluster.cql_connection_patient(node=tester.db_cluster.nodes[0],
                                                      user=DEFAULT_USER,
                                                      password=DEFAULT_USER_PASSWORD) as session:
            role_low = self._create_sla_auth(session=session, shares=low_share, index=auth_entity_name_index)
            role_high = self._create_sla_auth(session=session, shares=high_share, index=auth_entity_name_index)
            kwargs = {"-col": cassandra_stress_column_definition} if cassandra_stress_column_definition else {}

            # TODO: change stress_duration to 25. Now it is increased because of cluster rolling restart (workaround)
            stress_duration = 35
            read_cmds = [self.define_read_cassandra_stress_command(role=role_low,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs),
                         self.define_read_cassandra_stress_command(role=role_high,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs)
                         ]
            read_roles = [{"role": role_low, 'service_level': role_low.attached_service_level},
                          {"role": role_high, 'service_level': role_high.attached_service_level}]

            new_sl = None
            try:
                error_events.append(
                    self.run_stress_and_validate_scheduler_runtime_during_load(tester=tester, read_cmds=read_cmds,
                                                                               prometheus_stats=prometheus_stats,
                                                                               read_roles=read_roles,
                                                                               stress_queue=stress_queue))
                # Create new role and attach it instead of detached
                new_sl = ServiceLevel(session=session,
                                      name=SERVICE_LEVEL_NAME_TEMPLATE % ('800', auth_entity_name_index),
                                      shares=800).create()

                error_events.append(
                    self.attach_sl_and_validate_scheduler_runtime(tester=tester,
                                                                  new_service_level=new_sl,
                                                                  role_for_attach=role_low,
                                                                  read_roles=read_roles,
                                                                  prometheus_stats=prometheus_stats,
                                                                  sleep=600,
                                                                  restart_scylla=True))
            finally:
                self.verify_stress_threads(tester=tester, stress_queue=stress_queue)
                self.clean_auth(entities_list_of_dict=read_roles)
                if new_sl:
                    new_sl.drop()
                return error_events  # pylint: disable=lost-exception

    # pylint: disable=too-many-locals
    def test_increase_shares_during_load(self, tester, prometheus_stats, num_of_partitions,
                                         cassandra_stress_column_definition=None):
        low_share = 20
        high_share = 500
        error_events = []
        stress_queue = []
        auth_entity_name_index = self.unique_subsrtr_for_name()

        with tester.db_cluster.cql_connection_patient(node=tester.db_cluster.nodes[0],
                                                      user=DEFAULT_USER,
                                                      password=DEFAULT_USER_PASSWORD) as session:
            role_low = self._create_sla_auth(session=session, shares=low_share, index=auth_entity_name_index)
            role_high = self._create_sla_auth(session=session, shares=high_share, index=auth_entity_name_index)
            kwargs = {"-col": cassandra_stress_column_definition} if cassandra_stress_column_definition else {}

            stress_duration = 25
            read_cmds = [self.define_read_cassandra_stress_command(role=role_low,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs),
                         self.define_read_cassandra_stress_command(role=role_high,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs)
                         ]
            read_roles = [{"role": role_low, 'service_level': role_low.attached_service_level},
                          {"role": role_high, 'service_level': role_high.attached_service_level}]

            try:
                error_events.append(
                    self.run_stress_and_validate_scheduler_runtime_during_load(tester=tester, read_cmds=read_cmds,
                                                                               prometheus_stats=prometheus_stats,
                                                                               read_roles=read_roles,
                                                                               stress_queue=stress_queue))
                error_events.append(
                    self.alter_sl_and_validate_scheduler_runtime(tester=tester,
                                                                 service_level=role_low.attached_service_level,
                                                                 new_shares=900, read_roles=read_roles,
                                                                 prometheus_stats=prometheus_stats))
            finally:
                self.verify_stress_threads(tester=tester, stress_queue=stress_queue)
                self.clean_auth(entities_list_of_dict=read_roles)
                return error_events  # pylint: disable=lost-exception

    # pylint: disable=too-many-locals
    def test_decrease_shares_during_load(self, tester, prometheus_stats, num_of_partitions,
                                         cassandra_stress_column_definition=None):
        low_share = 800
        high_share = 500
        error_events = []
        stress_queue = []
        auth_entity_name_index = self.unique_subsrtr_for_name()

        with tester.db_cluster.cql_connection_patient(node=tester.db_cluster.nodes[0],
                                                      user=DEFAULT_USER,
                                                      password=DEFAULT_USER_PASSWORD) as session:
            role_low = self._create_sla_auth(session=session, shares=low_share, index=auth_entity_name_index)
            role_high = self._create_sla_auth(session=session, shares=high_share, index=auth_entity_name_index)
            kwargs = {"-col": cassandra_stress_column_definition} if cassandra_stress_column_definition else {}

            stress_duration = 25
            read_cmds = [self.define_read_cassandra_stress_command(role=role_low,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs),
                         self.define_read_cassandra_stress_command(role=role_high,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs)
                         ]
            read_roles = [{"role": role_low, 'service_level': role_low.attached_service_level},
                          {"role": role_high, 'service_level': role_high.attached_service_level}]

            try:
                error_events.append(
                    self.run_stress_and_validate_scheduler_runtime_during_load(tester=tester, read_cmds=read_cmds,
                                                                               prometheus_stats=prometheus_stats,
                                                                               read_roles=read_roles,
                                                                               stress_queue=stress_queue))
                error_events.append(
                    self.alter_sl_and_validate_scheduler_runtime(tester=tester,
                                                                 service_level=role_low.attached_service_level,
                                                                 new_shares=100, read_roles=read_roles,
                                                                 prometheus_stats=prometheus_stats))

            finally:
                self.verify_stress_threads(tester=tester, stress_queue=stress_queue)
                self.clean_auth(entities_list_of_dict=read_roles)
                return error_events  # pylint: disable=lost-exception

    # pylint: disable=too-many-locals
    def test_replace_service_level_using_detach_during_load(self, tester, prometheus_stats, num_of_partitions,
                                                            cassandra_stress_column_definition=None):
        low_share = 250
        high_share = 500
        error_events = []
        stress_queue = []
        auth_entity_name_index = self.unique_subsrtr_for_name()

        with tester.db_cluster.cql_connection_patient(node=tester.db_cluster.nodes[0],
                                                      user=DEFAULT_USER,
                                                      password=DEFAULT_USER_PASSWORD) as session:
            role_low = self._create_sla_auth(session=session, shares=low_share, index=auth_entity_name_index)
            role_high = self._create_sla_auth(session=session, shares=high_share, index=auth_entity_name_index)
            kwargs = {"-col": cassandra_stress_column_definition} if cassandra_stress_column_definition else {}

            stress_duration = 35
            read_cmds = [self.define_read_cassandra_stress_command(role=role_low,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs),
                         self.define_read_cassandra_stress_command(role=role_high,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs)
                         ]

            read_roles = [{"role": role_low, 'service_level': role_low.attached_service_level},
                          {"role": role_high, 'service_level': role_high.attached_service_level}]

            new_sl = None

            try:
                error_events.append(
                    self.run_stress_and_validate_scheduler_runtime_during_load(tester=tester, read_cmds=read_cmds,
                                                                               prometheus_stats=prometheus_stats,
                                                                               read_roles=read_roles,
                                                                               stress_queue=stress_queue))

                error_events.append(
                    self.detach_service_level_and_run_load(sl_for_detach=role_high.attached_service_level,
                                                           role_with_sl_to_detach=role_high,
                                                           sleep=600))
                self.refresh_role_in_list(role_to_refresh=role_high, read_roles=read_roles)

                # Create new role and attach it instead of detached
                new_sl = ServiceLevel(session=session,
                                      name=SERVICE_LEVEL_NAME_TEMPLATE % ('50', auth_entity_name_index),
                                      shares=50).create()

                error_events.append(
                    self.attach_sl_and_validate_scheduler_runtime(tester=tester,
                                                                  new_service_level=new_sl,
                                                                  role_for_attach=role_high,
                                                                  read_roles=read_roles,
                                                                  prometheus_stats=prometheus_stats,
                                                                  sleep=600,
                                                                  restart_scylla=True))

            finally:
                self.verify_stress_threads(tester=tester, stress_queue=stress_queue)
                self.clean_auth(entities_list_of_dict=read_roles)
                if new_sl:
                    new_sl.drop()
                return error_events  # pylint: disable=lost-exception

    # pylint: disable=too-many-locals
    def test_replace_service_level_using_drop_during_load(self, tester, prometheus_stats, num_of_partitions,
                                                          cassandra_stress_column_definition=None):
        low_share = 250
        high_share = 500
        error_events = []
        stress_queue = []
        auth_entity_name_index = self.unique_subsrtr_for_name()

        with tester.db_cluster.cql_connection_patient(node=tester.db_cluster.nodes[0],
                                                      user=DEFAULT_USER,
                                                      password=DEFAULT_USER_PASSWORD) as session:
            role_low = self._create_sla_auth(session=session, shares=low_share, index=auth_entity_name_index)
            role_high = self._create_sla_auth(session=session, shares=high_share, index=auth_entity_name_index)
            kwargs = {"-col": cassandra_stress_column_definition} if cassandra_stress_column_definition else {}

            # TODO: change stress_duration to 35. Now it is increased because of cluster rolling restart (workaround)
            stress_duration = 45
            read_cmds = [self.define_read_cassandra_stress_command(role=role_low,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs),
                         self.define_read_cassandra_stress_command(role=role_high,
                                                                   load_type=self.CACHE_ONLY_LOAD,
                                                                   c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                   threads=200,
                                                                   stress_duration_min=stress_duration,
                                                                   num_of_partitions=num_of_partitions,
                                                                   kwargs=kwargs)
                         ]

            read_roles = [{"role": role_low, "service_level": role_low.attached_service_level},
                          {"role": role_high, "service_level": role_high.attached_service_level}]

            new_sl = None

            try:
                error_events.append(
                    self.run_stress_and_validate_scheduler_runtime_during_load(tester=tester, read_cmds=read_cmds,
                                                                               prometheus_stats=prometheus_stats,
                                                                               read_roles=read_roles,
                                                                               stress_queue=stress_queue))

                error_events.append(
                    self.drop_service_level_and_run_load(sl_for_drop=role_low.attached_service_level,
                                                         role_with_sl_to_drop=role_low,
                                                         sleep=600))
                self.refresh_role_in_list(role_to_refresh=role_low, read_roles=read_roles)

                # Create new role and attach it instead of dropped
                new_sl = ServiceLevel(session=session,
                                      name=SERVICE_LEVEL_NAME_TEMPLATE % ('800', auth_entity_name_index),
                                      shares=800).create()

                error_events.append(
                    self.attach_sl_and_validate_scheduler_runtime(tester=tester,
                                                                  new_service_level=new_sl,
                                                                  role_for_attach=role_low,
                                                                  read_roles=read_roles,
                                                                  prometheus_stats=prometheus_stats,
                                                                  sleep=600,
                                                                  restart_scylla=True))

            finally:
                self.verify_stress_threads(tester=tester, stress_queue=stress_queue)
                self.clean_auth(entities_list_of_dict=read_roles)
                if new_sl:
                    new_sl.drop()
                return error_events  # pylint: disable=lost-exception

    # pylint: disable=too-many-locals,too-many-arguments
    def test_maximum_allowed_sls_with_max_shares_during_load(self, tester, prometheus_stats, num_of_partitions,
                                                             cassandra_stress_column_definition=None,
                                                             service_levels_amount=7):
        error_events = []
        stress_queue = []
        every_role_shares = 1000
        stress_duration = 20
        read_cmds = []
        read_roles = []

        with tester.db_cluster.cql_connection_patient(node=tester.db_cluster.nodes[0],
                                                      user=DEFAULT_USER,
                                                      password=DEFAULT_USER_PASSWORD) as session:
            roles = []
            kwargs = {"-col": cassandra_stress_column_definition} if cassandra_stress_column_definition else {}
            for _ in range(service_levels_amount):
                roles.append(self._create_sla_auth(session=session, shares=every_role_shares,
                                                   index=self.unique_subsrtr_for_name()))
                read_cmds.append(self.define_read_cassandra_stress_command(role=roles[-1],
                                                                           load_type=self.MIXED_LOAD,
                                                                           c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                           threads=50,
                                                                           stress_duration_min=stress_duration,
                                                                           num_of_partitions=num_of_partitions,
                                                                           max_rows_for_read=num_of_partitions,
                                                                           kwargs=kwargs))

                read_roles.append({"role": roles[-1], 'service_level': roles[-1].attached_service_level})

            try:
                error_events.append(
                    self.run_stress_and_validate_scheduler_runtime_during_load(tester=tester,
                                                                               read_cmds=read_cmds,
                                                                               prometheus_stats=prometheus_stats,
                                                                               read_roles=read_roles,
                                                                               stress_queue=stress_queue))

            finally:
                self.verify_stress_threads(tester=tester, stress_queue=stress_queue)
                self.clean_auth(entities_list_of_dict=read_roles)
                return error_events  # pylint: disable=lost-exception

    # pylint: disable=too-many-locals
    def test_seven_sls_with_different_shares_during_load(self, tester, prometheus_stats, num_of_partitions):
        error_events = []
        stress_queue = []
        auth_entity_name_index = self.unique_subsrtr_for_name()
        role_shares = [1000, 100, 200, 300, 500, 700, 800]
        stress_duration = 20
        read_cmds = []
        read_roles = []

        with tester.db_cluster.cql_connection_patient(node=tester.db_cluster.nodes[0],
                                                      user=DEFAULT_USER,
                                                      password=DEFAULT_USER_PASSWORD) as session:
            roles = []
            for shares in role_shares:
                roles.append(create_sla_auth(session=session, shares=shares, index=auth_entity_name_index))
                read_cmds.append(self.define_read_cassandra_stress_command(role=roles[-1],
                                                                           load_type=self.MIXED_LOAD,
                                                                           c_s_workload_type=self.WORKLOAD_THROUGHPUT,
                                                                           threads=50,
                                                                           stress_duration_min=stress_duration,
                                                                           num_of_partitions=num_of_partitions,
                                                                           max_rows_for_read=num_of_partitions))

                read_roles.append({"role": roles[-1], 'service_level': roles[-1].attached_service_level})

            try:
                error_events.append(
                    self.run_stress_and_validate_scheduler_runtime_during_load(tester=tester,
                                                                               read_cmds=read_cmds,
                                                                               prometheus_stats=prometheus_stats,
                                                                               read_roles=read_roles,
                                                                               stress_queue=stress_queue))

            finally:
                for stress in stress_queue:
                    tester.verify_stress_thread(cs_thread_pool=stress)
                self.clean_auth(entities_list_of_dict=read_roles)
                return error_events  # pylint: disable=lost-exception

    # pylint: disable=too-many-locals
    def test_sla_chaos(self, tester, num_of_partitions, cassandra_stress_column_definition=None):
        stress_duration = 5
        column_definition = {"-col": cassandra_stress_column_definition} if cassandra_stress_column_definition else {}
        with tester.db_cluster.cql_connection_patient(node=tester.db_cluster.nodes[0],
                                                      user=DEFAULT_USER,
                                                      password=DEFAULT_USER_PASSWORD) as session:
            self.chaos_scenario(tester=tester, session=session, num_of_partitions=num_of_partitions,
                                stress_duration=stress_duration, column_definition=column_definition)
