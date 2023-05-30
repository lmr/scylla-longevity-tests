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
# Copyright (c) 2022 ScyllaDB

import re
import time

from sdcm.sct_events import Severity
from sdcm.sct_events.system import TestFrameworkEvent

DEFAULT_USER = "cassandra"
DEFAULT_USER_PASSWORD = "cassandra"
STRESS_ROLE_NAME_TEMPLATE = 'role%s_%s'
STRESS_ROLE_PASSWORD_TEMPLATE = 'rolep%s'
SERVICE_LEVEL_NAME_TEMPLATE = 'sl%s_%s'
MAX_ALLOWED_SERVICE_LEVELS = 7


class LoaderUtilsMixin:
    """This mixin can be added to any class that inherits the 'ClusterTester' one"""

    def _create_counter_table(self):
        """
        workaround for the issue https://github.com/scylladb/scylla-tools-java/issues/32
        remove when resolved
        """
        node = self.db_cluster.nodes[0]
        with self.db_cluster.cql_connection_patient(node) as session:
            session.execute("""
                CREATE KEYSPACE IF NOT EXISTS keyspace1
                WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '3'} AND durable_writes = true;
            """)
            session.execute("""
                CREATE TABLE IF NOT EXISTS keyspace1.counter1 (
                    key blob PRIMARY KEY,
                    "C0" counter,
                    "C1" counter,
                    "C2" counter,
                    "C3" counter,
                    "C4" counter
                ) WITH COMPACT STORAGE
                    AND bloom_filter_fp_chance = 0.01
                    AND caching = '{"keys":"ALL","rows_per_partition":"ALL"}'
                    AND comment = ''
                    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
                    AND compression = {}
                    AND gc_grace_seconds = 864000
                    AND default_time_to_live = 0
                    AND max_index_interval = 2048
                    AND min_index_interval = 128
                    AND read_repair_chance = 0.0
                    AND dclocal_read_repair_chance = 0.1
                    AND memtable_flush_period_in_ms = 0
                    AND speculative_retry = '99.0PERCENTILE';
            """)

    def assemble_and_run_all_stress_cmd(self, stress_queue, stress_cmd, keyspace_num):
        if stress_cmd:
            # Stress: Same as in prepare_write - allow the load to be spread across all loaders when using multi ks
            if keyspace_num > 1 and self.params.get('round_robin'):
                self.log.debug("Using round_robin for multiple Keyspaces...")
                for i in range(1, keyspace_num + 1):
                    keyspace_name = self._get_keyspace_name(i)
                    params = {'keyspace_name': keyspace_name, 'round_robin': True, 'stress_cmd': stress_cmd}

                    self._run_all_stress_cmds(stress_queue, params)

            # The old method when we run all stress_cmds for all keyspace on the same loader, or in round-robin if defined in test yaml
            else:
                params = {'keyspace_num': keyspace_num, 'stress_cmd': stress_cmd,
                          'round_robin': self.params.get('round_robin')}
                self._run_all_stress_cmds(stress_queue, params)

    def _run_all_stress_cmds(self, stress_queue, params):
        stress_cmds = params['stress_cmd']
        if not isinstance(stress_cmds, list):
            stress_cmds = [stress_cmds]
        # In some cases we want the same stress_cmd to run several times (can be used with round_robin or not).
        stress_multiplier = self.params.get('stress_multiplier')
        if stress_multiplier > 1:
            stress_cmds *= stress_multiplier

        for stress_cmd in stress_cmds:
            stress_params = dict(params)
            stress_params.update({'stress_cmd': stress_cmd})

            # Due to an issue with scylla & cassandra-stress - we need to create the counter table manually
            if 'counter_' in stress_cmd:
                self._create_counter_table()

            if 'compression' in stress_cmd:
                if 'keyspace_name' not in stress_params:
                    compression_prefix = re.search('compression=(.*)Compressor', stress_cmd).group(1)
                    keyspace_name = "keyspace_{}".format(compression_prefix.lower())
                    stress_params.update({'keyspace_name': keyspace_name})

            # Run all stress commands
            self.log.debug('stress cmd: {}'.format(stress_cmd))
            if stress_cmd.startswith('scylla-bench'):
                stress_queue.append(self.run_stress_thread(stress_cmd=stress_cmd,
                                                           stats_aggregate_cmds=False,
                                                           round_robin=self.params.get('round_robin')))
            else:
                stress_queue.append(self.run_stress_thread(**stress_params))

            time.sleep(10)

        return stress_queue

    def run_stress_and_verify_threads(self, params=None):
        stress_queue = []

        self._run_all_stress_cmds(stress_queue, params=params)

        for queue in stress_queue:
            self.verify_stress_thread(cs_thread_pool=queue)

        return stress_queue

    @staticmethod
    def _get_keyspace_name(ks_number, keyspace_pref='keyspace'):
        return '{}{}'.format(keyspace_pref, ks_number)

    def _run_cql_commands(self, cmds, node=None):
        node = node if node else self.db_cluster.nodes[0]

        if not isinstance(cmds, list):
            cmds = [cmds]

        for cmd in cmds:
            # pylint: disable=no-member
            with self.db_cluster.cql_connection_patient(node) as session:
                session.execute(cmd)

    def _pre_create_keyspace(self):
        cmds = self.params.get('pre_create_keyspace')
        self._run_cql_commands(cmds)

    def run_post_prepare_cql_cmds(self):
        if post_prepare_cql_cmds := self.params.get('post_prepare_cql_cmds'):
            self.log.debug("Execute post prepare queries: %s", post_prepare_cql_cmds)
            self._run_cql_commands(post_prepare_cql_cmds)

    def run_prepare_write_cmd(self):
        # In some cases (like many keyspaces), we want to create the schema (all keyspaces & tables) before the load
        # starts - due to the heavy load, the schema propogation can take long time and c-s fails.
        prepare_write_cmd = self.params.get('prepare_write_cmd')
        keyspace_num = self.params.get('keyspace_num')
        write_queue = []
        verify_queue = []

        if not prepare_write_cmd:
            self.log.debug("No prepare write commands are configured to run. Continue with stress commands")
            return
        # When the load is too heavy for one loader when using MULTI-KEYSPACES, the load is spreaded evenly across
        # the loaders (round_robin).
        if keyspace_num > 1 and self.params.get('round_robin'):
            self.log.debug("Using round_robin for multiple Keyspaces...")
            for i in range(1, keyspace_num + 1):
                keyspace_name = self._get_keyspace_name(i)
                self._run_all_stress_cmds(write_queue, params={'stress_cmd': prepare_write_cmd,
                                                               'keyspace_name': keyspace_name,
                                                               'round_robin': True})
        # Not using round_robin and all keyspaces will run on all loaders
        else:
            self._run_all_stress_cmds(write_queue, params={'stress_cmd': prepare_write_cmd,
                                                           'keyspace_num': keyspace_num,
                                                           'round_robin': self.params.get('round_robin')})

        # In some cases we don't want the nemesis to run during the "prepare" stage in order to be 100% sure that
        # all keys were written succesfully
        if self.params.get('nemesis_during_prepare'):
            # Wait for some data (according to the param in the yaml) to be populated, for multi keyspace need to
            # pay attention to the fact it checks only on keyspace1
            self.db_cluster.wait_total_space_used_per_node(keyspace=None)
            self.db_cluster.start_nemesis()

        # Wait on the queue till all threads come back.
        # todo: we need to improve this part for some cases that threads are being killed and we don't catch it.
        for stress in write_queue:
            self.verify_stress_thread(cs_thread_pool=stress)

        # Run nodetool flush on all nodes to make sure nothing left in memory
        # I decided to comment this out for now, when we found the data corruption bug, we wanted to be on the safe
        # side, but I don't think we should continue with this approach.
        # If we decided to add this back in the future, we need to wrap it with try-except because it can run
        # in parallel to nemesis and it will fail on one of the nodes.
        # self._flush_all_nodes()

        # In case we would like to verify all keys were written successfully before we start other stress / nemesis
        prepare_verify_cmd = self.params.get('prepare_verify_cmd')
        if prepare_verify_cmd:
            self._run_all_stress_cmds(verify_queue, params={'stress_cmd': prepare_verify_cmd,
                                                            'keyspace_num': keyspace_num})

            for stress in verify_queue:
                self.verify_stress_thread(cs_thread_pool=stress)

        self.run_post_prepare_cql_cmds()

        prepare_wait_no_compactions_timeout = self.params.get('prepare_wait_no_compactions_timeout')
        if prepare_wait_no_compactions_timeout:
            for node in self.db_cluster.nodes:
                node.run_nodetool("compact")
            self.wait_no_compactions_running(n=prepare_wait_no_compactions_timeout)
        self.log.info('Prepare finished')

    @staticmethod
    def add_sla_credentials_to_stress_cmds(workload_names: list, roles, params, parent_class_name: str):
        def _set_credentials_to_cmd(cmd):
            if roles and "<sla credentials " in cmd:
                if 'user=' in cmd:
                    # if stress command is not defined as expected, stop the tests and fix it. Then re-run
                    raise EnvironmentError("Stress command is defined wrong. Credentials already applied. Remove "
                                           f"unnecessary and re-run the test. Command: {cmd}")

                index = re.search(r"<sla credentials (\d+)>", cmd)
                role_index = int(index.groups(0)[0]) if index else None
                if role_index is None:
                    # if stress command is not defined as expected, stop the tests and fix it. Then re-run
                    raise EnvironmentError("Stress command is defined wrong. Expected pattern '<credentials \\d>' was "
                                           f"not found. Fix the command and re-run the test. Command: {cmd}")
                sla_role_name = roles[role_index].name.replace('"', '')
                sla_role_password = roles[role_index].password
                return re.sub(r'<sla credentials \d+>', f'user={sla_role_name} password={sla_role_password}', cmd)
            return cmd

        try:
            for stress_op in workload_names:
                stress_cmds = []
                stress_params = params.get(stress_op)
                if isinstance(stress_params, str):
                    stress_params = [stress_params]

                if not stress_params:
                    continue

                for stress_cmd in stress_params:
                    # cover multitenant test
                    if isinstance(stress_cmd, list):
                        cmds = []
                        for current_cmd in stress_cmd:
                            cmds.append(_set_credentials_to_cmd(cmd=current_cmd))
                        stress_cmds.append(cmds)
                    else:
                        stress_cmds.append(_set_credentials_to_cmd(cmd=stress_cmd))

                params[stress_op] = stress_cmds
        except EnvironmentError as error_message:
            TestFrameworkEvent(source=parent_class_name, message=error_message, severity=Severity.CRITICAL).publish()
            raise
