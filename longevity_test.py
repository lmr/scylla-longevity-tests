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
import re
import time
import string
import tempfile
import itertools

import yaml
from cassandra import AlreadyExists, InvalidRequest


from sdcm.tester import ClusterTester


class LongevityTest(ClusterTester):
    """
    Test a Scylla cluster stability over a time period.
    """

    def _run_all_stress_cmds(self, stress_queue, params):
        stress_cmds = params['stress_cmd']
        if not isinstance(stress_cmds, list):
            stress_cmds = [stress_cmds]
        # In some cases we want the same stress_cmd to run several times (can be used with round_robin or not).
        stress_multiplier = self.params.get('stress_multiplier')
        if stress_multiplier > 1:
            stress_cmds *= stress_multiplier

        for stress_cmd in stress_cmds:
            params.update({'stress_cmd': stress_cmd})
            self._parse_stress_cmd(stress_cmd, params)

            # Run all stress commands
            self.log.debug('stress cmd: {}'.format(stress_cmd))
            if stress_cmd.startswith('scylla-bench'):
                stress_queue.append(self.run_stress_thread(stress_cmd=stress_cmd,
                                                           stats_aggregate_cmds=False,
                                                           round_robin=self.params.get('round_robin')))
            else:
                stress_queue.append(self.run_stress_thread(**params))

            time.sleep(10)

            # Remove "user profile" param for the next command
            if 'profile' in params:
                del params['profile']

            if 'keyspace_name' in params:
                del params['keyspace_name']

        return stress_queue

    def _parse_stress_cmd(self, stress_cmd, params):
        # Due to an issue with scylla & cassandra-stress - we need to create the counter table manually
        if 'counter_' in stress_cmd:
            self._create_counter_table()

        if 'compression' in stress_cmd:
            if 'keyspace_name' not in params:
                compression_prefix = re.search('compression=(.*)Compressor', stress_cmd).group(1)
                keyspace_name = "keyspace_{}".format(compression_prefix.lower())
                params.update({'keyspace_name': keyspace_name})

        return params

    default_params = {'timeout': 650000}

    @staticmethod
    def _get_keyspace_name(ks_number, keyspace_pref='keyspace'):
        return '{}{}'.format(keyspace_pref, ks_number)

    def _get_fullscan_params(self):
        params = {}
        fullscan = self.params.get('run_fullscan')
        if fullscan:
            fullscan = fullscan.split(',')
            params['ks.cf'] = fullscan[0].strip()
            params['interval'] = int(fullscan[1].strip())
            self.log.info('Fullscan target: {} Fullscan interval: {}'.format(params['ks.cf'],
                                                                             params['interval']))
        return params

    def run_pre_create_schema(self):
        pre_create_schema = self.params.get('pre_create_schema')
        keyspace_num = self.params.get('keyspace_num')
        if pre_create_schema:
            self._pre_create_schema(keyspace_num, scylla_encryption_options=self.params.get(
                'scylla_encryption_options'))

    def run_pre_create_keyspace(self):
        if self.params.get('pre_create_keyspace'):
            self._pre_create_keyspace()

    def run_prepare_write_cmd(self):
        # In some cases (like many keyspaces), we want to create the schema (all keyspaces & tables) before the load
        # starts - due to the heavy load, the schema propogation can take long time and c-s fails.
        prepare_write_cmd = self.params.get('prepare_write_cmd')
        keyspace_num = self.params.get('keyspace_num')
        write_queue = list()
        verify_queue = list()

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

        post_prepare_cql_cmds = self.params.get('post_prepare_cql_cmds')
        if post_prepare_cql_cmds:
            self._run_cql_commands(post_prepare_cql_cmds)

        prepare_wait_no_compactions_timeout = self.params.get('prepare_wait_no_compactions_timeout')
        if prepare_wait_no_compactions_timeout:
            for node in self.db_cluster.nodes:
                node.run_nodetool("compact")
            self.wait_no_compactions_running(n=prepare_wait_no_compactions_timeout)
        self.log.info('Prepare finished')

    def test_custom_time(self):
        """
        Run cassandra-stress with params defined in data_dir/scylla.yaml
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements

        self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(),
                                    tester_obj=self)
        stress_queue = list()

        # prepare write workload
        prepare_write_cmd = self.params.get('prepare_write_cmd')
        keyspace_num = self.params.get('keyspace_num')

        self.pre_create_alternator_tables()

        self.run_pre_create_keyspace()
        self.run_pre_create_schema()
        self.run_prepare_write_cmd()

        # Collect data about partitions and their rows amount
        validate_partitions = self.params.get('validate_partitions')
        table_name, primary_key_column, partitions_dict_before = '', '', {}
        if validate_partitions:
            table_name = self.params.get('table_name')
            primary_key_column = self.params.get('primary_key_column')
            self.log.debug('Save partitions info before reads')
            partitions_dict_before = self.collect_partitions_info(table_name=table_name,
                                                                  primary_key_column=primary_key_column,
                                                                  save_into_file_name='partitions_rows_before.log')
            if partitions_dict_before is None:
                validate_partitions = False

        stress_cmd = self.params.get('stress_cmd')
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

        customer_profiles = self.params.get('cs_user_profiles')
        if customer_profiles:
            cs_duration = self.params.get('cs_duration')
            for cs_profile in customer_profiles:
                assert os.path.exists(cs_profile), 'File not found: {}'.format(cs_profile)
                self.log.debug('Run stress test with user profile {}, duration {}'.format(cs_profile, cs_duration))
                profile_dst = os.path.join('/tmp', os.path.basename(cs_profile))
                with open(cs_profile) as pconf:
                    cont = pconf.readlines()
                    user_profile_table_count = self.params.get(  # pylint: disable=invalid-name
                        'user_profile_table_count')
                    for i in range(user_profile_table_count):
                        for cmd in [line.lstrip('#').strip() for line in cont if line.find('cassandra-stress') > 0]:
                            stress_cmd = (cmd.format(profile_dst, cs_duration))
                            params = {'stress_cmd': stress_cmd, 'profile': cs_profile}
                            self.log.debug('Stress cmd: {}'.format(stress_cmd))
                            self._run_all_stress_cmds(stress_queue, params)

        fullscan = self._get_fullscan_params()
        if fullscan:
            self.log.info('Fullscan target: {} Fullscan interval: {}'.format(fullscan['ks.cf'],
                                                                             fullscan['interval']))
            self.run_fullscan_thread(ks_cf=fullscan['ks.cf'], interval=fullscan['interval'])

        # Check if we shall wait for total_used_space or if nemesis wasn't started
        if not prepare_write_cmd or not self.params.get('nemesis_during_prepare'):
            self.db_cluster.wait_total_space_used_per_node(keyspace=None)
            self.db_cluster.start_nemesis()

        stress_read_cmd = self.params.get('stress_read_cmd')
        if stress_read_cmd:
            params = {'keyspace_num': keyspace_num, 'stress_cmd': stress_read_cmd}
            self._run_all_stress_cmds(stress_queue, params)

        for stress in stress_queue:
            self.verify_stress_thread(cs_thread_pool=stress)

        if (stress_read_cmd or stress_cmd) and validate_partitions:
            self.log.debug('Save partitions info after reads')
            partitions_dict_after = self.collect_partitions_info(table_name=table_name,
                                                                 primary_key_column=primary_key_column,
                                                                 save_into_file_name='partitions_rows_after.log')
            if partitions_dict_after is not None:
                self.assertEqual(partitions_dict_before, partitions_dict_after,
                                 msg='Row amount in partitions is not same before and after running of nemesis')

    def test_batch_custom_time(self):
        """
        The test runs like test_custom_time but designed for running multiple stress commands in batches.
        It take the keyspace_num and calculates the number of batches to run based on batch_size.
        For every batch, it runs the stress and verify them and only then moves to the next batch.

        Test assumes:
        - pre_create_schema (The test pre-creating the schema for all batches)
        - round_robin
        - No nemesis during prepare

        :param keyspace_num: Number of keyspaces to be batched (in future it can be enhanced with number of tables).
        :param batch_size: Number of stress commands to run together in a batch.
        """
        self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(), tester_obj=self)

        total_stress = self.params.get('keyspace_num')  # In future it may be 1 keyspace but multiple tables in it.
        batch_size = self.params.get('batch_size')

        prepare_write_cmd = self.params.get('prepare_write_cmd')
        if prepare_write_cmd:
            self._run_stress_in_batches(total_stress=total_stress, batch_size=batch_size,
                                        stress_cmd=prepare_write_cmd)

        self.db_cluster.start_nemesis()

        stress_cmd = self.params.get('stress_cmd')
        self._run_stress_in_batches(total_stress=batch_size, batch_size=batch_size,
                                    stress_cmd=stress_cmd)

    def test_user_batch_custom_time(self):
        """
        The test runs like test_custom_time but designed for running multiple stress commands on user profile with multiple
        tables in one keyspace

        It uses batches of `batch_size` size, until it reach `user_profile_table_count`
        For every batch, it runs the stress command in parallel and verify them and only then moves to the next batch.

        Test assumes:
        - pre_create_schema (The test pre-creating the schema for all batches)
        - round_robin
        - No nemesis during prepare

        :param user_profile_table_count: Number of tables to be batched.
        :param batch_size: Number of stress commands to run together in a batch.
        """

        self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(), tester_obj=self)

        batch_size = self.params.get('batch_size')

        if not self.params.get('reuse_cluster'):
            self._pre_create_templated_user_schema()

            # Start new nodes
            # we are starting this test case with only one db to make creating of the tables quicker
            # gossip with multiple node cluster make this painfully slower
            add_node_cnt = self.params.get('add_node_cnt')

            for _ in range(add_node_cnt):
                new_nodes = self.db_cluster.add_nodes(count=1, enable_auto_bootstrap=True)
                self.monitors.reconfigure_scylla_monitoring()
                self.db_cluster.wait_for_init(node_list=new_nodes)

        self.db_cluster.start_nemesis()

        stress_params_list = list()

        customer_profiles = self.params.get('cs_user_profiles')

        templated_table_counter = itertools.count()

        if customer_profiles:
            cs_duration = self.params.get('cs_duration')
            for cs_profile in customer_profiles:
                assert os.path.exists(cs_profile), 'File not found: {}'.format(cs_profile)
                self.log.debug('Run stress test with user profile {}, duration {}'.format(cs_profile, cs_duration))

                user_profile_table_count = self.params.get('user_profile_table_count')  # pylint: disable=invalid-name

                for _ in range(user_profile_table_count):
                    stress_params_list += self.create_templated_user_stress_params(next(templated_table_counter),
                                                                                   cs_profile)

            self._run_user_stress_in_batches(batch_size=batch_size,
                                             stress_params_list=stress_params_list)

    def _run_user_stress_in_batches(self, batch_size, stress_params_list):
        """
        run user profile in batches, while adding 4 stress-commands which are not with precreated tables
        and wait for them to finish

        :param batch_size: size of the batch
        :param stress_params_list: the list of all stress commands
        :return:
        """
        def chunks(_list, chunk_size):
            """Yield successive n-sized chunks from _list."""
            for i in range(0, len(_list), chunk_size):
                yield _list[i:i + chunk_size], i, i+chunk_size, len(_list) + i * 2

        for batch, _, _, extra_tables_idx in list(chunks(stress_params_list, batch_size)):

            stress_queue = list()
            batch_params = dict(round_robin=True, stress_cmd=[], timeout=self.params.get('cs_duration'))

            # add few stress threads with tables that weren't pre-created
            customer_profiles = self.params.get('cs_user_profiles')
            for cs_profile in customer_profiles:
                # for now we'll leave to just one fresh table, to kick schema update
                num_of_newly_created_tables = 1
                self._pre_create_templated_user_schema(batch_start=extra_tables_idx,
                                                       batch_end=extra_tables_idx+num_of_newly_created_tables)
                for i in range(num_of_newly_created_tables):
                    batch += self.create_templated_user_stress_params(extra_tables_idx + i, cs_profile=cs_profile)

            for params in batch:
                batch_params['stress_cmd'] += [params['stress_cmd']]

            self._run_all_stress_cmds(stress_queue, params=batch_params)
            for stress in stress_queue:
                self.verify_stress_thread(cs_thread_pool=stress)

    def _run_stress_in_batches(self, total_stress, batch_size, stress_cmd):
        stress_queue = list()
        pre_create_schema = self.params.get('pre_create_schema')

        if pre_create_schema:
            self._pre_create_schema(keyspace_num=total_stress,
                                    scylla_encryption_options=self.params.get('scylla_encryption_options'))

        num_of_batches = int(total_stress / batch_size)
        for batch in range(0, num_of_batches):
            for i in range(1 + batch * batch_size, (batch + 1) * batch_size + 1):
                keyspace_name = self._get_keyspace_name(i)
                self._run_all_stress_cmds(stress_queue, params={'stress_cmd': stress_cmd,
                                                                'keyspace_name': keyspace_name, 'round_robin': True})
            for stress in stress_queue:
                self.verify_stress_thread(cs_thread_pool=stress)

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
                    AND dclocal_read_repair_chance = 0.1
                    AND default_time_to_live = 0
                    AND gc_grace_seconds = 864000
                    AND max_index_interval = 2048
                    AND memtable_flush_period_in_ms = 0
                    AND min_index_interval = 128
                    AND read_repair_chance = 0.0
                    AND speculative_retry = '99.0PERCENTILE';
            """)

    @staticmethod
    def _get_columns_num_of_single_stress(single_stress_cmd):
        if '-col' not in single_stress_cmd:
            return None
        col_num = None
        params_list = single_stress_cmd.split()
        col_params_list = []
        for param in params_list[params_list.index('-col')+1:]:
            col_params_list.append(param.strip("'"))
            if param.endswith("'"):
                break
        for param in col_params_list:
            if param.startswith('n='):
                col_num = int(re.findall(r'\b\d+\b', param)[0])
                break
        return col_num

    def _get_prepare_write_cmd_columns_num(self):
        prepare_write_cmd = self.params.get('prepare_write_cmd')
        if not prepare_write_cmd:
            return None
        if isinstance(prepare_write_cmd, str):
            prepare_write_cmd = [prepare_write_cmd]
        return max([self._get_columns_num_of_single_stress(single_stress_cmd=stress) for stress in prepare_write_cmd])

    def _pre_create_schema(self, keyspace_num=1, in_memory=False, scylla_encryption_options=None):
        """
        For cases we are testing many keyspaces and tables, It's a possibility that we will do it better and faster than
        cassandra-stress.
        """

        self.log.debug('Pre Creating Schema for c-s with {} keyspaces'.format(keyspace_num))
        compaction_strategy = self.params.get('compaction_strategy')
        sstable_size = self.params.get('sstable_size')
        for i in range(1, keyspace_num+1):
            keyspace_name = 'keyspace{}'.format(i)
            self.create_keyspace(keyspace_name=keyspace_name, replication_factor=3)
            self.log.debug('{} Created'.format(keyspace_name))
            col_num = self._get_prepare_write_cmd_columns_num() or 5
            columns = {}
            for col_idx in range(col_num):
                cs_key = '"C'+str(col_idx)+'"'
                columns[cs_key] = 'blob'
            self.create_table(name='standard1', keyspace_name=keyspace_name, key_type='blob', read_repair=0.0, compact_storage=True,
                              columns=columns,
                              in_memory=in_memory, scylla_encryption_options=scylla_encryption_options,
                              compaction=compaction_strategy, sstable_size=sstable_size)

    def _pre_create_templated_user_schema(self, batch_start=None, batch_end=None):
        # pylint: disable=too-many-locals
        user_profile_table_count = self.params.get(  # pylint: disable=invalid-name
            'user_profile_table_count') or 0
        cs_user_profiles = self.params.get('cs_user_profiles')
        # read user-profile
        for profile_file in cs_user_profiles:
            profile_yaml = yaml.safe_load(open(profile_file))
            keyspace_definition = profile_yaml['keyspace_definition']
            keyspace_name = profile_yaml['keyspace']
            table_template = string.Template(profile_yaml['table_definition'])

            with self.db_cluster.cql_connection_patient(node=self.db_cluster.nodes[0]) as session:
                try:
                    session.execute(keyspace_definition)
                except AlreadyExists:
                    self.log.debug("keyspace [{}] exists".format(keyspace_name))

                if batch_start is not None and batch_end is not None:
                    table_range = range(batch_start, batch_end)
                else:
                    table_range = range(user_profile_table_count)
                self.log.debug('Pre Creating Schema for c-s with {} user tables'.format(user_profile_table_count))
                for i in table_range:
                    table_name = 'table{}'.format(i)
                    query = table_template.substitute(table_name=table_name)
                    try:
                        session.execute(query)
                    except AlreadyExists:
                        self.log.debug('table [{}] exists'.format(table_name))
                    self.log.debug('{} Created'.format(table_name))

                    for definition in profile_yaml.get('extra_definitions', []):
                        query = string.Template(definition).substitute(table_name=table_name)
                        try:
                            session.execute(query)
                        except (AlreadyExists, InvalidRequest) as exc:
                            self.log.debug('extra definition for [{}] exists [{}]'.format(table_name, str(exc)))

    def _pre_create_keyspace(self):
        cmds = self.params.get('pre_create_keyspace')
        self._run_cql_commands(cmds)

    def _run_cql_commands(self, cmds, node=None):
        node = node if node else self.db_cluster.nodes[0]

        if not isinstance(cmds, list):
            cmds = [cmds]

        for cmd in cmds:
            # pylint: disable=no-member
            with self.db_cluster.cql_connection_patient(node) as session:
                session.execute(cmd)

    def _flush_all_nodes(self):
        """
        This function will connect all db nodes in the cluster and run "nodetool flush" command.
        :return:
        """
        for node in self.db_cluster.nodes:
            node.run_nodetool("flush")

    def get_email_data(self):
        self.log.info("Prepare data for email")

        email_data = self._get_common_email_data()
        grafana_dataset = self.monitors.get_grafana_screenshot_and_snapshot(self.start_time) if self.monitors else {}
        email_data.update({"grafana_screenshots": grafana_dataset.get("screenshots", []),
                           "grafana_snapshots": grafana_dataset.get("snapshots", []),
                           "nemesis_details": self.get_nemesises_stats(),
                           "nemesis_name": self.params.get("nemesis_class_name"),
                           "scylla_ami_id": self.params.get("ami_id_db_scylla") or "-", })

        return email_data

    def create_templated_user_stress_params(self, idx, cs_profile):  # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        params_list = []
        cs_duration = self.params.get('cs_duration')

        with open(cs_profile) as pconf:
            cont = pconf.readlines()
            pconf.seek(0)
            template = string.Template(pconf.read())
            prefix, suffix = os.path.splitext(os.path.basename(cs_profile))
            table_name = "table%s" % idx

            with tempfile.NamedTemporaryFile(mode='w+', prefix=prefix, suffix=suffix, delete=False, encoding='utf-8') as file_obj:
                output = template.substitute(table_name=table_name)
                file_obj.write(output)
                profile_dst = file_obj.name

            # collect stress command from the comment in the end of the profile yaml
            # example:
            # cassandra-stress user profile={} cl=QUORUM 'ops(insert=1)' duration={} -rate threads=100 -pop 'dist=gauss(0..1M)'
            for cmd in [line.lstrip('#').strip() for line in cont if line.find('cassandra-stress') > 0]:
                stress_cmd = (cmd.format(profile_dst, cs_duration))
                params = {'stress_cmd': stress_cmd, 'profile': profile_dst}
                self.log.debug('Stress cmd: {}'.format(stress_cmd))
                params_list.append(params)

        return params_list
