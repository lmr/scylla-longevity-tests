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
# Copyright (c) 2020 ScyllaDB

import shutil
import sys
import os
import time
from enum import Enum
from textwrap import dedent
from typing import Optional

from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement  # pylint: disable=no-name-in-module

from sdcm import cluster
from sdcm.tester import ClusterTester
from sdcm.gemini_thread import GeminiStressThread


class Mode(Enum):
    DELTA = 1
    PREIMAGE = 2
    POSTIMAGE = 3


def mode_str(mode: Mode) -> str:
    return {
        Mode.DELTA: 'delta',
        Mode.PREIMAGE: 'preimage',
        Mode.POSTIMAGE: 'postimage'
    }[mode]


def print_file_to_stdout(path: str) -> None:
    with open(path, "r") as file:
        shutil.copyfileobj(file, sys.stdout)


def write_cql_result(res, path: str):
    """Write a CQL select result to a file.

    :param res: cql results
    :type res: ResultSet
    :param path: path to file
    :type path: str
    """
    with open(path, 'w') as file:
        for row in res:
            file.write(str(row) + '\n')
        file.flush()
        os.fsync(file.fileno())


SCYLLA_MIGRATE_URL = "https://kbr-scylla.s3-eu-west-1.amazonaws.com/scylla-migrate"
REPLICATOR_URL = "https://kbr-scylla.s3-eu-west-1.amazonaws.com/scylla-cdc-replicator-0.0.1-SNAPSHOT-jar-with-dependencies.jar"


class CDCReplicationTest(ClusterTester):
    KS_NAME = 'ks1'
    TABLE_NAME = 'table1'

    def collect_data_for_analysis(self,
                                  migrate_log_path: Optional[str], replicator_log_path: str,
                                  master_node: cluster.BaseNode, replica_node: cluster.BaseNode) -> None:
        if migrate_log_path:
            self.log.info('scylla-migrate log:')
            print_file_to_stdout(migrate_log_path)
        self.log.info('Replicator log:')
        print_file_to_stdout(replicator_log_path)

        with self.db_cluster.cql_connection_patient(node=master_node) as sess:
            self.log.info('Fetching master table...')
            res = sess.execute(SimpleStatement(f'select * from {self.KS_NAME}.{self.TABLE_NAME}',
                                               consistency_level=ConsistencyLevel.QUORUM, fetch_size=1000))
            write_cql_result(res, os.path.join(self.logdir, 'master-table'))

        with self.cs_db_cluster.cql_connection_patient(node=replica_node) as sess:
            self.log.info('Fetching replica table...')
            res = sess.execute(SimpleStatement(f'select * from {self.KS_NAME}.{self.TABLE_NAME}',
                                               consistency_level=ConsistencyLevel.QUORUM, fetch_size=1000))
            write_cql_result(res, os.path.join(self.logdir, 'replica-table'))

    def test_replication_cs(self) -> None:
        self.log.info('Using cassandra-stress to generate workload.')
        self.test_replication(False, Mode.DELTA)

    def test_replication_gemini(self, mode: Mode) -> None:
        self.log.info('Using gemini to generate workload. Mode: {}'.format(mode.name))
        self.test_replication(True, mode)

    def test_replication_gemini_delta(self) -> None:
        self.test_replication_gemini(Mode.DELTA)

    def test_replication_gemini_preimage(self) -> None:
        self.test_replication_gemini(Mode.PREIMAGE)

    def test_replication_gemini_postimage(self) -> None:
        self.test_replication_gemini(Mode.POSTIMAGE)

    # pylint: disable=too-many-statements,too-many-branches,too-many-locals
    def test_replication(self, is_gemini_test: bool, mode: Mode) -> None:
        assert is_gemini_test or (mode == Mode.DELTA), "cassandra-stress doesn't work with preimage/postimage modes"

        self.consistency_ok = False

        self.log.info('Waiting for the latest CDC generation to start...')
        # 2 * ring_delay (ring_delay = 30s) + leeway
        time.sleep(70)

        self.log.info('Starting stressor.')
        if is_gemini_test:
            stress_thread = GeminiStressThread(
                test_cluster=self.db_cluster,
                oracle_cluster=None,
                loaders=self.loaders,
                gemini_cmd=self.params.get('gemini_cmd'),
                timeout=self.get_duration(None),
                outputdir=self.loaders.logdir,
                params=self.params).run()
        else:
            stress_thread = self.run_stress_cassandra_thread(stress_cmd=self.params.get('stress_cmd'))

        self.log.info('Let stressor run for a while...')
        # Wait for C-S to create keyspaces/tables/UTs
        time.sleep(20)

        self.log.info('Fetching schema definitions from master cluster.')
        master_node = self.db_cluster.nodes[0]
        with self.db_cluster.cql_connection_patient(node=master_node) as sess:
            sess.cluster.refresh_schema_metadata()
            # For some reason, `refresh_schema_metadata` doesn't refresh immediatelly...
            time.sleep(10)
            ks = sess.cluster.metadata.keyspaces[self.KS_NAME]
            ut_ddls = [t[1].as_cql_query() for t in ks.user_types.items()]
            table_ddls = []
            for name, table in ks.tables.items():
                if name.endswith('_scylla_cdc_log'):
                    continue
                # Don't enable CDC on the replica cluster
                if 'cdc' in table.extensions:
                    del table.extensions['cdc']
                table_ddls.append(table.as_cql_query())

        if ut_ddls:
            self.log.info('User types:\n{}'.format('\n'.join(ut_ddls)))
        self.log.info('Table definitions:\n{}'.format('\n'.join(table_ddls)))

        self.log.info('Creating schema on replica cluster.')
        replica_node = self.cs_db_cluster.nodes[0]
        with self.cs_db_cluster.cql_connection_patient(node=replica_node) as sess:
            sess.execute(f"create keyspace if not exists {self.KS_NAME}"
                         " with replication = {'class': 'SimpleStrategy', 'replication_factor': 1}")
            for stmt in ut_ddls + table_ddls:
                sess.execute(stmt)

        self.log.info('Starting nemesis.')
        self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(), tester_obj=self)
        self.db_cluster.start_nemesis()

        loader_node = self.loaders.nodes[0]

        self.log.info('Installing tmux on loader node.')
        res = loader_node.remoter.run(cmd='sudo yum install -y tmux')
        if res.exit_status != 0:
            self.fail('Could not install tmux.')

        self.log.info('Getting scylla-migrate on loader node.')
        res = loader_node.remoter.run(cmd=f'wget {SCYLLA_MIGRATE_URL} -O scylla-migrate && chmod +x scylla-migrate')
        if res.exit_status != 0:
            self.fail('Could not obtain scylla-migrate.')

        self.log.info('Getting replicator on loader node.')
        res = loader_node.remoter.run(cmd=f'wget {REPLICATOR_URL} -O replicator.jar')
        if res.exit_status != 0:
            self.fail('Could not obtain CDC replicator.')

        # We run the replicator in a tmux session so that remoter.run returns immediately
        # (the replicator will run in the background). We redirect the output to a log file for later extraction.
        replicator_script = dedent("""
            (cat >runreplicator.sh && chmod +x runreplicator.sh && tmux new-session -d -s 'replicator' ./runreplicator.sh) <<'EOF'
            #!/bin/bash

            java -cp replicator.jar com.scylladb.scylla.cdc.replicator.Main -k {} -t {} -s {} -d {} -cl {} -m {} 2>&1 | tee replicatorlog
            EOF
        """.format(self.KS_NAME, self.TABLE_NAME, master_node.external_address, replica_node.external_address, 'one', mode_str(mode)))

        self.log.info('Replicator script:\n{}'.format(replicator_script))

        self.log.info('Starting replicator.')
        res = loader_node.remoter.run(cmd=replicator_script)
        if res.exit_status != 0:
            self.fail('Could not start CDC replicator.')

        self.log.info('Let stressor run for a while...')
        time.sleep(30)

        self.log.info('Stopping nemesis before bootstrapping a new node.')
        self.db_cluster.stop_nemesis(timeout=600)

        self.log.info('Let stressor run for a while...')
        time.sleep(30)

        self.log.info('Bootstrapping a new node...')
        new_node = self.db_cluster.add_nodes(count=1, enable_auto_bootstrap=True)[0]
        self.log.info('Waiting for new node to finish initializing...')
        self.db_cluster.wait_for_init(node_list=[new_node])
        self.monitors.reconfigure_scylla_monitoring()

        self.log.info('Bootstrapped, restarting nemesis.')
        self.db_cluster.start_nemesis()

        self.log.info('Waiting for stressor to finish...')
        if is_gemini_test:
            stress_results = self.verify_gemini_results(queue=stress_thread)
            self.log.info('gemini results: {}'.format(stress_results))
        else:
            stress_results = stress_thread.get_results()
            self.log.info('cassandra-stress results: {}'.format(list(stress_results)))

        self.log.info('Waiting for replicator to finish (sleeping 60s)...')
        time.sleep(60)

        self.log.info('Stopping nemesis.')
        self.db_cluster.stop_nemesis(timeout=600)

        self.log.info('Fetching replicator logs.')
        replicator_log_path = os.path.join(self.logdir, 'replicator.log')
        loader_node.remoter.receive_files(src='replicatorlog', dst=replicator_log_path)

        self.log.info('Comparing table contents using scylla-migrate...')
        res = loader_node.remoter.run(cmd='./scylla-migrate check --master-address {} --replica-address {}'
                                      ' --ignore-schema-difference {} {}.{} 2>&1 | tee migratelog'.format(
                                          master_node.external_address, replica_node.external_address,
                                          # Timestamps don't have to match in postimage mode
                                          '--no-writetime' if mode == Mode.POSTIMAGE else '',
                                          self.KS_NAME, self.TABLE_NAME))
        migrate_ok = res.ok

        migrate_log_path = os.path.join(self.logdir, 'scylla-migrate.log')
        loader_node.remoter.receive_files(src='migratelog', dst=migrate_log_path)
        with open(migrate_log_path) as file:
            self.consistency_ok = 'Consistency check OK.\n' in (line for line in file)

        if mode == Mode.PREIMAGE:
            with open(replicator_log_path) as file:
                self.consistency_ok = self.consistency_ok \
                        and not 'Inconsistency detected.\n' in (line for line in file)

        if not self.consistency_ok:
            self.log.error('Inconsistency detected.')
        if not migrate_ok:
            self.log.error('scylla-migrate command returned status {}'.format(res.exit_status))

        if self.consistency_ok and migrate_ok:
            self.log.info('Consistency check successful.')
        else:
            self.collect_data_for_analysis(
                migrate_log_path, replicator_log_path,
                master_node, replica_node)
            self.fail('Consistency check failed.')

    def get_email_data(self):
        self.log.info("Prepare data for email")

        email_data = self._get_common_email_data()
        grafana_dataset = self.monitors.get_grafana_screenshot_and_snapshot(self.start_time) if self.monitors else {}
        email_data.update({"grafana_screenshots": grafana_dataset.get("screenshots", []),
                           "grafana_snapshots": grafana_dataset.get("snapshots", []),
                           "nemesis_details": self.get_nemesises_stats(),
                           "nemesis_name": self.params.get("nemesis_class_name"),
                           "number_of_db_nodes": self.params.get('n_db_nodes'),
                           "scylla_ami_id": self.params.get("ami_id_db_scylla", "-"),
                           "scylla_instance_type": self.params.get('instance_type_db',
                                                                   self.params.get('gce_instance_type_db')),
                           "scylla_version": self.db_cluster.nodes[0].scylla_version if self.db_cluster else "N/A",
                           "number_of_oracle_nodes": self.params.get("n_test_oracle_db_nodes", 1),
                           "oracle_ami_id": self.params.get("ami_id_db_oracle"),
                           "oracle_db_version":
                               self.cs_db_cluster.nodes[0].scylla_version if self.cs_db_cluster else "N/A",
                           "oracle_instance_type": self.params.get("instance_type_db_oracle"),
                           "consistency_status": self.consistency_ok
                           })

        return email_data
