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

from pathlib import Path
import random
import time
import re
from functools import wraps

from pkg_resources import parse_version

from sdcm import wait
from sdcm.fill_db_data import FillDatabaseData
from sdcm.utils.version_utils import is_enterprise, get_node_supported_sstable_versions
from sdcm.sct_events.system import InfoEvent
from sdcm.sct_events.database import IndexSpecialColumnErrorEvent
from sdcm.sct_events.group_common_events import ignore_upgrade_schema_errors, ignore_ycsb_connection_refused


def truncate_entries(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        # Perform validation of truncate entries in case the new version is 3.1 or more
        node = args[0]
        if self.truncate_entries_flag:
            base_version = self.params.get('scylla_version')
            system_truncated = bool(parse_version(base_version) >= parse_version('3.1')
                                    and not is_enterprise(base_version))
            with self.db_cluster.cql_connection_patient(node, keyspace='truncate_ks') as session:
                self.cql_truncate_simple_tables(session=session, rows=self.insert_rows)
                self.validate_truncated_entries_for_table(session=session, system_truncated=system_truncated)

        func_result = func(self, *args, **kwargs)

        result = node.remoter.run('scylla --version')
        new_version = result.stdout
        if new_version and parse_version(new_version) >= parse_version('3.1'):
            # re-new connection
            with self.db_cluster.cql_connection_patient(node, keyspace='truncate_ks') as session:
                self.validate_truncated_entries_for_table(session=session, system_truncated=True)
                self.read_data_from_truncated_tables(session=session)
                self.cql_insert_data_to_simple_tables(session=session, rows=self.insert_rows)
        return func_result
    return inner


def check_reload_systemd_config(node):
    for i in ['scylla-server', 'scylla-jmx']:
        result = node.remoter.run('systemctl status %s' % i, ignore_status=True)
        if ".service changed on disk. Run 'systemctl daemon-reload' to reload units" in result.stderr:
            raise Exception("Systemd config is changed, but not reload automatically")


class UpgradeTest(FillDatabaseData):
    """
    Test a Scylla cluster upgrade.
    """
    orig_ver = None
    new_ver = None

    # `major_release` (eg: 2.1 <-> 2.2, 2017.1 <-> 2018.1)
    # `reinstall` (opensource <-> enterprise, enterprise <-> opensource)
    # `minor_release` (eg: 2.2.1 <-> 2.2.5, 2018.1.0 <-> 2018.1.1)
    upgrade_rollback_mode = None

    # expected format version after upgrade and nodetool upgradesstables called
    # would be recalculated after all the cluster finish upgrade
    expected_sstable_format_version = 'mc'

    insert_rows = None
    truncate_entries_flag = False

    def read_data_from_truncated_tables(self, session):
        session.execute("USE truncate_ks")
        truncate_query = 'SELECT COUNT(*) FROM {}'
        tables_name = self.get_tables_name_of_keyspace(session=session, keyspace_name='truncate_ks')
        for table_name in tables_name:
            count = self.rows_to_list(session.execute(truncate_query.format(table_name)))
            self.assertEqual(str(count[0][0]), '0',
                             msg='Expected that there is no data in the table truncate_ks.{}, but found {} rows'
                             .format(table_name, count[0][0]))

    def validate_truncated_entries_for_table(self, session, system_truncated=False):  # pylint: disable=invalid-name
        tables_id = self.get_tables_id_of_keyspace(session=session, keyspace_name='truncate_ks')

        for table_id in tables_id:
            if system_truncated:
                # validate truncation entries in the system.truncated table - expected entry
                truncated_time = self.get_truncated_time_from_system_truncated(session=session, table_id=table_id)
                self.assertTrue(truncated_time,
                                msg='Expected truncated entry in the system.truncated table, but it\'s not found')

            # validate truncation entries in the system.local table - not expected entry
            truncated_time = self.get_truncated_time_from_system_local(session=session)
            if system_truncated:
                self.assertEqual(truncated_time, [[None]],
                                 msg='Not expected truncated entry in the system.local table, but it\'s found')
            else:
                self.assertTrue(truncated_time,
                                msg='Expected truncated entry in the system.local table, but it\'s not found')

    @truncate_entries
    def upgrade_node(self, node, upgrade_sstables=True):
        # pylint: disable=too-many-branches,too-many-statements
        new_scylla_repo = self.params.get('new_scylla_repo')
        new_version = self.params.get('new_version')
        upgrade_node_packages = self.params.get('upgrade_node_packages')

        self.log.info('Upgrading a Node')
        node.upgrade_system()

        # We assume that if update_db_packages is not empty we install packages from there.
        # In this case we don't use upgrade based on new_scylla_repo(ignored sudo yum update scylla...)
        result = node.remoter.run('scylla --version')
        self.orig_ver = result.stdout

        if upgrade_node_packages:
            # update_scylla_packages
            node.remoter.send_files(upgrade_node_packages, '/tmp/scylla', verbose=True)
            # node.remoter.run('sudo yum update -y --skip-broken', connect_timeout=900)
            node.remoter.run('sudo yum install python34-PyYAML -y')
            # replace the packages
            node.remoter.run(r'rpm -qa scylla\*')
            # flush all memtables to SSTables
            node.run_nodetool("drain", timeout=3600, coredump_on_timeout=True)
            node.run_nodetool("snapshot")
            node.stop_scylla_server()
            # update *development* packages
            node.remoter.run('sudo rpm -UvhR --oldpackage /tmp/scylla/*development*', ignore_status=True)
            # and all the rest
            node.remoter.run('sudo rpm -URvh --replacefiles /tmp/scylla/*.rpm | true')
            node.remoter.run(r'rpm -qa scylla\*')
        elif new_scylla_repo:
            # backup the data
            node.remoter.run('sudo cp /etc/scylla/scylla.yaml /etc/scylla/scylla.yaml-backup')
            if node.is_rhel_like():
                node.remoter.run('sudo cp /etc/yum.repos.d/scylla.repo ~/scylla.repo-backup')
                node.remoter.run(
                    r'for conf in $( rpm -qc $(rpm -qa | grep scylla) | grep -v contains ); do sudo cp -v $conf $conf.autobackup; done')
            else:
                node.remoter.run('sudo cp /etc/apt/sources.list.d/scylla.list ~/scylla.list-backup')
                node.remoter.run(
                    r'for conf in $(cat /var/lib/dpkg/info/scylla-*server.conffiles /var/lib/dpkg/info/scylla-*conf.conffiles /var/lib/dpkg/info/scylla-*jmx.conffiles | grep -v init ); do sudo cp -v $conf $conf.backup; done')
            assert new_scylla_repo.startswith('http')
            node.download_scylla_repo(new_scylla_repo)
            # flush all memtables to SSTables
            node.run_nodetool("drain", timeout=3600, coredump_on_timeout=True)
            node.run_nodetool("snapshot")
            node.stop_scylla_server(verify_down=False)

            orig_is_enterprise = node.is_enterprise
            if node.is_rhel_like():
                result = node.remoter.run("sudo yum search scylla-enterprise 2>&1", ignore_status=True)
                new_is_enterprise = bool('scylla-enterprise.x86_64' in result.stdout or
                                         'No matches found' not in result.stdout)
            else:
                result = node.remoter.run("sudo apt-cache search scylla-enterprise", ignore_status=True)
                new_is_enterprise = 'scylla-enterprise' in result.stdout

            scylla_pkg = 'scylla-enterprise' if new_is_enterprise else 'scylla'
            if orig_is_enterprise != new_is_enterprise:
                self.upgrade_rollback_mode = 'reinstall'
            ver_suffix = r'\*{}'.format(new_version) if new_version else ''
            if self.upgrade_rollback_mode == 'reinstall':
                if node.is_rhel_like():
                    node.remoter.run(r'sudo yum remove scylla\* -y')
                    node.remoter.run('sudo yum install {}{} -y'.format(scylla_pkg, ver_suffix))
                    node.remoter.run(
                        r'for conf in $( rpm -qc $(rpm -qa | grep scylla) | grep -v contains ); do sudo cp -v $conf.autobackup $conf; done')
                else:
                    node.remoter.run(r'sudo apt-get remove scylla\* -y')
                    # fixme: add publick key
                    node.remoter.run(
                        r'sudo apt-get install {}{} -y '
                        r'-o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" '.format(scylla_pkg, ver_suffix))
                    node.remoter.run(r'for conf in $(cat /var/lib/dpkg/info/scylla-*server.conffiles /var/lib/dpkg/info/scylla-*conf.conffiles /var/lib/dpkg/info/scylla-*jmx.conffiles'
                                     r' | grep -v init ); do sudo cp -v $conf $conf.backup-2.1; done')
            else:
                if node.is_rhel_like():
                    node.remoter.run(r'sudo yum update {}{}\* -y'.format(scylla_pkg, ver_suffix))
                else:
                    node.remoter.run('sudo apt-get update')
                    node.remoter.run(
                        r'sudo apt-get dist-upgrade {} -y '
                        r'-o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" '.format(scylla_pkg))
        if self.params.get('test_sst3'):
            node.remoter.run("echo 'enable_sstables_mc_format: true' |sudo tee --append /etc/scylla/scylla.yaml")
        if self.params.get('test_upgrade_from_installed_3_1_0'):
            node.remoter.run("echo 'enable_3_1_0_compatibility_mode: true' |sudo tee --append /etc/scylla/scylla.yaml")
        authorization_in_upgrade = self.params.get('authorization_in_upgrade')
        if authorization_in_upgrade:
            node.remoter.run("echo 'authorizer: \"%s\"' |sudo tee --append /etc/scylla/scylla.yaml" %
                             authorization_in_upgrade)
        check_reload_systemd_config(node)
        # Current default 300s aren't enough for upgrade test of Debian 9.
        # Related issue: https://github.com/scylladb/scylla-cluster-tests/issues/1726
        node.start_scylla_server(verify_up_timeout=500)
        result = node.remoter.run('scylla --version')
        new_ver = result.stdout
        assert self.orig_ver != self.new_ver, "scylla-server version isn't changed"
        self.new_ver = new_ver

        if upgrade_sstables:
            self.upgradesstables_if_command_available(node)

    @truncate_entries
    def rollback_node(self, node, upgrade_sstables=True):
        # pylint: disable=too-many-branches,too-many-statements

        self.log.info('Rollbacking a Node')
        # fixme: auto identify new_introduced_pkgs, remove this parameter
        new_introduced_pkgs = self.params.get('new_introduced_pkgs')
        result = node.remoter.run('scylla --version')
        orig_ver = result.stdout
        # flush all memtables to SSTables
        node.run_nodetool("drain", timeout=3600, coredump_on_timeout=True)
        # backup the data
        node.run_nodetool("snapshot")
        node.stop_scylla_server(verify_down=False)

        if node.is_rhel_like():
            node.remoter.run('sudo cp ~/scylla.repo-backup /etc/yum.repos.d/scylla.repo')
            node.remoter.run('sudo chown root.root /etc/yum.repos.d/scylla.repo')
            node.remoter.run('sudo chmod 644 /etc/yum.repos.d/scylla.repo')
        else:
            node.remoter.run('sudo cp ~/scylla.list-backup /etc/apt/sources.list.d/scylla.list')
            node.remoter.run('sudo chown root.root /etc/apt/sources.list.d/scylla.list')
            node.remoter.run('sudo chmod 644 /etc/apt/sources.list.d/scylla.list')
        node.update_repo_cache()

        if re.findall(r'\d+.\d+', self.orig_ver)[0] == re.findall(r'\d+.\d+', self.new_ver)[0]:
            self.upgrade_rollback_mode = 'minor_release'

        if self.upgrade_rollback_mode == 'reinstall' or not node.is_rhel_like():
            if node.is_rhel_like():
                node.remoter.run(r'sudo yum remove scylla\* -y')
                node.remoter.run(r'sudo yum install %s -y' % node.scylla_pkg())
                node.remoter.run(
                    r'for conf in $( rpm -qc $(rpm -qa | grep scylla) | grep -v contains ); do sudo cp -v '
                    r'$conf.autobackup $conf; done')
            else:
                node.remoter.run(r'sudo apt-get remove scylla\* -y')
                node.remoter.run(
                    r'sudo apt-get install %s -y '
                    r'-o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" ' % node.scylla_pkg())
                node.remoter.run(
                    r'for conf in $(cat /var/lib/dpkg/info/scylla-*server.conffiles /var/lib/dpkg/info/scylla-*conf.conffiles /var/lib/dpkg/info/scylla-*jmx.conffiles | grep -v init ); do sudo cp -v $conf.backup $conf; done')

            node.remoter.run('sudo systemctl daemon-reload')

        elif self.upgrade_rollback_mode == 'minor_release':
            node.remoter.run(r'sudo yum downgrade scylla\*%s-\* -y' % self.orig_ver.split('-')[0])
        else:
            if new_introduced_pkgs:
                node.remoter.run('sudo yum remove %s -y' % new_introduced_pkgs)
            node.remoter.run(r'sudo yum downgrade scylla\* -y')
            if new_introduced_pkgs:
                node.remoter.run('sudo yum install %s -y' % node.scylla_pkg())
            if node.is_rhel_like():
                node.remoter.run(
                    r'for conf in $( rpm -qc $(rpm -qa | grep scylla) | grep -v contains ); do sudo cp -v $conf.autobackup $conf; done')
            else:
                node.remoter.run(
                    r'for conf in $(cat /var/lib/dpkg/info/scylla-*server.conffiles /var/lib/dpkg/info/scylla-*conf.conffiles /var/lib/dpkg/info/scylla-*jmx.conffiles | grep -v init ); do sudo cp -v $conf.backup $conf; done')

            node.remoter.run('sudo systemctl daemon-reload')

        node.remoter.run('sudo cp /etc/scylla/scylla.yaml-backup /etc/scylla/scylla.yaml')
        result = node.remoter.run('sudo find /var/lib/scylla/data/system')
        snapshot_name = re.findall(r"system/peers-[a-z0-9]+/snapshots/(\d+)\n", result.stdout)
        # cmd = r"DIR='/var/lib/scylla/data/system'; for i in `sudo ls $DIR`;do sudo test -e $DIR/$i/snapshots/%s && sudo find $DIR/$i/snapshots/%s -type f -exec sudo /bin/cp {} $DIR/$i/ \;; done" % (snapshot_name[0], snapshot_name[0])
        # recover the system tables
        if self.params.get('recover_system_tables'):
            node.remoter.send_files('./data_dir/recover_system_tables.sh', '/tmp/')
            node.remoter.run('bash /tmp/recover_system_tables.sh %s' % snapshot_name[0], verbose=True)
        if self.params.get('test_sst3'):
            node.remoter.run(
                r'sudo sed -i -e "s/enable_sstables_mc_format:/#enable_sstables_mc_format:/g" /etc/scylla/scylla.yaml')
        if self.params.get('test_upgrade_from_installed_3_1_0'):
            node.remoter.run(
                r'sudo sed -i -e "s/enable_3_1_0_compatibility_mode:/#enable_3_1_0_compatibility_mode:/g" /etc/scylla/scylla.yaml')
        if self.params.get('remove_authorization_in_rollback'):
            node.remoter.run('sudo sed -i -e "s/authorizer:/#authorizer:/g" /etc/scylla/scylla.yaml')
        # Current default 300s aren't enough for upgrade test of Debian 9.
        # Related issue: https://github.com/scylladb/scylla-cluster-tests/issues/1726
        node.start_scylla_server(verify_up_timeout=500)
        result = node.remoter.run('scylla --version')
        new_ver = result.stdout
        self.log.info('original scylla-server version is %s, latest: %s', orig_ver, new_ver)
        assert orig_ver != new_ver, "scylla-server version isn't changed"

        if upgrade_sstables:
            self.upgradesstables_if_command_available(node)

    def upgradesstables_if_command_available(self, node, queue=None):  # pylint: disable=invalid-name
        upgradesstables_available = False
        upgradesstables_supported = node.remoter.run(
            'nodetool help | grep -q upgradesstables && echo "yes" || echo "no"')
        if "yes" in upgradesstables_supported.stdout:
            upgradesstables_available = True
            self.log.info("calling upgradesstables")

            # NOTE: some 4.3.x and 4.4.x scylla images have nodetool with bug [1]
            # that is not yet fixed [3] there.
            # So, in such case we must use '/etc/scylla/cassandra' path for
            # the 'cassandra-rackdc.properties' file instead of the expected
            # one - '/etc/scylla' [2].
            # Example of the error:
            #     WARN  16:42:29,831 Unable to read cassandra-rackdc.properties
            #     error: DC or rack not found in snitch properties,
            #         check your configuration in: cassandra-rackdc.properties
            #
            # [1] https://github.com/scylladb/scylla-tools-java/commit/3eca0e35
            # [2] https://github.com/scylladb/scylla/issues/7930
            # [3] https://github.com/scylladb/scylla-tools-java/pull/232
            main_dir, subdir = Path("/etc/scylla"), "cassandra"
            filename = "cassandra-rackdc.properties"
            result = node.remoter.run(
                f"cp {main_dir / filename} {main_dir / subdir / filename}")

            node.run_nodetool(sub_cmd="upgradesstables", args="-a")
        if queue:
            queue.put(upgradesstables_available)
            queue.task_done()

    def get_highest_supported_sstable_version(self):  # pylint: disable=invalid-name
        """
        find the highest sstable format version supported in the cluster

        :return:
        """
        output = []
        for node in self.db_cluster.nodes:
            output.extend(get_node_supported_sstable_versions(node.system_log))
        return max(set(output))

    def wait_for_sstable_upgrade(self, node, queue=None):
        all_tables_upgraded = True

        def wait_for_node_to_finish():
            try:
                result = node.remoter.sudo(
                    r"find /var/lib/scylla/data/system -type f ! -path '*snapshots*' -printf %f\\n")
                all_sstable_files = result.stdout.splitlines()

                sstable_version_regex = re.compile(r'(\w+)-\d+-(.+)\.(db|txt|sha1|crc32)')

                sstable_versions = {sstable_version_regex.search(f).group(
                    1) for f in all_sstable_files if sstable_version_regex.search(f)}

                assert len(sstable_versions) == 1, "expected all table format to be the same found {}".format(sstable_versions)
                assert list(sstable_versions)[0] == self.expected_sstable_format_version, "expected to format version to be '{}', found '{}'".format(
                    self.expected_sstable_format_version, list(sstable_versions)[0])
            except Exception as ex:  # pylint: disable=broad-except
                self.log.warning(ex)
                return False
            else:
                return True

        try:
            self.log.info("Start waiting for upgardesstables to finish")
            wait.wait_for(func=wait_for_node_to_finish, step=30, timeout=900, throw_exc=True,
                          text="Waiting until upgardesstables is finished")
        except Exception:  # pylint: disable=broad-except
            all_tables_upgraded = False
        finally:
            if queue:
                queue.put(all_tables_upgraded)
                queue.task_done()

    default_params = {'timeout': 650000}

    def test_upgrade_cql_queries(self):
        """
        Run a set of different cql queries against various types/tables before
        and after upgrade of every node to check the consistency of data
        """
        self.truncate_entries_flag = False  # not perform truncate entries test
        self.log.info('Populate DB with many types of tables and data')
        self.fill_db_data()
        self.log.info('Run some Queries to verify data BEFORE UPGRADE')
        self.verify_db_data()

        self.log.info('Starting c-s write workload to pupulate 10M paritions')
        # YAML: stress_cmd: cassandra-stress write cl=QUORUM n=10000000 -schema 'replication(factor=3)' -port jmx=6868
        # -mode cql3 native -rate threads=1000 -pop seq=1..10000000
        stress_cmd = self._cs_add_node_flag(self.params.get('stress_cmd'))
        self.run_stress_thread(stress_cmd=stress_cmd)

        self.log.info('Sleeping for 360s to let cassandra-stress populate some data before the mixed workload')
        time.sleep(600)

        self.log.info('Starting c-s read workload for 60m')
        # YAML: stress_cmd_1: cassandra-stress read cl=QUORUM duration=60m -schema 'replication(factor=3)'
        # -port jmx=6868 -mode cql3 native -rate threads=100 -pop seq=1..10000000
        stress_cmd_1 = self._cs_add_node_flag(self.params.get('stress_cmd_1'))
        stress_queue = self.run_stress_thread(stress_cmd=stress_cmd_1)

        self.log.info('Sleeping for 300s to let cassandra-stress start before the upgrade...')
        time.sleep(300)

        nodes_num = len(self.db_cluster.nodes)
        # prepare an array containing the indexes
        indexes = list(range(nodes_num))
        # shuffle it so we will upgrade the nodes in a random order
        random.shuffle(indexes)

        # upgrade all the nodes in random order
        for i in indexes:
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[i]
            self.log.info('Upgrade Node %s begin', self.db_cluster.node_to_upgrade.name)
            self.upgrade_node(self.db_cluster.node_to_upgrade)
            time.sleep(300)
            self.log.info('Upgrade Node %s ended', self.db_cluster.node_to_upgrade.name)

        self.log.info('Run some Queries to verify data AFTER UPGRADE')
        self.verify_db_data()
        self.verify_stress_thread(stress_queue)

    def fill_and_verify_db_data(self, note, pre_fill=False, rewrite_data=True):
        if pre_fill:
            self.log.info('Populate DB with many types of tables and data')
            self.fill_db_data()
        self.log.info('Run some Queries to verify data %s', note)
        self.verify_db_data()
        if rewrite_data:
            self.log.info('Re-Populate DB with many types of tables and data')
            self.fill_db_data()

    # Added to cover the issue #5621: upgrade from 3.1 to 3.2 fails on std::logic_error (Column idx_token doesn't exist
    # in base and this view is not backing a secondary index)
    # @staticmethod
    def search_for_idx_token_error_after_upgrade(self, node, step):
        self.log.debug('Search for idx_token error. Step {}'.format(step))
        idx_token_error = list(node.follow_system_log(
            patterns=["Column idx_token doesn't exist"], start_from_beginning=True))
        if idx_token_error:
            IndexSpecialColumnErrorEvent(
                message=f'Node: {node.name}. Step: {step}. '
                        f'Found error: index special column "idx_token" is not recognized'
            ).publish()

    def test_rolling_upgrade(self):  # pylint: disable=too-many-locals,too-many-statements
        """
        Upgrade half of nodes in the cluster, and start special read workload
        during the stage. Checksum method is changed to xxhash from Scylla 2.2,
        we want to use this case to verify the read (cl=ALL) workload works
        well, upgrade all nodes to new version in the end.
        """
        # In case the target version >= 3.1 we need to perform test for truncate entries
        target_upgrade_version = self.params.get('target_upgrade_version')
        self.truncate_entries_flag = False
        if target_upgrade_version and parse_version(target_upgrade_version) >= parse_version('3.1') and \
                not is_enterprise(target_upgrade_version):
            self.truncate_entries_flag = True

        self.log.info('pre-test - prepare test keyspaces and tables')
        # prepare test keyspaces and tables before upgrade to avoid schema change during mixed cluster.
        self.prepare_keyspaces_and_tables()
        self.fill_and_verify_db_data('BEFORE UPGRADE', pre_fill=True)

        # write workload during entire test
        self.log.info('Starting c-s write workload during entire test')
        write_stress_during_entire_test = self.params.get('write_stress_during_entire_test')
        entire_write_cs_thread_pool = self.run_stress_thread(stress_cmd=write_stress_during_entire_test)

        # Let to write_stress_during_entire_test complete the schema changes
        self.metric_has_data(
            metric_query='collectd_cassandra_stress_write_gauge{type="ops", keyspace="keyspace_entire_test"}', n=10)

        # Prepare keyspace and tables for truncate test
        if self.truncate_entries_flag:
            self.insert_rows = 10
            self.fill_db_data_for_truncate_test(insert_rows=self.insert_rows)
            # Let to ks_truncate complete the schema changes
            time.sleep(120)

        # generate random order to upgrade
        nodes_num = len(self.db_cluster.nodes)
        # prepare an array containing the indexes
        indexes = list(range(nodes_num))
        # shuffle it so we will upgrade the nodes in a random order
        random.shuffle(indexes)

        self.log.info('pre-test - Run stress workload before upgrade')
        # complex workload: prepare write
        self.log.info('Starting c-s complex workload (5M) to prepare data')
        stress_cmd_complex_prepare = self.params.get('stress_cmd_complex_prepare')
        complex_cs_thread_pool = self.run_stress_thread(
            stress_cmd=stress_cmd_complex_prepare, profile='data_dir/complex_schema.yaml')

        # wait for the complex workload to finish
        self.verify_stress_thread(complex_cs_thread_pool)

        self.log.info('Will check paged query before upgrading nodes')
        self.paged_query()
        self.log.info('Done checking paged query before upgrading nodes')

        # prepare write workload
        self.log.info('Starting c-s prepare write workload (n=10000000)')
        prepare_write_stress = self.params.get('prepare_write_stress')
        prepare_write_cs_thread_pool = self.run_stress_thread(stress_cmd=prepare_write_stress)
        self.log.info('Sleeping for 60s to let cassandra-stress start before the upgrade...')
        self.metric_has_data(
            metric_query='collectd_cassandra_stress_write_gauge{type="ops", keyspace="keyspace1"}', n=5)

        # start gemini write workload
        # and cdc log reader
        if self.version_cdc_support():
            self.log.info("Start gemini and cdc stressor during upgrade")
            gemini_thread = self.run_gemini(self.params.get("gemini_cmd"))
            time.sleep(30)
            # Let to write_stress_during_entire_test complete the schema changes
            self.metric_has_data(
                metric_query='gemini_cql_requests', n=10)

            cdc_reader_thread = self.run_cdclog_reader_thread(self.params.get("stress_cdclog_reader_cmd"),
                                                              keyspace_name="ks1", base_table_name="table1")

        with ignore_upgrade_schema_errors():

            step = 'Step1 - Upgrade First Node '
            self.log.info(step)
            # upgrade first node
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[indexes[0]]
            self.log.info('Upgrade Node %s begin', self.db_cluster.node_to_upgrade.name)
            self.upgrade_node(self.db_cluster.node_to_upgrade)
            self.log.info('Upgrade Node %s ended', self.db_cluster.node_to_upgrade.name)
            self.db_cluster.node_to_upgrade.check_node_health()

            # wait for the prepare write workload to finish
            self.verify_stress_thread(prepare_write_cs_thread_pool)

            # read workload (cl=QUORUM)
            self.log.info('Starting c-s read workload (cl=QUORUM n=10000000)')
            stress_cmd_read_cl_quorum = self.params.get('stress_cmd_read_cl_quorum')
            read_stress_queue = self.run_stress_thread(stress_cmd=stress_cmd_read_cl_quorum)
            # wait for the read workload to finish
            self.verify_stress_thread(read_stress_queue)
            self.fill_and_verify_db_data('after upgraded one node')
            self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                          step=step+' - after upgraded one node')

            # read workload
            self.log.info('Starting c-s read workload for 10m')
            stress_cmd_read_10m = self.params.get('stress_cmd_read_10m')
            read_10m_cs_thread_pool = self.run_stress_thread(stress_cmd=stress_cmd_read_10m)

            self.log.info('Sleeping for 60s to let cassandra-stress start before the upgrade...')
            time.sleep(60)

            step = 'Step2 - Upgrade Second Node '
            self.log.info(step)
            # upgrade second node
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[indexes[1]]
            self.log.info('Upgrade Node %s begin', self.db_cluster.node_to_upgrade.name)
            self.upgrade_node(self.db_cluster.node_to_upgrade)
            self.log.info('Upgrade Node %s ended', self.db_cluster.node_to_upgrade.name)
            self.db_cluster.node_to_upgrade.check_node_health()

            # wait for the 10m read workload to finish
            self.verify_stress_thread(read_10m_cs_thread_pool)
            self.fill_and_verify_db_data('after upgraded two nodes')
            self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                          step=step+' - after upgraded two nodes')

            # read workload (60m)
            self.log.info('Starting c-s read workload for 60m')
            stress_cmd_read_60m = self.params.get('stress_cmd_read_60m')
            read_60m_cs_thread_pool = self.run_stress_thread(stress_cmd=stress_cmd_read_60m)
            self.log.info('Sleeping for 60s to let cassandra-stress start before the rollback...')
            time.sleep(60)

            self.log.info('Step3 - Rollback Second Node ')
            # rollback second node
            self.log.info('Rollback Node %s begin', self.db_cluster.nodes[indexes[1]].name)
            self.rollback_node(self.db_cluster.nodes[indexes[1]])
            self.log.info('Rollback Node %s ended', self.db_cluster.nodes[indexes[1]].name)
            self.db_cluster.nodes[indexes[1]].check_node_health()

        step = 'Step4 - Verify data during mixed cluster mode '
        self.log.info(step)
        self.fill_and_verify_db_data('after rollback the second node')
        self.log.info('Repair the first upgraded Node')
        self.db_cluster.nodes[indexes[0]].run_nodetool(sub_cmd='repair')
        self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                      step=step)

        with ignore_upgrade_schema_errors():

            step = 'Step5 - Upgrade rest of the Nodes '
            self.log.info(step)
            for i in indexes[1:]:
                self.db_cluster.node_to_upgrade = self.db_cluster.nodes[i]
                self.log.info('Upgrade Node %s begin', self.db_cluster.node_to_upgrade.name)
                self.upgrade_node(self.db_cluster.node_to_upgrade)
                self.log.info('Upgrade Node %s ended', self.db_cluster.node_to_upgrade.name)
                self.db_cluster.node_to_upgrade.check_node_health()
                self.fill_and_verify_db_data('after upgraded %s' % self.db_cluster.node_to_upgrade.name)
                self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                              step=step)

        self.log.info('Step6 - Verify stress results after upgrade ')
        self.log.info('Waiting for stress threads to complete after upgrade')
        # wait for the 60m read workload to finish
        self.verify_stress_thread(read_60m_cs_thread_pool)

        self.verify_stress_thread(entire_write_cs_thread_pool)

        self.log.info('Step7 - Upgrade sstables to latest supported version ')
        # figure out what is the last supported sstable version
        self.expected_sstable_format_version = self.get_highest_supported_sstable_version()

        # run 'nodetool upgradesstables' on all nodes and check/wait for all file to be upgraded
        upgradesstables = self.db_cluster.run_func_parallel(func=self.upgradesstables_if_command_available)

        # only check sstable format version if all nodes had 'nodetool upgradesstables' available
        if all(upgradesstables):
            self.log.info('Upgrading sstables if new version is available')
            tables_upgraded = self.db_cluster.run_func_parallel(func=self.wait_for_sstable_upgrade)
            assert all(tables_upgraded), "Failed to upgrade the sstable format {}".format(tables_upgraded)

        # Verify sstabledump
        self.log.info('Starting sstabledump to verify correctness of sstables')
        self.db_cluster.nodes[0].remoter.run(
            'for i in `sudo find /var/lib/scylla/data/keyspace_complex/ -type f |grep -v manifest.json |'
            'grep -v snapshots |head -n 1`; do echo $i; sudo sstabledump $i 1>/tmp/sstabledump.output || '
            'exit 1; done', verbose=True)

        self.log.info('Step8 - Run stress and verify after upgrading entire cluster ')
        self.log.info('Starting verify_stress_after_cluster_upgrade')
        verify_stress_after_cluster_upgrade = self.params.get(  # pylint: disable=invalid-name
            'verify_stress_after_cluster_upgrade')
        verify_stress_cs_thread_pool = self.run_stress_thread(stress_cmd=verify_stress_after_cluster_upgrade)
        self.verify_stress_thread(verify_stress_cs_thread_pool)

        # complex workload: verify data by simple read cl=ALL
        self.log.info('Starting c-s complex workload to verify data by simple read')
        stress_cmd_complex_verify_read = self.params.get('stress_cmd_complex_verify_read')
        complex_cs_thread_pool = self.run_stress_thread(
            stress_cmd=stress_cmd_complex_verify_read, profile='data_dir/complex_schema.yaml')
        # wait for the read complex workload to finish
        self.verify_stress_thread(complex_cs_thread_pool)

        self.log.info('Will check paged query after upgrading all nodes')
        self.paged_query()
        self.log.info('Done checking paged query after upgrading nodes')

        # After adjusted the workloads, there is a entire write workload, and it uses a fixed duration for catching
        # the data lose.
        # But the execute time of workloads are not exact, so let only use basic prepare write & read verify for
        # complex workloads,and comment two complex workloads.
        #
        # TODO: retest commented workloads and decide to enable or delete them.
        #
        # complex workload: verify data by multiple ops
        # self.log.info('Starting c-s complex workload to verify data by multiple ops')
        # stress_cmd_complex_verify_more = self.params.get('stress_cmd_complex_verify_more')
        # complex_cs_thread_pool = self.run_stress_thread(stress_cmd=stress_cmd_complex_verify_more,
        #                                                profile='data_dir/complex_schema.yaml')

        # wait for the complex workload to finish
        # self.verify_stress_thread(complex_cs_thread_pool)

        # complex workload: verify data by delete 1/10 data
        # self.log.info('Starting c-s complex workload to verify data by delete')
        # stress_cmd_complex_verify_delete = self.params.get('stress_cmd_complex_verify_delete')
        # complex_cs_thread_pool = self.run_stress_thread(stress_cmd=stress_cmd_complex_verify_delete,
        #                                                profile='data_dir/complex_schema.yaml')
        # wait for the complex workload to finish
        # self.verify_stress_thread(complex_cs_thread_pool)

        # During the test we filter and ignore some specific errors, but we want to allow only certain amount of them
        step = 'Step9 - Search for errors that we filter during the test '
        self.log.info(step)
        self.log.info('Checking how many failed_to_load_schem errors happened during the test')
        error_factor = 3
        schema_load_error_num = self.count_log_errors(search_pattern='Failed to load schema version',
                                                      step=step)
        # Warning example:
        # workload prioritization - update_service_levels_from_distributed_data: an error occurred while retrieving
        # configuration (exceptions::read_failure_exception (Operation failed for system_distributed.service_levels
        # - received 0 responses and 1 failures from 1 CL=ONE.))
        workload_prioritization_error_num = self.count_log_errors(search_pattern='workload prioritization.*read_failure_exception',
                                                                  step=step, search_for_idx_token_error=False)

        self.log.info(f'schema_load_error_num: {schema_load_error_num}; '
                      f'workload_prioritization_error_num: {workload_prioritization_error_num}')

        # Issue #https://github.com/scylladb/scylla-enterprise/issues/1391
        # By Eliran's comment: For 'Failed to load schema version' error which is expected and non offensive is
        # to count the 'workload prioritization' warning and subtract that amount from the amount of overall errors.
        load_error_num = schema_load_error_num - workload_prioritization_error_num
        assert load_error_num <= error_factor * 8 * \
            len(self.db_cluster.nodes), 'Only allowing shards_num * %d schema load errors per host during the ' \
                                        'entire test, actual: %d' % (
                error_factor, schema_load_error_num)

        self.log.info('Step10 - Verify that gemini and cdc stressor are not failed during upgrade')
        if self.version_cdc_support():
            self.verify_gemini_results(queue=gemini_thread)
            self.verify_cdclog_reader_results(cdc_reader_thread)

        self.log.info('all nodes were upgraded, and last workaround is verified.')

    def test_generic_cluster_upgrade(self):  # pylint: disable=too-many-locals,too-many-statements
        """
        Upgrade half of nodes in the cluster, and start special read workload
        during the stage. Checksum method is changed to xxhash from Scylla 2.2,
        we want to use this case to verify the read (cl=ALL) workload works
        well, upgrade all nodes to new version in the end.
        """
        # prepare workload (stress_before_upgrade)
        InfoEvent(message="Starting stress_before_upgrade - aka prepare step").publish()
        stress_before_upgrade = self.params.get('stress_before_upgrade')
        prepare_thread_pool = self.run_stress_thread(stress_cmd=stress_before_upgrade)
        if self.params.get('alternator_port'):
            self.pre_create_alternator_tables()
        InfoEvent(message="Waiting for stress_before_upgrade to finish").publish()
        self.verify_stress_thread(prepare_thread_pool)

        # Starting workload during entire upgrade
        InfoEvent(message="Starting stress_during_entire_upgrade workload").publish()
        stress_during_entire_upgrade = self.params.get('stress_during_entire_upgrade')
        stress_thread_pool = self.run_stress_thread(stress_cmd=stress_during_entire_upgrade)
        self.log.info('Sleeping for 60s to let cassandra-stress start before the rollback...')
        time.sleep(60)

        num_nodes_to_rollback = self.params.get('num_nodes_to_rollback')
        upgraded_nodes = []

        # generate random order to upgrade
        nodes_num = len(self.db_cluster.nodes)
        # prepare an array containing the indexes
        indexes = list(range(nodes_num))
        # shuffle it so we will upgrade the nodes in a random order
        random.shuffle(indexes)

        upgrade_sstables = self.params.get('upgrade_sstables')

        if num_nodes_to_rollback > 0:
            # Upgrade all nodes that should be rollback later
            for i in range(num_nodes_to_rollback):
                InfoEvent(message=f"Step{i + 1} - Upgrade node{i + 1}").publish()
                self.db_cluster.node_to_upgrade = self.db_cluster.nodes[indexes[i]]
                self.log.info(f'Upgrade Node {self.db_cluster.node_to_upgrade.name} begins')
                with ignore_ycsb_connection_refused():
                    self.upgrade_node(self.db_cluster.node_to_upgrade, upgrade_sstables=upgrade_sstables)
                self.log.info(f'Upgrade Node {self.db_cluster.node_to_upgrade.name} ended')
                self.db_cluster.node_to_upgrade.check_node_health()
                upgraded_nodes.append(self.db_cluster.node_to_upgrade)

            # Rollback all nodes that where upgraded (not necessarily in the same order)
            random.shuffle(upgraded_nodes)
            self.log.info(f'Upgraded Nodes to be rollback are: {upgraded_nodes}')
            for node in upgraded_nodes:
                InfoEvent(
                    message=f"Step{num_nodes_to_rollback + upgraded_nodes.index(node) + 1} - "
                            f"Rollback node{upgraded_nodes.index(node) + 1}"
                ).publish()
                self.log.info(f'Rollback Node {node} begin')
                with ignore_ycsb_connection_refused():
                    self.rollback_node(node, upgrade_sstables=upgrade_sstables)
                self.log.info(f'Rollback Node {node} ended')
                node.check_node_health()

        # Upgrade all nodes
        for i in range(nodes_num):
            InfoEvent(message=f"Step{num_nodes_to_rollback * 2 + i + 1} - Upgrade node{i + 1}").publish()
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[indexes[i]]
            self.log.info(f'Upgrade Node {self.db_cluster.node_to_upgrade.name} begins')
            with ignore_ycsb_connection_refused():
                self.upgrade_node(self.db_cluster.node_to_upgrade, upgrade_sstables=upgrade_sstables)
            self.log.info(f'Upgrade Node {self.db_cluster.node_to_upgrade.name} ended')
            self.db_cluster.node_to_upgrade.check_node_health()
            upgraded_nodes.append(self.db_cluster.node_to_upgrade)

        InfoEvent(message="All nodes were upgraded successfully").publish()

        InfoEvent(message="Waiting for stress_during_entire_upgrade to finish").publish()
        self.verify_stress_thread(stress_thread_pool)

        InfoEvent(message="Starting stress_after_cluster_upgrade").publish()
        stress_after_cluster_upgrade = self.params.get('stress_after_cluster_upgrade')
        stress_after_cluster_upgrade_pool = self.run_stress_thread(stress_cmd=stress_after_cluster_upgrade)
        self.verify_stress_thread(stress_after_cluster_upgrade_pool)

    def test_kubernetes_scylla_upgrade(self):
        """
        Run a set of different cql queries against various types/tables before
        and after upgrade of every node to check the consistency of data
        """
        self.truncate_entries_flag = False  # not perform truncate entries test
        self.log.info('Step1 - Populate DB with many types of tables and data')
        target_upgrade_version = self.params.get('new_version')
        if target_upgrade_version and parse_version(target_upgrade_version) >= parse_version('3.1') and \
                not is_enterprise(target_upgrade_version):
            self.truncate_entries_flag = True
        self.prepare_keyspaces_and_tables()
        self.log.info('Step2 - Populate some data before upgrading cluster')
        self.fill_and_verify_db_data('', pre_fill=True)
        self.log.info('Step3 - Starting c-s write workload')
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_w'))
            )
        )
        self.log.info('Step4 - Starting c-s read workload')
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_r'))
            )
        )
        self.log.info(f'Step5 - Upgrade cluster to {target_upgrade_version}')
        self.db_cluster.upgrade_scylla_cluster(target_upgrade_version)
        self.log.info('Step6 - Wait till cluster got upgraded')
        self.wait_till_scylla_is_upgraded_on_all_nodes(target_upgrade_version)

        self.log.info('Step7 - Upgrade sstables')
        if self.params.get('upgrade_sstables'):
            self.expected_sstable_format_version = self.get_highest_supported_sstable_version()
            upgradesstables = self.db_cluster.run_func_parallel(
                func=self.upgradesstables_if_command_available)
            # only check sstable format version if all nodes had 'nodetool upgradesstables' available
            if all(upgradesstables):
                self.log.info("Waiting until jmx is up across the board")
                self.wait_till_jmx_on_all_nodes()
                self.log.info('Upgrading sstables if new version is available')
                tables_upgraded = self.db_cluster.run_func_parallel(
                    func=self.wait_for_sstable_upgrade)
                assert all(tables_upgraded), f"Failed to upgrade the sstable format {tables_upgraded}"
        else:
            self.log.info("Upgrade of sstables is disabled")

        self.log.info('Step8 - Verify data after upgrade')
        self.fill_and_verify_db_data(note='after all nodes upgraded')
        self.log.info('Step9 - Starting c-s read workload')
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_r'))
            )
        )
        self.log.info('Step10 - Starting c-s write workload')
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_w'))
            )
        )
        self.log.info('Step11 - Starting c-s read workload')
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_r'))
            )
        )
        self.log.info('Step12 - Search for errors in scylla log')
        for node in self.db_cluster.nodes:
            self.search_for_idx_token_error_after_upgrade(node=node, step=f'{str(node)} after upgrade')
        self.log.info('Step13 - Checking how many failed_to_load_scheme errors happened during the test')
        error_factor = 3
        schema_load_error_num = self.count_log_errors(search_pattern='Failed to load schema version',
                                                      step='AFTER UPGRADE')
        # Warning example:
        # workload prioritization - update_service_levels_from_distributed_data: an error occurred while retrieving
        # configuration (exceptions::read_failure_exception (Operation failed for system_distributed.service_levels
        # - received 0 responses and 1 failures from 1 CL=ONE.))
        workload_prioritization_error_num = self.count_log_errors(
            search_pattern='workload prioritization.*read_failure_exception',
            step='AFTER UPGRADE',
            search_for_idx_token_error=False
        )
        self.log.info(f'schema_load_error_num: {schema_load_error_num}; '
                      f'workload_prioritization_error_num: {workload_prioritization_error_num}')

        # Issue #https://github.com/scylladb/scylla-enterprise/issues/1391
        # By Eliran's comment: For 'Failed to load schema version' error which is expected and non offensive is
        # to count the 'workload prioritization' warning and subtract that amount from the amount of overall errors.
        load_error_num = schema_load_error_num-workload_prioritization_error_num
        assert load_error_num <= error_factor * 8 * \
            len(self.db_cluster.nodes), 'Only allowing shards_num * %d schema load errors per host during the ' \
                                        'entire test, actual: %d' % (
                error_factor, schema_load_error_num)

    def wait_till_scylla_is_upgraded_on_all_nodes(self, target_version):
        def _is_cluster_upgraded():
            for node in self.db_cluster.nodes:
                if target_version != node.get_scylla_version() or not node.db_up:
                    return False
            return True
        wait.wait_for(func=_is_cluster_upgraded, step=30, timeout=900, throw_exc=True,
                      text="Waiting until all nodes in the cluster are upgraded")

    def wait_till_jmx_on_all_nodes(self):
        for node in self.db_cluster.nodes:
            node.wait_jmx_up(timeout=300)

    def count_log_errors(self, search_pattern, step, search_for_idx_token_error=True):
        schema_load_error_num = 0
        for node in self.db_cluster.nodes:
            errors = list(node.follow_system_log(patterns=[search_pattern], start_from_beginning=True))
            schema_load_error_num += len(errors)
            if search_for_idx_token_error:
                self.search_for_idx_token_error_after_upgrade(node=node, step=step)
        return schema_load_error_num

    def get_email_data(self):
        self.log.info('Prepare data for email')

        email_data = self._get_common_email_data()
        grafana_dataset = self.monitors.get_grafana_screenshot_and_snapshot(self.start_time) if self.monitors else {}
        email_data.update({
            "grafana_screenshots": grafana_dataset.get("screenshots", []),
            "grafana_snapshots": grafana_dataset.get("snapshots", []),
            "new_scylla_repo": self.params.get("new_scylla_repo"),
            "new_version": self.params.get("new_version"),
            "scylla_ami_id": self.params.get("ami_id_db_scylla") or "-",
        })

        return email_data
