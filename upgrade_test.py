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

# pylint: disable=too-many-lines
import traceback
from itertools import zip_longest, chain, count as itertools_count
import json
import random
import time
import re
from functools import wraps, cache
from typing import List
import contextlib

import cassandra
import tenacity
from argus.client.sct.types import Package
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement  # pylint: disable=no-name-in-module

from sdcm import wait
from sdcm.cluster import BaseNode
from sdcm.fill_db_data import FillDatabaseData
from sdcm.sct_events import Severity
from sdcm.stress_thread import CassandraStressThread
from sdcm.utils.common import ParallelObject
from sdcm.utils.decorators import retrying
from sdcm.utils.sstable.sstable_utils import get_sstable_data_dump_command
from sdcm.utils.user_profile import get_profile_content
from sdcm.utils.version_utils import (
    get_node_supported_sstable_versions,
    ComparableScyllaVersion,
    is_enterprise,
    get_node_enabled_sstable_version
)
from sdcm.sct_events.system import InfoEvent
from sdcm.sct_events.database import (
    IndexSpecialColumnErrorEvent,
    DatabaseLogEvent,
)
from sdcm.sct_events.filters import EventsSeverityChangerFilter
from sdcm.sct_events.group_common_events import (
    decorate_with_context,
    ignore_abort_requested_errors,
    ignore_topology_change_coordinator_errors,
    ignore_upgrade_schema_errors,
    ignore_ycsb_connection_refused,
)
from sdcm.utils import loader_utils
from sdcm.utils.features import CONSISTENT_TOPOLOGY_CHANGES_FEATURE
from sdcm.wait import wait_for
from sdcm.rest.raft_upgrade_procedure import RaftUpgradeProcedure
from test_lib.sla import create_sla_auth

NUMBER_OF_ROWS_FOR_TRUNCATE_TEST = 10


def truncate_entries(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        node = args[0]
        with self.db_cluster.cql_connection_patient(node, keyspace='truncate_ks', connect_timeout=600) as session:
            InfoEvent(message="Start truncate simple tables").publish()
            session.default_timeout = 60.0 * 5
            session.default_consistency_level = ConsistencyLevel.QUORUM
            try:
                self.cql_truncate_simple_tables(session=session, rows=NUMBER_OF_ROWS_FOR_TRUNCATE_TEST)
                InfoEvent(message="Finish truncate simple tables").publish()
            except cassandra.DriverException as details:
                InfoEvent(message=f"Failed truncate simple tables. Error: {str(details)}. Traceback: {traceback.format_exc()}",
                          severity=Severity.ERROR).publish()
            self.validate_truncated_entries_for_table(session=session, system_truncated=True)

        func_result = func(self, *args, **kwargs)

        # re-new connection
        with self.db_cluster.cql_connection_patient(node, keyspace='truncate_ks', connect_timeout=600) as session:
            session.default_timeout = 60.0 * 5
            session.default_consistency_level = ConsistencyLevel.QUORUM
            self.validate_truncated_entries_for_table(session=session, system_truncated=True)
            self.read_data_from_truncated_tables(session=session)
            self.cql_insert_data_to_simple_tables(session=session, rows=NUMBER_OF_ROWS_FOR_TRUNCATE_TEST)
        return func_result
    return inner


def check_reload_systemd_config(node):
    for i in ['scylla-server', 'scylla-jmx']:
        result = node.remoter.run('systemctl status %s' % i, ignore_status=True)
        if ".service changed on disk. Run 'systemctl daemon-reload' to reload units" in result.stderr:
            raise Exception("Systemd config is changed, but not reload automatically")


def backup_conf(node):
    if node.distro.is_rhel_like:
        node.remoter.run(
            r'for conf in $( rpm -qc $(rpm -qa | grep scylla) | grep -v contains ) '
            r'/etc/systemd/system/{var-lib-scylla,var-lib-systemd-coredump}.mount; '
            r'do if test -e $conf; then sudo cp -v $conf $conf.autobackup; fi; done')
    else:
        node.remoter.run(
            r'for conf in $(cat /var/lib/dpkg/info/scylla-*server.conffiles '
            r'/var/lib/dpkg/info/scylla-*conf.conffiles '
            r'/var/lib/dpkg/info/scylla-*jmx.conffiles | grep -v init ) '
            r'/etc/systemd/system/{var-lib-scylla,var-lib-systemd-coredump}.mount; '
            r'do if test -e $conf; then sudo cp -v $conf $conf.backup; fi; done')


def recover_conf(node):
    if node.distro.is_rhel_like:
        node.remoter.run(
            r'for conf in $( rpm -qc $(rpm -qa | grep scylla) | grep -v contains ) '
            r'/etc/systemd/system/{var-lib-scylla,var-lib-systemd-coredump}.mount; '
            r'do if test -e $conf.autobackup; then sudo cp -v $conf.autobackup $conf; fi; done')
    else:
        node.remoter.run(
            r'for conf in $(cat /var/lib/dpkg/info/scylla-*server.conffiles '
            r'/var/lib/dpkg/info/scylla-*conf.conffiles '
            r'/var/lib/dpkg/info/scylla-*jmx.conffiles | grep -v init ); do '
            r'if test -e $conf.backup; then sudo cp -v $conf.backup $conf; fi; done')


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class UpgradeTest(FillDatabaseData, loader_utils.LoaderUtilsMixin):
    """
    Test a Scylla cluster upgrade.
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.stack = contextlib.ExitStack()
        # ignoring those unsuppressed exceptions, till both ends of the upgrade would have https://github.com/scylladb/scylladb/pull/14681
        # see https://github.com/scylladb/scylladb/issues/14882 for details
        self.stack.enter_context(EventsSeverityChangerFilter(
            new_severity=Severity.WARNING,
            event_class=DatabaseLogEvent,
            regex=r".*std::runtime_error \(unknown exception\).*",
            extra_time_to_expiration=30,
        ))

    orig_ver = None
    new_ver = None

    # `major_release` (eg: 2.1 <-> 2.2, 2017.1 <-> 2018.1)
    # `reinstall` (opensource <-> enterprise, enterprise <-> opensource)
    # `minor_release` (eg: 2.2.1 <-> 2.2.5, 2018.1.0 <-> 2018.1.1)
    upgrade_rollback_mode = None

    # expected format version after upgrade and nodetool upgradesstables called
    # would be recalculated after all the cluster finish upgrade
    expected_sstable_format_version = 'mc'

    system_upgrade_timeout = 6 * 60

    @retrying(n=5)
    def _query_from_one_table(self, session, query, table_name) -> list:
        return self.rows_to_list(session.execute(SimpleStatement(query.format(table_name)), timeout=300))

    def read_data_from_truncated_tables(self, session):
        truncate_query = 'SELECT COUNT(*) FROM {}'
        tables_name = self.get_tables_name_of_keyspace(session=session, keyspace_name='truncate_ks')
        for table_name in tables_name:
            InfoEvent(message=f"Start read data from {table_name} tables").publish()
            try:
                count = self._query_from_one_table(session=session, query=truncate_query, table_name=table_name)
                self.assertEqual(str(count[0][0]), '0',
                                 msg='Expected that there is no data in the table truncate_ks.{}, but found {} rows'
                                 .format(table_name, count[0][0]))
                InfoEvent(message=f"Finish read data from {table_name} tables").publish()
            except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
                InfoEvent(message=f"Failed read data from {table_name} tables. Error: {str(details)}. Traceback: {traceback.format_exc()}",
                          severity=Severity.ERROR).publish()

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
        self._upgrade_node(node=node, upgrade_sstables=upgrade_sstables)

    @decorate_with_context(ignore_abort_requested_errors)
    # https://github.com/scylladb/scylla/issues/10447#issuecomment-1194155163
    def _upgrade_node(self, node, upgrade_sstables=True, new_scylla_repo=None, new_version=None):  # noqa: PLR0915
        # pylint: disable=too-many-branches,too-many-statements
        new_scylla_repo = new_scylla_repo or self.params.get('new_scylla_repo')
        new_version = new_version or self.params.get('new_version')
        upgrade_node_packages = self.params.get('upgrade_node_packages')

        scylla_yaml_updates = {}
        if not self.params.get('disable_raft'):
            scylla_yaml_updates.update({"consistent_cluster_management": True})
            scylla_yaml_updates.update({"force_schema_commit_log": True})

        if self.params.get("enable_tablets_on_upgrade"):
            scylla_yaml_updates.update({"enable_tablets": True})
        version = ComparableScyllaVersion(self.scylla_version)
        if self.is_enterprise and version >= "2025.1.0~dev" or not self.is_enterprise and version >= "6.3.0~dev":
            scylla_yaml_updates.update({"experimental_features": ["views-with-tablets"]})
        if self.params.get('test_sst3'):
            scylla_yaml_updates.update({"enable_sstables_mc_format": True})

        InfoEvent(message='Upgrading a Node').publish()
        # We assume that if update_db_packages is not empty we install packages from there.
        # In this case we don't use upgrade based on new_scylla_repo(ignored sudo yum update scylla...)
        result = node.remoter.run('scylla --version')
        self.orig_ver = result.stdout.strip()

        if upgrade_node_packages:
            InfoEvent(message='upgrade_node - started to "upgrade_node_packages').publish()
            # update_scylla_packages
            InfoEvent(message='upgrade_node - started sending upgrade packages to node').publish()
            node.remoter.send_files(upgrade_node_packages, '/tmp/scylla', verbose=True)
            InfoEvent(message='upgrade_node - ended sending upgrade packages to node').publish()
            # node.remoter.run('sudo yum update -y --skip-broken', connect_timeout=900)
            node.remoter.run('sudo yum install python34-PyYAML -y')
            # replace the packages
            node.remoter.run(r'rpm -qa scylla\*')
            # flush all memtables to SSTables
            node.run_nodetool("drain", timeout=15*60, coredump_on_timeout=True, long_running=True, retry=0)
            node.run_nodetool("snapshot")
            node.stop_scylla_server()
            # update *development* packages
            node.remoter.run('sudo rpm -UvhR --oldpackage /tmp/scylla/*development*', ignore_status=True)
            # and all the rest
            node.remoter.run('sudo rpm -URvh --replacefiles /tmp/scylla/*.rpm | true')
            node.remoter.run(r'rpm -qa scylla\*')
            InfoEvent(message='upgrade_node - ended to "upgrade_node_packages').publish()
        elif new_scylla_repo:
            InfoEvent(message='upgrade_node - started to create "new_scylla_repo"').publish()
            # backup the data
            node.remoter.run('sudo cp /etc/scylla/scylla.yaml /etc/scylla/scylla.yaml-backup')
            if node.distro.is_rhel_like:
                node.remoter.run('sudo cp /etc/yum.repos.d/scylla.repo ~/scylla.repo-backup')
            else:
                node.remoter.run('sudo cp /etc/apt/sources.list.d/scylla.list ~/scylla.list-backup')
            InfoEvent(message='upgrade_node - ended to create "new_scylla_repo"').publish()
            InfoEvent(message='upgrade_node - started to "backup_conf"').publish()
            backup_conf(node)
            InfoEvent(message='upgrade_node - ended to "backup_conf"').publish()
            assert new_scylla_repo.startswith('http')
            InfoEvent(message='upgrade_node - started to "download_scylla_repo"').publish()
            node.download_scylla_repo(new_scylla_repo)
            InfoEvent(message='upgrade_node - ended to "download_scylla_repo"').publish()
            # flush all memtables to SSTables
            InfoEvent(message='upgrade_node - started to "drain"').publish()
            node.run_nodetool("drain", timeout=15*60, coredump_on_timeout=True, long_running=True, retry=0)
            InfoEvent(message='upgrade_node - ended to "drain"').publish()
            InfoEvent(message='upgrade_node - started to "nodetool snapshot"').publish()
            node.run_nodetool("snapshot")
            InfoEvent(message='upgrade_node - ended to "nodetool snapshot"').publish()
            InfoEvent(message='upgrade_node - started to "stop_scylla_server"').publish()
            node.stop_scylla_server(verify_down=False)
            InfoEvent(message='upgrade_node - ended to "stop_scylla_server"').publish()

            orig_is_enterprise = node.is_enterprise
            if node.distro.is_rhel_like:
                result = node.remoter.run("sudo yum search scylla-enterprise 2>&1", ignore_status=True)
                new_is_enterprise = bool('scylla-enterprise.x86_64' in result.stdout or
                                         'No matches found' not in result.stdout)
            else:
                result = node.remoter.run("sudo apt-cache search scylla-enterprise", ignore_status=True)
                new_is_enterprise = 'scylla-enterprise' in result.stdout

            scylla_pkg = 'scylla-enterprise' if new_is_enterprise else 'scylla'
            ver_suffix = r'\*{}'.format(new_version) if new_version else ''
            scylla_pkg_ver = f"{scylla_pkg}{ver_suffix}"
            if orig_is_enterprise != new_is_enterprise:
                self.upgrade_rollback_mode = 'reinstall'
                if self.params.get('use_preinstalled_scylla'):
                    scylla_pkg_ver += f" {scylla_pkg}-machine-image{ver_suffix}"

            if self.upgrade_rollback_mode == 'reinstall':
                if node.distro.is_rhel_like:
                    InfoEvent(message='upgrade_node - starting to remove and install scylla on RHEL').publish()
                    node.remoter.run(r'sudo yum remove scylla\* -y')
                    node.remoter.run('sudo yum install {} -y'.format(scylla_pkg_ver))
                    InfoEvent(message='upgrade_node - ended to remove and install scylla on RHEL').publish()
                else:
                    InfoEvent(message='upgrade_node - starting to remove and install scylla on debian').publish()
                    node.remoter.sudo(r'apt-get remove scylla\* -y')
                    node.remoter.sudo(
                        f'DEBIAN_FRONTEND=noninteractive apt-get install {scylla_pkg_ver} -y'
                        f' -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"'
                    )
                    InfoEvent(message='upgrade_node - ended to remove and install scylla on debian').publish()
                InfoEvent(message='upgrade_node - starting to "recover_conf"').publish()
                recover_conf(node)
                InfoEvent(message='upgrade_node - ended to "recover_conf"').publish()
                InfoEvent(message='upgrade_node - starting to "daemon-reload"').publish()
                node.remoter.run('sudo systemctl daemon-reload')
                InfoEvent(message='upgrade_node - ended to "daemon-reload"').publish()
            else:  # noqa: PLR5501
                if node.distro.is_rhel_like:
                    InfoEvent(message='upgrade_node - starting to "yum update"').publish()
                    node.remoter.run(r'sudo yum update {}\* -y'.format(scylla_pkg_ver))
                    InfoEvent(message='upgrade_node - ended to "yum update"').publish()
                else:
                    InfoEvent(message='upgrade_node - starting to "apt-get update"').publish()
                    node.remoter.sudo('apt-get update')
                    node.remoter.sudo(
                        f'DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade {scylla_pkg} -y'
                        f' -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"'
                    )
                    InfoEvent(message='upgrade_node - ended to "apt-get update"').publish()

        InfoEvent(message='upgrade_node - fix /etc/scylla.d/io.conf arguments compatibility').publish()
        node.remoter.sudo(
            f"sed -i 's/--num-io-queues/--num-io-groups/' {node.add_install_prefix('/etc/scylla.d/io.conf')}")

        InfoEvent(message='upgrade_node - starting to "check_reload_systemd_config"').publish()
        check_reload_systemd_config(node)
        InfoEvent(message='upgrade_node - ended to "check_reload_systemd_config"').publish()
        # Current default 300s aren't enough for upgrade test of Debian 9.
        # Related issue: https://github.com/scylladb/scylla-cluster-tests/issues/1726
        InfoEvent(message='upgrade_node - starting to "run_scylla_sysconfig_setup"').publish()
        node.run_scylla_sysconfig_setup()
        InfoEvent(message='upgrade_node - ended to "run_scylla_sysconfig_setup"').publish()
        if scylla_yaml_updates:
            InfoEvent(message='upgrade_node - starting to "update_scylla_yaml"').publish()
            self._update_scylla_yaml_on_node(node_to_update=node, updates=scylla_yaml_updates)
            InfoEvent(message='upgrade_node - ended to "update_scylla_yaml"').publish()
        node.forget_scylla_version()
        node.drop_raft_property()
        InfoEvent(message='upgrade_node - starting to "start_scylla_server"').publish()
        node.start_scylla_server(verify_up_timeout=500)
        InfoEvent(message='upgrade_node - ended to "start_scylla_server"').publish()
        InfoEvent(message='upgrade_node - starting to "get_db_nodes_cpu_mode"').publish()
        self.db_cluster.get_db_nodes_cpu_mode()
        InfoEvent(message='upgrade_node - ended to "get_db_nodes_cpu_mode"').publish()
        result = node.remoter.run('scylla --version')
        new_ver = result.stdout.strip()
        assert self.orig_ver != new_ver, "scylla-server version isn't changed"
        self.new_ver = new_ver
        InfoEvent(message='upgrade_node - starting to "_update_argus_upgraded_version"').publish()
        self._update_argus_upgraded_version(node, new_ver)
        InfoEvent(message='upgrade_node - ended to "_update_argus_upgraded_version"').publish()
        if upgrade_sstables:
            InfoEvent(message='upgrade_node - starting to "upgradesstables_if_command_available"').publish()
            self.upgradesstables_if_command_available(node)
            InfoEvent(message='upgrade_node - ended to "upgradesstables_if_command_available"').publish()

        self.db_cluster.wait_all_nodes_un()

    def upgrade_os(self, nodes):
        def upgrade(node):
            InfoEvent(message=f'upgrade_node_system - starting to "upgrade_system" of the node {node.name}').publish()
            node.upgrade_system()
            InfoEvent(message=f'upgrade_node_system - ended to "upgrade_system" of the node {node.name}').publish()

        if self.params.get('upgrade_node_system'):
            InfoEvent(message='Upgrading OS on nodes').publish()
            parallel_obj = ParallelObject(objects=nodes, timeout=self.system_upgrade_timeout)
            parallel_obj.run(upgrade)

    @truncate_entries
    # https://github.com/scylladb/scylla/issues/10447#issuecomment-1194155163
    def rollback_node(self, node, upgrade_sstables=True):
        self._rollback_node(node=node, upgrade_sstables=upgrade_sstables)

    @decorate_with_context(ignore_abort_requested_errors)
    def _rollback_node(self, node, upgrade_sstables=True):
        # pylint: disable=too-many-branches,too-many-statements
        InfoEvent(message='Rollbacking a Node').publish()
        result = node.remoter.run('scylla --version')
        orig_ver = result.stdout.strip()
        # flush all memtables to SSTables
        node.run_nodetool("drain", timeout=15*60, coredump_on_timeout=True, long_running=True, retry=0)
        # backup the data
        node.run_nodetool("snapshot")
        node.stop_scylla_server(verify_down=False)

        if node.distro.is_rhel_like:
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

        if self.upgrade_rollback_mode == 'reinstall' or not node.distro.is_rhel_like:
            if node.distro.is_rhel_like:
                node.remoter.run(r'sudo yum remove scylla\* -y')
                node.remoter.run(r'sudo yum install %s -y' % node.scylla_pkg())
            else:
                node.remoter.sudo(r'apt-get remove scylla\* -y')
                node.remoter.sudo(
                    rf'DEBIAN_FRONTEND=noninteractive apt-get install {node.scylla_pkg()}\* -y'
                    rf' -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"'
                )
            recover_conf(node)
            node.remoter.run('sudo systemctl daemon-reload')

        elif self.upgrade_rollback_mode == 'minor_release':
            node.remoter.run(r'sudo yum downgrade scylla\*%s-\* -y' % self.orig_ver.split('-')[0])
        else:
            node.remoter.run(r'sudo yum downgrade scylla\* -y')
            recover_conf(node)
            node.remoter.run('sudo systemctl daemon-reload')

        result = node.remoter.run('scylla --version')
        new_ver = result.stdout.strip()

        node.remoter.run('sudo cp /etc/scylla/scylla.yaml-backup /etc/scylla/scylla.yaml')
        result = node.remoter.run('sudo find /var/lib/scylla/data/system')
        snapshot_name = re.findall(r"system/peers-[a-z0-9]+/snapshots/(\d+)\n", result.stdout)
        # cmd = (
        #     r"DIR='/var/lib/scylla/data/system'; "
        #     r"for i in `sudo ls $DIR`; do "
        #     r"    sudo test -e $DIR/$i/snapshots/%s && sudo find $DIR/$i/snapshots/%s -type f -exec sudo /bin/cp {} $DIR/$i/ \;; "
        #     r"done" % (snapshot_name[0], snapshot_name[0]))

        # recover the system tables - if it's downgrade from enterprise to OSS
        if (self.params.get('recover_system_tables') or
                (is_enterprise(orig_ver) and not is_enterprise(new_ver))):
            node.remoter.send_files('./data_dir/recover_system_tables.sh', '/tmp/')
            node.remoter.run('bash /tmp/recover_system_tables.sh %s' % snapshot_name[0], verbose=True)
        if self.params.get('test_sst3'):
            node.remoter.run(
                r'sudo sed -i -e "s/enable_sstables_mc_format:/#enable_sstables_mc_format:/g" /etc/scylla/scylla.yaml')
        if self.params.get('test_upgrade_from_installed_3_1_0'):
            node.remoter.run(
                r'sudo sed -i -e "s/enable_3_1_0_compatibility_mode:/#enable_3_1_0_compatibility_mode:/g" /etc/scylla/scylla.yaml')
        node.drop_raft_property()
        # Current default 300s aren't enough for upgrade test of Debian 9.
        # Related issue: https://github.com/scylladb/scylla-cluster-tests/issues/1726
        node.run_scylla_sysconfig_setup()
        node.start_scylla_server(verify_up_timeout=500)

        InfoEvent(message='original scylla-server version is %s, latest: %s' % (orig_ver, new_ver)).publish()
        assert orig_ver != new_ver, "scylla-server version isn't changed"

        if upgrade_sstables:
            self.upgradesstables_if_command_available(node)

        self.db_cluster.wait_all_nodes_un()

    @staticmethod
    def upgradesstables_if_command_available(node, queue=None):  # pylint: disable=invalid-name
        upgradesstables_available = False
        upgradesstables_supported = node.remoter.run(
            'nodetool help | grep -q upgradesstables && echo "yes" || echo "no"')
        if "yes" in upgradesstables_supported.stdout:
            upgradesstables_available = True
            InfoEvent(message="calling upgradesstables").publish()
            node.run_nodetool(sub_cmd="upgradesstables")
        if queue:
            queue.put(upgradesstables_available)
            queue.task_done()

    def get_highest_supported_sstable_version(self):  # pylint: disable=invalid-name
        """
        find the highest sstable format version supported in the cluster

        :return:
        """
        enabled_sstable_format_features = []
        for node in self.db_cluster.nodes:
            enabled_sstable_format_features.extend(get_node_supported_sstable_versions(node.system_log))

        if not enabled_sstable_format_features:
            for node in self.db_cluster.nodes:
                with self.db_cluster.cql_connection_patient_exclusive(node) as session:
                    enabled_sstable_format_features.extend(get_node_enabled_sstable_version(session))

        return max(set(enabled_sstable_format_features))

    def wait_for_sstable_upgrade(self, node, queue=None):
        all_tables_upgraded = True

        def wait_for_node_to_finish():
            try:
                result = node.remoter.sudo(
                    r"find /var/lib/scylla/data/system -type f ! -path '*snapshots*' -printf %f\\n")
                all_sstable_files = result.stdout.splitlines()

                sstable_version_regex = re.compile(r'(\w+)-[^-]+-(.+)\.(db|txt|sha1|crc32)')

                sstable_versions = {sstable_version_regex.search(f).group(
                    1) for f in all_sstable_files if sstable_version_regex.search(f)}

                assert len(sstable_versions) == 1, "expected all table format to be the same found {}".format(sstable_versions)
                assert list(sstable_versions)[0] == self.expected_sstable_format_version, (
                    "expected to format version to be '{}', found '{}'".format(
                        self.expected_sstable_format_version, list(sstable_versions)[0]))
            except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
                self.log.warning(ex)
                return False
            else:
                return True

        try:
            InfoEvent(message="Start waiting for upgardesstables to finish").publish()
            wait.wait_for(func=wait_for_node_to_finish, step=30, timeout=900, throw_exc=True,
                          text="Waiting until upgardesstables is finished")
        except tenacity.RetryError:
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
        self.upgrade_os(self.db_cluster.nodes)

        InfoEvent(message='Populate DB with many types of tables and data').publish()
        self.fill_db_data()
        InfoEvent(message='Run some Queries to verify data BEFORE UPGRADE').publish()
        self.verify_db_data()

        InfoEvent(message='Starting c-s write workload to pupulate 10M paritions').publish()
        # YAML: stress_cmd: cassandra-stress write cl=QUORUM n=10000000
        # -schema 'replication(strategy=NetworkTopologyStrategy,replication_factor=3)'
        # -mode cql3 native -rate threads=1000 -pop seq=1..10000000
        stress_cmd = self._cs_add_node_flag(self.params.get('stress_cmd'))
        self.run_stress_thread(stress_cmd=stress_cmd)

        InfoEvent(message='Sleeping for 360s to let cassandra-stress populate '
                          'some data before the mixed workload').publish()
        time.sleep(600)

        InfoEvent(message='Starting c-s read workload for 60m').publish()
        # YAML: stress_cmd_1: cassandra-stress read cl=QUORUM duration=60m
        # -schema 'replication(strategy=NetworkTopologyStrategy,replication_factor=3)'
        # -mode cql3 native -rate threads=100 -pop seq=1..10000000
        stress_cmd_1 = self._cs_add_node_flag(self.params.get('stress_cmd_1'))
        stress_queue = self.run_stress_thread(stress_cmd=stress_cmd_1)

        InfoEvent(message='Sleeping for 300s to let cassandra-stress start before the upgrade...').publish()
        time.sleep(300)

        nodes_num = len(self.db_cluster.nodes)
        # prepare an array containing the indexes
        indexes = list(range(nodes_num))
        # shuffle it so we will upgrade the nodes in a random order
        random.shuffle(indexes)

        # upgrade all the nodes in random order
        for i in indexes:
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[i]
            InfoEvent(message='Upgrade Node %s begin' % self.db_cluster.node_to_upgrade.name).publish()
            self.upgrade_node(self.db_cluster.node_to_upgrade)
            time.sleep(300)
            InfoEvent(message='Upgrade Node %s ended' % self.db_cluster.node_to_upgrade.name).publish()

        InfoEvent(message='Run some Queries to verify data AFTER UPGRADE').publish()
        self.verify_db_data()
        self.verify_stress_thread(stress_queue)

    def fill_and_verify_db_data(self, note, pre_fill=False, rewrite_data=True):
        if pre_fill:
            InfoEvent(message='Populate DB with many types of tables and data').publish()
            self.fill_db_data()
        InfoEvent(message='Run some Queries to verify data %s' % note).publish()
        self.verify_db_data()
        if rewrite_data:
            InfoEvent(message='Re-Populate DB with many types of tables and data').publish()
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

    @staticmethod
    def shuffle_nodes_and_alternate_dcs(nodes: List[BaseNode]) -> List[BaseNode]:
        """Shuffles provided nodes based on dc (alternates dc's).
        based on https://stackoverflow.com/a/21482016/3891194"""
        dc_nodes = {}
        for node in nodes:
            dc_nodes.setdefault(node.dc_idx, []).append(node)
        for dc in dc_nodes:  # pylint: disable=consider-using-dict-items
            random.shuffle(dc_nodes[dc])

        return [x for x in chain.from_iterable(zip_longest(*dc_nodes.values())) if x]

    @staticmethod
    def _update_scylla_yaml_on_node(node_to_update: BaseNode, updates: dict):
        with node_to_update.remote_scylla_yaml() as scylla_yaml:
            scylla_yaml.update(updates)

    def test_rolling_upgrade(self):  # pylint: disable=too-many-locals,too-many-statements,too-many-branches  # noqa: PLR0914, PLR0915
        """
        Upgrade half of nodes in the cluster, and start special read workload
        during the stage. Checksum method is changed to xxhash from Scylla 2.2,
        we want to use this case to verify the read (cl=ALL) workload works
        well, upgrade all nodes to new version in the end.
        """
        self.upgrade_os(self.db_cluster.nodes)

        InfoEvent(message='pre-test - prepare test keyspaces and tables').publish()
        # prepare test keyspaces and tables before upgrade to avoid schema change during mixed cluster.
        self.prepare_keyspaces_and_tables()
        InfoEvent(message='Running s-b to create schemas to avoid #11459').publish()
        large_partition_stress_during_upgrade = self.params.get('stress_before_upgrade')
        sb_create_schema = self.run_stress_thread(stress_cmd=f'{large_partition_stress_during_upgrade} -duration=1m')
        self.verify_stress_thread(sb_create_schema)
        self.fill_and_verify_db_data('BEFORE UPGRADE', pre_fill=True)

        # write workload during entire test
        InfoEvent(message='Starting c-s write workload during entire test').publish()
        write_stress_during_entire_test = self.params.get('write_stress_during_entire_test')
        entire_write_cs_thread_pool = self.run_stress_thread(stress_cmd=write_stress_during_entire_test)

        # Let to write_stress_during_entire_test complete the schema changes
        self.metric_has_data(
            metric_query='sct_cassandra_stress_write_gauge{type="ops", keyspace="keyspace_entire_test"}', n=10)

        # Prepare keyspace and tables for truncate test
        self.fill_db_data_for_truncate_test(insert_rows=NUMBER_OF_ROWS_FOR_TRUNCATE_TEST)

        # generate random order to upgrade
        nodes_num = len(self.db_cluster.nodes)
        # prepare an array containing the indexes
        indexes = list(range(nodes_num))
        # shuffle it so we will upgrade the nodes in a random order
        random.shuffle(indexes)

        InfoEvent(message='pre-test - Run stress workload before upgrade').publish()
        # complex workload: prepare write
        InfoEvent(message='Starting c-s complex workload (5M) to prepare data').publish()
        stress_cmd_complex_prepare = self.params.get('stress_cmd_complex_prepare')
        complex_cs_thread_pool = self.run_stress_thread(
            stress_cmd=stress_cmd_complex_prepare, profile='data_dir/complex_schema.yaml')

        # wait for the complex workload to finish
        self.verify_stress_thread(complex_cs_thread_pool)

        InfoEvent(message='Will check paged query before upgrading nodes').publish()
        self.paged_query()
        InfoEvent(message='Done checking paged query before upgrading nodes').publish()

        # prepare write workload
        InfoEvent(message='Starting c-s prepare write workload (n=10000000)').publish()
        prepare_write_stress = self.params.get('prepare_write_stress')
        prepare_write_cs_thread_pool = self.run_stress_thread(stress_cmd=prepare_write_stress)
        InfoEvent(message='Sleeping for 60s to let cassandra-stress start before the upgrade...').publish()
        self.metric_has_data(
            metric_query='sct_cassandra_stress_write_gauge{type="ops", keyspace="keyspace1"}', n=5)

        InfoEvent(message="Start gemini during upgrade").publish()
        gemini_cmd = self.params.get("gemini_cmd")
        if self.enable_cdc_for_tables:
            gemini_cmd += " --table-options \"cdc={'enabled': true}\""
        gemini_thread = self.run_gemini(gemini_cmd)
        self.metric_has_data(
            metric_query='gemini_cql_requests', n=10)

        with ignore_upgrade_schema_errors():

            step = 'Step1 - Upgrade First Node '
            InfoEvent(message=step).publish()
            # upgrade first node
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[indexes[0]]
            InfoEvent(message='Upgrade Node %s begin' % self.db_cluster.node_to_upgrade.name).publish()
            self.upgrade_node(self.db_cluster.node_to_upgrade)
            InfoEvent(message='Upgrade Node %s ended' % self.db_cluster.node_to_upgrade.name).publish()
            self.db_cluster.node_to_upgrade.check_node_health()

            # wait for the prepare write workload to finish
            self.verify_stress_thread(prepare_write_cs_thread_pool)

            # read workload (cl=QUORUM)
            InfoEvent(message='Starting c-s read workload (cl=QUORUM n=10000000)').publish()
            stress_cmd_read_cl_quorum = self.params.get('stress_cmd_read_cl_quorum')
            read_stress_queue = self.run_stress_thread(stress_cmd=stress_cmd_read_cl_quorum)
            # wait for the read workload to finish
            self.verify_stress_thread(read_stress_queue)
            InfoEvent(message='after upgraded one node').publish()
            self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                          step=step+' - after upgraded one node')

            # read workload
            InfoEvent(message='Starting c-s read workload for 10m').publish()
            stress_cmd_read_10m = self.params.get('stress_cmd_read_10m')
            read_10m_cs_thread_pool = self.run_stress_thread(stress_cmd=stress_cmd_read_10m)

            InfoEvent(message='Running s-b large partitions workload before and during upgrade').publish()
            large_partition_stress_during_upgrade = self.params.get('stress_before_upgrade')
            self.run_stress_thread(stress_cmd=large_partition_stress_during_upgrade)

            InfoEvent(message='Sleeping for 60s to let cassandra-stress and s-b start before the upgrade...').publish()
            time.sleep(60)

            step = 'Step2 - Upgrade Second Node '
            InfoEvent(message=step).publish()
            # upgrade second node
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[indexes[1]]
            InfoEvent(message='Upgrade Node %s begin' % self.db_cluster.node_to_upgrade.name).publish()
            self.upgrade_node(self.db_cluster.node_to_upgrade)
            InfoEvent(message='Upgrade Node %s ended' % self.db_cluster.node_to_upgrade.name).publish()
            self.db_cluster.node_to_upgrade.check_node_health()

            # wait for the 10m read workload to finish
            self.verify_stress_thread(read_10m_cs_thread_pool)
            self.fill_and_verify_db_data('after upgraded two nodes')
            self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                          step=step+' - after upgraded two nodes')

            # read workload (60m)
            InfoEvent(message='Starting c-s read workload for 60m').publish()
            stress_cmd_read_60m = self.params.get('stress_cmd_read_60m')
            read_60m_cs_thread_pool = self.run_stress_thread(stress_cmd=stress_cmd_read_60m)
            InfoEvent(message='Sleeping for 60s to let cassandra-stress start before the rollback...').publish()
            time.sleep(60)

            InfoEvent(message='Step3 - Rollback Second Node ').publish()
            # rollback second node
            InfoEvent(message='Rollback Node %s begin' % self.db_cluster.nodes[indexes[1]].name).publish()
            self.rollback_node(self.db_cluster.nodes[indexes[1]])
            InfoEvent(message='Rollback Node %s ended' % self.db_cluster.nodes[indexes[1]].name).publish()
            self.db_cluster.nodes[indexes[1]].check_node_health()

        step = 'Step4 - Verify data during mixed cluster mode '
        InfoEvent(message=step).publish()
        self.fill_and_verify_db_data('after rollback the second node')

        InfoEvent(message='Repair the first upgraded Node').publish()
        self.db_cluster.nodes[indexes[0]].run_nodetool(sub_cmd='repair', timeout=7200, coredump_on_timeout=True)
        self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade, step=step)

        with ignore_upgrade_schema_errors():

            step = 'Step5 - Upgrade rest of the Nodes '
            InfoEvent(message=step).publish()
            for i in indexes[1:]:
                self.db_cluster.node_to_upgrade = self.db_cluster.nodes[i]
                InfoEvent(message='Upgrade Node %s begin' % self.db_cluster.node_to_upgrade.name).publish()
                self.upgrade_node(self.db_cluster.node_to_upgrade)
                InfoEvent(message='Upgrade Node %s ended' % self.db_cluster.node_to_upgrade.name).publish()
                self.db_cluster.node_to_upgrade.check_node_health()
                self.fill_and_verify_db_data('after upgraded %s' % self.db_cluster.node_to_upgrade.name)
                self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                              step=step)
        InfoEvent(message='Step5.1 - run raft topology upgrade procedure').publish()
        self.run_raft_topology_upgrade_procedure()

        InfoEvent(message='Step6 - Verify stress results after upgrade ').publish()
        InfoEvent(message='Waiting for stress threads to complete after upgrade').publish()
        # wait for the 60m read workload to finish
        self.verify_stress_thread(read_60m_cs_thread_pool)

        self.verify_stress_thread(entire_write_cs_thread_pool)

        InfoEvent(message='Step7 - Upgrade sstables to latest supported version ').publish()
        # figure out what is the last supported sstable version
        self.expected_sstable_format_version = self.get_highest_supported_sstable_version()

        # run 'nodetool upgradesstables' on all nodes and check/wait for all file to be upgraded
        upgradesstables = self.db_cluster.run_func_parallel(func=self.upgradesstables_if_command_available)

        # only check sstable format version if all nodes had 'nodetool upgradesstables' available
        if all(upgradesstables):
            InfoEvent(message='Upgrading sstables if new version is available').publish()
            tables_upgraded = self.db_cluster.run_func_parallel(func=self.wait_for_sstable_upgrade)
            assert all(tables_upgraded), "Failed to upgrade the sstable format {}".format(tables_upgraded)

        # Verify sstabledump / scylla sstable dump-data
        InfoEvent(message='Starting sstabledump to verify correctness of sstables').publish()
        first_node = self.db_cluster.nodes[0]
        keyspace = "keyspace_complex"
        table = "user_with_ck"
        dump_cmd = get_sstable_data_dump_command(first_node, keyspace, table)
        first_node.remoter.run(
            f'for i in `sudo find /var/lib/scylla/data/{keyspace}/ -type f |grep -v manifest.json |'
            'grep -v snapshots |head -n 1`; do echo $i; '
            f'sudo {dump_cmd} $i 1>/tmp/sstabledump.output || '
            'exit 1; done', verbose=True)

        InfoEvent(message='Step8 - Run stress and verify after upgrading entire cluster').publish()
        InfoEvent(message='Starting verification stresses after cluster upgrade').publish()
        stress_after_cluster_upgrade = self.params.get('stress_after_cluster_upgrade')
        self.run_stress_thread(stress_cmd=stress_after_cluster_upgrade)
        verify_stress_after_cluster_upgrade = self.params.get(  # pylint: disable=invalid-name
            'verify_stress_after_cluster_upgrade')
        verify_stress_cs_thread_pool = self.run_stress_thread(stress_cmd=verify_stress_after_cluster_upgrade)
        self.verify_stress_thread(verify_stress_cs_thread_pool)

        # complex workload: verify data by simple read cl=ALL
        InfoEvent(message='Starting c-s complex workload to verify data by simple read').publish()
        stress_cmd_complex_verify_read = self.params.get('stress_cmd_complex_verify_read')
        complex_cs_thread_pool = self.run_stress_thread(
            stress_cmd=stress_cmd_complex_verify_read, profile='data_dir/complex_schema.yaml')
        # wait for the read complex workload to finish
        self.verify_stress_thread(complex_cs_thread_pool)

        InfoEvent(message='Will check paged query after upgrading all nodes').publish()
        self.paged_query()
        InfoEvent(message='Done checking paged query after upgrading nodes').publish()

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
        InfoEvent(message=step).publish()
        InfoEvent(message='Checking how many failed_to_load_schem errors happened during the test').publish()
        error_factor = 3
        schema_load_error_num = self.count_log_errors(search_pattern='Failed to load schema version',
                                                      step=step)
        # Warning example:
        # workload prioritization - update_service_levels_from_distributed_data: an error occurred while retrieving
        # configuration (exceptions::read_failure_exception (Operation failed for system_distributed.service_levels
        # - received 0 responses and 1 failures from 1 CL=ONE.))
        workload_prioritization_error_num = self.count_log_errors(search_pattern='workload prioritization.*read_failure_exception',
                                                                  step=step, search_for_idx_token_error=False)

        InfoEvent(message='schema_load_error_num: %s; workload_prioritization_error_num: %s' %
                          (schema_load_error_num, workload_prioritization_error_num)).publish()

        # Issue #https://github.com/scylladb/scylla-enterprise/issues/1391
        # By Eliran's comment: For 'Failed to load schema version' error which is expected and non offensive is
        # to count the 'workload prioritization' warning and subtract that amount from the amount of overall errors.
        load_error_num = schema_load_error_num - workload_prioritization_error_num
        assert load_error_num <= error_factor * 8 * \
            len(self.db_cluster.nodes), 'Only allowing shards_num * %d schema load errors per host during the ' \
                                        'entire test, actual: %d' % (
                error_factor, schema_load_error_num)

        InfoEvent(message='Step10 - Verify that gemini did not failed during upgrade').publish()
        self.verify_gemini_results(queue=gemini_thread)

        InfoEvent(message='all nodes were upgraded, and last workaround is verified.').publish()

    def run_raft_topology_upgrade_procedure(self):
        # wait features is enabled on nodes after upgrade
        feature_state_per_node = []
        for node in self.db_cluster.nodes:
            result = wait_for(func=self.db_cluster.is_features_enabled_on_node,
                              timeout=60,
                              step=f"Check feature enabled on node {node.name}",
                              throw_exc=False,
                              feature_list=[CONSISTENT_TOPOLOGY_CHANGES_FEATURE],
                              node=node)
            feature_state_per_node.append(result)
        if not all(feature_state_per_node):
            InfoEvent(message="Step5.1 - Consistent topology changes is not supported. Cluster stay with gossip topology mode").publish()
            return
        raft_upgrade = RaftUpgradeProcedure(self.db_cluster.nodes[0])
        raft_upgrade.start_upgrade_procedure()
        for node in self.db_cluster.nodes:
            RaftUpgradeProcedure(node).wait_upgrade_procedure_done()
        InfoEvent(message="Step5.1 - raft topology upgrade procedure done").publish()

    def _start_and_wait_for_node_upgrade(self, node: BaseNode, step: int) -> None:
        InfoEvent(
            message=f"Step {step} - Upgrade {node.name} from dc {node.dc_idx}").publish()
        InfoEvent(message='Upgrade Node %s begins' % node.name).publish()
        with ignore_ycsb_connection_refused(), ignore_topology_change_coordinator_errors():
            self.upgrade_node(node, upgrade_sstables=self.params.get('upgrade_sstables'))
        InfoEvent(message='Upgrade Node %s ended' % node.name).publish()
        node.check_node_health()

    def _start_and_wait_for_node_rollback(self, node: BaseNode, step: int) -> None:
        InfoEvent(
            message=f"Step {step} - "
                    f"Rollback {node.name} from dc {node.dc_idx}"
        ).publish()
        InfoEvent(message='Rollback Node %s begin' % node).publish()
        with ignore_ycsb_connection_refused():
            self.rollback_node(node, upgrade_sstables=self.params.get('upgrade_sstables'))
        InfoEvent(message='Rollback Node %s ended' % node).publish()
        node.check_node_health()

    def _run_stress_workload(self, workload_name: str, wait_for_finish: bool = False) -> [CassandraStressThread]:
        """Runs workload from param name specified in test-case yaml"""
        InfoEvent(message=f"Starting {workload_name}").publish()
        stress_commands = self.params.get(workload_name)
        workload_thread_pools = []
        if isinstance(stress_commands, str):
            stress_commands = [stress_commands]

        for stress_command in stress_commands:
            workload_thread_pools.append(self.run_stress_thread(stress_cmd=stress_command))

        if self.params.get('alternator_port'):
            self.pre_create_alternator_tables()
        if wait_for_finish is True:
            InfoEvent(message=f"Waiting for {workload_name} to finish").publish()
            for thread_pool in workload_thread_pools:
                self.verify_stress_thread(thread_pool)
        else:
            InfoEvent(message='Sleeping for 60s to let the stress command(s) start before the next steps...').publish()
            time.sleep(60)
        return workload_thread_pools

    def _add_sla_credentials_to_stress_commands(self, workloads_with_sla: list):
        if not workloads_with_sla:
            self.log.debug("No workloads supplied")
            return

        stress_during_entire_upgrade = self.params.get(workloads_with_sla[0])
        if not [cmd for cmd in stress_during_entire_upgrade if "<sla credentials " in cmd]:
            self.log.debug("No need to set SLA credentials for stress command: %s", stress_during_entire_upgrade)
            return

        roles = []
        service_level_shares = self.params.get("service_level_shares")
        # OSS Scylla version does not support "shares" option for Service Level
        if service_level_shares and not self.db_cluster.nodes[0].is_enterprise:
            service_level_shares = [None for _ in service_level_shares]

        with self.db_cluster.cql_connection_patient(node=self.db_cluster.nodes[0], user=loader_utils.DEFAULT_USER,
                                                    password=loader_utils.DEFAULT_USER_PASSWORD) as session:
            for index, shares in enumerate(service_level_shares):
                roles.append(create_sla_auth(session=session, shares=shares, index=str(index)))

        self.add_sla_credentials_to_stress_cmds(workload_names=workloads_with_sla, roles=roles,
                                                params=self.params, parent_class_name=self.__class__.__name__)

    def test_generic_cluster_upgrade(self):
        """
        Upgrade specified number of nodes in the cluster, and start special read workload
        during the stage. Checksum method is changed to xxhash from Scylla 2.2,
        we want to use this case to verify the read (cl=ALL) workload works
        well, upgrade all nodes to new version in the end.

        For multi-dc upgrades, alternates upgraded nodes between dc's.
        """
        self.upgrade_os(self.db_cluster.nodes)

        # Prepare keyspace and tables for truncate test
        self.fill_db_data_for_truncate_test(insert_rows=NUMBER_OF_ROWS_FOR_TRUNCATE_TEST)

        self._add_sla_credentials_to_stress_commands(workloads_with_sla=['stress_during_entire_upgrade',
                                                                         'stress_after_cluster_upgrade'])
        step = itertools_count(start=1)
        self._run_stress_workload("stress_before_upgrade", wait_for_finish=True)
        stress_thread_pools = self._run_stress_workload("stress_during_entire_upgrade", wait_for_finish=False)

        # generate random order to upgrade
        nodes_to_upgrade = self.shuffle_nodes_and_alternate_dcs(list(self.db_cluster.nodes))

        # Upgrade all nodes that should be rollback later
        upgraded_nodes = []
        for node_to_upgrade in nodes_to_upgrade[:self.params.get('num_nodes_to_rollback')]:
            self._start_and_wait_for_node_upgrade(node_to_upgrade, step=next(step))
            upgraded_nodes.append(node_to_upgrade)

        # Rollback all nodes that where upgraded (not necessarily in the same order)
        random.shuffle(upgraded_nodes)
        InfoEvent(message='Upgraded Nodes to be rollback are: %s' %
                  ", ".join(node.name for node in upgraded_nodes)).publish()
        for node in upgraded_nodes:
            self._start_and_wait_for_node_rollback(node, step=next(step))

        # Upgrade all nodes
        for node_to_upgrade in nodes_to_upgrade:
            self._start_and_wait_for_node_upgrade(node_to_upgrade, step=next(step))

        InfoEvent(message="All nodes were upgraded successfully").publish()

        InfoEvent(message='Run raft topology upgrade procedure').publish()
        self.run_raft_topology_upgrade_procedure()

        InfoEvent(message="Waiting for stress_during_entire_upgrade to finish").publish()
        for stress_thread_pool in stress_thread_pools:
            self.verify_stress_thread(stress_thread_pool)

        self._run_stress_workload("stress_after_cluster_upgrade", wait_for_finish=True)

    def test_cluster_upgrade_latency_regression(self):
        """Check latency regression after a ScyllaDB cluster upgrade using latte stress commands.

        Number of 'before' and 'after' commands must match. Their latency values will be compaired.

        - Write initial latte data (prepare_write_cmd?)
        - Wait for end of compactions
        - Read latte data generating report file (stress_before_upgrade)
        - Run a read latte stress (stress_during_entire_upgrade) not waiting for it's end
        - Upgrade the DB cluster
        * self.run_raft_topology_upgrade_procedure()
        - Wait for the end of the stress command (stress_during_entire_upgrade)
        - Wait for end of compactions
        - Read latte data (stress_after_cluster_upgrade) generating report file
        - Compare latte report files and raise SCT ERROR event if latencies are worse for more than 10%
        """
        self.upgrade_os(self.db_cluster.nodes)

        InfoEvent(message="Step1 - Populate DB data").publish()
        # Prepare keyspace and tables for truncate test
        self.fill_db_data_for_truncate_test(insert_rows=NUMBER_OF_ROWS_FOR_TRUNCATE_TEST)
        self.run_prepare_write_cmd()

        InfoEvent(message="Step2 - Run 'read' command before upgrade").publish()
        step = itertools_count(start=1)
        stress_before_upgrade_thread_pools = self._run_stress_workload(
            "stress_before_upgrade", wait_for_finish=False)
        stress_before_upgrade_results = []
        for stress_before_upgrade_thread_pool in stress_before_upgrade_thread_pools:
            stress_before_upgrade_results.append(self.get_stress_results(stress_before_upgrade_thread_pool))
        stress_during_entire_upgrade_thread_pools = self._run_stress_workload(
            "stress_during_entire_upgrade", wait_for_finish=False)

        InfoEvent(message="Step3 - Upgrade cluster to '%s' version" % self.params.get('new_version')).publish()

        InfoEvent(message="Upgrade part of nodes nodes before roll-back").publish()
        nodes_to_upgrade = self.shuffle_nodes_and_alternate_dcs(list(self.db_cluster.nodes))
        upgraded_nodes = []
        for node_to_upgrade in nodes_to_upgrade[:self.params.get('num_nodes_to_rollback')]:
            self._start_and_wait_for_node_upgrade(node_to_upgrade, step=next(step))
            upgraded_nodes.append(node_to_upgrade)

        # Rollback all nodes that where upgraded (not necessarily in the same order)
        random.shuffle(upgraded_nodes)
        InfoEvent(message="Roll-back following nodes: %s" % ", ".join(
            node.name for node in upgraded_nodes)).publish()
        for node in upgraded_nodes:
            self._start_and_wait_for_node_rollback(node, step=next(step))

        InfoEvent(message="Upgrade all nodes").publish()
        for node_to_upgrade in nodes_to_upgrade:
            self._start_and_wait_for_node_upgrade(node_to_upgrade, step=next(step))
        InfoEvent(message="All nodes were upgraded successfully").publish()

        InfoEvent(message="Step4 - Run raft topology upgrade procedure").publish()
        self.run_raft_topology_upgrade_procedure()

        InfoEvent(message="Step5 - Wait for stress_during_entire_upgrade to finish").publish()
        for stress_during_entire_upgrade_thread_pool in stress_during_entire_upgrade_thread_pools:
            self.verify_stress_thread(stress_during_entire_upgrade_thread_pool)
        self.wait_no_compactions_running(n=240, sleep_time=30)

        InfoEvent(message="Step6 - run 'stress_after_cluster_upgrade' stress command(s)").publish()
        time.sleep(60)
        stress_after_upgrade_thread_pools = self._run_stress_workload(
            "stress_after_cluster_upgrade", wait_for_finish=False)
        stress_after_upgrade_results = []
        for stress_after_upgrade_thread_pool in stress_after_upgrade_thread_pools:
            stress_after_upgrade_results.append(self.get_stress_results(stress_after_upgrade_thread_pool))

        self.log.info(
            "Going to compare following READ stress results:\nbefore upgrade: %s\nafter upgrade: %s",
            stress_before_upgrade_results, stress_after_upgrade_results)
        assert len(stress_before_upgrade_results) > 0
        for stress_before_upgrade_result in stress_before_upgrade_results:
            assert len(stress_before_upgrade_result) > 0
        assert len(stress_before_upgrade_results) == len(stress_after_upgrade_results)
        for stress_after_upgrade_result in stress_after_upgrade_results:
            assert len(stress_after_upgrade_result) > 0
        for i in range(len(stress_before_upgrade_results)):
            for j in range(len(stress_before_upgrade_results[i])):
                assert 'latency 99th percentile' in stress_before_upgrade_results[i][j]
                current_latency_before = float(stress_before_upgrade_results[i][j]['latency 99th percentile'])
                assert current_latency_before > 0
                assert 'latency 99th percentile' in stress_after_upgrade_results[i][j]
                current_latency_after = float(stress_after_upgrade_results[i][j]['latency 99th percentile'])
                assert current_latency_after > 0
                assert current_latency_after / current_latency_before < 1.2

    def test_kubernetes_scylla_upgrade(self):
        """
        Run a set of different cql queries against various types/tables before
        and after upgrade of every node to check the consistency of data
        """
        self.k8s_cluster.check_scylla_cluster_sa_annotations()
        InfoEvent(message='Step1 - Populate DB with many types of tables and data').publish()
        target_upgrade_version = self.params.get('new_version')
        self.prepare_keyspaces_and_tables()
        InfoEvent(message='Step2 - Populate some data before upgrading cluster').publish()
        self.fill_and_verify_db_data('', pre_fill=True)
        InfoEvent(message='Step3 - Starting c-s write workload').publish()
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_w'))
            )
        )
        InfoEvent(message='Step4 - Starting c-s read workload').publish()
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_r'))
            )
        )
        InfoEvent(message='Step5 - Upgrade cluster to %s' % target_upgrade_version).publish()
        self.db_cluster.upgrade_scylla_cluster(target_upgrade_version)
        InfoEvent(message='Step6 - Wait till cluster got upgraded').publish()
        self.wait_till_scylla_is_upgraded_on_all_nodes(target_upgrade_version)
        self.k8s_cluster.check_scylla_cluster_sa_annotations()

        InfoEvent(message='Step7 - Upgrade sstables').publish()
        if self.params.get('upgrade_sstables'):
            self.expected_sstable_format_version = self.get_highest_supported_sstable_version()
            upgradesstables = self.db_cluster.run_func_parallel(
                func=self.upgradesstables_if_command_available)
            # only check sstable format version if all nodes had 'nodetool upgradesstables' available
            if all(upgradesstables):
                InfoEvent(message="Waiting until jmx is up across the board").publish()
                self.wait_till_jmx_on_all_nodes()
                InfoEvent(message='Upgrading sstables if new version is available').publish()
                tables_upgraded = self.db_cluster.run_func_parallel(
                    func=self.wait_for_sstable_upgrade)
                assert all(tables_upgraded), f"Failed to upgrade the sstable format {tables_upgraded}"
        else:
            InfoEvent(message="Upgrade of sstables is disabled").publish()

        InfoEvent(message='Step8 - Verify data after upgrade').publish()
        self.fill_and_verify_db_data(note='after all nodes upgraded')
        InfoEvent(message='Step9 - Starting c-s read workload').publish()
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_r'))
            )
        )
        InfoEvent(message='Step10 - Starting c-s write workload').publish()
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_w'))
            )
        )
        InfoEvent(message='Step11 - Starting c-s read workload').publish()
        self.verify_stress_thread(
            self.run_stress_thread(
                stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_r'))
            )
        )
        InfoEvent(message='Step12 - Search for errors in scylla log').publish()
        for node in self.db_cluster.nodes:
            self.search_for_idx_token_error_after_upgrade(node=node, step=f'{str(node)} after upgrade')
        InfoEvent(message='Step13 - Checking how many failed_to_load_scheme errors happened during the test').publish()
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
        InfoEvent(message='schema_load_error_num: %s; workload_prioritization_error_num: %s' %
                          (schema_load_error_num, workload_prioritization_error_num)).publish()

        # Issue #https://github.com/scylladb/scylla-enterprise/issues/1391
        # By Eliran's comment: For 'Failed to load schema version' error which is expected and non offensive is
        # to count the 'workload prioritization' warning and subtract that amount from the amount of overall errors.
        load_error_num = schema_load_error_num-workload_prioritization_error_num
        assert load_error_num <= error_factor * 8 * \
            len(self.db_cluster.nodes), 'Only allowing shards_num * %d schema load errors per host during the ' \
                                        'entire test, actual: %d' % (
                error_factor, schema_load_error_num)

    def _get_current_operator_image_tag(self):
        return self.k8s_cluster.kubectl(
            "get deployment scylla-operator -o custom-columns=:..image --no-headers",
            namespace=self.k8s_cluster._scylla_operator_namespace  # pylint: disable=protected-access
        ).stdout.strip().split(":")[-1]

    def test_kubernetes_operator_upgrade(self):  # pylint: disable=too-many-locals,too-many-statements
        self.k8s_cluster.check_scylla_cluster_sa_annotations()

        InfoEvent(message='Step1 - Populate DB with data').publish()
        self.prepare_keyspaces_and_tables()
        self.fill_and_verify_db_data('', pre_fill=True)

        InfoEvent(message='Step2 - Run c-s write workload').publish()
        self.verify_stress_thread(self.run_stress_thread(
            stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_w'))))

        InfoEvent(message='Step3 - Run c-s read workload').publish()
        self.verify_stress_thread(self.run_stress_thread(
            stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_r'))))

        InfoEvent(message='Step4 - Upgrade scylla-operator').publish()
        old_scylla_pods_uids = {}
        for scylla_pod in self.db_cluster.nodes:
            old_scylla_pods_uids[scylla_pod.name] = scylla_pod.k8s_pod_uid
        progressing_last_transition_time_before_upgrade = self.k8s_cluster.kubectl(
            f"get --no-headers scyllacluster {self.params.get('k8s_scylla_cluster_name')}"
            " -o jsonpath='{.status.conditions[?(@.type==\"Progressing\")].lastTransitionTime}'",
            namespace=self.db_cluster.namespace).stdout.strip()
        base_docker_image_tag = self._get_current_operator_image_tag()
        upgrade_docker_image = self.params.get('k8s_scylla_operator_upgrade_docker_image') or ''
        self.k8s_cluster.upgrade_scylla_operator(
            self.params.get('k8s_scylla_operator_upgrade_helm_repo') or self.params.get(
                'k8s_scylla_operator_helm_repo'),
            self.params.get('k8s_scylla_operator_upgrade_chart_version') or 'latest',
            upgrade_docker_image)

        InfoEvent(message='Step5 - Validate scylla-operator version after upgrade').publish()
        actual_docker_image_tag = self._get_current_operator_image_tag()
        self.assertNotEqual(base_docker_image_tag, actual_docker_image_tag)
        expected_docker_image_tag = upgrade_docker_image.split(':')[-1]
        if not expected_docker_image_tag:
            operator_chart_info = self.k8s_cluster.helm(
                f"ls -n {self.k8s_cluster._scylla_operator_namespace} -o json")  # pylint: disable=protected-access
            expected_docker_image_tag = json.loads(operator_chart_info)[0]["app_version"]
        self.assertEqual(expected_docker_image_tag, actual_docker_image_tag)

        InfoEvent(message='Step6 - Wait for the update of Scylla cluster').publish()
        for scylla_pod in self.db_cluster.nodes[::-1]:
            scylla_pod.wait_till_k8s_pod_get_uid(
                ignore_uid=old_scylla_pods_uids[scylla_pod.name], throw_exc=True, timeout=900)
        for scylla_pod in self.db_cluster.nodes[::-1]:
            scylla_pod.wait_for_pod_readiness()

        for condition in ("Available=True", "Progressing=False", "Degraded=False"):
            self.k8s_cluster.kubectl_wait(
                f"scyllacluster {self.params.get('k8s_scylla_cluster_name')} --for=condition={condition}",
                namespace=self.db_cluster.namespace, timeout=600)
        scylla_cluster_conditions = json.loads(self.k8s_cluster.kubectl(
            f"get scyllacluster {self.params.get('k8s_scylla_cluster_name')} -o json",
            namespace=self.db_cluster.namespace).stdout.strip())["status"]["conditions"]
        for condition in scylla_cluster_conditions:
            if condition["type"] == "Progressing":
                assert condition["lastTransitionTime"] > progressing_last_transition_time_before_upgrade
            assert condition["reason"] != "Error", "'ScyllaCluster' failed to be updated: %s" % condition

        pods_with_wrong_image_tags = []
        pods = json.loads(self.k8s_cluster.kubectl(
            "get pods -o json", namespace=self.db_cluster.namespace).stdout.strip())["items"]
        for pod in pods:
            # NOTE: in pods we have 'containers' and 'initContainers'. So, walk over the both types
            #       and check that the 'scylla-operator' images are updated.
            for container_type in ("c", "initC"):
                for container in pod.get("status", {}).get(f"{container_type}ontainerStatuses", []):
                    image = container["image"].split("/")[-1]
                    image_name, image_tag = image.split(":")
                    if image_name == "scylla-operator" and image_tag != expected_docker_image_tag:
                        pods_with_wrong_image_tags.append({
                            "namespace": self.db_cluster.namespace,
                            "pod_name": pod["metadata"]["name"],
                            "container_name": container["name"],
                            "image": image,
                        })
        assert not pods_with_wrong_image_tags, (
            f"Found pods that have unexpected scylla-operator image tags.\n"
            f"Expected is '{expected_docker_image_tag}'.\n"
            f"Pods: {json.dumps(pods_with_wrong_image_tags, indent=2)}")

        self.k8s_cluster.check_scylla_cluster_sa_annotations()

        InfoEvent(message='Step7 - Add new member to the Scylla cluster').publish()
        peer_db_node = self.db_cluster.nodes[0]
        new_nodes = self.db_cluster.add_nodes(
            count=1,
            dc_idx=peer_db_node.dc_idx,
            rack=peer_db_node.rack,
            enable_auto_bootstrap=True)
        self.db_cluster.wait_for_init(node_list=new_nodes, timeout=40 * 60)
        self.db_cluster.wait_sts_rollout_restart(pods_to_wait=1)
        self.db_cluster.wait_for_nodes_up_and_normal(nodes=new_nodes)
        self.monitors.reconfigure_scylla_monitoring()
        self.k8s_cluster.check_scylla_cluster_sa_annotations()

        InfoEvent(message='Step8 - Verify data in the Scylla cluster').publish()
        self.fill_and_verify_db_data(note='after operator upgrade and scylla member addition')

        InfoEvent(message='Step9 - Run c-s read workload').publish()
        self.verify_stress_thread(self.run_stress_thread(
            stress_cmd=self._cs_add_node_flag(self.params.get('stress_cmd_r'))))

    def test_kubernetes_platform_upgrade(self):
        self.k8s_cluster.check_scylla_cluster_sa_annotations()

        InfoEvent(message="Step1 - Populate DB with data").publish()
        self.run_prepare_write_cmd()
        self.prepare_keyspaces_and_tables()
        self.fill_and_verify_db_data('', pre_fill=True)

        InfoEvent(message="Step2 - Start 'stress_during_entire_upgrade'").publish()
        stress_during_entire_upgrade_thread_pools = self._run_stress_workload(
            "stress_during_entire_upgrade", wait_for_finish=False)

        InfoEvent(message="Step3 - Upgrade kubernetes platform").publish()
        upgrade_version, scylla_pool = self.k8s_cluster.upgrade_kubernetes_platform(
            pod_objects=self.db_cluster.nodes,
            use_additional_scylla_nodepool=True)

        InfoEvent(message="Step4 - Validate K8S versions after the upgrade").publish()
        data_plane_versions = self.k8s_cluster.kubectl(
            f"get nodes --no-headers -l '{self.k8s_cluster.POOL_LABEL_NAME} "
            f"in ({self.k8s_cluster.AUXILIARY_POOL_NAME},{scylla_pool.name})' "
            "-o custom-columns=:.status.nodeInfo.kubeletVersion").stdout.strip().split()
        # Output example:
        #   GKE                EKS
        #   v1.19.13-gke.700   v1.21.2-eks-c1718fb
        #   v1.19.13-gke.700   v1.21.2-eks-c1718fb
        assert all(ver.startswith(f"v{upgrade_version}.") for ver in data_plane_versions), (
            f"Got unexpected K8S data plane version(s): {data_plane_versions}")

        control_plane_versions = self.k8s_cluster.kubectl("version --short").stdout.splitlines()
        # Output example:
        # Client Version: v1.20.4
        # Server Version: v1.19.13-gke.700
        # WARNING: foo
        assert any(ver.startswith(f"Server Version: v{upgrade_version}.")
                   for ver in control_plane_versions), (
            f"Got unexpected K8S control plane version: {control_plane_versions}")

        InfoEvent(message="Step5 - Check Scylla-operator pods").publish()
        self.k8s_cluster.kubectl_wait(
            "--all --for=condition=Ready pod",
            namespace=self.k8s_cluster._scylla_operator_namespace,  # pylint: disable=protected-access
            timeout=600)

        InfoEvent(message="Step6 - Check Scylla pods").publish()
        self.k8s_cluster.kubectl_wait(
            "--all --for=condition=Ready pod",
            namespace=self.db_cluster.namespace,
            timeout=600)
        self.k8s_cluster.check_scylla_cluster_sa_annotations()

        InfoEvent(message="Step7 - Add new member to the Scylla cluster").publish()
        peer_db_node = self.db_cluster.nodes[0]
        new_nodes = self.db_cluster.add_nodes(
            count=1,
            dc_idx=peer_db_node.dc_idx,
            rack=peer_db_node.rack,
            enable_auto_bootstrap=True)
        self.db_cluster.wait_for_init(node_list=new_nodes, timeout=40 * 60)
        self.db_cluster.wait_sts_rollout_restart(pods_to_wait=1)
        self.db_cluster.wait_for_nodes_up_and_normal(nodes=new_nodes)
        self.monitors.reconfigure_scylla_monitoring()
        self.k8s_cluster.check_scylla_cluster_sa_annotations()

        InfoEvent(message="Step8 - Wait for the end of 'stress_during_entire_upgrade'").publish()
        for stress_during_entire_upgrade_thread_pool in stress_during_entire_upgrade_thread_pools:
            self.verify_stress_thread(stress_during_entire_upgrade_thread_pool)

        InfoEvent(message="Step9 - Run 'fill_and_verify_db_data'").publish()
        self.fill_and_verify_db_data(note='after operator upgrade and scylla member addition')

        InfoEvent(message="Step10 - Run 'stress_cmd_r' to check all the preloaded data").publish()
        read_stress_threads = self._run_all_stress_cmds([], params={
            'stress_cmd': self.params.get('stress_cmd_r'),
            'round_robin': self.params.get('round_robin'),
        })
        for read_stress_thread in read_stress_threads:
            self.verify_stress_thread(read_stress_thread)

    def wait_till_scylla_is_upgraded_on_all_nodes(self, target_version: str) -> None:
        def _is_cluster_upgraded() -> bool:
            for node in self.db_cluster.nodes:
                node.forget_scylla_version()
                if node.scylla_version.replace("~", "-") != target_version.replace("~", "-") or not node.db_up:
                    return False
            return True
        wait.wait_for(
            func=_is_cluster_upgraded,
            step=30,
            text="Waiting until all nodes in the cluster are upgraded",
            timeout=900,
            throw_exc=True,
        )

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
        email_data.update({
            "new_scylla_repo": self.params.get("new_scylla_repo"),
            "new_version": self.new_ver,
            "scylla_ami_id": self.params.get("ami_id_db_scylla") or "-",
        })

        return email_data

    @cache
    def _update_argus_upgraded_version(self, node, new_version):
        try:
            ver = re.search(r"(?P<version>[\w.~]+)-0\.?(?P<date>[0-9]{8,8})\.(?P<commit_id>\w+)", new_version)
            if ver:
                package = Package(name="scylla-server-upgraded", date=ver.group("date"),
                                  version=ver.group("version"),
                                  revision_id=ver.group("commit_id"),
                                  build_id=node.get_scylla_build_id() or "#NO_BUILDID")
                self.log.info("Saving upgraded Scylla version...")
                self.test_config.argus_client().update_scylla_version(version=ver.group("version"))
                self.test_config.argus_client().submit_packages([package])
            else:
                self.log.warning("Couldn't extract version from %s", new_version)
        except Exception as exc:  # pylint: disable=broad-except
            self.log.exception("Failed to save upgraded Scylla version in Argus", exc_info=exc)


class UpgradeCustomTest(UpgradeTest):
    def test_custom_profile_rolling_upgrade(self):
        """
        Run a load of a custom profile.
        Upgrade half of nodes in the cluster, and run rollback.
        Upgrade all nodes to new version in the end.
        """
        cs_user_profiles = self.prepare_data_before_upgrade()
        self._custom_profile_rolling_upgrade(cs_user_profiles=cs_user_profiles)

    def test_custom_profile_sequential_rolling_upgrade(self):
        """
        Run a load of a custom profile.
        Perform sequential rolling upgrade with upgrade path:
        step 1. 2021.1 -> 2022.1
        step 2. 2022.1 -> 2023.1
        In each step: upgrade half of nodes in the cluster, and run rollback.
        Upgrade all nodes to new version in the end.
        """
        cs_user_profiles = self.prepare_data_before_upgrade()
        # Temporary solution. If we want to have sequential upgrade test, we will need to change the test parameters and build upgrade path
        new_version = "2022.1"
        new_scylla_repo = \
            "https://s3.amazonaws.com/downloads.scylladb.com/deb/ubuntu/scylla-2022.1.list" \
            "scylla.list"
        InfoEvent(message=f'Starting rolling upgrade test to {new_version}').publish()
        self._custom_profile_rolling_upgrade(cs_user_profiles=cs_user_profiles,
                                             new_version=new_version, new_scylla_repo=new_scylla_repo)
        InfoEvent(message=f'Finished rolling upgrade test to {new_version}').publish()

        new_version = "2023.1"
        new_scylla_repo = \
            "https://s3.amazonaws.com/downloads.scylladb.com/deb/ubuntu/scylla-2023.1.list"
        InfoEvent(message=f'Starting rolling upgrade test to {new_version}').publish()
        self._custom_profile_rolling_upgrade(cs_user_profiles=cs_user_profiles,
                                             new_version=new_version, new_scylla_repo=new_scylla_repo)
        InfoEvent(message=f'Finished rolling upgrade test to {new_version}').publish()

    def prepare_data_before_upgrade(self):
        InfoEvent(message='Running a prepare load for the initial custom data').publish()
        if not (prepare_cs_user_profiles := self.params.get('prepare_cs_user_profiles')):
            SyntaxError("Parameter 'prepare_cs_user_profiles' is not supplied")

        user_profiles, duration_per_cs_profile = self.parse_cs_user_profiles_param(prepare_cs_user_profiles)
        stress_before_upgrade = self.run_cs_user_profiles(cs_profiles=user_profiles,
                                                          duration_per_cs_profile=duration_per_cs_profile)
        for queue in stress_before_upgrade:
            self.verify_stress_thread(cs_thread_pool=queue)

        # write workload during entire test
        if not (cs_user_profiles := self.params.get('cs_user_profiles')):
            SyntaxError("Parameter 'cs_user_profiles' is not supplied")

        return cs_user_profiles

    def _custom_profile_rolling_upgrade(self, cs_user_profiles, new_scylla_repo=None, new_version=None):  # pylint: disable=too-many-locals,too-many-statements
        self.upgrade_os(self.db_cluster.nodes)

        InfoEvent(message='Starting write workload during entire test').publish()
        user_profiles, duration_per_cs_profile = self.parse_cs_user_profiles_param(cs_user_profiles)
        entire_write_thread_pool = self.run_cs_user_profiles(cs_profiles=user_profiles,
                                                             duration_per_cs_profile=duration_per_cs_profile)

        # Let wait for start writing data
        for stress_cmd in user_profiles:
            _, profile = get_profile_content(stress_cmd)
            keyspace_name = profile.get('keyspace')
            self.log.debug("keyspace_name: %s", keyspace_name)
            if not keyspace_name:
                continue

            try:
                self.metric_has_data(
                    metric_query='sct_cassandra_stress_user_gauge{type="ops", keyspace="%s"}' % keyspace_name, n=10)
            except Exception as err:  # pylint: disable=broad-except  # noqa: BLE001
                InfoEvent(
                    f"Get metrix data for keyspace {keyspace_name} failed with error: {err}", severity=Severity.ERROR).publish()

        # generate random order to upgrade
        nodes_num = len(self.db_cluster.nodes)
        # prepare an array containing the indexes
        indexes = list(range(nodes_num))
        # shuffle it so we will upgrade the nodes in a random order
        random.shuffle(indexes)

        with ignore_upgrade_schema_errors():

            step = 'Step1 - Upgrade First Node '
            InfoEvent(message=step).publish()
            # upgrade first node
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[indexes[0]]
            InfoEvent(message='Upgrade Node %s begin' % self.db_cluster.node_to_upgrade.name).publish()
            # Call "_upgrade_node" to prevent running truncate test
            self._upgrade_node(self.db_cluster.node_to_upgrade,
                               new_scylla_repo=new_scylla_repo, new_version=new_version)
            InfoEvent(message='Upgrade Node %s ended' % self.db_cluster.node_to_upgrade.name).publish()
            self.db_cluster.node_to_upgrade.check_node_health()

            InfoEvent(message='after upgraded one node').publish()
            self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                          step=step+' - after upgraded one node')

            step = 'Step2 - Upgrade Second Node '
            InfoEvent(message=step).publish()
            # upgrade second node
            self.db_cluster.node_to_upgrade = self.db_cluster.nodes[indexes[1]]
            InfoEvent(message='Upgrade Node %s begin' % self.db_cluster.node_to_upgrade.name).publish()
            # Call "_upgrade_node" to prevent running truncate test
            self._upgrade_node(self.db_cluster.node_to_upgrade,
                               new_scylla_repo=new_scylla_repo, new_version=new_version)
            InfoEvent(message='Upgrade Node %s ended' % self.db_cluster.node_to_upgrade.name).publish()
            self.db_cluster.node_to_upgrade.check_node_health()

            self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                          step=step+' - after upgraded two nodes')

            InfoEvent(message='Step3 - Rollback Second Node ').publish()
            # rollback second node
            InfoEvent(message='Rollback Node %s begin' % self.db_cluster.nodes[indexes[1]].name).publish()
            # Call "_rollback_node" to prevent running truncate test
            self._rollback_node(self.db_cluster.nodes[indexes[1]])
            InfoEvent(message='Rollback Node %s ended' % self.db_cluster.nodes[indexes[1]].name).publish()
            self.db_cluster.nodes[indexes[1]].check_node_health()

        step = 'Step4 - Verify data during mixed cluster mode '
        InfoEvent(message=step).publish()
        InfoEvent(message='Repair the first upgraded Node').publish()
        self.db_cluster.nodes[indexes[0]].run_nodetool(sub_cmd='repair')
        self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                      step=step)

        with ignore_upgrade_schema_errors():

            step = 'Step5 - Upgrade rest of the Nodes '
            InfoEvent(message=step).publish()
            for i in indexes[1:]:
                self.db_cluster.node_to_upgrade = self.db_cluster.nodes[i]
                InfoEvent(message='Upgrade Node %s begin' % self.db_cluster.node_to_upgrade.name).publish()
                # Call "_upgrade_node" to prevent running truncate test
                self._upgrade_node(self.db_cluster.node_to_upgrade,
                                   new_scylla_repo=new_scylla_repo, new_version=new_version)
                InfoEvent(message='Upgrade Node %s ended' % self.db_cluster.node_to_upgrade.name).publish()
                self.db_cluster.node_to_upgrade.check_node_health()
                self.search_for_idx_token_error_after_upgrade(node=self.db_cluster.node_to_upgrade,
                                                              step=step)

        InfoEvent(message='Step6 - Verify stress results after upgrade ').publish()
        InfoEvent(message='Waiting for stress threads to complete after upgrade').publish()

        for queue in entire_write_thread_pool:
            self.verify_stress_thread(cs_thread_pool=queue)

        InfoEvent(message='Step7 - Upgrade sstables to latest supported version ').publish()
        # figure out what is the last supported sstable version
        self.expected_sstable_format_version = self.get_highest_supported_sstable_version()

        # run 'nodetool upgradesstables' on all nodes and check/wait for all file to be upgraded
        upgradesstables = self.db_cluster.run_func_parallel(func=self.upgradesstables_if_command_available)

        # only check sstable format version if all nodes had 'nodetool upgradesstables' available
        if all(upgradesstables):
            InfoEvent(message='Upgrading sstables if new version is available').publish()
            tables_upgraded = self.db_cluster.run_func_parallel(func=self.wait_for_sstable_upgrade)
            assert all(tables_upgraded), "Failed to upgrade the sstable format {}".format(tables_upgraded)

        InfoEvent(message='all nodes were upgraded, and last workaround is verified.').publish()
