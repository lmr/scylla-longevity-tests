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

# pylint: disable=too-few-public-methods

import json
import time
import shutil
import logging
import os.path
import tempfile
import unittest
from datetime import datetime
from functools import cached_property
from typing import List
from weakref import proxy as weakproxy

import pytest
from invoke import Result

from sdcm.cluster import BaseNode, BaseCluster, BaseMonitorSet, BaseScyllaCluster
from sdcm.db_log_reader import DbLogReader
from sdcm.sct_events import Severity
from sdcm.sct_events.database import SYSTEM_ERROR_EVENTS_PATTERNS
from sdcm.sct_events.group_common_events import ignore_upgrade_schema_errors
from sdcm.sct_events.filters import DbEventsFilter
from sdcm.sct_events.system import InstanceStatusEvent
from sdcm.utils.common import get_keyspace_partition_ranges, keyspace_min_max_tokens
from sdcm.utils.distro import Distro

from unit_tests.dummy_remote import DummyRemote
from unit_tests.lib.events_utils import EventsUtilsMixin
from unit_tests.test_utils_common import DummyNode


class DummyDbCluster(BaseCluster, BaseScyllaCluster):  # pylint: disable=abstract-method
    # pylint: disable=super-init-not-called
    def __init__(self, nodes, params=None):
        self.nodes = nodes
        self.params = params
        self.added_password_suffix = False
        self.log = logging.getLogger(__name__)


class DummyDbLogReader(DbLogReader):
    def get_scylla_debuginfo_file(self):
        return "scylla_debug_info_file"


class TestBaseNode(unittest.TestCase, EventsUtilsMixin):
    @classmethod
    def setUpClass(cls):
        cls.setup_events_processes(events_device=True, events_main_device=False, registry_patcher=True)

    @cached_property
    def node(self):
        dummy_node = DummyNode(
            name='test_node',
            parent_cluster=None,
            base_logdir=self.temp_dir,
            ssh_login_info=dict(key_file='~/.ssh/scylla-test'),
        )
        dummy_node.init()
        dummy_node.remoter = DummyRemote()
        return dummy_node

    @cached_property
    def _db_log_reader(self):
        return DummyDbLogReader(
            system_log=self.node.system_log,
            remoter=self.node.remoter,
            node_name=str(self),
            system_event_patterns=SYSTEM_ERROR_EVENTS_PATTERNS,
            decoding_queue=None,
            log_lines=False
        )

    def _read_and_publish_events(self, log_text=None):
        if log_text:
            with tempfile.NamedTemporaryFile(mode='wt') as temp_log:
                self.node.system_log = temp_log.name

                for line in log_text.splitlines(keepends=True):
                    temp_log.write(line)
                    temp_log.flush()
                    self._db_log_reader._read_and_publish_events()  # pylint: disable=protected-access
        else:
            self._db_log_reader._read_and_publish_events()  # pylint: disable=protected-access

    @classmethod
    def tearDownClass(cls):
        cls.teardown_events_processes()

    def setUp(self):
        self.node.system_log = os.path.join(os.path.dirname(__file__), 'test_data', 'system.log')

    def test_search_system_log(self):
        critical_errors = list(self.node.follow_system_log(start_from_beginning=True))
        self.assertEqual(34, len(critical_errors))

    def test_search_system_log_specific_log(self):
        errors = list(self.node.follow_system_log(
            patterns=['Failed to load schema version'], start_from_beginning=True))
        self.assertEqual(len(errors), 2)

    def test_search_system_interlace_reactor_stall(self):
        self.node.system_log = os.path.join(os.path.dirname(__file__), 'test_data', 'system_interlace_stall.log')

        self._read_and_publish_events()

        with self.get_raw_events_log().open() as events_file:
            events = [json.loads(line) for line in events_file]

            event_a, event_b = events[-2], events[-1]
            print(event_a)
            print(event_b)

            assert event_a["type"] == "REACTOR_STALLED"
            assert event_a["line_number"] == 0
            assert event_b["type"] == "REACTOR_STALLED"
            assert event_b["line_number"] == 3

    def test_search_kernel_callstack(self):
        self.node.parent_cluster = {'params': {'print_kernel_callstack': True}}
        self.node.system_log = os.path.join(os.path.dirname(__file__), 'test_data', 'kernel_callstack.log')
        self._read_and_publish_events()
        with self.get_raw_events_log().open() as events_file:
            events = [json.loads(line) for line in events_file]

            event_a, event_b = events[-2], events[-1]
            print(event_a)
            print(event_b)

            assert event_a["type"] == "KERNEL_CALLSTACK"
            assert event_a["line_number"] == 2
            assert event_b["type"] == "KERNEL_CALLSTACK"
            assert event_b["line_number"] == 5

    def test_search_cdc_invalid_request(self):
        self.node.system_log = os.path.join(os.path.dirname(__file__), 'test_data', 'system_cdc_invalid_request.log')
        with ignore_upgrade_schema_errors():
            self._read_and_publish_events()

        time.sleep(0.1)
        with self.get_events_logger().events_logs_by_severity[Severity.ERROR].open() as events_file:
            cdc_err_events = [line for line in events_file if 'cdc - Could not retrieve CDC streams' in line]
            assert cdc_err_events != []

    def test_search_power_off(self):
        self.node.system_log = os.path.join(os.path.dirname(__file__), 'test_data', 'power_off.log')
        with DbEventsFilter(db_event=InstanceStatusEvent.POWER_OFF, node=self.node):
            self._read_and_publish_events()

        InstanceStatusEvent.POWER_OFF().add_info(
            node="A", line_number=22,
            line=f"{datetime.utcfromtimestamp(time.time() + 1):%Y-%m-%dT%H:%M:%S+00:00} "
                 "longevity-large-collections-12h-mas-db-node-c6a4e04e-1 !INFO    | systemd-logind: Powering Off..."
        ).publish()

        time.sleep(0.1)
        with self.get_events_logger().events_logs_by_severity[Severity.WARNING].open() as events_file:
            events = [line for line in events_file if 'Powering Off' in line]
            assert events

    def test_search_system_suppressed_messages(self):
        self.node.system_log = os.path.join(os.path.dirname(
            __file__), 'test_data', 'system_suppressed_messages.log')

        self._read_and_publish_events()

        with self.get_raw_events_log().open() as events_file:
            events = [json.loads(line) for line in events_file]

            event_a = events[-1]
            print(event_a)

            assert event_a["type"] == "SUPPRESSED_MESSAGES", 'Not expected event type {}'.format(event_a["type"])
            assert event_a["line_number"] == 6, 'Not expected event line number {}'.format(event_a["line_number"])

    def test_search_one_line_backtraces(self):
        self.node.system_log = os.path.join(os.path.dirname(__file__), 'test_data', 'system_one_line_backtrace.log')

        self._read_and_publish_events()

        with self.get_raw_events_log().open() as events_file:
            events = [json.loads(line) for line in events_file]

            event_backtrace1, event_backtrace2 = events[-3], events[-2]
            print(event_backtrace1)
            print(event_backtrace2)

            assert event_backtrace1["type"] == "BACKTRACE"
            assert event_backtrace1["raw_backtrace"]
            assert event_backtrace2["type"] == "BACKTRACE"
            assert event_backtrace2["raw_backtrace"]

    def test_gate_closed_ignored_exception_is_catched(self):
        self.node.system_log = os.path.join(os.path.dirname(__file__), 'test_data', 'gate_closed_ignored_exception.log')

        self._read_and_publish_events()

        with self.get_raw_events_log().open() as events_file:
            events = [json.loads(line) for line in events_file]

            event_backtrace1, event_backtrace2 = events[-2], events[-1]
            print(event_backtrace1)
            print(event_backtrace2)

            assert event_backtrace1["type"] == "GATE_CLOSED"
            assert event_backtrace1["line_number"] == 1
            assert event_backtrace2["type"] == "GATE_CLOSED"
            assert event_backtrace2["line_number"] == 3

    def test_compaction_stopped_exception_is_catched(self):
        self.node.system_log = os.path.join(os.path.dirname(__file__), 'test_data', 'compaction_stopped_exception.log')

        self._read_and_publish_events()

        with self.get_raw_events_log().open() as events_file:
            events = [json.loads(line) for line in events_file]

            event_backtrace1, event_backtrace2 = events[-3], events[-2]
            print(event_backtrace1)
            print(event_backtrace2)

            assert event_backtrace1["type"] == "COMPACTION_STOPPED"
            assert event_backtrace1["line_number"] == 0
            assert event_backtrace2["type"] == "COMPACTION_STOPPED"
            assert event_backtrace2["line_number"] == 1

    def test_appending_to_log(self):
        logs = """
INFO  2022-07-14 09:28:34,095 [shard 1] database - Flushing non-system tables
Reactor stalled for 32 ms on shard 1. Backtrace: 0x4e0d6e2 0x4e0c340 0x4e0d5f0 0x7f230af20a1f 0x14de339 0x14dc3b0 0x14d680c 0x14d5ff6 0x14e4d52 0x14d0238 0x1aec3c6 0x508c7e1
kernel callstack:
INFO  2022-07-14 09:28:35,102 [shard 1] database - Flushed non-system tables
        """

        self._read_and_publish_events(logs)

        with self.get_raw_events_log().open() as events_file:
            events = [json.loads(line) for line in events_file]
            reactor_stalls = [event for event in events if event["type"] == "REACTOR_STALLED"]
            assert len(reactor_stalls) == 1
            event = reactor_stalls[0]
            assert event["type"] == "REACTOR_STALLED"
            assert event["line_number"] == 2
            assert 'Reactor stalled for 32 ms on shard 1' in event['line']


class VersionDummyRemote:
    def __init__(self, test, results):
        self.test = weakproxy(test)
        self.results = iter(results)

    def run(self, cmd, *_, **__):
        expected_cmd, result = next(self.results)
        self.test.assertEqual(cmd, expected_cmd)
        return Result(exited=result[0], stdout=result[1], stderr=result[2])


class TestBaseNodeGetScyllaVersion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        self.node = DummyNode(name='test_node',
                              parent_cluster=None,
                              base_logdir=self.temp_dir,
                              ssh_login_info=dict(key_file='~/.ssh/scylla-test'))

    def test_no_scylla_binary_rhel_like(self):
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (127, "", "bash: scylla: command not found\n")),
            ("rpm --query --queryformat '%{VERSION}' scylla", (0, "3.3.rc1", "")),
        ))
        self.assertEqual("3.3.rc1", self.node.scylla_version)
        self.assertEqual("3.3.rc1", self.node.scylla_version_detailed)

    def test_no_scylla_binary_other(self):
        self.node.distro = Distro.DEBIAN9
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (127, "", "bash: scylla: command not found\n")),
            ("dpkg-query --show --showformat '${Version}' scylla", (0, "3.3~rc1-0.20200209.0d0c1d43188-1", "")),
        ))
        self.assertEqual("3.3.rc1", self.node.scylla_version)
        self.assertEqual("3.3.rc1-0.20200209.0d0c1d43188-1", self.node.scylla_version_detailed)

    def test_scylla(self):
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "3.3.rc1-0.20200209.0d0c1d43188\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))
        self.assertEqual("3.3.rc1", self.node.scylla_version)
        self.assertEqual("3.3.rc1-0.20200209.0d0c1d43188 with build-id xxx", self.node.scylla_version_detailed)

    def test_scylla_master(self):
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "666.development-0.20200205.2816404f575\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))
        self.assertEqual("666.development", self.node.scylla_version)
        self.assertEqual("666.development-0.20200205.2816404f575 with build-id xxx", self.node.scylla_version_detailed)

    def test_scylla_master_new_format(self):
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "4.4.dev-0.20200205.2816404f575\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))
        self.assertEqual("4.4.dev", self.node.scylla_version)
        self.assertEqual("4.4.dev-0.20200205.2816404f575 with build-id xxx", self.node.scylla_version_detailed)

    def test_scylla_enterprise(self):
        self.node.is_enterprise = True
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "2019.1.4-0.20191217.b59e92dbd\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))
        self.assertEqual("2019.1.4", self.node.scylla_version)
        self.assertEqual("2019.1.4-0.20191217.b59e92dbd with build-id xxx", self.node.scylla_version_detailed)

    def test_scylla_enterprise_no_scylla_binary(self):
        self.node.is_enterprise = True
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (127, "", "bash: scylla: command not found\n")),
            ("rpm --query --queryformat '%{VERSION}' scylla-enterprise", (0, "2019.1.4", "")),
        ))
        self.assertEqual("2019.1.4", self.node.scylla_version)
        self.assertEqual("2019.1.4", self.node.scylla_version_detailed)

    def test_scylla_binary_version_unparseable(self):
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "x.y.z\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))
        self.assertIsNone(self.node.scylla_version)
        self.assertEqual("x.y.z with build-id xxx", self.node.scylla_version_detailed)

    def test_get_scylla_version_from_second_attempt(self):
        self.node.distro = Distro.DEBIAN9
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (127, "", "bash: scylla: command not found\n")),
            ("dpkg-query --show --showformat '${Version}' scylla",
             (1, "", "dpkg-query: no packages found matching scylla\n")),
            ("/usr/bin/scylla --version", (127, "", "bash: scylla: command not found\n")),
            ("dpkg-query --show --showformat '${Version}' scylla",
             (1, "", "dpkg-query: no packages found matching scylla\n")),
        ))
        self.assertIsNone(self.node.scylla_version)
        self.assertIsNone(self.node.scylla_version_detailed)

        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "4.4.dev-0.20200205.2816404f575\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))
        self.assertEqual("4.4.dev", self.node.scylla_version)
        self.assertEqual("4.4.dev-0.20200205.2816404f575 with build-id xxx", self.node.scylla_version_detailed)

    def test_forget_scylla_version(self):
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "4.4.dev-0.20200205.2816404f575\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))
        self.assertEqual("4.4.dev", self.node.scylla_version)
        self.assertEqual("4.4.dev-0.20200205.2816404f575 with build-id xxx", self.node.scylla_version_detailed)

        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "3.3.rc1-0.20200209.0d0c1d43188\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))

        self.assertEqual("4.4.dev", self.node.scylla_version)
        self.assertEqual("4.4.dev-0.20200205.2816404f575 with build-id xxx", self.node.scylla_version_detailed)

        self.node.forget_scylla_version()

        self.assertEqual("3.3.rc1", self.node.scylla_version)
        self.assertEqual("3.3.rc1-0.20200209.0d0c1d43188 with build-id xxx", self.node.scylla_version_detailed)


class TestBaseMonitorSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        self.node = DummyNode(name='test_node',
                              parent_cluster=None,
                              base_logdir=self.temp_dir,
                              ssh_login_info=dict(key_file='~/.ssh/scylla-test'))
        self.db_cluster = DummyDbCluster([self.node])
        self.monitor_cluster = BaseMonitorSet({"db_cluster": self.db_cluster}, {})
        self.monitor_cluster.log = logging

    def test_monitoring_version(self):
        """
        verify that dev version are mapped to monitor master version
        """
        self.node.remoter = VersionDummyRemote(self, (
            ("/usr/bin/scylla --version", (0, "4.4.dev-0.20200205.2816404f575\n", "")),
            ("/usr/bin/scylla --build-id", (0, "xxx", "")),
        ))
        self.assertEqual(self.monitor_cluster.monitoring_version, "master")


class NodetoolDummyNode(BaseNode):  # pylint: disable=abstract-method

    def __init__(self, resp):  # pylint: disable=super-init-not-called
        self.resp = resp

    def run_nodetool(self, *args, **kwargs):  # pylint: disable=unused-argument
        return Result(exited=0, stderr="", stdout=self.resp)


class DummyScyllaCluster(BaseScyllaCluster, BaseCluster):  # pylint: disable=abstract-method
    nodes: List['NodetoolDummyNode']

    def __init__(self, params):  # pylint: disable=super-init-not-called
        self.nodes = params
        self.name = 'dummy_cluster'


class TestNodetool(unittest.TestCase):
    def test_describering_parsing(self):  # pylint: disable=no-self-use
        """ Test "nodetool describering" output parsing """
        resp = "\n".join(["Schema Version:00703362-03ed-3b41-afcb-ed34c1d1586c TokenRange:",
                          "TokenRange(start_token:-9193109213506951143, end_token:9202125676696964746, "
                          "endpoints:[127.0.49.3], rpc_endpoints:[127.0.49.3], "
                          "endpoint_details:[EndpointDetails(host:127.0.49.3, datacenter:datacenter1, rack:rack1)])",
                          "TokenRange(start_token:9154793403047166459, end_token:9156354786201613199, "
                          "endpoints:[127.0.49.3], rpc_endpoints:[127.0.49.3], "
                          "endpoint_details:[EndpointDetails(host:127.0.49.3, datacenter:datacenter1, rack:rack1)])",
                          ]
                         )
        node = NodetoolDummyNode(resp=resp)
        ranges = get_keyspace_partition_ranges(node=node, keyspace="")
        assert ranges == [{'start_token': -9193109213506951143,
                           'details': [{'host': '127.0.49.3', 'datacenter': 'datacenter1', 'rack': 'rack1'}],
                           'end_token': 9202125676696964746, 'endpoints': '127.0.49.3', 'rpc_endpoints': '127.0.49.3'},
                          {'start_token': 9154793403047166459,
                           'details': [{'host': '127.0.49.3', 'datacenter': 'datacenter1', 'rack': 'rack1'}],
                           'end_token': 9156354786201613199, 'endpoints': '127.0.49.3', 'rpc_endpoints': '127.0.49.3'}]

        min_token, max_token = keyspace_min_max_tokens(node=node, keyspace="")
        assert min_token == -9193109213506951143
        assert max_token == 9202125676696964746


class TestNodetoolStatus(unittest.TestCase):

    def test_can_get_nodetool_status_typical(self):  # pylint: disable=no-self-use
        resp = "\n".join(["Datacenter: eastus",
                          "==================",
                          "Status=Up/Down",
                          "|/ State=Normal/Leaving/Joining/Moving",
                          "--  Address   Load       Tokens       Owns    Host ID                               Rack",
                          "UN  10.0.59.34    21.71 GB   256          ?       e5bcb094-e4de-43aa-8dc9-b1bf74b3b346  1a",
                          "UN  10.0.198.153  ?          256          ?       fba174cd-917a-40f6-ab62-cc58efaaf301  1a"
                          ]
                         )
        node = NodetoolDummyNode(resp=resp)
        db_cluster = DummyScyllaCluster([node])

        status = db_cluster.get_nodetool_status()

        assert status == {'eastus':
                          {'10.0.59.34':
                           {'state': 'UN', 'load': '21.71GB', 'tokens': '256', 'owns': '?',
                            'host_id': 'e5bcb094-e4de-43aa-8dc9-b1bf74b3b346', 'rack': '1a'},
                           '10.0.198.153': {'state': 'UN', 'load': '?', 'tokens': '256', 'owns': '?',
                                            'host_id': 'fba174cd-917a-40f6-ab62-cc58efaaf301', 'rack': '1a'}}}

    def test_can_get_nodetool_status_ipv6(self):  # pylint: disable=no-self-use
        resp = "\n".join(["Datacenter: eu-north",
                          "====================",
                          "Status=Up/Down",
                          "|/ State=Normal/Leaving/Joining/Moving",
                          "--  Address                                 Load       Tokens       Owns    Host ID                Rack",
                          "UN  2a05:d016:cf8:de00:e07d:5832:c5c0:36a0  774 KB     256          ?       e2ed6943  1a",
                          "UN  2a05:d016:cf8:de00:339e:d0d:9446:1980   1.04 MB    256          ?       d67e8502  1a",
                          ]
                         )
        node = NodetoolDummyNode(resp=resp)
        db_cluster = DummyScyllaCluster([node])

        status = db_cluster.get_nodetool_status()

        assert status == {'eu-north':
                          {'2a05:d016:cf8:de00:e07d:5832:c5c0:36a0':
                           {'state': 'UN', 'load': '774KB', 'tokens': '256', 'owns': '?',
                            'host_id': 'e2ed6943', 'rack': '1a'},
                           '2a05:d016:cf8:de00:339e:d0d:9446:1980': {'state': 'UN', 'load': '1.04MB', 'tokens': '256', 'owns': '?',
                                                                     'host_id': 'd67e8502', 'rack': '1a'}}}

    def test_can_get_nodetool_status_azure(self):  # pylint: disable=no-self-use
        resp = "\n".join(["Datacenter: eastus",
                         "==================",
                          "Status=Up/Down",
                          "|/ State=Normal/Leaving/Joining/Moving",
                          "--  Address   Load       Tokens       Owns    Host ID                               Rack",
                          "UN  10.0.0.4  431 KB     256          ?       ed6af9a0-8c22-4813-ac9b-6fbeb462b687  ",
                          "UN  10.0.0.5  612 KB     256          ?       caa15869-cfb4-4229-85d7-0f4832986237  1",
                          "UN  10.0.0.6  806 KB     256          ?       3046ded9-ce17-4a3a-ac44-a3ada6916972  ",
                          ""
                          ]
                         )
        node = NodetoolDummyNode(resp=resp)
        db_cluster = DummyScyllaCluster([node])

        status = db_cluster.get_nodetool_status()

        assert status == {'eastus': {'10.0.0.4': {'host_id': 'ed6af9a0-8c22-4813-ac9b-6fbeb462b687',
                                                  'load': '431KB',
                                                  'owns': '?',
                                                  'rack': '',
                                                  'state': 'UN',
                                                  'tokens': '256'},
                                     '10.0.0.5': {'host_id': 'caa15869-cfb4-4229-85d7-0f4832986237',
                                                  'load': '612KB',
                                                  'owns': '?',
                                                  'rack': '1',
                                                  'state': 'UN',
                                                  'tokens': '256'},
                                     '10.0.0.6': {'host_id': '3046ded9-ce17-4a3a-ac44-a3ada6916972',
                                                  'load': '806KB',
                                                  'owns': '?',
                                                  'rack': '',
                                                  'state': 'UN',
                                                  'tokens': '256'}
                                     }
                          }


@pytest.mark.parametrize("grep_results,expected_core_number", (
    ("1", 1), ("2", 1), ("9", 1), ("10", 1), ("11", 1),

    ("1-7", 7), ("2-7", 6), ("1-6", 6), ("2-6", 5),

    ("1-7,9", 8), ("1,9-15", 8), ("1-7,9-15", 14), ("2-7,10-15", 12),

    ("1-7,9-15,17-23", 21),
    ("1-7,9-15,17", 15),
    ("1-7,9,17-23", 15),
    ("1,9-15,17-23", 15),
    ("1,9,17", 3),
    ("1,9-15,17", 9),
    ("1-7,9,17", 9),
    ("1,9,17-23", 9),

    ("1-7,9-15,17-23,25-31", 28),
    ("1,9-15,17-23,25-31", 22),
    ("1-7,9,17-23,25-31", 22),
    ("1-7,9-15,23,25-31", 22),
    ("1-7,9-15,17-23,31", 22),
    ("1-7,15,17,25-31", 16),
    ("1,9-15,17-23,31", 16),
    ("1,15,17-23,31", 10),
    ("1,15,17,31", 4),
    ("1,9-15,17,25-31", 16),
    ("1-7,9,17-23,25", 16),
))
def test_base_node_cpuset(grep_results, expected_core_number):
    dummy_node = DummyNode(
        name='dummy_node',
        parent_cluster=None,
        base_logdir=tempfile.mkdtemp(),
        ssh_login_info=dict(key_file='~/.ssh/scylla-test'),
    )
    dummy_node.init()
    grep_results_obj = type("FakeGrepResults", (), {"stdout": f'CPUSET="--cpuset {grep_results} "'})
    dummy_node.remoter = type("FakeRemoter", (), {
        "run": (lambda *args, **kwargs: grep_results_obj),
    })

    cpuset_value = dummy_node.cpuset

    assert isinstance(cpuset_value, int)
    assert cpuset_value == expected_core_number
