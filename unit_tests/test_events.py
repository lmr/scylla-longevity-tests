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
import logging
import unittest
import multiprocessing
from pathlib import Path
from datetime import datetime
from unittest import mock

from parameterized import parameterized

from sdcm.nemesis import UnsupportedNemesis
from sdcm.prometheus import start_metrics_server
from sdcm.sct_events.nodetool import NodetoolEvent
from sdcm.utils.decorators import timeout
from sdcm.sct_events import Severity
from sdcm.sct_events.system import CoreDumpEvent, TestFrameworkEvent, SoftTimeoutEvent
from sdcm.sct_events.filters import DbEventsFilter, EventsFilter, EventsSeverityChangerFilter
from sdcm.sct_events.loaders import YcsbStressEvent
from sdcm.sct_events.nemesis import DisruptionEvent
from sdcm.sct_events.database import DatabaseLogEvent
from sdcm.sct_events.file_logger import get_logger_event_summary

from unit_tests.lib.events_utils import EventsUtilsMixin

LOGGER = logging.getLogger(__name__)


class BaseEventsTest(unittest.TestCase, EventsUtilsMixin):
    killed = multiprocessing.Event()

    @classmethod
    def setUpClass(cls) -> None:
        start_metrics_server()
        cls.setup_events_processes(events_device=True, events_main_device=False, registry_patcher=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.teardown_events_processes()

    @classmethod
    def get_event_log_file(cls, name: str) -> str:
        if (log_file := Path(cls.temp_dir, "events_log", name)).exists():
            return log_file.read_text(encoding="utf-8")
        return ""

    @timeout(timeout=10, sleep_time=0.05)
    def wait_for_event_summary(self):
        return get_logger_event_summary(_registry=self.events_processes_registry)


class SctEventsTests(BaseEventsTest):  # pylint: disable=too-many-public-methods
    # increase the max length to see the full strings in the AssertionError which precedes the diff
    maxDiff = None

    def test_disruption_skipped_event(self):
        with self.assertRaises(UnsupportedNemesis), DisruptionEvent(
                nemesis_name="DeleteByRowsRange", node="target_node", publish_event=False) as nemesis_event:
            try:
                raise UnsupportedNemesis("This nemesis can run on scylla_bench test only")
            except UnsupportedNemesis as exc:
                nemesis_event.event_id = "c2561d8b-97ca-44fb-b5b1-8bcc0d437318"
                nemesis_event.skip(skip_reason=str(exc))
                nemesis_event.duration = 15
                raise

        self.assertEqual(
            str(nemesis_event),
            '(DisruptionEvent Severity.NORMAL) period_type=end event_id=c2561d8b-97ca-44fb-b5b1-8bcc0d437318 '
            'duration=15s: nemesis_name=DeleteByRowsRange target_node=target_node skipped skip_reason=This nemesis can '
            'run on scylla_bench test only'
        )

    def test_disruption_raised_critical_event(self):
        with self.assertRaises(ZeroDivisionError), DisruptionEvent(
                nemesis_name="DeleteByRowsRange", node="target_node", publish_event=False) as nemesis_event:
            nemesis_event.event_id = "c2561d8b-97ca-44fb-b5b1-8bcc0d437318"
            self.assertEqual(
                str(nemesis_event),
                '(DisruptionEvent Severity.NORMAL) period_type=begin event_id=c2561d8b-97ca-44fb-b5b1-8bcc0d437318: '
                'nemesis_name=DeleteByRowsRange target_node=target_node'
            )

            try:
                1 / 0
            except ZeroDivisionError:
                nemesis_event.add_error(['division by zero'])
                nemesis_event.full_traceback = "traceback.format_exc()"
                nemesis_event.severity = Severity.CRITICAL
                nemesis_event.duration = 15
                raise

        self.assertIn('division by zero', nemesis_event.errors_formatted)
        self.assertEqual(nemesis_event.severity, Severity.CRITICAL)
        self.assertEqual(nemesis_event.duration_formatted, '15s')

    def test_disruption_raised_error_event(self):
        with self.assertRaises(ZeroDivisionError), DisruptionEvent(
                nemesis_name="DeleteByRowsRange", node="target_node", publish_event=False) as nemesis_event:
            nemesis_event.event_id = "c2561d8b-97ca-44fb-b5b1-8bcc0d437318"
            self.assertEqual(
                str(nemesis_event),
                '(DisruptionEvent Severity.NORMAL) period_type=begin event_id=c2561d8b-97ca-44fb-b5b1-8bcc0d437318: '
                'nemesis_name=DeleteByRowsRange target_node=target_node'
            )

            try:
                1 / 0
            except ZeroDivisionError:
                nemesis_event.add_error(['division by zero'])
                nemesis_event.full_traceback = "traceback.format_exc()"
                nemesis_event.duration = 15
                raise

        self.assertIn('division by zero', nemesis_event.errors_formatted)
        self.assertEqual(nemesis_event.severity, Severity.ERROR)
        self.assertEqual(nemesis_event.duration_formatted, '15s')

    def test_disruption_error_event(self):
        with DisruptionEvent(nemesis_name="DeleteByRowsRange", node="target_node",
                             publish_event=False) as nemesis_event:
            nemesis_event.event_id = "c2561d8b-97ca-44fb-b5b1-8bcc0d437318"
            self.assertEqual(
                str(nemesis_event),
                '(DisruptionEvent Severity.NORMAL) period_type=begin event_id=c2561d8b-97ca-44fb-b5b1-8bcc0d437318: '
                'nemesis_name=DeleteByRowsRange target_node=target_node'
            )

            try:
                1 / 0
            except ZeroDivisionError:
                nemesis_event.add_error(['division by zero'])
                nemesis_event.full_traceback = "traceback.format_exc()"
                nemesis_event.duration = 15
                nemesis_event.severity = Severity.ERROR

        self.assertIn(
            '(DisruptionEvent Severity.ERROR) period_type=end event_id=c2561d8b-97ca-44fb-b5b1-8bcc0d437318 '
            'duration=15s: nemesis_name=DeleteByRowsRange target_node=target_node errors=division by zero\n',
            str(nemesis_event),
        )

    def test_disruption_normal_event(self):
        with DisruptionEvent(nemesis_name="DeleteByRowsRange",
                             node="target_node", publish_event=False) as nemesis_event:
            nemesis_event.event_id = "c2561d8b-97ca-44fb-b5b1-8bcc0d437318"
            self.assertEqual(
                str(nemesis_event),
                '(DisruptionEvent Severity.NORMAL) period_type=begin event_id=c2561d8b-97ca-44fb-b5b1-8bcc0d437318: '
                'nemesis_name=DeleteByRowsRange target_node=target_node'
            )
            nemesis_event.duration = 15

        self.assertEqual(
            str(nemesis_event),
            '(DisruptionEvent Severity.NORMAL) period_type=end event_id=c2561d8b-97ca-44fb-b5b1-8bcc0d437318 '
            'duration=15s: nemesis_name=DeleteByRowsRange target_node=target_node'
        )

    def test_filter(self):
        enospc_line_1 = \
            "[99.80.124.204] [stdout] Mar 31 09:08:10 warning|  [shard 8] commitlog - Exception in segment " \
            "reservation: storage_io_error (Storage I/O error: 28: No space left on device)"
        enospc_line_2 = \
            "2019-10-29T12:19:49+00:00  ip-172-30-0-184 !WARNING | scylla: [shard 2] storage_service - Commitlog " \
            "error: std::filesystem::__cxx11::filesystem_error (error system:28, filesystem error: open failed: No " \
            "space left on device [/var/lib/scylla/hints/2/172.30.0.116/HintsLog-1-36028797019122576.log])"

        log_content_before = self.get_event_log_file("events.log")

        # 13 events in total: 2 events per filter x 4 filters + 5 events.
        with self.wait_for_n_events(self.get_events_logger(), count=13, timeout=3):
            with DbEventsFilter(db_event=DatabaseLogEvent.NO_SPACE_ERROR), \
                    DbEventsFilter(db_event=DatabaseLogEvent.BACKTRACE, line="No space left on device"), \
                    DbEventsFilter(db_event=DatabaseLogEvent.DATABASE_ERROR, line="No space left on device"), \
                    DbEventsFilter(db_event=DatabaseLogEvent.FILESYSTEM_ERROR, line="No space left on device"):
                DatabaseLogEvent.NO_SPACE_ERROR().add_info(node="A", line_number=22, line=enospc_line_1).publish()
                DatabaseLogEvent.BACKTRACE().add_info(node="A", line_number=22, line=enospc_line_1).publish()
                DatabaseLogEvent.FILESYSTEM_ERROR().add_info(node="A", line_number=22, line=enospc_line_2).publish()
                DatabaseLogEvent.DATABASE_ERROR().add_info(node="A", line_number=22, line=enospc_line_1).publish()
                DatabaseLogEvent.NO_SPACE_ERROR().add_info(node="A", line_number=22, line=enospc_line_1).publish()

        self.assertEqual(log_content_before, self.get_event_log_file("events.log"))

    def test_general_filter(self):
        with self.wait_for_n_events(self.get_events_logger(), count=4, timeout=3):
            with EventsFilter(event_class=CoreDumpEvent):
                CoreDumpEvent(corefile_url="http://",
                              backtrace="asfasdfsdf",
                              node="node xy",
                              download_instructions="test_general_filter",
                              source_timestamp=1578998425.0).publish()  # Tue 2020-01-14 10:40:25 UTC
                TestFrameworkEvent(source="", source_method="").publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("TestFrameworkEvent", log_content)
        self.assertNotIn("test_general_filter", log_content)

    def test_general_filter_regex(self):
        with self.wait_for_n_events(self.get_events_logger(), count=4, timeout=3):
            with EventsFilter(regex=".*1234567890.*"):
                CoreDumpEvent(corefile_url="http://",
                              backtrace="asfasdfsdf",
                              node="node xy",
                              download_instructions="gsutil cp gs://upload.scylladb.com/core.scylla-jmx.996.1234567890"
                                                    ".3968.1566979933000/core.scylla-jmx.996.d173729352e34c76aaf8db334"
                                                    "2153c3e.3968.1566979933000000 .",
                              source_timestamp=1578998425.0).publish()  # Tue 2020-01-14 10:40:25 UTC
                TestFrameworkEvent(source="", source_method="").publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("TestFrameworkEvent", log_content)
        self.assertNotIn("1234567890", log_content)

    def test_severity_changer(self):
        extra_time_to_expiration = 10
        with self.wait_for_n_events(self.get_events_logger(), count=5, timeout=3):
            with EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                             event_class=TestFrameworkEvent,
                                             extra_time_to_expiration=extra_time_to_expiration):
                TestFrameworkEvent(source="critical that should be lowered #1",
                                   source_method="",
                                   severity=Severity.CRITICAL).publish()
            TestFrameworkEvent(source="critical that should be lowered #2",
                               source_method="",
                               severity=Severity.CRITICAL).publish()
            event = TestFrameworkEvent(
                source="critical that should not be lowered #3",
                source_method="",
                severity=Severity.CRITICAL)
            event.source_timestamp = time.time() + extra_time_to_expiration
            event.publish()

        log_content = self.get_event_log_file("warning.log")
        crit_log_content = self.get_event_log_file("critical.log")

        self.assertIn("TestFrameworkEvent", log_content)
        self.assertIn("critical that should be lowered #1", log_content)
        self.assertIn("critical that should be lowered #2", log_content)
        self.assertNotIn("critical that should not be lowered #3", log_content)
        self.assertIn("critical that should not be lowered #3", crit_log_content)

    def test_severity_changer_db_log(self):
        """
            See https://github.com/scylladb/scylla-cluster-tests/issues/2115
        """
        extra_time_to_expiration = 2
        # 1) Lower DatabaseLogEvent to WARNING for 1 sec.
        with self.wait_for_n_events(self.get_events_logger(), count=5, timeout=3):
            with EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                             event_class=DatabaseLogEvent,
                                             extra_time_to_expiration=extra_time_to_expiration):
                DatabaseLogEvent.NO_SPACE_ERROR() \
                    .add_info(node="A", line_number=22, line="error that should be lowered #1") \
                    .publish()
            DatabaseLogEvent.NO_SPACE_ERROR() \
                .add_info(node="A", line_number=22, line="error that should be lowered #2") \
                .publish()
            event = DatabaseLogEvent.NO_SPACE_ERROR().add_info(
                node="A", line_number=22, line="error that should not be lowered #3")
            event.source_timestamp = time.time() + extra_time_to_expiration
            event.publish()

        log_content = self.get_event_log_file("warning.log")
        error_log_content = self.get_event_log_file("error.log")

        self.assertIn("DatabaseLogEvent", log_content)
        self.assertIn("error that should be lowered #1", log_content)
        self.assertIn("error that should be lowered #2", log_content)
        self.assertNotIn("error that should not be lowered #3", log_content)
        self.assertIn("error that should not be lowered #3", error_log_content)

        # 2) One of the next DatabaseLogEvent event should expire the EventsSeverityChangerFilter
        #    (and not crash all subscribers)
        with self.wait_for_n_events(self.get_events_logger(), count=2, timeout=3):
            for _ in range(2):
                time.sleep(1)
                DatabaseLogEvent.NO_SPACE_ERROR() \
                    .add_info(node="A", line_number=22, line="error that shouldn't be lowered") \
                    .publish()

        log_content = self.get_event_log_file("error.log")

        self.assertIn("error that shouldn't be lowered", log_content)

    def test_ycsb_filter(self):
        with self.wait_for_n_events(self.get_events_logger(), count=4, timeout=3):
            with EventsFilter(event_class=YcsbStressEvent,
                              regex=".*Internal server error: exceptions::unavailable_exception.*"):
                YcsbStressEvent.error(
                    node="Node alternator-3h-silence--loader-node-bb90aa05-2"
                         " [34.251.153.122 | 10.0.220.55] (seed: False)",
                    stress_cmd="ycsb",
                    errors=["237951 [Thread-47] ERROR site.ycsb.db.DynamoDBClient"
                            " -com.amazonaws.AmazonServiceException: Internal server error:"
                            " exceptions::unavailable_exception (Cannot achieve consistency"
                            " level for cl LOCAL_ONE. Requires 1, alive 0) (Service: AmazonDynamoDBv2;"
                            " Status Code: 500; Error Code: Internal Server Error; Request ID: null)"]).publish()
                TestFrameworkEvent(source="", source_method="").publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("TestFrameworkEvent", log_content)
        self.assertNotIn("YcsbStressEvent", log_content)

        with self.wait_for_n_events(self.get_events_logger(), count=1):
            YcsbStressEvent.error(
                node="Node alternator-3h-silence--loader-node-bb90aa05-2"
                     " [34.251.153.122 | 10.0.220.55] (seed: False)",
                stress_cmd="ycsb",
                errors=["237951 [Thread-47] ERROR site.ycsb.db.DynamoDBClient"
                        " -com.amazonaws.AmazonServiceException: Internal server error:"
                        " exceptions::unavailable_exception (Cannot achieve consistency"
                        " level for cl LOCAL_ONE. Requires 1, alive 0) (Service: AmazonDynamoDBv2;"
                        " Status Code: 500; Error Code: Internal Server Error; Request ID: null)"]).publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("TestFrameworkEvent", log_content)
        self.assertIn("YcsbStressEvent", log_content)

    def test_filter_repair(self):
        failed_repaired_line = "2019-07-28T10:53:29+00:00  ip-10-0-167-91 !INFO    | scylla.bin: [shard 0] repair - " \
                               "Got error in row level repair: std::runtime_error (repair id 1 is aborted on shard 0)"

        # 9 events in total: 2 events per filter x 3 filters + 3 events.
        with self.wait_for_n_events(self.get_events_logger(), count=9, timeout=3):
            with DbEventsFilter(db_event=DatabaseLogEvent.DATABASE_ERROR,
                                line="repair's stream failed: streaming::stream_exception"), \
                    DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                                   line="Can not find stream_manager"), \
                    DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR, line="is aborted"):

                DatabaseLogEvent.RUNTIME_ERROR().add_info(node="A", line_number=22, line=failed_repaired_line).publish()
                DatabaseLogEvent.RUNTIME_ERROR().add_info(node="A", line_number=22, line=failed_repaired_line).publish()
                DatabaseLogEvent.NO_SPACE_ERROR().add_info(node="B", line_number=22, line="not filtered").publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("not filtered", log_content)
        self.assertNotIn("repair id 1", log_content)

    def test_filter_upgrade(self):
        known_failure_line = "!ERR     | scylla:  [shard 3] storage_proxy - Exception when communicating with " \
                             "10.142.0.56: std::runtime_error (Failed to load schema version " \
                             "b40e405f-462c-38f2-a90c-6f130ddbf6f3) "

        with self.wait_for_n_events(self.get_events_logger(), count=5, timeout=3):
            with DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR, line="Failed to load schema"):
                DatabaseLogEvent.RUNTIME_ERROR().add_info(node="A", line_number=22, line=known_failure_line).publish()
                DatabaseLogEvent.RUNTIME_ERROR().add_info(node="A", line_number=22, line=known_failure_line).publish()
                DatabaseLogEvent.NO_SPACE_ERROR().add_info(node="B", line_number=22, line="not filtered").publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("not filtered", log_content)
        self.assertNotIn("Exception when communicating", log_content)

    def test_filter_by_node(self):
        with self.wait_for_n_events(self.get_events_logger(), count=4, timeout=3):
            with DbEventsFilter(db_event=DatabaseLogEvent.NO_SPACE_ERROR, node="A"):
                DatabaseLogEvent.NO_SPACE_ERROR().add_info(node="A", line_number=22, line="this is filtered").publish()
                DatabaseLogEvent.NO_SPACE_ERROR().add_info(node="B", line_number=22, line="not filtered").publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("not filtered", log_content)
        self.assertNotIn("this is filtered", log_content)

    def test_filter_expiration(self):
        with self.wait_for_n_events(self.get_events_logger(), count=4, timeout=10):
            line_prefix = f"{datetime.utcnow():%Y-%m-%dT%H:%M:%S+00:00}"

            with DbEventsFilter(db_event=DatabaseLogEvent.NO_SPACE_ERROR, node="A"):
                DatabaseLogEvent.NO_SPACE_ERROR() \
                    .add_info(node="A", line_number=22, line=line_prefix + " this is filtered") \
                    .publish()

            line_prefix = f"{datetime.utcfromtimestamp(time.time() + 1):%Y-%m-%dT%H:%M:%S+00:00}"
            DatabaseLogEvent.NO_SPACE_ERROR() \
                .add_info(node="A", line_number=22,
                          line=line_prefix + " : this is not filtered") \
                .publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("this is not filtered", log_content)
        self.assertNotIn("this is filtered", log_content)

    def test_default_filters(self):
        with self.wait_for_n_events(self.get_events_logger(), count=2):
            DatabaseLogEvent.BACKTRACE() \
                .add_info(node="A",
                          line_number=22,
                          line="Jul 01 03:37:31 ip-10-0-127-151.eu-west-1.compute.internal"
                               " scylla[6026]:Rate-limit: supressed 4294967292 backtraces on shard 5") \
                .publish()
            DatabaseLogEvent.BACKTRACE() \
                .add_info(node="A", line_number=22, line="other back trace that shouldn't be filtered") \
                .publish()

        log_content = self.get_event_log_file("events.log")

        self.assertIn("other back trace", log_content)
        self.assertNotIn("supressed", log_content)

    def test_failed_stall_during_filter(self):
        with self.wait_for_n_events(self.get_events_logger(), count=5, timeout=3):
            with DbEventsFilter(db_event=DatabaseLogEvent.NO_SPACE_ERROR), \
                    DbEventsFilter(db_event=DatabaseLogEvent.BACKTRACE, line="No space left on device"):

                event = DatabaseLogEvent.REACTOR_STALLED()
                event.add_info(node="A",
                               line_number=22,
                               line="[99.80.124.204] [stdout] Mar 31 09:08:10 warning|  reactor stall 20").publish()

        self.assertEqual(event.severity, Severity.DEBUG)

    @parameterized.expand([(None, ''),
                           (2, '2s'),
                           (2.52, '2.52s'),
                           (326, '5m26s'),
                           (4598, '1h16m38s'),
                           (87400, '1d0h16m40s'),
                           ])
    def test_duration_format(self, duration_input, duration_formatted):
        event = NodetoolEvent(nodetool_command="scrub --skip-corrupted drop_table_during_repair_ks_0",
                              node='1.0.0.121', options="more options", publish_event=False)

        event.duration = duration_input
        self.assertEqual(duration_formatted, event.duration_formatted)

    @mock.patch('sdcm.sct_events.base.SctEvent.publish')
    def test_publish_called(self, publish):
        event = NodetoolEvent(nodetool_command="scrub", node='1.0.0.121',
                              options="", publish_event=True)
        event.begin_event()
        self.assertTrue(publish.called, "Publish function was not called unexpectedly")

    @mock.patch('sdcm.sct_events.base.SctEvent.publish')
    def test_publish_not_called(self, publish):
        event = NodetoolEvent(nodetool_command="scrub", node='1.0.0.121',
                              options="", publish_event=False)
        event.begin_event()
        self.assertFalse(publish.called, "Publish function was called unexpectedly")

    @staticmethod
    def test_soft_timeout():
        event = SoftTimeoutEvent(soft_timeout=0.1, operation="long-one", duration=0.2)
        event.publish_or_dump()
        event_data = str(event)

        assert event.trace
        assert "operation 'long-one' took 0.2s and soft-timeout was set to 0.1s" in event_data
