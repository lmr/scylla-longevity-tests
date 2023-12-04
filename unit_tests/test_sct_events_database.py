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

import unittest
import re

from sdcm.sct_events import Severity
from sdcm.sct_events.base import LogEvent
from sdcm.sct_events.database import \
    DatabaseLogEvent, FullScanEvent, IndexSpecialColumnErrorEvent, TOLERABLE_REACTOR_STALL, SYSTEM_ERROR_EVENTS, DataOperationsEvent, \
    DataOperationsCaseEvent


class TestDatabaseLogEvent(unittest.TestCase):
    def test_known_system_errors(self):
        self.assertTrue(issubclass(DatabaseLogEvent.NO_SPACE_ERROR, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.UNKNOWN_VERB, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.CLIENT_DISCONNECT, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.SEMAPHORE_TIME_OUT, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.LDAP_CONNECTION_RESET, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.SYSTEM_PAXOS_TIMEOUT, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.SERVICE_LEVEL_CONTROLLER, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.RESTARTED_DUE_TO_TIME_OUT, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.EMPTY_NESTED_EXCEPTION, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.DATABASE_ERROR, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.BAD_ALLOC, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.SCHEMA_FAILURE, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.RUNTIME_ERROR, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.FILESYSTEM_ERROR, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.STACKTRACE, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.BACKTRACE, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.ABORTING_ON_SHARD, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.SEGMENTATION, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.INTEGRITY_CHECK, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.REACTOR_STALLED, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.SUPPRESSED_MESSAGES, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.stream_exception, DatabaseLogEvent))
        self.assertTrue(issubclass(DatabaseLogEvent.DISK_ERROR, DatabaseLogEvent))

    def test_reactor_stalled_severity(self):
        event1 = DatabaseLogEvent.REACTOR_STALLED()
        self.assertEqual(event1.severity, Severity.DEBUG)

        self.assertIs(event1, event1.add_info(node="n1", line=f"{TOLERABLE_REACTOR_STALL-1} ms", line_number=1))
        self.assertEqual(event1.severity, Severity.DEBUG)
        self.assertEqual(event1.node, "n1")
        self.assertEqual(event1.line, f"{TOLERABLE_REACTOR_STALL-1} ms")
        self.assertEqual(event1.line_number, 1)

        event2 = DatabaseLogEvent.REACTOR_STALLED()
        self.assertEqual(event2.severity, Severity.DEBUG)
        self.assertIs(event2, event2.add_info(node="n2", line=f"{TOLERABLE_REACTOR_STALL} ms", line_number=2))
        self.assertEqual(event2.severity, Severity.ERROR)
        self.assertEqual(event2.node, "n2")
        self.assertEqual(event2.line, f"{TOLERABLE_REACTOR_STALL} ms")
        self.assertEqual(event2.line_number, 2)

    def test_kernel_callstack_severity(self):
        event1 = DatabaseLogEvent.KERNEL_CALLSTACK()
        self.assertEqual(event1.severity, Severity.DEBUG)

        self.assertIs(event1, event1.add_info(node="n1", line="kernel callstack 0xffffffffffffff80", line_number=1))
        self.assertEqual(event1.node, "n1")
        self.assertEqual(event1.line_number, 1)

        event2 = DatabaseLogEvent.REACTOR_STALLED()
        self.assertEqual(event2.severity, Severity.DEBUG)
        self.assertIs(event2, event2.add_info(node="n2", line="kernel callstack 0xffffffffffffff80", line_number=2))
        self.assertEqual(event2.node, "n2")
        self.assertEqual(event2.line_number, 2)

    def test_system_error_events_list(self):
        self.assertSetEqual(set(dir(DatabaseLogEvent)) - set(dir(LogEvent)),
                            {ev.type for ev in SYSTEM_ERROR_EVENTS})

    def test_disk_error_event(self):  # pylint: disable=line-too-long

        disk_error_event = DatabaseLogEvent.DISK_ERROR()
        # pylint: disable=line-too-long
        log_lines = """2022-02-07T06:13:14+00:00 longevity-tls-1tb-7d-4-6-db-node-5279f155-0-4 !    INFO |  [shard 8] compaction - [Compact keyspace1.standard1 089530c0-87dd-11ec-8382-519d84e34cb0] Compacting [/var/lib/scylla/data/keyspace1/standard1-b8e41570875811ec8382519d84e34cb0/md-287128-big-Data.db:level=2:origin=compaction,/var/lib/scylla/data/keyspace1/standard1-b8e41570875811ec8382519d84e34cb0/md-287112-big-Data.db:level=2:origin=compaction,/var/lib/scylla/data/keyspace1/standard1-b8e41570875811ec8382519d84e34cb0/md-290136-big-Data.db:level=2:origin=compaction,/var/lib/scylla/data/keyspace1/standard1-b8e41570875811ec8382519d84e34cb0/md-291192-big-Data.db:level=1:origin=compaction]
2022-02-07T06:13:14+00:00 longevity-tls-1tb-7d-4-6-db-node-5279f155-0-4 !     ERR | blk_update_request: critical medium error, dev nvme0n11, sector 4141328 op 0x0:(READ) flags 0x0 phys_seg 1 prio class 0
2022-02-07T06:13:14+00:00 longevity-tls-1tb-7d-4-6-db-node-5279f155-0-4 !     ERR |  [shard 6] storage_service - Shutting down communications due to I/O errors until operator intervention: Disk error: std::system_error (error system:61, No data available)
2022-02-07T06:13:14+00:00 longevity-tls-1tb-7d-4-6-db-node-5279f155-0-4 !    INFO |  [shard 0] storage_service - Stop transport: starts
2022-02-07T06:13:14+00:00 longevity-tls-1tb-7d-4-6-db-node-5279f155-0-4 !    INFO |  [shard 0] storage_service - Shutting down native transport
        """
        expected_error_data = {
            "line_number": 2,
            "line": "2022-02-07T06:13:14+00:00 longevity-tls-1tb-7d-4-6-db-node-5279f155-0-4 !     ERR |  [shard 6] storage_service - Shutting down communications due to I/O errors until operator intervention: Disk error: std::system_error (error system:61, No data available)",
            "node": "longevity-tls-1tb-7d-4-6-db-node-5279f155-0-4"
        }

        for num, line in enumerate(log_lines.splitlines()):
            if re.search(disk_error_event.regex, line):
                disk_error_event.add_info("longevity-tls-1tb-7d-4-6-db-node-5279f155-0-4",
                                          line, num)

        self.assertEqual(expected_error_data["node"], disk_error_event.node)
        self.assertEqual(expected_error_data["line_number"], disk_error_event.line_number)
        self.assertEqual(expected_error_data["line"], disk_error_event.line)


class TestFullScanEvent(unittest.TestCase):
    NODE_NAME = "db-node-1"
    KS_CF = "ks_cf"
    MSG = "msg"

    def test_no_message(self):
        event = FullScanEvent(node=self.NODE_NAME, ks_cf=self.KS_CF)
        self.assertIsNone(event.message)
        self.assertRegex(
            str(event),
            r"\(FullScanEvent Severity\.NORMAL\) period_type=not-set event_id=([\d\w-]{36}) "
            f"node={self.NODE_NAME} select_from={self.KS_CF}")

    def test_with_message(self):
        event = FullScanEvent(node=self.NODE_NAME, ks_cf=self.KS_CF, message=self.MSG)
        self.assertEqual(event.message, self.MSG)
        self.assertRegex(
            str(event),
            r"\(FullScanEvent Severity\.NORMAL\) period_type=not-set event_id=([\d\w-]{36}) "
            f"node={self.NODE_NAME} select_from={self.KS_CF} message={self.MSG}")


class TestIndexSpecialColumnErrorEvent(unittest.TestCase):
    def test_msgfmt(self):
        event = IndexSpecialColumnErrorEvent(message="m1")
        event.event_id = "ac449879-485a-4b06-8596-3fbe58881093"
        self.assertEqual(str(event),
                         "(IndexSpecialColumnErrorEvent Severity.ERROR) period_type=one-time "
                         "event_id=ac449879-485a-4b06-8596-3fbe58881093: message=m1")


class TestDataOperationsEvent(unittest.TestCase):
    def test_data_operation_event(self):
        event = DataOperationsEvent(message="m1", publish_event=False)
        event.event_id = "3c8e2362-a987-4eff-953f-9cd1ad2017e5"
        event.begin_event()
        self.assertEqual(
            str(event),
            "(DataOperationsEvent Severity.NORMAL) period_type=begin event_id=3c8e2362-a987-4eff-953f-9cd1ad2017e5: message=m1"
        )

        event.duration = 20.325
        event.end_event()
        self.assertEqual(
            str(event),
            "(DataOperationsEvent Severity.NORMAL) period_type=end event_id=3c8e2362-a987-4eff-953f-9cd1ad2017e5 duration=20.325s: "
            "message=m1"
        )


class TestDataOperationsCaseEvent(unittest.TestCase):
    def test_data_operation_event(self):
        event = DataOperationsCaseEvent(case="delete partition", message="m1", publish_event=False)
        event.event_id = "3c8e2362-a987-4eff-953f-9cd1ad2017e5"
        event.begin_event()
        self.assertEqual(
            str(event),
            "(DataOperationsCaseEvent Severity.NORMAL) period_type=begin event_id=3c8e2362-a987-4eff-953f-9cd1ad2017e5: "
            "case=delete partition message=m1"
        )

        event.duration = 20.325
        event.end_event()
        self.assertEqual(
            str(event),
            "(DataOperationsCaseEvent Severity.NORMAL) period_type=end event_id=3c8e2362-a987-4eff-953f-9cd1ad2017e5 duration=20.325s: "
            "case=delete partition message=m1"
        )
