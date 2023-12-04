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

import re
import logging
from functools import partial
from typing import Type, List, Tuple, Generic, Optional, NamedTuple, Pattern, Callable, Match

from sdcm.sct_events import Severity, SctEventProtocol
from sdcm.sct_events.base import SctEvent, LogEvent, LogEventProtocol, T_log_event, InformationalEvent, \
    EventPeriod

from sdcm.sct_events.continuous_event import ContinuousEventsRegistry, ContinuousEvent
from sdcm.sct_events.system import TestFrameworkEvent

TOLERABLE_REACTOR_STALL: int = 500  # ms

LOGGER = logging.getLogger(__name__)


class DatabaseLogEvent(LogEvent, abstract=True):
    WARNING: Type[LogEventProtocol]
    NO_SPACE_ERROR: Type[LogEventProtocol]
    UNKNOWN_VERB: Type[LogEventProtocol]
    CLIENT_DISCONNECT: Type[LogEventProtocol]
    SEMAPHORE_TIME_OUT: Type[LogEventProtocol]
    LDAP_CONNECTION_RESET: Type[LogEventProtocol]
    SYSTEM_PAXOS_TIMEOUT: Type[LogEventProtocol]
    SERVICE_LEVEL_CONTROLLER: Type[LogEventProtocol]
    GATE_CLOSED: Type[LogEventProtocol]
    RESTARTED_DUE_TO_TIME_OUT: Type[LogEventProtocol]
    EMPTY_NESTED_EXCEPTION: Type[LogEventProtocol]
    BAD_ALLOC: Type[LogEventProtocol]
    SCHEMA_FAILURE: Type[LogEventProtocol]
    RUNTIME_ERROR: Type[LogEventProtocol]
    DIRECTORY_NOT_EMPTY: Type[LogEventProtocol]
    FILESYSTEM_ERROR: Type[LogEventProtocol]
    STACKTRACE: Type[LogEventProtocol]
    RAFT_TRANSFER_SNAPSHOT_ERROR: Type[LogEventProtocol]
    DISK_ERROR: Type[LogEventProtocol]
    COMPACTION_STOPPED: Type[LogEventProtocol]

    # REACTOR_STALLED must be above BACKTRACE as it has "Backtrace" in its message
    REACTOR_STALLED: Type[LogEventProtocol]
    KERNEL_CALLSTACK: Type[LogEventProtocol]
    ABORTING_ON_SHARD: Type[LogEventProtocol]
    SEGMENTATION: Type[LogEventProtocol]
    CORRUPTED_SSTABLE: Type[LogEventProtocol]
    INTEGRITY_CHECK: Type[LogEventProtocol]
    SUPPRESSED_MESSAGES: Type[LogEventProtocol]
    stream_exception: Type[LogEventProtocol]
    RPC_CONNECTION: Type[LogEventProtocol]
    DATABASE_ERROR: Type[LogEventProtocol]
    BACKTRACE: Type[LogEventProtocol]


MILLI_RE = re.compile(r"(\d+) ms")


# pylint: disable=too-few-public-methods
class ReactorStalledMixin(Generic[T_log_event]):
    tolerable_reactor_stall: int = TOLERABLE_REACTOR_STALL

    def add_info(self: T_log_event, node, line: str, line_number: int) -> T_log_event:
        try:
            # Dynamically handle reactor stalls severity.
            if int(MILLI_RE.findall(line)[0]) >= self.tolerable_reactor_stall:
                self.severity = Severity.ERROR
        except (ValueError, IndexError, ):
            LOGGER.warning("failed to read REACTOR_STALLED line=[%s] ", line)
        return super().add_info(node=node, line=line, line_number=line_number)


DatabaseLogEvent.add_subevent_type("WARNING", severity=Severity.WARNING,
                                   regex=r"(^WARNING|!\s*?WARNING).*\[shard.*\]")
DatabaseLogEvent.add_subevent_type("NO_SPACE_ERROR", severity=Severity.ERROR,
                                   regex="No space left on device")
DatabaseLogEvent.add_subevent_type("UNKNOWN_VERB", severity=Severity.WARNING,
                                   regex="(unknown verb exception|unknown_verb_error)")
DatabaseLogEvent.add_subevent_type("CLIENT_DISCONNECT", severity=Severity.WARNING,
                                   regex=r"cql_server - exception while processing connection:")
DatabaseLogEvent.add_subevent_type("SEMAPHORE_TIME_OUT", severity=Severity.WARNING,
                                   regex="semaphore_timed_out")
# The below ldap-connection-reset is dependent on https://github.com/scylladb/scylla-enterprise/issues/2710
DatabaseLogEvent.add_subevent_type("LDAP_CONNECTION_RESET", severity=Severity.WARNING,
                                   regex=r".*ldap_connection - Seastar read failed: std::system_error \(error system:104, "
                                         r"recv: Connection reset by peer\).*")
# This scylla WARNING includes "exception" word and reported as ERROR. To prevent it I add the subevent below and locate
# it before DATABASE_ERROR. Message example:
# storage_proxy - Failed to apply mutation from 10.0.2.108#8: exceptions::mutation_write_timeout_exception
# (Operation timed out for system.paxos - received only 0 responses from 1 CL=ONE.)
DatabaseLogEvent.add_subevent_type("SYSTEM_PAXOS_TIMEOUT", severity=Severity.WARNING,
                                   regex="(mutation_write_|Operation timed out for system.paxos|"
                                         "Operation failed for system.paxos)")
DatabaseLogEvent.add_subevent_type("SERVICE_LEVEL_CONTROLLER", severity=Severity.WARNING,
                                   regex="Operation timed out for system_distributed.service_levels")
DatabaseLogEvent.add_subevent_type("GATE_CLOSED", severity=Severity.WARNING,
                                   regex="exception \"gate closed\" in no_wait handler ignored")
DatabaseLogEvent.add_subevent_type("RESTARTED_DUE_TO_TIME_OUT", severity=Severity.WARNING,
                                   regex="scylla-server.service.*State 'stop-sigterm' timed out")
DatabaseLogEvent.add_subevent_type("EMPTY_NESTED_EXCEPTION", severity=Severity.WARNING,
                                   regex=r"cql_server - exception while processing connection: "
                                         r"seastar::nested_exception \(seastar::nested_exception\)$")
DatabaseLogEvent.add_subevent_type("COMPACTION_STOPPED", severity=Severity.NORMAL,
                                   regex="compaction_stopped_exception")
DatabaseLogEvent.add_subevent_type("BAD_ALLOC", severity=Severity.ERROR,
                                   regex="std::bad_alloc")
DatabaseLogEvent.add_subevent_type("SCHEMA_FAILURE", severity=Severity.ERROR,
                                   regex="Failed to load schema version")
DatabaseLogEvent.add_subevent_type("RUNTIME_ERROR", severity=Severity.ERROR,
                                   regex="std::runtime_error")
# remove below workaround after dropping support for Scylla 2023.1 and 5.2 (see scylladb/scylla#13538)
DatabaseLogEvent.add_subevent_type("DIRECTORY_NOT_EMPTY", severity=Severity.NORMAL,
                                   regex="remove failed: Directory not empty")
DatabaseLogEvent.add_subevent_type("FILESYSTEM_ERROR", severity=Severity.ERROR,
                                   regex="filesystem_error")
DatabaseLogEvent.add_subevent_type("DISK_ERROR", severity=Severity.ERROR,
                                   regex=r"storage_service - .*due to I\/O errors.*Disk error: std::system_error")
DatabaseLogEvent.add_subevent_type("STACKTRACE", severity=Severity.ERROR,
                                   regex=r'^(?!.*libabsl).*stacktrace')
# scylladb/scylladb#12972
DatabaseLogEvent.add_subevent_type("RAFT_TRANSFER_SNAPSHOT_ERROR", severity=Severity.WARNING,
                                   regex=r"raft - \[[\w-]*\] Transferring snapshot to [\w-]* "
                                         r"failed with: seastar::rpc::remote_verb_error \(connection is closed\)")

# REACTOR_STALLED must be above BACKTRACE as it has "Backtrace" in its message
DatabaseLogEvent.add_subevent_type("REACTOR_STALLED", mixin=ReactorStalledMixin, severity=Severity.DEBUG,
                                   regex="Reactor stalled")
DatabaseLogEvent.add_subevent_type("KERNEL_CALLSTACK", severity=Severity.DEBUG,
                                   regex="kernel callstack: 0x.{16}")
DatabaseLogEvent.add_subevent_type("ABORTING_ON_SHARD", severity=Severity.ERROR,
                                   regex="Aborting on shard")
DatabaseLogEvent.add_subevent_type("SEGMENTATION", severity=Severity.ERROR,
                                   regex="segmentation")
DatabaseLogEvent.add_subevent_type("CORRUPTED_SSTABLE", severity=Severity.CRITICAL,
                                   regex="sstables::malformed_sstable_exception|invalid_mutation_fragment_stream")
DatabaseLogEvent.add_subevent_type("INTEGRITY_CHECK", severity=Severity.ERROR,
                                   regex="integrity check failed")
DatabaseLogEvent.add_subevent_type("SUPPRESSED_MESSAGES", severity=Severity.WARNING,
                                   regex="journal: Suppressed")
DatabaseLogEvent.add_subevent_type("stream_exception", severity=Severity.ERROR,
                                   regex="stream_exception")
DatabaseLogEvent.add_subevent_type("RPC_CONNECTION", severity=Severity.WARNING,
                                   regex=r'(^ERROR|!ERR).*rpc - client .*(connection dropped|fail to connect)')
DatabaseLogEvent.add_subevent_type("DATABASE_ERROR", severity=Severity.ERROR,
                                   regex=r"(^ERROR|!\s*?ERR).*\[shard.*\]")
DatabaseLogEvent.add_subevent_type("BACKTRACE", severity=Severity.ERROR,
                                   regex="^(?!.*audit:).*backtrace")
SYSTEM_ERROR_EVENTS = (
    DatabaseLogEvent.WARNING(),
    DatabaseLogEvent.NO_SPACE_ERROR(),
    DatabaseLogEvent.UNKNOWN_VERB(),
    DatabaseLogEvent.CLIENT_DISCONNECT(),
    DatabaseLogEvent.SEMAPHORE_TIME_OUT(),
    DatabaseLogEvent.LDAP_CONNECTION_RESET(),
    DatabaseLogEvent.SYSTEM_PAXOS_TIMEOUT(),
    DatabaseLogEvent.SERVICE_LEVEL_CONTROLLER(),
    DatabaseLogEvent.GATE_CLOSED(),
    DatabaseLogEvent.RESTARTED_DUE_TO_TIME_OUT(),
    DatabaseLogEvent.EMPTY_NESTED_EXCEPTION(),
    DatabaseLogEvent.COMPACTION_STOPPED(),
    DatabaseLogEvent.BAD_ALLOC(),
    DatabaseLogEvent.SCHEMA_FAILURE(),
    DatabaseLogEvent.RUNTIME_ERROR(),
    DatabaseLogEvent.DIRECTORY_NOT_EMPTY(),
    DatabaseLogEvent.FILESYSTEM_ERROR(),
    DatabaseLogEvent.DISK_ERROR(),
    DatabaseLogEvent.STACKTRACE(),
    DatabaseLogEvent.RAFT_TRANSFER_SNAPSHOT_ERROR(),

    # REACTOR_STALLED must be above BACKTRACE as it has "Backtrace" in its message
    DatabaseLogEvent.REACTOR_STALLED(),
    DatabaseLogEvent.KERNEL_CALLSTACK(),

    DatabaseLogEvent.ABORTING_ON_SHARD(),
    DatabaseLogEvent.SEGMENTATION(),
    DatabaseLogEvent.CORRUPTED_SSTABLE(),
    DatabaseLogEvent.INTEGRITY_CHECK(),
    DatabaseLogEvent.SUPPRESSED_MESSAGES(),
    DatabaseLogEvent.stream_exception(),
    DatabaseLogEvent.RPC_CONNECTION(),
    DatabaseLogEvent.DATABASE_ERROR(),
    DatabaseLogEvent.BACKTRACE(),
)
SYSTEM_ERROR_EVENTS_PATTERNS: List[Tuple[re.Pattern, LogEventProtocol]] = \
    [(re.compile(event.regex, re.IGNORECASE), event) for event in SYSTEM_ERROR_EVENTS]

# BACKTRACE_RE should match those:
# 2022-03-05T08:33:48+00:00 rolling-*-0-1 !    INFO |   /opt/scylladb/libreloc/libc.so.6+0x35a15
# 2022-03-05T08:33:48+00:00 rolling-*-0-1 !    INFO |   0x2e8653d
BACKTRACE_RE = re.compile(r'(?P<other_bt>/lib.*?\+0x[0-9a-f]*$)|'
                          r'(?P<scylla_bt>0x[0-9a-f]*$)', re.IGNORECASE)


class ScyllaHelpErrorEvent(SctEvent, abstract=True):
    duplicate: Type[SctEventProtocol]
    filtered: Type[SctEventProtocol]
    message: str

    def __init__(self, message: Optional[str] = None, severity=Severity.WARNING):
        super().__init__(severity=severity)

        # Don't include `message' to the state if it's None.
        if message is not None:
            self.message = message

    @property
    def msgfmt(self):
        fmt = super().msgfmt + ": type={0.type}"
        if hasattr(self, "message"):
            fmt += " message={0.message}"
        return fmt


ScyllaHelpErrorEvent.add_subevent_type("duplicate")
ScyllaHelpErrorEvent.add_subevent_type("filtered")


class ScyllaHousekeepingServiceEvent(InformationalEvent):
    def __init__(self, message: str, severity: Severity = Severity.NORMAL):
        super().__init__(severity=severity)

        self.message = message

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": message={0.message}"


class IndexSpecialColumnErrorEvent(InformationalEvent):
    def __init__(self, message: str, severity: Severity = Severity.ERROR):
        super().__init__(severity=severity)

        self.message = message

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": message={0.message}"


class ScyllaServerEventPattern(NamedTuple):
    pattern: Pattern
    period_func: Callable


class ScyllaServerEventPatternFuncs(NamedTuple):
    pattern: Pattern
    event_class: Type[ContinuousEvent]
    period_func: Callable


class ScyllaDatabaseContinuousEvent(ContinuousEvent, abstract=True):
    begin_pattern: str = NotImplemented
    end_pattern: str = NotImplemented
    continuous_hash_fields = ('node', 'shard')

    def __init__(self, node: str, shard: int = None, severity=Severity.UNKNOWN, publish_event=True):
        self.node = node
        self.shard = shard
        super().__init__(severity=severity, publish_event=publish_event)

    @property
    def msgfmt(self):
        node = " node={0.node}" if self.node else ""
        shard = " shard={0.shard}" if self.shard is not None else ""
        fmt = f"{super().msgfmt}{node}{shard}"

        return fmt


class ScyllaServerStatusEvent(ScyllaDatabaseContinuousEvent):
    begin_pattern = r'Starting Scylla Server'
    end_pattern = r'Stopping Scylla Server|Failed to start Scylla Server'

    def __init__(self, node: str, severity=Severity.NORMAL, **__):
        super().__init__(node=node, severity=severity)


class BootstrapEvent(ScyllaDatabaseContinuousEvent):
    begin_pattern = r'Starting to bootstrap'
    end_pattern = r'Bootstrap succeeded'

    def __init__(self, node: str, severity=Severity.NORMAL, **__):
        super().__init__(node=node, severity=severity)


class FullScanEvent(ScyllaDatabaseContinuousEvent):
    def __init__(self, node: str, ks_cf: str, message: Optional[str] = None, severity=Severity.NORMAL,
                 **kwargs):
        self.ks_cf = ks_cf
        self.message = message
        self.user = kwargs.get("user", None)
        self.password = kwargs.get("password", None)
        super().__init__(node=node, severity=severity)

    @property
    def msgfmt(self):
        fmt = super().msgfmt + " select_from={0.ks_cf}"
        if self.message:
            fmt += " message={0.message}"
        if self.user:
            fmt += " user={0.user}"
        if self.password:
            fmt += " password={0.password}"
        return fmt


class FullPartitionScanReversedOrderEvent(ScyllaDatabaseContinuousEvent):
    def __init__(self, node: str, ks_cf: str, message: Optional[str] = None, severity=Severity.NORMAL, **__):
        super().__init__(node=node, severity=severity)
        self.ks_cf = ks_cf
        self.message = message

    @property
    def msgfmt(self):
        fmt = super().msgfmt + " select_from={0.ks_cf}"
        if self.message:
            fmt += " message={0.message}"
        return fmt


class FullPartitionScanEvent(ScyllaDatabaseContinuousEvent):
    def __init__(self, node: str, ks_cf: str, message: Optional[str] = None, severity=Severity.NORMAL, **__):
        super().__init__(node=node, severity=severity)
        self.ks_cf = ks_cf
        self.message = message

    @property
    def msgfmt(self):
        fmt = super().msgfmt + " select_from={0.ks_cf}"
        if self.message:
            fmt += " message={0.message}"
        return fmt


class TombstoneGcVerificationEvent(ScyllaDatabaseContinuousEvent):
    def __init__(self, node: str, ks_cf: str, message: Optional[str] = None, severity=Severity.NORMAL, **__):
        super().__init__(node=node, severity=severity)
        self.ks_cf = ks_cf
        self.message = message

    @property
    def msgfmt(self):
        fmt = super().msgfmt + " select_from={0.ks_cf}"
        if self.message:
            fmt += " message={0.message}"
        return fmt


class FullScanAggregateEvent(ScyllaDatabaseContinuousEvent):
    def __init__(self, node: str, ks_cf: str, message: str | None = None, severity=Severity.NORMAL, **__):
        super().__init__(node=node, severity=severity)
        self.ks_cf = ks_cf
        self.message = message

    @property
    def msgfmt(self):
        fmt = super().msgfmt + " select_from={0.ks_cf}"
        if self.message:
            fmt += " message={0.message}"
        return fmt


class RepairEvent(ScyllaDatabaseContinuousEvent):
    begin_pattern = r'Repair 1 out of \d+ ranges, id=\[id=\d+, uuid=(?P<uuid>[\d\w-]{36})\w*\], shard=(?P<shard>\d+)'
    end_pattern = r'repair id \[id=\d+, uuid=(?P<uuid>[\d\w-]{36})\w*\] on shard (?P<shard>\d+) completed'
    publish_to_grafana = False
    save_to_files = False
    continuous_hash_fields = ('node', 'shard', 'uuid')

    def __init__(self, node: str, shard: int, uuid: str, severity=Severity.NORMAL, **__):
        self.uuid = uuid
        super().__init__(node=node, shard=shard, severity=severity)
        self.log_level = logging.DEBUG


class CompactionEvent(ScyllaDatabaseContinuousEvent):
    begin_pattern = r'\[shard (?P<shard>\d+)\] compaction - \[Compact (?P<table>\w+.\w+) ' \
                    r'(?P<compaction_process_id>.+)\] Compacting '
    end_pattern = r'\[shard (?P<shard>\d+)\] compaction - \[Compact (?P<table>\w+.\w+) ' \
                  r'(?P<compaction_process_id>.+)\] Compacted '
    publish_to_grafana = False
    save_to_files = False
    continuous_hash_fields = ('node', 'shard', 'table', 'compaction_process_id')

    def __init__(self, node: str, shard: int, table: str, compaction_process_id: str,  # pylint: disable=too-many-arguments
                 severity=Severity.NORMAL, **__):
        self.table = table
        self.compaction_process_id = compaction_process_id
        super().__init__(node=node, shard=shard, severity=severity)
        self.log_level = logging.DEBUG

    @property
    def msgfmt(self):
        table = " table={0.table}" if self.table is not None else ""
        compaction_process_id = " compaction_process_id={0.compaction_process_id}" \
            if self.compaction_process_id is not None else ""
        fmt = f"{super().msgfmt}{table}{compaction_process_id}"

        return fmt


class JMXServiceEvent(ScyllaDatabaseContinuousEvent):
    begin_pattern = r'Starting the JMX server'
    end_pattern = r'Stopped Scylla JMX'

    def __init__(self, node: str, severity=Severity.NORMAL, **__):
        super().__init__(node=node, severity=severity)


class ScyllaSysconfigSetupEvent(ScyllaDatabaseContinuousEvent):
    begin_pattern = r'Move current config files before running scylla_sysconfig setup'
    end_pattern = r'Finished running scylla_sysconfig_setup'

    def __init__(self, node: str, severity=Severity.NORMAL, **__):
        super().__init__(node=node, severity=severity, publish_event=True)


class ScyllaYamlUpdateEvent(InformationalEvent):
    def __init__(self, node_name: str, message: Optional[str] = None, diff: dict | None = None,
                 severity=Severity.NORMAL, **__):
        super().__init__(severity=severity)
        self.message = message or f"Updating scylla.yaml contents on node: {node_name}. Diff: {diff}"

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": message={0.message}"


class DataOperationsEvent(ContinuousEvent):
    def __init__(self, message: Optional[str] = None, severity=Severity.NORMAL, publish_event=True):
        self.message = message
        super().__init__(publish_event=publish_event, severity=severity)

    @property
    def msgfmt(self):
        fmt = super().msgfmt
        if self.message:
            fmt += ": message={0.message}"
        return fmt


class DataOperationsCaseEvent(ContinuousEvent):
    def __init__(self, case: str, message: Optional[str] = None, severity=Severity.NORMAL, publish_event=True):
        self.case = case
        self.message = message
        super().__init__(publish_event=publish_event, severity=severity)

    @property
    def msgfmt(self):
        fmt = super().msgfmt + ": case={0.case}"
        if self.message:
            fmt += " message={0.message}"
        return fmt


SCYLLA_DATABASE_CONTINUOUS_EVENTS = [
    ScyllaServerStatusEvent,
    BootstrapEvent,
    JMXServiceEvent,
    ScyllaSysconfigSetupEvent
]


def get_pattern_to_event_to_func_mapping(node: str) \
        -> List[ScyllaServerEventPatternFuncs]:
    """
    This function maps regex patterns, event classes and begin / end
    functions into ScyllaServerEventPatternFuncs object. Helper
    functions are delegated to find the event that should be the
    target of the start / stop action, or creating a new one.
    """
    mapping = []
    event_registry = ContinuousEventsRegistry()

    def _add_event(event_type: Type[ScyllaDatabaseContinuousEvent], match: Match):
        kwargs = match.groupdict()
        if "shard" in kwargs:
            kwargs["shard"] = int(kwargs["shard"])
        event_type(node=node, **kwargs).begin_event()

    def _end_event(event_type: Type[ScyllaDatabaseContinuousEvent], match: Match):
        kwargs = match.groupdict()
        continuous_hash = event_type.get_continuous_hash_from_dict({'node': node, **kwargs})
        if begin_event := event_registry.find_continuous_events_by_hash(continuous_hash):
            begin_event[-1].end_event()
            return
        TestFrameworkEvent(
            source=event_type.__name__,
            message=f"Did not find events of type {event_type} with hash {continuous_hash} ({kwargs})"
                    f" with period type {EventPeriod.BEGIN.value}",
            severity=Severity.DEBUG
        ).publish_or_dump()

    for event in SCYLLA_DATABASE_CONTINUOUS_EVENTS:
        mapping.append(ScyllaServerEventPatternFuncs(pattern=re.compile(event.begin_pattern),
                                                     event_class=event,
                                                     period_func=partial(_add_event, event_type=event)))
        mapping.append(ScyllaServerEventPatternFuncs(pattern=re.compile(event.end_pattern), event_class=event,
                                                     period_func=partial(_end_event, event_type=event)))

    return mapping
