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
import json
import time
import logging
from typing import Type, Optional, List, Tuple, Any

import dateutil.parser
from invoke.runners import Result

from sdcm.sct_events import Severity, SctEventProtocol
from sdcm.sct_events.base import LogEvent, LogEventProtocol, T_log_event, SctEvent
from sdcm.sct_events.stress_events import BaseStressEvent, StressEvent, StressEventProtocol

LOGGER = logging.getLogger(__name__)


class GeminiStressEvent(BaseStressEvent):
    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(self, node: Any,
                 cmd: str,
                 log_file_name: Optional[str] = None,
                 severity: Severity = Severity.NORMAL,
                 publish_event: bool = True):
        self.node = str(node)
        self.cmd = cmd
        self.log_file_name = log_file_name
        self.result = ""
        super().__init__(severity=severity, publish_event=publish_event)

    def add_result(self, result: Optional[Result]):
        if result != "":
            self.result += f"Exit code: {result.exited}\n"
            if result.stdout:
                self.result += f"Command output: {result.stdout.strip().splitlines()[-2:]}\n"
            if result.stderr:
                self.add_error([f"Command error: {result.stderr}\n"])

    @property
    def msgfmt(self):
        fmt = super().msgfmt + ":"
        if self.type:
            fmt += " type={0.type}"
        fmt += " node={0.node}"

        fmt += " gemini_cmd={0.cmd}"

        if self.result:
            fmt += "\nresult={0.result}"

        if self.errors:
            fmt += "\nerrors={0.errors}"

        return fmt


class HDRFileMissed(SctEvent):
    def __init__(self, message: str, severity=Severity.NORMAL):
        super().__init__(severity=severity)

        self.message = message


class CassandraStressEvent(StressEvent):
    ...


class ScyllaBenchEvent(StressEvent):
    ...


class LatteStressEvent(StressEvent):
    ...


class CassandraHarryEvent(StressEvent, abstract=True):
    failure: Type[StressEventProtocol]
    error: Type[SctEventProtocol]
    timeout: Type[StressEventProtocol]
    start: Type[StressEventProtocol]
    finish: Type[StressEventProtocol]

    @property
    def msgfmt(self):
        fmt = super(StressEvent, self).msgfmt + ": type={0.type} node={0.node} stress_cmd={0.stress_cmd}"
        if self.errors:
            return fmt + " error={0.errors_formatted}"
        return fmt


CassandraHarryEvent.add_stress_subevents(failure=Severity.CRITICAL, error=Severity.ERROR, timeout=Severity.ERROR)


class BaseYcsbStressEvent(StressEvent, abstract=True):
    pass


class YcsbStressEvent(BaseYcsbStressEvent, abstract=True):
    failure: Type[StressEventProtocol]
    error: Type[StressEventProtocol]
    start: Type[StressEventProtocol]
    finish: Type[StressEventProtocol]


YcsbStressEvent.add_stress_subevents(failure=Severity.CRITICAL, error=Severity.ERROR)


class CDCReaderStressEvent(BaseYcsbStressEvent, abstract=True):
    failure: Type[StressEventProtocol]
    error: Type[StressEventProtocol]
    start: Type[StressEventProtocol]
    finish: Type[StressEventProtocol]


CDCReaderStressEvent.add_stress_subevents(failure=Severity.CRITICAL, error=Severity.ERROR)


class NdBenchStressEvent(StressEvent, abstract=True):
    failure: Type[StressEventProtocol]
    error: Type[StressEventProtocol]
    start: Type[StressEventProtocol]
    finish: Type[StressEventProtocol]


NdBenchStressEvent.add_stress_subevents(start=Severity.NORMAL,
                                        finish=Severity.NORMAL,
                                        error=Severity.ERROR,
                                        failure=Severity.CRITICAL)


class NdBenchErrorEvent(LogEvent, abstract=True):
    Error: Type[LogEventProtocol]
    Failure: Type[LogEventProtocol]


NdBenchErrorEvent.add_subevent_type("Error", severity=Severity.ERROR, regex=r"ERROR")
NdBenchErrorEvent.add_subevent_type("Failure", severity=Severity.CRITICAL, regex=r"FAILURE|FAILED")


NDBENCH_ERROR_EVENTS = (
    NdBenchErrorEvent.Failure(),
    NdBenchErrorEvent.Error()
)

NDBENCH_ERROR_EVENTS_PATTERNS = [(re.compile(event.regex), event) for event in NDBENCH_ERROR_EVENTS]


class KclStressEvent(StressEvent, abstract=True):
    failure: Type[StressEventProtocol]
    start: Type[StressEventProtocol]
    finish: Type[StressEventProtocol]


KclStressEvent.add_stress_subevents(failure=Severity.ERROR)


class NoSQLBenchStressEvent(StressEvent):
    ...


class CassandraStressLogEvent(LogEvent, abstract=True):
    IOException: Type[LogEventProtocol]
    ConsistencyError: Type[LogEventProtocol]
    OperationOnKey: Type[LogEventProtocol]
    TooManyHintsInFlight: Type[LogEventProtocol]
    ShardAwareDriver: Type[LogEventProtocol]
    SchemaDisagreement: Type[LogEventProtocol]


class SchemaDisagreementErrorEvent(SctEvent):
    """Thrown when schema mismatch is detected with collected data for bug report. """

    def __init__(self, severity: Severity = Severity.UNKNOWN):
        super().__init__(severity=severity)
        self.sstable_links: list[str] = []
        self.gossip_info: dict = {}
        self.peers_info: dict = {}

    def add_sstable_link(self, link):
        self.sstable_links.append(link)

    def add_gossip_info(self, info):
        self.gossip_info = json.dumps({node.name: values for node, values in info.items()}, indent=2)

    def add_peers_info(self, info):
        self.peers_info = json.dumps({node.name: values for node, values in info.items()}, indent=2)

    @property
    def sstable_links_formatted(self):
        return "\n".join(self.sstable_links) if self.sstable_links else ""

    @property
    def msgfmt(self):
        fmt = super().msgfmt + ":"
        fmt += "\ncollected sstables:\n{0.sstable_links_formatted}"
        fmt += "\ngossip info:\n{0.gossip_info}"
        fmt += "\npeers info:\n{0.peers_info}"
        return fmt


# Task: https://trello.com/c/kGply3WI/2718-stress-failure-should-stop-the-test-immediately
CassandraStressLogEvent.add_subevent_type("OperationOnKey", severity=Severity.CRITICAL,
                                          regex=r"Operation x10 on key\(s\) \[")
# TODO: change TooManyHintsInFlight severity to CRITICAL, when we have more stable hinted handoff
# TODO: backpressure mechanism.
CassandraStressLogEvent.add_subevent_type("TooManyHintsInFlight", severity=Severity.ERROR,
                                          regex=r"Too many hints in flight|Too many in flight hints")

CassandraStressLogEvent.add_subevent_type("IOException", severity=Severity.ERROR,
                                          regex=r"java\.io\.IOException")
CassandraStressLogEvent.add_subevent_type("ConsistencyError", severity=Severity.ERROR,
                                          regex="Cannot achieve consistency level")
CassandraStressLogEvent.add_subevent_type("ShardAwareDriver", severity=Severity.NORMAL,
                                          regex="Using optimized driver")
CassandraStressLogEvent.add_subevent_type("SchemaDisagreement", severity=Severity.WARNING,
                                          regex="No schema agreement")


CS_ERROR_EVENTS = (
    CassandraStressLogEvent.TooManyHintsInFlight(),
    CassandraStressLogEvent.OperationOnKey(),
    CassandraStressLogEvent.IOException(),
    CassandraStressLogEvent.ConsistencyError(),
    CassandraStressLogEvent.SchemaDisagreement(),
)
CS_NORMAL_EVENTS = (CassandraStressLogEvent.ShardAwareDriver(), )

CS_ERROR_EVENTS_PATTERNS: List[Tuple[re.Pattern, LogEventProtocol]] = \
    [(re.compile(event.regex), event) for event in CS_ERROR_EVENTS]

CS_NORMAL_EVENTS_PATTERNS: List[Tuple[re.Pattern, LogEventProtocol]] = \
    [(re.compile(event.regex), event) for event in CS_NORMAL_EVENTS]


class ScyllaBenchLogEvent(LogEvent, abstract=True):
    ConsistencyError: Type[LogEventProtocol]
    DataValidationError: Type[LogEventProtocol]
    ParseDistributionError: Type[LogEventProtocol]


ScyllaBenchLogEvent.add_subevent_type("ConsistencyError", severity=Severity.ERROR, regex=r"received only")
# Scylla-bench data validation was added by https://github.com/scylladb/scylla-bench/commit/3eb53d8ce11e5ad26062bcc662edb31dda521ccf
ScyllaBenchLogEvent.add_subevent_type("DataValidationError", severity=Severity.CRITICAL,
                                      regex=r"doesn't match |failed to validate data|failed to verify checksum|corrupt checksum or data|"
                                            r"data corruption")
ScyllaBenchLogEvent.add_subevent_type(
    "ParseDistributionError",
    severity=Severity.CRITICAL,
    regex=r"missing parameter|unexpected parameter|unsupported"
          r"|invalid input value|distribution is invalid|distribution has invalid format")


SCYLLA_BENCH_ERROR_EVENTS = (
    ScyllaBenchLogEvent.ConsistencyError(),
    ScyllaBenchLogEvent.DataValidationError(),
    ScyllaBenchLogEvent.ParseDistributionError(),
)
SCYLLA_BENCH_ERROR_EVENTS_PATTERNS: List[Tuple[re.Pattern, LogEventProtocol]] = \
    [(re.compile(event.regex), event) for event in SCYLLA_BENCH_ERROR_EVENTS]

CASSANDRA_HARRY_ERROR_EVENTS = (
)
CASSANDRA_HARRY_ERROR_EVENTS_PATTERNS: List[Tuple[re.Pattern, LogEventProtocol]] = \
    [(re.compile(event.regex), event) for event in CASSANDRA_HARRY_ERROR_EVENTS]


class GeminiStressLogEvent(LogEvent[T_log_event], abstract=True):  # pylint: disable=too-many-instance-attributes
    SEVERITY_MAPPING = {
        "INFO": "NORMAL",
        "DEBUG": "NORMAL",
        "WARN": "WARNING",
        "ERROR": "ERROR",
        "FATAL": "CRITICAL",
    }

    GeminiEvent: Type[LogEventProtocol]

    def __init__(self, verbose=False):
        super().__init__(regex="", severity=Severity.CRITICAL)
        self.verbose = verbose

    def add_info(self: T_log_event, node, line: str, line_number: int) -> T_log_event:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            if self.verbose:
                LOGGER.debug("Failed to parse a line: %s", line.rstrip())
            self._ready_to_publish = False
            return self

        try:
            self.source_timestamp = dateutil.parser.parse(data.pop("T")).timestamp()
        except ValueError:
            pass

        self.event_timestamp = time.time()
        self.severity = Severity[self.SEVERITY_MAPPING[data.pop("L")]]

        self.line = data.pop("M")
        if data:
            self.line += " (" + " ".join(f'{key}="{value}"' for key, value in data.items()) + ")"

        self.line_number = line_number
        self.node = str(node)

        self._ready_to_publish = True
        return self

    @property
    def msgfmt(self) -> str:
        fmt = super(LogEvent, self).msgfmt + ":"
        if self.type:
            fmt += " type={0.type}"
        fmt += " line_number={0.line_number} node={0.node}"

        if self.line:
            fmt += "\nline={0.line}"

        return fmt


GeminiStressLogEvent.add_subevent_type("GeminiEvent")


class NoSQLBenchStressLogEvents(LogEvent, abstract=True):
    ProgressIndicatorStoppedEvent: Type[LogEventProtocol]
    ProgressIndicatorRunningEvent: Type[LogEventProtocol]
    ProgressIndicatorFinishedEvent: Type[LogEventProtocol]


NoSQLBenchStressLogEvents.add_subevent_type(
    "ProgressIndicatorStoppedEvent",
    severity=Severity.CRITICAL,
    regex=r'\d*\s+INFO\s+\[ProgressIndicator/logonly:\ds\]\s+PROGRESS\s+\w+:\s+\d{2}.\d{2}%/Stopped.*'
)

NoSQLBenchStressLogEvents.add_subevent_type(
    "ProgressIndicatorRunningEvent",
    severity=Severity.DEBUG,
    regex=r'\d*\s+INFO\s+\[ProgressIndicator/logonly:\ds\]\s+PROGRESS\s+\w+:\s+\d{2}.\d{2}%/Running.*'
)

NoSQLBenchStressLogEvents.add_subevent_type(
    "ProgressIndicatorFinishedEvent",
    severity=Severity.NORMAL,
    regex=r'\d*\s+INFO\s+\[ProgressIndicator/logonly:\ds\]\s+PROGRESS\s+\w+:\s+100.0%/Stopped\s+.*'
)


NOSQLBENCH_LOG_EVENTS = (
    NoSQLBenchStressLogEvents.ProgressIndicatorStoppedEvent(),
    NoSQLBenchStressLogEvents.ProgressIndicatorFinishedEvent(),
    NoSQLBenchStressLogEvents.ProgressIndicatorRunningEvent()
)

NOSQLBENCH_EVENT_PATTERNS = [(re.compile(event.regex), event) for event in NOSQLBENCH_LOG_EVENTS]
