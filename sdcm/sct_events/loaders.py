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
from typing import Type, Optional, List, Tuple

import dateutil.parser
from invoke.runners import Result

from sdcm.sct_events import Severity
from sdcm.sct_events.base import \
    SctEvent, SctEventProtocol, LogEvent, LogEventProtocol, T_log_event, \
    BaseStressEvent, StressEvent, StressEventProtocol


LOGGER = logging.getLogger(__name__)


class GeminiEvent(BaseStressEvent, abstract=True):
    error: Type[SctEventProtocol]
    warning: Type[SctEventProtocol]
    start: Type[SctEventProtocol]
    finish: Type[SctEventProtocol]

    def __init__(self, cmd: str, result: Optional[Result] = None, severity: Severity = Severity.NORMAL):
        super().__init__(severity=severity)

        self.cmd = cmd
        self.result = ""

        if result is not None:
            self.result += f"Exit code: {result.exited}\n"
            if result.stdout:
                self.result += f"Command output: {result.stdout.strip().splitlines()[-2:]}\n"
            if result.stderr:
                self.result += f"Command error: {result.stderr}\n"

    @property
    def msgfmt(self):
        fmt = super().msgfmt + ": type={0.type} gemini_cmd={0.cmd}"
        if self.result:
            fmt += "\n{0.result}"
        return fmt


GeminiEvent.add_stress_subevents(error=Severity.CRITICAL, warning=Severity.WARNING)


class CassandraStressEvent(StressEvent, abstract=True):
    failure: Type[StressEventProtocol]
    error: Type[StressEventProtocol]
    start: Type[StressEventProtocol]
    finish: Type[StressEventProtocol]

    @property
    def msgfmt(self):
        fmt = super(StressEvent, self).msgfmt + ": type={0.type} node={0.node}\n"
        if self.errors:
            return fmt + "{0.errors_formatted}"
        return fmt + "stress_cmd={0.stress_cmd}"


CassandraStressEvent.add_stress_subevents(failure=Severity.CRITICAL, error=Severity.ERROR)


class ScyllaBenchEvent(StressEvent, abstract=True):
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


ScyllaBenchEvent.add_stress_subevents(failure=Severity.CRITICAL, error=Severity.ERROR, timeout=Severity.ERROR)


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


ScyllaBenchEvent.add_stress_subevents(failure=Severity.CRITICAL, error=Severity.ERROR, timeout=Severity.ERROR)


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


class NdbenchStressEvent(StressEvent, abstract=True):
    failure: Type[StressEventProtocol]
    error: Type[StressEventProtocol]
    start: Type[StressEventProtocol]
    finish: Type[StressEventProtocol]


NdbenchStressEvent.add_stress_subevents(failure=Severity.CRITICAL, error=Severity.ERROR)


class KclStressEvent(StressEvent, abstract=True):
    failure: Type[StressEventProtocol]
    start: Type[StressEventProtocol]
    finish: Type[StressEventProtocol]


KclStressEvent.add_stress_subevents(failure=Severity.ERROR)


class CassandraStressLogEvent(LogEvent, abstract=True):
    IOException: Type[LogEventProtocol]
    ConsistencyError: Type[LogEventProtocol]


CassandraStressLogEvent.add_subevent_type("IOException", severity=Severity.ERROR,
                                          regex=r"java\.io\.IOException")
CassandraStressLogEvent.add_subevent_type("ConsistencyError", severity=Severity.ERROR,
                                          regex="Cannot achieve consistency level")


CS_ERROR_EVENTS = (
    CassandraStressLogEvent.IOException(),
    CassandraStressLogEvent.ConsistencyError(),
)
CS_ERROR_EVENTS_PATTERNS: List[Tuple[re.Pattern, LogEventProtocol]] = \
    [(re.compile(event.regex), event) for event in CS_ERROR_EVENTS]


class ScyllaBenchLogEvent(LogEvent, abstract=True):
    ConsistencyError: Type[LogEventProtocol]


ScyllaBenchLogEvent.add_subevent_type("ConsistencyError", severity=Severity.ERROR, regex=r"received only")


SCYLLA_BENCH_ERROR_EVENTS = (
    ScyllaBenchLogEvent.ConsistencyError(),
)
SCYLLA_BENCH_ERROR_EVENTS_PATTERNS: List[Tuple[re.Pattern, LogEventProtocol]] = \
    [(re.compile(event.regex), event) for event in SCYLLA_BENCH_ERROR_EVENTS]


class GeminiLogEvent(LogEvent[T_log_event], abstract=True):
    SEVERITY_MAPPING = {
        "INFO": "NORMAL",
        "DEBUG": "NORMAL",
        "WARN": "WARNING",
        "ERROR": "ERROR",
        "FATAL": "CRITICAL",
    }

    geminievent: Type[LogEventProtocol]

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
            self.timestamp = dateutil.parser.parse(data.pop("T")).timestamp()
        except ValueError:
            self.timestamp = time.time()

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
        return SctEvent.msgfmt + ": " + "type={0.type} line_number={0.line_number} node={0.node}\n" \
                                        "{0.line}"


GeminiLogEvent.add_subevent_type("geminievent")
