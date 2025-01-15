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
import sys
import time
from typing import Any, Optional, Sequence, Type, List, Tuple
from traceback import format_stack

from sdcm.sct_events import Severity
from sdcm.sct_events.base import SctEvent, SystemEvent, InformationalEvent, LogEvent, LogEventProtocol
from sdcm.sct_events.continuous_event import ContinuousEvent


class StartupTestEvent(SystemEvent):
    def __init__(self):
        super().__init__(severity=Severity.NORMAL)


class TestTimeoutEvent(SctEvent):
    def __init__(self, start_time: float, duration: int):
        super().__init__(severity=Severity.CRITICAL)
        self.start_time = start_time
        self.duration = duration

    @property
    def msgfmt(self) -> str:
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))
        return f"{super().msgfmt}, Test started at {start_time}, reached it's timeout ({self.duration} minute)"


class HWPerforanceEvent(SctEvent):
    def __init__(self, message: str, severity: Severity = Severity.NORMAL):
        super().__init__(severity)
        self.message = message

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": message={0.message}"


class TestFrameworkEvent(InformationalEvent):  # pylint: disable=too-many-instance-attributes
    __test__ = False  # Mark this class to be not collected by pytest.

    # pylint: disable=too-many-arguments
    def __init__(self,
                 source: Any,
                 source_method: Optional = None,
                 args: Optional[Sequence] = None,
                 kwargs: Optional[dict] = None,
                 message: Optional = None,
                 exception: Optional = None,
                 trace: Optional = None,
                 severity: Optional[Severity] = None):

        if severity is None:
            severity = Severity.ERROR
        super().__init__(severity=severity)

        self.source = str(source) if source else None
        self.source_method = str(source_method) if source_method else None
        self.exception = str(exception) if exception else None
        self.message = str(message) if message else None
        self.trace = "".join(format_stack(trace)) if trace else None
        self.args = args
        self.kwargs = kwargs

    @property
    def msgfmt(self) -> str:
        fmt = super().msgfmt + ", source={0.source}"

        if self.source_method:
            args = []
            if self.args:
                args.append("args={0.args}")
            if self.kwargs:
                args.append("kwargs={0.kwargs}")
            fmt += ".{0.source_method}(" + ", ".join(args) + ")"

        if self.message:
            fmt += " message={0.message}"
        if self.exception:
            fmt += "\nexception={0.exception}"
        if self.trace:
            fmt += "\nTraceback (most recent call last):\n{0.trace}"

        return fmt


class SoftTimeoutEvent(TestFrameworkEvent):

    def __init__(self, operation: str, duration: int | float, soft_timeout: int | float):
        message = f"operation '{operation}' took {duration}s and soft-timeout was set to {soft_timeout}s"
        super().__init__(source='SoftTimeout', severity=Severity.ERROR, trace=sys._getframe().f_back, message=message)


class ElasticsearchEvent(InformationalEvent):
    def __init__(self, doc_id: str, error: str):
        super().__init__(severity=Severity.ERROR)

        self.doc_id = doc_id
        self.error = error

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": doc_id={0.doc_id} error={0.error}"


class SpotTerminationEvent(InformationalEvent):
    def __init__(self, node: Any, message: str):
        super().__init__(severity=Severity.CRITICAL)

        self.node = str(node)
        self.message = message

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": node={0.node} message={0.message}"


class ScyllaRepoEvent(InformationalEvent):
    def __init__(self, url: str, error: str):
        super().__init__(severity=Severity.WARNING)

        self.url = url
        self.error = error

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": url={0.url} error={0.error}"


class CpuNotHighEnoughEvent(InformationalEvent):
    def __init__(self, message: str, severity=Severity.ERROR):
        super().__init__(severity=severity)

        self.message = message

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": message={0.message}"


class TestStepEvent(ContinuousEvent):

    def __init__(self,
                 step,
                 severity=Severity.NORMAL,
                 publish_event=True):  # pylint: disable=redefined-builtin,too-many-arguments
        self.step = step
        self.duration = None
        self.full_traceback = ""
        super().__init__(severity=severity, publish_event=publish_event)

    @property
    def msgfmt(self) -> str:
        fmt = super().msgfmt + ": step={0.step} "
        if self.errors:
            fmt += " errors={0.errors_formatted}"
        if self.full_traceback:
            fmt += "\n{0.full_traceback}"
        return fmt


class PerftuneResultEvent(InformationalEvent):
    def __init__(self, message: str, severity: Severity, trace: Optional = None):
        super().__init__(severity=severity)

        self.message = message
        self.trace = "".join(format_stack(trace)) if trace else None

    @property
    def msgfmt(self) -> str:
        fmt = super().msgfmt + ": message={0.message}"
        if self.trace:
            fmt += "\nTraceback (most recent call last):\n{0.trace}"
        return fmt


class InfoEvent(SctEvent):
    def __init__(self, message: str, severity=Severity.NORMAL):
        super().__init__(severity=severity)

        self.message = message

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": message={0.message}"


class ThreadFailedEvent(InformationalEvent):
    def __init__(self, message: str, traceback: Any):
        super().__init__(severity=Severity.ERROR)

        self.message = message
        self.traceback = str(traceback)

    @property
    def msgfmt(self) -> str:
        return super().msgfmt + ": message={0.message}\n{0.traceback}"


class AwsKmsEvent(ThreadFailedEvent):
    ...


class CoreDumpEvent(InformationalEvent):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 node: Any,
                 corefile_url: str,
                 backtrace: str,
                 download_instructions: str,
                 source_timestamp: Optional[float] = None):

        super().__init__(severity=Severity.ERROR)

        self.node = str(node)
        self.corefile_url = corefile_url
        self.backtrace = backtrace
        self.download_instructions = download_instructions
        if source_timestamp is not None:
            self.source_timestamp = source_timestamp

    @property
    def msgfmt(self) -> str:
        fmt = super().msgfmt + " "

        if self.node:
            fmt += "node={0.node}\n"
        if self.corefile_url:
            fmt += "corefile_url={0.corefile_url}\n"
        if self.backtrace:
            fmt += "backtrace={0.backtrace}\n"
            fmt += "Info about modules can be found in SCT logs by search for 'Coredump Modules info'\n"
        if self.download_instructions:
            fmt += "download_instructions:\n{0.download_instructions}\n"

        return fmt


class TestResultEvent(InformationalEvent, Exception):
    """An event that is published and raised at the end of the test.

    It holds and displays all errors of the tests and framework happened.
    """
    __test__ = False  # Mark this class to be not collected by pytest.

    _marker_width = 80
    _head = f"{' TEST RESULTS ':=^{_marker_width}}"
    _ending = "=" * _marker_width

    def __init__(self, test_status: str, events: dict, event_timestamp: Optional[float] = None):
        self._ok = test_status == "SUCCESS"
        super().__init__(severity=Severity.NORMAL if self._ok else Severity.ERROR)

        self.test_status = test_status
        self.events = events

        # Need to restore the event_timestamp on unpickling.  See `__reduce__()' also.
        if event_timestamp is not None:
            self.event_timestamp = event_timestamp

        # We don't publish this event.  Suppress warning about unpublished event on exit.
        self._ready_to_publish = False

    @property
    def events_formatted(self) -> str:
        result = []
        for event_group, events in self.events.items():
            if not events:
                continue
            result.append(f"""{f'{"":-<5} LAST {event_group} EVENT ':-<{self._marker_width - 2}}""")
            result.extend(events)
        return "\n".join(result)

    @property
    def msgfmt(self) -> str:
        if self._ok:
            return "{0._head}\n{0._ending}\nSUCCESS :)\n"
        return "{0._head}\n\n{0.events_formatted}\n{0._ending}\n{0.test_status} :(\n"

    def __reduce__(self):
        """Need to define it for pickling because of Exception class in MRO."""

        return type(self), (self.test_status, self.events, self.event_timestamp)


class InstanceStatusEvent(LogEvent, abstract=True):
    STARTUP: Type[LogEventProtocol]
    REBOOT: Type[LogEventProtocol]
    POWER_OFF: Type[LogEventProtocol]


InstanceStatusEvent.add_subevent_type("STARTUP", severity=Severity.WARNING, regex="kernel: Linux version")
InstanceStatusEvent.add_subevent_type("REBOOT", severity=Severity.WARNING,
                                      regex="Stopped target Host and Network Name Lookups")
InstanceStatusEvent.add_subevent_type("POWER_OFF", severity=Severity.WARNING, regex="Reached target Power-Off")

INSTANCE_STATUS_EVENTS = (
    InstanceStatusEvent.STARTUP(),
    InstanceStatusEvent.REBOOT(),
    InstanceStatusEvent.POWER_OFF(),
)

INSTANCE_STATUS_EVENTS_PATTERNS: List[Tuple[re.Pattern, LogEventProtocol]] = \
    [(re.compile(event.regex, re.IGNORECASE), event) for event in INSTANCE_STATUS_EVENTS]
