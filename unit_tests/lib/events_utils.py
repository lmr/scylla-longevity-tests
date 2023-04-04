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
import shutil
import tempfile
import unittest.mock
import multiprocessing as mp
from contextlib import contextmanager

from sdcm.sct_events.setup import EVENTS_DEVICE_START_DELAY, start_events_device, stop_events_device
from sdcm.sct_events.events_device import start_events_main_device, get_events_main_device
from sdcm.sct_events.file_logger import get_events_logger
from sdcm.sct_events.events_processes import EventsProcessesRegistry


class EventsUtilsMixin:
    temp_dir = None
    events_processes_registry = None
    events_processes_registry_patcher = None
    events_main_device = None

    @classmethod
    def setup_events_processes(cls, events_device: bool, events_main_device: bool, registry_patcher: bool):
        """TestConfig own copy of Events Device machinery."""

        mp.set_start_method('fork')
        cls.temp_dir = tempfile.mkdtemp()
        cls.events_processes_registry = EventsProcessesRegistry(log_dir=cls.temp_dir)
        if registry_patcher:
            cls.events_processes_registry_patcher = \
                unittest.mock.patch("sdcm.sct_events.base.SctEvent._events_processes_registry",
                                    cls.events_processes_registry)
            cls.events_processes_registry_patcher.start()
        if events_device:
            start_events_device(_registry=cls.events_processes_registry)
        elif events_main_device:
            start_events_main_device(_registry=cls.events_processes_registry)
            time.sleep(EVENTS_DEVICE_START_DELAY)
        cls.events_main_device = get_events_main_device(_registry=cls.events_processes_registry)

    @classmethod
    def teardown_events_processes(cls):
        stop_events_device(_registry=cls.events_processes_registry)
        if cls.events_processes_registry_patcher:
            cls.events_processes_registry_patcher.stop()
        shutil.rmtree(cls.temp_dir)

    @contextmanager
    def wait_for_n_events(self, subscriber, count: int, timeout: float = 1,  # pylint: disable=no-self-use
                          last_event_processing_delay: float = 1):
        last_event_n = subscriber.events_counter + count
        end_time = time.perf_counter() + timeout

        yield

        while time.perf_counter() < end_time and subscriber.events_counter < last_event_n:
            time.sleep(0.1)
        assert last_event_n <= subscriber.events_counter, \
            f"Subscriber {subscriber} didn't receive {count} events in {timeout} seconds"

        # Give a chance to the subscriber to handle last event received.
        time.sleep(last_event_processing_delay)

    @classmethod
    def get_events_logger(cls):
        return get_events_logger(_registry=cls.events_processes_registry)

    @classmethod
    def get_raw_events_log(cls):
        return get_events_main_device(_registry=cls.events_processes_registry).raw_events_log
