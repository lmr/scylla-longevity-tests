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
# Copyright (c) 2021 ScyllaDB

import os
import re
import uuid
import time
import logging
import concurrent.futures

from sdcm.loader import CassandraHarryStressExporter
from sdcm.prometheus import nemesis_metrics_obj
from sdcm.sct_events.loaders import CassandraHarryEvent, SCYLLA_BENCH_ERROR_EVENTS_PATTERNS
from sdcm.utils.common import FileFollowerThread, generate_random_string, convert_metric_to_ms
from sdcm.stress_thread import format_stress_cmd_error


LOGGER = logging.getLogger(__name__)


class CassandraHarryStressEventsPublisher(FileFollowerThread):
    def __init__(self, node, harry_log_filename):
        super().__init__()
        self.harry_log_filename = harry_log_filename
        self.node = str(node)

    def run(self):
        while not self.stopped():
            exists = os.path.isfile(self.harry_log_filename)
            if not exists:
                time.sleep(0.5)
                continue

            for line_number, line in enumerate(self.follow_file(self.harry_log_filename)):
                if self.stopped():
                    break

                for pattern, event in SCYLLA_BENCH_ERROR_EVENTS_PATTERNS:
                    if pattern.search(line):
                        event.add_info(node=self.node, line=line, line_number=line_number).publish()


class CassandraHarryThread:
    def __init__(self, stress_cmd, loader_set, timeout, node_list=None, round_robin=False, use_single_loader=False,
                 stop_test_on_failure=False, stress_num=1, credentials=None):
        if not node_list:
            node_list = []
        self.loader_set = loader_set
        self.stress_cmd = stress_cmd
        self.timeout = timeout
        self.use_single_loader = use_single_loader
        self.round_robin = round_robin
        self.node_list = node_list
        self.stress_num = stress_num
        if credentials and 'username=' not in self.stress_cmd:
            self.stress_cmd += " -username {} -password {}".format(*credentials)
        self.stop_test_on_failure = stop_test_on_failure

        self.executor = None
        self.results_futures = []
        self.shell_marker = generate_random_string(20)
        self.max_workers = 0

    def get_stress_results_bench(self):
        ret = []
        results = []

        LOGGER.debug('Wait for stress threads results')
        for future in concurrent.futures.as_completed(self.results_futures, timeout=self.timeout):
            results.append(future.result())

        for _, result in results:
            if not result:
                # Silently skip if stress command throwed error, since it was already reported in _run_stress
                continue
            output = result.stdout + result.stderr
            try:
                lines = output.splitlines()
                node_cs_res = self._parse_harry_summary(lines)  # pylint: disable=protected-access
                if node_cs_res:
                    ret.append(node_cs_res)
            except Exception as exc:
                CassandraHarryEvent.error(node="",
                                       stress_cmd=self.stress_cmd,
                                       errors=[f"Failed to proccess stress summary due to {exc}"]).publish()

        return ret

    def verify_results(self):
        harry_summary = []
        results = []
        errors = []

        LOGGER.debug('Wait for stress threads results')
        for future in concurrent.futures.as_completed(self.results_futures, timeout=self.timeout):
            results.append(future.result())

        for node, result in results:
            if not result:
                # Silently skip if stress command throwed error, since it was already reported in _run_stress
                continue
            output = result.stdout + result.stderr

            lines = output.splitlines()
            node_cs_res = self._parse_bench_summary(lines)  # pylint: disable=protected-access

            if node_cs_res:
                harry_summary.append(node_cs_res)

            for line in lines:
                if 'java.io.IOException' in line:
                    errors += ['%s: %s' % (node, line.strip())]

        return harry_summary, errors

    def _run_stress_harry(self, node, loader_idx, stress_cmd, node_list):

        CassandraHarryEvent.start(node=node, stress_cmd=stress_cmd).publish()
        os.makedirs(node.logdir, exist_ok=True)

        log_file_name = os.path.join(node.logdir, f'cassandra-harry-l{loader_idx}-{uuid.uuid4()}.log')
        # Select first seed node to send the scylla-bench cmds
        ips = node_list[0].private_ip_address

        with CassandraHarryStressExporter(instance_name=node.ip_address,
                                       metrics=nemesis_metrics_obj(),
                                       stress_log_filename=log_file_name,
                                       loader_idx=loader_idx), \
                CassandraHarryStressEventsPublisher(node=node, harry_log_filename=log_file_name):
            result = None
            try:
                result = node.remoter.run(
                    cmd=f"{stress_cmd} -nodes {ips}",
                    timeout=self.timeout,
                    log_file=log_file_name)
            except Exception as exc:  # pylint: disable=broad-except
                errors_str = format_stress_cmd_error(exc)
                event_type(
                    node=node,
                    stress_cmd=stress_cmd,
                    log_file_name=log_file_name,
                    errors=[errors_str, ],
                ).publish()
            else:
                CassandraHarryEvent.finish(node=node, stress_cmd=stress_cmd, log_file_name=log_file_name).publish()

        return node, result

    def run(self):
        if self.round_robin:
            loaders = [self.loader_set.get_loader()]
        else:
            loaders = self.loader_set.nodes if not self.use_single_loader else [self.loader_set.nodes[0]]
        LOGGER.debug("Round-Robin through loaders, Selected loader is {} ".format(loaders))

        self.max_workers = (os.cpu_count() or 1) * 5
        LOGGER.debug("Starting %d cassandra-stress Worker threads", self.max_workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

        for loader_idx, loader in enumerate(loaders):
            self.results_futures += [self.executor.submit(self._run_stress_harry,
                                                          *(loader, loader_idx, self.stress_cmd, self.node_list))]
            time.sleep(60)

        return self

    @staticmethod
    def _parse_harry_summary(lines):  # pylint: disable=too-many-branches
        if any(['Runner.java:255 - Completed' in line for line in lines]):
            results['status'] = 'completed'
        else:
            results['status'] = 'failed'
        return results

