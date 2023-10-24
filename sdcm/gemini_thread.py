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

import logging
import os
import re
import uuid
import random
import json
import time

from sdcm.sct_events import Severity
from sdcm.utils.common import FileFollowerThread
from sdcm.sct_events.loaders import GeminiStressEvent, GeminiStressLogEvent
from sdcm.stress_thread import DockerBasedStressThread
from sdcm.utils.docker_remote import RemoteDocker


LOGGER = logging.getLogger(__name__)


class NotGeminiErrorResult:  # pylint: disable=too-few-public-methods
    def __init__(self, error):
        self.exited = 1
        self.stdout = "n/a"
        self.stderr = str(error)


class GeminiEventsPublisher(FileFollowerThread):
    def __init__(self, node, gemini_log_filename, verbose=False, event_id=None):
        super().__init__()
        self.gemini_log_filename = gemini_log_filename
        self.node = str(node)
        self.verbose = verbose
        self.event_id = event_id

    def run(self):
        while not self.stopped():
            if not os.path.isfile(self.gemini_log_filename):
                time.sleep(0.5)
                continue
            for line_number, line in enumerate(self.follow_file(self.gemini_log_filename), start=1):
                gemini_event = GeminiStressLogEvent.GeminiEvent(verbose=self.verbose)
                gemini_event.add_info(node=self.node, line=line, line_number=line_number)
                gemini_event.event_id = self.event_id
                gemini_event.publish(warn_not_ready=False)

                if self.stopped():
                    break


class GeminiStressThread(DockerBasedStressThread):  # pylint: disable=too-many-instance-attributes

    DOCKER_IMAGE_PARAM_NAME = "stress_image.gemini"

    def __init__(self, test_cluster, oracle_cluster, loaders, stress_cmd, timeout=None, params=None):  # pylint: disable=too-many-arguments
        super().__init__(loader_set=loaders, stress_cmd=stress_cmd, timeout=timeout, params=params)
        self.test_cluster = test_cluster
        self.oracle_cluster = oracle_cluster
        self._gemini_result_file = None
        self.gemini_commands = []
        self.gemini_request_timeout = 180
        self.gemini_connect_timeout = 120

    @property
    def gemini_result_file(self):
        if not self._gemini_result_file:
            self._gemini_result_file = os.path.join("/", "gemini_result_{}.log".format(uuid.uuid4()))
        return self._gemini_result_file

    def _generate_gemini_command(self):
        seed = self.params.get('gemini_seed')
        table_options = self.params.get('gemini_table_options')
        if not seed:
            seed = random.randint(1, 100)
        test_nodes = ",".join(self.test_cluster.get_node_cql_ips())
        oracle_nodes = ",".join(self.oracle_cluster.get_node_cql_ips()) if self.oracle_cluster else None

        cmd = "./{} --test-cluster={} --outfile {} --seed {} --request-timeout {}s --connect-timeout {}s ".format(
            self.stress_cmd.strip(),
            test_nodes,
            self.gemini_result_file,
            seed,
            self.gemini_request_timeout,
            self.gemini_connect_timeout)
        if oracle_nodes:
            cmd += "--oracle-cluster={} ".format(oracle_nodes)
        if table_options:
            cmd += " ".join([f"--table-options \"{table_opt}\"" for table_opt in table_options])
        self.gemini_commands.append(cmd)
        return cmd

    def _run_stress(self, loader, loader_idx, cpu_idx):  # pylint: disable=too-many-locals

        cpu_options = ""
        if self.stress_num > 1:
            cpu_options = f'--cpuset-cpus="{cpu_idx}"'

        docker = cleanup_context = RemoteDocker(loader, self.docker_image_name,
                                                extra_docker_opts=f'{cpu_options} --label shell_marker={self.shell_marker}'
                                                                  f' --network=host --entrypoint=""')

        if not os.path.exists(loader.logdir):
            os.makedirs(loader.logdir, exist_ok=True)
        log_file_name = os.path.join(loader.logdir, 'gemini-l%s-c%s-%s.log' %
                                     (loader_idx, cpu_idx, uuid.uuid4()))
        LOGGER.debug('gemini local log: %s', log_file_name)

        gemini_cmd = self._generate_gemini_command()
        if gemini_schema := re.search(r'--schema (.*\.json)', gemini_cmd):
            schema_file_path = gemini_schema.group(1)
            docker.send_files(schema_file_path, schema_file_path)
        with cleanup_context, \
                GeminiEventsPublisher(node=loader, gemini_log_filename=log_file_name) as publisher, \
                GeminiStressEvent(node=loader, cmd=gemini_cmd, log_file_name=log_file_name) as gemini_stress_event:
            try:
                publisher.event_id = gemini_stress_event.event_id
                gemini_stress_event.log_file_name = log_file_name
                result = docker.run(cmd=gemini_cmd,
                                    timeout=self.timeout,
                                    ignore_status=False,
                                    log_file=log_file_name,
                                    retry=0,
                                    )
                # sleep to gather all latest log messages
                time.sleep(5)
            except Exception as details:  # pylint: disable=broad-except
                LOGGER.error(details)
                result = getattr(details, "result", NotGeminiErrorResult(details))

            if result.exited:
                gemini_stress_event.add_result(result=result)
                gemini_stress_event.severity = Severity.ERROR
            else:
                if result.stderr:
                    gemini_stress_event.add_result(result=result)
                    gemini_stress_event.severity = Severity.WARNING

            local_gemini_result_file = os.path.join(docker.node.logdir, os.path.basename(self.gemini_result_file))
            results_copied = docker.receive_files(src=self.gemini_result_file, dst=local_gemini_result_file)
            assert results_copied, "gemini results aren't available, did gemini even run ?"

        return docker, result, local_gemini_result_file

    def get_gemini_results(self):
        parsed_results = []
        raw_results = self.get_results()
        for _, _, local_gemini_result_file in raw_results:
            assert local_gemini_result_file, "gemini results aren't available, did gemini even run ?"
            with open(local_gemini_result_file, encoding="utf-8") as local_file:
                content = local_file.read()
                res = self._parse_gemini_summary_json(content)
                if res:
                    parsed_results.append(res)

        return parsed_results

    @staticmethod
    def verify_gemini_results(results):

        stats = {'results': [], 'errors': {}}
        if not results:
            LOGGER.error('Gemini results are not found')
            stats['status'] = 'FAILED'
        else:
            for res in results:
                stats['results'].append(res)
                for err_type in ['write_errors', 'read_errors', 'errors']:
                    if res.get(err_type, None):
                        LOGGER.error("Gemini {} errors: {}".format(err_type, res[err_type]))
                        stats['status'] = 'FAILED'
                        stats['errors'][err_type] = res[err_type]
        if not stats.get('status'):
            stats['status'] = "PASSED"

        return stats

    @staticmethod
    def _parse_gemini_summary_json(json_str):
        results = {'result': {}}
        try:
            results = json.loads(json_str)

        except Exception as details:  # pylint: disable=broad-except
            LOGGER.error("Invalid json document {}".format(details))

        return results.get('result')

    @staticmethod
    def _parse_gemini_summary(lines):
        results = {}
        enable_parse = False

        for line in lines:
            line.strip()
            if 'Results:' in line:
                enable_parse = True
                continue
            if "run completed" in line:
                enable_parse = False
                continue
            if not enable_parse:
                continue

            split_idx = line.index(':')
            key = line[:split_idx].strip()
            value = line[split_idx + 1:].split()[0]
            results[key] = int(value)
        return results
