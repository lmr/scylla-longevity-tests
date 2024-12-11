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
# Copyright (c) 2019 ScyllaDB

import os
import re
import time
import uuid
import logging
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
from itertools import chain
from pathlib import Path
from functools import cached_property

from sdcm.loader import CassandraStressExporter, CassandraStressHDRExporter
from sdcm.cluster import BaseLoaderSet, BaseNode
from sdcm.prometheus import nemesis_metrics_obj
from sdcm.reporting.tooling_reporter import CassandraStressVersionReporter
from sdcm.sct_events import Severity
from sdcm.utils.common import FileFollowerThread, get_data_dir_path, time_period_str_to_seconds, SoftTimeoutContext
from sdcm.utils.user_profile import get_profile_content, replace_scylla_qa_internal_path
from sdcm.sct_events.loaders import CassandraStressEvent, CS_ERROR_EVENTS_PATTERNS, CS_NORMAL_EVENTS_PATTERNS, HDRFileMissed
from sdcm.stress.base import DockerBasedStressThread
from sdcm.utils.docker_remote import RemoteDocker
from sdcm.utils.remote_logger import SSHLoggerBase


LOGGER = logging.getLogger(__name__)


class CassandraStressEventsPublisher(FileFollowerThread):
    def __init__(self, node: Any, cs_log_filename: str, event_id: str = None, stop_test_on_failure: bool = True):
        super().__init__()

        self.node = str(node)
        self.cs_log_filename = cs_log_filename
        self.event_id = event_id
        self.stop_test_on_failure = stop_test_on_failure

    def run(self) -> None:
        while not self.stopped():
            if not os.path.isfile(self.cs_log_filename):
                time.sleep(0.5)
                continue

            for line_number, line in enumerate(self.follow_file(self.cs_log_filename)):
                if self.stopped():
                    break

                for pattern, event in chain(CS_NORMAL_EVENTS_PATTERNS, CS_ERROR_EVENTS_PATTERNS):
                    if self.event_id:
                        # Connect the event to the stress load
                        event.event_id = self.event_id

                    if pattern.search(line):
                        if event.severity == Severity.CRITICAL and not self.stop_test_on_failure:
                            event = event.clone()  # so we don't change the severity to other stress threads
                            event.severity = Severity.ERROR
                        event.add_info(node=self.node, line=line, line_number=line_number).publish()
                        break  # Stop iterating patterns to avoid creating two events for one line of the log


class CSHDRFileLogger(SSHLoggerBase):
    VERBOSE_RETRIEVE = False

    def __init__(self, node: BaseNode, remote_log_file: str, target_log_file: str):
        super().__init__(node=node, target_log_file=target_log_file)
        self._child_process = None
        self._remote_log_file = remote_log_file
        self.target_log_file = target_log_file
        self._child_thread = ThreadPoolExecutor(max_workers=1)
        self._thread = None

    def start(self) -> None:
        LOGGER.debug("Start to read target_log_file: %s", self.target_log_file)
        self._termination_event.clear()
        self._thread = self._child_thread.submit(self._journal_thread)
        LOGGER.debug("Journal thread started for target_log_file: %s", self.target_log_file)

    def stop(self, timeout: float | None = None) -> None:
        self._termination_event.set()
        thread_cancelled = self._thread.cancel()
        LOGGER.debug("Is thread cancelled?: %s, target_log_file: %s", thread_cancelled, self.target_log_file)
        self._child_thread.shutdown(wait=False, cancel_futures=True)
        LOGGER.debug("Is thread shutdown?: %s, target_log_file: %s", self._thread.running(), self.target_log_file)
        if self._thread.running():
            self._thread.cancel()  # pylint: disable=no-member

    @cached_property
    def _logger_cmd_template(self) -> str:
        return f"tail -f {self._remote_log_file}"

    def validate_and_collect_hdr_file(self):
        """
        Validate that HDR file exists on the SCT runner.
        If it does not exist check if the file was created on the loader.
        If the HDR file found on the loader, try to copy to the runner.
        If the file is missed even on the loader - print error event.
        """
        if os.path.exists(self._target_log_file):
            return

        LOGGER.debug("'%s' file is not found on the runner. Try to find it on the loader %s",
                     self._target_log_file, self._node.name)
        HDRFileMissed(message=f"'{self._remote_log_file}' HDR file was not copied to the runner from loader",
                      severity=Severity.WARNING).publish()
        result = self._node.remoter.run(f"test -f {self._remote_log_file}", ignore_status=True)
        if not result.ok:
            HDRFileMissed(message=f"'{self._remote_log_file}' HDR file was not created on the loader {self._node.name}",
                          severity=Severity.ERROR).publish()
        try:
            LOGGER.debug("The '%s' file found on the loader %s", self._remote_log_file, self._node.name)
            self._node.remoter.receive_files(src=self._remote_log_file, dst=self._target_log_file)
        except Exception:  # noqa: BLE001
            HDRFileMissed(message=f"'{self._remote_log_file}' HDR file couldn't copied from loader {self._node.name}",
                          severity=Severity.ERROR).publish()

    # @raise_event_on_failure
    def _journal_thread(self) -> None:
        LOGGER.debug("Start journal thread. %s", self._remote_log_file)
        read_from_timestamp = None
        te_is_set = self._termination_event.is_set()
        while not te_is_set:
            LOGGER.debug("Start check if remoter ready. %s", self._remote_log_file)
            if self._is_ready_to_retrieve():
                LOGGER.debug("Remoter ready. %s", self._remote_log_file)
                self._retrieve(since=read_from_timestamp)
                LOGGER.debug("Retrieve finished. %s", self._remote_log_file)
                read_from_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            else:
                LOGGER.debug("Remoter is not ready. %s", self._remote_log_file)
                time.sleep(self.READINESS_CHECK_DELAY)
            te_is_set = self._termination_event.is_set()
            LOGGER.debug("_termination_event is set?: %s. %s", te_is_set, self._remote_log_file)

    def _is_ready_to_retrieve(self) -> bool:
        LOGGER.debug("Before remoter is_up. %s", self._remote_log_file)
        is_up = self._remoter.is_up()
        LOGGER.debug("After remoter is_up. Result: %s. %s", is_up, self._remote_log_file)
        return is_up

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.validate_and_collect_hdr_file()
        self.stop()


class CassandraStressThread(DockerBasedStressThread):  # pylint: disable=too-many-instance-attributes
    DOCKER_IMAGE_PARAM_NAME = 'stress_image.cassandra-stress'

    def __init__(self, loader_set, stress_cmd, timeout, stress_num=1, keyspace_num=1, keyspace_name='', compaction_strategy='',  # pylint: disable=too-many-arguments
                 profile=None, node_list=None, round_robin=False, client_encrypt=False, stop_test_on_failure=True,
                 params=None):
        super().__init__(loader_set=loader_set, stress_cmd=stress_cmd, timeout=timeout,
                         stress_num=stress_num, node_list=node_list,  # pylint: disable=too-many-arguments
                         round_robin=round_robin, stop_test_on_failure=stop_test_on_failure, params=params)
        self.keyspace_num = keyspace_num
        self.keyspace_name = keyspace_name
        self.profile = profile
        self.client_encrypt = client_encrypt
        self.stop_test_on_failure = stop_test_on_failure
        self.compaction_strategy = compaction_strategy

    def create_stress_cmd(self, cmd_runner, keyspace_idx, loader):  # pylint: disable=too-many-branches,too-many-locals
        stress_cmd = self.stress_cmd

        if "no-warmup" not in stress_cmd:
            # add no-warmup to stress_cmd if it's not there. See issue #5767
            stress_cmd = re.sub(r'(cassandra-stress [\w]+)', r'\1 no-warmup', stress_cmd)

        # When using cassandra-stress with "user profile" the profile yaml should be provided
        if 'profile' in stress_cmd and not self.profile:
            # support of using -profile in sct test-case yaml, assumes they exists data_dir
            # TODO: move those profile to their own directory
            cs_profile, profile = get_profile_content(stress_cmd)
            keyspace_name = profile['keyspace']
            self.profile = cs_profile
            self.keyspace_name = keyspace_name

        if self.keyspace_name:
            stress_cmd = stress_cmd.replace(" -schema ", " -schema keyspace={} ".format(self.keyspace_name))
        elif 'keyspace=' not in stress_cmd:  # if keyspace is defined in the command respect that
            stress_cmd = stress_cmd.replace(" -schema ", " -schema keyspace=keyspace{} ".format(keyspace_idx))

        if self.compaction_strategy and "compaction(" not in stress_cmd:
            stress_cmd = stress_cmd.replace(" -schema ", f" -schema 'compaction(strategy={self.compaction_strategy})' ")

        credentials = self.loader_set.get_db_auth()
        if credentials and 'user=' not in stress_cmd:
            # put the credentials into the right place into -mode section
            stress_cmd = re.sub(r'(-mode.*?)-', r'\1 user={} password={} -'.format(*credentials), stress_cmd)
        if self.client_encrypt and 'transport' not in stress_cmd:
            stress_cmd += \
                " -transport 'truststore=/etc/scylla/ssl_conf/client/cacerts.jks truststore-password=cassandra'"

        if (connection_bundle_file := self.node_list[0].parent_cluster.connection_bundle_file) and '-node' not in stress_cmd:
            stress_cmd += f" -cloudconf file={Path('/tmp') / connection_bundle_file.name}"
        elif self.node_list and '-node' not in stress_cmd:
            stress_cmd += " -node "
            if self.loader_set.test_config.MULTI_REGION:
                # The datacenter name can be received from "nodetool status" output. It's possible for DB nodes only,
                # not for loader nodes. So call next function for DB nodes
                datacenter_name_per_region = self.loader_set.get_datacenter_name_per_region(db_nodes=self.node_list)
                if loader_dc := datacenter_name_per_region.get(loader.region):
                    stress_cmd += f"datacenter={loader_dc} "
                else:
                    LOGGER.error("Not found datacenter for loader region '%s'. Datacenter per loader dict: %s",
                                 loader.region, datacenter_name_per_region)

            # if there are multiple rack/AZs configured, we'll try to configue c-s to pin to them
            rack_names = self.loader_set.get_rack_names_per_datacenter_and_rack_idx(db_nodes=self.node_list)
            node_list = self.node_list
            if len(set(rack_names.values())) > 1 and 'rack' in self._get_available_suboptions(cmd_runner, '-node'):
                if loader_rack := rack_names.get((str(loader.region), str(loader.rack))):
                    stress_cmd += f"rack={loader_rack} "
                    node_list = self.loader_set.get_nodes_per_datacenter_and_rack_idx(
                        db_nodes=self.node_list).get((str(loader.region), str(loader.rack)))

            node_ip_list = [n.cql_address for n in node_list]

            stress_cmd += ",".join(node_ip_list)
        if 'skip-unsupported-columns' in self._get_available_suboptions(cmd_runner, '-errors'):
            stress_cmd = self._add_errors_option(stress_cmd, ['skip-unsupported-columns'])
        return stress_cmd

    @staticmethod
    def _add_errors_option(stress_cmd: str, to_add: list) -> str:
        """
        Add suboption to -errors option, if such suboption is there, does not add or change it
        """
        to_add = list(to_add)
        current_error_option = next((option for option in stress_cmd.split(' -') if option.startswith('errors ')), None)
        if current_error_option is None:
            return f"{stress_cmd} -errors {' '.join(to_add)}"
        current_error_suboptions = current_error_option.split()[1:]
        new_error_suboptions = \
            list({suboption.split('=', 1)[0]: suboption for suboption in to_add + current_error_suboptions}.values())
        if len(new_error_suboptions) == len(current_error_suboptions):
            return stress_cmd
        return stress_cmd.replace(current_error_option, 'errors ' + ' '.join(new_error_suboptions))

    def _get_available_suboptions(self, loader, option, _cache={}):  # pylint: disable=dangerous-default-value
        if cached_value := _cache.get(option):
            return cached_value
        try:
            result = loader.run(
                cmd=f'cassandra-stress help {option} | grep "^Usage:"',
                timeout=self.timeout,
                ignore_status=True).stdout
        except Exception:  # pylint: disable=broad-except
            return []
        findings = re.findall(r' *\[([\w-]+?)[=?]*] *', result)
        _cache[option] = findings
        return findings

    @staticmethod
    def _disable_logging_for_cs(node, cmd_runner, _cache={}):  # pylint: disable=dangerous-default-value
        if not (node.is_kubernetes() or node.name in _cache):
            cmd_runner.run("cp /etc/scylla/cassandra/logback-tools.xml .", ignore_status=True)
            _cache[node.name] = 'done'

    @staticmethod
    def _add_hdr_log_option(stress_cmd: str, hdr_log_name: str) -> str:
        if "-log" in stress_cmd:
            if match := re.search(r"\s-log ([^-]*)-?", stress_cmd):
                cs_log_option = match.group(1)
                if "hdrfile" not in cs_log_option:
                    stress_cmd = stress_cmd.replace("-log", f"-log hdrfile={hdr_log_name}")
                else:
                    if replacing_hdr_file := re.search(r"hdrfile=(.*?)\s", cs_log_option):
                        stress_cmd = stress_cmd.replace(
                            f"hdrfile={replacing_hdr_file.group(1)}", f"hdrfile={hdr_log_name}")
        else:
            stress_cmd += f" -log hdrfile={hdr_log_name} interval=10s"

        return stress_cmd

    @staticmethod
    def _build_log_file_id(loader_idx, cpu_idx, keyspace_idx):
        return f"l{loader_idx}-c{cpu_idx}-k{keyspace_idx}-{uuid.uuid4()}"

    def _run_stress(self, loader, loader_idx, cpu_idx):
        pass

    def _run_cs_stress(self, loader, loader_idx, cpu_idx, keyspace_idx):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        cleanup_context = contextlib.nullcontext()
        os.makedirs(loader.logdir, exist_ok=True)

        # This tag will be output in the header of c-stress result,
        # we parse it to know the loader & cpu info in _parse_cs_summary().
        stress_cmd_opt = self.stress_cmd.split("cassandra-stress", 1)[1].split(None, 1)[0]

        log_id = self._build_log_file_id(loader_idx, cpu_idx, keyspace_idx)
        log_file_name = \
            os.path.join(loader.logdir, f'cassandra-stress-{stress_cmd_opt}-{log_id}.log')
        LOGGER.debug('cassandra-stress local log: %s', log_file_name)
        remote_hdr_file_name = f"cs-hdr-{stress_cmd_opt}-{log_id}.hdr"
        LOGGER.debug("cassandra-stress remote HDR log file: %s", remote_hdr_file_name)
        local_hdr_file_name = os.path.join(loader.logdir, remote_hdr_file_name)
        LOGGER.debug("cassandra-stress HDR local file %s", local_hdr_file_name)

        if "k8s" in self.params.get("cluster_backend"):
            cmd_runner = loader.remoter
            cmd_runner_name = loader.remoter.pod_name
        elif self.params.get("use_prepared_loaders"):
            cmd_runner = loader.remoter
            cmd_runner_name = loader.ip_address
        else:
            loader.remoter.run(f"touch $HOME/{remote_hdr_file_name}", ignore_status=True, verbose=False)
            cmd_runner_name = loader.ip_address

            cpu_options = ""
            cmd_runner = cleanup_context = RemoteDocker(loader, self.docker_image_name,
                                                        command_line="-c 'tail -f /dev/null'",
                                                        extra_docker_opts=f'{cpu_options} '
                                                        '--network=host '
                                                        '--security-opt seccomp=unconfined '
                                                        f'--label shell_marker={self.shell_marker}'
                                                        f' --entrypoint /bin/bash'
                                                        f' -w /'
                                                        f' -v $HOME/{remote_hdr_file_name}:/{remote_hdr_file_name}')

        stress_cmd = self.create_stress_cmd(cmd_runner, keyspace_idx, loader)
        if self.params.get('cs_debug'):
            cmd_runner.send_files(get_data_dir_path('logback-tools-debug.xml'),
                                  '/etc/scylla/cassandra/logback-tools.xml', delete_dst=True)
        if self.profile:
            loader_profile_path = os.path.join('/tmp', os.path.basename(self.profile))
            with open(self.profile, encoding="utf-8") as profile_file:
                LOGGER.info('Profile content:\n%s', profile_file.read())
            cmd_runner.send_files(self.profile, loader_profile_path, delete_dst=True)
            if 'scylla-qa-internal' in self.profile:
                LOGGER.info("Replace profile path %s in c-s command with actual %s",
                            self.profile, loader_profile_path)
                stress_cmd = replace_scylla_qa_internal_path(stress_cmd, loader_profile_path)

        if self.client_encrypt:
            ssl_conf_dir = Path(get_data_dir_path('ssl_conf', 'client'))
            for ssl_file in ssl_conf_dir.iterdir():
                if ssl_file.is_file():
                    cmd_runner.send_files(str(ssl_file),
                                          str(Path('/etc/scylla/ssl_conf/client') / ssl_file.name),
                                          verbose=True)

        if connection_bundle_file := self.connection_bundle_file:
            cmd_runner.send_files(str(connection_bundle_file),
                                  self.target_connection_bundle_file, delete_dst=True, verbose=True)

        if self.params.get("use_hdr_cs_histogram"):
            stress_cmd = self._add_hdr_log_option(stress_cmd, remote_hdr_file_name)
            hdr_logger_context = CSHDRFileLogger(
                node=loader,
                remote_log_file=remote_hdr_file_name,
                target_log_file=os.path.join(loader.logdir, remote_hdr_file_name),
            )
        else:
            hdr_logger_context = contextlib.nullcontext()

        LOGGER.info('Stress command:\n%s', stress_cmd)

        tag = f'TAG: loader_idx:{loader_idx}-cpu_idx:{cpu_idx}-keyspace_idx:{keyspace_idx}'

        if self.stress_num > 1:
            node_cmd = f'STRESS_TEST_MARKER={self.shell_marker}; taskset -c {cpu_idx} {stress_cmd}'
        else:
            node_cmd = f'STRESS_TEST_MARKER={self.shell_marker}; {stress_cmd}'
        node_cmd = f'echo {tag}; {node_cmd}'

        result = None
        self._disable_logging_for_cs(loader, cmd_runner)
        try:
            prefix,  *_ = stress_cmd.split("cassandra-stress", maxsplit=1)
            reporter = CassandraStressVersionReporter(
                cmd_runner, prefix, loader.parent_cluster.test_config.argus_client())
            reporter.report()
        except Exception:  # pylint: disable=broad-except
            LOGGER.info("Failed to collect cassandra-stress version information", exc_info=True)
        with cleanup_context, \
                CassandraStressExporter(instance_name=cmd_runner_name,
                                        metrics=nemesis_metrics_obj(),
                                        stress_operation=stress_cmd_opt,
                                        stress_log_filename=log_file_name,
                                        loader_idx=loader_idx, cpu_idx=cpu_idx), \
                CassandraStressEventsPublisher(node=loader, cs_log_filename=log_file_name,
                                               stop_test_on_failure=self.stop_test_on_failure) as publisher, \
                CassandraStressEvent(node=loader, stress_cmd=self.stress_cmd,
                                     log_file_name=log_file_name) as cs_stress_event, \
                CassandraStressHDRExporter(instance_name=cmd_runner_name,
                                           metrics=nemesis_metrics_obj(),
                                           stress_operation=stress_cmd_opt,
                                           stress_log_filename=local_hdr_file_name,
                                           loader_idx=loader_idx, cpu_idx=cpu_idx), \
                hdr_logger_context:
            publisher.event_id = cs_stress_event.event_id
            try:
                with SoftTimeoutContext(timeout=self.soft_timeout, operation="cassandra-stress"):
                    result = cmd_runner.run(cmd=node_cmd, timeout=self.hard_timeout, log_file=log_file_name, retry=0)
            except Exception as exc:  # pylint: disable=broad-except
                self.configure_event_on_failure(stress_event=cs_stress_event, exc=exc)

        return loader, result, cs_stress_event

    def run(self):
        self.configure_executer()
        for loader in self.loaders:
            loader_idx = loader.node_index
            for cpu_idx in range(self.stress_num):
                for ks_idx in range(1, self.keyspace_num + 1):
                    self.results_futures += [self.executor.submit(self._run_cs_stress,
                                                                  *(loader, loader_idx, cpu_idx, ks_idx))]
                    if loader_idx == 0 and cpu_idx == 0 and self.max_workers > 1:
                        # Wait for first stress thread to create the schema, before spawning new stress threads
                        time.sleep(30)

        return self

    def get_results(self) -> list[dict | None]:
        ret = []
        results = super().get_results()

        for _, result, event in results:
            if not result:
                # Silently skip if stress command threw error, since it was already reported in _run_stress
                continue
            output = result.stdout + result.stderr
            try:
                lines = output.splitlines()
                node_cs_res = BaseLoaderSet._parse_cs_summary(lines)  # pylint: disable=protected-access
                if node_cs_res:
                    ret.append(node_cs_res)
            except Exception as exc:  # pylint: disable=broad-except
                event.add_error([f"Failed to process stress summary due to {exc}"])
                event.severity = Severity.CRITICAL
                event.event_error()

        return ret

    def verify_results(self) -> (list[dict | None], list[str | None]):
        cs_summary = []
        errors = []

        results = super().get_results()

        for node, result, _ in results:
            if not result:
                # Silently skip if stress command threw error, since it was already reported in _run_stress
                continue
            output = result.stdout + result.stderr
            lines = output.splitlines()
            node_cs_res = BaseLoaderSet._parse_cs_summary(lines)  # pylint: disable=protected-access
            if node_cs_res:
                cs_summary.append(node_cs_res)
            for line in lines:
                if 'java.io.IOException' in line:
                    errors += ['%s: %s' % (node, line.strip())]

        return cs_summary, errors


stress_cmd_get_duration_pattern = re.compile(r' [-]{0,2}duration[\s=]+([\d]+[hms]+)')
stress_cmd_get_warmup_pattern = re.compile(r' [-]{0,2}warmup[\s=]+([\d]+[hms]+)')


def get_timeout_from_stress_cmd(stress_cmd: str) -> int | None:
    """Gets timeout in seconds based on duration and warmup arguments from stress command."""
    timeout = 0
    if duration_match := stress_cmd_get_duration_pattern.search(stress_cmd):
        timeout += time_period_str_to_seconds(duration_match.group(0))
    if warmup_match := stress_cmd_get_warmup_pattern.search(stress_cmd):
        timeout += time_period_str_to_seconds(warmup_match.group(0))
    if timeout == 0:
        return None
    else:
        # adding 15 minutes to timeout for general all others delays
        return timeout + 900
