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

#  pylint: disable=too-many-lines
import os
import json
import re
import time
import shutil
import fnmatch
import logging
import datetime
import tempfile
import traceback
from collections import OrderedDict
from typing import Optional, Tuple, List
from pathlib import Path
from functools import cached_property

import requests

import sdcm.monitorstack.ui as monitoring_ui
from sdcm.paths import SCYLLA_YAML_PATH, SCYLLA_PROPERTIES_PATH, SCYLLA_MANAGER_AGENT_YAML_PATH, \
    SCYLLA_MANAGER_YAML_PATH
from sdcm.provision import provisioner_factory
from sdcm.provision.network_configuration import ssh_connection_ip_type
from sdcm.provision.provisioner import ProvisionerError
from sdcm.remote import RemoteCmdRunnerBase, LocalCmdRunner
from sdcm.db_stats import PrometheusDBStats
from sdcm.sct_events.events_device import EVENTS_LOG_DIR, RAW_EVENTS_LOG
from sdcm.test_config import TestConfig
from sdcm.utils.common import (
    S3Storage,
    ParallelObject,
    list_instances_aws,
    list_instances_gce,
    remove_files,
    get_builder_by_test_id,
    get_testrun_dir,
    search_test_id_in_latest,
    filter_aws_instances_by_type,
    filter_gce_instances_by_type,
    get_sct_root_path,
    normalize_ipv6_url, create_remote_storage_dir,
)
from sdcm.utils.context_managers import environment
from sdcm.utils.decorators import retrying
from sdcm.utils.docker_utils import get_docker_bridge_gateway
from sdcm.utils.get_username import get_username
from sdcm.utils.k8s import KubernetesOps
from sdcm.utils.s3_remote_uploader import upload_remote_files_directly_to_s3
from sdcm.utils.gce_utils import gce_public_addresses, gce_private_addresses

LOGGER = logging.getLogger(__name__)


class CollectingNode:
    # pylint: disable=too-few-public-methods,too-many-instance-attributes
    logdir = None

    def __init__(self, name, ssh_login_info=None, instance=None, global_ip=None, grafana_ip=None, tags=None, logdir=None):  # pylint: disable=too-many-arguments
        if logdir:
            self.logdir = logdir
        self._containers = {}
        self.name = name
        if ssh_login_info is None:
            self.remoter = LocalCmdRunner()
        else:
            self.remoter = RemoteCmdRunnerBase.create_remoter(**ssh_login_info)
        self.ssh_login_info = ssh_login_info
        self._instance = instance
        self.external_address = global_ip
        if grafana_ip is None:
            self.grafana_address = global_ip
        else:
            self.grafana_address = grafana_ip
        self.tags = {**(tags or {}), "Name": self.name, }


class PrometheusSnapshotErrorException(Exception):
    pass


class BaseLogEntity:  # pylint: disable=too-few-public-methods
    """Base class for log entity

    LogEntity any file, command, operation, complex actions
    which require to be logged, stored locally, uploaded to
    S3 storage
    """
    collect_timeout = 300
    _params = {}

    def __init__(self, name, command="", search_locally=False, collect_from_parent=False):
        """
         collect_from_parent: relevant for cluster:
                            - there are logs of every node
                            - there are logs that relevant for whole cluster
                            If this parameter is True - collect logs on cluster level (cluster folder), not under node folder
        """
        self.name = name
        self.cmd = command
        self.search_locally = search_locally
        self.collect_from_parent = collect_from_parent

    def set_params(self, params):
        self._params = params

    def collect(self, node, local_dst, remote_dst=None, local_search_path=None):  # pylint: disable=unused-argument,no-self-use
        raise Exception('Should be implemented in child class')


class BaseMonitoringEntity(BaseLogEntity):
    def get_monitoring_base_dir(self, node):
        # Avoid cyclic dependencies
        if hasattr(node, "parent_cluster") and node.parent_cluster:
            return node.parent_cluster.monitor_install_path_base
        return self.get_monitoring_stack(self._params.get('cluster_backend')).get_monitor_install_path_base(node)

    @staticmethod
    def get_monitoring_stack(backend):
        if backend == 'aws':
            from sdcm.cluster_aws import MonitorSetAWS  # pylint: disable=import-outside-toplevel
            return MonitorSetAWS
        elif backend == 'docker':
            from sdcm.cluster_docker import MonitorSetDocker  # pylint: disable=import-outside-toplevel
            return MonitorSetDocker
        elif backend == 'gce':
            from sdcm.cluster_gce import MonitorSetGCE  # pylint: disable=import-outside-toplevel
            return MonitorSetGCE
        elif backend == 'baremetal':
            from sdcm.cluster_baremetal import MonitorSetPhysical  # pylint: disable=import-outside-toplevel
            return MonitorSetPhysical
        from sdcm.cluster import BaseMonitorSet  # pylint: disable=import-outside-toplevel
        return BaseMonitorSet

    def get_monitoring_version(self, node):
        try:
            basedir = self.get_monitoring_base_dir(node)
            result = node.remoter.run(
                rf"ls {basedir} | grep 'scylla-monitoring-src\|scylla-grafana-monitoring-scylla-monitoring'",
                ignore_status=True, verbose=False)
            name = result.stdout.strip()
            if not name:
                LOGGER.error("Dir with scylla monitoring stack was not found")
                return None, None, None
            result = node.remoter.run(
                f"cat {basedir}/{name}/monitor_version", ignore_status=True, verbose=False)
        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.error("Failed to get monitoring version: %s", details)
            return None, None, None

        try:
            monitor_version, scylla_version = result.stdout.strip().split(':')
        except ValueError:
            monitor_version = None
            scylla_version = None
        return name, monitor_version, scylla_version


class CommandLog(BaseLogEntity):  # pylint: disable=too-few-public-methods
    """Command to get log and save output result to file on remote host

    CommandLog which allow to save the output (usually log output)
    to file. The log output should be produced by the command

    Extends:
        BaseLogEntity
    """

    def collect(self, node, local_dst, remote_dst=None, local_search_path=None) -> Optional[str]:
        if not node or not node.remoter or remote_dst is None:
            return None
        remote_logfile = LogCollector.collect_log_remotely(node=node,
                                                           cmd=self.cmd,
                                                           log_filename=os.path.join(remote_dst, self.name))
        LogCollector.receive_log(node=node,
                                 remote_log_path=remote_logfile,
                                 local_dir=local_dst,
                                 timeout=self.collect_timeout)
        return os.path.join(local_dst, os.path.basename(remote_logfile))


class FileLog(CommandLog):
    """Log File Entinty

    Allow to find log file locally first, if it have not
    found
    Run command to collect log data remotely if command provided

    Extends:
        CommandLogEntity
    """
    @staticmethod
    def find_local_files(search_in_dir, search_pattern, except_patterns="collected_logs"):
        local_files = []
        for root, _, files in os.walk(search_in_dir):
            for f in files:  # pylint: disable=invalid-name
                full_path = os.path.join(root, f)
                if except_patterns in full_path:
                    continue
                if full_path.endswith(search_pattern) or fnmatch.fnmatch(full_path, "*{}".format(search_pattern)):
                    if os.path.islink(full_path):
                        full_path = os.path.realpath(full_path)
                    local_files.append(full_path)
        return local_files

    @staticmethod
    def find_on_builder(builder, file_name, search_in_dir="/"):  # pylint: disable=unused-argument
        result = builder.remoter.run('find {search_in_dir} -name {file_name}'.format(**locals()),
                                     ignore_status=True)
        if not result.exited and not result.stderr:
            path = result.stdout.strip()
        else:
            path = None

        return path

    def _is_file_collected(self, local_dst):
        for collected_file in os.listdir(local_dst):
            if self.name in collected_file or fnmatch.fnmatch(collected_file, "*{}".format(self.name)):
                return True
        return False

    def collect(self, node, local_dst, remote_dst=None, local_search_path=None):
        os.makedirs(local_dst, exist_ok=True)
        if self.search_locally and local_search_path:
            search_pattern = self.name if not node else "/".join([node.name, self.name])
            local_logfiles = self.find_local_files(local_search_path, search_pattern)
            for logfile in local_logfiles:
                shutil.copy(src=logfile, dst=local_dst)

        if self.cmd and not self._is_file_collected(local_dst):
            super().collect(node, local_dst, remote_dst)

        return local_dst

    def collect_from_builder(self, builder, local_dst, search_in_dir) -> None:
        if file_path := self.find_on_builder(builder, self.name, search_in_dir):
            builder.remoter.receive_files(file_path, local_dst, timeout=self.collect_timeout)


class DirLog(FileLog):
    """Get files that match provided filter keeping their dir placement the same.

    It is useful when you want to copy whole dir with lots of files,
    which could have dynamic names.

    Usage example:
         DirLog(name='some-dir-name-with-files/*', search_locally=True)
    """

    def collect(self, node, local_dst, remote_dst=None, local_search_path=None):
        os.makedirs(local_dst, exist_ok=True)
        dst_logfiles = []
        if self.search_locally and local_search_path:
            local_logfiles = self.find_local_files(local_search_path, self.name)
            for logfile in local_logfiles:
                relative_path = logfile.split(local_search_path)[-1]
                if relative_path[0] == "/":
                    relative_path = relative_path[1:]
                current_dst = Path(local_dst) / relative_path
                os.makedirs(str(current_dst).rsplit("/", 1)[0], exist_ok=True)
                shutil.copy(src=logfile, dst=current_dst)
                dst_logfiles.append(str(current_dst))
        return dst_logfiles

    def collect_from_builder(self, builder, local_dst, search_in_dir) -> None:
        # TODO: implement it to be able to gather whole dirs on remote nodes
        LOGGER.warning(
            "'DirLog' class doesn't support 'collect_from_builder' method. "
            "It is to be implemented. Ignoring gathering following logs: '%s'", self.name)


class PrometheusSnapshots(BaseMonitoringEntity):
    """Get Prometheus snapshot entity

    Specific for Prometheus Log entity which allow to
    collect Prometheus snapshot or data dir

    Extends:
        BaseLogEntity
    """
    monitoring_data_dir_name = "scylla-monitoring-data"
    collect_timeout = 3000

    def __init__(self, *args, **kwargs):
        self.monitoring_data_dir = kwargs.pop('monitoring_data_dir', None)
        super().__init__(*args, **kwargs)

        self.current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    @retrying(n=3, sleep_time=10, allowed_exceptions=(PrometheusSnapshotErrorException, Exception),
              message='Create prometheus snapshot')
    def create_prometheus_snapshot(self, node):
        prometheus_client = PrometheusDBStats(host=node.external_address)
        result = prometheus_client.create_snapshot()
        if result and "success" in result['status']:
            snapshot_dir = os.path.join(self.monitoring_data_dir,
                                        "snapshots",
                                        result['data']['name'])
            return snapshot_dir
        else:
            raise PrometheusSnapshotErrorException(result)

    def get_prometheus_snapshot_remote(self, node) -> Optional[str]:
        try:
            snapshot_dir = self.create_prometheus_snapshot(node)
        except (PrometheusSnapshotErrorException, Exception) as details:  # pylint: disable=broad-except
            LOGGER.warning("Create prometheus snapshot failed %s.\nUse prometheus data directory", details)
            node.remoter.run('docker stop aprom', ignore_status=True)
            snapshot_dir = self.monitoring_data_dir

        LOGGER.info(snapshot_dir)
        archive_logfile = LogCollector.archive_log_remotely(node, snapshot_dir, f"{self.name}_{self.current_datetime}")
        node.remoter.run('docker start aprom', ignore_status=True)

        if not archive_logfile:
            return None

        LOGGER.info(archive_logfile)
        return archive_logfile

    def setup_monitor_data_dir(self, node):
        if self.monitoring_data_dir:
            return
        base_dir = self.get_monitoring_base_dir(node)
        self.monitoring_data_dir = os.path.join(base_dir, self.monitoring_data_dir_name)

    def collect(self, node, local_dst, remote_dst=None, local_search_path=None) -> Optional[str]:
        self.setup_monitor_data_dir(node)
        if remote_snapshot_archive := self.get_prometheus_snapshot_remote(node):
            LogCollector.receive_log(node,
                                     remote_log_path=remote_snapshot_archive,
                                     local_dir=local_dst,
                                     timeout=self.collect_timeout)
            return os.path.join(local_dst, os.path.basename(remote_snapshot_archive))
        return None


class MonitoringStack(BaseMonitoringEntity):
    """LogEntinty to collect MonitoringStack

    Collect monitoring stack

    Extends:
        BaseLogEntity

    Variables:
        grafana_port {number} -- Grafana server port
    """
    grafana_port = 3000
    collect_timeout = 1800

    def get_monitoring_data_stack(self, node, local_dist) -> str:
        monitor_base_dir = self.get_monitoring_base_dir(node)
        monitor_install_dir_name, monitor_branch, monitor_version = self.get_monitoring_version(node)

        LOGGER.info("%s, %s, %s", monitor_install_dir_name, monitor_branch, monitor_version)

        if not monitor_branch or not monitor_version:
            return ""

        with open(os.path.join(tempfile.mkdtemp(), "annotations.json"), "w", encoding="utf-8") as annotations_file:
            annotations_file.write(self.get_grafana_annotations(node.grafana_address))
        node.remoter.send_files(src=annotations_file.name,
                                dst=os.path.join(monitor_base_dir,
                                                 monitor_install_dir_name,
                                                 "sct_monitoring_addons"))

        archive_name = os.path.join(monitor_base_dir,
                                    f"monitoring_data_stack_{monitor_branch}_{monitor_version}.tar.gz")
        if not node.remoter.run(f"tar czvf '{archive_name}' --exclude='gpx_scylladb_*' -C '{monitor_base_dir}' '{monitor_install_dir_name}'",
                                ignore_status=True).ok:
            return ""
        if not check_archive(node.remoter, archive_name):
            LOGGER.error("Archive with monitoring data stack (%s) is corrupted.", archive_name)
            return ""
        node.remoter.receive_files(src=archive_name, dst=local_dist, timeout=self.collect_timeout)

        return os.path.join(local_dist, os.path.basename(archive_name))

    def get_grafana_annotations(self, grafana_ip: str) -> str:
        try:
            res = requests.get(f"http://{grafana_ip}:{self.grafana_port}/api/annotations")
            if res.ok:
                return res.text
        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.warning("Unable to get Grafana annotations [%s]", details)
        return ""

    @staticmethod
    @retrying(n=3, sleep_time=3, message="Search dashboard...", raise_on_exceeded=False)
    def search_dashboard(grafana_ip: str, port: int, query: str) -> list:
        search_api_url = f"http://{grafana_ip}:{port}/api/search?query={query}"
        resp = requests.get(search_api_url)
        if not resp.ok:
            LOGGER.error("Search dashboards by query '%s' failed: %s %s", query, resp.status_code, resp.content)
            return []
        return resp.json()

    @staticmethod
    def get_dashboard_by_title(grafana_ip: str, port: int, title: str) -> Optional[dict]:
        dashboards = MonitoringStack.search_dashboard(grafana_ip, port, title)
        if not dashboards:
            LOGGER.error("Dashboard with title '%s' was not found", title)
            return None
        return next((dashboard for dashboard in dashboards if title in dashboard["title"]), None)

    def collect(self, node, local_dst, remote_dst=None, local_search_path=None):
        local_archive = self.get_monitoring_data_stack(node, local_dst)
        if not local_archive:
            LOGGER.error('Monitoring stack were not collected')
        return local_archive


class GrafanaEntity(BaseMonitoringEntity):  # pylint: disable=too-few-public-methods
    """Base Gragana log entity

    Base class to support various Grafana log entity

    Extends:
        BaseLogEntity

    Variables:
        grafana_entity_names {list} -- list of links to collect
        grafana_port {number} -- Grafana server port
        grafana_entity_url_tmpl {str} -- template of grafana url to collect
        phantomjs_base {str} -- name and version of phantomjs package
    """
    base_grafana_dashboards = [
        monitoring_ui.OverviewDashboard(),
        monitoring_ui.ServerMetricsNemesisDashboard(),
    ]

    grafana_port = 3000
    grafana_entity_url_tmpl = "http://{node_ip}:{grafana_port}/render{path}?from={st}&to=now&refresh=1d"
    sct_base_path = get_sct_root_path()

    def __init__(self, *args, **kwargs):
        test_start_time = kwargs.pop("test_start_time", None)
        if not test_start_time:
            # set test start time previous 6 hours
            test_start_time = time.time() - (6 * 3600)
        self.start_time = str(test_start_time).split('.', maxsplit=1)[0] + '000'
        self.grafana_dashboards = self.base_grafana_dashboards + kwargs.pop("extra_entities", [])
        super().__init__(*args, **kwargs)

    def get_version(self, node):
        _, _, version = self.get_monitoring_version(node)
        if version:
            version = version.replace('.', '-')
            return version
        LOGGER.warning("Monitoring version was not found")
        return None


class GrafanaScreenShot(GrafanaEntity):
    """Collect Grafana screenshot

    Collect grafana screenshot

    Extends:
        GrafanaEntity
    """

    def get_grafana_screenshot(self, node, local_dst):
        """
            Take screenshot of the Grafana per-server-metrics dashboard and upload to S3
        """
        screenshots = []
        version = self.get_version(node)
        if not version:
            return screenshots

        try:
            for dashboard in self.grafana_dashboards:
                try:
                    dashboard_metadata = MonitoringStack.get_dashboard_by_title(
                        grafana_ip=normalize_ipv6_url(node.grafana_address),
                        port=self.grafana_port,
                        title=dashboard.title)
                    if not dashboard_metadata:
                        LOGGER.error("Dashboard with title '%s' was not found", dashboard.title)
                        continue

                    grafana_url = self.grafana_entity_url_tmpl.format(
                        node_ip=normalize_ipv6_url(node.grafana_address),
                        grafana_port=self.grafana_port,
                        path=dashboard_metadata["url"],
                        st=self.start_time)
                    screenshot_path = os.path.join(local_dst,
                                                   "%s-%s-%s-%s.png" % (self.name,
                                                                        dashboard.name,
                                                                        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                                                                        node.name))
                    LOGGER.debug("Get screenshot for url %s, save to %s", grafana_url, screenshot_path)
                    # deliberately specifying params as string to be able to use kiosk mode
                    # since requests can put a param without value, if using dict
                    # https://github.com/psf/requests/issues/2651
                    with requests.get(grafana_url, stream=True,
                                      params=f"width={dashboard.resolution[0]}&height=-1&kiosk") as response:
                        response.raise_for_status()
                        with open(screenshot_path, 'wb') as output_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                output_file.write(chunk)
                    screenshots.append(screenshot_path)
                except Exception as details:  # pylint: disable=broad-except
                    LOGGER.error("Error get screenshot %s: %s", dashboard.name, details, exc_info=True)

            return screenshots

        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.error("Error taking monitor screenshot: %s, traceback: %s", details, traceback.format_exc())
            return []

    def collect(self, node, local_dst, remote_dst=None, local_search_path=None):
        node.logdir = local_dst
        os.makedirs(local_dst, exist_ok=True)
        return self.get_grafana_screenshot(node, local_dst)


class LogCollector:
    """Base class for LogCollector types

        Base class implements interface for collecting
        various LogEntities on different types of Clusters/RemoteHosts

        Variables:
            node_remote_dir {str} -- name of remote dir on remote host
            _current_run {str} -- DateTime of current collecting log run
            cluster_log_type {str} -- Type of cluster
            log_entities {list} -- List of log entities, which should be collected on remote hosts
            USER {str} -- name of user, for which search local file log versions
    """
    _current_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cluster_log_type = 'base'
    cluster_dir_prefix = 'base'
    log_entities = []
    node_remote_dir = '/tmp'
    collect_timeout = 300

    @property
    def current_run(self):
        return LogCollector._current_run

    def __init__(self, nodes, test_id, storage_dir, params):
        self.test_id = test_id
        self.nodes = nodes
        self.local_dir = self.create_local_storage_dir(storage_dir)
        self.params = params
        for entity in self.log_entities:
            if self.params:
                entity.set_params(self.params)

    def create_local_storage_dir(self, base_local_dir):
        local_dir = os.path.join(base_local_dir, self.current_run,
                                 "{}-{}".format(self.cluster_log_type, self.test_id[:8]))
        try:
            os.makedirs(local_dir, exist_ok=True)
        except OSError as details:
            if not os.path.exists(local_dir):
                LOGGER.error("Folder is not created. {}".format(details))
                raise
        return local_dir

    def create_remote_storage_dir(self, node, path=''):
        if not path:
            path = node.name
        try:
            remote_dir = os.path.join(self.node_remote_dir, path)
            result = node.remoter.run('mkdir -p {}'.format(remote_dir), ignore_status=True)

            if result.exited > 0:
                LOGGER.error(
                    'Remote storing folder not created.\n{}'.format(result))
                remote_dir = self.node_remote_dir

        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.error("Error during creating remote directory %s", details)
            remote_dir = self.node_remote_dir

        return remote_dir

    @staticmethod
    def collect_log_remotely(node, cmd: str, log_filename: str) -> Optional[str]:
        if not node.remoter:
            return None
        collect_log_command = f"{cmd} > '{log_filename}' 2>&1"
        node.remoter.run(collect_log_command, ignore_status=True, verbose=True)
        result = node.remoter.run(f"test -f '{log_filename}'", ignore_status=True)
        return log_filename if result.ok else None

    @staticmethod
    def archive_log_remotely(node, log_filename: str, archive_name: Optional[str] = None) -> Optional[str]:
        if not node.remoter:
            return None
        archive_dir, log_filename = os.path.split(log_filename)
        archive_name = os.path.join(archive_dir, archive_name or log_filename) + ".tar.gz"
        if not node.remoter.run(f"tar czf '{archive_name}' -C '{archive_dir}' '{log_filename}'", ignore_status=True).ok:
            LOGGER.error("Unable to archive log `%s' to `%s'", log_filename, archive_name)
            return None
        if not check_archive(node.remoter, archive_name):
            return None
        return archive_name

    @staticmethod
    def receive_log(node, remote_log_path, local_dir, timeout=300):
        os.makedirs(local_dir, exist_ok=True)
        if node.remoter:
            node.remoter.receive_files(src=remote_log_path,
                                       dst=local_dir,
                                       timeout=timeout)
        return local_dir

    def collect_logs(self, local_search_path: Optional[str] = None) -> list[str]:
        def collect_logs_per_node(node):
            LOGGER.info('Collecting logs on host: %s', node.name)
            remote_node_dir = self.create_remote_storage_dir(node)
            local_node_dir = os.path.join(self.local_dir, node.name)
            local_parent_dir = self.local_dir
            for log_entity in self.log_entities:
                try:
                    log_entity.collect(node=node,
                                       local_dst=local_parent_dir if log_entity.collect_from_parent else local_node_dir,
                                       remote_dst=remote_node_dir,
                                       local_search_path=local_search_path)
                except Exception as details:  # pylint: disable=unused-variable, broad-except  # noqa: BLE001
                    LOGGER.error("Error occured during collecting of %s on host: %s\n%s",
                                 log_entity.name, node.name, details)

        LOGGER.debug("Nodes list %s", [node.name for node in self.nodes])

        if not self.nodes and not os.listdir(self.local_dir):
            LOGGER.warning('No nodes found for %s cluster. Logs will not be collected', self.cluster_log_type)
            return []
        if workers_number := len(self.nodes):
            workers_number = min(workers_number, 30)
            try:
                ParallelObject(self.nodes, num_workers=workers_number, timeout=self.collect_timeout).run(
                    collect_logs_per_node, ignore_exceptions=True)
            except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
                LOGGER.error('Error occured during collecting logs %s', details)

        if not os.listdir(self.local_dir):
            LOGGER.warning('Directory %s is empty', self.local_dir)
            return []

        final_archive = self.archive_to_tarfile(self.local_dir)
        if not final_archive:
            return []
        s3_link = upload_archive_to_s3(final_archive, f"{self.test_id}/{self.current_run}")
        remove_files(self.local_dir)
        remove_files(final_archive)
        return [s3_link]

    def collect_logs_for_inactive_nodes(self, local_search_path=None):
        node_names = {node.name for node in self.nodes}
        if not local_search_path:
            return
        for root, _, _ in os.walk(local_search_path):
            if self.cluster_dir_prefix in root:
                for entity in self.log_entities:
                    entity.collect(CollectingNode(name=root), self.local_dir, local_search_path=root)
                    node_dirs = {dir_name for dir_name in os.listdir(root)
                                 if os.path.isdir(os.path.join(root, dir_name))}
                    if len(node_names) != len(node_dirs):
                        inactive_nodes = node_dirs.difference(node_names)
                        for dir_name in inactive_nodes:
                            entity.collect(CollectingNode(name=dir_name),
                                           os.path.join(self.local_dir, dir_name),
                                           local_search_path=os.path.join(root, dir_name))

    def update_db_info(self):
        pass

    def _compress_file(self, src_path: str, src_name: str) -> str:  # pylint: disable=no-self-use
        archive_name = f"{src_name}.tar.zst"
        LocalCmdRunner().run(cmd=f"tar --zstd -cf {archive_name} -C {os.path.dirname(src_path)} {src_name}")

        return archive_name

    def archive_to_tarfile(self, src_path: str, add_test_id_to_archive: bool = False) -> str:
        src_name = os.path.basename(src_path)
        if add_test_id_to_archive:
            # Add test_id to the archive name when archive is created per log file, like: sct.log, email_data.json
            extension = f".{src_name.split('.')[-1]}"
            if extension in ['.log', '.json']:
                src_name = src_name.replace(extension, f"-{self.test_id.split('-')[0]}{extension}")
        try:
            return self._compress_file(src_path, src_name)
        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.error("Error during archive creation. Details: \n%s", details)
            return None


class ScyllaLogCollector(LogCollector):
    """ScyllaDB cluster log collecting

    Collect on each node the logs for Scylla DB cluster

    Extends:
        LogCollector

    Variables:
        log_entities {list} -- LogEntities to collect
        cluster_log_type {str} -- cluster type name
    """
    log_entities = [FileLog(name='system.log',
                            command="sudo journalctl --no-tail --no-pager -u scylla-ami-setup.service "
                                    "-u scylla-image-setup.service -u scylla-io-setup.service -u scylla-server.service "
                                    "-u scylla-jmx.service -u scylla-housekeeping-restart.service "
                                    "-u scylla-housekeeping-daily.service -o short-precise", search_locally=True),
                    FileLog(name='system_*',
                            search_locally=True),
                    FileLog(name='kallsyms_*',
                            search_locally=True),
                    FileLog(name='lsof_*',
                            search_locally=True),
                    FileLog(name='netstat_*',
                            search_locally=True),
                    FileLog(name='schema.log',
                            search_locally=True,
                            collect_from_parent=True),
                    FileLog(name='system_schema_tables.log',
                            search_locally=True,
                            collect_from_parent=True),
                    CommandLog(name='cpu_info',
                               command='cat /proc/cpuinfo'),
                    CommandLog(name='mem_info',
                               command='cat /proc/meminfo'),
                    CommandLog(name='interrupts',
                               command='cat /proc/interrupts'),
                    CommandLog(name='vmstat',
                               command='cat /proc/vmstat'),
                    CommandLog(name='scylla.yaml',
                               # Fixme: command=f'cat {self.add_install_prefix(SCYLLA_YAML_PATH)}'),
                               command=f'cat {SCYLLA_YAML_PATH}'),
                    CommandLog(name='coredumps.info',
                               command='sudo coredumpctl info'),
                    CommandLog(name='io-properties.yaml',
                               command='cat /etc/scylla.d/io_properties.yaml'),
                    CommandLog(name='dmesg.log',
                               command='sudo dmesg -P'),
                    CommandLog(name='systemctl.status',
                               command='sudo systemctl status --all --full --no-pager'),
                    CommandLog(name='cassandra-rackdc.properties',
                               command=f'cat {SCYLLA_PROPERTIES_PATH}'),
                    CommandLog(name='scylla-manager-agent.yaml',
                               command=f'cat {SCYLLA_MANAGER_AGENT_YAML_PATH}'),
                    CommandLog(name='setup_scripts_errors.log',
                               command='for i in /var/tmp/scylla/*.log;do echo [$i]; cat $i;done'),
                    CommandLog(name='scylla_doctor.vitals.json',
                               command='cat *.vitals.json'),
                    CommandLog(name='cloud-init-output.log',
                               command='cat /var/log/cloud-init-output.log'),
                    CommandLog(name='cloud-init.log',
                               command='cat /var/log/cloud-init.log'),
                    ]

    cmd = "test -f /etc/scylla/ssl_conf/{0} && cat /etc/scylla/ssl_conf/{0}"
    log_entities.extend([
        FileLog(name='db.crt', command=cmd.format('db.crt')),
        FileLog(name='client-facing.crt', command=cmd.format('client-facing.crt')),
        FileLog(name='cqlshrc', command=cmd.format('client/cqlshrc')),
    ])

    cluster_log_type = "db-cluster"
    cluster_dir_prefix = "db-cluster"
    collect_timeout = 600

    def collect_logs(self, local_search_path=None) -> list[str]:
        self.collect_logs_for_inactive_nodes(local_search_path)
        return super().collect_logs(local_search_path)


def save_kallsyms_map(node):

    LOGGER.info('Saving kallsyms map from host: %s', node.name)
    if remote_node_dir := create_remote_storage_dir(node):
        uptime = datetime.datetime.strptime(node.remoter.run('uptime -s', ignore_status=True).stdout.strip(),
                                            '%Y-%m-%d %H:%M:%S').strftime("%Y%m%d_%H%M%S")
        kallsyms_name = f'kallsyms_{uptime}'
        kallsyms_file_path = os.path.join(node.logdir, kallsyms_name)
        if os.path.exists(kallsyms_file_path):
            LOGGER.debug("The kallsyms file '%s' already exists and not changed. Not collecting it's map",
                         kallsyms_file_path)
            return
        log_entity = CommandLog(name=kallsyms_name,
                                command='sudo cat /proc/kallsyms')

        try:
            log_entity.collect(node, node.logdir, remote_node_dir)
        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.error("Error occurred during collecting kallsyms on host: %s\n%s", node.name, details)


def collect_diagnostic_data(node):
    log_entities = [
        # make names unique to avoid skipping
        CommandLog(name=f'lsof_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}', command='sudo lsof'),
        CommandLog(name=f'netstat_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}', command='sudo netstat -tanp'),
    ]
    collect_log_entities(node, log_entities)


def collect_log_entities(node, log_entities: List[BaseLogEntity]):
    """Collects diagnostics data from node - used during the test.

    Log Entities should have unique names, otherwise won't be collected."""
    LOGGER.info('Collecting diagnostics data from: %s', node.name)
    if remote_node_dir := create_remote_storage_dir(node):
        for log_entity in log_entities:
            if os.path.exists(os.path.join(node.logdir, log_entity.name)):
                LOGGER.debug("Diagnostic file '%s' already exists and not changed. Skipping collection.",
                             log_entity.name)
                continue
            try:
                log_entity.collect(node, node.logdir, remote_node_dir)
                LOGGER.debug("Diagnostic file '%s' collected", log_entity.name)
            except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
                LOGGER.error("Error occurred during collecting diagnostics data on host: %s\n%s", node.name, details)


class LoaderLogCollector(LogCollector):
    cluster_log_type = "loader-set"
    cluster_dir_prefix = "loader-set"

    log_entities = [
        FileLog(name='system.log',
                command="sudo journalctl --no-tail --no-pager",
                search_locally=True),
        FileLog(name='*cassandra-stress*.log',
                search_locally=True),
        FileLog(name='*ycsb*.log',
                search_locally=True),
        FileLog(name='*gemini-l*.log',
                search_locally=True),
        FileLog(name='*gemini_*_statements_*.log',
                search_locally=True),
        FileLog(name='gemini_result*.log',
                search_locally=True),
        FileLog(name='cdclogreader*.log',
                search_locally=True),
        FileLog(name='scylla-bench-l*.log',
                search_locally=True),
        FileLog(name='kcl-l*.log',
                search_locally=True),
        FileLog(name='*cassandra-harry*.log',
                search_locally=True),
        FileLog(name="*cs-hdr-*.hdr",
                search_locally=True),
        FileLog(name='*latte*',
                search_locally=True),
        FileLog(name='test.crt',
                command="test -f /etc/scylla/ssl_conf/{0} && cat /etc/scylla/ssl_conf/{0}".format('test.crt'))
    ]

    def collect_logs(self, local_search_path=None) -> list[str]:
        self.collect_logs_for_inactive_nodes(local_search_path)
        return super().collect_logs(local_search_path)


class MonitorLogCollector(LogCollector):
    """SCT monitor set log collector

    Collect required log entites for Monitor set nodes

    Extends:
        LogCollector

    """
    log_entities = [
        FileLog(name='system.log',
                command="sudo journalctl --no-tail --no-pager",
                search_locally=True),
        CommandLog(name='aprom.log',
                   command='docker logs --details -t aprom'),
        CommandLog(name='agraf.log',
                   command='docker logs --details -t agraf'),
        CommandLog(name='aalert.log',
                   command='docker logs --details -t aalert'),
        CommandLog(name='scylla-manager.yaml',
                   command=f'cat {SCYLLA_MANAGER_YAML_PATH}'),
        FileLog(name='scylla_manager.log',
                command='sudo journalctl -u scylla-manager.service --no-tail',
                search_locally=True),
        FileLog(name='manager_scylla_backend.log',
                command='sudo journalctl -u scylla-server.service -u scylla-jmx.service --no-tail',
                search_locally=True),
        PrometheusSnapshots(name='prometheus_data'),
        MonitoringStack(name='monitoring-stack'),
        GrafanaScreenShot(name='grafana-screenshot'),
    ]
    cluster_log_type = "monitor-set"
    cluster_dir_prefix = "monitor-set"
    collect_timeout = 3600


class SirenManagerLogCollector(LogCollector):
    log_entities = [
        FileLog(name="system.log",
                command="sudo journalctl --no-tail --no-pager",
                search_locally=True),
        FileLog(name="scylla_manager.log",
                command="sudo journalctl -u scylla-manager.service --no-tail",
                search_locally=True),
    ]
    cluster_log_type = "siren-manager-set"
    cluster_dir_prefix = "siren-manager-set"
    collect_timeout = 3600


class BaseSCTLogCollector(LogCollector):
    """logs for hydra test run

    Find and collect local version of files
    for sct

    Extends:
        LogCollector

    """
    log_entities = [
        FileLog(name='profile.stats',
                search_locally=True),
        FileLog(name='email_data.json',
                search_locally=True),
        FileLog(name='left_processes.log',
                search_locally=True),
        FileLog(name='raw_events.log',
                search_locally=True),
        FileLog(name='events.log',
                search_locally=True),
        FileLog(name='output.log',
                search_locally=True),
        FileLog(name='critical.log',
                search_locally=True),
        FileLog(name='warning.log',
                search_locally=True),
        FileLog(name='error.log',
                search_locally=True),
        FileLog(name='normal.log',
                search_locally=True),
        FileLog(name='debug.log',
                search_locally=True),
        FileLog(name='summary.log',
                search_locally=True),
        FileLog(name='cdc-replicator.log',
                search_locally=True),
        FileLog(name='scylla-migrate.log',
                search_locally=True),
        FileLog(name='argus.log',
                search_locally=True),
        FileLog(name=r'*debug.json',
                search_locally=True),
        FileLog(name='result_gradual_increase.log'),
        FileLog(name='partition_range_scan_diff_*.log',
                search_locally=True),
        FileLog(name='junit.xml',
                search_locally=True),
    ]
    cluster_log_type = 'sct-runner-events'
    cluster_dir_prefix = 'sct-runner-events'
    too_big_log_size = 3*1024*1024*1024

    def collect_logs(self, local_search_path: Optional[str] = None) -> list[str]:
        for ent in self.log_entities:
            ent.collect(None, self.local_dir, None, local_search_path=local_search_path)
        if not os.listdir(self.local_dir):
            LOGGER.warning('No any local files')
            LOGGER.info('Searching on builders')
            builders = get_builder_by_test_id(self.test_id)

            for obj in builders:
                builder = CollectingNode(name=obj['builder']['name'],
                                         ssh_login_info={
                                             "hostname": obj['builder']['public_ip'],
                                             "user": obj['builder']['user'],
                                             "key_file": obj["builder"]['key_file']},
                                         instance=None,
                                         global_ip=obj['builder']['public_ip'])
                for ent in self.log_entities:
                    ent.collect_from_builder(builder, self.local_dir, obj["path"])

            if not os.listdir(self.local_dir):
                LOGGER.warning('Nothing found')
                return []

        return self.create_archive_and_upload()

    def get_files_size(self) -> int:
        total_size = 0
        for dirpath, _, filenames in os.walk(self.local_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                # skip if it is symbolic link
                if not os.path.islink(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size

    @property
    def is_collect_to_a_single_archive(self) -> bool:
        return self.get_files_size() < self.too_big_log_size

    def create_single_archive_and_upload(self) -> list[str]:
        final_archive = self.archive_to_tarfile(self.local_dir)

        s3_link = upload_archive_to_s3(final_archive, f"{self.test_id}/{self.current_run}")
        remove_files(self.local_dir)
        remove_files(final_archive)
        return [s3_link]

    def create_archive_per_file_and_upload(self) -> list[str]:
        s3_links = []
        for root, _, files in os.walk(self.local_dir):
            for current_file in files:
                file_path = os.path.join(root, current_file)
                LOGGER.info(file_path)
                file_archive = self.archive_to_tarfile(file_path, add_test_id_to_archive=True)
                LOGGER.info(file_archive)
                s3_link = upload_archive_to_s3(file_archive, f"{self.test_id}/{self.current_run}")
                s3_links.append(s3_link)
                remove_files(file_path)
                remove_files(file_archive)
        return s3_links

    def create_archive_and_upload(self) -> list[str]:
        if self.is_collect_to_a_single_archive:
            s3_links = self.create_single_archive_and_upload()
        else:
            LOGGER.info("SCT log files are too big, uploading them separately")
            s3_links = self.create_archive_per_file_and_upload()

        return s3_links


class PythonSCTLogCollector(BaseSCTLogCollector):
    log_entities = [
        FileLog(name='sct.log', search_locally=True),
    ]
    cluster_log_type = 'sct-runner-python-log'
    cluster_dir_prefix = 'sct-runner-python-log'

    def _compress_file(self, src_path: str, src_name: str) -> list[str]:
        """
        In case of big SCT.log file, it will be split into several GZ files by bash script
        that uses minimum disc space to mitigate concern: larger SCT disks translates to higher costs.
        Also, GZ is used instead of TAR.GZ because of the same concern
        and because the TAR cli cannot be used to read the anonymous pipeline,
        only files, that will double the used space.
        """
        if os.path.getsize(src_path) < self.too_big_log_size:
            return [super()._compress_file(src_path, src_name)]
        else:
            runner = LocalCmdRunner()
            runner.run(
                f"bash {os.path.join(os.path.dirname(__file__), 'log_archive.sh')} "
                f"{src_path} {self.too_big_log_size} {src_name}")
            res = runner.run(f"ls *{src_name}.zst")
            return res.stdout.rstrip("\n").split("\n")

    def create_archive_and_upload(self) -> list[str]:
        file_archives = self.archive_to_tarfile(os.path.join(self.local_dir, "sct.log"), add_test_id_to_archive=True)
        s3_links = []
        for file_archive in file_archives:
            s3_links.append(upload_archive_to_s3(file_archive, f"{self.test_id}/{self.current_run}"))
            remove_files(file_archive)
        remove_files(self.local_dir)
        return s3_links


class KubernetesAPIServerLogCollector(BaseSCTLogCollector):
    """Gather K8S API server logs."""
    log_entities = [
        DirLog(name='kube-apiserver/*', search_locally=True),
    ]
    audit_file_name_prefix = "kube-apiserver-audit"
    api_call_stats_filename = "api-call-stats.json"

    cluster_log_type = "kubernetes-apiserver"
    cluster_dir_prefix = "k8s-"
    collect_timeout = 600

    def collect_logs(self, local_search_path: Optional[str] = None) -> list[str]:
        logfiles, apiserver_logdir = [], ""
        for ent in self.log_entities:
            output = ent.collect(None, self.local_dir, None, local_search_path=local_search_path)
            if isinstance(output, list):
                logfiles.extend(output)
            elif isinstance(output, str):
                logfiles.append(output)
        for logfile in logfiles:
            if 'kube-apiserver' in logfile:
                apiserver_logdir = logfile.rsplit("/", maxsplit=1)[0]
                break
        if not apiserver_logdir:
            LOGGER.warning("No log files found at '%s' related to K8S API server", local_search_path)
            return []
        else:
            LOGGER.info("Found K8S API server logs at '%s'. Calculating API call stats", apiserver_logdir)
            self.generate_apiserver_call_stats_file(apiserver_logdir)
            return self.create_single_archive_and_upload()

    def generate_apiserver_call_stats_file(self, apiserver_logdir: str) -> None:  # pylint: disable=no-self-use
        dst_file_path = f"{apiserver_logdir}/{self.api_call_stats_filename}"
        results, results_list = {}, []

        if os.path.exists(dst_file_path):
            LOGGER.warning("Kube API call stats file already exists at '%s'", dst_file_path)
            return
        if not os.path.exists(apiserver_logdir):
            LOGGER.warning(
                "Kube API server logs dir '%s' doesn't exist. Cannot calculate Kube API call stats",
                apiserver_logdir)
            return

        # Parse the K8S API server logs
        for log_file in os.listdir(apiserver_logdir):
            if not log_file.startswith(self.audit_file_name_prefix):
                continue
            with open(f"{apiserver_logdir}/{log_file}", mode="r", encoding="utf-8") as current_log_file:
                for line in current_log_file.readlines():
                    data = json.loads(line)
                    username = data.get("user", {}).get('username')
                    if not username:
                        continue
                    if username not in results:
                        results[username] = {"user": username, "total": 0, "verbs": {}}
                    verb = data.get('verb')
                    if not verb:
                        continue
                    if verb not in results[username]["verbs"]:
                        results[username]["verbs"][verb] = 1
                    else:
                        results[username]["verbs"][verb] += 1
                    results[username]["total"] += 1

        # Sort the end data
        for result in results.values():
            result['verbs'] = OrderedDict(sorted(result['verbs'].items()))
            results_list.append(result)
        results_list = sorted(results_list, key=lambda datum: datum['total'], reverse=True)

        # Write the end data to the file
        with open(dst_file_path, mode='w', encoding="utf-8") as dst_file:
            dst_file.write(json.dumps(results_list, indent=2, sort_keys=False))
        LOGGER.info("Created K8S API call stats file at '%s'.", dst_file_path)


class KubernetesLogCollector(BaseSCTLogCollector):
    """Gather K8S logs."""
    log_entities = [
        DirLog(name='cert_manager.log', search_locally=True),
        DirLog(name='scylla_manager.log', search_locally=True),
        DirLog(name='scylla_operator.log', search_locally=True),
        DirLog(name='haproxy_ingress.log', search_locally=True),
        DirLog(name='*_cluster_events.log', search_locally=True),
        DirLog(name='kubectl.version', search_locally=True),
        DirLog(name='cluster-scoped-resources/*', search_locally=True),
        DirLog(name='namespace-scoped-resources/*', search_locally=True),
    ]
    cluster_log_type = "kubernetes"
    cluster_dir_prefix = "k8s-"
    collect_timeout = 600

    def _find_test_run_subdir_by_test_id(self, base_logdir) -> str:
        for sub_dir in next(os.walk(base_logdir))[1]:
            if sub_dir == 'latest':
                continue
            sub_test_id_file = os.path.join(base_logdir, sub_dir, "test_id")
            if not os.path.isfile(sub_test_id_file):
                continue
            with open(sub_test_id_file, mode='r', encoding="utf8") as test_id_file:
                if test_id_file.read().strip() == self.test_id:
                    return os.path.join(base_logdir, sub_dir)
        return ""

    def _find_k8s_subdirs(self, test_run_logdir) -> str:
        k8s_subdirs = []
        for sub_dir in next(os.walk(test_run_logdir))[1]:
            if self.test_id[:6] not in sub_dir:
                continue
            sub_files = next(os.walk(os.path.join(test_run_logdir, sub_dir)))[2]
            if any(file_name in sub_files for file_name in ('cert_manager.log', 'scylla_operator.log')):
                k8s_subdirs.append(os.path.join(test_run_logdir, sub_dir))
        return k8s_subdirs

    def collect_logs(self, local_search_path: Optional[str] = None) -> list[str]:
        try:
            base_logdir = TestConfig.base_logdir()
            test_run_logdir = self._find_test_run_subdir_by_test_id(base_logdir)
            if not test_run_logdir:
                return super().collect_logs(local_search_path=local_search_path)
            k8s_logdirs = self._find_k8s_subdirs(test_run_logdir)
            with environment(_SCT_TEST_LOGDIR=test_run_logdir):
                for k8s_logdir in k8s_logdirs:
                    if os.path.isdir(k8s_logdir) and not os.path.isdir(
                            os.path.join(k8s_logdir, "namespaces")):
                        for _, _, current_k8s_logdir_sub_file in next(
                                os.walk(os.path.join(k8s_logdir, '.kube'))):
                            if not current_k8s_logdir_sub_file.endswith("config"):
                                continue
                            with environment(KUBECONFIG=os.path.join(
                                    k8s_logdir, '.kube', current_k8s_logdir_sub_file)):
                                KubernetesOps.gather_k8s_logs(k8s_logdir)
                                break
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.warning("Got following failure processing the K8S logs: %s", exc)
        return super().collect_logs(local_search_path=local_search_path)


class KubernetesMustGatherLogCollector(BaseSCTLogCollector):
    """Gather K8S logs using the 'must-gather' scylla-operator command."""
    log_entities = [
        DirLog(name='must-gather/*', search_locally=True),
    ]
    cluster_log_type = "kubernetes-must-gather"
    cluster_dir_prefix = "k8s-"
    collect_timeout = 600

    def collect_logs(self, local_search_path: Optional[str] = None) -> list[str]:
        return super().collect_logs(local_search_path=local_search_path)


class JepsenLogCollector(LogCollector):
    cluster_log_type = "jepsen-data"
    cluster_dir_prefix = "jepsen-data"

    def collect_logs(self, local_search_path: Optional[str] = None) -> list[str]:
        s3_link = []
        if self.nodes:
            jepsen_node = self.nodes[0]
            if jepsen_archive := self.archive_log_remotely(jepsen_node, "./jepsen-scylla", "jepsen-data"):
                self.receive_log(jepsen_node, jepsen_archive, self.local_dir)
                s3_link.append(upload_archive_to_s3(
                    archive_path=os.path.join(self.local_dir, os.path.basename(jepsen_archive)),
                    storing_path=f"{self.test_id}/{self.current_run}",
                ))
            remove_files(self.local_dir)
        return s3_link


class ParallelTimelinesReportCollector(BaseSCTLogCollector):
    """
    Collect HTML file with parallel timelines report and upload it to S3
    """
    log_entities = [
        FileLog(name='parallel-timelines-report.html',
                search_locally=True)
    ]
    cluster_log_type = "parallel-timelines-report"
    cluster_dir_prefix = "parallel-timelines-report"

    @property
    def is_collect_to_a_single_archive(self) -> bool:
        return True


class SSTablesCollector(BaseSCTLogCollector):
    """
    Collect corrupted sstables from db node.
    """
    cluster_log_type = "corrupted-sstables"
    cluster_dir_prefix = "corrupted-sstables"
    sstable_path_regexp = re.compile(r'[./\w\-]+\.db')

    def get_sstable_details(self, error_msg: str) -> Tuple[str, str, str, str]:
        sstable_path = self.sstable_path_regexp.findall(error_msg)[0]
        data_path, keyspace, table_dir, sstable_name = sstable_path.rsplit("/", 3)
        return f"{data_path}/{keyspace}/{table_dir}", keyspace, table_dir.split("-")[0], sstable_name

    def collect_logs(self, local_search_path: Optional[str] = None) -> list[str]:  # pylint: disable=too-many-locals
        try:
            raw_events_file_path = Path(self.local_dir).parent.parent.parent / EVENTS_LOG_DIR / RAW_EVENTS_LOG
            with open(raw_events_file_path, "r", encoding="utf-8") as events_file:
                for raw_line in events_file.readlines():
                    event = json.loads(raw_line)
                    if event.get("type") == "CORRUPTED_SSTABLE":
                        try:
                            sstable_dir, keyspace, table_name, sstable_name = self.get_sstable_details(
                                event.get("line"))
                            node_name = event.get("node")
                            break
                        except IndexError:
                            LOGGER.warning("Couldn't get sstable details from event line.")
                else:
                    LOGGER.info("CORRUPTED_SSTABLE error event not found. Skipping sstables collection.")
                    return []
            node: CollectingNode = [node for node in self.nodes if node.name == node_name][0]
            LOGGER.info("Collecting sstables for node %s...", node.name)
            result = node.remoter.run(f"nodetool snapshot {keyspace} -cf {table_name}")
            try:
                snapshot_dir = result.stdout.split("Snapshot directory: ")[1].strip()
            except IndexError:
                LOGGER.error("Cannot extract snapshot directory from stdout: %s", result.stdout)
                return []
            snapshot_path = f"{sstable_dir}/snapshots/{snapshot_dir}"
            s3_link = upload_remote_files_directly_to_s3(
                node.ssh_login_info, [snapshot_path], s3_bucket=S3Storage.bucket_name,
                s3_key=f"{self.test_id}/{self.current_run}/corrupted-sstables-{keyspace}-{table_name}.tar.gz",
                max_size_gb=20, public_read_acl=True)
            if not s3_link:
                # upload malformed sstable along with several others and schema file
                malformed_files = node.remoter.run(
                    f"ls {snapshot_path}/{sstable_name.rsplit('-', 1)[0]}*", ignore_status=True).stdout.split()
                recent_sstables = node.remoter.run(f"ls -t {snapshot_path}/m?-* | head -n30").stdout.split()
                s3_link = upload_remote_files_directly_to_s3(
                    node.ssh_login_info, malformed_files + recent_sstables + [f"{snapshot_path}/schema.cql"],
                    s3_bucket=S3Storage.bucket_name,
                    s3_key=f"{self.test_id}/{self.current_run}/corrupted-sstables-{keyspace}-{table_name}.tar.gz",
                    max_size_gb=80, public_read_acl=True)
        except Exception as error:  # pylint: disable=broad-except
            LOGGER.exception("failed collecting malformed sstables:\n%s", error, exc_info=error)
            return []
        return [s3_link]


class SSLConfCollector(BaseSCTLogCollector):
    """
    Cluster SSL/TLS configuration collector.

    Collects all SSL/TLS artifacts generated by the SCT runner for the test.
    This includes:
        - Certificates and keys for nodes and the CA
        - Java truststore for c-s commands
        - cqlshrc configuration file
    """
    log_entities = [
        DirLog(name='ssl_conf/*', search_locally=True),
    ]
    cluster_log_type = 'ssl-conf'


class Collector:  # pylint: disable=too-many-instance-attributes,
    """Collector instance

    Collector instance which should be run to collect logs and additional info
    as separate stage in pipeline on as subcommand in hydra

    """

    def __init__(self, test_id=None, test_dir=None, params=None):
        """Constructor of Collector object

        Build Collector instance to run collecting log processes for running instances

        Keyword Arguments:
            test_id {str} -- Test Id. Name of hydra docker or from log file. if
                             not defined, then test id will be search localy (default: {None})
            test_dir {str} -- where to search and store log files (default: {None})
            params  {SCTConfiguration}  -- SCTConfiguration object (sdcm/sct_config.py)
        """

        self.base_dir = os.environ.get('HOME')
        self._test_id = test_id
        self.backend = params['cluster_backend']
        self.params = params
        self.storage_dir = None
        self._test_dir = test_dir
        self.db_cluster = []
        self.monitor_set = []
        self.siren_manager_set = []
        self.loader_set = []
        self.kubernetes_set = []
        self.sct_set = []
        self.pt_report_set = []
        self.cluster_log_collectors = {}
        if self.backend.startswith("k8s"):
            self.cluster_log_collectors |= {
                KubernetesLogCollector: self.kubernetes_set,
                KubernetesMustGatherLogCollector: self.kubernetes_set,
                KubernetesAPIServerLogCollector: self.kubernetes_set,
            }
        self.cluster_log_collectors |= {
            ScyllaLogCollector: self.db_cluster,
            BaseSCTLogCollector: self.sct_set,
            PythonSCTLogCollector: self.sct_set,
            LoaderLogCollector: self.loader_set,
            MonitorLogCollector: self.monitor_set,
            SirenManagerLogCollector: self.siren_manager_set,
            SSTablesCollector: self.db_cluster,
            JepsenLogCollector: self.loader_set,
            ParallelTimelinesReportCollector: self.pt_report_set,
        }
        if self.params.get('server_encrypt') or self.params.get('client_encrypt'):
            self.cluster_log_collectors[SSLConfCollector] = self.sct_set

    @property
    def test_id(self):
        return self._test_id

    @cached_property
    def tags(self):
        return {"RunByUser": get_username(),
                "TestId": self.test_id,
                "keep_action": "terminate", }

    @property
    def sct_result_dir(self):
        return self._test_dir if self._test_dir else os.path.join(self.base_dir, "sct-results")

    def define_test_id(self):
        if not self._test_id:
            self._test_id = search_test_id_in_latest(self.sct_result_dir)

    def find_and_append_cloud_manager_instance_to_collecting_nodes(self):
        try:
            from cluster_cloud import get_manager_instance_by_cluster_id  # pylint: disable=import-outside-toplevel
        except ImportError:
            LOGGER.error("Couldn't collect Siren manager logs, cluster_cloud module isn't installed")
            return

        cloud_cluster_id = self.params.get("cloud_cluster_id")
        if not cloud_cluster_id:
            LOGGER.error("Cloud cluster id is not found. Probably the cluster has not been created")
            return

        try:
            instance = get_manager_instance_by_cluster_id(cluster_id=cloud_cluster_id)
            if not instance:
                raise ValueError(f"Cloud manager for the cluster {cloud_cluster_id} not found")
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.error("Failed to get cloud manager instance. Error: %s", exc)
            return

        LOGGER.info("Manager instance: %s", instance)

        ip_address = instance["privateip"] if instance["publicip"].startswith("proxy://") else instance["publicip"]
        ssh_login_info = {
            "hostname": ip_address,
            "user": "centos",
            "key_file": self.params.get("cloud_credentials_path"),
        }

        LOGGER.info("Manager instance ssh_login_info: %s", ssh_login_info)

        self.siren_manager_set.append(
            CollectingNode(
                name=instance["externalid"],
                ssh_login_info=ssh_login_info,
                instance=instance,
                global_ip=ip_address,
            ),
        )

    def get_aws_ip_address(self, instance):
        return instance['PublicIpAddress'] if ssh_connection_ip_type(self.params) == 'public' else instance['PrivateIpAddress']

    def get_aws_instances_by_testid(self):
        instances = list_instances_aws({"TestId": self.test_id}, running=True)
        filtered_instances = filter_aws_instances_by_type(instances)
        for instance in filtered_instances['db_nodes']:
            name = [tag['Value']
                    for tag in instance['Tags'] if tag['Key'] == 'Name']
            self.db_cluster.append(CollectingNode(name=name[0],
                                                  ssh_login_info={
                                                      "hostname": self.get_aws_ip_address(instance),
                                                      "user": self.params.get('ami_db_scylla_user'),
                                                      "key_file": self.params.get('user_credentials_path')},
                                                  instance=instance,
                                                  global_ip=self.get_aws_ip_address(instance),
                                                  tags={**self.tags, "NodeType": "scylla-db", }))
        for instance in filtered_instances['monitor_nodes']:
            name = [tag['Value']
                    for tag in instance['Tags'] if tag['Key'] == 'Name']
            self.monitor_set.append(CollectingNode(name=name[0],
                                                   ssh_login_info={
                                                       "hostname": self.get_aws_ip_address(instance),
                                                       "user": self.params.get('ami_monitor_user'),
                                                       "key_file": self.params.get('user_credentials_path')},
                                                   instance=instance,
                                                   global_ip=self.get_aws_ip_address(instance),
                                                   tags={**self.tags, "NodeType": "monitor", }))
        if self.params.get("use_cloud_manager"):
            self.find_and_append_cloud_manager_instance_to_collecting_nodes()

        for instance in filtered_instances['loader_nodes']:
            name = [tag['Value']
                    for tag in instance['Tags'] if tag['Key'] == 'Name']
            self.loader_set.append(CollectingNode(name=name[0],
                                                  ssh_login_info={
                                                      "hostname": self.get_aws_ip_address(instance),
                                                      "user": self.params.get('ami_loader_user'),
                                                      "key_file": self.params.get('user_credentials_path')},
                                                  instance=instance,
                                                  global_ip=self.get_aws_ip_address(instance),
                                                  tags={**self.tags, "NodeType": "loader", }))
        for instance in filtered_instances['kubernetes_nodes']:
            name = [tag['Value']
                    for tag in instance['Tags'] if tag['Key'] == 'Name']
            self.kubernetes_set.append(CollectingNode(name=name[0],
                                                      ssh_login_info={
                "hostname": self.get_aws_ip_address(instance),
                "user": self.params.get('ami_loader_user'),
                "key_file": self.params.get('user_credentials_path')},
                instance=instance,
                global_ip=self.get_aws_ip_address(instance),
                tags={**self.tags, "NodeType": "loader", }))

    def get_gce_ip_address(self, instance):
        return gce_public_addresses(instance)[0] if ssh_connection_ip_type(self.params) == 'public' else gce_private_addresses(instance)[0]

    def get_gce_instances_by_testid(self):
        instances = list_instances_gce({"TestId": self.test_id}, running=True)
        filtered_instances = filter_gce_instances_by_type(instances)
        for instance in filtered_instances['db_nodes']:
            self.db_cluster.append(CollectingNode(name=instance.name,
                                                  ssh_login_info={
                                                      "hostname": self.get_gce_ip_address(instance),
                                                      "user": self.params.get('gce_image_username'),
                                                      "key_file": self.params.get('user_credentials_path')},
                                                  instance=instance,
                                                  global_ip=self.get_gce_ip_address(instance),
                                                  tags={**self.tags, "NodeType": "scylla-db", }))
        for instance in filtered_instances['monitor_nodes']:
            self.monitor_set.append(CollectingNode(name=instance.name,
                                                   ssh_login_info={
                                                       "hostname": self.get_gce_ip_address(instance),
                                                       "user": self.params.get('gce_image_username'),
                                                       "key_file": self.params.get('user_credentials_path')},
                                                   instance=instance,
                                                   global_ip=self.get_gce_ip_address(instance),
                                                   tags={**self.tags, "NodeType": "monitor", }))
        for instance in filtered_instances['loader_nodes']:
            self.loader_set.append(CollectingNode(name=instance.name,
                                                  ssh_login_info={
                                                      "hostname": self.get_gce_ip_address(instance),
                                                      "user": self.params.get('gce_image_username'),
                                                      "key_file": self.params.get('user_credentials_path')},
                                                  instance=instance,
                                                  global_ip=self.get_gce_ip_address(instance),
                                                  tags={**self.tags, "NodeType": "loader", }))
        for instance in filtered_instances['kubernetes_nodes']:
            self.kubernetes_set.append(CollectingNode(name=instance.name,
                                                      ssh_login_info={
                                                          "hostname": self.get_gce_ip_address(instance),
                                                          "user": self.params.get('gce_image_username'),
                                                          "key_file": self.params.get('user_credentials_path')},
                                                      instance=instance,
                                                      global_ip=self.get_gce_ip_address(instance),
                                                      tags={**self.tags, "NodeType": "loader", }))
        if self.params.get("use_cloud_manager"):
            self.find_and_append_cloud_manager_instance_to_collecting_nodes()

    def create_collecting_nodes(self):
        try:
            provisioners = provisioner_factory.discover_provisioners(backend=self.backend, test_id=self.test_id)
            instances = sum([provisioner.list_instances() for provisioner in provisioners], [])
            collecting_nodes = [CollectingNode(name=instance.name,
                                               ssh_login_info={
                                                   "hostname": instance.public_ip_address or instance.private_ip_address,
                                                   "user": instance.user_name,
                                                   "key_file": f"~/.ssh/{instance.ssh_key_name}"},
                                               instance=instance,
                                               global_ip=instance.public_ip_address,
                                               tags=instance.tags) for instance in instances]
            for c_node in collecting_nodes:
                match c_node.tags.get("NodeType"):
                    case "scylla-db" | "oracle-db":
                        self.db_cluster.append(c_node)
                    case "monitor":
                        self.monitor_set.append(c_node)
                    case "loader":
                        self.loader_set.append(c_node)
            if self.params.get("use_cloud_manager"):
                self.find_and_append_cloud_manager_instance_to_collecting_nodes()
        except ProvisionerError:
            LOGGER.debug("get_running_cluster_sets: unknown backend type: %s", self.backend)

    def get_docker_instances_by_testid(self):
        instances = list_instances_gce({"TestId": self.test_id}, running=True)
        filtered_instances = filter_gce_instances_by_type(instances)
        for instance in filtered_instances['db_nodes']:
            self.db_cluster.append(CollectingNode(name=instance.name,
                                                  ssh_login_info={
                                                      "hostname": self.get_gce_ip_address(instance),
                                                      "user": 'scylla-test',
                                                      "key_file": self.params.get('user_credentials_path')},
                                                  instance=instance,
                                                  global_ip=self.get_gce_ip_address(instance),
                                                  tags={**self.tags, "NodeType": "scylla-db", }))
        self.monitor_set.append(CollectingNode(name=f"monitor-node-{self.test_id}-0",
                                               global_ip='127.0.0.1',
                                               grafana_ip=get_docker_bridge_gateway(LocalCmdRunner()),
                                               tags={**self.tags, "NodeType": "monitor", }))
        for instance in filtered_instances['loader_nodes']:
            self.loader_set.append(CollectingNode(name=instance.name,
                                                  ssh_login_info={
                                                      "hostname": self.get_gce_ip_address(instance),
                                                      "user": 'scylla-test',
                                                      "key_file": self.params.get('user_credentials_path')},
                                                  instance=instance,
                                                  global_ip=self.get_gce_ip_address(instance),
                                                  tags={**self.tags, "NodeType": "loader", }))

    def get_running_cluster_sets(self, backend):
        if backend in ("aws", "aws-siren", "k8s-eks"):
            self.get_aws_instances_by_testid()
        elif backend in ("gce", "gce-siren", "k8s-gke"):
            self.get_gce_instances_by_testid()
        elif backend == 'docker':
            self.get_docker_instances_by_testid()
        else:
            self.create_collecting_nodes()

    def run(self):
        """Run collect logs process as standalone operation

        Run collect log operations as standalon process or run
        as single stage in pipeline for defined cluster sets
        """
        results = {}
        self.define_test_id()
        if not self.test_id:
            LOGGER.warning("No test_id provided or found")
            return results
        self.get_running_cluster_sets(self.backend)

        local_dir_with_logs = get_testrun_dir(self.sct_result_dir, self.test_id)
        LOGGER.info("Found sct result directory with logs: %s", local_dir_with_logs)

        self.create_base_storage_dir(local_dir_with_logs)
        LOGGER.info("Created directory to storing collected logs: %s", self.storage_dir)
        for cluster_log_collector, nodes in self.cluster_log_collectors.items():
            log_collector = cluster_log_collector(nodes,
                                                  test_id=self.test_id,
                                                  storage_dir=self.storage_dir,
                                                  params=self.params)
            LOGGER.info("Start collect logs for cluster %s", log_collector.cluster_log_type)
            try:
                if result := log_collector.collect_logs(local_search_path=local_dir_with_logs):
                    results[log_collector.cluster_log_type] = result
                    LOGGER.info("collected data for %s\n%s\n", log_collector.cluster_log_type, result)
                else:
                    LOGGER.warning("There are no logs collected for %s", log_collector.cluster_log_type)
            except Exception:  # pylint: disable=broad-except  # noqa: BLE001
                LOGGER.warning(
                    "%s is not able to collect logs. Moving to the next log collector",
                    log_collector.__class__.__name__, exc_info=True)
        return results

    def create_base_storage_dir(self, test_dir=None):
        date_time_formatted = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_dir = os.path.basename(test_dir) if test_dir else date_time_formatted
        self.storage_dir = os.path.join(self.sct_result_dir, log_dir, 'collected_logs')
        os.makedirs(self.storage_dir, exist_ok=True)
        if not os.path.exists(os.path.join(os.path.dirname(self.storage_dir), "test_id")):
            with open(os.path.join(os.path.dirname(self.storage_dir), "test_id"), "w", encoding="utf-8") as f:  # pylint: disable=invalid-name
                f.write(self.test_id)


def check_archive(remoter, path: str) -> bool:
    """Ensure that given path is a good and not empty archive."""

    if path.endswith(".tar.gz"):
        cmd = f"tar tzf '{path}'"
    elif path.endswith(".tar.zst"):
        cmd = f"tar --zstd -tf '{path}'"
    elif path.endswith(".zip"):
        cmd = f"unzip -qql '{path}'"
    elif path.endswith(".gz"):
        cmd = f"gzip -t '{path}' && gzip -lv '{path}'"
    else:
        raise ValueError(f"Unsupported archive type: {path}")
    result = remoter.run(cmd, ignore_status=True)
    archive_is_ok = result.ok and bool(result.stdout.strip())
    if not archive_is_ok:
        LOGGER.error("Archive `%s' is corrupted: `%s' returns %d\n-- STDOUT: --\n%s\n\n-- STDERR: --\n%s",
                     path, cmd, result.exit_status, result.stdout, result.stderr)

    return archive_is_ok


def upload_archive_to_s3(archive_path: str, storing_path: str) -> Optional[str]:
    if not check_archive(LocalCmdRunner(), archive_path):
        LOGGER.error("File `%s' will not be uploaded", archive_path)
        return None
    return S3Storage().upload_file(file_path=archive_path, dest_dir=storing_path)
