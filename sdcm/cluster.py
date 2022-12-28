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
# Copyright (c) 2016 ScyllaDB

# pylint: disable=too-many-lines
import contextlib
import queue
import logging
import os
import shutil
import sys
import random
import re
import tempfile
import threading
import time
import traceback
import uuid
import itertools
import json
import ipaddress

from typing import List, Optional, Dict, Union, Set, Iterable, ContextManager, Any
from datetime import datetime
from textwrap import dedent
from functools import cached_property, wraps
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import packaging.version

import yaml
import requests
from paramiko import SSHException
from tenacity import RetryError
from invoke.exceptions import UnexpectedExit, Failure, CommandTimedOut
from cassandra import ConsistencyLevel
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster as ClusterDriver  # pylint: disable=no-name-in-module
from cassandra.cluster import NoHostAvailable  # pylint: disable=no-name-in-module
from cassandra.policies import RetryPolicy
from cassandra.policies import WhiteListRoundRobinPolicy, HostFilterPolicy, RoundRobinPolicy
from cassandra.query import SimpleStatement  # pylint: disable=no-name-in-module

from argus.db.cloud_types import ResourceState, CloudInstanceDetails, CloudResource
from argus.db.db_types import NemesisStatus
from sdcm.argus_test_run import ArgusTestRun
from sdcm.node_exporter_setup import NodeExporterSetup
from sdcm.db_log_reader import DbLogReader
from sdcm.mgmt import AnyManagerCluster, ScyllaManagerError
from sdcm.mgmt.common import get_manager_repo_from_defaults, get_manager_scylla_backend
from sdcm.prometheus import start_metrics_server, PrometheusAlertManagerListener, AlertSilencer
from sdcm.log import SDCMAdapter
from sdcm.provision.common.configuration_script import ConfigurationScriptBuilder
from sdcm.provision.scylla_yaml import ScyllaYamlNodeAttrBuilder
from sdcm.provision.scylla_yaml.certificate_builder import ScyllaYamlCertificateAttrBuilder

from sdcm.provision.scylla_yaml.cluster_builder import ScyllaYamlClusterAttrBuilder
from sdcm.provision.scylla_yaml.scylla_yaml import ScyllaYaml
from sdcm.provision.helpers.certificate import install_client_certificate, install_encryption_at_rest_files
from sdcm.remote import RemoteCmdRunnerBase, LOCALRUNNER, NETWORK_EXCEPTIONS, shell_script_cmd, RetryableNetworkException
from sdcm.remote.remote_file import remote_file, yaml_file_to_dict, dict_to_yaml_file
from sdcm import wait, mgmt
from sdcm.sct_config import SCTConfiguration
from sdcm.sct_events.continuous_event import ContinuousEventsRegistry
from sdcm.utils import properties
from sdcm.utils.benchmarks import ScyllaClusterBenchmarkManager
from sdcm.utils.common import (
    S3Storage,
    ScyllaCQLSession,
    PageFetcher,
    deprecation,
    get_data_dir_path,
    verify_scylla_repo_file,
    normalize_ipv6_url,
    download_dir_from_cloud,
    generate_random_string,
    prepare_and_start_saslauthd_service
)
from sdcm.utils.ci_tools import get_test_name
from sdcm.utils.distro import Distro
from sdcm.utils.install import InstallMode
from sdcm.utils.docker_utils import ContainerManager, NotFound, docker_hub_login
from sdcm.utils.health_checker import check_nodes_status, check_node_status_in_gossip_and_nodetool_status, \
    check_schema_version, check_nulls_in_peers, check_schema_agreement_in_gossip_and_peers, \
    CHECK_NODE_HEALTH_RETRIES, CHECK_NODE_HEALTH_RETRY_DELAY
from sdcm.utils.decorators import NoValue, retrying, log_run_info, optional_cached_property
from sdcm.utils.remotewebbrowser import WebDriverContainerMixin
from sdcm.test_config import TestConfig
from sdcm.utils.version_utils import SCYLLA_VERSION_RE, get_gemini_version, get_systemd_version, assume_version
from sdcm.sct_events import Severity
from sdcm.sct_events.base import LogEvent
from sdcm.sct_events.health import ClusterHealthValidatorEvent
from sdcm.sct_events.system import TestFrameworkEvent, INSTANCE_STATUS_EVENTS_PATTERNS, InfoEvent, SoftTimeoutEvent
from sdcm.sct_events.grafana import set_grafana_url
from sdcm.sct_events.database import SYSTEM_ERROR_EVENTS_PATTERNS, ScyllaHelpErrorEvent
from sdcm.sct_events.nodetool import NodetoolEvent
from sdcm.sct_events.decorators import raise_event_on_failure
from sdcm.utils.auto_ssh import AutoSshContainerMixin
from sdcm.monitorstack.ui import AlternatorDashboard
from sdcm.logcollector import GrafanaSnapshot, GrafanaScreenShot, PrometheusSnapshots, upload_archive_to_s3, \
    save_kallsyms_map
from sdcm.utils.ldap import LDAP_SSH_TUNNEL_LOCAL_PORT, LDAP_BASE_OBJECT, LDAP_PASSWORD, LDAP_USERS, \
    LDAP_PORT, DEFAULT_PWD_SUFFIX
from sdcm.utils.remote_logger import get_system_logging_thread
from sdcm.utils.scylla_args import ScyllaArgParser
from sdcm.utils.file import File
from sdcm.utils import cdc
from sdcm.coredump import CoredumpExportSystemdThread
from sdcm.keystore import KeyStore
from sdcm.paths import (
    SCYLLA_YAML_PATH,
    SCYLLA_PROPERTIES_PATH,
    SCYLLA_MANAGER_YAML_PATH,
    SCYLLA_MANAGER_AGENT_YAML_PATH,
    SCYLLA_MANAGER_TLS_CERT_FILE,
    SCYLLA_MANAGER_TLS_KEY_FILE,
)
from sdcm.sct_provision.aws.user_data import ScyllaUserDataBuilder


# Test duration (min). Parameter used to keep instances produced by tests that
# are supposed to run longer than 24 hours from being killed
SCYLLA_DIR = "/var/lib/scylla"
TEST_USER = 'scylla-test'
INSTALL_DIR = f"/home/{TEST_USER}/scylladb"

DB_LOG_PATTERN_RESHARDING_START = "(?i)database - Resharding"
DB_LOG_PATTERN_RESHARDING_FINISH = "(?i)storage_service - Restarting a node in NORMAL"

SPOT_TERMINATION_CHECK_DELAY = 5

MINUTE_IN_SEC: int = 60
HOUR_IN_SEC: int = 60 * MINUTE_IN_SEC
MAX_TIME_WAIT_FOR_NEW_NODE_UP: int = HOUR_IN_SEC * 8
MAX_TIME_WAIT_FOR_ALL_NODES_UP: int = MAX_TIME_WAIT_FOR_NEW_NODE_UP + HOUR_IN_SEC
MAX_TIME_WAIT_FOR_DECOMMISSION: int = HOUR_IN_SEC * 6

LOGGER = logging.getLogger(__name__)


def remove_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


class NodeError(Exception):

    def __init__(self, msg=None):
        super().__init__()
        self.msg = msg

    def __str__(self):
        if self.msg is not None:
            return self.msg
        else:
            return ""


class PrometheusSnapshotErrorException(Exception):
    pass


class ScyllaRequirementError(Exception):
    pass


class NodeStayInClusterAfterDecommission(Exception):
    """ raise after decommission finished but node stay in cluster"""


def prepend_user_prefix(user_prefix: str, base_name: str):
    return '%s-%s' % (user_prefix, base_name)


class UserRemoteCredentials():

    def __init__(self, key_file):
        self.type = 'user'
        self.key_file = key_file
        self.name = os.path.basename(self.key_file)
        self.key_pair_name = self.name

    def __str__(self):
        return "Key Pair %s -> %s" % (self.name, self.key_file)

    def write_key_file(self):
        pass

    def destroy(self):
        pass


class BaseNode(AutoSshContainerMixin, WebDriverContainerMixin):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    CQL_PORT = 9042
    CQL_SSL_PORT = 9142
    MANAGER_AGENT_PORT = 10001
    MANAGER_SERVER_PORT = 5080
    OLD_MANAGER_PORT = 56080

    log = LOGGER

    GOSSIP_STATUSES_FILTER_OUT = ['LEFT',    # in case the node was decommissioned
                                  'removed',  # in case the node was removed by nodetool removenode
                                  'BOOT',    # node during boot and not exists in the cluster yet and they will remain
                                             # in the gossipinfo 3 days.
                                             # It's expected behaviour and we won't send the error in this case
                                  'shutdown'  # when node was removed it may take more time to update the gossip info
                                  ]

    SYSTEM_EVENTS_PATTERNS = SYSTEM_ERROR_EVENTS_PATTERNS + INSTANCE_STATUS_EVENTS_PATTERNS

    def __init__(self, name, parent_cluster, ssh_login_info=None, base_logdir=None, node_prefix=None, dc_idx=0, rack=0):  # pylint: disable=too-many-arguments,unused-argument
        self.name = name
        self.rack = rack
        self.parent_cluster = parent_cluster  # reference to the Cluster object that the node belongs to
        self.test_config = TestConfig()
        self.ssh_login_info = ssh_login_info
        self.logdir = os.path.join(base_logdir, self.name) if base_logdir else None
        self.dc_idx = dc_idx

        self.argus_resource: CloudResource | None = None
        self._containers = {}
        self.is_seed = False

        self.remoter: Optional[RemoteCmdRunnerBase] = None

        self._spot_monitoring_thread = None
        self._journal_thread = None
        self._docker_log_process = None
        self._public_ip_address_cached = None
        self._private_ip_address_cached = None
        self._ipv6_ip_address_cached = None
        self._maximum_number_of_cores_to_publish = 10

        self.last_line_no = 1
        self.last_log_position = 0
        self._continuous_events_registry = ContinuousEventsRegistry()
        self._coredump_thread: Optional[CoredumpExportSystemdThread] = None
        self._db_log_reader_thread = None
        self._scylla_manager_journal_thread = None
        self._decoding_backtraces_thread = None
        self._init_system = None
        self.db_init_finished = False

        self._short_hostname = None
        self._alert_manager: Optional[PrometheusAlertManagerListener] = None

        self.termination_event = threading.Event()
        self.lock = threading.Lock()

        self._running_nemesis = None

        # We should disable bootstrap when we create nodes to establish the cluster,
        # if we want to add more nodes when the cluster already exists, then we should
        # enable bootstrap.
        self.enable_auto_bootstrap = True

        # If node is a replacement for a dead node, store dead node private ip here
        self.replacement_node_ip = None

        self._kernel_version = None
        self._uuid = None

    def init(self) -> None:
        if self.logdir:
            os.makedirs(self.logdir, exist_ok=True)
        self.log = SDCMAdapter(self.log, extra={"prefix": str(self)})
        if self.ssh_login_info:
            self.ssh_login_info["hostname"] = self.external_address
        self._init_remoter(self.ssh_login_info)
        # Start task threads after ssh is up, otherwise the dense ssh attempts from task
        # threads will make SCT builder to be blocked by sshguard of gce instance.
        self.wait_ssh_up(verbose=True)
        if not self.test_config.REUSE_CLUSTER:
            self.set_hostname()
            self.configure_remote_logging()
        self.start_task_threads()
        self._init_port_mapping()

        self.set_keep_alive()
        if self.node_type == "db" and not self.is_kubernetes() \
                and self.parent_cluster.params.get("print_kernel_callstack"):
            try:
                self.remoter.sudo(shell_script_cmd("""\
                echo 'kernel.perf_event_paranoid = 0' >> /etc/sysctl.conf
                sysctl -p
                """), verbose=True)
            except Exception:  # pylint: disable=broad-except
                LOGGER.error("Encountered an unhadled exception while changing 'perf_event_paranoid' value",
                             exc_info=True)
        self._init_argus_resource()

    def _init_argus_resource(self):
        try:
            run = ArgusTestRun.get()
            shards = -1 if "db-node" in self.instance_name else self.cpu_cores
            instance_details = CloudInstanceDetails(public_ip=self.public_ip_address, region=self.region,
                                                    provider=self.parent_cluster.cluster_backend,
                                                    private_ip=self.ip_address, creation_time=int(time.time()),
                                                    shards_amount=shards)
            resource = CloudResource(name=self.name, state=ResourceState.RUNNING,
                                     instance_info=instance_details, resource_type=self.node_type)
            self.argus_resource = resource
            run.run_info.resources.attach_resource(resource)
            run.save()
        except Exception:  # pylint: disable=broad-except
            LOGGER.error("Encountered an unhandled exception while interacting with Argus", exc_info=True)

    def argus_resource_set_shards(self) -> CloudResource | None:
        try:
            if not self.argus_resource:
                return None
            shards = self.scylla_shards if "db-node" in self.instance_name else self.cpu_cores
            shards = int(shards) if shards else 0
            self.argus_resource.instance_info.shards_amount = shards
            return self.argus_resource
        except Exception:  # pylint: disable=broad-except
            LOGGER.error("Encountered an unhandled exception while interacting with Argus", exc_info=True)
            return None

    def _destroy_argus_resource(self):
        try:
            run = ArgusTestRun.get()
            if self.argus_resource in run.run_info.resources.allocated_resources:
                reason = self.running_nemesis if self.running_nemesis else "GracefulShutdown"
                run.run_info.resources.detach_resource(self.argus_resource, reason=reason)
                run.save()
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error saving resource state to Argus", exc_info=True)

    def _init_remoter(self, ssh_login_info):
        self.remoter = RemoteCmdRunnerBase.create_remoter(**ssh_login_info)
        self.log.debug(self.remoter.ssh_debug_cmd())

    def _init_port_mapping(self):
        if self.test_config.IP_SSH_CONNECTIONS == 'public':
            if self.test_config.RSYSLOG_ADDRESS:
                try:
                    ContainerManager.destroy_container(self, "auto_ssh:rsyslog", ignore_keepalive=True)
                except NotFound:
                    pass
                ContainerManager.run_container(self, "auto_ssh:rsyslog",
                                               local_port=self.test_config.RSYSLOG_ADDRESS[1],
                                               remote_port=self.test_config.RSYSLOG_SSH_TUNNEL_LOCAL_PORT)
            if self.test_config.LDAP_ADDRESS and self.parent_cluster.node_type == "scylla-db":
                try:
                    ContainerManager.destroy_container(self, "auto_ssh:ldap", ignore_keepalive=True)
                except NotFound:
                    pass
                ContainerManager.run_container(self, "auto_ssh:ldap",
                                               local_port=self.test_config.LDAP_ADDRESS[1],
                                               remote_port=LDAP_SSH_TUNNEL_LOCAL_PORT)

    @property
    def region(self):
        raise NotImplementedError()

    @property
    def host_id(self):
        full_nodetool_status = self.parent_cluster.get_nodetool_status(verification_node=self)
        for data_center in full_nodetool_status:
            if self.ip_address in full_nodetool_status[data_center]:
                return full_nodetool_status[data_center][self.ip_address]['host_id']
        raise AssertionError(f"Could not find the requested node {self.ip_address} in nodetool status")

    @property
    def db_node_instance_type(self) -> Optional[str]:
        backend = self.parent_cluster.cluster_backend()
        if backend in ("aws", "aws-siren"):
            return self.parent_cluster.params.get("instance_type_db")
        elif backend == "azure":
            return self.parent_cluster.params.get('azure_instance_type_db')
        elif backend in ("gce", "gce-siren"):
            return self.parent_cluster.params.get("gce_instance_type_db")
        elif backend == "docker":
            return "docker"
        else:
            self.log.warning("Unrecognized backend type, defaulting to 'Unknown' for"
                             "db instance type.")
            return None

    @property
    def _proposed_scylla_yaml_properties(self) -> dict:
        node_params_builder = ScyllaYamlNodeAttrBuilder(params=self.parent_cluster.params, node=self)
        certificate_params_builder = ScyllaYamlCertificateAttrBuilder(params=self.parent_cluster.params, node=self)
        return node_params_builder.dict(exclude_none=True) | certificate_params_builder.dict(exclude_none=True)

    @property
    def proposed_scylla_yaml(self) -> ScyllaYaml:
        """
        A scylla yaml that is suggested for the node, it is calculated of the SCTConfiguration,
        and BaseNode entity is providing ip addresses
        """
        scylla_yml = ScyllaYaml(**self._proposed_scylla_yaml_properties)
        if self.enable_auto_bootstrap is not None:
            scylla_yml.auto_bootstrap = self.enable_auto_bootstrap
        if self.replacement_node_ip:
            scylla_yml.replace_address_first_boot = self.replacement_node_ip
        if append_scylla_yaml := self.parent_cluster.params.get('append_scylla_yaml'):
            append_scylla_yaml = yaml.safe_load(append_scylla_yaml)
            if any(substr in append_scylla_yaml for substr in (
                    "system_key_directory", "system_info_encryption", "kmip_hosts")):
                install_encryption_at_rest_files(self.remoter)
            scylla_yml.update(append_scylla_yaml)

        return scylla_yml

    def refresh_ip_address(self):
        # Invalidate ip address cache
        self._private_ip_address_cached = self._public_ip_address_cached = self._ipv6_ip_address_cached = None
        self.__dict__.pop('cql_ip_address', None)

        if self.ssh_login_info["hostname"] == self.external_address:
            return

        self.ssh_login_info["hostname"] = self.external_address
        self.remoter.stop()
        self._init_remoter(self.ssh_login_info)
        self._init_port_mapping()

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**self.parent_cluster.tags,
                "Name": str(self.name), "UserName": str(self.ssh_login_info.get('user'))}

    def _set_keep_alive(self):
        ContainerManager.set_all_containers_keep_alive(self)
        return True

    def set_keep_alive(self):
        node_type = None if self.parent_cluster is None else self.parent_cluster.node_type
        if self.test_config.should_keep_alive(node_type) and self._set_keep_alive():
            self.log.info("Keep this node alive")

    @property
    def short_hostname(self):
        if not self._short_hostname:
            try:
                self._short_hostname = self.remoter.run('hostname -s').stdout.strip()
            except Exception:  # pylint: disable=broad-except
                return "no_booted_yet"
        return self._short_hostname

    @property
    def system_log(self):
        orig_log_path = os.path.join(self.logdir, 'system.log')

        if self.test_config.RSYSLOG_ADDRESS:
            rsys_log_path = os.path.join(self.test_config.logdir(), 'hosts', self.short_hostname, 'messages.log')
            if os.path.exists(rsys_log_path) and (not os.path.islink(orig_log_path)):
                os.symlink(os.path.relpath(rsys_log_path, self.logdir), orig_log_path)
            return rsys_log_path
        else:
            return orig_log_path

    database_log = system_log

    @property
    def continuous_events_registry(self) -> ContinuousEventsRegistry:
        return self._continuous_events_registry

    @property
    def running_nemesis(self):
        return self._running_nemesis

    @running_nemesis.setter
    def running_nemesis(self, nemesis):
        """Set name of nemesis which is started on node

        Decorators:
            running_nemesis.setter

        Arguments:
            nemesis {str} -- class name of Nemesis
        """

        # Only one Nemesis could run on node. Limitation
        # of first version for X parallel Nemesis

        with self.lock:
            self._running_nemesis = nemesis

    @cached_property
    def distro(self):
        self.log.debug("Trying to detect Linux distribution...")
        _distro = Distro.from_os_release(self.remoter.run("cat /etc/os-release", ignore_status=True, retry=5).stdout)
        self.log.info("Detected Linux distribution: %s", _distro.name)
        return _distro

    @cached_property
    def is_nonroot_install(self):
        return self.parent_cluster.params.get("unified_package") \
            and self.parent_cluster.params.get("nonroot_offline_install")

    @property
    def is_client_encrypt(self):
        result = self.remoter.run(
            f"grep ^client_encryption_options: {self.add_install_prefix(SCYLLA_YAML_PATH)} "
            f"-A 3 | grep enabled | awk '{{print $2}}'", ignore_status=True)
        return 'true' in result.stdout.lower()

    @property
    def cpu_cores(self) -> Optional[int]:
        try:
            result = self.remoter.run("nproc", ignore_status=True)
            return int(result.stdout)
        except Exception as details:  # pylint: disable=broad-except
            self.log.error("Failed to get number of cores due to the %s", details)
        return None

    @property
    def scylla_shards(self):
        """
        Priority of selecting number of shards for Scylla is defined in
        <dist.common.scripts.scylla_util.scylla_cpuinfo.nr_shards> and has following order:
        1) SMP
        2) CPUSET
        3) Number of cores on the machine
        """
        return self.smp or self.cpuset or self.cpu_cores

    @property
    def cpuset(self):
        # If Scylla is installed as nonroot then we can't run scylla_prepare
        # and perftune so cpuset doesn't help (by Avi). So, Ignore it.
        if self.is_nonroot_install:
            return ''

        try:
            # Possible output of grep:
            #   'CPUSET="--cpuset 1 "'
            # or
            #   'CPUSET="--cpuset 1-7 "'
            # or
            #   'CPUSET="--cpuset 1-7,9 "'
            # or
            #   'CPUSET="--cpuset 1,9-15 "'
            # or
            #   'CPUSET="--cpuset 1-7,9-15 "'
            # or
            #   'CPUSET="--cpuset 1-7,9-15,17-23,25-31 "'
            # And so on...
            grep_result = self.remoter.run('grep "^CPUSET" /etc/scylla.d/cpuset.conf')
        except Exception as exc:  # pylint: disable=broad-except
            self.log.error(f"Failed to get CPUSET. Error: {exc}")
            return ''

        scylla_cpu_set = re.findall(r'(\d+)-(\d+)|(\d+)', grep_result.stdout)
        self.log.debug(f"CPUSET on node {self.name}: {grep_result}")
        core_counter = 0
        for range_or_singular in scylla_cpu_set:
            # Output of finding is of the following structure:
            #   [('1', '7', ''), ('', '', '15')]
            # First element is caught range, second element is caught singular
            core_counter += 1
            if not range_or_singular[2]:
                core_counter += int(range_or_singular[1]) - int(range_or_singular[0])
        return core_counter or ''

    @property
    def smp(self):
        """
        Example of SCYLLA_ARGS:
        SCYLLA_ARGS="--blocked-reactor-notify-ms 500 --abort-on-lsa-bad-alloc 1 --abort-on-seastar-bad-alloc
        --abort-on-internal-error 1 --abort-on-ebadf 1 --enable-sstable-key-validation 1 --smp 6 --log-to-syslog 1
        --log-to-stdout 0 --default-log-level info --network-stack posix"
        """
        # If Scylla is installed as nonroot, we don't append SCYLLA_ARGS in the sysconfig/scylla-server.
        # So it's not needed to check SMP
        if self.is_nonroot_install:
            return ''

        try:
            grep_result = self.remoter.sudo(f'grep "^SCYLLA_ARGS=" {self.scylla_server_sysconfig_path}')
        except Exception as exc:  # pylint: disable=broad-except
            self.log.error(f"Failed to get SCYLLA_ARGS. Error: {exc}")
            return ''

        scylla_smp = re.search(r'--smp\s(\d+)', grep_result.stdout)
        self.log.debug(f"SMP on node {self.name}: {scylla_smp.groups() if scylla_smp else scylla_smp}")

        return scylla_smp.group(1) if scylla_smp else ''

    @property
    def scylla_server_sysconfig_path(self):
        return f"/etc/{'sysconfig' if self.distro.is_rhel_like or self.distro.is_sles else 'default'}/scylla-server"

    def scylla_random_shards(self):
        max_shards = self.cpu_cores
        min_shards = round(self.cpu_cores / 2)
        scylla_shards = random.randrange(start=min_shards, stop=max_shards)
        # scylla_shards value (will be used as SMP) shouldn't exceed CPUs amount in cpuset.conf
        if self.cpuset and scylla_shards > self.cpuset:
            scylla_shards = self.cpuset

        self.log.info(f"Random shards: {scylla_shards}")
        return scylla_shards

    @property
    def is_server_encrypt(self):
        result = self.remoter.run(
            f"grep '^server_encryption_options:' {self.add_install_prefix(SCYLLA_YAML_PATH)}", ignore_status=True)
        return 'server_encryption_options' in result.stdout.lower()

    def extract_seeds_from_scylla_yaml(self):
        yaml_dst_path = os.path.join(tempfile.mkdtemp(prefix='sct'), 'scylla.yaml')
        wait.wait_for(func=self.remoter.receive_files, step=10, text='Waiting for copying scylla.yaml', timeout=300,
                      throw_exc=True, src=self.add_install_prefix(SCYLLA_YAML_PATH), dst=yaml_dst_path)
        with open(yaml_dst_path, encoding="utf-8") as yaml_stream:
            try:
                conf_dict = yaml.safe_load(yaml_stream)
            except Exception:
                yaml_stream.seek(0)
                self.log.error('Parsing failed. Scylla YAML config contents:')
                self.log.exception(yaml_stream.read())
                raise

            try:
                node_seeds = conf_dict['seed_provider'][0]['parameters'][0].get('seeds')
            except Exception as details:
                self.log.debug('Loaded YAML data structure: %s', conf_dict)
                raise ValueError('Exception determining seed node ips') from details

            if node_seeds:
                return node_seeds.split(',')
            else:
                raise Exception('Seeds not found in the scylla.yaml')

    @staticmethod
    def is_kubernetes() -> bool:
        return False

    def is_centos7(self):
        deprecation("consider to use node.distro.is_centos7 property instead")
        return self.distro.is_centos7

    def is_rhel7(self):
        deprecation("consider to use node.distro.is_rhel7 property instead")
        return self.distro.is_rhel7

    def is_rhel_like(self):
        deprecation("consider to use node.distro.is_rhel_like property instead")
        return self.distro.is_rhel_like

    def is_ubuntu14(self):
        deprecation("consider to use node.distro.is_ubuntu14 property instead")
        return self.distro.is_ubuntu14

    def is_ubuntu16(self):
        deprecation("consider to use node.distro.is_ubuntu16 property instead")
        return self.distro.is_ubuntu16

    def is_ubuntu18(self):
        deprecation("consider to use node.distro.is_ubuntu18 property instead")
        return self.distro.is_ubuntu18

    def is_ubuntu(self):
        deprecation("consider to use node.distro.is_ubuntu property instead")
        return self.distro.is_ubuntu

    def is_debian8(self):
        deprecation("consider to use node.distro.is_debian8 property instead")
        return self.distro.is_debian8

    def is_debian9(self):
        deprecation("consider to use node.distro.is_debian9 property instead")
        return self.distro.is_debian9

    def is_debian(self):
        deprecation("consider to use node.distro.is_debian property instead")
        return self.distro.is_debian

    # pylint: disable=too-many-arguments
    def pkg_install(self, pkgs, apt_pkgs=None, ubuntu14_pkgs=None, ubuntu16_pkgs=None,
                    debian8_pkgs=None, debian9_pkgs=None, ubuntu18_pkgs=None):
        """
        Support to install packages to multiple distros

        :param pkgs: default package name string
        :param apt_pkgs: special package name string for apt-get
        """
        # pylint: disable=too-many-return-statements
        # install packages for special debian like system
        if self.is_ubuntu14() and ubuntu14_pkgs:
            self.remoter.run('sudo apt-get install -y %s' % ubuntu14_pkgs)
            return
        if self.is_ubuntu16() and ubuntu16_pkgs:
            self.remoter.run('sudo apt-get install -y %s' % ubuntu16_pkgs)
            return
        if self.is_ubuntu18() and ubuntu18_pkgs:
            self.remoter.run('sudo apt-get install -y %s' % ubuntu18_pkgs)
            return
        if self.is_ubuntu14() and ubuntu14_pkgs:
            self.remoter.run('sudo apt-get install -y %s' % ubuntu14_pkgs)
            return
        if self.is_debian8() and debian8_pkgs:
            self.remoter.run('sudo apt-get install -y %s' % debian8_pkgs)
            return
        if self.is_debian9() and debian9_pkgs:
            self.remoter.run('sudo apt-get install -y %s' % debian9_pkgs)
            return

        # install general packages for debian like system
        if apt_pkgs:
            self.remoter.run('sudo apt-get install -y %s' % apt_pkgs)
            return

        if not self.is_rhel_like():
            self.log.error('Install packages for unknown distro by yum')
        self.remoter.run('sudo yum install -y %s' % pkgs)

    @staticmethod
    def is_docker() -> bool:
        return False

    @staticmethod
    def is_gce() -> bool:
        return False

    def scylla_pkg(self):
        return 'scylla-enterprise' if self.is_enterprise else 'scylla'

    def file_exists(self, file_path: str) -> Optional[bool]:
        try:
            return self.remoter.sudo(f"test -e '{file_path}'", ignore_status=True).ok
        except Exception as details:  # pylint: disable=broad-except
            self.log.error("Error checking if file %s exists: %s", file_path, details)
            return None

    def stop_network_interface(self, interface_name="eth1"):
        if self.is_rhel_like():
            shutdown_interface_command = "/sbin/ifdown {}"
        else:
            shutdown_interface_command = "ip link set {} down"
        self.remoter.sudo(shutdown_interface_command.format(interface_name))

    def start_network_interface(self, interface_name="eth1"):
        if self.is_rhel_like():
            startup_interface_command = "/sbin/ifup {}"
        elif self.distro.uses_systemd and \
                self.remoter.run("systemctl is-active --quiet systemd-networkd.service", ignore_status=True).ok:
            startup_interface_command = "networkctl reconfigure {}"
        else:
            startup_interface_command = "ip link set {} up"
        self.remoter.sudo(startup_interface_command.format(interface_name))

    @cached_property
    def is_enterprise(self):
        if self.distro.is_rhel_like:
            result = self.remoter.sudo("yum search scylla-enterprise 2>&1", ignore_status=True).stdout
            if 'One of the configured repositories failed (Extra Packages for Enterprise Linux 7 - x86_64)' in result:
                return "enterprise" in self.remoter.sudo("cat /etc/yum.repos.d/scylla.repo").stdout
            return "scylla-enterprise.x86_64" in result or "No matches found" not in result
        elif self.distro.is_sles:
            result = self.remoter.sudo("zypper search scylla-enterprise 2>&1", ignore_status=True).stdout
            return "scylla-enterprise" in result or "No matching items found" not in result
        return "scylla-enterprise" in self.remoter.sudo("apt-cache search scylla-enterprise", ignore_status=True).stdout

    @property
    def public_ip_address(self) -> Optional[str]:
        if self._public_ip_address_cached is None:
            self._public_ip_address_cached = self._get_public_ip_address()
        return self._public_ip_address_cached

    def _get_public_ip_address(self) -> Optional[str]:
        public_ips, _ = self._refresh_instance_state()
        if public_ips:
            return public_ips[0]
        else:
            return None

    @property
    def private_ip_address(self) -> Optional[str]:
        if self._private_ip_address_cached is None:
            self._private_ip_address_cached = self._get_private_ip_address()
        return self._private_ip_address_cached

    def _get_private_ip_address(self) -> Optional[str]:
        _, private_ips = self._refresh_instance_state()
        if private_ips:
            return private_ips[0]
        else:
            return None

    @property
    def ipv6_ip_address(self) -> Optional[str]:
        if self._ipv6_ip_address_cached is None:
            self._ipv6_ip_address_cached = self._get_ipv6_ip_address()
        return self._ipv6_ip_address_cached

    def _get_ipv6_ip_address(self) -> Optional[str]:
        raise NotImplementedError()

    def get_all_ip_addresses(self):
        public_ipv4_addresses, private_ipv4_addresses = self._refresh_instance_state()
        return list(set(public_ipv4_addresses + private_ipv4_addresses + [self._get_ipv6_ip_address()]))

    def _wait_public_ip(self):
        public_ips, _ = self._refresh_instance_state()
        while not public_ips:
            time.sleep(1)
            public_ips, _ = self._refresh_instance_state()

    def _wait_private_ip(self):
        _, private_ips = self._refresh_instance_state()
        while not private_ips:
            time.sleep(1)
            _, private_ips = self._refresh_instance_state()

    def _refresh_instance_state(self):
        raise NotImplementedError()

    @cached_property
    def cql_ip_address(self):
        with self.remote_scylla_yaml() as scylla_yaml:
            return scylla_yaml.broadcast_rpc_address if scylla_yaml.broadcast_rpc_address else self.ip_address

    @property
    def ip_address(self):
        if self.test_config.IP_SSH_CONNECTIONS == "ipv6":
            return self.ipv6_ip_address
        elif self.test_config.INTRA_NODE_COMM_PUBLIC:
            return self.public_ip_address
        else:
            return self.private_ip_address

    @property
    def external_address(self):
        """
        the communication address for usage between the test and the nodes
        :return:
        """
        if self.test_config.IP_SSH_CONNECTIONS == "ipv6":
            return self.ipv6_ip_address
        elif self.test_config.IP_SSH_CONNECTIONS == 'public' or self.test_config.INTRA_NODE_COMM_PUBLIC:
            return self.public_ip_address
        else:
            return self.private_ip_address

    @property
    def grafana_address(self):
        """
        the communication address for usage between the test and grafana server
        :return:
        """
        return self.external_address

    @property
    def scylla_listen_address(self) -> str:
        """The address the Scylla is bound.

        Use it for localhost connections (e.g., cqlsh)
        """
        return self.ip_address

    @property
    def instance_name(self) -> str:
        """
        Return name of the instance related to the node, when node is running in the cloud, or on the docker
        """
        return self.name

    @property
    def is_spot(self):
        return False

    def check_spot_termination(self):
        """Check if a spot instance termination was initiated by the cloud.

        :return: a delay to a next check if the instance termination was started, otherwise 0
        """
        raise NotImplementedError('Derived classes must implement check_spot_termination')

    def spot_monitoring_thread(self):
        while True:
            next_check_delay = self.check_spot_termination() or SPOT_TERMINATION_CHECK_DELAY
            if self.termination_event.wait(next_check_delay):
                self.check_spot_termination()
                break

    def start_spot_monitoring_thread(self):
        self._spot_monitoring_thread = threading.Thread(
            target=self.spot_monitoring_thread, name='SpotMonitoringThread', daemon=True)
        self._spot_monitoring_thread.start()

    @property
    def init_system(self):
        deprecation("consider to use node.distro.uses_systemd property instead")
        if self._init_system is None:
            result = self.remoter.run('journalctl --version',
                                      ignore_status=True)
            if result.exit_status == 0:
                self._init_system = 'systemd'
            else:
                self._init_system = 'sysvinit'

        return self._init_system

    def start_journal_thread(self):
        logs_transport = self.parent_cluster.params.get("logs_transport")
        self._journal_thread = get_system_logging_thread(
            logs_transport=logs_transport,
            node=self,
            target_log_file=self.system_log,
        )
        if self._journal_thread:
            self.log.debug("Use %s as logging daemon", type(self._journal_thread).__name__)
            self._journal_thread.start()
        else:
            if logs_transport == 'rsyslog':
                self.log.debug("Use no logging daemon since log transport is rsyslog")
            elif logs_transport == 'syslog-ng':
                self.log.debug("Use no logging daemon since log transport is syslog-ng")
            else:
                TestFrameworkEvent(
                    source=self.__class__.__name__,
                    source_method='start_journal_thread',
                    message="Got no logging daemon by unknown reason"
                ).publish_or_dump()

    def start_coredump_thread(self):
        self._coredump_thread = CoredumpExportSystemdThread(self, self._maximum_number_of_cores_to_publish)
        self._coredump_thread.start()

    def start_db_log_reader_thread(self):
        self._db_log_reader_thread = DbLogReader(
            system_log=self.system_log,
            remoter=self.remoter,
            node_name=str(self.name),
            system_event_patterns=SYSTEM_ERROR_EVENTS_PATTERNS,
            decoding_queue=self.test_config.DECODING_QUEUE,
            log_lines=self.parent_cluster.params.get('logs_transport') in ['rsyslog', 'syslog-ng']
        )
        self._db_log_reader_thread.start()

    def start_alert_manager_thread(self):
        self._alert_manager = PrometheusAlertManagerListener(self.external_address, stop_flag=self.termination_event)
        self._alert_manager.start()

    def silence_alert(self, alert_name, duration=None, start=None, end=None):
        return AlertSilencer(self._alert_manager, alert_name, duration, start, end)

    def __str__(self):
        return 'Node %s [%s | %s%s] (seed: %s)' % (
            self.name,
            self.public_ip_address,
            self.private_ip_address,
            " | %s" % self.ipv6_ip_address if self.test_config.IP_SSH_CONNECTIONS == "ipv6" else "",
            self.is_seed)

    def restart(self):
        raise NotImplementedError('Derived classes must implement restart')

    def hard_reboot(self):  # pylint: disable=no-self-use
        # Need to re-implement this method if the backend supports hard reboot.
        raise Exception("The backend doesn't support hard_reboot")

    def soft_reboot(self):  # pylint: disable=no-self-use
        try:
            self.remoter.run('sudo reboot', ignore_status=True, retry=0)
        except Exception:  # pylint: disable=broad-except
            pass

    @property
    def uptime(self):
        return datetime.strptime(self.remoter.run('uptime -s', ignore_status=True).stdout.strip(), '%Y-%m-%d %H:%M:%S')

    def reboot(self, hard=True, verify_ssh=True):
        pre_uptime = self.uptime

        def uptime_changed():
            try:
                post_uptime = self.uptime
                # In one job, I found the `uptime -s` result increased 1 second without real
                # reboot, it might be caused by normal timedrift. So we should not treat it as
                # reboot finish if the uptime change is very short.
                #
                # The uptime is the time kernel to start, so the normal time gap between two
                # uptime should contain system start time of first reboot, and the bios reset
                # time of second reboot. In the problem job, one complete reboot costed 3 mins
                # and 27 seconds.
                #
                # The real gap time is effected many factors (instance type, system load, cloud
                # platform load, enabled services in system, etc), so here we just expect the gap
                # time is larger than 5 seconds.
                #
                # The added a time gap check will ignore short uptime change before real reboot.
                return pre_uptime != post_uptime and (post_uptime - pre_uptime).seconds > 5
            except SSHException as ex:
                self.log.debug("Network isn't available, reboot might already start, %s" % ex)
                return False
            except Exception as ex:  # pylint: disable=broad-except
                self.log.debug('Failed to get uptime during reboot, %s' % ex)
                return False

        if hard:
            self.log.debug('Hardly rebooting node')
            self.hard_reboot()
        else:
            self.log.debug('Softly rebooting node')
            if not self.remoter.is_up(60):
                raise RuntimeError('Target host is down')
            self.soft_reboot()

        # wait until the reboot is executed
        wait.wait_for(func=uptime_changed, step=10, timeout=60*45, throw_exc=True)

        if verify_ssh:
            self.wait_ssh_up()

    @cached_property
    def node_type(self) -> 'str':
        if 'db-node' in self.name:
            self.log.info('node_type = db')
            return 'db'
        if 'monitor' in self.name:
            self.log.info('node_type = monitor')
            return 'monitor'
        if 'loader' in self.name:
            self.log.info('node_type = loader')
            return 'loader'
        return 'unknown'

    @log_run_info
    def start_task_threads(self):
        self.start_journal_thread()
        if self.is_spot or self.is_gce():
            self.start_spot_monitoring_thread()
        if self.node_type == 'db':
            self.start_coredump_thread()
            self.start_db_log_reader_thread()
        elif self.node_type == 'loader':
            self.start_coredump_thread()
        elif self.node_type == 'monitor':
            # TODO: start alert manager thread here when start_task_threads will be run after node setup
            # self.start_alert_manager_thread()
            if self.test_config.BACKTRACE_DECODING:
                self.start_decode_on_monitor_node_thread()

    def get_backtraces(self):
        if not self._coredump_thread:
            return
        self._coredump_thread.process_coredumps()

    @property
    def n_coredumps(self):
        if not self._coredump_thread:
            return 0
        return self._coredump_thread.n_coredumps

    @log_run_info
    def stop_task_threads(self):
        if self.termination_event.is_set():
            return
        self.log.info('Set termination_event')
        self.termination_event.set()
        if self._coredump_thread and self._coredump_thread.is_alive():
            self._coredump_thread.stop()
        if self._db_log_reader_thread and self._db_log_reader_thread.is_alive():
            self._db_log_reader_thread.stop()
        if self._alert_manager and self._alert_manager.is_alive():
            self._alert_manager.stop()

    @log_run_info
    def wait_till_tasks_threads_are_stopped(self, timeout: float = 120):
        await_bucket = []
        if self._spot_monitoring_thread:
            await_bucket.append(self._spot_monitoring_thread)
        if self._db_log_reader_thread:
            await_bucket.append(self._db_log_reader_thread)
        if self._alert_manager:
            await_bucket.append(self._alert_manager)
        if self._decoding_backtraces_thread:
            await_bucket.append(self._decoding_backtraces_thread)
        end_time = time.perf_counter() + timeout
        while await_bucket and end_time > time.perf_counter():
            for thread in await_bucket.copy():
                if not thread.is_alive():
                    await_bucket.remove(thread)
            time.sleep(1)

        if self._coredump_thread:
            self._coredump_thread.join(20*60)
        if self._journal_thread:
            self._journal_thread.stop(timeout // 10)
        if self._scylla_manager_journal_thread:
            self.stop_scylla_manager_log_capture(timeout // 10)
        self._decoding_backtraces_thread = None
        if self._docker_log_process:
            self._docker_log_process.kill()

    def get_installed_packages(self):
        """Get installed packages on node

        Execute package manager command and get
        list of all installed packages
        """
        if self.is_ubuntu() or self.is_debian():
            cmd = 'dpkg-query --show'
        else:
            cmd = 'rpm -qa'

        try:
            result = self.remoter.run(cmd, verbose=False)
            return result.stdout.strip()
        except Exception as details:  # pylint: disable=broad-except
            self.log.error('Error retrieving installed packages: %s',
                           details)
            return None

    def destroy(self):
        self.stop_task_threads()
        ContainerManager.destroy_all_containers(self)
        self._destroy_argus_resource()
        LOGGER.info("%s destroyed", self)

    def wait_ssh_up(self, verbose=True, timeout=500):
        text = None
        if verbose:
            text = '%s: Waiting for SSH to be up' % self
        wait.wait_for(func=self.remoter.is_up, step=10, text=text, timeout=timeout, throw_exc=True)

    def is_port_used(self, port: int, service_name: str) -> bool:
        # Check that "ss" is present and install if absent
        ss_version_result = self.remoter.run("ss --version || echo ss_not_found")
        if "ss_not_found" in ss_version_result.stdout:
            self.log.debug(
                f"Failed to get 'ss' binary version\n"
                f"stdout: {ss_version_result.stdout}\n"
                f"stderr: {ss_version_result.stderr}")
            if self.is_rhel_like():
                self.remoter.sudo("yum install -y iproute", ignore_status=True)
            elif self.distro.is_sles:
                self.remoter.sudo("zypper install -y iproute", ignore_status=True)
            else:
                self.remoter.sudo("apt-get install -y iproute2", ignore_status=True)

        try:
            # Path to `ss' is /usr/sbin/ss for RHEL-like distros and /bin/ss for Debian-based.  Unfortunately,
            # /usr/sbin is not always in $PATH, so need to set it explicitly.
            #
            # Output of `ss -ln' command in case of used port:
            #   $ ss -ln '( sport = :8000 )'
            #   Netid State      Recv-Q Send-Q     Local Address:Port                    Peer Address:Port
            #   tcp   LISTEN     0      5                      *:8000                               *:*
            #
            # And if there are no processes listening on the port:
            #   $ ss -ln '( sport = :8001 )'
            #   Netid State      Recv-Q Send-Q     Local Address:Port                    Peer Address:Port
            #
            # Can't avoid the header by using `-H' option because of ss' core on Ubuntu 18.04.
            cmd = f"PATH=/bin:/usr/sbin ss -ln '( sport = :{port} )'"
            return len(self.remoter.run(cmd, verbose=False).stdout.splitlines()) > 1
        except Exception as details:  # pylint: disable=broad-except
            self.log.error("Error checking for '%s' on port %s: %s", service_name, port, details)
            return False

    def db_up(self):
        return self.is_port_used(port=self.CQL_PORT, service_name="scylla-server")

    def jmx_up(self):
        return self.is_port_used(port=7199, service_name="scylla-jmx")

    def cs_installed(self, cassandra_stress_bin=None):
        if cassandra_stress_bin is None:
            cassandra_stress_bin = '/usr/bin/cassandra-stress'
        return self.file_exists(cassandra_stress_bin)

    @staticmethod
    def _parse_cfstats(cfstats_output):
        stat_dict = {}
        for line in cfstats_output.splitlines()[1:]:
            # Example of line of cfstats output:
            #       Space used (total): 123456
            stat_line = [element for element in line.strip().split(':') if
                         element]
            if stat_line:
                try:
                    try:
                        # Fix for space_node_threshold: if there are a few tables in the keyspace and space is used by the
                        # table, that arrives last in the cfstats output, will be less then space_node_threshold,
                        # the nemesis never will be run. Because of this, we sum space of all tables in the keyspace
                        # This function is used just for wait_total_space_used_per_node, so I fix "Space used.." statistics only
                        current_value = stat_dict[stat_line[0]] if 'Space used' in stat_line[0] \
                                                                   and stat_line[0] in stat_dict else 0
                        if '.' in stat_line[1].split()[0]:
                            stat_dict[stat_line[0]] = float(stat_line[1].split()[0]) + current_value
                        else:
                            stat_dict[stat_line[0]] = int(stat_line[1].split()[0]) + current_value
                    except IndexError:
                        continue
                except ValueError:
                    stat_dict[stat_line[0]] = stat_line[1].split()[0]
        return stat_dict

    def _get_tcpdump_logs(self, tcpdump_id):
        try:
            pcap_name = 'tcpdump-%s.pcap' % tcpdump_id
            pcap_tmp_file = os.path.join('/tmp', pcap_name)
            pcap_file = os.path.join(self.logdir, pcap_name)
            self.remoter.run('sudo tcpdump -vv -i lo port 10000 -w %s > /dev/null 2>&1' %
                             pcap_tmp_file, ignore_status=True)
            self.remoter.receive_files(src=pcap_tmp_file, dst=pcap_file)  # pylint: disable=not-callable
        except Exception as details:  # pylint: disable=broad-except
            self.log.error('Error running tcpdump on lo, tcp port 10000: %s',
                           str(details))

    def _is_storage_virtualized(self):
        return self.is_docker()

    def fstrim_scylla_disks(self):
        if not self._is_storage_virtualized():
            self.remoter.sudo("fstrim -v /var/lib/scylla")
        else:
            TestFrameworkEvent(
                source=self.__class__.__name__,
                source_method='fstrim_scylla_disks',
                message="fstrim'ming of Scylla disks was skipped",
                severity=Severity.WARNING, ).publish_or_dump()

    def get_cfstats(self, keyspace, tcpdump=False):
        def keyspace_available():
            self.run_nodetool("flush", ignore_status=True, timeout=300)
            res = self.run_nodetool(sub_cmd='cfstats', args=keyspace, ignore_status=True, timeout=300)
            return res.exit_status == 0
        tcpdump_id = uuid.uuid4()
        if tcpdump:
            self.log.info('START tcpdump thread uuid: %s', tcpdump_id)
            tcpdump_thread = threading.Thread(target=self._get_tcpdump_logs, name='TcpDumpUploadingThread',
                                              kwargs={'tcpdump_id': tcpdump_id}, daemon=True)
            tcpdump_thread.start()
        wait.wait_for(keyspace_available, timeout=600, step=60,
                      text='Waiting until keyspace {} is available'.format(keyspace), throw_exc=False)
        try:
            # Don't need NodetoolEvent when waiting for space_node_threshold before start the nemesis, not publish it
            result = self.run_nodetool(sub_cmd='cfstats', args=keyspace, timeout=300,
                                       warning_event_on_exception=(Failure, UnexpectedExit), publish_event=False)
        except (Failure, UnexpectedExit):
            self.log.error('nodetool error - see tcpdump thread uuid %s for '
                           'debugging info', tcpdump_id)
            raise
        finally:
            if tcpdump:
                self.remoter.run('sudo killall tcpdump', ignore_status=True)
                self.log.info('END tcpdump thread uuid: %s', tcpdump_id)
        return self._parse_cfstats(result.stdout)

    def wait_jmx_up(self, verbose=True, timeout=None):
        text = None
        if verbose:
            text = '%s: Waiting for JMX service to be up' % self
        wait.wait_for(func=self.jmx_up, step=60, text=text, timeout=timeout, throw_exc=True)

    def wait_jmx_down(self, verbose=True, timeout=None):
        text = None
        if verbose:
            text = '%s: Waiting for JMX service to be down' % self
        wait.wait_for(func=lambda: not self.jmx_up(), step=60, text=text, timeout=timeout, throw_exc=True)

    @property
    def uuid(self):
        if not self._uuid and not self.is_nonroot_install:
            uuid_path = '/var/lib/scylla-housekeeping/housekeeping.uuid'
            uuid_result = self.remoter.run('test -e %s' % uuid_path, ignore_status=True, verbose=True)
            uuid_exists = uuid_result.ok
            if uuid_exists:
                result = self.remoter.run('cat %s' % uuid_path, verbose=True)
                self._uuid = result.stdout.strip()
        return self._uuid

    def _report_housekeeping_uuid(self, verbose=False):
        """
        report uuid of test db nodes to ScyllaDB
        """
        mark_path = '/var/lib/scylla-housekeeping/housekeeping.uuid.marked'
        cmd = 'curl "https://i6a5h9l1kl.execute-api.us-east-1.amazonaws.com/prod/check_version?uu=%s&mark=scylla"'

        mark_exists = self.remoter.run('test -e %s' % mark_path, ignore_status=True, verbose=verbose).ok
        if self.uuid and not mark_exists:
            self.remoter.run(cmd % self.uuid, ignore_status=True)
            if self.is_docker():
                self.remoter.sudo('touch %s' % mark_path, verbose=verbose)
            else:
                self.remoter.sudo('touch %s' % mark_path, verbose=verbose, user='scylla')

    def wait_db_up(self, verbose=True, timeout=3600):
        text = None
        if verbose:
            text = '%s: Waiting for DB services to be up' % self
        wait.wait_for(func=self.db_up, step=60, text=text, timeout=timeout, throw_exc=True)
        self.db_init_finished = True
        try:
            self._report_housekeeping_uuid(verbose=True)
        except Exception as details:  # pylint: disable=broad-except
            self.log.error('Failed to report housekeeping uuid. Error details: %s', details)

    def is_manager_agent_up(self, port=None):
        port = port if port else self.MANAGER_AGENT_PORT
        # When the agent is IP, it should answer an https request of https://NODE_IP:10001/ping with status code 204
        response = requests.get(f"https://{normalize_ipv6_url(self.external_address)}:{port}/ping", verify=False)
        return response.status_code == 204

    def wait_manager_agent_up(self, verbose=True, timeout=180):
        text = None
        if verbose:
            text = '%s: Waiting for manager agent to be up' % self
        wait.wait_for(func=self.is_manager_agent_up, step=10, text=text, timeout=timeout, throw_exc=True)

    def is_manager_server_up(self, port=None):
        port = port if port else self.MANAGER_SERVER_PORT
        # When the manager has started,
        # it should answer an http request of https://127.0.0.1:5080/ping with status code 204
        # The port is only open locally, hence using curl instead
        curl_output = self.remoter.run(
            f'''curl --write-out "%{{http_code}}\n" --silent --output /dev/null "http://127.0.0.1:{port}/ping"''',
            verbose=True, ignore_status=True)
        http_status_code = int(curl_output.stdout.strip())
        return http_status_code == 204

    def wait_manager_server_up(self, verbose=True, timeout=300, port=None):
        text = None
        if verbose:
            text = '%s: Waiting for manager server to be up' % self
        try:
            wait.wait_for(func=self.is_manager_server_up, port=port,
                          step=10, text=text, timeout=timeout, throw_exc=True)
        except RetryError:
            wait.wait_for(func=self.is_manager_server_up, port=self.OLD_MANAGER_PORT,
                          step=10, text=text, timeout=timeout, throw_exc=True)

    def restart_manager_server(self, port=None):
        self.remoter.sudo("systemctl restart scylla-manager")
        self.wait_manager_server_up(port=port)

    # Configuration node-exporter.service when use IPv6

    def set_web_listen_address(self):
        node_exporter_file = '/usr/lib/systemd/system/node-exporter.service'
        find_web_param = self.remoter.run('grep "web.listen-address" %s' % node_exporter_file,
                                          ignore_status=True)
        if find_web_param.exit_status == 1:
            cmd = """sudo sh -c "sed -i 's|ExecStart=/usr/bin/node_exporter  --collector.interrupts|""" \
                  """ExecStart=/usr/bin/node_exporter  --collector.interrupts """ \
                  """--web.listen-address="[%s]:9100"|g' %s" """ % (self.ip_address, node_exporter_file)
            self.remoter.run(cmd)
            self.remoter.run('sudo systemctl restart node-exporter.service')

    def apt_running(self):
        try:
            result = self.remoter.run('sudo lsof /var/lib/dpkg/lock', ignore_status=True)
            return result.exit_status == 0
        except Exception as details:  # pylint: disable=broad-except
            self.log.error('Failed to check if APT is running in the background. Error details: %s', details)
            return False

    def wait_apt_not_running(self, verbose=True):
        text = None
        if verbose:
            text = '%s: Waiting for apt to finish running in the background' % self
        wait.wait_for(func=lambda: not self.apt_running(), step=60,
                      text=text, throw_exc=False)

    def wait_db_down(self, verbose=True, timeout=3600, check_interval=60):
        text = None
        if verbose:
            text = '%s: Waiting for DB services to be down' % self
        wait.wait_for(func=lambda: not self.db_up(), step=check_interval, text=text, timeout=timeout, throw_exc=True)

    def wait_cs_installed(self, verbose=True):
        text = None
        if verbose:
            text = '%s: Waiting for cassandra-stress' % self
        wait.wait_for(func=self.cs_installed, step=60,
                      text=text, throw_exc=False)

    def mark_log(self):
        """
        Returns "a mark" to the current position of this node Cassandra log.
        This is for use with the from_mark parameter of watch_log_for_* methods,
        allowing to watch the log from the position when this method was called.
        """
        if not os.path.exists(self.system_log):
            return 0
        with open(self.system_log, encoding="utf-8") as log_file:
            log_file.seek(0, os.SEEK_END)
            return log_file.tell()

    def follow_system_log(
            self,
            patterns: Optional[List[Union[str, re.Pattern, LogEvent]]] = None,
            start_from_beginning: bool = False
    ) -> Iterable[str]:
        stream = File(self.system_log)
        if not start_from_beginning:
            stream.move_to_end()
        if not patterns:
            patterns = [p[0] for p in SYSTEM_ERROR_EVENTS_PATTERNS]
        regexps = []
        for pattern in patterns:
            if isinstance(pattern, re.Pattern):
                regexps.append(pattern)
            elif isinstance(pattern, str):
                regexps.append(re.compile(pattern, flags=re.IGNORECASE))
            elif isinstance(pattern, LogEvent):
                regexps.append(re.compile(pattern.regex, flags=re.IGNORECASE))
        return stream.read_lines_filtered(*regexps)

    def start_decode_on_monitor_node_thread(self):
        self._decoding_backtraces_thread = threading.Thread(
            target=self.decode_backtrace, name='DecodeOnMonitorNodeThread', daemon=True)
        self._decoding_backtraces_thread.daemon = True
        self._decoding_backtraces_thread.start()

    def decode_backtrace(self):
        scylla_debug_file = None
        while True:
            event = None
            obj = None
            try:
                obj = self.test_config.DECODING_QUEUE.get(timeout=5)
                if obj is None:
                    break
                event = obj["event"]
                if not scylla_debug_file:
                    scylla_debug_file = self.copy_scylla_debug_info(obj["node"], obj["debug_file"])
                output = self.decode_raw_backtrace(scylla_debug_file, " ".join(event.raw_backtrace.split('\n')))
                event.backtrace = output.stdout
            except queue.Empty:
                pass
            except Exception as details:  # pylint: disable=broad-except
                self.log.error("failed to decode backtrace %s", details)
            finally:
                if event:
                    event.ready_to_publish()
                    event.publish()

            if self.termination_event.is_set() and self.test_config.DECODING_QUEUE.empty():
                break

    def copy_scylla_debug_info(self, node_name: str, debug_file: str):
        """Copy scylla debug file from db-node to monitor-node

        Copy via builder
        :param node_name: db node name
        :type node_name: str
        :param scylla_debug_file: path to scylla_debug_file on db-node
        :type scylla_debug_file: str
        :returns: path on monitor node
        :rtype: {str}
        """

        db_nodes = self.parent_cluster.targets['db_cluster'].nodes
        db_node = next(iter([n for n in db_nodes if n.name == node_name]), None)
        assert db_node, f"Node named: {node_name} wasn't found"

        base_scylla_debug_file = os.path.basename(debug_file)
        transit_scylla_debug_file = os.path.join(db_node.parent_cluster.logdir,
                                                 base_scylla_debug_file)
        final_scylla_debug_file = os.path.join("/tmp", base_scylla_debug_file)

        if not os.path.exists(transit_scylla_debug_file):
            db_node.remoter.receive_files(debug_file, transit_scylla_debug_file)
        res = self.remoter.run(
            "test -f {}".format(final_scylla_debug_file), ignore_status=True, verbose=False)
        if res.exited != 0:
            self.remoter.send_files(transit_scylla_debug_file,  # pylint: disable=not-callable
                                    final_scylla_debug_file)
        self.log.info("File on monitor node %s: %s", self, final_scylla_debug_file)
        self.log.info("Remove transit file: %s", transit_scylla_debug_file)
        os.remove(transit_scylla_debug_file)
        return final_scylla_debug_file

    def decode_raw_backtrace(self, scylla_debug_file, raw_backtrace):
        """run decode backtrace on monitor node

        Decode backtrace on monitor node
        :param scylla_debug_file: file path on db-node
        :type scylla_debug_file: str
        :param raw_backtrace: string with backtrace data
        :type raw_backtrace: str
        :returns: result of bactrace
        :rtype: {str}
        """
        return self.remoter.run('addr2line -Cpife {0} {1}'.format(scylla_debug_file, raw_backtrace), verbose=True)

    def get_scylla_build_id(self) -> Optional[str]:
        for scylla_executable in ("/usr/bin/scylla", "/opt/scylladb/libexec/scylla", ):
            build_id_result = self.remoter.run(f"{scylla_executable} --build-id", ignore_status=True)
            if build_id_result.ok:
                return build_id_result.stdout.strip()
        return None

    def datacenter_setup(self, datacenters):
        cmd = "sudo sh -c 'echo \"\ndc={}\nrack=RACK1\nprefer_local=true\ndc_suffix={}\n\" >> /etc/scylla/cassandra-rackdc.properties'"
        region_name = datacenters[self.dc_idx]
        ret = re.findall('-([a-z]+).*-', region_name)
        if ret:
            dc_suffix = 'scylla_node_{}'.format(ret[0])
        else:
            dc_suffix = region_name.replace('-', '_')

        cmd = cmd.format(datacenters[self.dc_idx], dc_suffix)
        self.remoter.run(cmd)

    def _remote_yaml(self, path):
        self.log.debug("Update %s YAML file", path)
        return remote_file(remoter=self.remoter,
                           remote_path=path,
                           serializer=dict_to_yaml_file,
                           deserializer=yaml_file_to_dict,
                           sudo=True)

    def _remote_properties(self, path):
        self.log.debug("Update %s properties configuration file", path)
        return remote_file(remoter=self.remoter,
                           remote_path=path,
                           serializer=properties.serialize,
                           deserializer=properties.deserialize,
                           sudo=True)

    def remote_cassandra_rackdc_properties(self):
        return self._remote_properties(path=self.add_install_prefix(abs_path=SCYLLA_PROPERTIES_PATH))

    @contextlib.contextmanager
    def remote_scylla_yaml(self) -> ContextManager[ScyllaYaml]:
        with self._remote_yaml(path=self.add_install_prefix(abs_path=SCYLLA_YAML_PATH)) as scylla_yaml:
            new_scylla_yaml = ScyllaYaml(**scylla_yaml)
            old_scylla_yaml = new_scylla_yaml.copy()
            yield new_scylla_yaml
            diff = old_scylla_yaml.diff(new_scylla_yaml)
            if not diff:
                LOGGER.debug("%s: scylla.yaml hasn't been changed", self)
                return
            scylla_yaml.clear()
            scylla_yaml.update(
                new_scylla_yaml.dict(
                    exclude_none=True, exclude_unset=True, exclude_defaults=True,
                    # NOTE: explicit fields included into yaml no matter what,
                    #  they are needed for nodetool to operate properly
                    explicit=['partitioner', 'commitlog_sync', 'commitlog_sync_period_in_ms', 'endpoint_snitch']
                )
            )
            LOGGER.debug("%s: scylla.yaml will be updated to:\n%s", self, scylla_yaml)

    def remote_manager_yaml(self):
        return self._remote_yaml(path=SCYLLA_MANAGER_YAML_PATH)

    def remote_manager_agent_yaml(self):
        return self._remote_yaml(path=SCYLLA_MANAGER_AGENT_YAML_PATH)

    def get_saslauthd_config(self):
        if self.test_config.LDAP_ADDRESS is None:
            return {}
        ldap_server_ip = '127.0.0.1' if self.test_config.IP_SSH_CONNECTIONS == 'public' \
            else self.test_config.LDAP_ADDRESS[0]
        ldap_port = LDAP_SSH_TUNNEL_LOCAL_PORT if self.test_config.IP_SSH_CONNECTIONS == 'public' \
            else self.test_config.LDAP_ADDRESS[1]
        return {'ldap_servers': f'ldap://{ldap_server_ip}:{ldap_port}/',
                'ldap_search_base': f'ou=Person,{LDAP_BASE_OBJECT}',
                'ldap_bind_dn': f'cn=admin,{LDAP_BASE_OBJECT}',
                'ldap_bind_pw': LDAP_PASSWORD}

    @staticmethod
    def get_saslauthd_ms_ad_config():
        ldap_ms_ad_credentials = KeyStore().get_ldap_ms_ad_credentials()
        ldap_server_ip = ldap_ms_ad_credentials["server_address"]
        ldap_port = LDAP_PORT
        ldap_search_base = f'OU=People,{LDAP_BASE_OBJECT}'
        ldap_bind_dn = ldap_ms_ad_credentials['ldap_bind_dn']
        ldap_bind_pw = ldap_ms_ad_credentials['admin_password']

        return {'ldap_servers': f'ldap://{ldap_server_ip}:{ldap_port}/',
                'ldap_search_base': ldap_search_base,
                'ldap_filter': '(cn=%u)',
                'ldap_bind_dn': ldap_bind_dn,
                'ldap_bind_pw': ldap_bind_pw}

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,unused-argument,too-many-branches
    def config_setup(self,
                     append_scylla_args='',
                     debug_install=False,
                     **_
                     ):
        with self.remote_scylla_yaml() as scylla_yml:
            scylla_yml.update(
                self.parent_cluster.proposed_scylla_yaml,
                self.proposed_scylla_yaml
            )

        self.process_scylla_args(append_scylla_args)

        if debug_install:
            if self.distro.is_rhel_like:
                self.remoter.sudo("yum install -y scylla-gdb", verbose=True, ignore_status=True)
            elif self.distro.is_sles:
                self.remoter.sudo("zypper install -y scylla-gdb", verbose=True, ignore_status=True)

        if self.init_system == "systemd":
            self.fix_scylla_server_systemd_config()

    def fix_scylla_server_systemd_config(self):
        systemd_version = get_systemd_version(self.remoter.run("systemctl --version", ignore_status=True).stdout)
        if systemd_version >= 240:
            self.log.debug("systemd version %d >= 240: we can change FinalKillSignal", systemd_version)
            self.remoter.sudo(shell_script_cmd("""\
                mkdir -p /etc/systemd/system/scylla-server.service.d
                cat <<EOF > /etc/systemd/system/scylla-server.service.d/override.conf
                [Service]
                FinalKillSignal=SIGABRT
                EOF
                systemctl daemon-reload
            """))

    def process_scylla_args(self, append_scylla_args=''):
        if append_scylla_args:
            scylla_help = self.remoter.run(
                f"{self.add_install_prefix('/usr/bin/scylla')} --help", ignore_status=True).stdout
            scylla_help_seastar = self.remoter.run(
                f"{self.add_install_prefix('/usr/bin/scylla')} --help-seastar", ignore_status=True).stdout
            scylla_arg_parser = ScyllaArgParser.from_scylla_help(
                help_text=f"{scylla_help}\n{scylla_help_seastar}",
                duplicate_cb=lambda dups: ScyllaHelpErrorEvent.duplicate(
                    message=f"Scylla help contains duplicate for the following arguments: {','.join(dups)}"
                ).publish()
            )
            append_scylla_args = scylla_arg_parser.filter_args(
                append_scylla_args,
                unknown_args_cb=lambda args: ScyllaHelpErrorEvent.filtered(
                    message="Following arguments are filtered out: " + ','.join(args)
                ).publish()
            )
        if self.parent_cluster.params.get('db_nodes_shards_selection') == 'random':
            append_scylla_args += f" --smp {self.scylla_random_shards()}"

        if append_scylla_args:
            self.log.debug("Append following args to scylla: `%s'", append_scylla_args)
            self.remoter.sudo(
                f"sed -i '/{append_scylla_args}/! s/SCYLLA_ARGS=\"/&{append_scylla_args} /' "
                f"{self.scylla_server_sysconfig_path}")

    def config_client_encrypt(self):
        install_client_certificate(self.remoter)

    @retrying(n=3, sleep_time=10, allowed_exceptions=(AssertionError,), message="Retrying on getting scylla repo")
    def download_scylla_repo(self, scylla_repo):
        if not scylla_repo:
            self.log.error("Scylla YUM repo file url is not provided, it should be defined in configuration YAML!!!")
            return
        if self.is_rhel_like():
            repo_path = '/etc/yum.repos.d/scylla.repo'
            self.remoter.sudo('curl --retry 5 --retry-max-time 300 -o %s -L %s' % (repo_path, scylla_repo))
            self.remoter.sudo('chown root:root %s' % repo_path)
            self.remoter.sudo('chmod 644 %s' % repo_path)
            result = self.remoter.run('cat %s' % repo_path, verbose=True)
            verify_scylla_repo_file(result.stdout, is_rhel_like=True)
        elif self.distro.is_sles:
            repo_path = '/etc/zypp/repos.d/scylla.repo'
            self.remoter.sudo('curl --retry 5 --retry-max-time 300 -o %s -L %s' % (repo_path, scylla_repo))
            self.remoter.sudo('chown root:root %s' % repo_path)
            self.remoter.sudo('chmod 644 %s' % repo_path)
            result = self.remoter.run('cat %s' % repo_path, verbose=True)
            verify_scylla_repo_file(result.stdout, is_rhel_like=True)
        else:
            repo_path = '/etc/apt/sources.list.d/scylla.list'
            self.remoter.sudo('curl --retry 5 --retry-max-time 300 -o %s -L %s' % (repo_path, scylla_repo))
            result = self.remoter.run('cat %s' % repo_path, verbose=True)
            verify_scylla_repo_file(result.stdout, is_rhel_like=False)
            self.remoter.sudo("mkdir -p /etc/apt/keyrings")
            for apt_key in self.parent_cluster.params.get("scylla_apt_keys"):
                self.remoter.sudo(f"apt-key adv --keyserver keyserver.ubuntu.com --recv-keys {apt_key}", retry=3)
                self.remoter.sudo(f"gpg --homedir /tmp --no-default-keyring --keyring /etc/apt/keyrings/scylladb.gpg "
                                  f"--keyserver hkp://keyserver.ubuntu.com:80 --recv-keys {apt_key}", retry=3)
        self.update_repo_cache()

    def download_scylla_manager_repo(self, scylla_repo: str) -> None:
        if self.distro.is_rhel_like:
            repo_path = '/etc/yum.repos.d/scylla-manager.repo'
        elif self.distro.is_sles:
            repo_path = '/etc/zypp/repos.d/scylla-manager.repo'
        else:
            repo_path = '/etc/apt/sources.list.d/scylla-manager.list'
        self.remoter.sudo(f"curl -o {repo_path} -L {scylla_repo}")

        # Prevent issue https://github.com/scylladb/scylla/issues/9683
        if self.distro.is_oel8:
            self.remoter.sudo(f"chmod 644 {repo_path}")

        if self.distro.is_debian_like:
            self.remoter.sudo("mkdir -p /etc/apt/keyrings")
            for apt_key in self.parent_cluster.params.get("scylla_apt_keys"):
                self.remoter.sudo(f"apt-key adv --keyserver keyserver.ubuntu.com --recv-keys {apt_key}", retry=3)
                self.remoter.sudo(f"gpg --homedir /tmp --no-default-keyring --keyring /etc/apt/keyrings/scylladb.gpg "
                                  f"--keyserver hkp://keyserver.ubuntu.com:80 --recv-keys {apt_key}", retry=3)
            self.remoter.sudo("apt-get update", ignore_status=True)

    @retrying(n=30, sleep_time=15, allowed_exceptions=UnexpectedExit)
    def install_package(self,
                        package_name: str,
                        wait_step: int = 30,
                        wait_timeout: int = 60,
                        wait_for_package_manager: bool = True) -> None:
        if self.distro.is_ubuntu and wait_for_package_manager:
            wait.wait_for(func=self.is_apt_lock_free, step=wait_step,
                          timeout=wait_timeout, text='Checking if package manager is free',
                          throw_exc=False)
        if self.distro.is_rhel_like:
            pkg_cmd = 'yum'
        elif self.distro.is_sles:
            pkg_cmd = 'zypper'
        else:
            pkg_cmd = 'apt-get'

            # A workaround for: https://github.com/scylladb/scylla-pkg/issues/2578
            if package_name == "scylla-manager-agent":
                self.remoter.sudo('apt --fix-broken install -y')

        self.remoter.sudo(f'{pkg_cmd} install -y {package_name}')

    def is_apt_lock_free(self) -> bool:
        result = self.remoter.sudo("lsof /var/lib/dpkg/lock", ignore_status=True)
        return result.exit_status == 1

    def install_manager_agent(self, package_path: Optional[str] = None) -> None:
        package_name = "scylla-manager-agent"
        if package_path:
            package_name = f"{package_path}scylla-manager-agent*"
        elif self.parent_cluster.params.get("scylla_mgmt_agent_address"):
            self.download_scylla_manager_repo(self.parent_cluster.params.get("scylla_mgmt_agent_address"))
        else:
            manager_version = self.parent_cluster.params.get("manager_version")
            agent_repo_url = get_manager_repo_from_defaults(manager_version, self.distro)
            self.download_scylla_manager_repo(agent_repo_url)

        self.install_package(package_name)

        tls_cert_file = tls_key_file = None
        if self.remoter.sudo("scyllamgr_ssl_cert_gen", ignore_status=True).ok:
            tls_cert_file = SCYLLA_MANAGER_TLS_CERT_FILE
            tls_key_file = SCYLLA_MANAGER_TLS_KEY_FILE

        with self.remote_manager_agent_yaml() as manager_agent_yaml:
            manager_agent_yaml["auth_token"] = self.test_config.test_id()
            manager_agent_yaml["tls_cert_file"] = tls_cert_file
            manager_agent_yaml["tls_key_file"] = tls_key_file
            manager_agent_yaml["prometheus"] = f":{self.parent_cluster.params.get('manager_prometheus_port')}"

        self.remoter.sudo(shell_script_cmd("""\
            systemctl restart scylla-manager-agent
            systemctl enable scylla-manager-agent
        """))

        manager_agent_version = self.remoter.run("scylla-manager-agent --version").stdout
        self.log.info("node %s has scylla-manager-agent version %s", self.name, manager_agent_version)

    def update_manager_agent_config(self, region: Optional[str] = None) -> None:
        backup_backend = self.parent_cluster.params.get("backup_bucket_backend")
        backup_backend_config = {}
        if backup_backend == "s3":
            if region and region != self.region:
                backup_backend_config["region"] = region
        elif backup_backend == "gcs":
            pass
        elif backup_backend == "azure":
            backup_backend_config["account"] = self.test_config.backup_azure_blob_credentials["account"]
            backup_backend_config["key"] = self.test_config.backup_azure_blob_credentials["key"]
        else:
            raise ValueError(f"{backup_backend=} is not supported")

        with self.remote_manager_agent_yaml() as manager_agent_yaml:
            manager_agent_yaml[backup_backend] = backup_backend_config

        self.remoter.sudo("systemctl restart scylla-manager-agent")
        self.wait_manager_agent_up()

    def upgrade_manager_agent(self, scylla_mgmt_address: str, start_agent_after_upgrade: bool = True) -> None:
        self.download_scylla_manager_repo(scylla_mgmt_address)
        if self.is_rhel_like():
            self.remoter.sudo("yum update scylla-manager-agent -y")
        elif self.distro.is_sles:
            self.remoter.sudo("zypper update scylla-manager-agent -y")
        else:
            self.remoter.sudo("apt-get update", ignore_status=True)
            self.remoter.sudo("apt-get install -y scylla-manager-agent")
        self.remoter.sudo("scyllamgr_agent_setup -y")
        if start_agent_after_upgrade:
            if self.is_docker():
                self.remoter.sudo("supervisorctl start scylla-manager-agent")
            else:
                self.remoter.sudo("systemctl start scylla-manager-agent")

    def clean_scylla_data(self):
        """Clean all scylla data file

        Commands are taken from instruction in docs.
        See https://docs.scylladb.com/operating-scylla/procedures/cluster-management/clear_data/
        """
        clean_commands_list = [
            "rm -rf /var/lib/scylla/data/*",
            "find /var/lib/scylla/commitlog -type f -delete",
            "find /var/lib/scylla/hints -type f -delete",
            "find /var/lib/scylla/view_hints -type f -delete"
        ]
        self.log.debug("Clean all files from scylla data dirs")
        for cmd in clean_commands_list:
            self.remoter.sudo(cmd, ignore_status=True)

    def clean_scylla(self):
        """
        Uninstall scylla
        """
        self.stop_scylla_server(verify_down=False, ignore_status=True)
        if self.is_rhel_like():
            self.remoter.sudo('yum remove -y scylla\\*')
        elif self.distro.is_sles:
            self.remoter.sudo('zypper remove -y scylla\\*', ignore_status=True)
        else:
            self.remoter.sudo('rm -f /etc/apt/sources.list.d/scylla.list')
            self.remoter.sudo('apt-get remove -y scylla\\*', ignore_status=True)
        self.update_repo_cache()
        self.clean_scylla_data()

    def update_repo_cache(self):
        try:
            if self.is_rhel_like():
                # The yum makecache command was removed from here since not needed and recommended.
                # In the past it also caused ERROR 404 of yum, reference https://wiki.centos.org/yum-errors
                # This fixes https://github.com/scylladb/scylla-cluster-tests/issues/4977
                self.remoter.sudo('yum clean all')
                self.remoter.sudo('rm -rf /var/cache/yum/')
            elif self.distro.is_sles:
                self.remoter.sudo('zypper clean all')
                self.remoter.sudo('rm -rf /var/cache/zypp/')
                self.remoter.sudo('zypper refresh', retry=3)
            else:
                self.remoter.sudo('apt-get clean all')
                self.remoter.sudo('rm -rf /var/cache/apt/')
                self.remoter.sudo('apt-get update', retry=3)
                self.remoter.run("echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections")
        except Exception as ex:  # pylint: disable=broad-except
            self.log.error('Failed to update repo cache: %s', ex)

    def upgrade_system(self):
        if self.is_rhel_like():
            # update system to latest
            result = self.remoter.run('ls /etc/yum.repos.d/epel.repo', ignore_status=True)
            if result.exit_status == 0:
                self.remoter.run('sudo yum update -y --skip-broken --disablerepo=epel', retry=3)
            else:
                self.remoter.run('sudo yum update -y --skip-broken', retry=3)
        elif self.distro.is_sles:
            self.remoter.sudo('zypper update -y -l')
        else:
            self.remoter.run(
                'sudo DEBIAN_FRONTEND=noninteractive apt-get '
                '-o Dpkg::Options::="--force-confold" '
                '-o Dpkg::Options::="--force-confdef" '
                'upgrade -y ', retry=3)
        # update repo cache after upgrade
        self.update_repo_cache()

    def install_scylla(self, scylla_repo):
        """
        Download and install scylla on node
        :param scylla_repo: scylla repo file URL
        """
        self.log.info("Installing Scylla...")
        if self.is_rhel_like():
            # `screen' package is missed in CentOS/RHEL 8. Should be installed from EPEL repository.
            if self.distro.is_centos8 or self.distro.is_rhel8 or self.distro.is_oel8 or self.distro.is_rocky8:
                self.install_epel()
            self.remoter.run("sudo yum remove -y abrt")  # https://docs.scylladb.com/operating-scylla/admin/#core-dumps
            self.remoter.run('sudo yum install -y rsync tcpdump screen')
            self.download_scylla_repo(scylla_repo)
            # hack cause of broken caused by EPEL
            self.remoter.run('sudo yum install -y python36-PyYAML', ignore_status=True)
            self.remoter.run('sudo yum install -y {}'.format(self.scylla_pkg()))
            self.remoter.run('sudo yum install -y scylla-gdb', ignore_status=True)
        elif self.distro.is_sles15:
            self.remoter.sudo('zypper install -y rsync tcpdump screen')
            self.download_scylla_repo(scylla_repo)
            # self.remoter.sudo('zypper mr -e Python_2_Module_x86_64:SLE-Module-Python2-15-SP3-Pool')
            # self.remoter.sudo('zypper mr -e Python_2_Module_x86_64:SLE-Module-Python2-15-SP3-Updates')
            # self.remoter.sudo('zypper mr -e Legacy_Module_x86_64:SLE-Module-Legacy15-SP3-Updates')
            # self.remoter.sudo('zypper mr -e Legacy_Module_x86_64:SLE-Module-Legacy15-SP3-Pool')
            self.remoter.sudo('SUSEConnect --product sle-module-legacy/15.3/x86_64')
            self.remoter.sudo('SUSEConnect --product sle-module-python2/15.3/x86_64')
            self.remoter.sudo('zypper install -y python2-PyYAML', ignore_status=True)
            self.remoter.sudo('zypper install -y python3-PyYAML', ignore_status=True)
            self.remoter.sudo('zypper install -y {}'.format(self.scylla_pkg()))
            self.remoter.sudo('zypper install -y scylla-gdb', ignore_status=True)
        else:
            if self.is_ubuntu14():
                self.remoter.run('sudo apt-get install software-properties-common -y')
                self.remoter.run('sudo add-apt-repository -y ppa:openjdk-r/ppa')
                self.remoter.run('sudo add-apt-repository -y ppa:scylladb/ppa')
                self.remoter.run('sudo apt-get update')
                self.remoter.run('sudo apt-get install -y openjdk-8-jre-headless')
                self.remoter.run('sudo update-java-alternatives --jre-headless -s java-1.8.0-openjdk-amd64')
            elif self.distro.is_ubuntu:
                install_prereqs = dedent("""
                    export DEBIAN_FRONTEND=noninteractive
                    apt-get install software-properties-common -y
                """)
                self.remoter.run('sudo bash -cxe "%s"' % install_prereqs)
            elif self.is_debian8():
                self.remoter.run("sudo sed -i -e 's/jessie-updates/stable-updates/g' /etc/apt/sources.list")
                self.remoter.run(
                    'echo "deb http://archive.debian.org/debian jessie-backports main" '
                    '|sudo tee /etc/apt/sources.list.d/backports.list')
                self.remoter.run(
                    r"sudo sed -i -e 's/:\/\/.*\/debian jessie-backports /:\/\/archive.debian.org\/debian"
                    r" jessie-backports /g' /etc/apt/sources.list.d/*.list")
                self.remoter.run(
                    "echo 'Acquire::Check-Valid-Until \"false\";' |sudo tee /etc/apt/apt.conf.d/99jessie-backports")
                self.remoter.run('sudo apt-get update')
                self.remoter.run('sudo apt-get install gnupg-curl -y')
                self.remoter.run(
                    'sudo apt-key adv --fetch-keys https://download.opensuse.org/repositories/home:/'
                    'scylladb:/scylla-3rdparty-jessie/Debian_8.0/Release.key')
                self.remoter.run(
                    'echo "deb http://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-jessie/'
                    'Debian_8.0/ /" |sudo tee /etc/apt/sources.list.d/scylla-3rdparty.list')
                self.remoter.run('sudo apt-get update')
                self.remoter.run('sudo apt-get install -y openjdk-8-jre-headless -t jessie-backports')
                self.remoter.run('sudo update-java-alternatives --jre-headless -s java-1.8.0-openjdk-amd64')
            elif self.is_debian9():
                install_debian_9_prereqs = dedent("""
                    export DEBIAN_FRONTEND=noninteractive
                    apt-get update
                    apt-get install apt-transport-https -y
                    apt-get install gnupg1-curl dirmngr -y
                    apt-key adv --fetch-keys https://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-stretch/Debian_9.0/Release.key
                    echo 'deb http://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-stretch/Debian_9.0/ /' > /etc/apt/sources.list.d/scylla-3rdparty.list
                """)
                self.remoter.run('sudo bash -cxe "%s"' % install_debian_9_prereqs)
            elif self.distro.is_debian10 or self.distro.is_debian11:
                install_debian_10_prereqs = dedent("""
                    export DEBIAN_FRONTEND=noninteractive
                    apt-get update
                    apt-get install apt-transport-https -y
                    apt-get install gnupg1-curl dirmngr -y
                    curl https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | sudo apt-key add -
                    apt-get install software-properties-common -y
                    add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/
                """)
                self.remoter.run('sudo bash -cxe "%s"' % install_debian_10_prereqs)

            self.remoter.run(
                'sudo DEBIAN_FRONTEND=noninteractive apt-get '
                '-o Dpkg::Options::="--force-confold" '
                '-o Dpkg::Options::="--force-confdef" '
                'upgrade -y ')
            self.remoter.run('sudo apt-get install -y rsync tcpdump screen')
            self.download_scylla_repo(scylla_repo)
            self.remoter.run('sudo apt-get update')
            self.remoter.run(
                'sudo apt-get install -y '
                ' {} '.format(self.scylla_pkg()))

    def offline_install_scylla(self, unified_package, nonroot):
        """
        Offline install scylla by unified package.
        """
        # Download unified package
        self.remoter.run(f'curl {unified_package} -o ./unified_package.tar.gz')

        if nonroot:
            additional_pkgs = ''
        else:
            additional_pkgs = 'xfsprogs mdadm'

        # scylla current cqlsh still need python2 to work
        if self.distro.is_ubuntu18:
            additional_pkgs += ' python2.7'
        else:
            additional_pkgs += ' python2'

        # Offline install does't provide openjdk-8, it has to be installed in advance
        # https://github.com/scylladb/scylla-jmx/issues/127
        if self.is_rhel_like():
            self.remoter.run(f'sudo yum install -y java-1.8.0-openjdk {additional_pkgs}')
        elif self.distro.is_sles:
            raise Exception("Offline install on SLES isn't supported")
        elif self.distro.is_debian10 or self.distro.is_debian11:
            self.remoter.run(f'sudo apt-get install -y openjdk-11-jre-headless {additional_pkgs}')
        else:
            self.remoter.run(f'sudo apt-get install -y openjdk-8-jre-headless {additional_pkgs}')
            self.remoter.run('sudo update-java-alternatives --jre-headless '
                             '-s java-1.8.0-openjdk-${dpkg-architecture -q DEB_BUILD_ARCH}')

        package_version_cmds_v2 = dedent("""
            tar -xzO --wildcards -f ./unified_package.tar.gz .relocatable_package_version
        """)
        package_version_cmds_v3 = dedent("""
            tar -xzO --wildcards -f ./unified_package.tar.gz scylla-*/.relocatable_package_version
        """)
        result = self.remoter.run('bash -cxe "%s"' % package_version_cmds_v3, ignore_status=True)
        if not result.ok:
            logging.info('v3 version of .relocatable_package_version does not detected, retry with v2 version')
            result = self.remoter.run('bash -cxe "%s"' % package_version_cmds_v2)
        package_version = packaging.version.parse(result.stdout.strip())

        if nonroot:
            # Make sure env variable (XDG_RUNTIME_DIR) is set, which is necessary for systemd user
            if not 'XDG_RUNTIME_DIR=' in self.remoter.run('env').stdout:
                # Reload the env variables by ssh reconnect
                self.remoter.run('env', verbose=True, change_context=True)
                assert 'XDG_RUNTIME_DIR' in self.remoter.run('env', verbose=True).stdout
            if package_version < packaging.version.parse('3'):
                install_cmds = dedent("""
                    tar xvfz ./unified_package.tar.gz
                    ./install.sh --nonroot
                    sudo rm -f /tmp/scylla.yaml
                """)
            else:
                install_cmds = dedent("""
                    tar xvfz ./unified_package.tar.gz
                    cd ./scylla-*
                    ./install.sh --nonroot
                    cd -
                    sudo rm -f /tmp/scylla.yaml
                """)
            # Known issue: https://github.com/scylladb/scylla/issues/7071
            self.remoter.run('bash -cxe "%s"' % install_cmds)
        else:
            if package_version < packaging.version.parse('3'):
                install_cmds = dedent("""
                    tar xvfz ./unified_package.tar.gz
                    ./install.sh --housekeeping
                    rm -f /tmp/scylla.yaml
                """)
            else:
                install_cmds = dedent("""
                    tar xvfz ./unified_package.tar.gz
                    cd ./scylla-*
                    ./install.sh --housekeeping
                    cd -
                    rm -f /tmp/scylla.yaml
                """)
            self.remoter.run('sudo bash -cxe "%s"' % install_cmds)

    def web_install_scylla(self, scylla_version: Optional[str] = None) -> None:
        """
        Install scylla by Scylla Web Installer Script. Try to use install the latest
        unstable build for the test branch, generally it should install same version
        as the unstable scylla_repo.

        https://github.com/scylladb/scylla-web-install
        """
        is_enterprise, version = assume_version(self.parent_cluster.params, scylla_version)
        product_type = '--scylla-product scylla-enterprise' if is_enterprise else ''
        self.remoter.run(
            f"curl -sSf get.scylladb.com/server | sudo bash -s -- --scylla-version {version} {product_type}")

    def install_scylla_debuginfo(self) -> None:
        if self.distro.is_rhel_like:
            cmd = fr"yum install -y {self.scylla_pkg()}-debuginfo-{self.scylla_version}\*"
        elif self.distro.is_sles:
            cmd = fr"zypper install -y {self.scylla_pkg()}-debuginfo-{self.scylla_version}\*"
        else:
            cmd = fr"apt-get install -y {self.scylla_pkg()}-server-dbg={self.scylla_version}\*"

        self.log.debug("Installing Scylla debug info...")
        self.remoter.sudo(cmd, ignore_status=True)

    def is_scylla_installed(self, raise_if_not_installed=False):
        if self.distro.is_rhel_like or self.distro.is_sles:
            result = self.remoter.run(f'rpm -q {self.scylla_pkg()}', verbose=False, ignore_status=True)
        elif self.distro.is_ubuntu or self.distro.is_debian:
            result = self.remoter.run(f'dpkg-query --show {self.scylla_pkg()}', verbose=False, ignore_status=True)
        else:
            raise ValueError(f"Unsupported Linux distribution: {self.distro}")
        if result.exit_status == 0:
            return True
        elif raise_if_not_installed:
            raise Exception(f"There is no pre-installed ScyllaDB on {self}")
        return False

    def get_scylla_binary_version(self) -> Optional[str]:
        result = self.remoter.run(f"{self.add_install_prefix('/usr/bin/scylla')} --version", ignore_status=True)
        if result.ok:
            return result.stdout.strip()
        self.log.debug("Unable to get ScyllaDB version using `%s':\n%s\n%s",
                       result.command, result.stdout, result.stderr)
        return None

    def get_scylla_package_version(self) -> Optional[str]:
        if self.distro.is_rhel_like:
            cmd = f"rpm --query --queryformat '%{{VERSION}}' {self.scylla_pkg()}"
        else:
            cmd = f"dpkg-query --show --showformat '${{Version}}' {self.scylla_pkg()}"
        result = self.remoter.run(cmd, ignore_status=True)
        if result.ok:
            return result.stdout.strip()
        self.log.debug("Unable to get ScyllaDB version using `%s':\n%s\n%s",
                       result.command, result.stdout, result.stderr)
        return None

    @optional_cached_property
    def scylla_version_detailed(self) -> Optional[str]:
        scylla_version = self.get_scylla_binary_version()
        if scylla_version is None:
            if scylla_version := self.get_scylla_package_version():
                return scylla_version.replace("~", ".")
            self.log.warning("All attempts to get ScyllaDB version failed. Looks like there is no ScyllaDB installed.")
            raise NoValue
        if build_id := self.get_scylla_build_id():
            scylla_version += f" with build-id {build_id}"
            self.log.info("Found ScyllaDB version with details: %s", scylla_version)
        self.log.debug(f"self.scylla_version_detailed={scylla_version}")
        return scylla_version

    @optional_cached_property
    def scylla_version(self) -> Optional[str]:
        if scylla_version := self.scylla_version_detailed:
            if match := SCYLLA_VERSION_RE.match(scylla_version):
                scylla_version = match.group()
                self.log.info("Found ScyllaDB version: %s", scylla_version)
                return scylla_version
            self.log.debug("Unable to parse ScyllaDB version string: `%s'", scylla_version)
        raise NoValue

    def forget_scylla_version(self) -> None:
        self.__dict__.pop("scylla_version_detailed", None)
        self.__dict__.pop("scylla_version", None)

    @log_run_info("Detecting disks")
    def detect_disks(self, nvme=True):
        """
        Detect local disks
        :param nvme: NVMe(True) or SCSI(False) disk
        :return: list of disk names
        """
        patt = (r'nvme*n*', r'nvme\d+n\d+') if nvme else (r'sd[b-z]', r'sd\w+')
        result = self.remoter.run(f"ls /dev/{patt[0]}", ignore_status=True)
        disks = re.findall(r'/dev/{}'.format(patt[1]), result.stdout)
        # filter out the used disk, the free disk doesn't have partition.
        disks = [i for i in disks if disks.count(i) == 1]
        assert disks, 'Failed to find disks!'
        self.log.debug("Found disks: %s", disks)
        return disks

    @property
    def kernel_version(self):
        if not self._kernel_version:
            res = self.remoter.run("uname -r", ignore_status=True)
            if res.exit_status:
                self._kernel_version = "unknown"
            else:
                self._kernel_version = res.stdout.strip()
            self.log.info("Found kernel version: {}".format(self._kernel_version))
        return self._kernel_version

    def increase_jmx_heap_memory(self, jmx_memory):
        jmx_file = '/opt/scylladb/jmx/scylla-jmx'
        self.log.info('changing scylla-jmx heap memory to avoid 5k crashing jmx')
        self.remoter.run(f"sudo sed -i 's/Xmx256m/Xmx{jmx_memory}m/' {jmx_file}")
        self.log.info('changed scylla-jmx heap memory')
        self.remoter.run(f"sudo grep Xmx {jmx_file}")
        self.log.info('result after changing scylla-jmx heap above')

    @log_run_info
    def scylla_setup(self, disks, devname: str):
        """
        TestConfig scylla
        :param disks: list of disk names
        """
        extra_setup_args = self.parent_cluster.params.get('append_scylla_setup_args')
        result = self.remoter.run('sudo /usr/lib/scylla/scylla_setup --help')
        if '--swap-directory' in result.stdout:
            # swap setup is supported
            extra_setup_args += ' --swap-directory / '
        if self.parent_cluster.params.get('unified_package'):
            extra_setup_args += ' --no-verify-package '

        if self.parent_cluster.params.get('workaround_kernel_bug_for_iotune'):
            self.log.warning(dedent("""
                Kernel version is {}. Due to known kernel bug in this version using predefined iotune.
                related issue: https://github.com/scylladb/scylla/issues/5181
                known kernel bug will cause scylla_io_setup fails in executing iotune.
                The kernel bug doesn't occur all the time, so we can get some succeed gce instance.
                the config files are copied from a succeed GCE instance (same instance type, same test
            """.format(self.kernel_version)))
            self.remoter.run('sudo /usr/lib/scylla/scylla_setup --nic {} --disks {} --no-io-setup {}'
                             .format(devname, ','.join(disks), extra_setup_args))
            for conf in ['io.conf', 'io_properties.yaml']:
                self.remoter.send_files(src=os.path.join('./configurations/', conf),  # pylint: disable=not-callable
                                        dst='/tmp/')
                self.remoter.run('sudo mv /tmp/{0} /etc/scylla.d/{0}'.format(conf))
        else:
            self.remoter.run('sudo /usr/lib/scylla/scylla_setup --nic {} --disks {} --setup-nic-and-disks {}'
                             .format(devname, ','.join(disks), extra_setup_args))

        result = self.remoter.run('cat /proc/mounts')
        assert ' /var/lib/scylla ' in result.stdout, "RAID setup failed, scylla directory isn't mounted correctly"
        self.remoter.run('sudo sync')
        self.log.info('io.conf right after setup')
        self.remoter.run('sudo cat /etc/scylla.d/io.conf')

        if not self.is_ubuntu14():
            self.remoter.run('sudo systemctl enable scylla-server.service')
            self.remoter.run('sudo systemctl enable scylla-jmx.service')

    def upgrade_mgmt(self, scylla_mgmt_address, start_manager_after_upgrade=True):
        self.log.debug("Upgrade scylla-manager via repo: %s", scylla_mgmt_address)
        self.download_scylla_manager_repo(scylla_mgmt_address)
        if self.distro.is_rhel_like:
            self.remoter.sudo("yum update scylla-manager-server scylla-manager-client -y")
        elif self.distro.is_sles:
            self.remoter.sudo("zypper update scylla-manager-server scylla-manager-client -y")
        else:
            self.remoter.sudo("apt-get update", ignore_status=True)
            # Upgrade should update packages of:
            # 1) scylla-manager
            # 2) scylla-manager-client
            # 3) scylla-manager-server
            self.remoter.sudo('DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade '
                              '-o Dpkg::Options::="--force-confold" '
                              '-o Dpkg::Options::="--force-confdef" '
                              'scylla-manager-server scylla-manager-client -y ')
        time.sleep(3)
        if start_manager_after_upgrade:
            if self.is_docker():
                self.remoter.sudo("supervisorctl start scylla-manager")
            else:
                self.remoter.sudo("systemctl restart scylla-manager.service")
            time.sleep(5)

    # pylint: disable=too-many-branches
    def install_mgmt(self, package_url: Optional[str] = None) -> None:
        self.log.debug("Install scylla-manager")

        if self.is_docker():
            self.remoter.sudo(
                "yum remove -y scylla scylla-jmx scylla-tools scylla-tools-core scylla-server scylla-conf")

        if self.distro.is_rhel_like:
            self.install_epel()
            self.remoter.sudo("yum install python36-PyYAML -y", retry=3)
        elif self.distro.is_sles:
            self.remoter.sudo("zypper install python36-PyYAML -y", retry=3)
        elif not self.distro.is_debian_like:
            raise ValueError(f"Unsupported Linux distribution: {self.distro}")

        package_names = "scylla-manager-server scylla-manager-client"
        if package_url:
            package_names = f"{package_url}scylla-manager-server* {package_url}scylla-manager-client*"
        elif self.parent_cluster.params.get("scylla_mgmt_address"):
            self.download_scylla_manager_repo(self.parent_cluster.params.get("scylla_mgmt_address"))
        else:
            manager_version = self.parent_cluster.params.get("manager_version")
            manager_repo_url = get_manager_repo_from_defaults(manager_version, self.distro)
            self.download_scylla_manager_repo(manager_repo_url)
        self.install_package(package_names)

        self.log.debug("Copying TLS files from data_dir to the node")
        self.remoter.send_files(src="./data_dir/ssl_conf", dst="/tmp/")

        if self.is_docker():
            try:
                self.remoter.run("echo no | sudo scyllamgr_setup")
            except Exception as ex:  # pylint: disable=broad-except
                self.log.warning(ex)
        else:
            self.remoter.run("echo yes | sudo scyllamgr_setup")

        # Copy the private SSH key and generate the public key.
        self.remoter.send_files(src=self.ssh_login_info["key_file"], dst="/tmp/scylla-test")
        self.remoter.sudo(shell_script_cmd("""\
            chmod 0400 /tmp/scylla-test
            chown scylla-manager:scylla-manager /tmp/scylla-test
            ssh-keygen -y -f /tmp/scylla-test > /tmp/scylla-test-pub
        """))

        tls_cert_file = tls_key_file = None
        if self.remoter.sudo("scyllamgr_ssl_cert_gen", ignore_status=True).ok:
            tls_cert_file = SCYLLA_MANAGER_TLS_CERT_FILE
            tls_key_file = SCYLLA_MANAGER_TLS_KEY_FILE

        with self.remote_manager_yaml() as manager_yaml:
            manager_yaml["tls_cert_file"] = tls_cert_file
            manager_yaml["tls_key_file"] = tls_key_file
            manager_yaml["prometheus"] = f":{self.parent_cluster.params.get('manager_prometheus_port')}"

        if self.is_docker():
            self.remoter.sudo("supervisorctl restart scylla-manager")
            res = self.remoter.sudo("supervisorctl status scylla-manager")
        else:
            self.remoter.sudo("systemctl restart scylla-manager.service")
            res = self.remoter.sudo("systemctl status scylla-manager.service")

        if not res or "Active: failed" in res.stdout:
            raise ScyllaManagerError(f"Scylla-Manager is not properly installed or not running: {res}")

        self.start_scylla_manager_log_capture()

    def retrieve_scylla_manager_log(self):
        mgmt_log_name = os.path.join(self.logdir, 'scylla_manager.log')
        cmd = "sudo journalctl -u scylla-manager -f"
        self.remoter.run(cmd, ignore_status=True, verbose=True, log_file=mgmt_log_name)

    def scylla_manager_log_thread(self):
        while not self.termination_event.is_set():
            self.retrieve_scylla_manager_log()

    def start_scylla_manager_log_capture(self):
        self._scylla_manager_journal_thread = threading.Thread(
            target=self.scylla_manager_log_thread, name='ScyllaManagerJournalThread', daemon=True)
        self._scylla_manager_journal_thread.start()

    def stop_scylla_manager_log_capture(self, timeout=10):
        cmd = "sudo pkill -f \"sudo journalctl -u scylla-manager -f\""
        self.remoter.run(cmd, ignore_status=True, verbose=True)
        self._scylla_manager_journal_thread.join(timeout)
        self._scylla_manager_journal_thread = None

    def wrap_cmd_with_permission(self, cmd):
        """Generate commandline prefix according to the privilege"""
        if self.is_nonroot_install:
            return f'{cmd} --user'
        else:
            return f'sudo {cmd}'

    @property
    def systemctl(self):
        return self.wrap_cmd_with_permission('systemctl')

    @property
    def journalctl(self):
        return self.wrap_cmd_with_permission('journalctl')

    def add_install_prefix(self, abs_path):
        """
        nonroot install included all files inside a install root directory,
        it's different with root install.
        """
        assert os.path.isabs(abs_path), f"abs_path ({abs_path}) should be an absolute path"
        if not self.is_nonroot_install:
            return abs_path
        checklist = {
            SCYLLA_YAML_PATH: f'{INSTALL_DIR}{SCYLLA_YAML_PATH}',
            SCYLLA_PROPERTIES_PATH: f'{INSTALL_DIR}{SCYLLA_PROPERTIES_PATH}',
            '/etc/scylla.d/io.conf': f'{INSTALL_DIR}/etc/scylla.d/io.conf',
            '/usr/bin/scylla': f'{INSTALL_DIR}/bin/scylla',
            '/usr/bin/nodetool': f'{INSTALL_DIR}/share/cassandra/bin/nodetool',
            '/usr/bin/cqlsh': f'{INSTALL_DIR}/share/cassandra/bin/cqlsh',
            '/usr/bin/cassandra-stress': f'{INSTALL_DIR}/share/cassandra/bin/cassandra-stress',
        }
        return checklist.get(abs_path, INSTALL_DIR + abs_path)

    def _service_cmd(self, service_name: str, cmd: str, timeout: int = 500, ignore_status=False):
        if self.is_ubuntu14():
            cmd = f'sudo service {service_name} {cmd}'
        else:
            cmd = f'{self.systemctl} {cmd} {service_name}.service'
        return self.remoter.run(cmd, timeout=timeout, ignore_status=ignore_status)

    def get_service_status(self, service_name: str, timeout: int = 500, ignore_status=False):
        return self._service_cmd(service_name=service_name, cmd='status', timeout=timeout, ignore_status=ignore_status)

    def start_service(self, service_name: str, timeout: int = 500, ignore_status=False):
        self._service_cmd(service_name=service_name, cmd='start', timeout=timeout, ignore_status=ignore_status)

    def stop_service(self, service_name: str, timeout=500, ignore_status=False):
        self._service_cmd(service_name=service_name, cmd='stop', timeout=timeout, ignore_status=ignore_status)

    def restart_service(self, service_name: str, timeout=500, ignore_status=False):
        self._service_cmd(service_name=service_name, cmd='restart', timeout=timeout, ignore_status=ignore_status)

    @property
    def verify_up_timeout(self):
        return 300

    def start_scylla_server(self, verify_up=True, verify_down=False, timeout=500, verify_up_timeout=None):
        verify_up_timeout = verify_up_timeout or self.verify_up_timeout
        if verify_down:
            self.wait_db_down(timeout=timeout)
        self.start_service(service_name='scylla-server', timeout=timeout)
        if verify_up:
            self.wait_db_up(timeout=verify_up_timeout)

    def start_scylla_jmx(self, verify_up=True, verify_down=False, timeout=300, verify_up_timeout=300):
        if verify_down:
            self.wait_jmx_down(timeout=timeout)
        self.start_service(service_name='scylla-jmx', timeout=timeout)
        if verify_up:
            self.wait_jmx_up(timeout=verify_up_timeout)

    @log_run_info
    def start_scylla(self, verify_up=True, verify_down=False, timeout=500):
        self.start_scylla_server(verify_up=verify_up, verify_down=verify_down, timeout=timeout)
        if verify_up:
            self.wait_jmx_up(timeout=timeout)

    @retrying(n=3, sleep_time=5, allowed_exceptions=NETWORK_EXCEPTIONS + (CommandTimedOut, ),
              message="Failed to stop scylla.server, retrying...")
    def stop_scylla_server(self, verify_up=False, verify_down=True, timeout=300, ignore_status=False):
        if verify_up:
            self.wait_db_up(timeout=timeout)
        self.stop_service(service_name='scylla-server', timeout=timeout, ignore_status=ignore_status)
        if verify_down:
            self.wait_db_down(timeout=timeout)

    def stop_scylla_jmx(self, verify_up=False, verify_down=True, timeout=300):
        if verify_up:
            self.wait_jmx_up(timeout=timeout)
        self.stop_service(service_name='scylla-jmx', timeout=timeout)
        if verify_down:
            self.wait_jmx_down(timeout=timeout)

    @log_run_info
    def stop_scylla(self, verify_up=False, verify_down=True, timeout=300):
        self.stop_scylla_server(verify_up=verify_up, verify_down=verify_down, timeout=timeout)
        if verify_down:
            self.wait_jmx_down(timeout=timeout)

    def restart_scylla_server(self, verify_up_before=False, verify_up_after=True, timeout=500, ignore_status=False):
        if verify_up_before:
            self.wait_db_up(timeout=timeout)
        self.restart_service(service_name='scylla-server', timeout=timeout, ignore_status=ignore_status)
        if verify_up_after:
            self.wait_db_up(timeout=timeout)

    def restart_scylla_jmx(self, verify_up_before=False, verify_up_after=True, timeout=300):
        if verify_up_before:
            self.wait_jmx_up(timeout=timeout)
        self.restart_service(service_name='scylla-jmx', timeout=timeout)
        if verify_up_after:
            self.wait_jmx_up(timeout=timeout)

    @log_run_info
    def restart_scylla(self, verify_up_before=False, verify_up_after=True, timeout=500):
        self.restart_scylla_server(verify_up_before=verify_up_before, verify_up_after=verify_up_after, timeout=timeout)
        if verify_up_after:
            self.wait_jmx_up(timeout=timeout)

    def prepare_files_for_archive(self, fileslist):
        """Prepare files for creating archives on node

        Create directories structure and copy files
        from absolute path of files in fileslist
        in node /tmp/node_name and return path to created
        structure
        Ex.:
            filelists = ['/proc/cpuinfo', /var/log/system.log]

            result directory:
            /tmp/<node_name>/proc/cpuinfo
                            /var/log/system.log

        Arguments:
            fileslist {list} -- list of file with fullpaths on node

        Returns:
            [type] -- path to created directory structure
        """
        root_dir = '/tmp/%s' % self.name
        self.remoter.run('mkdir -p %s' % root_dir, ignore_status=True)
        for f in fileslist:
            if self.file_exists(f):
                old_full_path = os.path.dirname(f)[1:]
                new_full_path = os.path.join(root_dir, old_full_path)
                self.remoter.run('mkdir -p %s' % new_full_path, ignore_status=True)
                self.remoter.run('cp -r %s %s' % (f, new_full_path), ignore_status=True)
        return root_dir

    def generate_coredump_file(self, restart_scylla=True, node_name: str = None, message: str = None):
        message = message or "Generating a Scylla core dump by SIGQUIT..\n"
        if node_name:
            message += f"Node: {node_name}"
        InfoEvent(severity=Severity.ERROR, message=message)
        self.remoter.sudo("pkill -f --signal 3 /usr/bin/scylla")
        self.wait_db_down(timeout=600)
        if restart_scylla:
            self.log.debug('Restart scylla server')
            self.stop_scylla(timeout=600)
            self.start_scylla(timeout=600)

    def get_console_output(self):
        # TODO add to each type of node
        # comment raising exception. replace with log warning
        # raise NotImplementedError('Derived classes must implement get_console_output')
        self.log.warning('Method get_console_output is not implemented for %s' % self.__class__.__name__)
        return ''

    def get_console_screenshot(self):
        # TODO add to each type of node
        # comment raising exception. replace with log warning
        # raise NotImplementedError('Derived classes must implement get_console_output')
        self.log.warning('Method get_console_screenshot is not implemented for %s' % self.__class__.__name__)
        return b''

    # Default value of murmur3_partitioner_ignore_msb_bits parameter is 12
    def restart_node_with_resharding(self, murmur3_partitioner_ignore_msb_bits: int = 12) -> None:
        """
        Resharding is started during Scylla startup and Scylla initialization has to wait until resharding done
        """
        self.stop_scylla()
        # Change murmur3_partitioner_ignore_msb_bits parameter to cause resharding.
        with self.remote_scylla_yaml() as scylla_yml:
            scylla_yml.murmur3_partitioner_ignore_msb_bits = murmur3_partitioner_ignore_msb_bits

        search_reshard_start = self.follow_system_log(patterns=[DB_LOG_PATTERN_RESHARDING_START])
        search_reshard_finish = self.follow_system_log(patterns=[DB_LOG_PATTERN_RESHARDING_FINISH])
        start_scylla_timeout = 7200
        self.start_scylla(timeout=start_scylla_timeout)
        resharding_started = list(search_reshard_start)
        resharding_finished = list(search_reshard_finish)

        if resharding_started:
            self.log.debug(f'Resharding has been started successfully '
                           f'(murmur3_partitioner_ignore_msb_bits={murmur3_partitioner_ignore_msb_bits})')
        else:
            raise Exception(f'Resharding has not been started '
                            f'(murmur3_partitioner_ignore_msb_bits={murmur3_partitioner_ignore_msb_bits}) '
                            'Check the log for the details')

        if resharding_finished:
            self.log.debug(f'Resharding has been finished successfully '
                           f'(murmur3_partitioner_ignore_msb_bits={murmur3_partitioner_ignore_msb_bits})')
        else:
            raise Exception(f'Resharding has not been finished within {start_scylla_timeout}'
                            f'(murmur3_partitioner_ignore_msb_bits={murmur3_partitioner_ignore_msb_bits}) '
                            'Check the log for the details')

    def _gen_nodetool_cmd(self, sub_cmd, args, options):
        credentials = self.parent_cluster.get_db_auth()
        if credentials:
            options += "-u {} -pw '{}' ".format(*credentials)
        return f"{self.add_install_prefix('/usr/bin/nodetool')} {options} {sub_cmd} {args}"

    # pylint: disable=inconsistent-return-statements
    def run_nodetool(self, sub_cmd, args="", options="", timeout=None,
                     ignore_status=False, verbose=True, coredump_on_timeout=False,
                     warning_event_on_exception=None, error_message="", publish_event=True, retry=1):
        """
            Wrapper for nodetool command.
            Command format: nodetool [options] command [args]

        :param sub_cmd: subcommand like status
        :param args: arguments for the subcommand
        :param options: nodetool options:
            -h  --host  Hostname or IP address.
            -p  --port  Port number.
            -pwf    --password-file Password file path.
            -pw --password  Password.
            -u  --username  Username.
        :param timeout: time for command execution
        :param ignore_status: don't throw exception if the command fails
        :param coredump_on_timeout: Send signal SIGQUIT to scylla process
        :param warning_event_on_exception: create WARNING instead of ERROR NodetoolEvent severity if the exception is
                                           in the list
        :param error_message: additional error message to exception message
        :param publish_event: publish event or not
        :return: Remoter result object
        """
        cmd = self._gen_nodetool_cmd(sub_cmd, args, options)

        with NodetoolEvent(nodetool_command=sub_cmd,
                           node=self.name,
                           options=options,
                           publish_event=publish_event) as nodetool_event:
            try:
                result = \
                    self.remoter.run(cmd, timeout=timeout, ignore_status=ignore_status, verbose=verbose, retry=retry)
                self.log.debug("Command '%s' duration -> %s s" % (result.command, result.duration))

                nodetool_event.duration = result.duration
                return result
            except Exception as details:  # pylint: disable=broad-except
                if isinstance(details, RetryableNetworkException):
                    details = details.original
                if coredump_on_timeout and details.__class__.__name__.endswith("CommandTimedOut"):
                    message = f"Generating a Scylla core dump by SIGQUIT due to a nodetool command '{sub_cmd}' timeout"
                    self.generate_coredump_file(node_name=self.name, message=message)

                nodetool_event.add_error([f"{error_message}{str(details)}"])
                nodetool_event.full_traceback = traceback.format_exc()

                if warning_event_on_exception and isinstance(details, warning_event_on_exception):
                    nodetool_event.severity = Severity.WARNING
                else:
                    raise

    def node_health_events(self) -> Iterator[ClusterHealthValidatorEvent]:
        nodes_status = self.get_nodes_status()
        peers_details = self.get_peers_info() or {}
        gossip_info = self.get_gossip_info() or {}

        return itertools.chain(
            check_nodes_status(
                nodes_status=nodes_status,
                current_node=self,
                removed_nodes_list=self.parent_cluster.dead_nodes_ip_address_list),
            check_node_status_in_gossip_and_nodetool_status(
                gossip_info=gossip_info,
                nodes_status=nodes_status,
                current_node=self),
            check_schema_version(
                gossip_info=gossip_info,
                peers_details=peers_details,
                nodes_status=nodes_status,
                current_node=self),
            check_nulls_in_peers(
                gossip_info=gossip_info,
                peers_details=peers_details,
                current_node=self),
        )

    def check_node_health(self, retries: int = CHECK_NODE_HEALTH_RETRIES) -> None:
        # Task 1443: ClusterHealthCheck is bottle neck in scale test and create a lot of noise in 5000 tables test.
        # Disable it
        if not self.parent_cluster.params.get('cluster_health_check'):
            return

        for retry_n in range(1, retries+1):
            LOGGER.debug("Check the health of the node `%s' [attempt #%d]", self.name, retry_n)
            events = self.node_health_events()
            event = next(events, None)
            if event is None:
                LOGGER.debug("Node `%s' is healthy", self.name)
                break
            if retry_n == retries:  # publish health validation events on the last retry.
                LOGGER.debug("One or more node `%s' health validation has failed", self.name)
                event.publish()
                for event in events:
                    event.publish()
                break

            event.dont_publish()

            LOGGER.debug("Wait for %d secs before next try to validate the health of the node `%s'",
                         CHECK_NODE_HEALTH_RETRY_DELAY, self.name)
            time.sleep(CHECK_NODE_HEALTH_RETRY_DELAY)

    def get_nodes_status(self):
        nodes_status = {}
        try:
            statuses = self.parent_cluster.get_nodetool_status(verification_node=self)

            for dc, dc_status in statuses.items():
                for node_ip, node_properties in dc_status.items():
                    if node := self.parent_cluster.find_node_by_ip(node_ip):
                        nodes_status[node] = {'status': node_properties['state'], 'dc': dc}
                    else:
                        if node_ip:
                            LOGGER.error("Get nodes statuses. Failed to find a node in cluster by IP: %s", node_ip)

        except Exception as error:  # pylint: disable=broad-except
            ClusterHealthValidatorEvent.NodeStatus(
                severity=Severity.WARNING,
                node=self.name,
                message=f"Unable to get nodetool status from `{self.name}': {error=}",
            ).publish()
        return nodes_status

    @retrying(n=5, sleep_time=5, raise_on_exceeded=False)
    def get_peers_info(self):
        columns = (
            'peer', 'data_center', 'host_id', 'rack', 'release_version',
            'rpc_address', 'schema_version', 'supported_features',
        )
        peers_details = {}
        with self.parent_cluster.cql_connection_patient_exclusive(self) as session:
            result = session.execute(f"select {', '.join(columns)} from system.peers")
            cql_results = result.all()
        err = ''
        for row in cql_results:
            peer = row.peer
            try:
                ipaddress.ip_address(row.peer)
            except ValueError as exc:
                current_err = f"Peer '{peer}' is not an IP address, err: {exc}\n"
                LOGGER.warning(current_err)
                err += current_err
                continue

            if node := self.parent_cluster.find_node_by_ip(peer):
                peers_details[node] = row._asdict()
            else:
                current_err = f"'get_peers_info' failed to find a node by IP: {peer}\n"
                LOGGER.error(current_err)
                err += current_err

        if not (peers_details or err):
            LOGGER.error(
                "No data, no errors. Check the output from the cql command for the correctness:\n%s",
                cql_results)
        return peers_details

    @retrying(n=5, sleep_time=10, raise_on_exceeded=False)
    def get_gossip_info(self):
        gossip_info = self.run_nodetool('gossipinfo', verbose=False, warning_event_on_exception=(Exception,),
                                        publish_event=False)
        LOGGER.debug("get_gossip_info: %s", gossip_info)
        gossip_node_schemas = {}
        schema = ip = status = dc = ''
        for line in gossip_info.stdout.split():
            if line.startswith('SCHEMA:'):
                schema = line.replace('SCHEMA:', '')
            elif line.startswith('RPC_ADDRESS:'):
                ip = line.replace('RPC_ADDRESS:', '')
            elif line.startswith('STATUS:'):
                status = line.replace('STATUS:', '').split(',')[0]
            elif line.startswith('DC:'):
                dc = line.replace('DC:', '').split(',')[0]

            if schema and ip and status:
                if node := self.parent_cluster.find_node_by_ip(ip):
                    gossip_node_schemas[node] = {'schema': schema, 'status': status, 'dc': dc}
                elif status in self.GOSSIP_STATUSES_FILTER_OUT:
                    LOGGER.debug("Get gossip info. Node with IP %s is not found in the cluster. Node status is: %s",
                                 ip, status)
                else:
                    LOGGER.error("Get gossip info. Failed to find a node with status %s in the cluster by IP: %s",
                                 status, ip)

                schema = ip = status = dc = ''

        return gossip_node_schemas

    def print_node_running_nemesis(self, node_ip):
        node = self.parent_cluster.get_node_by_ip(node_ip)
        if not node:
            return ''

        return f' ({node.running_nemesis} nemesis target node)' if node.running_nemesis else ' (not target node)'

    def _gen_cqlsh_cmd(self, command, keyspace, timeout, host, port, connect_timeout):
        """cqlsh [options] [host [port]]"""
        credentials = self.parent_cluster.get_db_auth()
        auth_params = "-u {} -p '{}'".format(*credentials) if credentials else ''
        use_keyspace = "--keyspace {}".format(keyspace) if keyspace else ""
        ssl_params = '--ssl' if self.parent_cluster.params.get("client_encrypt") else ''
        options = "--no-color {auth_params} {use_keyspace} --request-timeout={timeout} " \
                  "--connect-timeout={connect_timeout} {ssl_params}".format(
                      auth_params=auth_params, use_keyspace=use_keyspace, timeout=timeout,
                      connect_timeout=connect_timeout, ssl_params=ssl_params)

        if self.parent_cluster.connection_bundle_file and (
                packaging.version.LegacyVersion(self.scylla_version) >= packaging.version.LegacyVersion('5.2.0~dev')):

            connection_bundle_file = self.parent_cluster.connection_bundle_file
            target_connection_bundle_file = str(Path('/tmp/') / connection_bundle_file.name)
            self.remoter.send_files(str(connection_bundle_file), target_connection_bundle_file)

            return f'cqlsh {options} -e "{command}" --cloudconf {target_connection_bundle_file}'
        return 'cqlsh {options} -e "{command}" {host} {port}'.format(options=options, command=command, host=host,
                                                                     port=port)

    def run_cqlsh(self, cmd, keyspace=None, port=None, timeout=120, verbose=True, split=False, target_db_node=None,
                  connect_timeout=60, num_retry_on_failure=1):
        """Runs CQL command using cqlsh utility"""
        cmd = self._gen_cqlsh_cmd(command=cmd, keyspace=keyspace, timeout=timeout,
                                  host=self.scylla_listen_address if not target_db_node else target_db_node.ip_address,
                                  port=port if port else self.CQL_PORT,
                                  connect_timeout=connect_timeout)
        while num_retry_on_failure:
            try:
                cqlsh_out = self.remoter.run(cmd, timeout=timeout + 30,  # we give 30 seconds to cqlsh timeout mechanism to work
                                             verbose=verbose)
                break
            except Exception:  # pylint: disable=broad-except
                num_retry_on_failure -= 1
                if not num_retry_on_failure:
                    raise

        # stdout of cqlsh example:
        #      pk
        #      ----
        #       2
        #       3
        #
        #      (10 rows)
        return cqlsh_out if not split else list(map(str.strip, cqlsh_out.stdout.splitlines()))

    def run_startup_script(self):
        startup_script_remote_path = '/tmp/sct-startup.sh'

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(self.test_config.get_startup_script())
            tmp_file.flush()
            self.remoter.send_files(src=tmp_file.name, dst=startup_script_remote_path)  # pylint: disable=not-callable

        cmds = dedent("""
                chmod +x {0}
                {0}
            """.format(startup_script_remote_path))

        result = self.remoter.run("sudo bash -ce '%s'" % cmds)
        LOGGER.debug(result.stdout)

    def create_swap_file(self, size=1024):
        """Create swap file on instance

        Create swap file on instance with size 1MB * 1024 = 1GB
        :param size: size of swap file in MB, defaults to 1024MB
        :type size: number, optional
        """
        commands = dedent("""sudo /bin/dd if=/dev/zero of=/var/sct_configured_swapfile bs=1M count={}
                          sudo /sbin/mkswap /var/sct_configured_swapfile
                          sudo chmod 600 /var/sct_configured_swapfile
                          sudo /sbin/swapon /var/sct_configured_swapfile""".format(size))
        self.log.info("Add swap file to %s", self)
        result = self.remoter.run(commands, ignore_status=True)
        if not result.ok:
            self.log.warning("Swap file was not created on node %s.\nError details: %s", self, result.stderr)
        result = self.remoter.run("grep /sct_configured_swapfile /proc/swaps", ignore_status=True)
        if "sct_configured_swapfile" not in result.stdout:
            self.log.warning("Swap file is not used on node %s.\nError details: %s", self, result.stderr)

    def set_hostname(self):
        self.log.warning('Method set_hostname is not implemented for %s' % self.__class__.__name__)

    def configure_remote_logging(self):
        if self.parent_cluster.params.get('logs_transport') not in ['rsyslog', 'syslog-ng']:
            return
        script = ConfigurationScriptBuilder(
            syslog_host_port=self.test_config.get_logging_service_host_port(),
            logs_transport=self.parent_cluster.params.get('logs_transport'),
            disable_ssh_while_running=False,
            hostname=self.name,
        ).to_string()
        self.remoter.sudo(shell_script_cmd(script, quote="'"))

    @property
    def scylla_packages_installed(self) -> List[str]:
        if self.distro.is_rhel_like or self.distro.is_sles:
            cmd = "rpm -qa 'scylla*'"
        else:
            cmd = "dpkg-query --show 'scylla*'"
        result = self.remoter.run(cmd, ignore_status=True)
        if result.exited == 0:
            return result.stdout.splitlines()
        return []

    def get_scylla_config_param(self, config_param_name, verbose=True):
        """
        Get Scylla configuration parameter that not exists in the scylla.yaml
        """
        try:
            request_out = self.remoter.run(
                f'sudo curl --request GET http://localhost:10000/v2/config/{config_param_name}')
            if "No such config entry" in request_out.stdout:
                self.log.error(
                    f'Failed to retreive value of {config_param_name} parameter. Error: {request_out.stdout}')
                return None
            if verbose:
                self.log.debug(f'{config_param_name} parameter value: {request_out.stdout}')
            return request_out.stdout
        except Exception as e:  # pylint: disable=broad-except
            self.log.error(f'Failed to retreive value of {config_param_name} parameter. Error: {e}')
            return None

    def install_epel(self):
        """
        Standard repositories might not provide all the packages that can be installed on CentOS, RHEL,
        or Amazon Linux-based distributions. Enabling the EPEL repository provides additional options for
        package installation.
        """
        if not self.distro.is_rhel_like:
            raise Exception('EPEL can only be installed for RHEL like distros')

        if self.distro.is_amazon2:
            # Enable amazon2-extras repo for installing epel
            # Reference: https://aws.amazon.com/amazon-linux-2/faqs/#Amazon_Linux_Extras
            self.remoter.run('sudo amazon-linux-extras install epel')

        if self.distro.is_rhel8:
            self.remoter.run(
                "sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm")
        elif self.distro.is_oel8:
            self.remoter.run('sudo yum install -y oracle-epel-release-el8')
        else:
            self.remoter.run('sudo yum install -y epel-release', retry=3)

    def is_machine_image_configured(self):
        smi_configured_path = "/etc/scylla/machine_image_configured"
        result = self.remoter.run(cmd=f"test -e {smi_configured_path}", ignore_status=True, verbose=False)
        return result.ok

    def wait_for_machine_image_configured(self):
        self.log.info("Waiting for Scylla Machine Image setup to finish...")
        wait.wait_for(self.is_machine_image_configured, step=10, timeout=300, throw_exc=False)

    def get_sysctl_properties(self) -> Dict[str, str]:
        sysctl_properties = {}
        result = self.remoter.sudo("sysctl -a", ignore_status=True)

        if not result.ok:
            self.log.error(f"sysctl command failed: {result}")
            return sysctl_properties

        for line in result.stdout.strip().split("\n"):
            try:
                name, value = line.strip().split("=", 1)
                sysctl_properties.update({name.strip(): value.strip()})
            except ValueError:
                self.log.error(f"Could not parse sysctl line: {line}")

        return sysctl_properties

    def set_seed_flag(self, is_seed):
        self.is_seed = is_seed
        if hasattr(self.log, 'extra'):
            self.log.extra['prefix'] = str(self)

    def disable_daily_triggered_services(self):
        if self.distro.uses_systemd and (self.is_ubuntu() or self.is_debian()):
            LOGGER.debug("Disabling 'apt-daily' and 'apt-daily-upgrade' services...")
            self.remoter.sudo('systemctl disable apt-daily.timer')
            self.remoter.sudo('systemctl disable apt-daily-upgrade.timer')
            self.remoter.sudo('systemctl disable apt-daily.service', ignore_status=True)
            self.remoter.sudo('systemctl disable apt-daily-upgrade.service', ignore_status=True)
            self.remoter.sudo('systemctl stop apt-daily.timer', ignore_status=True)
            self.remoter.sudo('systemctl stop apt-daily-upgrade.timer', ignore_status=True)
            self.remoter.sudo('systemctl stop apt-daily.service', ignore_status=True)
            self.remoter.sudo('systemctl stop apt-daily-upgrade.service', ignore_status=True)
            self.wait_apt_not_running()
            self.remoter.sudo('rm -f /etc/apt/apt.conf.d/*unattended-upgrades', ignore_status=True)
            self.remoter.sudo('rm -f /etc/apt/apt.conf.d/*auto-upgrades', ignore_status=True)
            self.remoter.sudo('rm -f /etc/apt/apt.conf.d/*periodic', ignore_status=True)
            self.remoter.sudo('rm -f /etc/apt/apt.conf.d/*update-notifier', ignore_status=True)
            self.remoter.sudo('apt-get remove -y unattended-upgrades', ignore_status=True)
            self.remoter.sudo('apt-get remove -y update-manager', ignore_status=True)

    def get_nic_devices(self) -> List:
        """Returns list of ethernet network interfaces"""
        result = self.remoter.run('/sbin/ip -o link show |grep ether |awk -F": " \'{print $2}\'', verbose=True)
        return result.stdout.strip().split()

    def run_scylla_sysconfig_setup(self) -> bool:
        """
        Run the scylla_sysconfig_setup script that
        sets the values in /etc/scylla.d for:
        - cpuset.conf
        - perftune.yaml
        These values are used by Scylla to decide
        on how to balance rx queues with the logical
        core number.

        Returns a bool indicating if a restart is
        required if the previous values
        of cpuset.conf are different from the
        new ones. Note: restarting might trigger a
        resharding in Scylla.

        Created to implement the procedure described in:
        https://github.com/scylladb/scylla-docs/issues/4126
        """
        # backup old config if it exists
        self.log.info("Move current config files before running scylla_sysconfig setup...")
        self.remoter.sudo("mv /etc/scylla.d/cpuset.conf /etc/scylla.d/cpuset.conf.old", ignore_status=True)
        self.remoter.sudo("mv /etc/scylla.d/perftune.yaml /etc/scylla.d/perftune.yaml.old", ignore_status=True)

        # run as sudo scylla_sysconfig_setup
        self.log.info("Running scylla_sysconfig_setup as sudo...")
        nic_name = self.get_nic_devices()[0]
        syscofnig_result = self.remoter.sudo(f"scylla_sysconfig_setup --nic {nic_name} "
                                             f"--homedir /var/lib/scylla --confdir /etc/scylla")
        self.log.debug("Sysconfig command result:\nstdout: %s\nstderr: %s",
                       syscofnig_result.stdout, syscofnig_result.stderr)

        # compare old output witn new in scylla.d, use diff -q so we'll only get output if there is a difference
        self.log.info("Comparing old config files vs new ones after running scylla_sysconfig_setup...")
        cpuset_diff = self.remoter.run(
            "diff -q /etc/scylla.d/cpuset.conf /etc/scylla.d/cpuset.conf.old", ignore_status=True)

        self.log.info("Comparison result:\nstdout: %s\nstderr: %s", cpuset_diff.stdout, cpuset_diff.stderr)

        # if the diff command failed, we don't have all the needed config files, so revert the backup
        if cpuset_diff.return_code != 0:
            self.remoter.sudo("mv /etc/scylla.d/cpuset.conf.old /etc/scylla.d/cpuset.conf", ignore_status=True)
            self.remoter.sudo("mv /etc/scylla.d/perftune.yaml.old /etc/scylla.d/perftune.yaml", ignore_status=True)
            self.log.info("Finished running scylla_sysconfig_setup")
            return False

        # restart is needed if the config files differ
        restart_needed = bool(cpuset_diff.stdout)

        self.log.info("Finished running scylla_sysconfig_setup")
        return restart_needed

    def get_perftune_yaml(self) -> str:
        cmd = self.remoter.run("sudo cat /etc/scylla.d/perftune.yaml", ignore_status=True)
        if cmd.return_code == 0:
            return cmd.stdout
        else:
            return cmd.stderr

    def get_list_of_sstables(self, keyspace_name: str, table_name: str, suffix: str = "-Statistics.db") -> List[str]:
        statistics_files = []
        find_cmd = f"find /var/lib/scylla/data/{keyspace_name}/{table_name}-* -name *{suffix}"

        result = self.remoter.run(find_cmd, ignore_status=True, verbose=True)

        if result.ok:
            statistics_files = result.stdout.strip().split()

        return statistics_files

    def reload_config(self):
        """
        Reloads scylla configuration without restarting scylla.
        Works only for some parameters (marked as liveUpdate in config.cc)
        """
        result = self.remoter.run("ps -C scylla -o pid --no-headers", ignore_status=True)
        if result.ok:
            pid = result.stdout.strip()
        else:
            LOGGER.error("Failed to obtain scylla pid - config not reloaded")
            return
        self.remoter.run(f"sudo kill -s SIGHUP {pid}")


class FlakyRetryPolicy(RetryPolicy):

    """
    A retry policy that retries 5 times
    """

    def _retry_message(self, msg, retry_num):
        if retry_num < 5:
            LOGGER.debug("%s. Attempt #%d", msg, retry_num)
            return self.RETRY, None
        return self.RETHROW, None

    # pylint: disable=too-many-arguments
    def on_read_timeout(self, query, consistency, required_responses,
                        received_responses, data_retrieved, retry_num):
        return self._retry_message(msg="Retrying read after timeout", retry_num=retry_num)

    # pylint: disable=too-many-arguments
    def on_write_timeout(self, query, consistency, write_type,
                         required_responses, received_responses, retry_num):
        return self._retry_message(msg="Retrying write after timeout", retry_num=retry_num)

    # pylint: disable=too-many-arguments
    def on_unavailable(self, query, consistency, required_replicas, alive_replicas, retry_num):
        return self._retry_message(msg="Retrying request after UE", retry_num=retry_num)


# pylint: disable=too-many-instance-attributes
@dataclass
class DeadNode:
    name: str
    public_ip: str
    private_ip: str
    ipv6_ip: str
    ip_address: str  # it's depend on SSH connection type (ip_ssh_connections): private|public|ipv6
    shards: int
    termination_time: str
    terminated_by_nemesis: str = ""
    ip: str = ""

    def __post_init__(self):
        if not self.ip:
            self.ip = f"{self.public_ip} | {self.private_ip}{f' | {self.ipv6_ip}' if self.ipv6_ip else ''}"


class BaseCluster:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    Cluster of Node objects.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, cluster_uuid=None, cluster_prefix='cluster', node_prefix='node', n_nodes=3, params=None,
                 region_names=None, node_type=None, extra_network_interface=False, add_nodes=True):
        self.extra_network_interface = extra_network_interface
        if params is None:
            params = {}
        if cluster_uuid is None:
            self.uuid = self.test_config.test_id()
        else:
            self.uuid = cluster_uuid
        self.node_type = node_type
        self.shortid = str(self.uuid)[:8]
        self.name = '%s-%s' % (cluster_prefix, self.shortid)
        self.node_prefix = '%s-%s' % (node_prefix, self.shortid)
        self._node_index = 0
        # I wanted to avoid some parameter passing
        # from the tester class to the cluster test.

        self.init_log_directory()

        self.log = SDCMAdapter(LOGGER, extra={'prefix': str(self)})
        self.log.info('Init nodes')
        self.nodes = []
        self.instance_provision = params.get('instance_provision')
        self.params = params
        self.datacenter = region_names or []
        self.dead_nodes_list = []
        self.use_ldap_authentication = self.params.get('use_ldap_authentication')
        # default 'cassandra' password is weak password, MS AD doesn't allow to use it.
        self.added_password_suffix = False
        self.node_benchmark_manager = ScyllaClusterBenchmarkManager()

        if self.test_config.REUSE_CLUSTER:
            # get_node_ips_param should be defined in child
            self._node_public_ips = self.params.get(self.get_node_ips_param(public_ip=True)) or []
            self._node_private_ips = self.params.get(self.get_node_ips_param(public_ip=False)) or []
            self.log.debug('Node public IPs: {}, private IPs: {}'.format(self._node_public_ips, self._node_private_ips))

        # NOTE: following is needed in case of K8S where we init multiple DB clusters first
        #       and only then we add nodes to it calling code in parallel.
        if add_nodes:
            if isinstance(n_nodes, list):
                for dc_idx, num in enumerate(n_nodes):
                    self.add_nodes(num, dc_idx=dc_idx, enable_auto_bootstrap=self.auto_bootstrap)
            elif isinstance(n_nodes, int):  # legacy type
                self.add_nodes(n_nodes, enable_auto_bootstrap=self.auto_bootstrap)
            else:
                raise ValueError('Unsupported type: {}'.format(type(n_nodes)))
            self.run_node_benchmarks()
        self.coredumps = {}
        super().__init__()

    @cached_property
    def test_config(self) -> TestConfig:  # pylint: disable=no-self-use
        return TestConfig()

    @property
    def auto_bootstrap(self):
        """
        When enabled scylla nodes will be bootstraped automatically on first boot
        """
        return not self.params.get('use_legacy_cluster_init')

    @cached_property
    def tags(self) -> Dict[str, str]:
        key = self.node_type if "db" not in self.node_type else "db"
        action = self.params.get(f"post_behavior_{key}_nodes")
        return {**self.test_config.common_tags(),
                "NodeType": str(self.node_type),
                "keep_action": "terminate" if action == "destroy" else "", }

    @property
    def dead_nodes_ip_address_list(self):
        return [node.ip_address for node in self.dead_nodes_list]

    def find_node_by_ip(self, node_ip):
        for node in self.nodes:
            if node_ip in node.get_all_ip_addresses():
                return node

        return None

    def init_log_directory(self):
        assert '_SCT_TEST_LOGDIR' in os.environ
        self.logdir = os.path.join(os.environ['_SCT_TEST_LOGDIR'], self.name)
        os.makedirs(self.logdir, exist_ok=True)

    def nodes_by_region(self, nodes=None) -> dict:
        """:returns {region_name: [list of nodes]}"""
        nodes = nodes if nodes else self.nodes
        grouped_by_region = defaultdict(list)
        for node in nodes:
            grouped_by_region[node.region].append(node)
        return grouped_by_region

    def send_file(self, src, dst, verbose=False):
        for loader in self.nodes:
            loader.remoter.send_files(src=src, dst=dst, verbose=verbose)

    def run(self, cmd, verbose=False):
        for loader in self.nodes:
            loader.remoter.run(cmd=cmd, verbose=verbose)

    def run_func_parallel(self, func, node_list=None):
        if node_list is None:
            node_list = self.nodes

        _queue = queue.Queue()
        for node in node_list:
            setup_thread = threading.Thread(target=func, args=(node, _queue), daemon=True)
            setup_thread.start()

        results = []
        while len(results) != len(node_list):
            try:
                results.append(_queue.get(block=True, timeout=5))
            except queue.Empty:
                pass
        return results

    def get_backtraces(self):
        for node in self.nodes:
            try:
                node.get_backtraces()
                if node.n_coredumps > 0:
                    self.coredumps[node.name] = node.n_coredumps
            except Exception as ex:  # pylint: disable=broad-except
                self.log.exception("Unable to get coredump status from node {node}: {ex}".format(node=node, ex=ex))

    def node_setup(self, node, verbose=False, timeout=3600):
        raise NotImplementedError("Derived class must implement 'node_setup' method!")

    def get_node_ips_param(self, public_ip=True):
        raise NotImplementedError("Derived class must implement 'get_node_ips_param' method!")

    def wait_for_init(self):
        raise NotImplementedError("Derived class must implement 'wait_for_init' method!")

    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False):
        """
        :param count: number of nodes to add
        :param ec2_user_data:
        :param dc_idx: datacenter index, used as an index for self.datacenter list
        :return: list of Nodes
        """
        raise NotImplementedError("Derived class must implement 'add_nodes' method!")

    def prepare_user_data(self, enable_auto_bootstrap=False):
        user_data_format_version = '3'

        # Using pre-built images, hence read the format version from configuration
        if self.params.get('use_preinstalled_scylla'):
            if self.node_type == 'scylla-db':
                user_data_format_version = self.params.get('user_data_format_version') or user_data_format_version
            elif self.node_type == 'oracle-db':
                user_data_format_version = self.params.get(
                    'oracle_user_data_format_version') or user_data_format_version

        user_data_builder = ScyllaUserDataBuilder(cluster_name=self.name,
                                                  bootstrap=enable_auto_bootstrap,
                                                  user_data_format_version=user_data_format_version, params=self.params,
                                                  syslog_host_port=self.test_config.get_logging_service_host_port())
        return user_data_builder.to_string()

    def get_node_private_ips(self):
        return [node.private_ip_address for node in self.nodes]

    def get_node_public_ips(self):
        return [node.public_ip_address for node in self.nodes]

    def get_node_cql_ips(self):
        return [node.cql_ip_address for node in self.nodes]

    def get_node_database_errors(self):
        errors = {}
        for node in self.nodes:
            node_errors = list(node.follow_system_log(start_from_beginning=True))
            if node_errors:
                errors.update({node.name: node_errors})
        return errors

    def destroy(self):
        self.log.info('Destroy nodes')
        for node in self.nodes:
            node.destroy()

    def terminate_node(self, node):
        if node.ip_address not in self.dead_nodes_ip_address_list:
            self.dead_nodes_list.append(DeadNode(
                name=node.name,
                public_ip=node.public_ip_address,
                private_ip=node.private_ip_address,
                ipv6_ip=node.ipv6_ip_address if self.test_config.IP_SSH_CONNECTIONS == "ipv6" else '',
                ip_address=node.ip_address,
                shards=node.scylla_shards,
                termination_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                terminated_by_nemesis=node.running_nemesis))

        if node in self.nodes:
            self.nodes.remove(node)
        node.destroy()

    def get_db_auth(self):
        if self.params.get('use_ldap') and self.params.get('are_ldap_users_on_scylla'):
            user = LDAP_USERS[0]
            password = LDAP_PASSWORD
        else:
            user = self.params.get('authenticator_user')
            password = self.params.get('authenticator_password')
            # default password of Role cassandra is `cassandra`, the weak password isn't allowed in MS-AD.
            if self.added_password_suffix:
                password = self.params.get('authenticator_password') + DEFAULT_PWD_SUFFIX
        return (user, password) if user and password else None

    def write_node_public_ip_file(self):
        public_ip_file_path = os.path.join(self.logdir, 'public_ips')
        with open(public_ip_file_path, 'w', encoding="utf-8") as public_ip_file:
            public_ip_file.write("%s" % "\n".join(self.get_node_public_ips()))
            public_ip_file.write("\n")

    def write_node_private_ip_file(self):
        private_ip_file_path = os.path.join(self.logdir, 'private_ips')
        with open(private_ip_file_path, 'w', encoding="utf-8") as private_ip_file:
            private_ip_file.write("%s" % "\n".join(self.get_node_private_ips()))
            private_ip_file.write("\n")

    def set_keep_alive_on_failure(self):
        for node in self.nodes:
            if hasattr(node, "set_keep_alive"):
                node.set_keep_alive()

    def get_node_by_ip(self, node_ip, datacenter=None):
        full_node_ip = f"{datacenter}.{node_ip}" if datacenter else node_ip
        for node in self.nodes:
            region_ip = f"{node.datacenter}.{node.ip_address}" if datacenter else node.ip_address
            if region_ip == full_node_ip:
                return node
        return None

    def _create_session(self, node, keyspace, user, password, compression,
                        # pylint: disable=too-many-arguments, too-many-locals
                        protocol_version, load_balancing_policy=None,
                        port=None, ssl_opts=None, node_ips=None, connect_timeout=None,
                        verbose=True, connection_bundle_file=None):
        if not port:
            port = node.CQL_PORT

        if protocol_version is None:
            protocol_version = 3

        if user is None and password is None:
            credentials = self.get_db_auth()
            user, password = credentials if credentials else (None, None)

        if user is not None:
            auth_provider = PlainTextAuthProvider(username=user, password=password)
        else:
            auth_provider = None

        if ssl_opts is None and self.params.get('client_encrypt'):
            ssl_opts = {'ca_certs': './data_dir/ssl_conf/client/catest.pem'}
        self.log.debug(str(ssl_opts))
        kwargs = dict(contact_points=node_ips, port=port, ssl_options=ssl_opts)
        if connection_bundle_file:
            kwargs = dict(scylla_cloud=connection_bundle_file)
        cluster_driver = ClusterDriver(auth_provider=auth_provider,
                                       compression=compression,
                                       protocol_version=protocol_version,
                                       load_balancing_policy=load_balancing_policy,
                                       default_retry_policy=FlakyRetryPolicy(),
                                       connect_timeout=connect_timeout, **kwargs
                                       )
        session = cluster_driver.connect()

        # temporarily increase client-side timeout to 1m to determine
        # if the cluster is simply responding slowly to requests
        session.default_timeout = 60.0

        if keyspace is not None:
            session.set_keyspace(keyspace)

        # override driver default consistency level of LOCAL_QUORUM
        session.default_consistency_level = ConsistencyLevel.ONE

        return ScyllaCQLSession(session, cluster_driver, verbose)

    def cql_connection(self, node, keyspace=None, user=None,  # pylint: disable=too-many-arguments
                       password=None, compression=True, protocol_version=None,
                       port=None, ssl_opts=None, connect_timeout=100, verbose=True):
        if connection_bundle_file := node.parent_cluster.connection_bundle_file:
            wlrr = None
            node_ips = []
        else:
            node_ips = self.get_node_cql_ips()
            wlrr = WhiteListRoundRobinPolicy(node_ips)
        return self._create_session(node=node, keyspace=keyspace, user=user, password=password,
                                    compression=compression, protocol_version=protocol_version,
                                    load_balancing_policy=wlrr, port=port, ssl_opts=ssl_opts, node_ips=node_ips,
                                    connect_timeout=connect_timeout, verbose=verbose,
                                    connection_bundle_file=connection_bundle_file)

    def cql_connection_exclusive(self, node, keyspace=None, user=None,  # pylint: disable=too-many-arguments,too-many-locals
                                 password=None, compression=True,
                                 protocol_version=None, port=None,
                                 ssl_opts=None, connect_timeout=100, verbose=True):
        if connection_bundle_file := node.parent_cluster.connection_bundle_file:
            # TODO: handle the case of multiple datacenters
            bundle_yaml = yaml.safe_load(connection_bundle_file.open('r', encoding='utf-8'))
            node_domain = None
            for _, connection_data in bundle_yaml.get('datacenters', {}).items():
                node_domain = connection_data.get('nodeDomain').strip()
            assert node_domain, f"didn't found nodeDomain in bundle [{connection_bundle_file}]"

            def host_filter(host):
                return str(host.host_id) == str(node.host_id) or node_domain == host.endpoint._server_name  # pylint: disable=protected-access
            wlrr = HostFilterPolicy(child_policy=RoundRobinPolicy(), predicate=host_filter)
            node_ips = []
        else:
            node_ips = [node.cql_ip_address]
            wlrr = WhiteListRoundRobinPolicy(node_ips)
        return self._create_session(node=node, keyspace=keyspace, user=user, password=password,
                                    compression=compression, protocol_version=protocol_version,
                                    load_balancing_policy=wlrr, port=port, ssl_opts=ssl_opts, node_ips=node_ips,
                                    connect_timeout=connect_timeout, verbose=verbose,
                                    connection_bundle_file=connection_bundle_file)

    @retrying(n=8, sleep_time=15, allowed_exceptions=(NoHostAvailable,))
    def cql_connection_patient(self, node, keyspace=None,
                               # pylint: disable=too-many-arguments,unused-argument
                               user=None, password=None,
                               compression=True, protocol_version=None,
                               port=None, ssl_opts=None, connect_timeout=100, verbose=True):
        """
        Returns a connection after it stops throwing NoHostAvailables.

        If the timeout is exceeded, the exception is raised.
        """
        kwargs = locals()
        del kwargs["self"]
        return self.cql_connection(**kwargs)

    @retrying(n=8, sleep_time=15, allowed_exceptions=(NoHostAvailable,))
    def cql_connection_patient_exclusive(self, node, keyspace=None,
                                         # pylint: disable=invalid-name,too-many-arguments,unused-argument
                                         user=None, password=None,
                                         compression=True,
                                         protocol_version=None,
                                         port=None, ssl_opts=None, connect_timeout=100, verbose=True):
        """
        Returns a connection after it stops throwing NoHostAvailables.

        If the timeout is exceeded, the exception is raised.
        """
        # pylint: disable=unused-argument
        kwargs = locals()
        del kwargs["self"]
        return self.cql_connection_exclusive(**kwargs)

    def get_non_system_ks_cf_list(self, db_node,  # pylint: disable=too-many-arguments
                                  filter_out_table_with_counter=False, filter_out_mv=False, filter_empty_tables=True) -> List[str]:
        return self.get_any_ks_cf_list(db_node, filter_out_table_with_counter=filter_out_table_with_counter,
                                       filter_out_mv=filter_out_mv, filter_empty_tables=filter_empty_tables,
                                       filter_out_system=True, filter_out_cdc_log_tables=True)

    def get_any_ks_cf_list(self, db_node,  # pylint: disable=too-many-arguments
                           filter_out_table_with_counter=False, filter_out_mv=False, filter_empty_tables=True,
                           filter_out_system=False, filter_out_cdc_log_tables=False) -> List[str]:
        regular_column_names = ["keyspace_name", "table_name"]
        materialized_view_column_names = ["keyspace_name", "view_name"]
        regular_table_names, materialized_view_table_names = set(), set()

        def execute_cmd(cql_session, entity_type):
            result = set()
            is_column_type = entity_type == "column"
            column_names = regular_column_names
            if is_column_type:
                cmd = f"SELECT {column_names[0]}, {column_names[1]}, type FROM system_schema.columns"
            elif entity_type == "view":
                column_names = materialized_view_column_names
                cmd = f"SELECT {column_names[0]}, {column_names[1]} FROM system_schema.views"
            else:
                raise ValueError(f"The following value '{entity_type}' not supported")

            session.default_fetch_size = 1000
            session.default_consistency_level = ConsistencyLevel.ONE
            execute_result = session.execute_async(cmd)
            fetcher = PageFetcher(execute_result).request_all()
            current_rows = fetcher.all_data()

            # for row in cql_session.execute(cmd).current_rows:
            for row in current_rows:
                is_valid_table = True
                table_name = f"{getattr(row, column_names[0])}.{getattr(row, column_names[1])}"

                if filter_out_system and getattr(row, column_names[0]).startswith(("system", "alternator_usertable")):
                    is_valid_table = False
                elif is_column_type and (filter_out_table_with_counter and "counter" in row.type):
                    is_valid_table = False
                elif filter_out_cdc_log_tables and getattr(row, column_names[1]).endswith(cdc.options.CDC_LOGTABLE_SUFFIX):
                    is_valid_table = False
                elif is_column_type and filter_empty_tables:
                    current_rows = 0
                    # Scylls issue https://github.com/scylladb/scylla/issues/7186
                    # Problem to read from system_schema.dropped_columns, column "dropped_time":
                    # cassandra.DriverException: Failed decoding result column "dropped_time" of type timestamp:
                    # date value out of range
                    if table_name == 'system_schema.dropped_columns':
                        continue

                    try:
                        current_rows = cql_session.execute(f"SELECT * FROM {table_name} LIMIT 1").current_rows
                    except Exception as exc:  # pylint: disable=broad-except
                        self.log.warning(f'Failed to get rows from {table_name} table. Error: {exc}')

                    if not current_rows:
                        is_valid_table = False

                if is_valid_table:
                    result.add(table_name)
            return result

        with self.cql_connection_patient(db_node) as session:
            regular_table_names = execute_cmd(cql_session=session, entity_type="column")
            if regular_table_names and filter_out_mv:
                materialized_view_table_names = execute_cmd(cql_session=session, entity_type="view")
        if not regular_table_names:
            return []

        return list(regular_table_names - materialized_view_table_names)

    def is_table_has_data(self, session, table_name: str) -> (bool, Optional[Exception]):
        """
        Return True if `table_name` has data in it.
        Checking if a result page with any data is received.
        """
        try:
            result = session.execute(SimpleStatement(f"SELECT * FROM {table_name}", fetch_size=10))
            return bool(len(result.one())), None

        except Exception as exc:  # pylint: disable=broad-except
            self.log.warning(f'Failed to get rows from {table_name} table. Error: {exc}')
            return False, exc

    def get_all_tables_with_cdc(self, db_node: BaseNode) -> List[str]:
        """Return list of all tables with enabled cdc feature

        Get all non system keyspaces and tables and return
        list of keyspace.table with enable cdc feature
        :param db_node: [description]
        :type db_node: BaseNode
        :returns: [description]
        :rtype: {List[str]}
        """
        all_ks_cf = self.get_any_ks_cf_list(db_node,
                                            filter_out_system=True,
                                            filter_out_mv=True)
        self.log.debug(all_ks_cf)
        ks_tables_with_cdc = [ks_cf_cdc[:-len(cdc.options.CDC_LOGTABLE_SUFFIX)]
                              for ks_cf_cdc in all_ks_cf if ks_cf_cdc.endswith(cdc.options.CDC_LOGTABLE_SUFFIX)]
        self.log.debug(ks_tables_with_cdc)

        return ks_tables_with_cdc

    def get_all_tables_with_twcs(self, db_node: BaseNode) -> List[Dict[str, Union[Dict[str, Any], int, str]]]:
        """ Get list of keyspace.table with TWCS
            example of returning dict: {
                "name": ks.test,
                "compaction": {"class": "TimeWindowCompactionStrategy",
                                "compaction_window_size": 1,
                                "compaction_window_unit": "MINUTES"},
                "gc": 100,
                "dttl": 100,
                }
        """
        all_ks_cf = self.get_any_ks_cf_list(db_node, filter_out_system=True, filter_out_mv=True)
        self.log.debug("Found user's tables: %s", all_ks_cf)
        twcs_tables_list = []
        twcs_query_search = "SELECT default_time_to_live as dttl, gc_grace_seconds as gc, compaction from system_schema.tables\
                             WHERE keyspace_name='{ks}' and table_name='{cf}'"

        with self.cql_connection_patient(db_node) as session:
            for ks_cf in all_ks_cf:
                ks, cf = ks_cf.split(".")
                result = session.execute(twcs_query_search.format(ks=ks, cf=cf))

                if result and result[0].compaction["class"] == "TimeWindowCompactionStrategy":
                    twcs_tables_list.append({"name": ks_cf,
                                             "compaction": dict(result[0].compaction),
                                             "gc": result[0].gc,
                                             "dttl": result[0].dttl})

        return twcs_tables_list

    def run_node_benchmarks(self):
        if not self.params.get("run_db_node_benchmarks") or not self.nodes:
            return

        self.node_benchmark_manager.add_nodes(self.nodes)
        self.node_benchmark_manager.install_benchmark_tools()
        self.node_benchmark_manager.run_benchmarks()
        self.get_node_benchmarks_results()

    def get_node_benchmarks_results(self):
        if not self.params.get("run_db_node_benchmarks") or not self.nodes:
            return None

        return self.node_benchmark_manager.comparison

    @property
    def cluster_backend(self):
        return self.params.get("cluster_backend")


class NodeSetupFailed(Exception):
    def __init__(self, node, error_msg, traceback_str=""):
        super().__init__(error_msg)
        self.node = node
        self.error_msg = error_msg
        self.traceback_str = "\n" + traceback_str if traceback_str else ""

    def __str__(self):
        return f"[{self.node}] NodeSetupFailed: {self.error_msg}{self.traceback_str}"

    def __repr__(self):
        return self.__str__()


class NodeSetupTimeout(Exception):
    pass


def wait_for_init_wrap(method):  # pylint: disable=too-many-statements
    """
    Wraps wait_for_init class method.
    Run setup of nodes simultaneously and wait for all the setups finished.
    Raise exception if setup failed or timeout expired.
    """
    @wraps(method)
    def wrapper(*args, **kwargs):  # pylint: disable=too-many-statements
        cl_inst = args[0]
        LOGGER.debug('Class instance: %s', cl_inst)
        LOGGER.debug('Method kwargs: %s', kwargs)
        node_list = kwargs.get('node_list', None) or cl_inst.nodes
        timeout = kwargs.get('timeout', None)
        # remove all arguments which is not supported by BaseScyllaCluster.node_setup method
        setup_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ["node_list", "check_node_health", "wait_for_db_logs"]}

        _queue = queue.Queue()

        @raise_event_on_failure
        def node_setup(_node: BaseNode):
            exception_details = None
            try:
                cl_inst.node_setup(_node, **setup_kwargs)
            except Exception as ex:  # pylint: disable=broad-except
                exception_details = (str(ex), traceback.format_exc())
            try:
                _node.argus_resource_set_shards()
                ArgusTestRun.get().save()
            except Exception:  # pylint: disable=broad-except
                LOGGER.warning("Failure settings shards for node %s in Argus.", _node)
                LOGGER.debug("Exception details:\n", exc_info=True)
            _queue.put((_node, exception_details))
            _queue.task_done()

        def verify_node_setup(start_time):
            time_elapsed = time.perf_counter() - start_time
            try:
                node, setup_exception = _queue.get(block=True, timeout=5)
                if setup_exception:
                    raise NodeSetupFailed(node=node, error_msg=setup_exception[0], traceback_str=setup_exception[1])
                results.append(node)
                cl_inst.log.info("(%d/%d) nodes ready, node %s. Time elapsed: %d s",
                                 len(results), len(node_list), str(node), int(time_elapsed))
            except queue.Empty:
                pass
            if timeout and time_elapsed / 60 > timeout:
                msg = 'TIMEOUT [%d min]: Waiting for node(-s) setup(%d/%d) expired!' % (
                    timeout, len(results), len(node_list))
                cl_inst.log.error(msg)
                raise NodeSetupTimeout(msg)

        start_time = time.perf_counter()
        init_nodes = []
        results = []

        if isinstance(cl_inst, BaseScyllaCluster):
            # Update installed scylla before node setup, scylla server will be start at the end of node_setup
            cl_inst.update_db_binary(node_list, start_service=False)
            cl_inst.update_db_packages(node_list, start_service=False)

        for node in node_list:
            if isinstance(cl_inst, BaseScyllaCluster) \
                    and not getattr(cl_inst, 'params', {}).get('use_legacy_cluster_init'):
                init_nodes.append(node)
                start_time = time.perf_counter()
                node_setup(node)
                verify_node_setup(start_time)
            else:
                setup_thread = threading.Thread(target=node_setup, name='NodeSetupThread',
                                                args=(node,), daemon=True)
                setup_thread.start()
                if isinstance(cl_inst, BaseScyllaCluster):
                    cl_inst.log.info("Wait 120 seconds before next node setup")
                    time.sleep(120)

        while len(results) != len(node_list):
            verify_node_setup(start_time)

        if isinstance(cl_inst, BaseScyllaCluster):
            cl_inst.wait_for_nodes_up_and_normal(nodes=node_list, verification_node=node_list[0], timeout=timeout)

        time_elapsed = time.perf_counter() - start_time
        cl_inst.log.debug('TestConfig duration -> %s s', int(time_elapsed))

        method(*args, **kwargs)
    return wrapper


class ClusterNodesNotReady(Exception):
    pass


class BaseScyllaCluster:  # pylint: disable=too-many-public-methods, too-many-instance-attributes, too-many-statements
    node_setup_requires_scylla_restart = True
    name: str
    nodes: List[BaseNode]
    log: logging.Logger

    def __init__(self, *args, **kwargs):
        self.nemesis_termination_event = threading.Event()
        self.nemesis = []
        self.nemesis_threads = []
        self.nemesis_count = 0
        self.test_config = TestConfig()
        self._node_cycle = None
        self.params = kwargs.get('params', {})
        super().__init__(*args, **kwargs)

    def get_node_ips_param(self, public_ip=True):
        if self.test_config.MIXED_CLUSTER:
            return 'oracle_db_nodes_public_ip' if public_ip else 'oracle_db_nodes_private_ip'
        return 'db_nodes_public_ip' if public_ip else 'db_nodes_private_ip'

    def get_scylla_args(self):
        # pylint: disable=no-member
        return self.params.get('append_scylla_args_oracle') if self.name.find('oracle') > 0 else \
            self.params.get('append_scylla_args')

    def get_rack_nodes(self, rack: int) -> list:
        return sorted([node for node in self.nodes if node.rack == rack], key=lambda n: n.name)

    @cached_property
    def proposed_scylla_yaml(self) -> ScyllaYaml:
        """
        A scylla yaml that cluster imposes on nodes, it is calculated of the SCTConfiguration,
        and TestConfig is providing LDAP-related information
        """
        cluster_params_builder = ScyllaYamlClusterAttrBuilder(
            cluster_name=self.name,
            params=self.params,
            test_config=self.test_config,
            msldap_server_info=KeyStore().get_ldap_ms_ad_credentials()
        )
        return ScyllaYaml(**cluster_params_builder.dict(exclude_unset=True, exclude_none=True))

    @property
    def connection_bundle_file(self) -> Path | None:
        bundle_file = self.params.get("k8s_connection_bundle_file")
        return Path(bundle_file) if bundle_file else None

    @property
    def racks(self) -> Set[int]:
        return {node.rack for node in self.nodes}

    def set_seeds(self, wait_for_timeout=300, first_only=False):
        seeds_selector = self.params.get('seeds_selector')
        seeds_num = self.params.get('seeds_num')

        seed_nodes_ips = None
        if first_only:
            node = self.nodes[0]
            node.wait_ssh_up()
            seed_nodes_ips = [node.ip_address]

        elif (seeds_selector == 'reflector'
                or self.test_config.REUSE_CLUSTER
                or self.params.get('db_type') == 'cloud_scylla'):
            node = self.nodes[0]
            node.wait_ssh_up()
            # When cluster just started, seed IP in the scylla.yaml may be like '127.0.0.1'
            # In this case we want to ignore it and wait, when reflector will select real node and update scylla.yaml
            seed_nodes_ips = wait.wait_for(self.get_seed_selected_by_reflector,
                                           step=10, text='Waiting for seed is selected by reflector',
                                           timeout=wait_for_timeout, throw_exc=True)
        else:
            if seeds_selector == 'random':
                selected_nodes = random.sample(self.nodes, seeds_num)
            # seeds_selector == 'first'
            else:
                selected_nodes = self.nodes[:seeds_num]

            seed_nodes_ips = [node.ip_address for node in selected_nodes]

        for node in self.nodes:
            if node.ip_address in seed_nodes_ips:
                node.set_seed_flag(True)

        assert seed_nodes_ips, "We should have at least one selected seed by now"

    @property
    def seed_nodes_ips(self):
        seed_nodes_ips = [node.ip_address for node in self.nodes if node.is_seed]
        assert seed_nodes_ips, "We should have at least one selected seed by now"
        return seed_nodes_ips

    @property
    def seed_nodes(self):
        seed_nodes = [node for node in self.nodes if node.is_seed]
        assert seed_nodes, "We should have at least one selected seed by now"
        return seed_nodes

    @property
    def non_seed_nodes(self):
        return [node for node in self.nodes if not node.is_seed]

    def validate_seeds_on_all_nodes(self):
        for node in self.nodes:
            yaml_seeds_ips = node.extract_seeds_from_scylla_yaml()
            for ip in yaml_seeds_ips:
                assert ip in self.seed_nodes_ips, \
                    'Wrong seed IP {act_ip} in the scylla.yaml on the {node_name} node. ' \
                    'Expected {exp_ips}'.format(node_name=node.name,
                                                exp_ips=self.seed_nodes_ips,
                                                act_ip=ip)

    @contextlib.contextmanager
    def patch_params(self) -> SCTConfiguration:
        new_params, old_params = self.params.copy(), self.params
        yield new_params
        if new_params != old_params:
            self.params = new_params
            for node in self.nodes:
                node.config_setup(append_scylla_args=self.get_scylla_args())
                node.restart_scylla()

    def enable_client_encrypt(self):
        self.log.debug("Enabling client encryption on nodes")
        with self.patch_params() as params:
            params['client_encrypt'] = True

    def disable_client_encrypt(self):
        with self.patch_params() as params:
            params['client_encrypt'] = False

    def _update_db_binary(self, new_scylla_bin, node_list, start_service=True):
        self.log.debug('User requested to update DB binary...')

        def update_scylla_bin(node, _queue):
            node.log.info('Updating DB binary')
            node.remoter.send_files(new_scylla_bin, '/tmp/scylla', verbose=True)

            # scylla binary is moved to different directory after relocation
            # check if the installed binary is the relocated one or not
            relocated_binary = '/opt/scylladb/libexec/scylla.bin'
            if node.file_exists(relocated_binary):
                binary_path = relocated_binary
            else:
                binary_path = '/usr/bin/scylla'

            self._wait_for_preinstalled_scylla(node)
            # replace the binary
            node.remoter.sudo(shell_script_cmd(f"""\
                cp -f {binary_path} {binary_path}.origin
                cp -f /tmp/scylla {binary_path}
                chown root:root {binary_path}
                chmod +x {binary_path}
            """))
            _queue.put(node)
            _queue.task_done()

        def stop_scylla(node, _queue):
            node.stop_scylla(verify_down=True, verify_up=False)
            _queue.put(node)
            _queue.task_done()

        def start_scylla(node, _queue):
            node.start_scylla(verify_down=True, verify_up=True)
            _queue.put(node)
            _queue.task_done()

        start_time = time.time()

        # First, stop *all* non seed nodes
        self.run_func_parallel(func=stop_scylla, node_list=self.non_seed_nodes)  # pylint: disable=no-member
        # First, stop *all* seed nodes
        self.run_func_parallel(func=stop_scylla, node_list=self.seed_nodes)  # pylint: disable=no-member
        # Then, update bin only on requested nodes
        self.run_func_parallel(func=update_scylla_bin, node_list=node_list)  # pylint: disable=no-member
        if start_service:
            # Start all seed nodes
            self.run_func_parallel(func=start_scylla, node_list=self.seed_nodes)  # pylint: disable=no-member
            # Start all non seed nodes
            self.run_func_parallel(func=start_scylla, node_list=self.non_seed_nodes)  # pylint: disable=no-member

        time_elapsed = time.time() - start_time
        self.log.debug('Update DB binary duration -> %s s', int(time_elapsed))

    def _update_db_packages(self, new_scylla_bin, node_list, start_service=True):
        self.log.debug('User requested to update DB packages...')

        def check_package_suites_distro(node, extension):
            files_fit_extension = node.remoter.run(f'ls /tmp/scylla/*.{extension}', ignore_status=True, verbose=True)
            if not files_fit_extension.ok:
                TestFrameworkEvent(
                    source=self.__class__.__name__,
                    source_method='update_scylla_packages',
                    message="Did not get the right packages for this distro",
                    severity=Severity.CRITICAL
                ).publish()

        def update_scylla_packages(node, _queue):
            node.log.info('Updating DB packages')
            node.remoter.run('mkdir /tmp/scylla')
            node.remoter.send_files(new_scylla_bin, '/tmp/scylla', verbose=True)

            self._wait_for_preinstalled_scylla(node)
            logging.info("unzipping any tar.gz rpms")
            node.remoter.run('tar -xvf /tmp/scylla/*.tar.gz -C /tmp/scylla/', ignore_status=True, verbose=True)

            # replace the packages
            node.log.debug('Node distro is %s, will check the package suffix received', node.distro)
            if node.distro.is_rhel_like:
                check_package_suites_distro(node, 'rpm')
                node.log.info('Installed RPMs before replacing with new RPM files')
                node.remoter.run('yum list installed | grep scylla', verbose=True)
                node.log.debug('Will replace the RPM files with the ones provided as a parameter to the test')
                node.remoter.sudo('rpm -URvh --replacefiles /tmp/scylla/*.rpm', ignore_status=False, verbose=True)
                node.log.info('Installed RPMs after replacing with new RPM files')
                node.remoter.run('yum list installed | grep scylla', verbose=True)
            elif node.distro.is_ubuntu:
                check_package_suites_distro(node, 'deb')

                @retrying(n=6, sleep_time=10, allowed_exceptions=(UnexpectedExit,))
                def dpkg_force_install():
                    node.remoter.sudo(shell_script_cmd("yes Y | dpkg --force-depends -i /tmp/scylla/scylla*"),
                                      ignore_status=False, verbose=True)

                node.log.info('Installed .deb packages before replacing with new .DEB files')
                node.log.info(node.scylla_packages_installed)
                dpkg_force_install()
                node.log.info('Installed .deb packages after replacing with new .DEB files')
                node.log.info(node.scylla_packages_installed)
            _queue.put(node)
            _queue.task_done()

        def stop_scylla(node, _queue):
            node.stop_scylla(verify_down=True, verify_up=False)
            _queue.put(node)
            _queue.task_done()

        def start_scylla(node, _queue):
            node.start_scylla(verify_down=True, verify_up=True)
            _queue.put(node)
            _queue.task_done()

        start_time = time.time()

        if len(node_list) == 1:
            # Stop only new nodes
            self.run_func_parallel(func=stop_scylla, node_list=node_list)  # pylint: disable=no-member
            # Then, update packages only on requested node
            self.run_func_parallel(func=update_scylla_packages, node_list=node_list)  # pylint: disable=no-member
            if start_service:
                # Start new nodes
                self.run_func_parallel(func=start_scylla, node_list=node_list)  # pylint: disable=no-member
        else:
            # First, stop *all* non seed nodes
            self.run_func_parallel(func=stop_scylla, node_list=self.non_seed_nodes)  # pylint: disable=no-member
            # First, stop *all* seed nodes
            self.run_func_parallel(func=stop_scylla, node_list=self.seed_nodes)  # pylint: disable=no-member
            # Then, update packages only on requested nodes
            self.run_func_parallel(func=update_scylla_packages, node_list=node_list)  # pylint: disable=no-member
            if start_service:
                # Start all seed nodes
                self.run_func_parallel(func=start_scylla, node_list=self.seed_nodes)  # pylint: disable=no-member
                # Start all non seed nodes
                self.run_func_parallel(func=start_scylla, node_list=self.non_seed_nodes)  # pylint: disable=no-member

        time_elapsed = time.time() - start_time
        self.log.debug('Update DB packages duration -> %s s', int(time_elapsed))

    def update_seed_provider(self):
        for node in self.nodes:
            with node.remote_scylla_yaml() as scylla_yml:
                scylla_yml.seed_provider = node.proposed_scylla_yaml.seed_provider

    def update_db_binary(self, node_list=None, start_service=True):
        if node_list is None:
            node_list = self.nodes

        new_scylla_bin = self.params.get('update_db_binary')
        if new_scylla_bin:
            self._update_db_binary(new_scylla_bin, node_list, start_service=start_service)

    def update_db_packages(self, node_list=None, start_service=True):
        new_scylla_bin = self.params.get('update_db_packages')
        if new_scylla_bin:
            if node_list is None:
                node_list = self.nodes
            self._update_db_packages(new_scylla_bin, node_list, start_service=start_service)

    @retrying(n=3, sleep_time=5)
    def get_nodetool_status(self, verification_node=None):  # pylint: disable=too-many-locals
        """
            Runs nodetool status and generates status structure.
            Status format:
            status = {
                "datacenter1": {
                    "ip1": {
                        'state': state,
                        'load': load,
                        'tokens': tokens,
                        'owns': owns,
                        'host_id': host_id,
                        'rack': rack
                    }
                }
            }
        :param verification_node: node to run the nodetool on
        :return: dict
        """
        if not verification_node:
            verification_node = random.choice(self.nodes)
        status = {}
        res = verification_node.run_nodetool('status', publish_event=False)

        data_centers = res.stdout.split("Datacenter: ")
        # see TestNodetoolStatus test in test_cluster.py
        pattern = re.compile(
            r"(?P<state>\w{2})\s+"
            r"(?P<ip>[\w:.]+)\s+"
            r"(?P<load>[\d.]+ [\w]+|\?)\s+"
            r"(?P<tokens>[\d]+)\s+"
            r"(?P<owns>[\w?]+)\s+"
            r"(?P<host_id>[\w-]+)\s{2,}"
            r"(?P<rack>[\w]+|$)")

        for dc in data_centers:
            if not dc:
                continue
            lines = dc.splitlines()
            dc_name = lines[0]
            status[dc_name] = {}
            for line in lines[1:]:
                match = pattern.match(line)
                if not match:
                    continue
                node_info = match.groupdict()
                node_ip = node_info.pop("ip")
                # NOTE: following replacement is needed for the K8S case where
                #       registered IP is different than the one used for network connections
                if verification_node.is_kubernetes():
                    for node in self.nodes:
                        if node_ip in node.get_all_ip_addresses() and node_ip != node.ip_address:
                            node_ip = node.ip_address
                node_info["load"] = node_info["load"].replace(" ", "")
                status[dc_name][node_ip] = node_info
        return status

    @staticmethod
    def get_nodetool_info(node):
        """
            Runs nodetool info and generates status structure.
            Info format:

            :param node: node to run the nodetool on
            :return: dict
        """
        res = node.run_nodetool('info')
        # Removing unnecessary lines from the output
        proper_yaml_output = "\n".join([line for line in res.stdout.splitlines() if ":" in line])
        info_res = yaml.safe_load(proper_yaml_output)
        return info_res

    def check_cluster_health(self):
        # Task 1443: ClusterHealthCheck is bottle neck in scale test and create a lot of noise in 5000 tables test.
        # Disable it
        if not self.params.get('cluster_health_check'):
            self.log.debug('Cluster health check disabled')
            return

        with ClusterHealthValidatorEvent() as chc_event:
            # Don't run health check in case parallel nemesis.
            # TODO: find how to recognize, that nemesis on the node is running
            if self.nemesis_count == 1:
                for node in self.nodes:
                    node.check_node_health()
            else:
                chc_event.message = "Test runs with parallel nemesis. Nodes health checks are disabled."
                return

            self.check_nodes_running_nemesis_count()
            chc_event.message = "Cluster health check finished"

    def check_nodes_running_nemesis_count(self):
        nodes_running_nemesis = [node for node in self.nodes if node.running_nemesis]

        # Support parallel nemesis: nemesis_count is amount of nemesises that runs in parallel
        if len(nodes_running_nemesis) <= self.nemesis_count:
            return

        message = "; ".join(f"{node.ip_address} ({'seed' if node.is_seed else 'non-seed'}): {node.running_nemesis}"
                            for node in nodes_running_nemesis)
        ClusterHealthValidatorEvent.NodesNemesis(
            severity=Severity.WARNING,
            message=f"There are more then expected nodes running nemesis: {message}",
        ).publish()

    @retrying(n=6, sleep_time=10, allowed_exceptions=(AssertionError,))
    def wait_for_schema_agreement(self):

        for node in self.nodes:
            err = check_schema_agreement_in_gossip_and_peers(node)
            assert not err, err
        self.log.debug('Schema agreement is reached')
        return True

    def check_nodes_up_and_normal(self, nodes=None, verification_node=None):
        """Checks via nodetool that node joined the cluster and reached 'UN' state"""
        if not nodes:
            nodes = self.nodes
        status = self.get_nodetool_status(verification_node=verification_node)
        up_statuses = []
        for node in nodes:
            found_node_status = False
            for dc_status in status.values():
                ip_status = dc_status.get(node.ip_address)
                if ip_status:
                    found_node_status = True
                    up_statuses.append(ip_status["state"] == "UN")
                    break
            if not found_node_status:
                up_statuses.append(False)
        if not all(up_statuses):
            raise ClusterNodesNotReady("Not all nodes joined the cluster")

    def get_nodes_up_and_normal(self, verification_node=None):
        """Checks via nodetool that node joined the cluster and reached 'UN' state"""
        status = self.get_nodetool_status(verification_node=verification_node)
        up_nodes = []
        for node in self.nodes:
            for dc_status in status.values():
                ip_status = dc_status.get(node.ip_address)
                if ip_status:
                    if ip_status["state"] == "UN":
                        up_nodes.append(node)
        return up_nodes

    def get_node_status_dictionary(self, ip_address=None, verification_node=None):
        """Get node status dictionary via nodetool (in case it's not found return None)"""
        node_status = None
        if ip_address is None:
            return node_status
        status = self.get_nodetool_status(verification_node=verification_node)
        for dc_status in status.values():
            ip_status = dc_status.get(ip_address)
            if ip_status:
                node_status = ip_status
                break
        return node_status

    def wait_for_nodes_up_and_normal(self, nodes=None, verification_node=None, iterations=60, sleep_time=3, timeout=0):  # pylint: disable=too-many-arguments
        @retrying(n=iterations, sleep_time=sleep_time, allowed_exceptions=NETWORK_EXCEPTIONS + (ClusterNodesNotReady,),
                  message="Waiting for nodes to join the cluster", timeout=timeout)
        def _wait_for_nodes_up_and_normal():
            self.check_nodes_up_and_normal(nodes=nodes, verification_node=verification_node)

        _wait_for_nodes_up_and_normal()

    def get_test_keyspaces(self):
        """Function returning a list of non-system keyspaces (created by test)"""
        keyspaces = self.nodes[0].run_cqlsh("describe keyspaces").stdout.split()
        return [ks for ks in keyspaces if not ks.startswith("system")]

    def cfstat_reached_threshold(self, key, threshold, keyspaces=None):
        """
        Find whether a certain cfstat key in all nodes reached a certain value.

        :param key: cfstat key, example, 'Space used (total)'.
        :param threshold: threshold value for cfstats key. Example, 2432043080.
        :param keyspace: keyspace name or table full name(example: 'keyspace1.standard1'). If keyspaces is None,
                         receive all
        :return: Whether all nodes reached that threshold or not.
        """
        if not keyspaces:
            keyspaces = self.get_test_keyspaces()

        self.log.debug("Waiting for threshold: %s" % (threshold))
        node = self.nodes[0]
        node_space = 0
        # Calculate space on the disk of all test keyspaces on the one node.
        # It's decided to check the threshold on one node only
        for keyspace_name in keyspaces:
            self.log.debug("Get cfstats on the node %s for %s keyspace" %
                           (node.name, keyspace_name))
            node_space += node.get_cfstats(keyspace_name)[key]
        self.log.debug("Current cfstats on the node %s for %s keyspaces: %s" %
                       (node.name, keyspaces, node_space))
        reached_threshold = True
        if node_space < threshold:
            reached_threshold = False
        if reached_threshold:
            self.log.debug("Done waiting on cfstats: %s" % node_space)
        return reached_threshold

    def wait_total_space_used_per_node(self, size=None, keyspace='keyspace1'):
        if size is None:
            size = int(self.params.get('space_node_threshold'))
        if size:
            if keyspace and not isinstance(keyspace, list):
                keyspace = [keyspace]
            key = 'Space used (total)'
            wait.wait_for(func=self.cfstat_reached_threshold, step=10, timeout=600,
                          text="Waiting until cfstat '%s' reaches value '%s'" % (key, size),
                          key=key, threshold=size, keyspaces=keyspace, throw_exc=False)

    def add_nemesis(self, nemesis, tester_obj):
        for nem in nemesis:
            for _ in range(nem['num_threads']):
                self.nemesis.append(nem['nemesis'](tester_obj=tester_obj,
                                                   termination_event=self.nemesis_termination_event))
        self.nemesis_count = sum(nem['num_threads'] for nem in nemesis)

    def clean_nemesis(self):
        self.nemesis = []

    @log_run_info("Start nemesis threads on cluster")
    def start_nemesis(self, interval=None, cycles_count: int = -1):
        self.log.info('Clear _nemesis_termination_event')
        self.nemesis_termination_event.clear()
        for nemesis in self.nemesis:
            nemesis_thread = threading.Thread(target=nemesis.run, name='NemesisThread',
                                              args=(interval, cycles_count), daemon=True)
            nemesis_thread.start()
            self.nemesis_threads.append(nemesis_thread)

    @log_run_info("Stop nemesis threads on cluster")
    def stop_nemesis(self, timeout=10):
        if self.nemesis_termination_event.is_set():
            return
        self.log.info('Set _nemesis_termination_event')
        self.log.debug("There are %s nemesis threads currently running", len(self.nemesis_threads))
        self.nemesis_termination_event.set()
        threads_tracebacks = []
        # pylint: disable=protected-access
        current_thread_frames = sys._current_frames()
        for nemesis_thread in self.nemesis_threads:
            nemesis_thread.join(timeout)
            if nemesis_thread.is_alive():
                stack_trace = traceback.format_stack(current_thread_frames[nemesis_thread.ident])
                threads_tracebacks.append("\n".join(stack_trace))
        self.argus_terminate_remaining_nemeses(stack_trace="\n".join(threads_tracebacks))

    def argus_terminate_remaining_nemeses(self, stack_trace="") -> None:
        try:
            run = ArgusTestRun.get()
            running_nemeses = []
            for nemesis in run.run_info.results.nemesis_data:
                if nemesis.nemesis_status == NemesisStatus.RUNNING:
                    running_nemeses.append(nemesis)

            self.log.info("Attempting to terminate %s running nemeses in Argus", len(running_nemeses))
            for nemesis in running_nemeses:
                self.log.debug("Marking nemesis \"%s\" as terminated in argus...", nemesis.name)
                nemesis.complete(f"Nemesis termination at the end of the test\n{stack_trace}")
                nemesis.nemesis_status = NemesisStatus.TERMINATED
            run.save()
        except Exception:  # pylint: disable=broad-except
            self.log.warning("Error recording nemesis termination information in Argus")
            self.log.debug(exc_info=True)

    def scylla_configure_non_root_installation(self, node, devname, verbose, timeout):
        node.stop_scylla_server(verify_down=False)
        node.remoter.run(f'{INSTALL_DIR}/sbin/scylla_setup --nic {devname} --no-raid-setup --no-io-setup',
                         verbose=True, ignore_status=True)
        node.remoter.send_files(src='./configurations/io.conf', dst=f'{INSTALL_DIR}/etc/scylla.d/')
        node.remoter.send_files(src='./configurations/io_properties.yaml', dst=f'{INSTALL_DIR}/etc/scylla.d/')
        node.remoter.run(
            fr"sed -ie 's/io-properties-file=/io-properties-file=\/home\/{TEST_USER}\/scylladb/g' {INSTALL_DIR}/etc/scylla.d/io.conf")
        node.remoter.run(
            fr"sed -ie 's/mountpoint: .*/mountpoint: \/home\/{TEST_USER}\/scylladb/g' {INSTALL_DIR}/etc/scylla.d/io_properties.yaml")

        # simple config
        node.remoter.run(
            f"echo 'cluster_name: \"{self.name}\"' >> {INSTALL_DIR}/etc/scylla/scylla.yaml")  # pylint: disable=no-member
        node.remoter.run(
            f"sed -ie 's/- seeds: .*/- seeds: {node.ip_address}/g' {INSTALL_DIR}/etc/scylla/scylla.yaml")
        node.remoter.run(
            f"sed -ie 's/^listen_address: .*/listen_address: {node.ip_address}/g' {INSTALL_DIR}/etc/scylla/scylla.yaml")
        node.remoter.run(
            f"sed -ie 's/^rpc_address: .*/rpc_address: {node.ip_address}/g' {INSTALL_DIR}/etc/scylla/scylla.yaml")

        node.start_scylla_server(verify_up=False, verify_up_timeout=timeout)
        node.wait_db_up(verbose=verbose, timeout=timeout)
        node.wait_jmx_up(verbose=verbose, timeout=200)

    def copy_preconfigured_iotune_files(self, node):
        self.log.info("This AMI need to be tweaked for io.conf and properties")
        for conf in ['io.conf', 'io_properties.yaml']:
            node.remoter.send_files(src=os.path.join('./configurations/', conf),
                                    # pylint: disable=not-callable
                                    dst='/tmp/')
            node.remoter.run('sudo mv /tmp/{0} /etc/scylla.d/{0}'.format(conf))

    def node_setup(self, node: BaseNode, verbose: bool = False, timeout: int = 3600):  # pylint: disable=too-many-branches,too-many-statements
        node.wait_ssh_up(verbose=verbose, timeout=timeout)
        if node.distro.is_centos8 or node.distro.is_rhel8 or node.distro.is_oel8 or node.distro.is_rocky8:
            node.remoter.sudo('systemctl stop iptables', ignore_status=True)
            node.remoter.sudo('systemctl disable iptables', ignore_status=True)
            node.remoter.sudo('systemctl stop firewalld', ignore_status=True)
            node.remoter.sudo('systemctl disable firewalld', ignore_status=True)

        if self.params.get('logs_transport') == 'ssh':
            node.install_package('python3')

        node.update_repo_cache()

        install_scylla = True

        if self.params.get("use_preinstalled_scylla") and node.is_scylla_installed(raise_if_not_installed=True):
            install_scylla = False

        if not self.test_config.REUSE_CLUSTER:
            node.disable_daily_triggered_services()
            nic_devname = node.get_nic_devices()[0]
            if install_scylla:
                self._scylla_install(node)
            else:
                self.log.info("Waiting for preinstalled Scylla")
                self._wait_for_preinstalled_scylla(node)
                self.log.info("Done waiting for preinstalled Scylla")
                if self.params.get('workaround_kernel_bug_for_iotune'):
                    self.copy_preconfigured_iotune_files(node)
            if self.params.get('print_kernel_callstack'):
                save_kallsyms_map(node=node)
            if node.is_nonroot_install:
                self.scylla_configure_non_root_installation(node=node, devname=nic_devname,
                                                            verbose=verbose, timeout=timeout)
                return

            if self.test_config.BACKTRACE_DECODING:
                node.install_scylla_debuginfo()

            if self.test_config.MULTI_REGION:
                node.datacenter_setup(self.datacenter)  # pylint: disable=no-member
            node.config_setup(append_scylla_args=self.get_scylla_args())

            self._scylla_post_install(node, install_scylla, nic_devname)

            # prepare and start saslauthd service
            if self.params.get('prepare_saslauthd'):
                prepare_and_start_saslauthd_service(node)

            if self.node_setup_requires_scylla_restart:
                node.stop_scylla_server(verify_down=False)
                node.clean_scylla_data()
                node.remoter.sudo(cmd="rm -f /etc/scylla/ami_disabled", ignore_status=True)

                if self.is_additional_data_volume_used():
                    result = node.remoter.sudo(cmd="scylla_io_setup")
                    if result.ok:
                        self.log.info("Scylla_io_setup result: %s", result.stdout)

                node.start_scylla_server(verify_up=False)

            # code to increase java heap memory to scylla-jmx (because of #7609)
            if jmx_memory := self.params.get("jmx_heap_memory"):
                node.increase_jmx_heap_memory(jmx_memory)
                node.restart_scylla_jmx()

            self.log.debug('io.conf right after reboot: %s', node.remoter.sudo('cat /etc/scylla.d/io.conf').stdout)

            if self.params.get('use_mgmt'):
                self.install_scylla_manager(node)
        else:
            self._reuse_cluster_setup(node)

        node.wait_db_up(verbose=verbose, timeout=timeout)
        nodes_status = node.get_nodes_status()
        check_nodes_status(nodes_status=nodes_status, current_node=node)

        self.clean_replacement_node_ip(node)

    def install_scylla_manager(self, node):
        pkgs_url = self.params.get("scylla_mgmt_pkg")
        pkg_path = None
        if pkgs_url:
            pkg_path = download_dir_from_cloud(pkgs_url)
            node.remoter.run(f"mkdir -p {pkg_path}")
            node.remoter.send_files(src=f"{pkg_path}*.rpm", dst=pkg_path)
        node.install_manager_agent(package_path=pkg_path)
        node.update_manager_agent_config(region=self.params.get("backup_bucket_region"))

    def _scylla_install(self, node):
        node.update_repo_cache()
        node.clean_scylla()

        install_mode = self.params.get('install_mode')
        try:
            mode = InstallMode(install_mode)
        except Exception as ex:
            raise ValueError(f'Invalid install mode: {install_mode}, err: {ex}') from ex

        if mode == InstallMode.WEB:
            node.web_install_scylla(scylla_version=self.params.get('scylla_version'))

        if self.params.get('unified_package'):
            node.offline_install_scylla(unified_package=self.params.get('unified_package'),
                                        nonroot=self.params.get('nonroot_offline_install'))
        else:
            node.install_scylla(scylla_repo=self.params.get('scylla_repo'))

    @staticmethod
    def _wait_for_preinstalled_scylla(node):
        pass

    @staticmethod
    def _scylla_post_install(node: BaseNode, new_scylla_installed: bool, devname: str) -> None:
        if new_scylla_installed:
            try:
                disks = node.detect_disks(nvme=True)
            except AssertionError:
                disks = node.detect_disks(nvme=False)
            node.scylla_setup(disks, devname)

    def _reuse_cluster_setup(self, node):
        pass

    @staticmethod
    def clean_replacement_node_ip(node):
        if node.replacement_node_ip:
            # If this is a replacement node, we need to set back configuration in case
            # when scylla-server process will be restarted
            node.replacement_node_ip = None
            node.remoter.run(
                f'sudo sed -i -e "s/^replace_address_first_boot:/# replace_address_first_boot:/g" '
                f'{node.add_install_prefix(SCYLLA_YAML_PATH)}')

    @staticmethod
    def verify_logging_from_nodes(nodes_list):
        absent = []
        for node in nodes_list:
            if not os.path.exists(node.system_log):
                absent.append(node)
        if absent:
            raise Exception(
                "No db log from the following nodes:\n%s\nfiles expected to be find:\n%s" % (
                    '\n'.join(map(lambda node: node.name, absent)),
                    '\n'.join(map(lambda node: node.system_log, absent))
                ))
        return True

    @wait_for_init_wrap
    def wait_for_init(self, node_list=None, verbose=False, timeout=None, check_node_health=True):  # pylint: disable=unused-argument
        node_list = node_list or self.nodes
        wait.wait_for(
            func=self.verify_logging_from_nodes,
            step=20,
            text="wait for db logs",
            timeout=300,
            throw_exc=True,
            nodes_list=node_list,
        )
        self.log.info("%s nodes configured and started.", node_list)

        # If wait_for_init is called during cluster initialization we may want this validation will be performed,
        # but if it was called from nemesis, we don't need it in the middle of nemesis. It may cause to not relevant
        # failures
        if check_node_health:
            node_list[0].check_node_health()

    def restart_scylla(self, nodes=None, random_order=False):
        nodes_to_restart = (nodes or self.nodes)[:]  # create local copy of nodes list
        if random_order:
            random.shuffle(nodes_to_restart)
        self.log.info("Going to restart Scylla on %s", [n.name for n in nodes_to_restart])
        for node in nodes_to_restart:
            node.stop_scylla(verify_down=True)
            node.start_scylla(verify_up=True)
            self.log.debug("'%s' restarted.", node.name)

    def get_seed_selected_by_reflector(self, node=None):
        """
        Check if reflector updated the scylla.yaml with selected seed IP
        """
        if not node:
            node = self.nodes[0]

        seed_nodes_ips = node.extract_seeds_from_scylla_yaml()
        # When cluster just started, seed IP in the scylla.yaml may be like '127.0.0.1'
        # In this case we want to ignore it and wait, when reflector will select real node and update scylla.yaml
        return [n.ip_address for n in self.nodes if n.ip_address in seed_nodes_ips]

    def get_node(self):
        if not self._node_cycle:
            self._node_cycle = itertools.cycle(self.nodes)
        return next(self._node_cycle)

    def backup_keyspace(self, ks):
        backup_name = generate_random_string(10)
        for node in self.nodes:
            node.run_nodetool('drain')
            node.run_nodetool('flush')

        for node in self.nodes:
            node.stop_scylla_server()

        for node in self.nodes:
            node.remoter.sudo(f'cp -r "/var/lib/scylla/data/{ks}" "/var/lib/scylla/data/{backup_name}"')

        for node in self.nodes:
            node.start_scylla_server()

        return ks, backup_name

    def restore_keyspace(self, backup_data):
        ks, backup_name = backup_data
        for node in self.nodes:
            node.stop_scylla_server()

        for node in self.nodes:
            node.remoter.sudo(shell_script_cmd(f"""\
                rm -rf '/var/lib/scylla/data/{ks}'
                cp -r '/var/lib/scylla/data/{backup_name}' '/var/lib/scylla/data/{ks}'
            """))

        for node in self.nodes:
            node.start_scylla_server()

        for node in self.nodes:
            node.run_nodetool('repair')

    def verify_decommission(self, node):
        def get_node_ip_list(verification_node):
            try:
                ip_node_list = []
                status = self.get_nodetool_status(verification_node)
                for nodes_ips in status.values():
                    ip_node_list.extend(nodes_ips.keys())
                return ip_node_list
            except Exception as details:  # pylint: disable=broad-except
                LOGGER.error(str(details))
                return None

        target_node_ip = node.ip_address
        undecommission_nodes = [n for n in self.nodes if n != node]

        verification_node = random.choice(undecommission_nodes)
        node_ip_list = get_node_ip_list(verification_node)
        while verification_node == node or node_ip_list is None:
            verification_node = random.choice(undecommission_nodes)
            node_ip_list = get_node_ip_list(verification_node)

        decommission_done = list(node.follow_system_log(
            patterns=['DECOMMISSIONING: done'], start_from_beginning=True))

        if target_node_ip in node_ip_list and not decommission_done:
            cluster_status = self.get_nodetool_status(verification_node)
            error_msg = ('Node that was decommissioned %s still in the cluster. '
                         'Cluster status info: %s' % (node,
                                                      cluster_status))

            LOGGER.error('Decommission %s FAIL', node)
            LOGGER.error(error_msg)
            raise NodeStayInClusterAfterDecommission(error_msg)

        LOGGER.info('Decommission %s PASS', node)
        self.terminate_node(node)  # pylint: disable=no-member
        self.test_config.tester_obj().monitors.reconfigure_scylla_monitoring()

    def decommission(self, node: BaseNode, timeout: int | float = None, soft_timeout: int | float = None):
        with SoftTimeoutEvent(soft_timeout=soft_timeout, operation="decommission"):
            node.run_nodetool("decommission", timeout=timeout)
        self.verify_decommission(node)

    @property
    def scylla_manager_node(self) -> BaseNode:
        return self.test_config.tester_obj().monitors.nodes[0]

    @property
    def scylla_manager_auth_token(self) -> str:
        return self.test_config.tester_obj().monitors.mgmt_auth_token

    @property
    def scylla_manager_cluster_name(self):
        return self.name

    def get_cluster_manager(self, create_cluster_if_not_exists: bool = True) -> AnyManagerCluster:
        if not self.params.get('use_mgmt'):
            raise ScyllaManagerError('Scylla-manager configuration is not defined!')
        manager_tool = mgmt.get_scylla_manager_tool(manager_node=self.scylla_manager_node, scylla_cluster=self)
        LOGGER.debug("sctool version is : {}".format(manager_tool.sctool.version))
        cluster_name = self.scylla_manager_cluster_name  # pylint: disable=no-member
        mgr_cluster = manager_tool.get_cluster(cluster_name)
        if not mgr_cluster and create_cluster_if_not_exists:
            self.log.debug("Could not find cluster : {} on Manager. Adding it to Manager".format(cluster_name))
            return self.create_cluster_manager(cluster_name, manager_tool=manager_tool)
        return mgr_cluster

    def create_cluster_manager(self, cluster_name: str, manager_tool=None, host_ip=None):
        if manager_tool is None:
            manager_tool = mgmt.get_scylla_manager_tool(manager_node=self.scylla_manager_node, scylla_cluster=self)
        if host_ip is None:
            host_ip = self.nodes[0].ip_address
        return manager_tool.add_cluster(name=cluster_name, host=host_ip, auth_token=self.scylla_manager_auth_token)

    def is_additional_data_volume_used(self) -> bool:
        """return true if additional data volume is configured

        :returns: if ebs volumes attached return true
        :rtype: {bool}
        """
        return self.params.get("data_volume_disk_num") > 0 or self.params.get('gce_pd_standard_disk_size_db') > 0

    def fstrim_scylla_disks_on_nodes(self):
        # if used ebs volumes with aws backend fstrim is not supported
        # fstrim: /var/lib/scylla: the discard operation is not supported
        if self.is_additional_data_volume_used():
            self.log.info("fstrim is not supported on additional data volume")
            return

        for node in self.nodes:
            node.fstrim_scylla_disks()

    def get_db_nodes_cpu_mode(self):
        results = {}

        for node in self.nodes:
            perftune_config = node.get_perftune_yaml()
            pattern = re.compile(r"(?!mode: )\wq")
            if "mode" in perftune_config:
                results.update({node.name: pattern.search(perftune_config).group()})
            else:
                results.update({node.name: perftune_config})

        self.log.info("DB nodes CPU modes: %s", results)
        return results


class BaseLoaderSet():

    def __init__(self, params):
        self._loader_cycle = None
        self.params = params
        self._gemini_version = None
        self.sb_write_timeseries_ts = None

    @property
    def gemini_version(self):
        if not self._gemini_version:
            try:
                result = self.nodes[0].remoter.run(f'docker run --rm {self.params.get("stress_image.gemini")} '
                                                   f'gemini --version', ignore_status=True)
                if result.ok:
                    self._gemini_version = get_gemini_version(result.stdout)
            except Exception as details:  # pylint: disable=broad-except
                self.log.error("Error get gemini version: %s", details)
        return self._gemini_version

    def node_setup(self, node, verbose=False, db_node_address=None, **kwargs):  # pylint: disable=unused-argument
        # pylint: disable=too-many-statements,too-many-branches

        self.log.info('Setup in BaseLoaderSet')
        node.wait_ssh_up(verbose=verbose)
        # add swap file
        if not TestConfig.REUSE_CLUSTER:
            swap_size = self.params.get("loader_swap_size")
            if not swap_size:
                self.log.info("Swap file for the loader is not configured")
            else:
                node.create_swap_file(swap_size)
        # update repo cache and system after system is up
        node.update_repo_cache()

        if TestConfig().REUSE_CLUSTER:
            self.kill_stress_thread()
            return

        node_exporter_setup = NodeExporterSetup()
        node_exporter_setup.install(node)

        if self.params.get("bare_loaders"):
            self.log.info("Don't install anything because bare loaders requested")
            return

        if self.params.get('client_encrypt'):
            node.config_client_encrypt()

        # Install java-8 for workaround hdrhistogram
        if node.is_rhel_like():
            node.remoter.sudo('yum install -y java-1.8.0-openjdk-devel', verbose=True, ignore_status=True)
            node.remoter.sudo("ln -sf /usr/lib/jvm/java-1.8.0-openjdk-1.8.0.*/jre/bin/java* /etc/alternatives/java",
                              verbose=True, ignore_status=True)
        else:
            node.remoter.sudo('apt install -y openjdk-8-jre', verbose=True, ignore_status=True)
            node.remoter.sudo("ln -sf /usr/lib/jvm/java-1.8.0-openjdk-*/bin/java* /etc/alternatives/java",
                              verbose=True, ignore_status=True)

        result = node.remoter.run('test -e ~/PREPARED-LOADER', ignore_status=True)
        node.remoter.sudo("bash -cxe \"echo '*\t\thard\tcore\t\tunlimited\n*\t\tsoft\tcore\t\tunlimited' "
                          ">> /etc/security/limits.d/20-coredump.conf\"")
        if result.exit_status == 0:
            self.log.debug('Skip loader setup for using a prepared AMI')
            return

        if node.is_ubuntu14():
            install_java_script = dedent("""
                apt-get install software-properties-common -y
                add-apt-repository -y ppa:openjdk-r/ppa
                add-apt-repository -y ppa:scylladb/ppa
                apt-get update
                apt-get install -y openjdk-8-jre-headless
                update-java-alternatives --jre-headless -s java-1.8.0-openjdk-amd64
            """)
            node.remoter.run('sudo bash -cxe "%s"' % install_java_script)

        elif node.is_debian8():
            install_java_script = dedent(r"""
                sed -i -e 's/jessie-updates/stable-updates/g' /etc/apt/sources.list
                echo 'deb http://archive.debian.org/debian jessie-backports main' |sudo tee /etc/apt/sources.list.d/backports.list
                sed -e 's/:\/\/.*\/debian jessie-backports /:\/\/archive.debian.org\/debian jessie-backports /g' /etc/apt/sources.list.d/*.list
                echo 'Acquire::Check-Valid-Until false;' |sudo tee /etc/apt/apt.conf.d/99jessie-backports
                apt-get update
                apt-get install gnupg-curl -y
                apt-key adv --fetch-keys https://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-jessie/Debian_8.0/Release.key
                echo 'deb http://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-jessie/Debian_8.0/ /' |sudo tee /etc/apt/sources.list.d/scylla-3rdparty.list
                apt-get update
                apt-get install -y openjdk-8-jre-headless -t jessie-backports
                update-java-alternatives --jre-headless -s java-1.8.0-openjdk-amd64
            """)
            node.remoter.run('sudo bash -cxe "%s"' % install_java_script)
        elif node.distro.is_debian10 or node.distro.is_debian11:
            node.remoter.sudo(shell_script_cmd("""\
                apt-get update
                apt-get install -y openjdk-11-jre openjdk-11-jre-headless
            """))

        scylla_repo_loader = self.params.get('scylla_repo_loader')
        if not scylla_repo_loader:
            scylla_repo_loader = self.params.get('scylla_repo')
        node.download_scylla_repo(scylla_repo_loader)
        if node.is_rhel_like():
            node.remoter.run('sudo yum install -y {}-tools'.format(node.scylla_pkg()))
            node.remoter.sudo('yum install -y java-1.8.0-openjdk-devel', verbose=True, ignore_status=True)
            node.remoter.sudo("ln -sf /usr/lib/jvm/java-1.8.0-openjdk-*/jre/bin/java /etc/alternatives/java",
                              verbose=True, ignore_status=True)
        else:
            node.remoter.run('sudo apt-get update')
            node.remoter.run('sudo apt-get install -y '
                             ' {}-tools '.format(node.scylla_pkg()))
            node.remoter.sudo('apt instally -y openjdk-8-jre', verbose=True, ignore_status=True)
            node.remoter.sudo("ln -sf /usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java* /etc/alternatives/java",
                              verbose=True, ignore_status=True)

        if db_node_address is not None:
            node.remoter.run("echo 'export DB_ADDRESS=%s' >> $HOME/.bashrc" % db_node_address)

        node.wait_cs_installed(verbose=verbose)

        scylla_repo_loader = self.params.get('scylla_repo_loader')
        if not scylla_repo_loader:
            scylla_repo_loader = self.params.get('scylla_repo')
        node.download_scylla_repo(scylla_repo_loader)
        if node.is_rhel_like():
            node.remoter.run('sudo yum install -y {}-tools'.format(node.scylla_pkg()))
        else:
            node.remoter.run('sudo apt-get update')
            node.remoter.run('sudo apt-get install -y '
                             ' {}-tools '.format(node.scylla_pkg()))

        if db_node_address is not None:
            node.remoter.run("echo 'export DB_ADDRESS=%s' >> $HOME/.bashrc" % db_node_address)

        node.wait_cs_installed(verbose=verbose)

        # install docker
        docker_install = dedent("""
            curl -fsSL get.docker.com --retry 5 --retry-max-time 300 -o get-docker.sh
            sh get-docker.sh
            systemctl enable docker
            systemctl start docker
        """)
        node.remoter.run('sudo bash -cxe "%s"' % docker_install)

        node.remoter.run('sudo usermod -aG docker $USER', change_context=True)

        # Login to Docker Hub.
        docker_hub_login(remoter=node.remoter)

    @wait_for_init_wrap
    def wait_for_init(self, verbose=False, db_node_address=None):
        pass

    @staticmethod
    def get_node_ips_param(public_ip=True):
        return 'loaders_public_ip' if public_ip else 'loaders_private_ip'

    def get_loader(self):
        if not self._loader_cycle:
            self._loader_cycle = itertools.cycle(self.nodes)
        return next(self._loader_cycle)

    def kill_stress_thread(self):
        if self.nodes and self.nodes[0].is_kubernetes() and self.params.get(
                'k8s_loader_run_type') != 'static':
            for node in self.nodes:
                node.remoter.stop()
        else:
            if self.params.get("use_prepared_loaders"):
                self.kill_cassandra_stress_thread()
            else:
                self.kill_docker_loaders()

    def kill_cassandra_stress_thread(self):
        search_cmds = [
            'pgrep -f .*cassandra.*',
            'pgrep -f cassandra.stress',
            'pgrep -f cassandra-stress'
        ]

        def kill_cs_process(loader, filter_cmd):
            list_of_processes = loader.remoter.run(cmd=filter_cmd,
                                                   verbose=True, ignore_status=True)
            if not list_of_processes.stdout.strip():
                return True
            loader.remoter.run(cmd=f'{filter_cmd} | xargs -I{{}}  kill -TERM {{}}',
                               verbose=True, ignore_status=True)
            return False

        for loader in self.nodes:
            try:
                for search_cmd in search_cmds:
                    wait.wait_for(kill_cs_process, text="Search and kill c-s processes", timeout=30, throw_exc=False,
                                  loader=loader, filter_cmd=search_cmd)

            except Exception as ex:  # pylint: disable=broad-except
                self.log.warning("failed to kill stress-command on [%s]: [%s]",
                                 str(loader), str(ex))

    def kill_docker_loaders(self):
        for loader in self.nodes:
            try:
                loader.remoter.run(cmd='docker ps -a -q | xargs docker rm -f', verbose=True, ignore_status=True)
                self.log.info("Killed docker loader on node: %s", loader.name)
            except Exception as ex:  # pylint: disable=broad-except
                self.log.warning("failed to kill docker stress command on [%s]: [%s]",
                                 str(loader), str(ex))

    @staticmethod
    def _parse_cs_summary(lines):
        """
        Parsing c-stress results, only parse the summary results.
        Collect results of all nodes and return a dictionaries' list,
        the new structure data will be easy to parse, compare, display or save.
        """
        results = {}
        enable_parse = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Parse loader & cpu info
            if line.startswith('TAG:'):
                # TAG: loader_idx:1-cpu_idx:0-keyspace_idx:1
                ret = re.findall(r"TAG: loader_idx:(\d+)-cpu_idx:(\d+)-keyspace_idx:(\d+)", line)
                results['loader_idx'] = ret[0][0]
                results['cpu_idx'] = ret[0][1]
                results['keyspace_idx'] = ret[0][2]
                continue
            if line.startswith('Username:'):
                # Mode:
                # ...
                #   Username: null
                #   Password: null
                results['username'] = line.split('Username:')[1].strip()
            if line.startswith('Results:'):
                # Results:
                # Op rate                   :    9,999 op/s  [WRITE: 9,999 op/s]
                # Partition rate            :    9,999 pk/s  [WRITE: 9,999 pk/s]
                # Row rate                  :    9,999 row/s [WRITE: 9,999 row/s]
                # ....
                enable_parse = True
                continue
            if line == '':
                continue
            if line == 'END':
                break
            if not enable_parse:
                continue
            split_idx = line.find(':')
            if split_idx < 0:
                continue
            # Op rate                   :    9,999 op/s  [WRITE: 9,999 op/s]
            # Partition rate            :    9,999 pk/s  [WRITE: 9,999 pk/s]
            # Row rate                  :    9,999 row/s [WRITE: 9,999 row/s]
            # Latency mean              :    1.1 ms [WRITE: 1.1 ms]
            # Latency median            :    0.6 ms [WRITE: 0.6 ms]
            # Latency 95th percentile   :    2.3 ms [WRITE: 2.3 ms]
            # Latency 99th percentile   :    5.4 ms [WRITE: 5.4 ms]
            # Latency 99.9th percentile :   23.7 ms [WRITE: 23.7 ms]
            # Latency max               : 15787.4 ms [WRITE: 15,787.4 ms]
            # Total partitions          : 108,000,096 [WRITE: 108,000,096]
            # Total errors              :          0 [WRITE: 0]
            # Total GC count            : 0
            # Total GC memory           : 0.000 KiB
            # Total GC time             :    0.0 seconds
            key = line[:split_idx].strip().lower()
            value = line[split_idx + 1:].split()[0].replace(",", "")
            results[key] = value
            match = re.findall(r'\[READ:\s([\d,]+\.\d+)\sms,\sWRITE:\s([\d,]+\.\d)\sms\]', line)
            if match:  # parse results for mixed workload
                results['%s read' % key] = match[0][0]
                results['%s write' % key] = match[0][1]

        if not enable_parse:
            LOGGER.warning('Cannot find summary in c-stress results: %s', lines[-10:])
            return {}
        return results


class BaseMonitorSet:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    # This is a Mixin for monitoring cluster and should not be inherited
    DB_NODES_IP_ADDRESS = 'ip_address'
    json_file_params_for_replace = {"$test_name": get_test_name()}

    def __init__(self, targets, params, monitor_id: str = None):
        self.targets = targets
        self.params = params
        self.local_metrics_addr = start_metrics_server()  # start prometheus metrics server locally and return local ip
        self.sct_ip_port = self.set_local_sct_ip()
        self.grafana_port = 3000
        self.prometheus_retention = "365d"
        self.monitor_branch = self.params.get('monitor_branch')
        self._monitor_install_path_base = None
        self.phantomjs_installed = False
        self.grafana_start_time = 0
        self._sct_dashboard_json_file = None
        self.test_config = TestConfig()
        self.monitor_id = monitor_id or self.test_config.test_id()

    @staticmethod
    @retrying(n=5)
    def get_monitor_install_path_base(node):
        return os.path.join(node.remoter.run("echo $HOME").stdout.strip(), "sct-monitoring")

    @property
    def monitor_install_path_base(self):
        if not self._monitor_install_path_base:
            self._monitor_install_path_base = self.get_monitor_install_path_base(self.nodes[0])
        return self._monitor_install_path_base

    @property
    def monitor_install_path(self):
        return os.path.join(self.monitor_install_path_base, "scylla-monitoring-src")

    @property
    def monitoring_conf_dir(self):
        return os.path.join(self.monitor_install_path, "config")

    @property
    def monitoring_data_dir(self):
        return os.path.join(self.monitor_install_path_base, "scylla-monitoring-data")

    @staticmethod
    def get_node_ips_param(public_ip=True):
        param_name = 'monitor_nodes_public_ip' if public_ip else 'monitor_nodes_private_ip'
        return param_name

    @property
    def monitoring_version(self):
        scylla_version = self.targets["db_cluster"].nodes[0].scylla_version
        self.log.debug("Using %s ScyllaDB version to derive monitoring version", scylla_version)
        if scylla_version and "dev" not in scylla_version:
            if version := re.match(r"(\d+\.\d+)", scylla_version):
                return version.group(1)
        return "master"

    @property
    def sct_dashboard_json_file(self):
        if not self._sct_dashboard_json_file:
            sct_dashboard_json_filename = f"scylla-dash-per-server-nemesis.{self.monitoring_version}.json"
            sct_dashboard_json_path = get_data_dir_path(sct_dashboard_json_filename)
            if not os.path.exists(sct_dashboard_json_path):
                sct_dashboard_json_filename = "scylla-dash-per-server-nemesis.master.json"
                sct_dashboard_json_path = get_data_dir_path(sct_dashboard_json_filename)

            # NOTE: use separate path per monitoring to support multi-tenant setups in K8S
            original_path_parts = sct_dashboard_json_path.split("/")
            current_sct_dashboard_json_path = "/".join(
                original_path_parts[:-1]
                + [f"sct-monitor-{generate_random_string(5)}-{original_path_parts[-1]}"])
            shutil.copyfile(sct_dashboard_json_path, current_sct_dashboard_json_path)

            self._sct_dashboard_json_file = current_sct_dashboard_json_path
            self.sct_dashboard_json_file_content_update(update_params=self.json_file_params_for_replace,
                                                        json_file=self._sct_dashboard_json_file)
        return self._sct_dashboard_json_file

    @staticmethod
    def sct_dashboard_json_file_content_update(update_params: dict, json_file: str):
        # Read json data to the string
        with open(json_file, encoding="utf-8") as file:
            json_data = file.read()

        for param, value in update_params.items():
            json_data = json_data.replace(param, value)

        with open(json_file, 'w', encoding="utf-8") as file:
            json.dump(json.loads(json_data), file, indent=2)

    def node_setup(self, node, **kwargs):  # pylint: disable=unused-argument
        self.log.info('TestConfig in BaseMonitorSet')
        node.wait_ssh_up()
        # add swap file
        if not self.test_config.REUSE_CLUSTER:
            monitor_swap_size = self.params.get("monitor_swap_size")
            if not monitor_swap_size:
                self.log.info("Swap file for the monitor is not configured")
            else:
                node.create_swap_file(monitor_swap_size)
        # update repo cache and system after system is up
        node.update_repo_cache()
        self.mgmt_auth_token = self.monitor_id  # pylint: disable=attribute-defined-outside-init

        if self.test_config.REUSE_CLUSTER:
            self.configure_scylla_monitoring(node)
            self.restart_scylla_monitoring(sct_metrics=True)
            set_grafana_url(f"http://{normalize_ipv6_url(node.external_address)}:{self.grafana_port}")
            return

        self.install_scylla_monitoring(node)
        self.configure_scylla_monitoring(node)
        try:
            self.start_scylla_monitoring(node)
        except (Failure, UnexpectedExit):
            self.restart_scylla_monitoring()
        # The time will be used in url of Grafana monitor,
        # the data from this point to the end of test will
        # be captured.
        self.grafana_start_time = time.time()
        set_grafana_url(f"http://{normalize_ipv6_url(node.external_address)}:{self.grafana_port}")
        # since monitoring node is started last (after db nodes and loader) we can't actually set the timeout
        # for starting the alert manager thread (since it depends on DB cluster size and num of loaders)
        node.start_alert_manager_thread()  # remove when start task threads will be started after node setup
        if self.params.get("use_mgmt"):
            self.install_scylla_manager(node)

    def install_scylla_manager(self, node):
        if self.params.get("scylla_repo_m"):
            scylla_repo = self.params.get("scylla_repo_m")
        else:
            manager_scylla_backend_version = self.params.get("manager_scylla_backend_version")
            scylla_repo = get_manager_scylla_backend(manager_scylla_backend_version, node.distro)
        node.install_scylla(scylla_repo=scylla_repo)
        package_path = self.params.get("scylla_mgmt_pkg")
        if package_path:
            node.remoter.run(f"mkdir -p {package_path}")
            node.remoter.send_files(src=f"{package_path}*.rpm", dst=package_path)
        node.install_mgmt(package_url=package_path)
        self.nodes[0].wait_manager_server_up()

    def configure_ngrok(self):
        port = self.local_metrics_addr.split(':')[1]

        requests.delete('http://localhost:4040/api/tunnels/sct')

        tunnel = {
            "addr": port,
            "proto": "http",
            "name": "sct",
            "bind_tls": False
        }
        res = requests.post('http://localhost:4040/api/tunnels', json=tunnel)
        assert res.ok, "failed to add a ngrok tunnel [{}, {}]".format(res, res.text)
        ngrok_address = res.json()['public_url'].replace('http://', '')

        return "{}:80".format(ngrok_address)

    def set_local_sct_ip(self):

        ngrok_name = self.params.get('sct_ngrok_name')
        if ngrok_name:
            return self.configure_ngrok()

        sct_public_ip = self.params.get('sct_public_ip')
        if sct_public_ip:
            return sct_public_ip + ':' + self.local_metrics_addr.split(':')[1]
        else:
            return self.local_metrics_addr

    @wait_for_init_wrap
    def wait_for_init(self, *args, **kwargs):
        pass

    @staticmethod
    def install_scylla_monitoring_prereqs(node):  # pylint: disable=invalid-name
        if node.is_rhel_like():
            node.install_epel()
            node.update_repo_cache()
            prereqs_script = dedent("""
                yum install -y unzip wget
                yum install -y python36
                yum install -y python36-pip
                python3 -m pip install --upgrade pip
                python3 -m pip install pyyaml
                curl -fsSL get.docker.com --retry 5 --retry-max-time 300 -o get-docker.sh
                sh get-docker.sh
                systemctl start docker
            """)
        elif node.is_ubuntu():
            prereqs_script = dedent("""
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
                sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
                sudo apt-get update
                sudo apt-get install -y docker docker.io
                apt-get install -y software-properties-common
                apt-get install -y python3 python3-dev
                apt-get install -y python-setuptools unzip wget
                apt-get install -y python3-pip
                python3 -m pip install pyyaml
                python3 -m pip install -I -U psutil
                systemctl start docker
            """)
        elif node.distro.is_debian9 or node.distro.is_debian10 or node.distro.is_debian11:
            node.remoter.run(
                cmd="sudo apt install -y apt-transport-https ca-certificates curl software-properties-common gnupg2")
            node.remoter.run('curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -', retry=3)
            node.remoter.run(
                cmd='sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"')
            node.remoter.run(cmd="sudo apt update")
            node.remoter.run(cmd="sudo apt install -y docker-ce")
            node.remoter.run(cmd="sudo DEBIAN_FRONTEND=noninteractive apt install -y python3")
            node.remoter.run(cmd="sudo apt install -y python-setuptools wget unzip python3-pip")
            prereqs_script = dedent("""
                cat /etc/debian_version
            """)
        else:
            raise ValueError('Unsupported Distribution type: {}'.format(str(node.distro)))

        node.remoter.run(cmd="sudo bash -ce '%s'" % prereqs_script)
        node.remoter.run("sudo usermod -aG docker $USER", change_context=True)
        if node.is_debian9():
            node.reboot(hard=False)
        else:
            node.remoter.run(cmd='sudo systemctl restart docker', timeout=60)

        docker_hub_login(remoter=node.remoter)

    def download_scylla_monitoring(self, node):
        install_script = dedent("""
            sudo rm -rf {0.monitor_install_path_base}
            mkdir -p {0.monitor_install_path_base}
            cd {0.monitor_install_path_base}
            wget https://github.com/scylladb/scylla-monitoring/archive/{0.monitor_branch}.zip
            rm -rf ./tmp {0.monitor_install_path} 2>/dev/null
            unzip {0.monitor_branch}.zip -d ./tmp
            mv ./tmp/scylla-monitoring-{0.monitor_branch}/ {0.monitor_install_path}
            rm -rf ./tmp 2>/dev/null
        """.format(self))
        node.remoter.run("bash -ce '%s'" % install_script)
        if node.is_ubuntu():
            node.remoter.run(f'sed -i "s/python3/python3.6/g" {self.monitor_install_path}/*.py')

    @staticmethod
    def start_node_exporter(node):
        start_node_exporter_script = dedent('''
            docker run --restart=always -d --net="host" --pid="host" -v "/:/host:ro,rslave" --cap-add=SYS_TIME \
            quay.io/prometheus/node-exporter --path.rootfs=/host
        ''')
        node.remoter.run("bash -ce '%s'" % start_node_exporter_script)

    def configure_scylla_monitoring(self, node, sct_metrics=True, alert_manager=True):  # pylint: disable=too-many-locals
        cloud_prom_bearer_token = self.params.get('cloud_prom_bearer_token')

        if sct_metrics:
            temp_dir = tempfile.mkdtemp()
            template_fn = "prometheus.yml.template"
            prometheus_yaml_template = os.path.join(self.monitor_install_path, "prometheus", template_fn)
            local_template_tmp = os.path.join(temp_dir, template_fn + ".orig")
            local_template = os.path.join(temp_dir, template_fn)
            node.remoter.receive_files(src=prometheus_yaml_template,
                                       dst=local_template_tmp)
            with open(local_template_tmp, encoding="utf-8") as output_file:
                templ_yaml = yaml.safe_load(output_file)
                self.log.debug("Configs %s" % templ_yaml)
            loader_targets_list = ["[%s]:9100" % getattr(node, self.DB_NODES_IP_ADDRESS)
                                   for node in self.targets["loaders"].nodes]

            # remove those jobs if exists, for support of 'reuse_cluster: true'
            def remove_sct_metrics(metric):
                return metric['job_name'] not in ['stress_metrics', 'sct_metrics']
            templ_yaml["scrape_configs"] = list(filter(remove_sct_metrics, templ_yaml["scrape_configs"]))

            scrape_configs = templ_yaml["scrape_configs"]
            scrape_configs.append(dict(job_name="stress_metrics", honor_labels=True,
                                       static_configs=[dict(targets=loader_targets_list)]))

            if cloud_prom_bearer_token:
                cloud_prom_path = self.params.get('cloud_prom_path')
                cloud_prom_host = self.params.get('cloud_prom_host')
                scrape_configs.append(dict(job_name="scylla_cloud_cluster", honor_labels=True, scrape_interval='15s',
                                           metrics_path=cloud_prom_path, scheme='https',
                                           params={'match[]': ['{job=~".+"}']},
                                           bearer_token=cloud_prom_bearer_token,
                                           static_configs=[dict(targets=[cloud_prom_host])]))

            if self.params.get('gemini_cmd'):
                gemini_loader_targets_list = ["%s:2112" % getattr(node, self.DB_NODES_IP_ADDRESS)
                                              for node in self.targets["loaders"].nodes]
                scrape_configs.append(dict(job_name="gemini_metrics", honor_labels=True,
                                           static_configs=[dict(targets=gemini_loader_targets_list)]))

            nosqlbench_cmds = self.params.get('stress_cmd')
            if nosqlbench_cmds and "nosqlbench" in nosqlbench_cmds[0]:
                graphite_exporter_target_list = [f"{node.ip_address}:9108" for node in self.targets["loaders"].nodes]
                scrape_configs.append(dict(job_name="nosqlbench_metrics", honor_labels=True,
                                           static_configs=[dict(targets=graphite_exporter_target_list)]))

            if self.sct_ip_port:
                scrape_configs.append(dict(job_name="sct_metrics", honor_labels=True,
                                           static_configs=[dict(targets=[self.sct_ip_port])]))
            with open(local_template, "w", encoding="utf-8") as output_file:
                yaml.safe_dump(templ_yaml, output_file, default_flow_style=False)  # to remove tag !!python/unicode
            node.remoter.send_files(src=local_template, dst=prometheus_yaml_template, delete_dst=True)

            LOCALRUNNER.run(f"rm -rf {temp_dir}", ignore_status=True)

        self.reconfigure_scylla_monitoring()
        if alert_manager:
            self.configure_alert_manager(node)

    def configure_alert_manager(self, node):
        alertmanager_conf_file = os.path.join(self.monitor_install_path, "prometheus", "prometheus.rules.yml")
        conf = dedent("""

            # Alert for 99% cassandra stress write spikes
              - alert: CassandraStressWriteTooSlow
                expr: sct_cassandra_stress_write_gauge{type="lat_perc_99"} > 1000
                for: 1s
                labels:
                  severity: "1"
                  sct_severity: "ERROR"
                annotations:
                  description: "Cassandra Stress write latency more than 1000ms"
                  summary: "Cassandra Stress write latency is more than 1000ms during 1 sec period of time"

            # Alert for YCSB error spikes
              - alert: YCSBTooManyErrors
                expr: sum(rate(sct_ycsb_read_failed_gauge{type="count"}[1m])) > 5 OR sum(rate(sct_ycsb_update_failed_gauge{type="count"}[1m])) > 5  OR sum(rate(sct_ycsb_insert_failed_gauge{type="count"}[1m])) > 5
                for: 1s
                labels:
                  severity: "4"
                  sct_severity: "CRITICAL"
                annotations:
                  description: "YCSB errors more than 5 errors per min"
                  summary:  "YCSB errors more than 5 errors per min"

            # Alert for YCSB validation error spikes
              - alert: YCSBTooManyVerifyErrors
                expr: sum(rate(sct_ycsb_verify_gauge{type="ERROR"}[1m])) > 5 OR sum(rate(sct_ycsb_verify_gauge{type="UNEXPECTED_STATE"}[1m])) > 5
                for: 1s
                labels:
                  severity: "4"
                  sct_severity: "CRITICAL"
                annotations:
                  description: "YCSB verify errors more than 5 errors per min"
                  summary:  "YCSB verify errors more than 5 errors per min"
        """)
        with tempfile.NamedTemporaryFile("w") as alert_cont_tmp_file:
            alert_cont_tmp_file.write(conf)
            alert_cont_tmp_file.flush()
            node.remoter.send_files(src=alert_cont_tmp_file.name, dst=alert_cont_tmp_file.name)
            node.remoter.run("bash -ce 'cat %s >> %s'" % (alert_cont_tmp_file.name, alertmanager_conf_file))

    @retrying(n=5, sleep_time=10, allowed_exceptions=(Failure, UnexpectedExit),
              message="Waiting for restarting scylla monitoring")
    def restart_scylla_monitoring(self, sct_metrics=False):
        for node in self.nodes:
            self.stop_scylla_monitoring(node)
            # We use sct_metrics=False, alert_manager=False since they should be configured once
            self.configure_scylla_monitoring(node, sct_metrics=sct_metrics, alert_manager=False)
            self.start_scylla_monitoring(node)

    @retrying(n=5, sleep_time=10, allowed_exceptions=(Failure, UnexpectedExit),
              message="Waiting for reconfiguring scylla monitoring")
    def reconfigure_scylla_monitoring(self):
        for node in self.nodes:
            cluster_backend = self.params.get("cluster_backend")
            monitoring_targets = []
            for db_node in self.targets["db_cluster"].nodes:
                monitoring_targets.append(f"[{getattr(db_node, self.DB_NODES_IP_ADDRESS)}]:9180")
            monitoring_targets = " ".join(monitoring_targets)
            if self.params.get("ip_ssh_connections") != "ipv6":
                monitoring_targets = monitoring_targets.replace("[", "").replace("]", "")
            node.remoter.sudo(shell_script_cmd(f"""\
                cd {self.monitor_install_path}
                mkdir -p {self.monitoring_conf_dir}
                export PATH=/usr/local/bin:$PATH  # hack to enable running on docker
                python3 genconfig.py -s -n -d {self.monitoring_conf_dir} {monitoring_targets}
            """), verbose=True)

            if cluster_backend != "docker":
                node.remoter.sudo(f"docker run --rm -v {self.monitoring_conf_dir}:/workdir"
                                  f" mikefarah/yq:3 yq w -i node_exporter_servers.yml"
                                  f" '[0].targets[+]' ''[{node.private_ip_address}]:9100''", verbose=True)

            if self.params.get("cloud_prom_bearer_token"):
                node.remoter.sudo(shell_script_cmd(f"""\
                    echo "targets: [] " > {self.monitoring_conf_dir}/scylla_servers.yml
                    echo "targets: [] " > {self.monitoring_conf_dir}/node_exporter_servers.yml
                """), verbose=True)

            if self.params.get("scylla_rsyslog_setup"):
                for db_node in self.targets["db_cluster"].nodes:
                    db_node.remoter.sudo(shell_script_cmd(
                        f"[ -f /etc/rsyslog.d/scylla.conf ] || scylla_rsyslog_setup --remote-server {node.ip_address}"
                    ), verbose=True)

    def _create_manager_prometheus_yaml(self, node):
        manager_prometheus_port = self.params.get("manager_prometheus_port")
        # Manager server prometheus:
        manager_prometheus_target_list = [f"127.0.0.1:{manager_prometheus_port}"]
        full_yaml_string = "- targets:\n"
        for prometheus_target in manager_prometheus_target_list:
            full_yaml_string += f"  - {prometheus_target}\n"

        node.remoter.sudo(f"echo '{str(full_yaml_string)}' > "
                          f"{self.monitor_install_path}/prometheus/scylla_manager_servers.yml")
        # Writing directly to the yaml in the conf dir results in permission denied
        node.remoter.sudo(f"cp {self.monitor_install_path}/prometheus/scylla_manager_servers.yml "
                          f"{self.monitoring_conf_dir}/scylla_manager_servers.yml")

    def start_scylla_monitoring(self, node):
        self._create_manager_prometheus_yaml(node=node)
        labels = " ".join(f"--label {key}={value}" for key, value in node.tags.items())
        scylla_manager_servers_arg = ""
        if self.params.get("use_mgmt"):
            scylla_manager_servers_arg = f'-N `realpath "{self.monitoring_conf_dir}/scylla_manager_servers.yml"`'
        run_script = dedent(f"""
            cd -P {self.monitor_install_path}
            mkdir -p {self.monitoring_data_dir}
            echo "" > UA.sh
            PATH=$PATH:/usr/sbin ./start-all.sh \
            -D "{labels}" \
            -s `realpath "{self.monitoring_conf_dir}/scylla_servers.yml"` \
            -n `realpath "{self.monitoring_conf_dir}/node_exporter_servers.yml"` \
            {scylla_manager_servers_arg} \
            -d `realpath "{self.monitoring_data_dir}"` -l -v master,{self.monitoring_version} \
            -b "--web.enable-admin-api --storage.tsdb.retention.time={self.prometheus_retention}" \
            -c 'GF_USERS_DEFAULT_THEME=dark'
        """)
        node.remoter.run("bash -ce '%s'" % run_script, verbose=True)
        self.add_sct_dashboards_to_grafana(node)
        self.save_sct_dashboards_config(node)
        self.save_monitoring_version(node)

    def save_monitoring_version(self, node):
        node.remoter.run(
            'echo "{0.monitor_branch}:{0.monitoring_version}" > \
            {0.monitor_install_path}/monitor_version'.format(self), ignore_status=True)

    def add_sct_dashboards_to_grafana(self, node):

        def _register_grafana_json(json_filename):
            url = "'http://{0}:{1.grafana_port}/api/dashboards/db'".format(normalize_ipv6_url(node.external_address),
                                                                           self)
            result = LOCALRUNNER.run('curl -g -XPOST -i %s --data-binary @%s -H "Content-Type: application/json"' %
                                     (url, json_filename))
            return result.exited == 0

        wait.wait_for(_register_grafana_json, step=10,
                      text="Waiting to register '%s'..." % self.sct_dashboard_json_file,
                      json_filename=self.sct_dashboard_json_file, throw_exc=False)

    def save_sct_dashboards_config(self, node):
        sct_monitoring_addons_dir = os.path.join(self.monitor_install_path, 'sct_monitoring_addons')

        node.remoter.run('mkdir -p {}'.format(sct_monitoring_addons_dir), ignore_status=True)
        node.remoter.send_files(src=self.sct_dashboard_json_file, dst=sct_monitoring_addons_dir)

    @log_run_info
    def install_scylla_monitoring(self, node):
        self.install_scylla_monitoring_prereqs(node)
        self.download_scylla_monitoring(node)
        if not self.params.get('cluster_backend') == 'docker':
            self.start_node_exporter(node)

    def get_grafana_annotations(self, node):
        annotations_url = "http://{node_ip}:{grafana_port}/api/annotations?limit=10000"
        try:
            res = requests.get(url=annotations_url.format(node_ip=normalize_ipv6_url(node.grafana_address),
                                                          grafana_port=self.grafana_port))
            if res.ok:
                return res.content
        except Exception as ex:  # pylint: disable=broad-except
            LOGGER.warning("unable to get grafana annotations [%s]", str(ex))
        return ""

    def set_grafana_annotations(self, node, annotations_data):
        annotations_url = "http://{node_ip}:{grafana_port}/api/annotations"
        res = requests.post(url=annotations_url.format(node_ip=normalize_ipv6_url(node.grafana_address),
                                                       grafana_port=self.grafana_port),
                            data=annotations_data, headers={'Content-Type': 'application/json'})
        self.log.info("posting annotations result: %s", res)

    def stop_scylla_monitoring(self, node):
        kill_script = dedent(f"""
            cd {self.monitor_install_path}
            ./kill-all.sh
        """)
        node.remoter.run("bash -ce '%s'" % kill_script)

    def get_grafana_screenshot_and_snapshot(self, test_start_time: Optional[int] = None) -> dict[str, list[str]]:
        """
            Take screenshot of the Grafana per-server-metrics dashboard and upload to S3
        """
        if not test_start_time:
            self.log.error("No start time for test")
            return {}

        screenshot_links = []
        snapshot_links = []
        for node in self.nodes:
            screenshot_links.extend(self.get_grafana_screenshots(node, test_start_time))
            snapshot_links.extend(self.get_grafana_snapshots(node, test_start_time))

        return {'screenshots': screenshot_links, 'snapshots': snapshot_links}

    def get_grafana_screenshots(self, node: BaseNode, test_start_time: float) -> list[str]:
        screenshot_links = []
        grafana_extra_dashboards = []
        if 'alternator_port' in self.params:
            grafana_extra_dashboards = [AlternatorDashboard()]
        date_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        screenshot_collector = GrafanaScreenShot(name="grafana-screenshot",
                                                 test_start_time=test_start_time,
                                                 extra_entities=grafana_extra_dashboards)
        screenshot_files = screenshot_collector.collect(node, self.logdir)
        for screenshot in screenshot_files:
            s3_path = "{test_id}/{date}".format(test_id=self.test_config.test_id(), date=date_time)
            screenshot_links.append(S3Storage().upload_file(screenshot, s3_path))

        return screenshot_links

    def get_grafana_snapshots(self, node: BaseNode, test_start_time: float) -> list[str]:
        snapshot_links = []
        grafana_extra_dashboards = []
        if 'alternator_port' in self.params:
            grafana_extra_dashboards = [AlternatorDashboard()]

        snapshots_collector = GrafanaSnapshot(name="grafana-snapshot",
                                              test_start_time=test_start_time,
                                              extra_entities=grafana_extra_dashboards)
        snapshots_data = snapshots_collector.collect(node, self.logdir)
        snapshot_links.extend(snapshots_data.get('links', []))

        return snapshot_links

    def upload_annotations_to_s3(self):
        annotations_url = ''
        if not self.nodes:
            return annotations_url
        try:
            annotations = self.get_grafana_annotations(self.nodes[0])
            if annotations:
                annotations_url = S3Storage().generate_url('annotations.json', self.monitor_id)
                self.log.info("Uploading 'annotations.json' to {s3_url}".format(
                    s3_url=annotations_url))
                response = requests.put(annotations_url, data=annotations, headers={
                                        'Content-type': 'application/json; charset=utf-8'})
                response.raise_for_status()
        except Exception:  # pylint: disable=broad-except
            self.log.exception("failed to upload annotations to S3")

        return annotations_url

    @log_run_info
    def download_monitor_data(self) -> str:
        if not self.nodes:
            return ""
        try:
            if snapshot_archive := PrometheusSnapshots(name='prometheus_snapshot').collect(self.nodes[0], self.logdir):
                self.log.debug("Snapshot local path: %s", snapshot_archive)
                return upload_archive_to_s3(snapshot_archive, self.monitor_id)
        except Exception as details:  # pylint: disable=broad-except
            self.log.error("Error downloading prometheus data dir: %s", details)
        return ""


class NoMonitorSet():

    def __init__(self):

        self.log = SDCMAdapter(LOGGER, extra={'prefix': str(self)})
        self.nodes = []

    def __str__(self):
        return 'NoMonitorSet'

    def wait_for_init(self):
        self.log.info('Monitor nodes disabled for this run')

    def get_backtraces(self):
        pass

    def get_monitor_snapshot(self):
        pass

    def reconfigure_scylla_monitoring(self):
        pass

    def download_monitor_data(self):
        pass

    def destroy(self):
        pass

    def collect_logs(self, storage_dir):
        pass

    def get_grafana_screenshot_and_snapshot(self, test_start_time=None):  # pylint: disable=unused-argument,no-self-use,invalid-name
        return {}


class LocalNode(BaseNode):
    def __init__(self, name, parent_cluster,   # pylint: disable=too-many-arguments,unused-argument
                 ssh_login_info=None, base_logdir=None, node_prefix=None, dc_idx=0, rack=0):

        super().__init__(name=name, parent_cluster=parent_cluster, ssh_login_info=ssh_login_info,
                         base_logdir=base_logdir, node_prefix=node_prefix, dc_idx=dc_idx, rack=rack)

    def _init_port_mapping(self):
        pass

    def _init_remoter(self, ssh_login_info):
        self.remoter = LOCALRUNNER

    def _refresh_instance_state(self):
        return ['127.0.0.1'], ['127.0.0.1']

    @property
    def region(self):
        return "local"

    def set_keep_alive(self):
        pass

    def restart(self):
        pass

    def _get_ipv6_ip_address(self):
        pass

    def check_spot_termination(self):
        pass


class LocalK8SHostNode(LocalNode):
    def configure_remote_logging(self):
        self.log.debug("No need to configure remote logging on k8s")
