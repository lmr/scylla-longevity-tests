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

import queue
import getpass
import logging
import os
import random as librandom
import re
import tempfile
import threading
import time
import traceback
import uuid
import itertools
import json
import ipaddress

from collections import defaultdict
from sdcm.sct_events import Severity
from typing import List, Optional, Dict, Union, Set
from textwrap import dedent
from datetime import datetime
from functools import cached_property, wraps
from tenacity import RetryError

import yaml
import requests
from cassandra import ConsistencyLevel
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster as ClusterDriver  # pylint: disable=no-name-in-module
from cassandra.cluster import NoHostAvailable  # pylint: disable=no-name-in-module
from cassandra.policies import RetryPolicy
from cassandra.policies import WhiteListRoundRobinPolicy

from invoke.exceptions import UnexpectedExit, Failure, CommandTimedOut
from paramiko import SSHException

from sdcm.collectd import ScyllaCollectdSetup
from sdcm.mgmt import AnyManagerCluster, ScyllaManagerError
from sdcm.mgmt.cli import update_config_file, SCYLLA_MANAGER_YAML_PATH
from sdcm.prometheus import start_metrics_server, PrometheusAlertManagerListener, AlertSilencer
from sdcm.log import SDCMAdapter
from sdcm.remote import RemoteCmdRunnerBase, LOCALRUNNER, NETWORK_EXCEPTIONS, shell_script_cmd
from sdcm.remote.remote_file import remote_file
from sdcm import wait, mgmt
from sdcm.utils import alternator
from sdcm.utils.common import deprecation, get_data_dir_path, verify_scylla_repo_file, S3Storage, get_my_ip, \
    get_latest_gemini_version, normalize_ipv6_url, download_dir_from_cloud, generate_random_string, ScyllaCQLSession, \
    SCYLLA_YAML_PATH, get_test_name
from sdcm.utils.distro import Distro
from sdcm.utils.docker_utils import ContainerManager, NotFound

from sdcm.utils.health_checker import check_nodes_status, check_node_status_in_gossip_and_nodetool_status, \
    check_schema_version, check_nulls_in_peers, check_schema_agreement_in_gossip_and_peers, \
    CHECK_NODE_HEALTH_RETRIES, CHECK_NODE_HEALTH_RETRY_DELAY
from sdcm.utils.decorators import retrying, log_run_info
from sdcm.utils.get_username import get_username
from sdcm.utils.remotewebbrowser import WebDriverContainerMixin
from sdcm.utils.version_utils import SCYLLA_VERSION_RE, BUILD_ID_RE, get_gemini_version, get_systemd_version
from sdcm.sct_events.base import LogEvent
from sdcm.sct_events.health import ClusterHealthValidatorEvent
from sdcm.sct_events.system import TestFrameworkEvent
from sdcm.sct_events.database import SYSTEM_ERROR_EVENTS_PATTERNS, BACKTRACE_RE
from sdcm.sct_events.grafana import set_grafana_url
from sdcm.sct_events.decorators import raise_event_on_failure
from sdcm.utils.auto_ssh import AutoSshContainerMixin
from sdcm.utils.rsyslog import RSYSLOG_SSH_TUNNEL_LOCAL_PORT
from sdcm.logcollector import GrafanaSnapshot, GrafanaScreenShot, PrometheusSnapshots, upload_archive_to_s3
from sdcm.utils.ldap import LDAP_SSH_TUNNEL_LOCAL_PORT, LDAP_BASE_OBJECT, LDAP_PASSWORD, LDAP_USERS, LDAP_ROLE
from sdcm.utils.remote_logger import get_system_logging_thread
from sdcm.utils.scylla_args import ScyllaArgParser
from sdcm.utils.file import File
from sdcm.utils import cdc
from sdcm.coredump import CoredumpExportSystemdThread


CREDENTIALS = []
DEFAULT_USER_PREFIX = getpass.getuser()
# Test duration (min). Parameter used to keep instances produced by tests that
# are supposed to run longer than 24 hours from being killed
TEST_DURATION = 60
IP_SSH_CONNECTIONS = 'private'
TASK_QUEUE = 'task_queue'
RES_QUEUE = 'res_queue'
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCYLLA_DIR = "/var/lib/scylla"
TEST_USER = 'scylla-test'
INSTALL_DIR = f"/home/{TEST_USER}/scylladb"

INSTANCE_PROVISION_ON_DEMAND = 'on_demand'
SPOT_TERMINATION_CHECK_DELAY = 5

LOGGER = logging.getLogger(__name__)


def set_ip_ssh_connections(ip_type):
    # pylint: disable=global-statement
    global IP_SSH_CONNECTIONS
    IP_SSH_CONNECTIONS = ip_type


def set_duration(duration):
    # pylint: disable=global-statement
    global TEST_DURATION
    TEST_DURATION = duration


def remove_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


class Setup:
    KEEP_ALIVE_DB_NODES = False
    KEEP_ALIVE_LOADER_NODES = False
    KEEP_ALIVE_MONITOR_NODES = False

    REUSE_CLUSTER = False
    MIXED_CLUSTER = False
    MULTI_REGION = False
    BACKTRACE_DECODING = False
    INTRA_NODE_COMM_PUBLIC = False
    RSYSLOG_ADDRESS = None
    LDAP_ADDRESS = None
    DECODING_QUEUE = None
    USE_LEGACY_CLUSTER_INIT = False
    AUTO_BOOTSTRAP = True

    _test_id = None
    _test_name = None
    _logdir = None
    _tester_obj = None

    @classmethod
    def test_id(cls):
        return cls._test_id

    @classmethod
    def set_test_id(cls, test_id):
        if not cls._test_id:
            cls._test_id = str(test_id)
            test_id_file_path = os.path.join(cls.logdir(), "test_id")
            with open(test_id_file_path, "w") as test_id_file:
                test_id_file.write(str(test_id))
        else:
            LOGGER.warning("TestID already set!")

    @classmethod
    def tester_obj(cls):
        return cls._tester_obj

    @classmethod
    def set_tester_obj(cls, tester_obj):
        if not cls._tester_obj:
            cls._tester_obj = tester_obj

    @classmethod
    def base_logdir(cls) -> str:
        return os.path.expanduser(os.environ.get("_SCT_LOGDIR", "~/sct-results"))

    @classmethod
    def make_new_logdir(cls, update_latest_symlink: bool, postfix: str = "") -> str:
        base = cls.base_logdir()
        logdir = os.path.join(base, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + postfix)
        os.makedirs(logdir, exist_ok=True)
        LOGGER.info("New directory created: %s", logdir)
        if update_latest_symlink:
            latest_symlink = os.path.join(base, "latest")
            if os.path.islink(latest_symlink):
                os.remove(latest_symlink)
            os.symlink(os.path.relpath(logdir, base), latest_symlink)
            LOGGER.info("Symlink `%s' updated to `%s'", latest_symlink, logdir)
        return logdir

    @classmethod
    def logdir(cls) -> str:
        if not cls._logdir:
            cls._logdir = cls.make_new_logdir(update_latest_symlink=True)
            os.environ['_SCT_TEST_LOGDIR'] = cls._logdir
        return cls._logdir

    @classmethod
    def test_name(cls):
        return cls._test_name

    @classmethod
    def set_test_name(cls, test_name):
        cls._test_name = test_name

    @classmethod
    def set_multi_region(cls, multi_region):
        cls.MULTI_REGION = multi_region

    @classmethod
    def set_decoding_queue(cls):
        cls.DECODING_QUEUE = queue.Queue()

    @classmethod
    def set_intra_node_comm_public(cls, intra_node_comm_public):
        cls.INTRA_NODE_COMM_PUBLIC = intra_node_comm_public

    @classmethod
    def reuse_cluster(cls, val=False):
        cls.REUSE_CLUSTER = val

    @classmethod
    def keep_cluster(cls, node_type, val='destroy'):
        if "db_nodes" in node_type:
            cls.KEEP_ALIVE_DB_NODES = bool(val == 'keep')
        elif "loader_nodes" in node_type:
            cls.KEEP_ALIVE_LOADER_NODES = bool(val == 'keep')
        elif "monitor_nodes" in node_type:
            cls.KEEP_ALIVE_MONITOR_NODES = bool(val == 'keep')

    @classmethod
    def should_keep_alive(cls, node_type: Optional[str]) -> bool:
        if TEST_DURATION >= 24 * 60:
            return True
        if node_type is None:
            return False
        if "db" in node_type:
            return cls.KEEP_ALIVE_DB_NODES
        if "loader" in node_type:
            return cls.KEEP_ALIVE_LOADER_NODES
        if "monitor" in node_type:
            return cls.KEEP_ALIVE_MONITOR_NODES
        return False

    @classmethod
    def mixed_cluster(cls, val=False):
        cls.MIXED_CLUSTER = val

    @classmethod
    def common_tags(cls) -> Dict[str, str]:
        job_name = os.environ.get('JOB_NAME')
        tags = dict(RunByUser=get_username(),
                    TestName=str(cls.test_name()),
                    TestId=str(cls.test_id()),
                    version=job_name.split('/', 1)[0] if job_name else "unknown")

        build_tag = os.environ.get('BUILD_TAG')
        if build_tag:
            tags["JenkinsJobTag"] = build_tag

        return tags

    @classmethod
    def configure_ldap(cls, node, use_ssl=False):
        ContainerManager.run_container(node, "ldap")
        if use_ssl:
            port = node.ldap_ports['ldap_ssl_port']
        else:
            port = node.ldap_ports['ldap_port']
        address = get_my_ip()
        cls.LDAP_ADDRESS = (address, port)

    @classmethod
    def configure_rsyslog(cls, node, enable_ngrok=False):
        ContainerManager.run_container(node, "rsyslog", logdir=cls.logdir())
        port = node.rsyslog_port
        LOGGER.info("rsyslog listen on port %s (config: %s)", port, node.rsyslog_confpath)

        if enable_ngrok:
            requests.delete('http://localhost:4040/api/tunnels/rsyslogd')

            tunnel = {
                "addr": port,
                "proto": "tcp",
                "name": "rsyslogd",
                "bind_tls": False
            }
            res = requests.post('http://localhost:4040/api/tunnels', json=tunnel)
            assert res.ok, "failed to add a ngrok tunnel [{}, {}]".format(res, res.text)
            ngrok_address = res.json()['public_url'].replace('tcp://', '')

            address, port = ngrok_address.split(':')
        else:
            address = get_my_ip()

        cls.RSYSLOG_ADDRESS = (address, port)

    @classmethod
    def get_startup_script(cls):
        post_boot_script = '#!/bin/bash'
        post_boot_script += dedent(r'''
               sudo sed -i 's/#MaxSessions \(.*\)$/MaxSessions 1000/' /etc/ssh/sshd_config
               sudo sed -i 's/#MaxStartups \(.*\)$/MaxStartups 60/' /etc/ssh/sshd_config
               sudo sed -i 's/#LoginGraceTime \(.*\)$/LoginGraceTime 15s/' /etc/ssh/sshd_config
               sudo systemctl restart sshd
               ''')
        if cls.RSYSLOG_ADDRESS:

            if IP_SSH_CONNECTIONS == 'public' or Setup.MULTI_REGION:
                post_boot_script += dedent('''
                       sudo echo 'action(type="omfwd" Target="{0}" Port="{1}" Protocol="tcp")'>> /etc/rsyslog.conf
                       sudo systemctl restart rsyslog
                       '''.format('127.0.0.1', RSYSLOG_SSH_TUNNEL_LOCAL_PORT))
            else:
                post_boot_script += dedent('''
                       sudo echo 'action(type="omfwd" Target="{0}" Port="{1}" Protocol="tcp")'>> /etc/rsyslog.conf
                       sudo systemctl restart rsyslog
                       '''.format(*cls.RSYSLOG_ADDRESS))  # pylint: disable=not-an-iterable

        post_boot_script += dedent(r'''
               sed -i -e 's/^\*[[:blank:]]*soft[[:blank:]]*nproc[[:blank:]]*4096/*\t\tsoft\tnproc\t\tunlimited/' \
               /etc/security/limits.d/20-nproc.conf
               echo -e '*\t\thard\tnproc\t\tunlimited' >> /etc/security/limits.d/20-nproc.conf
               ''')
        return post_boot_script


class NodeError(Exception):

    def __init__(self, msg=None):
        super(NodeError, self).__init__()
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


def prepend_user_prefix(user_prefix, base_name):
    if not user_prefix:
        user_prefix = DEFAULT_USER_PREFIX
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

    def __init__(self, name, parent_cluster, ssh_login_info=None, base_logdir=None, node_prefix=None, dc_idx=0, rack=0):  # pylint: disable=too-many-arguments,unused-argument
        self.name = name
        self.rack = rack
        self.parent_cluster = parent_cluster  # reference to the Cluster object that the node belongs to
        self.ssh_login_info = ssh_login_info
        self.logdir = os.path.join(base_logdir, self.name) if base_logdir else None
        self.dc_idx = dc_idx

        self._containers = {}
        self.is_seed = False

        self.remoter = None

        self._spot_monitoring_thread = None
        self._journal_thread = None
        self._docker_log_process = None
        self._public_ip_address_cached = None
        self._private_ip_address_cached = None
        self._ipv6_ip_address_cached = None
        self._maximum_number_of_cores_to_publish = 10

        self.last_line_no = 1
        self.last_log_position = 0
        self._coredump_thread: Optional[CoredumpExportSystemdThread] = None
        self._db_log_reader_thread = None
        self._scylla_manager_journal_thread = None
        self._decoding_backtraces_thread = None
        self._init_system = None
        self.db_init_finished = False

        self._short_hostname = None
        self._alert_manager: Optional[PrometheusAlertManagerListener] = None

        self._system_log_errors_index = []
        self._exclude_system_log_from_being_logged = [
            ' !INFO    | sshd[',
            ' !INFO    | systemd:',
            ' !INFO    | systemd-logind:',
            ' !INFO    | sudo:',
            ' !INFO    | sshd[',
            ' !INFO    | dhclient['
        ]
        self.termination_event = threading.Event()
        self.lock = threading.Lock()

        self._running_nemesis = None

        # We should disable bootstrap when we create nodes to establish the cluster,
        # if we want to add more nodes when the cluster already exists, then we should
        # enable bootstrap.
        self.enable_auto_bootstrap = True

        self.scylla_version = ''
        self.scylla_version_detailed = ''

        # If node is a replacement for a dead node, store dead node private ip here
        self.replacement_node_ip = None

        self._kernel_version = None
        self._cassandra_stress_version = None

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
        if not Setup.REUSE_CLUSTER:
            self.set_hostname()

        self.start_task_threads()
        self._init_port_mapping()

        self.set_keep_alive()

    def _init_remoter(self, ssh_login_info):
        self.remoter = RemoteCmdRunnerBase.create_remoter(**ssh_login_info)
        self.log.debug(self.remoter.ssh_debug_cmd())

    def _init_port_mapping(self):
        if IP_SSH_CONNECTIONS == 'public' or Setup.MULTI_REGION:
            if Setup.RSYSLOG_ADDRESS:
                try:
                    ContainerManager.destroy_container(self, "auto_ssh:rsyslog", ignore_keepalive=True)
                except NotFound:
                    pass
                ContainerManager.run_container(self, "auto_ssh:rsyslog",
                                               local_port=Setup.RSYSLOG_ADDRESS[1],
                                               remote_port=RSYSLOG_SSH_TUNNEL_LOCAL_PORT)
            if Setup.LDAP_ADDRESS and self.parent_cluster.node_type == "scylla-db":
                try:
                    ContainerManager.destroy_container(self, "auto_ssh:ldap", ignore_keepalive=True)
                except NotFound:
                    pass
                ContainerManager.run_container(self, "auto_ssh:ldap",
                                               local_port=Setup.LDAP_ADDRESS[1],
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

    def refresh_ip_address(self):
        # Invalidate ip address cache
        self._private_ip_address_cached = self._public_ip_address_cached = self._ipv6_ip_address_cached = None

        if self.ssh_login_info["hostname"] == self.external_address:
            return

        self.ssh_login_info["hostname"] = self.external_address
        self.remoter.stop()
        self._init_remoter(self.ssh_login_info)
        self._init_port_mapping()

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**self.parent_cluster.tags,
                "Name": str(self.name), }

    def _set_keep_alive(self):
        ContainerManager.set_all_containers_keep_alive(self)
        return True

    def set_keep_alive(self):
        node_type = None if self.parent_cluster is None else self.parent_cluster.node_type
        if Setup.should_keep_alive(node_type) and self._set_keep_alive():
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

        if Setup.RSYSLOG_ADDRESS:
            rsys_log_path = os.path.join(Setup.logdir(), 'hosts', self.short_hostname, 'messages.log')
            if os.path.exists(rsys_log_path) and (not os.path.islink(orig_log_path)):
                os.symlink(os.path.relpath(rsys_log_path, self.logdir), orig_log_path)
            return rsys_log_path
        else:
            return orig_log_path

    database_log = system_log

    @property
    def cassandra_stress_version(self):
        if not self._cassandra_stress_version:
            result = self.remoter.run(cmd="cassandra-stress version", ignore_status=True, verbose=True)
            match = re.search("Version: (.*)", result.stdout)
            if match:
                self._cassandra_stress_version = match.group(1)
            else:
                self.log.error("C-S version not found!")
                self._cassandra_stress_version = "unknown"
        return self._cassandra_stress_version

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
        self.log.info("Trying to detect Linux distribution...")
        return Distro.from_os_release(self.remoter.run("cat /etc/os-release", ignore_status=True, retry=5).stdout)

    @cached_property
    def is_nonroot_install(self):
        return self.parent_cluster.params.get("unified_package") and self.parent_cluster.params.get("nonroot_offline_install")

    @property
    def is_client_encrypt(self):
        result = self.remoter.run(
            f"grep ^client_encryption_options: {self.add_install_prefix(SCYLLA_YAML_PATH)} -A 3 | grep enabled | awk '{{print $2}}'", ignore_status=True)
        return 'true' in result.stdout.lower()

    @property
    def cpu_cores(self):
        result = self.remoter.run("nproc", ignore_status=True)
        cores = result.stdout.lower()
        cores = int(cores) if cores else None
        return cores

    @property
    def is_server_encrypt(self):
        result = self.remoter.run(
            f"grep '^server_encryption_options:' {self.add_install_prefix(SCYLLA_YAML_PATH)}", ignore_status=True)
        return 'server_encryption_options' in result.stdout.lower()

    def extract_seeds_from_scylla_yaml(self):
        yaml_dst_path = os.path.join(tempfile.mkdtemp(prefix='sct'), 'scylla.yaml')
        wait.wait_for(func=self.remoter.receive_files, step=10, text='Waiting for copying scylla.yaml', timeout=300,
                      throw_exc=True, src=self.add_install_prefix(SCYLLA_YAML_PATH), dst=yaml_dst_path)
        with open(yaml_dst_path, 'r') as yaml_stream:
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
                raise ValueError('Exception determining seed node ips: %s' % details)

            if node_seeds:
                return node_seeds.split(',')
            else:
                raise Exception('Seeds not found in the scylla.yaml')

    def is_kubernetes(self) -> bool:
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

    @cached_property
    def is_enterprise(self):
        if self.distro.is_rhel_like:
            result = self.remoter.sudo("yum search scylla-enterprise 2>&1", ignore_status=True).stdout
            if 'One of the configured repositories failed (Extra Packages for Enterprise Linux 7 - x86_64)' in result:
                return "enterprise" in self.remoter.sudo("cat /etc/yum.repos.d/scylla.repo").stdout
            return "scylla-enterprise.x86_64" in result or "No matches found" not in result
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

    @property
    def ip_address(self):
        if IP_SSH_CONNECTIONS == "ipv6":
            return self.ipv6_ip_address
        elif Setup.INTRA_NODE_COMM_PUBLIC:
            return self.public_ip_address
        else:
            return self.private_ip_address

    @property
    def external_address(self):
        """
        the communication address for usage between the test and the nodes
        :return:
        """
        if IP_SSH_CONNECTIONS == "ipv6":
            return self.ipv6_ip_address
        elif IP_SSH_CONNECTIONS == 'public' or Setup.INTRA_NODE_COMM_PUBLIC:
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
            if self.termination_event.isSet():
                break
            try:
                self.wait_ssh_up(verbose=False)
            except Exception as ex:  # pylint: disable=broad-except
                self.log.warning("Unable to connect to '%s'. Probably the node was terminated or is still booting. "
                                 "Error details: '%s'", self.name, ex)
                continue
            next_check_delay = self.check_spot_termination() or SPOT_TERMINATION_CHECK_DELAY
            time.sleep(next_check_delay)

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
        log_transport = self.parent_cluster.params.get("logs_transport")
        self._journal_thread = get_system_logging_thread(
            logs_transport=log_transport,
            node=self,
            target_log_file=self.system_log,
        )
        if self._journal_thread:
            self.log.info("Use %s as logging daemon", type(self._journal_thread).__name__)
            self._journal_thread.start()
        else:
            if log_transport == 'rsyslog':
                self.log.info("Use no logging daemon since log transport is rsyslog")
            else:
                TestFrameworkEvent(
                    source=self.__class__.__name__,
                    source_method='start_journal_thread',
                    message="Got no logging daemon by unknown reason"
                ).publish_or_dump()

    @raise_event_on_failure
    def db_log_reader_thread(self):
        """
        Keep reporting new events from db log, every 30 seconds.
        """
        while not self.termination_event.isSet():
            self.termination_event.wait(15)
            try:
                self._read_system_log_and_publish_events(start_from_beginning=False,
                                                         exclude_from_logging=self._exclude_system_log_from_being_logged)
            except Exception:  # pylint: disable=broad-except
                self.log.exception("failed to read db log")
            except (SystemExit, KeyboardInterrupt) as ex:
                self.log.debug("db_log_reader_thread() stopped by %s", ex.__class__.__name__)

    def start_coredump_thread(self):
        self._coredump_thread = CoredumpExportSystemdThread(self, self._maximum_number_of_cores_to_publish)
        self._coredump_thread.start()

    def start_db_log_reader_thread(self):
        self._db_log_reader_thread = threading.Thread(
            target=self.db_log_reader_thread, name='LogReaderThread', daemon=True)
        self._db_log_reader_thread.start()

    def start_alert_manager_thread(self):
        self._alert_manager = PrometheusAlertManagerListener(self.external_address, stop_flag=self.termination_event)
        self._alert_manager.start()

    def silence_alert(self, alert_name, duration=None, start=None, end=None):
        return AlertSilencer(self._alert_manager, alert_name, duration, start, end)

    def __str__(self):
        return 'Node %s [%s | %s%s] (seed: %s)' % (self.name,
                                                   self.public_ip_address,
                                                   self.private_ip_address,
                                                   " | %s" % self.ipv6_ip_address if IP_SSH_CONNECTIONS == "ipv6" else "",
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
        wait.wait_for(func=uptime_changed, step=10, timeout=500, throw_exc=True)

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
        if self.is_spot:
            self.start_spot_monitoring_thread()
        if self.node_type == 'db':
            self.start_coredump_thread()
            self.start_db_log_reader_thread()
        elif self.node_type == 'loader':
            self.start_coredump_thread()
        elif self.node_type == 'monitor':
            # TODO: start alert manager thread here when start_task_threads will be run after node setup
            # self.start_alert_manager_thread()
            if Setup.BACKTRACE_DECODING:
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
        if self.termination_event.isSet():
            return
        self.log.info('Set termination_event')
        self.termination_event.set()
        if self._coredump_thread and self._coredump_thread.is_alive():
            self._coredump_thread.stop()
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
        LOGGER.info("%s destroyed", self)

    def wait_ssh_up(self, verbose=True, timeout=500):
        text = None
        if verbose:
            text = '%s: Waiting for SSH to be up' % self
        wait.wait_for(func=self.remoter.is_up, step=10, text=text, timeout=timeout, throw_exc=True)

    def is_port_used(self, port: int, service_name: str) -> bool:
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

    def get_cfstats(self, keyspace, tcpdump=False):
        def keyspace_available():
            self.run_nodetool("flush", ignore_status=True, timeout=60)
            res = self.run_nodetool(sub_cmd='cfstats', args=keyspace, ignore_status=True, timeout=60)
            return res.exit_status == 0
        tcpdump_id = uuid.uuid4()
        if tcpdump:
            self.log.info('START tcpdump thread uuid: %s', tcpdump_id)
            tcpdump_thread = threading.Thread(target=self._get_tcpdump_logs, name='TcpDumpUploadingThread',
                                              kwargs={'tcpdump_id': tcpdump_id}, daemon=True)
            tcpdump_thread.start()
        wait.wait_for(keyspace_available, timeout=120, step=60,
                      text='Waiting until keyspace {} is available'.format(keyspace))
        try:
            result = self.run_nodetool(sub_cmd='cfstats', args=keyspace, timeout=60)
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

    def _report_housekeeping_uuid(self, verbose=False):
        """
        report uuid of test db nodes to ScyllaDB
        """
        uuid_path = '/var/lib/scylla-housekeeping/housekeeping.uuid'
        mark_path = '/var/lib/scylla-housekeeping/housekeeping.uuid.marked'
        cmd = 'curl "https://i6a5h9l1kl.execute-api.us-east-1.amazonaws.com/prod/check_version?uu=%s&mark=scylla"'

        uuid_exists = self.remoter.run('test -e %s' % uuid_path, ignore_status=True, verbose=verbose).ok
        mark_exists = self.remoter.run('test -e %s' % mark_path, ignore_status=True, verbose=verbose).ok
        if uuid_exists and not mark_exists:
            result = self.remoter.run('cat %s' % uuid_path, verbose=verbose)
            self.remoter.run(cmd % result.stdout.strip(), ignore_status=True)
            if self.is_docker():
                self.remoter.run('touch %s' % mark_path, verbose=verbose)
            else:
                self.remoter.sudo('touch %s' % mark_path, verbose=verbose, user='scylla')

    def wait_db_up(self, verbose=True, timeout=3600):
        text = None
        if verbose:
            text = '%s: Waiting for DB services to be up' % self
        wait.wait_for(func=self.db_up, step=60, text=text, timeout=timeout, throw_exc=True)
        self.db_init_finished = True
        try:
            self._report_housekeeping_uuid()
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

    def wait_manager_server_up(self, verbose=True, timeout=300):
        text = None
        if verbose:
            text = '%s: Waiting for manager server to be up' % self
        try:
            wait.wait_for(func=self.is_manager_server_up, step=10, text=text, timeout=timeout, throw_exc=True)
        except RetryError:
            wait.wait_for(func=self.is_manager_server_up, port=self.OLD_MANAGER_PORT,
                          step=10, text=text, timeout=timeout, throw_exc=True)

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
                      text=text)

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
                      text=text)

    def mark_log(self):
        """
        Returns "a mark" to the current position of this node Cassandra log.
        This is for use with the from_mark parameter of watch_log_for_* methods,
        allowing to watch the log from the position when this method was called.
        """
        if not os.path.exists(self.system_log):
            return 0
        with open(self.system_log) as log_file:
            log_file.seek(0, os.SEEK_END)
            return log_file.tell()

    def follow_system_log(
            self,
            patterns: Optional[List[Union[str, re.Pattern, LogEvent]]] = None,
            start_from_beginning: bool = False
    ) -> List[str]:
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

    def _read_system_log_and_publish_events(self,
                                            start_from_beginning: bool = False,
                                            exclude_from_logging: List[str] = None) -> None:
        """Search for all known patterns listed in `sdcm.sct_events.database.SYSTEM_ERROR_EVENTS'."""

        # pylint: disable=too-many-branches,too-many-locals,too-many-statements

        backtraces = []
        index = 0

        if not os.path.exists(self.system_log):
            return

        if start_from_beginning:
            start_search_from_byte = 0
            last_line_no = 0
        else:
            start_search_from_byte = self.last_log_position
            last_line_no = self.last_line_no

        with open(self.system_log, 'r') as db_file:
            if start_search_from_byte:
                db_file.seek(start_search_from_byte)
            for index, line in enumerate(db_file, start=last_line_no):
                json_log = None
                if line[0] == '{':
                    try:
                        json_log = json.loads(line)
                    except Exception:
                        pass
                if not start_from_beginning and Setup.RSYSLOG_ADDRESS:
                    line = line.strip()
                    if not exclude_from_logging:
                        LOGGER.debug(line)
                    else:
                        exclude = False
                        for pattern in exclude_from_logging:
                            if pattern in line:
                                exclude = True
                                break
                        if not exclude:
                            LOGGER.debug(line)
                if json_log:
                    continue
                match = BACKTRACE_RE.search(line)
                one_line_backtrace = []
                if match and backtraces:
                    data = match.groupdict()
                    if data['other_bt']:
                        backtraces[-1]['backtrace'] += [data['other_bt'].strip()]
                    if data['scylla_bt']:
                        backtraces[-1]['backtrace'] += [data['scylla_bt'].strip()]
                elif "backtrace:" in line.lower() and "0x" in line:
                    # This part handles the backtrases are printed in one line.
                    # Example:
                    # [shard 2] seastar - Exceptional future ignored: exceptions::mutation_write_timeout_exception
                    # (Operation timed out for system.paxos - received only 0 responses from 1 CL=ONE.),
                    # backtrace:   0x3316f4d#012  0x2e2d177#012  0x189d397#012  0x2e76ea0#012  0x2e770af#012
                    # 0x2eaf065#012  0x2ebd68c#012  0x2e48d5d#012  /opt/scylladb/libreloc/libpthread.so.0+0x94e1#012
                    splitted_line = re.split("backtrace:", line, flags=re.IGNORECASE)
                    for trace_line in splitted_line[1].split():
                        if trace_line.startswith('0x') or 'scylladb/lib' in trace_line:
                            one_line_backtrace.append(trace_line)

                if index not in self._system_log_errors_index or start_from_beginning:
                    # for each line use all regexes to match, and if found send an event
                    for pattern, event in SYSTEM_ERROR_EVENTS_PATTERNS:
                        match = pattern.search(line)
                        if match:
                            self._system_log_errors_index.append(index)
                            cloned_event = event.clone().add_info(node=self, line_number=index, line=line)
                            backtraces.append(dict(event=cloned_event, backtrace=[]))
                            break  # Stop iterating patterns to avoid creating two events for one line of the log

                if one_line_backtrace and backtraces:
                    backtraces[-1]['backtrace'] = one_line_backtrace

            if not start_from_beginning:
                self.last_line_no = index if index else last_line_no
                self.last_log_position = db_file.tell() + 1

        traces_count = 0
        for backtrace in backtraces:
            backtrace['event'].raw_backtrace = "\n".join(backtrace['backtrace'])
            if backtrace['event'].type == 'BACKTRACE':
                traces_count += 1

        # A filter function to attach the backtrace to the correct error and not to the backtraces.
        # If the error is within 10 lines and the last isn't backtrace type, the backtrace would be
        # appended to the previous error.
        def filter_backtraces(backtrace):
            last_error = filter_backtraces.last_error
            try:
                if (last_error and
                        backtrace['event'].line_number <= filter_backtraces.last_error.line_number + 20
                        and not filter_backtraces.last_error.type == 'BACKTRACE' and backtrace['event'].type == 'BACKTRACE'):
                    last_error.raw_backtrace = "\n".join(backtrace['backtrace'])
                    backtrace['event'].dont_publish()
                    return False
                return True
            finally:
                filter_backtraces.last_error = backtrace['event']

        # support interlaced reactor stalled
        for _ in range(traces_count):
            filter_backtraces.last_error = None
            backtraces = list(filter(filter_backtraces, backtraces))

        for backtrace in backtraces:
            if Setup.BACKTRACE_DECODING and backtrace["event"].raw_backtrace:
                scylla_debug_info = self.get_scylla_debuginfo_file()
                self.log.debug("Debug info file %s", scylla_debug_info)
                Setup.DECODING_QUEUE.put({
                    "node": self,
                    "debug_file": scylla_debug_info,
                    "event": backtrace["event"],
                })
            else:
                backtrace["event"].publish()

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
                obj = Setup.DECODING_QUEUE.get(timeout=5)
                if obj is None:
                    Setup.DECODING_QUEUE.task_done()
                    break
                event = obj["event"]
                if not scylla_debug_file:
                    scylla_debug_file = self.copy_scylla_debug_info(obj["node"], obj["debug_file"])
                output = self.decode_raw_backtrace(scylla_debug_file, " ".join(event.raw_backtrace.split('\n')))
                event.backtrace = output.stdout
                Setup.DECODING_QUEUE.task_done()
            except queue.Empty:
                pass
            except Exception as details:  # pylint: disable=broad-except
                self.log.error("failed to decode backtrace %s", details)
            finally:
                if event:
                    event.publish()

            if self.termination_event.isSet() and Setup.DECODING_QUEUE.empty():
                break

    def copy_scylla_debug_info(self, node, debug_file):
        """Copy scylla debug file from db-node to monitor-node

        Copy via builder
        :param node: db node
        :type node: BaseNode
        :param scylla_debug_file: path to scylla_debug_file on db-node
        :type scylla_debug_file: str
        :returns: path on monitor node
        :rtype: {str}
        """
        base_scylla_debug_file = os.path.basename(debug_file)
        transit_scylla_debug_file = os.path.join(node.parent_cluster.logdir,
                                                 base_scylla_debug_file)
        final_scylla_debug_file = os.path.join("/tmp", base_scylla_debug_file)

        if not os.path.exists(transit_scylla_debug_file):
            node.remoter.receive_files(debug_file, transit_scylla_debug_file)
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
            output = self.remoter.run(f"readelf -n {scylla_executable}", ignore_status=True).stdout
            if match := BUILD_ID_RE.search(output):
                return match.group("build_id")
        return None

    def get_scylla_debuginfo_file(self):
        """
        Lookup the scylla debug information, in various places it can be.

        :return the path to the scylla debug information
        :rtype str
        """
        # first try default location
        scylla_debug_info = '/usr/lib/debug/bin/scylla.debug'
        results = self.remoter.run('[[ -f {} ]]'.format(scylla_debug_info), ignore_status=True)
        if results.exit_status == 0:
            return scylla_debug_info

        # then try the relocatable location
        results = self.remoter.run('ls /usr/lib/debug/opt/scylladb/libexec/scylla*.debug', ignore_status=True)
        if results.stdout.strip():
            return results.stdout.strip()

        # then look it up base on the build id
        if build_id := self.get_scylla_build_id():
            scylla_debug_info = "/usr/lib/debug/.build-id/{0}/{1}.debug".format(build_id[:2], build_id[2:])
            results = self.remoter.run('[[ -f {} ]]'.format(scylla_debug_info), ignore_status=True)
            if results.exit_status == 0:
                return scylla_debug_info

        raise Exception("Couldn't find scylla debug information")

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
        file_name = os.path.split(path)[1].split(".", maxsplit=1)[0].title()
        self.log.debug("Update {} YAML configuration file ({})".format(file_name, path))
        return remote_file(remoter=self.remoter,
                           remote_path=path,
                           serializer=yaml.safe_dump,
                           deserializer=yaml.safe_load,
                           sudo=True)

    def remote_scylla_yaml(self, path=SCYLLA_YAML_PATH):
        path = self.add_install_prefix(path)
        return self._remote_yaml(path=path)

    def remote_manager_yaml(self, path=SCYLLA_MANAGER_YAML_PATH):
        return self._remote_yaml(path=path)

    @staticmethod
    def get_ldap_config():
        ldap_server_ip = '127.0.0.1' if IP_SSH_CONNECTIONS == 'public' or Setup.MULTI_REGION else Setup.LDAP_ADDRESS[0]
        ldap_port = LDAP_SSH_TUNNEL_LOCAL_PORT if IP_SSH_CONNECTIONS == 'public' or Setup.MULTI_REGION else \
            Setup.LDAP_ADDRESS[1]
        return {'role_manager': 'com.scylladb.auth.LDAPRoleManager',
                'ldap_url_template': f'ldap://{ldap_server_ip}:{ldap_port}/'
                                     f'{LDAP_BASE_OBJECT}?cn?sub?(uniqueMember='
                                     f'uid={{USER}},ou=Person,{LDAP_BASE_OBJECT})',
                'ldap_attr_role': 'cn',
                'ldap_bind_dn': f'cn=admin,{LDAP_BASE_OBJECT}',
                'ldap_bind_passwd': LDAP_PASSWORD}

    def create_ldap_users_on_scylla(self):
        if not self.parent_cluster.ldap_configured:
            self.run_cqlsh(f'CREATE ROLE \'{LDAP_ROLE}\' WITH SUPERUSER=true')
            for user in LDAP_USERS:
                self.run_cqlsh(f'CREATE ROLE \'{user}\' WITH login=true AND password=\'{LDAP_PASSWORD}\'')
            self.parent_cluster.ldap_configured = True

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    def config_setup(self,
                     seed_address=None,
                     cluster_name=None,
                     enable_exp=True,
                     endpoint_snitch=None,
                     yaml_file=SCYLLA_YAML_PATH,
                     broadcast=None,
                     authenticator=None,
                     server_encrypt=None,
                     client_encrypt=None,
                     append_scylla_yaml=None,
                     append_scylla_args=None,
                     debug_install=False,
                     hinted_handoff="enabled",
                     murmur3_partitioner_ignore_msb_bits=None,
                     authorizer=None,
                     alternator_port=None,
                     listen_on_all_interfaces=False,
                     ip_ssh_connections=None,
                     alternator_enforce_authorization=False,
                     internode_compression=None,
                     internode_encryption=None,
                     ldap=False):
        with self.remote_scylla_yaml(yaml_file) as scylla_yml:
            if seed_address:
                # Set seeds
                scylla_yml['seed_provider'] = [
                    dict(class_name='org.apache.cassandra.locator.SimpleSeedProvider',
                         parameters=[dict(seeds=seed_address)])]

                # NOTICE: the following configuration always have to use private_ip_address for multi-region to work
                # Set listen_address
                scylla_yml['listen_address'] = self.private_ip_address
                # Set rpc_address
                scylla_yml['rpc_address'] = self.private_ip_address

            if listen_on_all_interfaces:
                # Set listen_address
                scylla_yml['listen_address'] = "0.0.0.0"
                # Set rpc_address
                scylla_yml['rpc_address'] = "0.0.0.0"

            if broadcast:
                # Set broadcast_address
                scylla_yml['broadcast_address'] = broadcast

                # Set broadcast_rpc_address
                scylla_yml['broadcast_rpc_address'] = broadcast

            if cluster_name:
                scylla_yml['cluster_name'] = cluster_name

            # disable hinted handoff (it is enabled by default in Scylla). Expected values: "enabled"/"disabled"
            if hinted_handoff == 'disabled':
                scylla_yml['hinted_handoff_enabled'] = False

            if ip_ssh_connections == 'ipv6':
                self.log.debug('Enable IPv6 DNS lookup')
                scylla_yml['enable_ipv6_dns_lookup'] = True

                scylla_yml['prometheus_address'] = self.ip_address
                scylla_yml['broadcast_rpc_address'] = self.ip_address
                scylla_yml['listen_address'] = self.ip_address
                scylla_yml['rpc_address'] = self.ip_address

            if murmur3_partitioner_ignore_msb_bits:
                self.log.debug('Change murmur3_partitioner_ignore_msb_bits to {}'.format(
                    murmur3_partitioner_ignore_msb_bits))
                scylla_yml['murmur3_partitioner_ignore_msb_bits'] = int(murmur3_partitioner_ignore_msb_bits)

            if enable_exp:
                scylla_yml['experimental'] = True

            if endpoint_snitch:
                scylla_yml['endpoint_snitch'] = endpoint_snitch

            if not client_encrypt:
                scylla_yml['client_encryption_options'] = dict(enabled=False)

            if self.enable_auto_bootstrap:
                scylla_yml['auto_bootstrap'] = True
            else:
                if 'auto_bootstrap' in scylla_yml:
                    scylla_yml['auto_bootstrap'] = False

            if authenticator in ['AllowAllAuthenticator', 'PasswordAuthenticator']:
                scylla_yml['authenticator'] = authenticator

            if authorizer in ['AllowAllAuthorizer', 'CassandraAuthorizer']:
                scylla_yml['authorizer'] = authorizer

            if server_encrypt or client_encrypt:
                self.config_client_encrypt()
            if server_encrypt:
                scylla_yml['server_encryption_options'] = dict(internode_encryption=internode_encryption,
                                                               certificate='/etc/scylla/ssl_conf/db.crt',
                                                               keyfile='/etc/scylla/ssl_conf/db.key',
                                                               truststore='/etc/scylla/ssl_conf/cadb.pem')

            if client_encrypt:
                scylla_yml['client_encryption_options'] = dict(enabled=True,
                                                               certificate='/etc/scylla/ssl_conf/client/test.crt',
                                                               keyfile='/etc/scylla/ssl_conf/client/test.key',
                                                               truststore='/etc/scylla/ssl_conf/client/catest.pem')

            if self.replacement_node_ip:
                scylla_yml['replace_address_first_boot'] = self.replacement_node_ip
            else:
                if 'replace_address_first_boot' in scylla_yml:
                    del scylla_yml['replace_address_first_boot']

            if alternator_port:
                scylla_yml['alternator_port'] = alternator_port
                scylla_yml['alternator_write_isolation'] = alternator.enums.WriteIsolation.ALWAYS_USE_LWT.value

            if alternator_enforce_authorization:
                scylla_yml['alternator_enforce_authorization'] = True
            else:
                scylla_yml['alternator_enforce_authorization'] = False

            if internode_compression:
                scylla_yml['internode_compression'] = internode_compression

            if ldap:
                scylla_yml.update(self.get_ldap_config())

            if append_scylla_yaml:
                scylla_yml.update(yaml.safe_load(append_scylla_yaml))

        if append_scylla_yaml:
            if any(substr in append_scylla_yaml for substr in ("system_key_directory",
                                                               "system_info_encryption",
                                                               "kmip_hosts:", )):
                self.remoter.send_files(src="./data_dir/encrypt_conf", dst="/tmp/")
                self.remoter.sudo(shell_script_cmd("""\
                    rm -rf /etc/encrypt_conf
                    mv -f /tmp/encrypt_conf /etc
                    mkdir -p /etc/scylla/encrypt_conf /etc/encrypt_conf/system_key_dir
                    chown -R scylla:scylla /etc/scylla /etc/encrypt_conf
                """))
                self.remoter.sudo("md5sum /etc/encrypt_conf/*.pem", ignore_status=True)

        if append_scylla_args:
            scylla_help = self.remoter.run(
                f"{self.add_install_prefix('/usr/bin/scylla')} --help", ignore_status=True).stdout
            scylla_arg_parser = ScyllaArgParser.from_scylla_help(scylla_help)
            append_scylla_args = scylla_arg_parser.filter_args(append_scylla_args)

        if append_scylla_args:
            self.log.debug("Append following args to scylla: `%s'", append_scylla_args)
            scylla_server_config = f"/etc/{'sysconfig' if self.distro.is_rhel_like else 'default'}/scylla-server"
            self.remoter.sudo(
                f"sed -i '/{append_scylla_args}/! s/SCYLLA_ARGS=\"/&{append_scylla_args} /' {scylla_server_config}")

        if debug_install and self.distro.is_rhel_like:
            self.remoter.sudo("yum install -y scylla-gdb", verbose=True, ignore_status=True)

        if self.init_system == "systemd":
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

    def config_client_encrypt(self):
        self.remoter.send_files(src='./data_dir/ssl_conf', dst='/tmp/')  # pylint: disable=not-callable
        setup_script = dedent("""
            mkdir -p ~/.cassandra/
            cp /tmp/ssl_conf/client/cqlshrc ~/.cassandra/
            sudo mkdir -p /etc/scylla/
            sudo rm -rf /etc/scylla/ssl_conf/
            sudo mv -f /tmp/ssl_conf/ /etc/scylla/
        """)
        self.remoter.run('bash -cxe "%s"' % setup_script)

    @retrying(n=3, sleep_time=10, allowed_exceptions=(AssertionError,), message="Retrying on getting scylla repo")
    def download_scylla_repo(self, scylla_repo):
        if not scylla_repo:
            self.log.error("Scylla YUM repo file url is not provided, it should be defined in configuration YAML!!!")
            return
        if self.is_rhel_like():
            repo_path = '/etc/yum.repos.d/scylla.repo'
            self.remoter.run('sudo curl -o %s -L %s' % (repo_path, scylla_repo))
            self.remoter.run('sudo chown root:root %s' % repo_path)
            self.remoter.run('sudo chmod 644 %s' % repo_path)
            result = self.remoter.run('cat %s' % repo_path, verbose=True)
            verify_scylla_repo_file(result.stdout, is_rhel_like=True)
        else:
            repo_path = '/etc/apt/sources.list.d/scylla.list'
            self.remoter.run('sudo curl -o %s -L %s' % (repo_path, scylla_repo))
            result = self.remoter.run('cat %s' % repo_path, verbose=True)
            verify_scylla_repo_file(result.stdout, is_rhel_like=False)
        self.update_repo_cache()

    def download_scylla_manager_repo(self, scylla_repo):
        if self.is_rhel_like():
            repo_path = '/etc/yum.repos.d/scylla-manager.repo'
        else:
            repo_path = '/etc/apt/sources.list.d/scylla-manager.list'
        self.remoter.run('sudo curl -o %s -L %s' % (repo_path, scylla_repo))
        if not self.is_rhel_like():
            self.remoter.run(cmd="sudo apt-get update", ignore_status=True)

    def install_manager_agent(self, package_path=None):
        auth_token = Setup.test_id()
        manager_prometheus_port = self.parent_cluster.params.get("manager_prometheus_port")
        if package_path:
            package_name = '{}scylla-manager-agent*'.format(package_path)
        else:
            self.download_scylla_manager_repo(
                self.parent_cluster.params.get("scylla_mgmt_agent_repo") or
                self.parent_cluster.params.get("scylla_mgmt_repo"))
            package_name = 'scylla-manager-agent'
        package_mgr = "yum" if self.distro.is_rhel_like else "apt-get"
        install_and_config_agent_command = dedent(r"""
            {} install -y {}
            sed -i 's/# auth_token:.*$/auth_token: {}/' /etc/scylla-manager-agent/scylla-manager-agent.yaml
            scyllamgr_ssl_cert_gen
            sed -i 's/#tls_cert_file/tls_cert_file/' /etc/scylla-manager-agent/scylla-manager-agent.yaml
            sed -i 's/#tls_key_file/tls_key_file/' /etc/scylla-manager-agent/scylla-manager-agent.yaml
            sed -i 's/#prometheus: .*/prometheus: :{}/' /etc/scylla-manager-agent/scylla-manager-agent.yaml
            systemctl restart scylla-manager-agent
            systemctl enable scylla-manager-agent
        """.format(package_mgr, package_name, auth_token, manager_prometheus_port))
        self.remoter.run('sudo bash -cxe "%s"' % install_and_config_agent_command)
        version = self.remoter.run('scylla-manager-agent --version').stdout
        self.log.info(f'node {self.name} has scylla-manager-agent version {version}')

    def upgrade_manager_agent(self, scylla_mgmt_repo, start_agent_after_upgrade=True):
        self.download_scylla_manager_repo(scylla_mgmt_repo)
        if self.is_rhel_like():
            self.remoter.run('sudo yum update scylla-manager-agent -y')
        else:
            self.remoter.run(cmd="sudo apt-get update", ignore_status=True)
            self.remoter.run('sudo apt-get install -y scylla-manager-agent')
        self.remoter.run("sudo scyllamgr_agent_setup -y")
        if start_agent_after_upgrade:
            if self.is_docker():
                self.remoter.run('sudo supervisorctl start scylla-manager-agent')
            else:
                self.remoter.run("sudo systemctl start scylla-manager-agent")

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
            self.remoter.run('sudo yum remove -y scylla\\*')
        else:
            self.remoter.run('sudo rm -f /etc/apt/sources.list.d/scylla.list')
            self.remoter.run('sudo apt-get remove -y scylla\\*', ignore_status=True)
        self.update_repo_cache()
        self.clean_scylla_data()

    def update_repo_cache(self):
        try:
            if self.is_rhel_like():
                # try to avoid ERROR 404 of yum, reference https://wiki.centos.org/yum-errors
                self.remoter.run('sudo yum clean all')
                self.remoter.run('sudo rm -rf /var/cache/yum/')
                self.remoter.run('sudo yum makecache', retry=3)
            else:
                self.remoter.run('sudo apt-get clean all')
                self.remoter.run('sudo rm -rf /var/cache/apt/')
                self.remoter.run('sudo apt-get update', retry=3)
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
        else:
            self.remoter.run(
                'sudo DEBIAN_FRONTEND=noninteractive apt-get --force-yes -o Dpkg::Options::="--force-confold" -o Dpkg::Options::="--force-confdef" upgrade -y', retry=3)
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
            if self.distro.is_centos8 or self.distro.is_rhel8 or self.distro.is_oel8:
                self.install_epel()
            self.remoter.run('sudo yum install -y rsync tcpdump screen')
            self.download_scylla_repo(scylla_repo)
            # hack cause of broken caused by EPEL
            self.remoter.run('sudo yum install -y python36-PyYAML', ignore_status=True)
            self.remoter.run('sudo yum install -y {}'.format(self.scylla_pkg()))
            self.remoter.run('sudo yum install -y scylla-gdb', ignore_status=True)
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
                    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6B2BFD3660EF3F5B
                    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 17723034C56D4B19
                    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5e08fbd8b5d6ec9c
                    # add-apt-repository -y ppa:scylladb/ppa
                    # apt-get update
                """)
                self.remoter.run('sudo bash -cxe "%s"' % install_prereqs)
            elif self.is_debian8():
                self.remoter.run("sudo sed -i -e 's/jessie-updates/stable-updates/g' /etc/apt/sources.list")
                self.remoter.run(
                    'echo "deb http://archive.debian.org/debian jessie-backports main" |sudo tee /etc/apt/sources.list.d/backports.list')
                self.remoter.run(
                    r"sudo sed -i -e 's/:\/\/.*\/debian jessie-backports /:\/\/archive.debian.org\/debian jessie-backports /g' /etc/apt/sources.list.d/*.list")
                self.remoter.run(
                    "echo 'Acquire::Check-Valid-Until \"false\";' |sudo tee /etc/apt/apt.conf.d/99jessie-backports")
                self.remoter.run('sudo apt-get update')
                self.remoter.run('sudo apt-get install gnupg-curl -y')
                self.remoter.run(
                    'sudo apt-key adv --fetch-keys https://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-jessie/Debian_8.0/Release.key')
                self.remoter.run('sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 17723034C56D4B19')
                self.remoter.run(
                    'echo "deb http://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-jessie/Debian_8.0/ /" |sudo tee /etc/apt/sources.list.d/scylla-3rdparty.list')
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
                    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 17723034C56D4B19
                    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5E08FBD8B5D6EC9C
                    echo 'deb http://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-stretch/Debian_9.0/ /' > /etc/apt/sources.list.d/scylla-3rdparty.list
                """)
                self.remoter.run('sudo bash -cxe "%s"' % install_debian_9_prereqs)
            elif self.distro.is_debian10:
                install_debian_10_prereqs = dedent("""
                    export DEBIAN_FRONTEND=noninteractive
                    apt-get update
                    apt-get install apt-transport-https -y
                    apt-get install gnupg1-curl dirmngr -y
                    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 17723034C56D4B19
                    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5E08FBD8B5D6EC9C
                    curl https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | sudo apt-key add -
                    apt-get install software-properties-common -y
                    add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/
                """)
                self.remoter.run('sudo bash -cxe "%s"' % install_debian_10_prereqs)

            self.remoter.run(
                'sudo DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Options::="--force-confold" -o Dpkg::Options::="--force-confdef" upgrade -y')
            self.remoter.run('sudo apt-get install -y rsync tcpdump screen')
            self.download_scylla_repo(scylla_repo)
            self.remoter.run('sudo apt-get update')
            self.remoter.run(
                'sudo apt-get install -y -o Dpkg::Options::="--force-overwrite" -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" '
                '--allow-unauthenticated {}'.format(self.scylla_pkg()))

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

        # Offline install does't provide openjdk-8, it has to be installed in advance
        # https://github.com/scylladb/scylla-jmx/issues/127
        if self.is_rhel_like():
            self.remoter.run(f'sudo yum install -y java-1.8.0-openjdk {additional_pkgs}')
        elif self.distro.is_debian10:
            self.remoter.run(f'sudo apt-get install -y openjdk-11-jre-headless {additional_pkgs}')
        else:
            self.remoter.run(f'sudo apt-get install -y openjdk-8-jre-headless {additional_pkgs}')
            self.remoter.run('sudo update-java-alternatives --jre-headless -s java-1.8.0-openjdk-amd64')

        if nonroot:
            # Make sure env variable (XDG_RUNTIME_DIR) is set, which is necessary for systemd user
            if not 'XDG_RUNTIME_DIR=' in self.remoter.run('env').stdout:
                # Reload the env variables by ssh reconnect
                self.remoter.run('env', verbose=True, change_context=True)
                assert 'XDG_RUNTIME_DIR' in self.remoter.run('env', verbose=True).stdout
            install_cmds = dedent(f"""
                tar xvfz ./unified_package.tar.gz
                ./install.sh --nonroot
                sudo rm -f /tmp/scylla.yaml
            """)
            # Known issue: https://github.com/scylladb/scylla/issues/7071
            self.remoter.run('bash -cxe "%s"' % install_cmds)
        else:
            install_cmds = dedent(f"""
                tar xvfz ./unified_package.tar.gz
                ./install.sh --housekeeping
                rm -f /tmp/scylla.yaml
            """)
            self.remoter.run('sudo bash -cxe "%s"' % install_cmds)

    def install_scylla_debuginfo(self):
        self.log.info("Installing Scylla debug info...")
        if not self.scylla_version:
            self.get_scylla_version()
        if self.is_rhel_like():
            self.remoter.run(
                r'sudo yum install -y {0}-debuginfo-{1}\*'.format(self.scylla_pkg(), self.scylla_version), ignore_status=True)
        else:
            self.remoter.run(r'sudo apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" --allow-unauthenticated {0}-server-dbg={1}\*'
                             .format(self.scylla_pkg(), self.scylla_version), ignore_status=True)

    def is_scylla_installed(self):
        if self.distro.is_rhel_like:
            result = self.remoter.run(f'rpm -q {self.scylla_pkg()}', verbose=False, ignore_status=True)
        elif self.distro.is_ubuntu or self.distro.is_debian:
            result = self.remoter.run(f'dpkg-query --show {self.scylla_pkg()}', verbose=False, ignore_status=True)
        else:
            raise ValueError(f"Unsupported Linux distribution: {self.distro}")
        return result.exit_status == 0

    def get_scylla_version(self) -> str:
        self.scylla_version = self.scylla_version_detailed = ""
        scylla_path = self.add_install_prefix("/usr/bin/scylla")
        result = self.remoter.run(f"{scylla_path} --version", ignore_status=True)
        if result.ok:
            self.scylla_version_detailed = result.stdout.strip()
            if build_id := self.get_scylla_build_id():
                self.scylla_version_detailed += f" with build-id {build_id}"
                self.log.info(f"Found ScyllaDB version with details: {self.scylla_version_detailed}")
        else:
            if self.distro.is_rhel_like:
                cmd = f"rpm --query --queryformat '%{{VERSION}}' {self.scylla_pkg()}"
            else:
                cmd = f"dpkg-query --show --showformat '${{Version}}' {self.scylla_pkg()}"
            result = self.remoter.run(cmd, ignore_status=True)

        if match := result.ok and SCYLLA_VERSION_RE.match(result.stdout):
            self.scylla_version = match.group().replace("~", ".")
            self.log.info("Found ScyllaDB version: %s", self.scylla_version)
        else:
            self.log.debug("Unable to get or parse ScyllaDB version using `%s':\n%s\n%s",
                           cmd, result.stdout, result.stderr)

        if not self.scylla_version:
            self.log.warning(
                "All attempts to get ScyllaDB version failed. Looks like there is no ScyllaDB installed.")

        if not self.scylla_version_detailed:
            self.scylla_version_detailed = self.scylla_version

        return self.scylla_version

    @log_run_info("Detecting disks")
    def detect_disks(self, nvme=True):
        """
        Detect local disks
        :param nvme: NVMe(True) or SCSI(False) disk
        :return: list of disk names
        """
        patt = (r'nvme*n*', r'nvme\d+n\d+') if nvme else (r'sd[b-z]', r'sd\w+')
        result = self.remoter.run('ls /dev/{}'.format(patt[0]))
        disks = re.findall(r'/dev/{}'.format(patt[1]), result.stdout)
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
        Setup scylla
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

    def upgrade_mgmt(self, scylla_mgmt_repo, start_manager_after_upgrade=True):
        self.download_scylla_manager_repo(scylla_mgmt_repo)
        self.log.debug('Upgrade scylla-manager via repo: {}'.format(scylla_mgmt_repo))
        if self.is_rhel_like():
            self.remoter.run('sudo yum update scylla-manager -y')
        else:
            self.remoter.run(cmd="sudo apt-get update", ignore_status=True)
            # Upgrade should update packages of:
            # 1) scylla-manager
            # 2) scylla-manager-client
            # 3) scylla-manager-server
            self.remoter.run('sudo apt-get dist-upgrade scylla-manager-server scylla-manager-client -y -o '
                             'Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" --force-yes'
                             ' --allow-unauthenticated')
        time.sleep(3)
        if start_manager_after_upgrade:
            if self.is_docker():
                self.remoter.run('sudo supervisorctl start scylla-manager')
            else:
                self.remoter.run('sudo systemctl restart scylla-manager.service')
            time.sleep(5)

    # pylint: disable=too-many-branches,too-many-statements
    def install_mgmt(self, scylla_mgmt_repo, auth_token, package_url=None):
        self.log.debug('Install scylla-manager')
        rsa_id_dst = '/tmp/scylla-test'
        rsa_id_dst_pub = '/tmp/scylla-test-pub'
        mgmt_user = 'scylla-manager'
        if not (self.is_rhel_like() or self.is_debian() or self.is_ubuntu()):
            raise ValueError('Unsupported Distribution type: {}'.format(str(self.distro)))
        if self.is_rhel_like():
            self.install_epel()
            self.remoter.run('sudo yum install python36-PyYAML -y', retry=3)
        else:
            self.remoter.run('sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6B2BFD3660EF3F5B', retry=3)
            self.remoter.run('sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 17723034C56D4B19', retry=3)
            self.remoter.run('sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5E08FBD8B5D6EC9C', retry=3)

        self.log.debug("Copying TLS files from data_dir to node")
        self.remoter.send_files(src='./data_dir/ssl_conf', dst='/tmp/')  # pylint: disable=not-callable

        if package_url:
            package_names = '{0}scylla-manager-server* {0}scylla-manager-client*'.format(package_url)
        else:
            self.download_scylla_manager_repo(scylla_mgmt_repo)
            package_names = 'scylla-manager'
        if self.is_docker():
            self.remoter.run('sudo yum remove -y scylla scylla-jmx scylla-tools scylla-tools-core'
                             ' scylla-server scylla-conf')

        if self.is_rhel_like():
            self.remoter.run('sudo yum install -y {}'.format(package_names))
        else:
            self.remoter.run(cmd="sudo apt-get update", ignore_status=True)
            self.remoter.run(
                'sudo apt-get install -y {}{}'.format(package_names, ' --force-yes' if not self.is_debian9() else ''))

        if self.is_docker():
            try:
                self.remoter.run('echo no| sudo scyllamgr_setup')
            except Exception as ex:  # pylint: disable=broad-except
                self.log.warning(ex)
        else:
            self.remoter.run('echo yes| sudo scyllamgr_setup')
        self.remoter.send_files(src=self.ssh_login_info['key_file'], dst=rsa_id_dst)  # pylint: disable=not-callable
        ssh_config_script = dedent("""
                chmod 0400 {rsa_id_dst}
                chown {mgmt_user}:{mgmt_user} {rsa_id_dst}
                ssh-keygen -y -f {rsa_id_dst} > {rsa_id_dst_pub}
        """.format(mgmt_user=mgmt_user, rsa_id_dst=rsa_id_dst, rsa_id_dst_pub=rsa_id_dst_pub))  # generate ssh public key from private key.
        self.remoter.run('sudo bash -cxe "%s"' % ssh_config_script)

        if self.is_docker():
            self.remoter.run('sudo supervisorctl restart scylla-manager')
            res = self.remoter.run('sudo supervisorctl status scylla-manager')
        else:
            self.remoter.run('sudo systemctl restart scylla-manager.service')
            res = self.remoter.run('sudo systemctl status scylla-manager.service')

        manager_prometheus_port = self.parent_cluster.params.get("manager_prometheus_port")
        if self.is_rhel_like():  # TODO: Add debian and ubuntu support
            configuring_manager_command = dedent("""
            scyllamgr_ssl_cert_gen
            sed -i 's/#tls_cert_file/tls_cert_file/' /etc/scylla-manager/scylla-manager.yaml
            sed -i 's/#tls_key_file/tls_key_file/' /etc/scylla-manager/scylla-manager.yaml
            sed -i 's/#prometheus: .*/prometheus: :{}/' /etc/scylla-manager/scylla-manager.yaml
            systemctl restart scylla-manager
            """.format(manager_prometheus_port))  # pylint: disable=too-many-format-args
            self.remoter.run('sudo bash -cxe "%s"' % configuring_manager_command)

        if not res or "Active: failed" in res.stdout:
            raise ScyllaManagerError("Scylla-Manager is not properly installed or not running: {}".format(res))

        self.start_scylla_manager_log_capture()

    def retrieve_scylla_manager_log(self):
        mgmt_log_name = os.path.join(self.logdir, 'scylla_manager.log')
        cmd = "sudo journalctl -u scylla-manager -f"
        self.remoter.run(cmd, ignore_status=True, verbose=True, log_file=mgmt_log_name)

    def scylla_manager_log_thread(self):
        while not self.termination_event.isSet():
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
            '/etc/scylla.d/io.conf': f'{INSTALL_DIR}/etc/scylla.d/io.conf',
            '/usr/bin/scylla': f'{INSTALL_DIR}/bin/scylla',
            '/usr/bin/nodetool': f'{INSTALL_DIR}/share/cassandra/bin/nodetool',
            '/usr/bin/cqlsh': f'{INSTALL_DIR}/share/cassandra/bin/cqlsh',
            '/usr/bin/cassandra-stress': f'{INSTALL_DIR}/share/cassandra/bin/cassandra-stress',
        }
        return checklist.get(abs_path, INSTALL_DIR + abs_path)

    def start_scylla_server(self, verify_up=True, verify_down=False, timeout=500, verify_up_timeout=300):
        if verify_down:
            self.wait_db_down(timeout=timeout)
        if self.is_ubuntu14():
            self.remoter.run('sudo service scylla-server start', timeout=timeout)
        else:
            self.remoter.run(f'{self.systemctl} start scylla-server.service', timeout=timeout)
        if verify_up:
            self.wait_db_up(timeout=verify_up_timeout)

    def start_scylla_jmx(self, verify_up=True, verify_down=False, timeout=300, verify_up_timeout=300):
        if verify_down:
            self.wait_jmx_down(timeout=timeout)
        if self.is_ubuntu14():
            self.remoter.run('sudo service scylla-jmx start', timeout=timeout)
        else:
            self.remoter.run(f'{self.systemctl} start scylla-jmx.service', timeout=timeout)
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
        if self.is_ubuntu14():
            self.remoter.run('sudo service scylla-server stop', timeout=timeout, ignore_status=ignore_status)
        else:
            self.remoter.run(f'{self.systemctl} stop scylla-server.service',
                             timeout=timeout, ignore_status=ignore_status)
        if verify_down:
            self.wait_db_down(timeout=timeout)

    def stop_scylla_jmx(self, verify_up=False, verify_down=True, timeout=300):
        if verify_up:
            self.wait_jmx_up(timeout=timeout)
        if not self.is_ubuntu14():
            self.remoter.run(f'{self.systemctl} stop scylla-jmx.service', timeout=timeout)
        else:
            self.remoter.sudo('service scylla-jmx stop', timeout=timeout)
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
        if self.distro.is_ubuntu14:
            self.remoter.run("sudo service scylla-server restart",
                             timeout=timeout, ignore_status=ignore_status)
        else:
            self.remoter.run(f"{self.systemctl} restart scylla-server.service",
                             timeout=timeout, ignore_status=ignore_status)
        if verify_up_after:
            self.wait_db_up(timeout=timeout)

    def restart_scylla_jmx(self, verify_up_before=False, verify_up_after=True, timeout=300):
        if verify_up_before:
            self.wait_jmx_up(timeout=timeout)
        if self.distro.is_ubuntu14:
            self.remoter.run("sudo service scylla-jmx restart", timeout=timeout)
        else:
            self.remoter.run(f"{self.systemctl} restart scylla-jmx.service", timeout=timeout)
        if verify_up_after:
            self.wait_jmx_up(timeout=timeout)

    @log_run_info
    def restart_scylla(self, verify_up_before=False, verify_up_after=True, timeout=500):
        self.restart_scylla_server(verify_up_before=verify_up_before, verify_up_after=verify_up_after, timeout=timeout)
        if verify_up_after:
            self.wait_jmx_up(timeout=timeout)

    def enable_client_encrypt(self):
        SCYLLA_YAML_PATH_TMP = "/tmp/scylla.yaml"
        self.remoter.run("sudo cat {} | grep -v '<client_encrypt>' > {}".format(SCYLLA_YAML_PATH, SCYLLA_YAML_PATH_TMP))
        self.remoter.run("sudo mv -f {} {}".format(SCYLLA_YAML_PATH_TMP, SCYLLA_YAML_PATH))
        self.parent_cluster.node_config_setup(node=self, client_encrypt=True)
        self.stop_scylla()
        self.start_scylla()

    def disable_client_encrypt(self):
        self.parent_cluster.node_config_setup(node=self, client_encrypt=False)
        self.stop_scylla()
        self.start_scylla()

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

    def generate_coredump_file(self, restart_scylla=True):
        self.log.info('Generate scylla core')
        self.remoter.run("sudo pkill -f --signal 3 /usr/bin/scylla")
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

    def _resharding_status(self, status):
        """
        Check is there's Reshard listed in the "nodetool compactionstats" output
        status : expected values: "start" or "finish"
        """
        patt = re.compile('RESHARD|RESHAP')
        result = self.run_nodetool("compactionstats")
        found = patt.search(result.stdout)
        # wait_for_status=='finish': If 'RESHARD' is not found in the compactionstats output, return True -
        # means resharding was finished
        # wait_for_status=='start: If 'RESHARD' is found in the compactionstats output, return True -
        # means resharding was started

        return bool(found) if status == 'start' else not bool(found)

    # Default value of murmur3_partitioner_ignore_msb_bits parameter is 12
    def _restart_node_with_resharding(self, murmur3_partitioner_ignore_msb_bits: int = 12):
        self.stop_scylla()
        # Change murmur3_partitioner_ignore_msb_bits parameter to cause resharding.
        with self.remote_scylla_yaml() as scylla_yml:
            scylla_yml["murmur3_partitioner_ignore_msb_bits"] = murmur3_partitioner_ignore_msb_bits
        search_reshard = self.follow_system_log(patterns=['Reshard', 'Reshap'])
        self.start_scylla(timeout=7200)
        return search_reshard

    def restart_node_with_resharding(self, murmur3_partitioner_ignore_msb_bits: int = 12) -> None:
        search_reshard = self._restart_node_with_resharding(murmur3_partitioner_ignore_msb_bits)

        resharding_started = wait.wait_for(func=self._resharding_status, step=5, timeout=180,
                                           text="Wait for re-sharding to be started", status='start')
        if not resharding_started:
            resharding_started = list(search_reshard)
            if resharding_started:
                # If re-sharding was found running in the log but didn't reported by "nodetool compactionstats",
                # it means that re-sharding was started and finished already
                self.log.debug(f'Resharding has been finished successfully '
                               f'(murmur3_partitioner_ignore_msb_bits={murmur3_partitioner_ignore_msb_bits})')
            else:
                raise Exception(f'Resharding has not been started '
                                f'(murmur3_partitioner_ignore_msb_bits={murmur3_partitioner_ignore_msb_bits}) '
                                'Check the log for the details')
            return
        else:
            # Decrease nodetool compactionstats calls from 5sec to 1min to avoid the noise
            resharding_finished = wait.wait_for(func=self._resharding_status, step=60,
                                                text="Wait for re-sharding to be finished", status='finish')

            if not resharding_finished:
                raise Exception('Resharding was not finished! '
                                f'(murmur3_partitioner_ignore_msb_bits={murmur3_partitioner_ignore_msb_bits}) '
                                'Check the log for the details')
            else:
                self.log.debug('Resharding has been finished successfully '
                               f'(murmur3_partitioner_ignore_msb_bits={murmur3_partitioner_ignore_msb_bits})')

    def _gen_nodetool_cmd(self, sub_cmd, args, options):
        credentials = self.parent_cluster.get_db_auth()
        if credentials:
            options += '-u {} -pw {} '.format(*credentials)
        return f"{self.add_install_prefix('/usr/bin/nodetool')} {options} {sub_cmd} {args}"

    def run_nodetool(self, sub_cmd, args="", options="", timeout=None,
                     ignore_status=False, verbose=True, coredump_on_timeout=False):
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
        :return: Remoter result object
        """
        cmd = self._gen_nodetool_cmd(sub_cmd, args, options)
        try:
            result = self.remoter.run(cmd, timeout=timeout, ignore_status=ignore_status, verbose=verbose)
            self.log.debug("Command '%s' duration -> %s s" % (result.command, result.duration))
            return result
        except Exception as details:  # pylint: disable=broad-except
            self.log.critical(f"Command '{cmd}' error: {details}")
            if coredump_on_timeout and isinstance(details, CommandTimedOut):
                self.generate_coredump_file()
            raise

    def check_node_health(self, retries: int = CHECK_NODE_HEALTH_RETRIES) -> None:
        # Task 1443: ClusterHealthCheck is bottle neck in scale test and create a lot of noise in 5000 tables test.
        # Disable it
        if not self.parent_cluster.params.get('cluster_health_check'):
            return

        for retry_n in range(1, retries+1):
            LOGGER.debug("Check the health of the node `%s' [attempt #%d]", self.name, retry_n)

            nodes_status = self.get_nodes_status()
            peers_details = self.get_peers_info() or {}
            gossip_info = self.get_gossip_info() or {}

            events = itertools.chain(
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
                    current_node=self), )

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
                    nodes_status[node_ip] = {'status': node_properties['state'], 'dc': dc}

        except Exception as exc:
            ClusterHealthValidatorEvent.NodeStatus(
                severity=Severity.WARNING,
                node=self.name,
                message=f"Unable to get nodetool status from `{self.name}': {exc}",
            ).publish()
        return nodes_status

    @retrying(n=5, sleep_time=5, raise_on_exceeded=False)
    def get_peers_info(self):
        cql_result = self.run_cqlsh('select peer, data_center, host_id, rack, release_version, '
                                    'rpc_address, schema_version, supported_features from system.peers',
                                    split=True, verbose=False)
        # peer | data_center | host_id | rack | release_version | rpc_address | schema_version | supported_features
        # ------+-------------+---------+------+-----------------+-------------+----------------+--------------------

        peers_details = {}
        for line in cql_result:
            line_splitted = line.split('|')
            if len(line_splitted) < 8:
                continue
            peer = line_splitted[0].strip()
            try:
                ipaddress.ip_address(peer)
            except ValueError:
                continue
            peers_details[peer] = {
                'data_center': line_splitted[1].strip(),
                'host_id': line_splitted[2].strip(),
                'rack': line_splitted[3].strip(),
                'release_version': line_splitted[4].strip(),
                'rpc_address': line_splitted[5].strip(),
                'schema_version': line_splitted[6].strip(),
                'supported_features': line_splitted[7].strip(),
            }

        return peers_details

    @retrying(n=5, sleep_time=10, raise_on_exceeded=False)
    def get_gossip_info(self):
        gossip_info = self.run_nodetool('gossipinfo', verbose=False)
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
                gossip_node_schemas[ip] = {'schema': schema, 'status': status, 'dc': dc}
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
        auth_params = '-u {} -p {}'.format(*credentials) if credentials else ''
        use_keyspace = "--keyspace {}".format(keyspace) if keyspace else ""
        ssl_params = '--ssl' if self.parent_cluster.params.get("client_encrypt") else ''
        options = "--no-color {auth_params} {use_keyspace} --request-timeout={timeout} " \
                  "--connect-timeout={connect_timeout} {ssl_params}".format(
                      auth_params=auth_params, use_keyspace=use_keyspace, timeout=timeout,
                      connect_timeout=connect_timeout, ssl_params=ssl_params)
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
            except Exception:
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
            tmp_file.write(Setup.get_startup_script())
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
        self.log.info("Add swap file to loader %s", self)
        result = self.remoter.run(commands, ignore_status=True)
        if not result.ok:
            self.log.warning("Swap file was not created on loader node %s.\nError details: %s", self, result.stderr)
        result = self.remoter.run("grep /sct_configured_swapfile /proc/swaps", ignore_status=True)
        if "sct_configured_swapfile" not in result.stdout:
            self.log.warning("Swap file is not used on loader node %s.\nError details: %s", self, result.stderr)

    def set_hostname(self):
        self.log.warning('Method set_hostname is not implemented for %s' % self.__class__.__name__)

    @property
    def scylla_packages_installed(self) -> List[str]:
        if self.distro.is_rhel_like:
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
        wait.wait_for(self.is_machine_image_configured, step=10, timeout=300)

    def get_sysctl_output(self) -> dict[str, str]:
        properties = {}
        result = self.remoter.sudo("sysctl -a", ignore_status=True)

        if not result.ok:
            self.log.error(f"sysctl command failed: {result}")
            return properties

        for line in result.stdout.strip().split("\n"):
            try:
                name, value = line.strip().split("=", 1)
                properties.update({name.strip(): value.strip()})
            except ValueError:
                self.log.error(f"Could not parse sysctl line: {line}")

        return properties

    def set_seed_flag(self, is_seed):
        self.is_seed = is_seed
        if hasattr(self.log, 'extra'):
            self.log.extra['prefix'] = str(self)


class FlakyRetryPolicy(RetryPolicy):

    """
    A retry policy that retries 5 times
    """

    def _retry_message(self, msg, retry_num):
        if retry_num < 5:
            LOGGER.debug("%s. Attempt #%d", msg, retry_num)
            return self.RETRY, None
        return self.RETHROW, None

    def on_read_timeout(self, query, consistency, required_responses,
                        received_responses, data_retrieved, retry_num):
        return self._retry_message(msg="Retrying read after timeout", retry_num=retry_num)

    def on_write_timeout(self, query, consistency, write_type,
                         required_responses, received_responses, retry_num):
        return self._retry_message(msg="Retrying write after timeout", retry_num=retry_num)

    def on_unavailable(self, query, consistency, required_replicas, alive_replicas, retry_num):
        return self._retry_message(msg="Retrying request after UE", retry_num=retry_num)


class BaseCluster:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    Cluster of Node objects.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, cluster_uuid=None, cluster_prefix='cluster', node_prefix='node', n_nodes=3, params=None,
                 region_names=None, node_type=None, extra_network_interface=False):
        self.extra_network_interface = extra_network_interface
        if params is None:
            params = {}
        if cluster_uuid is None:
            self.uuid = Setup.test_id()
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
        self.dead_nodes_ip_address_list = set()

        if Setup.REUSE_CLUSTER:
            # get_node_ips_param should be defined in child
            self._node_public_ips = self.params.get(self.get_node_ips_param(public_ip=True)) or []
            self._node_private_ips = self.params.get(self.get_node_ips_param(public_ip=False)) or []
            self.log.debug('Node public IPs: {}, private IPs: {}'.format(self._node_public_ips, self._node_private_ips))

        if isinstance(n_nodes, list):
            for dc_idx, num in enumerate(n_nodes):
                self.add_nodes(num, dc_idx=dc_idx, enable_auto_bootstrap=Setup.AUTO_BOOTSTRAP)
        elif isinstance(n_nodes, int):  # legacy type
            self.add_nodes(n_nodes, enable_auto_bootstrap=Setup.AUTO_BOOTSTRAP)
        else:
            raise ValueError('Unsupported type: {}'.format(type(n_nodes)))
        self.coredumps = dict()
        self.ldap_configured = False
        self.latency_results = dict()
        super(BaseCluster, self).__init__()

    @cached_property
    def tags(self) -> Dict[str, str]:
        key = self.node_type if "db" not in self.node_type else "db"
        action = self.params.get(f"post_behavior_{key}_nodes")
        return {**Setup.common_tags(),
                "NodeType": str(self.node_type),
                "keep_action": "terminate" if action == "destroy" else "", }

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

    def get_node_private_ips(self):
        return [node.private_ip_address for node in self.nodes]

    def get_node_public_ips(self):
        return [node.public_ip_address for node in self.nodes]

    def get_node_external_ips(self):
        return [node.external_address for node in self.nodes]

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
            self.dead_nodes_ip_address_list.add(node.ip_address)
        if node in self.nodes:
            self.nodes.remove(node)
        node.destroy()

    def get_db_auth(self):
        if self.params.get('use_ldap_authorization') and self.ldap_configured:
            user = LDAP_USERS[0]
            password = LDAP_PASSWORD
        else:
            user = self.params.get('authenticator_user')
            password = self.params.get('authenticator_password')
        return (user, password) if user and password else None

    def write_node_public_ip_file(self):
        public_ip_file_path = os.path.join(self.logdir, 'public_ips')
        with open(public_ip_file_path, 'w') as public_ip_file:
            public_ip_file.write("%s" % "\n".join(self.get_node_public_ips()))
            public_ip_file.write("\n")

    def write_node_private_ip_file(self):
        private_ip_file_path = os.path.join(self.logdir, 'private_ips')
        with open(private_ip_file_path, 'w') as private_ip_file:
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
                        verbose=True):
        if not port:
            port = node.CQL_PORT

        if protocol_version is None:
            protocol_version = 3

        authenticator = self.params.get('authenticator')
        if authenticator and authenticator == 'PasswordAuthenticator':
            user = self.params.get('authenticator_user')
            password = self.params.get('authenticator_password')

        if user is not None:
            auth_provider = PlainTextAuthProvider(username=user, password=password)
        else:
            auth_provider = None

        if ssl_opts is None and self.params.get('client_encrypt'):
            ssl_opts = {'ca_certs': './data_dir/ssl_conf/client/catest.pem'}
        self.log.debug(str(ssl_opts))
        cluster_driver = ClusterDriver(node_ips, auth_provider=auth_provider,
                                       compression=compression,
                                       protocol_version=protocol_version,
                                       load_balancing_policy=load_balancing_policy,
                                       default_retry_policy=FlakyRetryPolicy(),
                                       port=port, ssl_options=ssl_opts,
                                       connect_timeout=connect_timeout)
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
        node_ips = self.get_node_external_ips()
        wlrr = WhiteListRoundRobinPolicy(node_ips)
        return self._create_session(node=node, keyspace=keyspace, user=user, password=password,
                                    compression=compression, protocol_version=protocol_version,
                                    load_balancing_policy=wlrr, port=port, ssl_opts=ssl_opts, node_ips=node_ips,
                                    connect_timeout=connect_timeout, verbose=verbose)

    def cql_connection_exclusive(self, node, keyspace=None, user=None,  # pylint: disable=too-many-arguments
                                 password=None, compression=True,
                                 protocol_version=None, port=None,
                                 ssl_opts=None, connect_timeout=100, verbose=True):
        node_ips = [node.external_address]
        wlrr = WhiteListRoundRobinPolicy(node_ips)
        return self._create_session(node=node, keyspace=keyspace, user=user, password=password,
                                    compression=compression, protocol_version=protocol_version,
                                    load_balancing_policy=wlrr, port=port, ssl_opts=ssl_opts, node_ips=node_ips,
                                    connect_timeout=connect_timeout, verbose=verbose)

    @retrying(n=8, sleep_time=15, allowed_exceptions=(NoHostAvailable,))
    def cql_connection_patient(self, node, keyspace=None,  # pylint: disable=too-many-arguments
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

            for row in cql_session.execute(cmd).current_rows:
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


class NodeSetupFailed(Exception):
    def __init__(self, node, error_msg, traceback_str=""):
        super(NodeSetupFailed, self).__init__(error_msg)
        self.node = node
        self.error_msg = error_msg
        self.traceback_str = "\n" + traceback_str if traceback_str else ""

    def __str__(self):
        return f"[{self.node}] NodeSetupFailed: {self.error_msg}{self.traceback_str}"

    def __repr__(self):
        return self.__str__()


class NodeSetupTimeout(Exception):
    pass


def wait_for_init_wrap(method):
    """
    Wraps wait_for_init class method.
    Run setup of nodes simultaneously and wait for all the setups finished.
    Raise exception if setup failed or timeout expired.
    """
    @wraps(method)
    def wrapper(*args, **kwargs):
        cl_inst = args[0]
        LOGGER.debug('Class instance: %s', cl_inst)
        LOGGER.debug('Method kwargs: %s', kwargs)
        node_list = kwargs.get('node_list', None) or cl_inst.nodes
        timeout = kwargs.get('timeout', None)
        # remove all arguments which is not supported by BaseScyllaCluster.node_setup method
        setup_kwargs = {k: kwargs[k] for k in kwargs if k not in ["node_list", "check_node_health"]}

        _queue = queue.Queue()

        @raise_event_on_failure
        def node_setup(_node):
            exception_details = None
            try:
                cl_inst.node_setup(_node, **setup_kwargs)
            except Exception as ex:  # pylint: disable=broad-except
                exception_details = (str(ex), traceback.format_exc())
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
            if isinstance(cl_inst, BaseScyllaCluster) and not Setup.USE_LEGACY_CLUSTER_INIT:
                init_nodes.append(node)
                start_time = time.perf_counter()
                node_setup(node)
                verify_node_setup(start_time)
            else:
                setup_thread = threading.Thread(target=node_setup, name='NodeSetupThread',
                                                args=(node,), daemon=True)
                setup_thread.start()
                time.sleep(120)

        while len(results) != len(node_list):
            verify_node_setup(start_time)

        if isinstance(cl_inst, BaseScyllaCluster):
            cl_inst.wait_for_nodes_up_and_normal(nodes=node_list, verification_node=node_list[0])

        time_elapsed = time.perf_counter() - start_time
        cl_inst.log.debug('Setup duration -> %s s', int(time_elapsed))

        method(*args, **kwargs)
    return wrapper


class ClusterNodesNotReady(Exception):
    pass


class BaseScyllaCluster:  # pylint: disable=too-many-public-methods, too-many-instance-attributes
    node_setup_requires_scylla_restart = True
    name: str
    nodes: List[BaseNode]
    params: dict
    log: logging.Logger

    def __init__(self, *args, **kwargs):
        self.nemesis_termination_event = threading.Event()
        self.nemesis = []
        self.nemesis_threads = []
        self.nemesis_count = 0
        self._node_cycle = None
        super(BaseScyllaCluster, self).__init__(*args, **kwargs)

    @staticmethod
    def get_node_ips_param(public_ip=True):
        if Setup.MIXED_CLUSTER:
            return 'oracle_db_nodes_public_ip' if public_ip else 'oracle_db_nodes_private_ip'
        return 'db_nodes_public_ip' if public_ip else 'db_nodes_private_ip'

    def get_scylla_args(self):
        # pylint: disable=no-member
        return self.params.get('append_scylla_args_oracle') if self.name.find('oracle') > 0 else \
            self.params.get('append_scylla_args')

    def get_rack_nodes(self, rack: int) -> list:
        return sorted([node for node in self.nodes if node.rack == rack], key=lambda n: n.name)

    @property
    def racks(self) -> Set[int]:
        return set([node.rack for node in self.nodes])

    def set_seeds(self, wait_for_timeout=300, first_only=False):
        seeds_selector = self.params.get('seeds_selector')
        seeds_num = self.params.get('seeds_num')
        cluster_backend = self.params.get('cluster_backend')

        seed_nodes_ips = None
        if first_only:
            node = self.nodes[0]
            node.wait_ssh_up()
            seed_nodes_ips = [node.ip_address]

        elif seeds_selector == 'reflector' or Setup.REUSE_CLUSTER or cluster_backend == 'aws-siren':
            node = self.nodes[0]
            node.wait_ssh_up()
            # When cluster just started, seed IP in the scylla.yaml may be like '127.0.0.1'
            # In this case we want to ignore it and wait, when reflector will select real node and update scylla.yaml
            seed_nodes_ips = wait.wait_for(self.get_seed_selected_by_reflector,
                                           step=10, text='Waiting for seed is selected by reflector',
                                           timeout=wait_for_timeout, throw_exc=True)
        else:
            if seeds_selector == 'random':
                selected_nodes = librandom.sample(self.nodes, seeds_num)
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

    def enable_client_encrypt(self):
        for node in self.nodes:
            self.log.debug("Enabling client encryption on node")
            node.enable_client_encrypt()

    def disable_client_encrypt(self):
        for node in self.nodes:
            self.log.debug("Disabling client encryption on node")
            node.disable_client_encrypt()

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

        def update_scylla_packages(node, _queue):
            node.log.info('Updating DB packages')
            node.remoter.run('mkdir /tmp/scylla')
            node.remoter.send_files(new_scylla_bin, '/tmp/scylla', verbose=True)

            self._wait_for_preinstalled_scylla(node)
            # replace the packages
            node.remoter.run('yum list installed | grep scylla')
            node.remoter.sudo('rpm -URvh --replacepkgs --replacefiles /tmp/scylla/*.rpm',
                              ignore_status=False, verbose=True)
            node.remoter.run('yum list installed | grep scylla')
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
        seed_provider = [{
            "class_name": "org.apache.cassandra.locator.SimpleSeedProvider",
            "parameters": [{
                "seeds": ",".join(self.seed_nodes_ips),
            }, ],
        }, ]
        for node in self.nodes:
            with node.remote_scylla_yaml() as scylla_yml:
                scylla_yml["seed_provider"] = seed_provider

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

    def get_node_info_list(self, verification_node):
        """
            !!! Deprecated !!!!
            use self.get_nodetool_status instead
        """
        assert verification_node in self.nodes
        cmd_result = verification_node.run_nodetool('status')
        node_info_list = []
        for line in cmd_result.stdout.splitlines():
            line = line.strip()
            if line.startswith('UN'):
                try:
                    status, ip, load, _, tokens, owns, host_id, rack = line.split()
                    node_info = {'status': status,
                                 'ip': ip,
                                 'load': load,
                                 'tokens': tokens,
                                 'owns': owns,
                                 'host_id': host_id,
                                 'rack': rack}
                    # Cassandra banners have nodetool status output as well.
                    # Need to guarantee unique set of results.
                    node_ips = [node_info['ip']
                                for node_info in node_info_list]
                    if node_info['ip'] not in node_ips:
                        node_info_list.append(node_info)
                except ValueError:
                    pass
        return node_info_list

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
            verification_node = librandom.choice(self.nodes)
        status = {}
        res = verification_node.run_nodetool('status')
        data_centers = res.stdout.strip().split("Datacenter: ")
        for dc in data_centers:
            if dc:
                lines = dc.splitlines()
                dc_name = lines[0]
                status[dc_name] = {}
                for line in lines[1:]:
                    if line.startswith('--'):  # ignore the title line in result
                        continue
                    try:
                        splitted_line = line.split()
                        # Regulary nodetool status returns node load as "21.71 GB"
                        # Example: "UN  10.0.59.34    21.71 GB   256          ?       e5bcb094-e4de-43aa-8dc9-b1bf74b3b346  1a"
                        # But it may be the "?" instead and has no load_unit. Add empty string to prevent the failure
                        # Example: "UN  10.0.198.153  ?          256          ?       fba174cd-917a-40f6-ab62-cc58efaaf301  1a"
                        if len(splitted_line) == 7 and splitted_line[3].isdigit():
                            splitted_line.insert(3, '')
                        state, ip, load, load_unit, tokens, owns, host_id, rack = splitted_line
                        node_info = {'state': state,
                                     'load': '%s%s' % (load, load_unit),
                                     'tokens': tokens,
                                     'owns': owns,
                                     'host_id': host_id,
                                     'rack': rack,
                                     }
                        status[dc_name][ip] = node_info
                    except ValueError:
                        pass
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

        # Don't run health check in case parallel nemesis.
        # TODO: find how to recognize, that nemesis on the node is running
        if self.nemesis_count == 1:
            for node in self.nodes:
                node.check_node_health()
        else:
            ClusterHealthValidatorEvent.Info(
                message="Test runs with parallel nemesis. Nodes health checks are disabled.",
            ).publish()

        self.check_nodes_running_nemesis_count()
        ClusterHealthValidatorEvent.Done(message="Cluster health check finished").publish()

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
            assert check_schema_agreement_in_gossip_and_peers(node), 'Schema agreement is not reached'
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

    @retrying(n=60, sleep_time=3, allowed_exceptions=NETWORK_EXCEPTIONS + (ClusterNodesNotReady,),
              message="Waiting for nodes to join the cluster")
    def wait_for_nodes_up_and_normal(self, nodes, verification_node=None):
        self.check_nodes_up_and_normal(nodes=nodes, verification_node=verification_node)

    def get_scylla_version(self):
        if not self.nodes[0].scylla_version:
            scylla_version = self.nodes[0].get_scylla_version()

            if scylla_version:
                for node in self.nodes:
                    node.scylla_version = scylla_version

    def get_test_keyspaces(self):
        out = self.nodes[0].run_cqlsh('select keyspace_name from system_schema.keyspaces',
                                      split=True)
        return [ks.strip() for ks in out[3:-3] if 'system' not in ks]

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
            wait.wait_for(func=self.cfstat_reached_threshold, step=10, timeout=300,
                          text="Waiting until cfstat '%s' reaches value '%s'" % (key, size),
                          key=key, threshold=size, keyspaces=keyspace)

    def add_nemesis(self, nemesis, tester_obj):
        for nem in nemesis:
            for _ in range(nem['num_threads']):
                self.nemesis.append(nem['nemesis'](tester_obj=tester_obj,
                                                   termination_event=self.nemesis_termination_event))
        self.nemesis_count = sum(nem['num_threads'] for nem in nemesis)

    def clean_nemesis(self):
        self.nemesis = []

    @log_run_info("Start nemesis threads on cluster")
    def start_nemesis(self, interval=None):
        self.log.info('Clear _nemesis_termination_event')
        self.nemesis_termination_event.clear()
        for nemesis in self.nemesis:
            nemesis_thread = threading.Thread(target=nemesis.run, name='NemesisThread', args=(interval,), daemon=True)
            nemesis_thread.start()
            self.nemesis_threads.append(nemesis_thread)

    @log_run_info("Stop nemesis threads on cluster")
    def stop_nemesis(self, timeout=10):
        if self.nemesis_termination_event.is_set():
            return
        self.log.info('Set _nemesis_termination_event')
        self.nemesis_termination_event.set()
        for nemesis_thread in self.nemesis_threads:
            nemesis_thread.join(timeout)
        self.nemesis_threads = []

    def node_config_setup(self, node, seed_address=None, endpoint_snitch=None, murmur3_partitioner_ignore_msb_bits=None, client_encrypt=None):  # pylint: disable=too-many-arguments,invalid-name
        node.config_setup(seed_address=seed_address,
                          cluster_name=self.name,  # pylint: disable=no-member
                          enable_exp=self.params.get('experimental'),
                          endpoint_snitch=endpoint_snitch,
                          authenticator=self.params.get('authenticator'),
                          server_encrypt=self.params.get('server_encrypt'),
                          client_encrypt=client_encrypt if client_encrypt is not None else self.params.get(
                              'client_encrypt'),
                          append_scylla_yaml=self.params.get('append_scylla_yaml'),
                          append_scylla_args=self.get_scylla_args(),
                          hinted_handoff=self.params.get('hinted_handoff'),
                          authorizer=self.params.get('authorizer'),
                          alternator_port=self.params.get('alternator_port'),
                          murmur3_partitioner_ignore_msb_bits=murmur3_partitioner_ignore_msb_bits,
                          alternator_enforce_authorization=self.params.get('alternator_enforce_authorization'),
                          internode_compression=self.params.get('internode_compression'),
                          internode_encryption=self.params.get('internode_encryption'),
                          ldap=self.params.get('use_ldap_authorization'))

    def node_setup(self, node: BaseNode, verbose: bool = False, timeout: int = 3600):  # pylint: disable=too-many-branches
        node.wait_ssh_up(verbose=verbose, timeout=timeout)

        install_scylla = True

        if self.params.get("use_preinstalled_scylla"):
            if node.is_scylla_installed():
                install_scylla = False
            else:
                raise Exception("There is no pre-installed ScyllaDB")

        if not Setup.REUSE_CLUSTER:
            result = node.remoter.run('/sbin/ip -o link show |grep ether |awk -F": " \'{print $2}\'', verbose=True)
            devname = result.stdout.strip()
            if install_scylla:
                self._scylla_install(node)
            else:
                self._wait_for_preinstalled_scylla(node)
            if node.is_nonroot_install:
                node.stop_scylla_server(verify_down=False)
                node.remoter.run(f'{INSTALL_DIR}/sbin/scylla_setup --nic {devname} --no-raid-setup --no-io-setup',
                                 verbose=True, ignore_status=True)
                node.remoter.send_files(src='./configurations/io.conf', dst=f'{INSTALL_DIR}/etc/scylla.d/')
                node.remoter.send_files(src='./configurations/io_properties.yaml', dst=f'{INSTALL_DIR}/etc/scylla.d/')
                node.remoter.run(
                    f"sed -ie 's/io-properties-file=/io-properties-file=\/home\/{TEST_USER}\/scylladb/g' {INSTALL_DIR}/etc/scylla.d/io.conf")
                node.remoter.run(
                    f"sed -ie 's/mountpoint: .*/mountpoint: \/home\/{TEST_USER}\/scylladb/g' {INSTALL_DIR}/etc/scylla.d/io_properties.yaml")

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
                return

            if Setup.BACKTRACE_DECODING:
                node.install_scylla_debuginfo()

            if Setup.MULTI_REGION:
                node.datacenter_setup(self.datacenter)  # pylint: disable=no-member
            self.node_config_setup(node, ','.join(self.seed_nodes_ips), self.get_endpoint_snitch())

            self._scylla_post_install(node, install_scylla, devname)

            if self.node_setup_requires_scylla_restart:
                node.stop_scylla_server(verify_down=False)
                node.clean_scylla_data()
                node.remoter.sudo(cmd="rm -f /etc/scylla/ami_disabled", ignore_status=True)

                if self.is_ebs_volumes_attached():
                    result = node.remoter.sudo(cmd="scylla_io_setup")
                    if result.ok:
                        self.log.info("Scylla_io_setup result: %s", result.stdout)
                    else:
                        self.log.error(result)
                        raise Exception(f"{result.return_code}\n{result.stdout}\n{result.stderr}")
                node.start_scylla_server(verify_up=False)

            # code to increase java heap memory to scylla-jmx (because of #7609)
            jmx_memory = self.params.get("jmx_heap_memory")
            if jmx_memory:
                node.increase_jmx_heap_memory(jmx_memory)
                node.restart_scylla_jmx()

            self.log.info('io.conf right after reboot')
            node.remoter.sudo('cat /etc/scylla.d/io.conf')

            if self.params.get('use_mgmt'):
                self.install_scylla_manager(node)
        else:
            self._reuse_cluster_setup(node)

        node.wait_db_up(verbose=verbose, timeout=timeout)
        nodes_status = node.get_nodes_status()
        check_nodes_status(nodes_status=nodes_status, current_node=node)

        self.clean_replacement_node_ip(node)

    def install_scylla_manager(self, node):
        pkgs_url = self.params.get('scylla_mgmt_pkg')
        pkg_path = None
        if pkgs_url:
            pkg_path = download_dir_from_cloud(pkgs_url)
            node.remoter.run('mkdir -p {}'.format(pkg_path))
            node.remoter.send_files(src='{}*.rpm'.format(pkg_path), dst=pkg_path)
        node.install_manager_agent(package_path=pkg_path)
        cluster_backend = self.params.get("cluster_backend")
        if cluster_backend == "aws":
            region = self.params.get('region_name').split()
            update_config_file(node=node, region=region[0])
        else:
            self.log.warning("Backend '%s' not support. Won't configure manager backups!", cluster_backend)

    def _scylla_install(self, node):
        node.update_repo_cache()
        if node.init_system == 'systemd' and (node.is_ubuntu() or node.is_debian()):
            node.remoter.sudo('systemctl disable apt-daily.timer')
            node.remoter.sudo('systemctl disable apt-daily-upgrade.timer')
            node.remoter.sudo('systemctl stop apt-daily.timer', ignore_status=True)
            node.remoter.sudo('systemctl stop apt-daily-upgrade.timer', ignore_status=True)
        node.clean_scylla()
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

    def get_endpoint_snitch(self, default_multi_region="GossipingPropertyFileSnitch"):
        endpoint_snitch = self.params.get('endpoint_snitch')
        if Setup.MULTI_REGION:
            if not endpoint_snitch:
                endpoint_snitch = default_multi_region
        return endpoint_snitch

    @staticmethod
    def clean_replacement_node_ip(node):
        if node.replacement_node_ip:
            # If this is a replacement node, we need to set back configuration in case
            # when scylla-server process will be restarted
            node.replacement_node_ip = None
            node.remoter.run(
                f'sudo sed -i -e "s/^replace_address_first_boot:/# replace_address_first_boot:/g" {node.add_install_prefix(SCYLLA_YAML_PATH)}')

    @staticmethod
    def verify_logging_from_nodes(nodes_list):
        for node in nodes_list:
            if not os.path.exists(node.system_log):
                error_msg = "No db log from node [%s] " % node
                raise Exception(error_msg)
        return True

    @wait_for_init_wrap
    def wait_for_init(self, node_list=None, verbose=False, timeout=None, check_node_health=True):  # pylint: disable=unused-argument, too-many-arguments
        """
        Scylla cluster setup.
        :param node_list: List of nodes to watch for init.
        :param verbose: Whether to print extra info while watching for init.
        :param timeout: timeout in minutes to wait for init to be finished
        :param check_node_health: select if run node health check or not
        :return:
        """
        node_list = node_list or self.nodes
        self.get_scylla_version()

        wait.wait_for(self.verify_logging_from_nodes, nodes_list=node_list,
                      text="wait for db logs", step=20, timeout=300, throw_exc=True)

        self.log.info("{} nodes configured and started.".format(node_list))

        # If wait_for_init is called during cluster initialization we may want this validation will be performed,
        # but if it was called from nemesis, we don't need it in the middle of nemesis. It may cause to not relevant
        # failures
        if check_node_health:
            node_list[0].check_node_health()

    def restart_scylla(self, nodes=None, random=False):
        if nodes:
            nodes_to_restart = nodes
        else:
            nodes_to_restart = self.nodes
        if random:
            nodes_to_restart = librandom.sample(nodes_to_restart, len(nodes_to_restart))
        self.log.info("Going to restart Scylla on %s" % [n.name for n in nodes_to_restart])
        for node in nodes_to_restart:
            node.stop_scylla(verify_down=True)
            node.start_scylla(verify_up=True)
            self.log.debug("'{0.name}' restarted.".format(node))

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

    def decommission(self, node):
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
        node.run_nodetool("decommission")
        verification_node = librandom.choice(self.nodes)
        node_ip_list = get_node_ip_list(verification_node)
        while verification_node == node or node_ip_list is None:
            verification_node = librandom.choice(self.nodes)
            node_ip_list = get_node_ip_list(verification_node)

        if target_node_ip in node_ip_list:
            cluster_status = self.get_nodetool_status(verification_node)
            error_msg = ('Node that was decommissioned %s still in the cluster. '
                         'Cluster status info: %s' % (node,
                                                      cluster_status))

            LOGGER.error('Decommission %s FAIL', node)
            LOGGER.error(error_msg)
            raise NodeStayInClusterAfterDecommission(error_msg)

        LOGGER.info('Decommission %s PASS', node)
        self.terminate_node(node)  # pylint: disable=no-member
        Setup.tester_obj().monitors.reconfigure_scylla_monitoring()

    @property
    def scylla_manager_node(self) -> BaseNode:
        return Setup.tester_obj().monitors.nodes[0]

    @property
    def scylla_manager_auth_token(self) -> str:
        return Setup.tester_obj().monitors.mgmt_auth_token

    @property
    def scylla_manager_cluster_name(self):
        return self.name

    def get_cluster_manager(self, create_cluster_if_not_exists: bool = True) -> AnyManagerCluster:
        if not self.params.get('use_mgmt'):
            raise ScyllaManagerError('Scylla-manager configuration is not defined!')
        manager_tool = mgmt.get_scylla_manager_tool(manager_node=self.scylla_manager_node, scylla_cluster=self)
        LOGGER.debug("sctool version is : {}".format(manager_tool.version))
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

    def is_ebs_volumes_attached(self) -> bool:
        """check data device type usage

        check if scylla cluster uses the
        ebs volumes for instance on aws cloud

        :returns: if ebs volumes attached return true
        :rtype: {bool}
        """
        backend = self.params.get('cluster_backend')
        data_device_type = self.params.get("data_device")

        if backend == "aws" and data_device_type == "attached":
            self.log.info("EBS volumes attached")
            return True

        return False


class BaseLoaderSet():

    def __init__(self, params):
        self._loader_cycle = None
        self.params = params
        self._gemini_version = None
        self._gemini_base_path = None

    @property
    def gemini_version(self):
        if not self._gemini_version:
            try:
                result = self.nodes[0].remoter.run('cd $HOME; ./gemini --version', ignore_status=True)
                if result.ok:
                    self._gemini_version = get_gemini_version(result.stdout)
            except Exception as details:  # pylint: disable=broad-except
                self.log.error("Error get gemini version: %s", details)
        return self._gemini_version

    @property
    def gemini_base_path(self):
        if not self._gemini_base_path:
            result = self.nodes[0].remoter.run("echo $HOME", ignore_status=True)
            self._gemini_base_path = result.stdout.strip()
        return self._gemini_base_path

    def install_gemini(self, node):
        gemini_version = self.params.get('gemini_version')
        if gemini_version.lower() == 'latest':
            gemini_version = get_latest_gemini_version()

        gemini_url = 'http://downloads.scylladb.com/gemini/{0}/gemini_{0}_Linux_x86_64.tar.gz'.format(gemini_version)
        # TODO: currently schema is not used by gemini tool need to store the schema
        #       in data_dir for each test
        gemini_schema_url = self.params.get('gemini_schema_url')
        if not gemini_url or not gemini_schema_url:
            self.log.warning('Gemini URLs should be defined to run the gemini tool')
        else:
            gemini_tar = os.path.basename(gemini_url)  # pylint: disable=unused-variable
            install_gemini_script = dedent(f"""
                cd {self.gemini_base_path}
                rm -rf gemini*
                curl -LO {gemini_url}
                tar -xvf {gemini_tar}
                chmod a+x gemini
                curl -LO  {gemini_schema_url}
            """)
            node.remoter.run("bash -cxe '%s'" % install_gemini_script)
            self.log.debug('Gemini version {}'.format(self.gemini_version))

    def node_setup(self, node, verbose=False, db_node_address=None, **kwargs):  # pylint: disable=unused-argument
        # pylint: disable=too-many-statements

        self.log.info('Setup in BaseLoaderSet')
        node.wait_ssh_up(verbose=verbose)
        # add swap file
        if not Setup.REUSE_CLUSTER:
            swap_size = self.params.get("loader_swap_size")
            if not swap_size:
                self.log.info("Swap file for the loader is not configured")
            else:
                node.create_swap_file(swap_size)
        # update repo cache and system after system is up
        node.update_repo_cache()

        if Setup.REUSE_CLUSTER:
            self.kill_stress_thread()
            return

        collectd_setup = ScyllaCollectdSetup()
        collectd_setup.install(node)
        self.install_gemini(node=node)
        if self.params.get('client_encrypt'):
            node.config_client_encrypt()

        result = node.remoter.run('test -e ~/PREPARED-LOADER', ignore_status=True)
        node.remoter.sudo("bash -cxe \"echo '*\t\thard\tcore\t\tunlimited\n*\t\tsoft\tcore\t\tunlimited' "
                          ">> /etc/security/limits.d/20-coredump.conf\"")
        if result.exit_status == 0:
            # Update existing scylla-bench to latest
            node.remoter.run('go get -u github.com/scylladb/scylla-bench')
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
                apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 17723034C56D4B19
                echo 'deb http://download.opensuse.org/repositories/home:/scylladb:/scylla-3rdparty-jessie/Debian_8.0/ /' |sudo tee /etc/apt/sources.list.d/scylla-3rdparty.list
                apt-get update
                apt-get install -y openjdk-8-jre-headless -t jessie-backports
                update-java-alternatives --jre-headless -s java-1.8.0-openjdk-amd64
            """)
            node.remoter.run('sudo bash -cxe "%s"' % install_java_script)
        elif node.distro.is_debian10:
            node.remoter.sudo(shell_script_cmd("""\
                apt-get update
                apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5E08FBD8B5D6EC9C
                apt-get install -y openjdk-11-jre openjdk-11-jre-headless
            """))

        scylla_repo_loader = self.params.get('scylla_repo_loader')
        if not scylla_repo_loader:
            scylla_repo_loader = self.params.get('scylla_repo')
        node.download_scylla_repo(scylla_repo_loader)
        if node.is_rhel_like():
            node.remoter.run('sudo yum install -y {}-tools'.format(node.scylla_pkg()))
        else:
            node.remoter.run('sudo apt-get update')
            node.remoter.run('sudo apt-get install -y -o Dpkg::Options::="--force-confdef"'
                             ' -o Dpkg::Options::="--force-confold" -o Dpkg::Options::="--force-overwrite"'
                             ' --allow-unauthenticated {}-tools'.format(node.scylla_pkg()))

        if db_node_address is not None:
            node.remoter.run("echo 'export DB_ADDRESS=%s' >> $HOME/.bashrc" % db_node_address)

        node.wait_cs_installed(verbose=verbose)

        self.install_scylla_bench(node)

        # install docker
        docker_install = dedent("""
            curl -fsSL get.docker.com -o get-docker.sh
            sh get-docker.sh
            systemctl enable docker
            systemctl start docker
        """)
        node.remoter.run('sudo bash -cxe "%s"' % docker_install)

        node.remoter.run('sudo usermod -aG docker $USER', change_context=True)

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
        self.kill_cassandra_stress_thread()
        self.kill_stress_thread_bench()
        self.kill_gemini_thread()
        self.kill_docker_loaders()

    def kill_cassandra_stress_thread(self):
        for loader in self.nodes:
            try:
                loader.remoter.run(cmd='pgrep -f cassandra-stress | xargs -I{}  kill -TERM -{}',
                                   verbose=False, ignore_status=True)
            except Exception as ex:  # pylint: disable=broad-except
                self.log.warning("failed to kill stress-command on [%s]: [%s]",
                                 str(loader), str(ex))

    def kill_docker_loaders(self):
        for loader in self.nodes:
            try:
                loader.remoter.run(cmd='docker ps -a -q | docker rm -f', verbose=False, ignore_status=True)
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
            match = re.findall(r'.*READ:(\d+), WRITE:(\d+)]', line)
            if match:  # parse results for mixed workload
                results['%s read' % key] = match[0][0]
                results['%s write' % key] = match[0][1]

        if not enable_parse:
            LOGGER.warning('Cannot find summary in c-stress results: %s', lines[-10:])
            return {}
        return results

    @staticmethod
    def _parse_cs_results(lines):
        results = dict()
        results['time'] = []
        results['ops'] = []
        results['totalops'] = []
        results['latmax'] = []
        results['lat999'] = []
        results['lat99'] = []
        results['lat95'] = []
        for line in lines:
            line.strip()
            if line.startswith('total,'):
                items = line.split(',')
                totalops = items[1]
                ops = items[2]
                lat95 = items[7]
                lat99 = items[8]
                lat999 = items[9]
                latmax = items[10]
                time_point = items[11]
                results['time'].append(time_point)
                results['totalops'].append(totalops)
                results['ops'].append(ops)
                results['lat95'].append(lat95)
                results['lat99'].append(lat99)
                results['lat999'].append(lat999)
                results['latmax'].append(latmax)
        return results

    def kill_stress_thread_bench(self):
        for loader in self.nodes:
            sb_active = loader.remoter.run(cmd='pgrep -f scylla-bench', verbose=False, ignore_status=True)
            if sb_active.exit_status == 0:
                kill_result = loader.remoter.run('pkill -f -SIGINT scylla-bench', ignore_status=True)
                if kill_result.exit_status != 0:
                    self.log.warning('Terminate scylla-bench on node %s:\n%s',
                                     loader, kill_result)

    def kill_gemini_thread(self):
        for loader in self.nodes:
            sb_active = loader.remoter.run(cmd='pgrep -f gemini', verbose=False, ignore_status=True)
            if sb_active.exit_status == 0:
                kill_result = loader.remoter.run('pkill -f -QUIT gemini', ignore_status=True)
                if kill_result.exit_status != 0:
                    self.log.warning('Terminate gemini on node %s:\n%s', loader, kill_result)

    @staticmethod
    def install_scylla_bench(node):
        if node.distro.is_rhel_like:
            node.remoter.sudo("yum install -y git")
        else:
            node.remoter.sudo(shell_script_cmd("""\
                apt-get update
                apt-get install -y git
            """))
        node.remoter.sudo(shell_script_cmd("""\
            curl -LO https://storage.googleapis.com/golang/go1.13.linux-amd64.tar.gz
            tar -C /usr/local -xvzf go1.13.linux-amd64.tar.gz
            echo 'export GOPATH=$HOME/go' >> $HOME/.bash_profile
            echo 'export PATH=$PATH:/usr/local/go/bin' >> $HOME/.bash_profile
            source $HOME/.bash_profile
            go get -u github.com/scylladb/scylla-bench
        """))


class BaseMonitorSet():  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    # This is a Mixin for monitoring cluster and should not be inherited

    json_file_params_for_replace = {"$test_name": get_test_name()}

    def __init__(self, targets, params):
        self.targets = targets
        self.params = params
        self.local_metrics_addr = start_metrics_server()  # start prometheus metrics server locally and return local ip
        self.sct_ip_port = self.set_local_sct_ip()
        self.grafana_port = 3000
        self.monitor_branch = self.params.get('monitor_branch')
        self._monitor_install_path_base = None
        self.phantomjs_installed = False
        self.grafana_start_time = 0
        self._sct_dashboard_json_file = None

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
    def scylla_version(self):
        return self.targets["db_cluster"].nodes[0].scylla_version

    @property
    def monitoring_version(self):
        self.log.debug("Using %s ScyllaDB version to derive monitoring version" %
                       self.scylla_version)
        version = re.match(r"(\d+\.\d+)", self.scylla_version)
        if not version or 'dev' in self.scylla_version:
            return 'master'
        else:
            return version.group(1)

    @property
    def is_enterprise(self):
        return self.targets["db_cluster"].nodes[0].is_enterprise

    @property
    def sct_dashboard_json_file(self):
        if not self._sct_dashboard_json_file:
            sct_dashboard_json_filename = f"scylla-dash-per-server-nemesis.{self.monitoring_version}.json"
            sct_dashboard_json_path = get_data_dir_path(sct_dashboard_json_filename)
            if not os.path.exists(sct_dashboard_json_path):
                sct_dashboard_json_filename = "scylla-dash-per-server-nemesis.master.json"
                sct_dashboard_json_path = get_data_dir_path(sct_dashboard_json_filename)
            self._sct_dashboard_json_file = sct_dashboard_json_path
            self.sct_dashboard_json_file_content_update(update_params=self.json_file_params_for_replace,
                                                        json_file=self._sct_dashboard_json_file)
        return self._sct_dashboard_json_file

    @staticmethod
    def sct_dashboard_json_file_content_update(update_params: dict, json_file: str):
        # Read json data to the string
        with open(json_file, 'r') as file:
            json_data = file.read()

        for param, value in update_params.items():
            json_data = json_data.replace(param, value)

        with open(json_file, 'w') as file:
            json.dump(json.loads(json_data), file, indent=2)

    def node_setup(self, node, **kwargs):  # pylint: disable=unused-argument
        self.log.info('Setup in BaseMonitorSet')
        node.wait_ssh_up()
        # add swap file
        if not Setup.REUSE_CLUSTER:
            monitor_swap_size = self.params.get("monitor_swap_size")
            if not monitor_swap_size:
                self.log.info("Swap file for the monitor is not configured")
            else:
                node.create_swap_file(monitor_swap_size)
        # update repo cache and system after system is up
        node.update_repo_cache()
        self.mgmt_auth_token = Setup.test_id()  # pylint: disable=attribute-defined-outside-init

        if Setup.REUSE_CLUSTER:
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
            self.install_scylla_manager(node, auth_token=self.mgmt_auth_token)

    def install_scylla_manager(self, node, auth_token):
        if self.params.get('use_mgmt'):
            node.install_scylla(scylla_repo=self.params.get('scylla_repo_m'))
            package_path = self.params.get('scylla_mgmt_pkg')
            if package_path:
                node.remoter.run('mkdir -p {}'.format(package_path))
                node.remoter.send_files(src='{}*.rpm'.format(package_path), dst=package_path)
            node.install_mgmt(scylla_mgmt_repo=self.params.get('scylla_mgmt_repo'), auth_token=auth_token,
                              package_url=package_path)
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
                curl -fsSL get.docker.com -o get-docker.sh
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
                add-apt-repository -y ppa:deadsnakes/ppa
                apt-get update
                apt-get install -y python3.6 python3.6-dev
                apt-get install -y python-setuptools unzip wget
                apt install -y python3-pip
                python3.6 -m pip install --upgrade pip
                python3.6 -m pip install pyyaml
                pip3 install -I -U psutil
                systemctl start docker
            """)
        elif node.is_debian9():
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
        start_node_exporter_script = dedent(f'''
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
            with open(local_template_tmp) as output_file:
                templ_yaml = yaml.safe_load(output_file)
                self.log.debug("Configs %s" % templ_yaml)
            loader_targets_list = ["[%s]:9103" % n.ip_address for n in self.targets["loaders"].nodes]

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
                gemini_loader_targets_list = ["%s:2112" % n.ip_address for n in self.targets["loaders"].nodes]
                scrape_configs.append(dict(job_name="gemini_metrics", honor_labels=True,
                                           static_configs=[dict(targets=gemini_loader_targets_list)]))

            if self.sct_ip_port:
                scrape_configs.append(dict(job_name="sct_metrics", honor_labels=True,
                                           static_configs=[dict(targets=[self.sct_ip_port])]))
            with open(local_template, "w") as output_file:
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
                expr: collectd_cassandra_stress_write_gauge{type="lat_perc_99"} > 1000
                for: 1s
                labels:
                  severity: "1"
                  sct_severity: "ERROR"
                annotations:
                  description: "Cassandra Stress write latency more than 1000ms"
                  summary: "Cassandra Stress write latency is more than 1000ms during 1 sec period of time"

            # Alert for YCSB error spikes
              - alert: YCSBTooManyErrors
                expr: sum(rate(collectd_ycsb_read_failed_gauge{type="count"}[1m])) > 5 OR sum(rate(collectd_ycsb_update_failed_gauge{type="count"}[1m])) > 5  OR sum(rate(collectd_ycsb_insert_failed_gauge{type="count"}[1m])) > 5
                for: 1s
                labels:
                  severity: "4"
                  sct_severity: "CRITICAL"
                annotations:
                  description: "YCSB errors more than 5 errors per min"
                  summary:  "YCSB errors more than 5 errors per min"

            # Alert for YCSB validation error spikes
              - alert: YCSBTooManyVerifyErrors
                expr: sum(rate(collectd_ycsb_verify_gauge{type="ERROR"}[1m])) > 5 OR sum(rate(collectd_ycsb_verify_gauge{type="UNEXPECTED_STATE"}[1m])) > 5
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
            monitoring_targets = " ".join(f"[{n.ip_address}]:9180" for n in self.targets["db_cluster"].nodes)
            node.remoter.sudo(shell_script_cmd(f"""\
                cd {self.monitor_install_path}
                mkdir -p {self.monitoring_conf_dir}
                export PATH=/usr/local/bin:$PATH  # hack to enable running on docker
                python3 genconfig.py -s -n -d {self.monitoring_conf_dir} {monitoring_targets}
            """), verbose=True)

            if self.params.get("cluster_backend") != "docker":
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

    def start_scylla_monitoring(self, node):
        node.remoter.run("cp {0.monitor_install_path}/prometheus/scylla_manager_servers.example.yml"
                         " {0.monitor_install_path}/prometheus/scylla_manager_servers.yml".format(self))
        labels = " ".join(f"--label {key}={value}" for key, value in node.tags.items())
        run_script = dedent(f"""
            cd -P {self.monitor_install_path}
            mkdir -p {self.monitoring_data_dir}
            ./start-all.sh \
            -D "{labels}" \
            -s `realpath "{self.monitoring_conf_dir}/scylla_servers.yml"` \
            -n `realpath "{self.monitoring_conf_dir}/node_exporter_servers.yml"` \
            -d `realpath "{self.monitoring_data_dir}"` -l -v master,{self.monitoring_version} -b "-web.enable-admin-api"
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
                      json_filename=self.sct_dashboard_json_file)

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

    def get_grafana_screenshot_and_snapshot(self, test_start_time=None):
        """
            Take screenshot of the Grafana per-server-metrics dashboard and upload to S3
        """
        if not test_start_time:
            self.log.error("No start time for test")
            return {}
        date_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        grafana_extra_entities = []
        if 'alternator_port' in self.params:
            grafana_extra_entities = [
                {
                    'name': 'alternator',
                    'path': 'd/alternator-{version}/{dashboard_name}',
                    'resolution': '1920px*4000px'
                }
            ]

        screenshot_links = []
        snapshots = []
        for node in self.nodes:
            screenshot_collector = GrafanaScreenShot(name="grafana-screenshot",
                                                     test_start_time=test_start_time,
                                                     extra_entities=grafana_extra_entities)
            screenshot_files = screenshot_collector.collect(node, self.logdir)
            for screenshot in screenshot_files:
                s3_path = "{test_id}/{date}".format(test_id=Setup.test_id(), date=date_time)
                screenshot_links.append(S3Storage().upload_file(screenshot, s3_path))

            snapshots_collector = GrafanaSnapshot(name="grafana-snapshot",
                                                  test_start_time=test_start_time,
                                                  extra_entities=grafana_extra_entities)
            snapshots_data = snapshots_collector.collect(node, self.logdir)
            snapshots.extend(snapshots_data.get('links', []))
        return {'screenshots': screenshot_links, 'snapshots': snapshots}

    def upload_annotations_to_s3(self):
        annotations_url = ''
        if not self.nodes:
            return annotations_url
        try:
            annotations = self.get_grafana_annotations(self.nodes[0])
            if annotations:
                annotations_url = S3Storage().generate_url('annotations.json', Setup.test_id())
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
                return upload_archive_to_s3(snapshot_archive, Setup.test_id())
        except Exception as details:
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
