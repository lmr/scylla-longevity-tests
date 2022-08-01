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
from collections import defaultdict
from dataclasses import asdict

import logging
import os
import re
import stat
import time
import unittest
import unittest.mock
from typing import NamedTuple, Optional, Union, List, Dict, Any
from uuid import uuid4, UUID
from functools import wraps, cached_property, cache
import threading
import signal
import sys
import json

import yaml
from invoke.exceptions import UnexpectedExit, Failure

from cassandra.concurrent import execute_concurrent_with_args  # pylint: disable=no-name-in-module
from cassandra import ConsistencyLevel

from argus.db.db_types import TestStatus, PackageVersion
from sdcm import nemesis, cluster_docker, cluster_k8s, cluster_baremetal, db_stats, wait
from sdcm.cluster import NoMonitorSet, SCYLLA_DIR, TestConfig, UserRemoteCredentials, BaseLoaderSet, BaseMonitorSet, \
    BaseScyllaCluster, BaseNode, MAX_TIME_WAIT_FOR_ALL_NODES_UP
from sdcm.argus_test_run import ArgusTestRun
from sdcm.cluster_azure import ScyllaAzureCluster, LoaderSetAzure, MonitorSetAzure
from sdcm.cluster_gce import ScyllaGCECluster
from sdcm.cluster_gce import LoaderSetGCE
from sdcm.cluster_gce import MonitorSetGCE
from sdcm.cluster_aws import CassandraAWSCluster
from sdcm.cluster_aws import ScyllaAWSCluster
from sdcm.cluster_aws import LoaderSetAWS
from sdcm.cluster_aws import MonitorSetAWS
from sdcm.cluster_k8s import mini_k8s, gke, eks, LOADER_CLUSTER_CONFIG
from sdcm.cluster_k8s.eks import MonitorSetEKS
from sdcm.provision.azure.provisioner import AzureProvisioner
from sdcm.provision.provisioner import provisioner_factory
from sdcm.scan_operation_thread import FullScanThread, FullPartitionScanThread
from sdcm.nosql_thread import NoSQLBenchStressThread
from sdcm.scylla_bench_thread import ScyllaBenchThread
from sdcm.cassandra_harry_thread import CassandraHarryThread
from sdcm.utils.aws_utils import init_monitoring_info_from_params, get_ec2_network_configuration, get_ec2_services, \
    get_common_params, init_db_info_from_params, ec2_ami_get_root_device_name
from sdcm.utils.common import format_timestamp, wait_ami_available, tag_ami, update_certificates, \
    download_dir_from_cloud, get_post_behavior_actions, get_testrun_status, download_encrypt_keys, PageFetcher, \
    rows_to_list, make_threads_be_daemonic_by_default, ParallelObject, clear_out_all_exit_hooks, \
    change_default_password
from sdcm.utils.get_username import get_username
from sdcm.utils.decorators import log_run_info, retrying
from sdcm.utils.ldap import LDAP_USERS, LDAP_PASSWORD, LDAP_ROLE, LDAP_BASE_OBJECT, \
    LdapConfigurationError, LdapServerType
from sdcm.utils.log import configure_logging, handle_exception
from sdcm.db_stats import PrometheusDBStats
from sdcm.results_analyze import PerformanceResultsAnalyzer, SpecifiedStatsPerformanceAnalyzer, \
    LatencyDuringOperationsPerformanceAnalyzer
from sdcm.sct_config import init_and_verify_sct_config
from sdcm.sct_events import Severity
from sdcm.sct_events.setup import start_events_device, stop_events_device
from sdcm.sct_events.system import InfoEvent, TestFrameworkEvent, TestResultEvent, TestTimeoutEvent
from sdcm.sct_events.file_logger import get_events_grouped_by_category, get_logger_event_summary
from sdcm.sct_events.events_analyzer import stop_events_analyzer
from sdcm.sct_events.grafana import start_posting_grafana_annotations
from sdcm.stress_thread import CassandraStressThread, get_timeout_from_stress_cmd
from sdcm.gemini_thread import GeminiStressThread
from sdcm.utils.log_time_consistency import DbLogTimeConsistencyAnalyzer
from sdcm.utils.threads_and_processes_alive import gather_live_processes_and_dump_to_file, \
    gather_live_threads_and_dump_to_file
from sdcm.ycsb_thread import YcsbStressThread
from sdcm.ndbench_thread import NdBenchStressThread
from sdcm.kcl_thread import KclStressThread, CompareTablesSizesThread
from sdcm.localhost import LocalHost
from sdcm.cdclog_reader_thread import CDCLogReaderThread
from sdcm.logcollector import (
    KubernetesAPIServerLogCollector,
    KubernetesLogCollector,
    LoaderLogCollector,
    MonitorLogCollector,
    SCTLogCollector,
    ScyllaLogCollector,
    SirenManagerLogCollector,
)
from sdcm.send_email import build_reporter, read_email_data_from_file, get_running_instances_for_email_report, \
    save_email_data_to_file
from sdcm.utils import alternator
from sdcm.utils.profiler import ProfilerFactory
from sdcm.remote import RemoteCmdRunnerBase
from sdcm.utils.gce_utils import get_gce_services
from sdcm.utils.auth_context import temp_authenticator
from sdcm.keystore import KeyStore
from sdcm.utils.latency import calculate_latency

CLUSTER_CLOUD_IMPORT_ERROR = ""
try:
    import cluster_cloud
except ImportError as import_exc:
    cluster_cloud = None
    CLUSTER_CLOUD_IMPORT_ERROR = str(import_exc)

configure_logging(exception_handler=handle_exception, variables={'log_dir': TestConfig().logdir()})

try:
    from botocore.vendored.requests.packages.urllib3.contrib.pyopenssl import extract_from_urllib3

    # Don't use pyOpenSSL in urllib3 - it causes an ``OpenSSL.SSL.Error``
    # exception when we try an API call on an idled persistent connection.
    # See https://github.com/boto/boto3/issues/220
    extract_from_urllib3()
except ImportError:
    pass


TEST_LOG = logging.getLogger(__name__)


def teardown_on_exception(method):
    """
    Ensure that resources used in test are cleaned upon unhandled exceptions. and every process are stopped, and logs
    are uploaded

    :param method: ScyllaClusterTester method to wrap.
    :return: Wrapped method.
    """

    @wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            TestFrameworkEvent(
                source=args[0].__class__.__name__,
                source_method='SetUp',
                exception=exc
            ).publish_or_dump()
            TEST_LOG.exception("Exception in %s. Will call tearDown", method.__name__)
            args[0].tearDown()
            raise

    return wrapper


class silence:  # pylint: disable=invalid-name
    """
    A decorator and context manager that catch, log and store any exception that
        happened within wrapped function or within context clause.
    Two ways of using it:
    1) as decorator
        class someclass:
            @silence(verbose=True)
            def my_method(self):
                ...

    2) as context manager
        class someclass:
            def my_method(self):
                with silence(parent=self, name='Step #1'):
                    ...
                with silence(parent=self, name='Step #2'):
                    ...
    """
    test = None
    name: str = None

    def __init__(self, parent=None, name: str = None, verbose: bool = False):
        self.parent = parent
        self.name = name
        self.verbose = verbose
        self.log = logging.getLogger(self.__class__.__name__)

    def __enter__(self):
        pass

    def __call__(self, funct):
        def decor(*args, **kwargs):
            if self.name is None:
                name = funct.__name__
            else:
                name = self.name
            result = None
            try:
                self.log.debug("Silently running '%s'", name)
                result = funct(*args, **kwargs)
                self.log.debug("Finished '%s'. No errors were silenced.", name)
            except Exception as exc:  # pylint: disable=broad-except
                self.log.debug("Finished '%s'. %s exception was silenced.", name, str(type(exc)))
                self._store_test_result(args[0], exc, exc.__traceback__, name)
            return result

        return decor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            self.log.debug("Finished '%s'. No errors were silenced.", self.name)
        else:
            self.log.debug("Finished '%s'. %s exception was silenced.", self.name, str(exc_type))
            self._store_test_result(self.parent, exc_val, exc_tb, self.name)
        return True

    @staticmethod
    def _store_test_result(parent, exc_val, exc_tb, name):
        TestFrameworkEvent(
            source=parent.__class__.__name__,
            source_method=name,
            message=f'{name} (silenced) failed with:',
            exception=exc_val,
            severity=getattr(exc_val, 'severity', Severity.ERROR),
            trace=exc_tb.tb_frame,
        ).publish_or_dump(default_logger=parent.log)


class CriticalTestFailure(BaseException):
    pass


def critical_failure_handler(signum, frame):  # pylint: disable=unused-argument
    try:
        if TestConfig().tester_obj().teardown_started:
            TEST_LOG.info("A critical event happened during tearDown")
            return
    except Exception:  # pylint: disable=broad-except
        pass
    raise CriticalTestFailure("Critical Error has failed the test")  # pylint: disable=raise-missing-from


signal.signal(signal.SIGUSR2, critical_failure_handler)


class SchemaVersion(NamedTuple):
    schema_id: str
    node_ips: List[str]


class ClusterInformation(NamedTuple):
    name: str
    snitch: str
    partitioner: str
    schema_versions: List[SchemaVersion]


class ClusterTester(db_stats.TestStatsMixin, unittest.TestCase):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    log = None
    localhost = None
    events_processes_registry = None
    monitors: BaseMonitorSet = None
    loaders: BaseLoaderSet = None
    db_cluster: BaseScyllaCluster = None
    k8s_cluster: Union[eks.EksCluster, gke.GkeCluster, mini_k8s.LocalKindCluster, None] = None

    def __init__(self, *args):  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
        super().__init__(*args)
        self.result = None
        self._results = []
        self.status = "RUNNING"
        self.start_time = time.time()
        self.teardown_started = False
        self._init_params()
        reuse_cluster_id = self.params.get('reuse_cluster')
        if reuse_cluster_id:
            self.test_config.reuse_cluster(True)
            self.test_config.set_test_id(reuse_cluster_id)
        else:
            # Test id is set by Hydra or generated if running without Hydra
            self.test_config.set_test_id(self.params.get('test_id') or uuid4())
        self.test_config.set_test_name(self.id())
        self.test_config.set_tester_obj(self)
        self._init_logging()
        RemoteCmdRunnerBase.set_default_ssh_transport(self.params.get('ssh_transport'))

        self._profile_factory = None
        if self.params.get('enable_test_profiling'):
            self._profile_factory = ProfilerFactory(os.path.join(self.logdir, 'profile.stats'))
            self._profile_factory.activate()

        ip_ssh_connections = self.params.get(key='ip_ssh_connections')
        self.log.debug("IP used for SSH connections is '%s'",
                       ip_ssh_connections)
        self.test_config.set_ip_ssh_connections(ip_ssh_connections)
        self._duration = self.params.get(key='test_duration')
        post_behavior_db_nodes = self.params.get('post_behavior_db_nodes')
        self.log.debug('Post behavior for db nodes %s', post_behavior_db_nodes)
        self.test_config.keep_cluster(node_type='db_nodes', val=post_behavior_db_nodes)
        post_behavior_monitor_nodes = self.params.get('post_behavior_monitor_nodes')
        self.log.debug('Post behavior for monitor nodes %s', post_behavior_monitor_nodes)
        self.test_config.keep_cluster(node_type='monitor_nodes', val=post_behavior_monitor_nodes)
        post_behavior_loader_nodes = self.params.get('post_behavior_loader_nodes')
        self.log.debug('Post behavior for loader nodes %s', post_behavior_loader_nodes)
        self.test_config.keep_cluster(node_type='loader_nodes', val=post_behavior_loader_nodes)
        self.test_config.set_duration(self._duration)
        cluster_backend = self.params.get('cluster_backend')
        if cluster_backend == 'aws':
            self.test_config.set_multi_region(len(self.params.get('region_name').split()) > 1)
        elif cluster_backend == 'gce':
            self.test_config.set_multi_region(len(self.params.get('gce_datacenter').split()) > 1)

        if self.params.get("backup_bucket_backend") == "azure":
            self.test_config.set_backup_azure_blob_credentials()

        self.test_config.BACKTRACE_DECODING = self.params.get('backtrace_decoding')
        if self.test_config.BACKTRACE_DECODING:
            self.test_config.set_decoding_queue()
        self.test_config.set_intra_node_comm_public(self.params.get(
            'intra_node_comm_public'))

        # for saving test details in DB
        self.create_stats = self.params.get(key='store_perf_results')
        self.scylla_dir = SCYLLA_DIR
        self.left_processes_log = os.path.join(self.logdir, 'left_processes.log')
        self.scylla_hints_dir = os.path.join(self.scylla_dir, "hints")
        self._logs = {}
        self.timeout_thread = None
        self.email_reporter = build_reporter(self.__class__.__name__,
                                             self.params.get('email_recipients'),
                                             self.logdir)
        self.log.info("Initialized Argus TestRun with test id %s", self.argus_test_run.id)
        self._move_kubectl_config()
        self.localhost = self._init_localhost()
        self.test_config.set_rsyslog_imjournal_rate_limit(
            interval=self.params.get("rsyslog_imjournal_rate_limit_interval"),
            burst=self.params.get("rsyslog_imjournal_rate_limit_burst"),
        )
        if self.params.get("logs_transport") == 'rsyslog':
            self.test_config.configure_rsyslog(self.localhost, enable_ngrok=False)
        elif self.params.get("logs_transport") == 'syslog-ng':
            self.test_config.configure_syslogng(self.localhost)

        self.alternator: alternator.api.Alternator = alternator.api.Alternator(sct_params=self.params)
        self.alternator = alternator.api.Alternator(sct_params=self.params)

        if self.params.get("use_ldap"):
            self._init_ldap()

        start_events_device(log_dir=self.logdir, _registry=self.events_processes_registry)
        time.sleep(0.5)
        InfoEvent(message=f"TEST_START test_id={self.test_config.test_id()}").publish()

    @cached_property
    def argus_test_run(self):
        try:
            argus_test_run = ArgusTestRun.from_sct_config(test_id=UUID(self.test_id),
                                                          sct_config=self.params)
            argus_test_run.save()
            return argus_test_run
        except Exception as exc:  # pylint: disable=broad-except
            self.log.warning("Unable to set up Argus connection: %s", exc.args[0])
            self.log.debug("Error details: ", exc_info=True)
            return unittest.mock.MagicMock()

    def argus_update_status(self, status: TestStatus):
        try:
            self.argus_test_run.run_info.results.status = status
            self.argus_test_run.save()
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error saving test status to Argus", exc_info=True)

    def argus_collect_packages(self):
        try:
            self.log.info("Collecting packages for Argus...")
            versions = self.get_scylla_versions()
            kernel_version = self.db_cluster.nodes[0].kernel_version
            kernel_package = PackageVersion(name="kernel", date="", version=kernel_version, revision_id="", build_id="")
            self.argus_test_run.run_info.details.packages.append(kernel_package)
            for package_name, package_info in versions.items():
                package = PackageVersion(name=package_name, date=package_info.get("date", "#NO_DATE"),
                                         version=package_info.get("version", "#NO_VERSION"),
                                         revision_id=package_info.get("commit_id", "#NO_COMMIT"),
                                         build_id=package_info.get("build_id", "#NO_BUILDID"))
                self.argus_test_run.run_info.details.packages.append(package)
            self.log.info("Saving collected packages...")
            self.argus_test_run.save()
        except Exception:  # pylint: disable=broad-except
            self.log.error("Unable to collect package versions for Argus - skipping...", exc_info=True)

    def argus_get_scylla_version(self):
        try:
            self.log.info("Collection Scylla version for argus...")
            version_regex = re.compile(r'(([\w.~]+)-(0.)?([0-9]{8,8}).(\w+).)')
            version_str = self.db_cluster.nodes[0].get_scylla_binary_version()
            if (match := version_regex.match(version_str)):
                self.argus_test_run.run_info.details.scylla_version = match.group(2)
                self.argus_test_run.save()
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error getting scylla version for argus", exc_info=True)

    def argus_finalize_test_run(self):
        try:
            stat_map = {
                "SUCCESS": TestStatus.PASSED,
                "FAILED": TestStatus.FAILED
            }
            self.argus_test_run.run_info.details.set_test_end_time()
            self.argus_update_status(stat_map.get(self.get_test_status(), TestStatus.FAILED))
            last_events = get_events_grouped_by_category(limit=100, _registry=self.events_processes_registry)
            for severity, messages in last_events.items():
                for message in messages:
                    self.argus_test_run.run_info.results.max_stored_events = 100
                    self.argus_test_run.run_info.results.add_event(severity, message)
            self.argus_test_run.save()
            ArgusTestRun.destroy()
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error committing test events to Argus", exc_info=True)

    def argus_collect_logs(self, log_links: dict[str, list[str] | str]):
        try:
            for name, link in log_links.items():
                self.argus_test_run.run_info.logs.add_log(name, link)
            self.argus_test_run.save()
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error saving logs to Argus", exc_info=True)

    def _init_ldap(self):
        self.params['are_ldap_users_on_scylla'] = False

        match self.params.get("ldap_server_type"):
            case LdapServerType.MS_AD:
                self.log.debug("Configuring LDAP for MS Active Directory Services")
                self._init_ldap_ms_ad()
            case LdapServerType.OPENLDAP:
                self.log.debug("Configuring LDAP for OpenLDAP Server")
                self._init_ldap_openldap()
            case _:
                raise LdapConfigurationError("LDAP Configuration requested, but no mode provided")

    def _init_ldap_ms_ad(self):
        ldap_ms_ad_credentials = KeyStore().get_ldap_ms_ad_credentials()
        self.test_config.LDAP_ADDRESS = ldap_ms_ad_credentials["server_address"]

    def _init_ldap_openldap(self):
        self.configure_ldap(node=self.localhost, use_ssl=False)

    def _setup_ldap_roles(self, db_cluster: BaseScyllaCluster):
        self.log.debug("Configuring LDAP Roles.")
        node = db_cluster.nodes[0]
        node.run_cqlsh(f'CREATE ROLE \'{LDAP_ROLE}\' WITH SUPERUSER=true')
        for user in LDAP_USERS:
            node.run_cqlsh(f'CREATE ROLE \'{user}\' WITH login=true')
        node.run_cqlsh(f'ALTER ROLE \'{LDAP_USERS[0]}\' with SUPERUSER=true and password=\'{LDAP_PASSWORD}\'')
        self.params['are_ldap_users_on_scylla'] = True

    def configure_ldap(self, node, use_ssl=False):
        self.test_config.configure_ldap(node=node, use_ssl=use_ssl)
        ldap_username = f'cn=admin,{LDAP_BASE_OBJECT}'
        ldap_role = LDAP_ROLE
        ldap_users = LDAP_USERS.copy()
        ldap_address = list(self.test_config.LDAP_ADDRESS).copy()
        unique_members_list = [f'uid={user},ou=Person,{LDAP_BASE_OBJECT}' for user in ldap_users]
        user_password = LDAP_PASSWORD  # not in use not for authorization, but must be in the config
        role_entry = [
            f'cn={ldap_role},{LDAP_BASE_OBJECT}',
            ['groupOfUniqueNames', 'simpleSecurityObject', 'top'],
            {
                'uniqueMember': unique_members_list,
                'userPassword': user_password
            }
        ]
        self.localhost.add_ldap_entry(ip=ldap_address[0], ldap_port=ldap_address[1],
                                      user=ldap_username, password=LDAP_PASSWORD, ldap_entry=role_entry)

        organizational_unit_entry = [
            f'ou=Person,{LDAP_BASE_OBJECT}',
            ['organizationalUnit', 'top'],
            {
                'ou': 'Person'
            }
        ]
        self.localhost.add_ldap_entry(ip=ldap_address[0], ldap_port=ldap_address[1],
                                      user=ldap_username, password=LDAP_PASSWORD, ldap_entry=organizational_unit_entry)

        # Built-in user also need to be added in ldap server, otherwise it can't log in to create LDAP_USERS
        for user in [self.params.get('authenticator_user')] + LDAP_USERS:
            password = LDAP_PASSWORD if user in LDAP_USERS else self.params.get("authenticator_password")
            user_entry = [
                f'uid={user},ou=Person,{LDAP_BASE_OBJECT}',
                ['uidObject', 'organizationalPerson', 'top'],
                {
                    'userPassword': password,
                    'sn': 'PersonSn',
                    'cn': 'PersonCn'
                }
            ]
            self.localhost.add_ldap_entry(ip=ldap_address[0], ldap_port=ldap_address[1],
                                          user=ldap_username, password=LDAP_PASSWORD, ldap_entry=user_entry)

    def _init_test_timeout_thread(self) -> threading.Timer:
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time + int(self.test_duration) * 60))
        self.log.info('Test start time %s, duration is %s and timeout set to %s',
                      start_time, self.test_duration, end_time)
        self.argus_update_status(TestStatus.RUNNING)

        def kill_the_test(start: float, duration: int):
            TestTimeoutEvent(start_time=start, duration=duration).publish()

        thread = threading.Timer(60 * int(self.test_duration), kill_the_test,
                                 kwargs={'start': self.start_time, 'duration': self.test_duration})
        thread.daemon = True
        thread.start()
        return thread

    def _init_localhost(self):
        return LocalHost(user_prefix=self.params.get("user_prefix"), test_id=self.test_config.test_id())

    def _move_kubectl_config(self):
        secure_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        os.environ['KUBECONFIG'] = self.kubectl_config_path
        if not os.path.exists(self.kubectl_config_path):
            os.makedirs(os.path.dirname(self.kubectl_config_path), exist_ok=True)
            with open(self.kubectl_config_path, 'w', encoding="utf-8") as kube_config_file:
                kube_config_file.write('')
                kube_config_file.flush()
        os.chmod(os.path.dirname(self.kubectl_config_path), mode=secure_mode)
        os.chmod(self.kubectl_config_path, mode=secure_mode)
        os.environ['HELM_CONFIG_HOME'] = self.helm_config_path
        if not os.path.exists(self.helm_config_path):
            os.makedirs(os.path.dirname(self.helm_config_path), exist_ok=True)
        os.chmod(os.path.dirname(self.helm_config_path), mode=secure_mode)

    @cached_property
    def kubectl_config_path(self):
        return os.path.join(os.path.expanduser(self.logdir), '.kube/config')

    @cached_property
    def helm_config_path(self):
        return os.path.join(os.path.expanduser(self.logdir), '.helm')

    def _init_params(self):
        self.params = init_and_verify_sct_config()

    def _init_logging(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.logdir = self.test_config.logdir()

    def run(self, result=None):
        self.result = self.defaultTestResult() if result is None else result
        result = super().run(self.result)
        return result

    @property
    def latency_results_file(self):
        return TestConfig.latency_results_file()

    @property
    def reliable_replication_factor(self) -> int:
        """
        Calculates reliable replication factor for tables to have for a test.
        Needed in a case when you want to create a table to run queries on it,
          but don't know what is the replication factor to set to make sure that
          queries won't fail due to the lack of nodes in the cluster.
        :return:
        """
        n_db_nodes = str(self.params.get('n_db_nodes'))
        return min([int(nodes_num) for nodes_num in n_db_nodes.split() if int(nodes_num) > 0])

    @property
    def test_id(self):
        return TestConfig.test_id()

    @property
    def test_duration(self):
        return self._duration

    def get_duration(self, duration: int):
        """Calculate duration based on test_duration

        Calculate duration for stress threads

        Arguments:
            duration {int} -- time duration in minutes

        Returns:
            int -- time duration in seconds
        """
        if not duration:
            duration = self.test_duration
        return duration * 60 + 600

    @staticmethod
    def legacy_init_nodes(db_cluster):
        db_cluster.set_seeds()

        # Init seed nodes
        db_cluster.wait_for_init(node_list=db_cluster.seed_nodes)

        # Init non-seed nodes
        if db_cluster.non_seed_nodes:
            db_cluster.wait_for_init(node_list=db_cluster.non_seed_nodes)

    @staticmethod
    def init_nodes(db_cluster):
        db_cluster.set_seeds(first_only=True)
        db_cluster.wait_for_init(node_list=db_cluster.nodes)
        db_cluster.set_seeds()
        db_cluster.update_seed_provider()

    @staticmethod
    def update_certificates():
        update_certificates()

    def get_event_summary(self) -> dict:
        return get_logger_event_summary(_registry=self.events_processes_registry)

    def get_test_status(self) -> str:
        summary = self.get_event_summary()
        if summary.get('ERROR', 0) or summary.get('CRITICAL', 0):
            return 'FAILED'
        return 'SUCCESS'

    def kill_test(self, backtrace_with_reason):
        test_pid = os.getpid()
        self.result.addFailure(self.test_config.tester_obj(), backtrace_with_reason)
        os.kill(test_pid, signal.SIGUSR2)

    def download_db_packages(self):
        # download rpms for update_db_packages
        self.params['update_db_packages'] = download_dir_from_cloud(self.params.get('update_db_packages'))

    def download_encrypt_keys(self):  # pylint: disable=no-self-use
        download_encrypt_keys()

    @property
    def is_encrypt_keys_needed(self):
        append_scylla_yaml = self.params.get('append_scylla_yaml')
        return append_scylla_yaml and (
            'system_key_directory' in append_scylla_yaml or
            'system_info_encryption' in append_scylla_yaml or
            'kmip_hosts:' in append_scylla_yaml
        )

    @teardown_on_exception
    @log_run_info
    def setUp(self):  # pylint: disable=too-many-branches,too-many-statements
        self.credentials = []
        self.db_cluster = None

        # NOTE: following are used only on K8S for testing multi-tenancy case
        self.db_clusters_multitenant: list[BaseScyllaCluster] = []
        self.loaders_multitenant = []
        self.monitors_multitenant = []
        self.prometheus_db_multitenant = []

        self.cs_db_cluster = None
        self.loaders = None
        self.monitors = None
        self.siren_manager = None
        self.k8s_cluster = None
        self.connections = []
        make_threads_be_daemonic_by_default()
        self.update_certificates()

        # download rpms for update_db_packages
        if self.params.get('update_db_packages'):
            self.download_db_packages()
        if self.is_encrypt_keys_needed:
            self.download_encrypt_keys()

        self.init_resources()

        if self.db_cluster and not self.db_clusters_multitenant:
            self.db_clusters_multitenant = [self.db_cluster]
        for db_cluster in self.db_clusters_multitenant:
            if not (db_cluster and db_cluster.nodes):
                continue
            if self.params.get('use_legacy_cluster_init'):
                self.legacy_init_nodes(db_cluster=db_cluster)
            else:
                self.init_nodes(db_cluster=db_cluster)

            if self.params.get('use_ldap'):
                with temp_authenticator(db_cluster.nodes[0], "org.apache.cassandra.auth.PasswordAuthenticator"):
                    # running `set_system_auth_rf()` before changing authorization/authentication protocols
                    self.set_system_auth_rf(db_cluster=db_cluster)

                    self._setup_ldap_roles(db_cluster=db_cluster)
                    if self.params.get('ldap_server_type') == LdapServerType.MS_AD:
                        change_default_password(node=db_cluster.nodes[0],
                                                user=self.params.get('authenticator_user'),
                                                password=self.params.get('authenticator_password'))
                        # TODO: update proper loader and monitor in multi-tenant case.
                        #       Now it updates only the first ones all the time.
                        self.loaders.added_password_suffix = True
                        # TODO: Replace with strong password generation
                        self.monitors.added_password_suffix = True
            else:
                self.set_system_auth_rf(db_cluster=db_cluster)

            db_node_address = db_cluster.nodes[0].ip_address
            if self.loaders and not self.loaders_multitenant:
                self.loaders_multitenant = [self.loaders]
            for loaders in self.loaders_multitenant:
                if loaders:
                    loaders.wait_for_init(db_node_address=db_node_address)

        # cs_db_cluster is created in case MIXED_CLUSTER. For example, gemini test
        if self.cs_db_cluster:
            init_value = self.params["use_mgmt"]
            self.params["use_mgmt"] = False
            self.init_nodes(db_cluster=self.cs_db_cluster)
            self.params["use_mgmt"] = init_value

        if self.create_stats:
            self.create_test_stats()
            # sync test_start_time with ES
            self.start_time = self.get_test_start_time()

        if self.monitors and not self.monitors_multitenant:
            self.monitors_multitenant = [self.monitors]
        for monitors in self.monitors_multitenant:
            monitors.wait_for_init()

        self.argus_collect_packages()
        self.argus_get_scylla_version()

        # cancel reuse cluster - for new nodes added during the test
        self.test_config.reuse_cluster(False)

        for monitors in self.monitors_multitenant:
            if monitors and monitors.nodes:
                self.prometheus_db_multitenant.append(
                    PrometheusDBStats(host=monitors.nodes[0].external_address))
        self.prometheus_db = (self.prometheus_db_multitenant or [None])[0]

        self.start_time = time.time()
        self.timeout_thread = self._init_test_timeout_thread()

        for db_cluster in self.db_clusters_multitenant:
            if db_cluster:
                db_cluster.validate_seeds_on_all_nodes()

    def set_system_auth_rf(self, db_cluster=None):
        if not db_cluster:
            db_cluster = self.db_cluster
        # No need to change system tables when running via scylla-cloud
        # Also, when running a Alternator via scylla-cloud, we don't have CQL access to the cluster
        if self.params.get('db_type') == 'cloud_scylla':
            # TODO: move this skip to siren-tools when possible
            self.log.warning("Skipping this function due this job run from Siren cloud!")
            return
        # change RF of system_auth
        system_auth_rf = self.params.get('system_auth_rf')
        if system_auth_rf > 1 and not self.test_config.REUSE_CLUSTER:
            self.log.info('change RF of system_auth to %s', system_auth_rf)
            node = db_cluster.nodes[0]
            credentials = db_cluster.get_db_auth()
            username, password = credentials if credentials else (None, None)
            with db_cluster.cql_connection_patient(node, user=username, password=password) as session:
                session.execute("ALTER KEYSPACE system_auth WITH replication = "
                                "{'class': 'org.apache.cassandra.locator.SimpleStrategy', "
                                "'replication_factor': %s};" % system_auth_rf)
            self.log.info('repair system_auth keyspace ...')
            node.run_nodetool(sub_cmd="repair", args="-- system_auth")

    @cache
    def pre_create_alternator_tables(self):
        node = self.db_cluster.nodes[0]
        if self.params.get('alternator_port'):
            self.log.info("Going to create alternator tables")
            if self.params.get('alternator_enforce_authorization'):
                with self.db_cluster.cql_connection_patient(self.db_cluster.nodes[0]) as session:
                    session.execute("""
                        INSERT INTO system_auth.roles (role, salted_hash) VALUES (%s, %s)
                    """, (self.params.get('alternator_access_key_id'),
                          self.params.get('alternator_secret_access_key')))

            schema = self.params.get("dynamodb_primarykey_type")
            self.alternator.create_table(node=node, schema=schema)
            self.alternator.create_table(node=node, schema=schema, isolation=alternator.enums.WriteIsolation.FORBID_RMW,
                                         table_name='usertable_no_lwt')

    def get_nemesis_class(self):
        """
        Get a Nemesis class from parameters.

        :return: Nemesis class.
        :rtype: nemesis.Nemesis derived class
        """
        nemesis_threads = []
        list_class_name = self.params.get('nemesis_class_name')
        for klass in list_class_name.split(' '):
            try:
                nemesis_name, num = klass.strip().split(':')
                nemesis_name = nemesis_name.strip()
                num = num.strip()

            except ValueError:
                nemesis_name = klass.split(':')[0]
                num = 1
            nemesis_threads.append({'nemesis': getattr(nemesis, nemesis_name), 'num_threads': int(num)})

        return nemesis_threads

    def get_cluster_gce(self, loader_info, db_info, monitor_info):
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        if loader_info['n_nodes'] is None:
            loader_info['n_nodes'] = int(self.params.get('n_loaders'))
        if loader_info['type'] is None:
            loader_info['type'] = self.params.get('gce_instance_type_loader')
        if loader_info['root_disk_type'] is None:
            loader_info['root_disk_type'] = self.params.get('gce_root_disk_type_loader')
        if loader_info['disk_size'] is None:
            loader_info['disk_size'] = self.params.get('root_disk_size_loader')
        if loader_info['n_local_ssd'] is None:
            loader_info['n_local_ssd'] = self.params.get('gce_n_local_ssd_disk_loader')
        if db_info['n_nodes'] is None:
            n_db_nodes = self.params.get('n_db_nodes')
            if isinstance(n_db_nodes, int):  # legacy type
                db_info['n_nodes'] = [n_db_nodes]
            elif isinstance(n_db_nodes, str):  # latest type to support multiple datacenters
                db_info['n_nodes'] = [int(n) for n in n_db_nodes.split()]
            else:
                self.fail('Unsupported parameter type: {}'.format(type(n_db_nodes)))
        cpu = self.params.get('gce_instance_type_cpu_db')
        # unit is GB
        mem = self.params.get('gce_instance_type_mem_db')
        if cpu and mem:
            db_info['type'] = 'custom-{}-{}-ext'.format(cpu, int(mem) * 1024)
        if db_info['type'] is None:
            db_info['type'] = self.params.get('gce_instance_type_db')
        if db_info['root_disk_type'] is None:
            db_info['root_disk_type'] = self.params.get('gce_root_disk_type_db')
        if db_info['disk_size'] is None:
            db_info['disk_size'] = self.params.get('root_disk_size_db')
        if db_info['n_local_ssd'] is None:
            db_info['n_local_ssd'] = self.params.get('gce_n_local_ssd_disk_db')
        if monitor_info['n_nodes'] is None:
            monitor_info['n_nodes'] = self.params.get('n_monitor_nodes')
        if monitor_info['type'] is None:
            monitor_info['type'] = self.params.get('gce_instance_type_monitor')
        if monitor_info['root_disk_type'] is None:
            monitor_info['root_disk_type'] = self.params.get('gce_root_disk_type_monitor')
        if monitor_info['disk_size'] is None:
            monitor_info['disk_size'] = self.params.get('root_disk_size_monitor')
        if monitor_info['n_local_ssd'] is None:
            monitor_info['n_local_ssd'] = self.params.get('gce_n_local_ssd_disk_monitor')

        user_prefix = self.params.get('user_prefix')

        services = get_gce_services(self.params.get('gce_datacenter').split())
        assert len(services) in (1, len(db_info['n_nodes']),)
        gce_datacenter, services = list(services.keys()), list(services.values())
        TEST_LOG.info("Using GCE AZs: %s", gce_datacenter)

        user_credentials = self.params.get('user_credentials_path')
        self.credentials.append(UserRemoteCredentials(key_file=user_credentials))

        gce_image_db = self.params.get('gce_image_db')
        if not gce_image_db:
            gce_image_db = self.params.get('gce_image')
        gce_image_monitor = self.params.get('gce_image_monitor')
        if not gce_image_monitor:
            gce_image_monitor = self.params.get('gce_image')
        cluster_additional_disks = {'pd-ssd': self.params.get('gce_pd_ssd_disk_size_db'),
                                    'pd-standard': self.params.get('gce_pd_standard_disk_size_db')}

        service_accounts = KeyStore().get_gcp_service_accounts()
        db_type = self.params.get('db_type')
        common_params = dict(gce_image_username=self.params.get('gce_image_username'),
                             gce_network=self.params.get('gce_network'),
                             credentials=self.credentials,
                             user_prefix=user_prefix,
                             params=self.params,
                             gce_datacenter=gce_datacenter,
                             )

        # create provisioners
        gce_provisioners = []
        for region in gce_datacenter:
            gce_provisioners.append(
                provisioner_factory.create_provisioner(backend="gce",
                                                       test_id=self.test_id,
                                                       region=region,
                                                       params=self.params)
            )

        if db_type == 'cloud_scylla':
            cloud_credentials = self.params.get('cloud_credentials_path')

            credentials = [UserRemoteCredentials(key_file=cloud_credentials)]
            params = dict(
                n_nodes=[0],
                user_prefix=self.params.get('user_prefix'),
                credentials=credentials,
                params=self.params,
            )
            if not cluster_cloud:
                raise ImportError(f"cluster_cloud isn't installed. {CLUSTER_CLOUD_IMPORT_ERROR}")
            self.db_cluster = cluster_cloud.ScyllaCloudCluster(**params)
        else:
            self.db_cluster = ScyllaGCECluster(gce_image=gce_image_db,
                                               gce_image_type=db_info['root_disk_type'],
                                               gce_image_size=db_info['disk_size'],
                                               gce_n_local_ssd=db_info['n_local_ssd'],
                                               gce_instance_type=db_info['type'],
                                               services=services,
                                               provisioners=gce_provisioners,
                                               n_nodes=db_info['n_nodes'],
                                               add_disks=cluster_additional_disks,
                                               service_accounts=service_accounts,
                                               **common_params)

        loader_additional_disks = {'pd-ssd': self.params.get('gce_pd_ssd_disk_size_loader')}
        self.loaders = LoaderSetGCE(gce_image=self.params.get('gce_image_loader'),
                                    gce_image_type=loader_info['root_disk_type'],
                                    gce_image_size=loader_info['disk_size'],
                                    gce_n_local_ssd=loader_info['n_local_ssd'],
                                    gce_instance_type=loader_info['type'],
                                    service=services[:1],
                                    provisioners=gce_provisioners,
                                    n_nodes=loader_info['n_nodes'],
                                    add_disks=loader_additional_disks,
                                    **common_params)

        if monitor_info['n_nodes'] > 0:
            monitor_additional_disks = {'pd-ssd': self.params.get('gce_pd_ssd_disk_size_monitor')}
            self.monitors = MonitorSetGCE(gce_image=gce_image_monitor,
                                          gce_image_type=monitor_info['root_disk_type'],
                                          gce_image_size=monitor_info['disk_size'],
                                          gce_n_local_ssd=monitor_info['n_local_ssd'],
                                          gce_instance_type=monitor_info['type'],
                                          service=services[:1],
                                          provisioners=gce_provisioners,
                                          n_nodes=monitor_info['n_nodes'],
                                          add_disks=monitor_additional_disks,
                                          targets=dict(db_cluster=self.db_cluster,
                                                       loaders=self.loaders),
                                          **common_params)
        else:
            self.monitors = NoMonitorSet()

    def get_cluster_azure(self, loader_info, db_info, monitor_info):
        # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        regions = self.params.get('azure_region_name')
        test_id = str(TestConfig().test_id())
        provisioners: List[AzureProvisioner] = []
        for region in regions:
            provisioners.append(provisioner_factory.create_provisioner(backend="azure", test_id=test_id,
                                region=region))
        if db_info['n_nodes'] is None:
            n_db_nodes = self.params.get('n_db_nodes')
            if isinstance(n_db_nodes, int):  # legacy type
                db_info['n_nodes'] = [n_db_nodes]
            elif isinstance(n_db_nodes, str):  # latest type to support multiple datacenters
                db_info['n_nodes'] = [int(n) for n in n_db_nodes.split()]
            else:
                self.fail('Unsupported parameter type: {}'.format(type(n_db_nodes)))
        db_info['type'] = self.params.get('azure_instance_type_db')
        if loader_info['n_nodes'] is None:
            n_loader_nodes = self.params.get('n_loaders')
            if isinstance(n_loader_nodes, int):  # legacy type
                loader_info['n_nodes'] = [n_loader_nodes]
            elif isinstance(n_loader_nodes, str):  # latest type to support multiple datacenters
                loader_info['n_nodes'] = [int(n) for n in n_loader_nodes.split()]
            else:
                self.fail('Unsupported parameter type: {}'.format(type(n_loader_nodes)))
        azure_image = self.params.get("azure_image_db")
        user_prefix = self.params.get('user_prefix')
        self.credentials.append(UserRemoteCredentials(key_file="~/.ssh/scylla-test"))

        common_params = dict(
            credentials=self.credentials,
            user_prefix=user_prefix,
            params=self.params,
            region_names=None,
        )
        self.db_cluster = ScyllaAzureCluster(image_id=azure_image,
                                             root_disk_size=db_info['disk_size'],
                                             instance_type=db_info['type'],
                                             provisioners=provisioners,
                                             n_nodes=db_info['n_nodes'],
                                             user_name=self.params.get('azure_image_username'),
                                             **common_params)
        self.loaders = LoaderSetAzure(
            image_id=self.params.get('azure_image_loader'),
            root_disk_size=loader_info['disk_size'],
            instance_type="Standard_D2s_v3",
            provisioners=provisioners,
            n_nodes=loader_info['n_nodes'],
            user_name=self.params.get('ami_loader_user'),
            **common_params)
        if monitor_info['n_nodes'] is None:
            monitor_info['n_nodes'] = self.params.get('n_monitor_nodes')
        if monitor_info['n_nodes'] > 0:
            azure_image_monitor = self.params.get('azure_image_monitor')
            if not azure_image_monitor:
                azure_image_monitor = self.params.get('azure_image')
            self.monitors = MonitorSetAzure(image_id=azure_image_monitor,
                                            instance_type=self.params.get('azure_instance_type_monitor'),
                                            root_disk_size=self.params.get('azure_root_disk_type_monitor'),
                                            provisioners=provisioners,
                                            n_nodes=self.params.get('n_monitor_nodes'),
                                            targets=dict(db_cluster=self.db_cluster,
                                                         loaders=self.loaders),
                                            user_name=self.params.get('ami_monitor_user'),
                                            **common_params)
        else:
            self.monitors = NoMonitorSet()

    def get_cluster_aws(self, loader_info, db_info, monitor_info):
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        regions = self.params.get('region_name').split()

        if loader_info['n_nodes'] is None:
            n_loader_nodes = self.params.get('n_loaders')
            if isinstance(n_loader_nodes, int):  # legacy type
                loader_info['n_nodes'] = [n_loader_nodes]
            elif isinstance(n_loader_nodes, str):  # latest type to support multiple datacenters
                loader_info['n_nodes'] = [int(n) for n in n_loader_nodes.split()]
            else:
                self.fail('Unsupported parameter type: {}'.format(type(n_loader_nodes)))
        if loader_info['type'] is None:
            loader_info['type'] = self.params.get('instance_type_loader')
        if loader_info['disk_size'] is None:
            loader_info['disk_size'] = self.params.get('root_disk_size_loader')
        if loader_info['device_mappings'] is None:
            if loader_info['disk_size']:
                loader_info['device_mappings'] = [{
                    "DeviceName": ec2_ami_get_root_device_name(image_id=self.params.get('ami_id_loader').split()[0],
                                                               region=regions[0]),
                    "Ebs": {
                        "VolumeSize": loader_info['disk_size'],
                        "VolumeType": "gp2"
                    }
                }]
            else:
                loader_info['device_mappings'] = []

        init_db_info_from_params(db_info, params=self.params, regions=regions)
        init_monitoring_info_from_params(monitor_info, params=self.params, regions=regions)
        user_prefix = self.params.get('user_prefix')

        user_credentials = self.params.get('user_credentials_path')

        regions = self.params.get('region_name').split()
        services = get_ec2_services(regions)

        for _ in regions:
            self.credentials.append(UserRemoteCredentials(key_file=user_credentials))

        ami_ids = self.params.get('ami_id_db_scylla').split()
        for idx, ami_id in enumerate(ami_ids):
            wait_ami_available(services[idx].meta.client, ami_id)

        common_params = get_common_params(params=self.params, regions=regions, credentials=self.credentials,
                                          services=services)

        def create_cluster(db_type='scylla'):
            cl_params = dict(
                ec2_instance_type=db_info['type'],
                ec2_block_device_mappings=db_info['device_mappings'],
                n_nodes=db_info['n_nodes']
            )
            cl_params.update(common_params)
            if db_type == 'scylla':
                return ScyllaAWSCluster(
                    ec2_ami_id=self.params.get('ami_id_db_scylla').split(),
                    ec2_ami_username=self.params.get('ami_db_scylla_user'),
                    **cl_params)
            elif db_type == 'cassandra':
                return CassandraAWSCluster(
                    ec2_ami_id=self.params.get('ami_id_db_cassandra').split(),
                    ec2_ami_username=self.params.get('ami_db_cassandra_user'),
                    **cl_params)
            elif db_type == 'mixed_scylla':
                self.test_config.mixed_cluster(True)
                return ScyllaAWSCluster(
                    ec2_ami_id=self.params.get('ami_id_db_oracle').split(),
                    ec2_ami_username=self.params.get('ami_db_scylla_user'),
                    ec2_instance_type=self.params.get('instance_type_db_oracle'),
                    ec2_block_device_mappings=db_info['device_mappings'],
                    n_nodes=[self.params.get('n_test_oracle_db_nodes')],
                    node_type='oracle-db',
                    **(common_params | {'user_prefix': user_prefix + '-oracle'}),
                )
            elif db_type == 'cloud_scylla':
                cloud_credentials = self.params.get('cloud_credentials_path')

                credentials = [UserRemoteCredentials(key_file=cloud_credentials)]
                params = dict(
                    n_nodes=[0],
                    user_prefix=self.params.get('user_prefix'),
                    credentials=credentials,
                    params=self.params,
                )
                if not cluster_cloud:
                    raise ImportError(f"cluster_cloud isn't installed. {CLUSTER_CLOUD_IMPORT_ERROR}")
                return cluster_cloud.ScyllaCloudCluster(**params)
            return None

        db_type = self.params.get('db_type')
        if db_type in ('scylla', 'cassandra'):
            self.db_cluster = create_cluster(db_type)
        elif db_type == 'mixed':
            self.db_cluster = create_cluster('scylla')
            self.cs_db_cluster = create_cluster('cassandra')
        elif db_type == 'mixed_scylla':
            self.db_cluster = create_cluster('scylla')
            self.cs_db_cluster = create_cluster('mixed_scylla')
        elif db_type == 'cloud_scylla':
            self.db_cluster = create_cluster('cloud_scylla')
        else:
            self.log.error('Incorrect parameter db_type: %s',
                           self.params.get('db_type'))

        self.loaders = LoaderSetAWS(
            ec2_ami_id=self.params.get('ami_id_loader').split(),
            ec2_ami_username=self.params.get('ami_loader_user'),
            ec2_instance_type=loader_info['type'],
            ec2_block_device_mappings=loader_info['device_mappings'],
            n_nodes=loader_info['n_nodes'],
            **common_params)

        if monitor_info['n_nodes'] > 0:
            self.monitors = MonitorSetAWS(
                ec2_ami_id=self.params.get('ami_id_monitor').split(),
                ec2_ami_username=self.params.get('ami_monitor_user'),
                ec2_instance_type=monitor_info['type'],
                ec2_block_device_mappings=monitor_info['device_mappings'],
                n_nodes=monitor_info['n_nodes'],
                targets=dict(db_cluster=self.db_cluster,
                             loaders=self.loaders),
                **common_params)
        else:
            self.monitors = NoMonitorSet()

    def get_cluster_docker(self):
        self.credentials.append(UserRemoteCredentials(key_file=self.params.get('user_credentials_path')))

        container_node_params = dict(docker_image=self.params.get('docker_image'),
                                     docker_image_tag=self.params.get('scylla_version'),
                                     node_key_file=self.credentials[0].key_file)
        common_params = dict(user_prefix=self.params.get('user_prefix'),
                             params=self.params)

        self.db_cluster = cluster_docker.ScyllaDockerCluster(n_nodes=[self.params.get("n_db_nodes"), ],
                                                             **container_node_params, **common_params)
        self.loaders = cluster_docker.LoaderSetDocker(n_nodes=self.params.get("n_loaders"),
                                                      **container_node_params, **common_params)
        self.monitors = cluster_docker.MonitorSetDocker(n_nodes=self.params.get("n_monitor_nodes"),
                                                        targets=dict(db_cluster=self.db_cluster,
                                                                     loaders=self.loaders),
                                                        **common_params)

    def get_cluster_baremetal(self):
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches

        user_credentials = self.params.get('user_credentials_path')
        self.credentials.append(UserRemoteCredentials(key_file=user_credentials))
        params = dict(
            n_nodes=[self.params.get('n_db_nodes')],
            public_ips=self.params.get('db_nodes_public_ip'),
            private_ips=self.params.get('db_nodes_private_ip'),
            user_prefix=self.params.get('user_prefix'),
            credentials=self.credentials,
            params=self.params,
            targets=dict(db_cluster=self.db_cluster, loaders=self.loaders),
        )
        self.db_cluster = cluster_baremetal.ScyllaPhysicalCluster(**params)

        params['n_nodes'] = int(self.params.get('n_loaders'))
        params['public_ips'] = self.params.get('loaders_public_ip')
        params['private_ips'] = self.params.get('loaders_private_ip')
        self.loaders = cluster_baremetal.LoaderSetPhysical(**params)

        params['n_nodes'] = self.params.get('n_monitor_nodes')
        params['public_ips'] = self.params.get('monitor_nodes_public_ip')
        params['private_ips'] = self.params.get('monitor_nodes_private_ip')
        self.monitors = cluster_baremetal.MonitorSetPhysical(**params)

    def get_cluster_k8s_local_kind_cluster(self):
        self.credentials.append(UserRemoteCredentials(key_file=self.params.get('user_credentials_path')))

        container_node_params = dict(
            docker_image=self.params.get('docker_image'),
            docker_image_tag=self.params.get('scylla_version'),
            node_key_file=self.credentials[0].key_file,
        )
        self.k8s_cluster = mini_k8s.LocalKindCluster(
            software_version=self.params.get("mini_k8s_version"),
            user_prefix=self.params.get("user_prefix"),
            params=self.params,
        )
        self.k8s_cluster.deploy()
        for namespace in ('kube-system', 'local-path-storage'):
            self.k8s_cluster.set_nodeselector_for_deployments(
                pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME, namespace=namespace)

        scylla_pool = mini_k8s.MinimalK8SNodePool(
            k8s_cluster=self.k8s_cluster,
            name=self.k8s_cluster.SCYLLA_POOL_NAME,
            num_nodes=self.params.get("n_db_nodes"),
            image_type="fake-image-type",
            instance_type="fake-instance-type")
        self.k8s_cluster.deploy_node_pool(scylla_pool, wait_till_ready=False)
        self.k8s_cluster.install_static_local_volume_provisioner(node_pool=scylla_pool)

        self.k8s_cluster.deploy_cert_manager(pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME)
        self.k8s_cluster.deploy_scylla_operator(pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME)
        if self.params.get('use_mgmt'):
            self.k8s_cluster.deploy_scylla_manager(pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME)

        self.db_cluster = mini_k8s.LocalMinimalScyllaPodCluster(
            k8s_cluster=self.k8s_cluster,
            scylla_cluster_name=self.params.get("k8s_scylla_cluster_name"),
            user_prefix=self.params.get("user_prefix"),
            n_nodes=self.params.get("k8s_n_scylla_pods_per_cluster") or self.params.get("n_db_nodes"),
            params=self.params,
            node_pool=scylla_pool,
        )

        if self.params.get('k8s_deploy_monitoring'):
            self.k8s_cluster.deploy_monitoring_cluster(
                is_manager_deployed=self.params.get('use_mgmt')
            )
        if self.params.get("n_loaders"):
            self.loaders = cluster_docker.LoaderSetDocker(
                **container_node_params,
                n_nodes=self.params.get("n_loaders"),
                user_prefix=self.params.get("user_prefix"),
                params=self.params,
            )

        if self.params.get("n_monitor_nodes") > 0:
            self.monitors = cluster_docker.MonitorSetDocker(
                n_nodes=self.params.get("n_monitor_nodes"),
                targets=dict(
                    db_cluster=self.db_cluster,
                    loaders=self.loaders),
                user_prefix=self.params.get("user_prefix"),
                params=self.params,
            )
        else:
            self.monitors = NoMonitorSet()

    @staticmethod
    def _add_and_wait_for_cluster_nodes_in_parallel(clusters):
        def _add_and_wait_for_cluster_nodes(cluster):
            if cluster.node_type == 'loader':
                count = cluster.params.get("k8s_n_loader_pods_per_cluster") or cluster.params.get("n_loaders")
            elif cluster.node_type == 'monitor':
                count = 1
            else:
                count = cluster.params.get("k8s_n_scylla_pods_per_cluster") or cluster.params.get("n_db_nodes")
            cluster.add_nodes(count=count, enable_auto_bootstrap=cluster.auto_bootstrap)

        add_nodes_in_parallel = ParallelObject(
            timeout=3600, objects=[[cluster] for cluster in clusters], num_workers=len(clusters))
        add_nodes_in_parallel.run(
            func=_add_and_wait_for_cluster_nodes,
            unpack_objects=True, ignore_exceptions=False)

    def get_cluster_k8s_gke(self):
        self.credentials.append(UserRemoteCredentials(key_file=self.params.get('user_credentials_path')))

        services = get_gce_services(self.params.get('gce_datacenter').split())
        assert len(services) == 1, "Doesn't support multi DC setup for `k8s-gke' backend"

        services, gce_datacenter = list(services.values()), list(services.keys())
        self.k8s_cluster = gke.GkeCluster(
            gke_cluster_version=self.params.get("gke_cluster_version"),
            gke_k8s_release_channel=self.params.get("gke_k8s_release_channel"),
            gce_disk_size=self.params.get("root_disk_size_db"),
            gce_disk_type=self.params.get("gce_root_disk_type_db"),
            gce_network=self.params.get("gce_network"),
            services=services,
            user_prefix=self.params.get("user_prefix"),
            gce_instance_type='n1-standard-2',
            n_nodes=2,
            params=self.params,
            gce_datacenter=gce_datacenter,
        )
        self.k8s_cluster.deploy()

        # NOTE: between GKE cluster creation and addition of new node pools we need
        # several minutes gap to avoid "repair" status of a cluster when API server goes down.
        # So, deploy apps specific to default-pool in between above mentioned deployment steps.
        self.k8s_cluster.set_nodeselector_for_deployments(
            pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME, namespace="kube-system")
        self.k8s_cluster.deploy_cert_manager(pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME)
        self.k8s_cluster.deploy_scylla_operator()
        if self.params.get('use_mgmt'):
            self.k8s_cluster.deploy_scylla_manager(pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME)

        loader_pool = None
        if self.params.get("n_loaders"):
            loader_pool = gke.GkeNodePool(
                name=self.k8s_cluster.LOADER_POOL_NAME,
                instance_type=self.params.get("gce_instance_type_loader"),
                num_nodes=self.params.get("n_loaders"),
                k8s_cluster=self.k8s_cluster)
            self.k8s_cluster.deploy_node_pool(loader_pool, wait_till_ready=False)

        monitor_pool = None
        if self.params.get('k8s_deploy_monitoring'):
            monitor_pool = gke.GkeNodePool(
                name=self.k8s_cluster.MONITORING_POOL_NAME,
                local_ssd_count=self.params.get("gce_n_local_ssd_disk_monitor"),
                disk_size=self.params.get("root_disk_size_monitor"),
                disk_type=self.params.get("gce_root_disk_type_monitor"),
                instance_type=self.params.get("gce_instance_type_monitor"),
                num_nodes=1,
                k8s_cluster=self.k8s_cluster)
            self.k8s_cluster.deploy_node_pool(monitor_pool, wait_till_ready=False)

        scylla_pool = gke.GkeNodePool(
            name=self.k8s_cluster.SCYLLA_POOL_NAME,
            local_ssd_count=self.params.get("gce_n_local_ssd_disk_db"),
            disk_size=self.params.get("root_disk_size_db"),
            disk_type=self.params.get("gce_root_disk_type_db"),
            instance_type=self.params.get("gce_instance_type_db"),
            num_nodes=self.params.get("n_db_nodes"),
            k8s_cluster=self.k8s_cluster)
        self.k8s_cluster.deploy_node_pool(scylla_pool, wait_till_ready=False)

        self.k8s_cluster.wait_all_node_pools_to_be_ready()
        self.k8s_cluster.prepare_k8s_scylla_nodes(node_pool=scylla_pool)

        for i in range(self.k8s_cluster.tenants_number):
            self.db_clusters_multitenant.append(gke.GkeScyllaPodCluster(
                k8s_cluster=self.k8s_cluster,
                scylla_cluster_name=self.params.get("k8s_scylla_cluster_name") + (f"-{i + 1}" if i else ""),
                user_prefix=self.params.get("user_prefix"),
                n_nodes=self.params.get("k8s_n_scylla_pods_per_cluster") or self.params.get("n_db_nodes"),
                params=self.params,
                node_pool=scylla_pool,
                add_nodes=False,
            ))
        self.db_cluster = self.db_clusters_multitenant[0]

        if self.params.get('k8s_deploy_monitoring'):
            self.k8s_cluster.deploy_monitoring_cluster(
                is_manager_deployed=self.params.get('use_mgmt'),
                node_pool=monitor_pool
            )

        if self.params.get("n_loaders"):
            for i in range(self.k8s_cluster.tenants_number):
                self.loaders_multitenant.append(cluster_k8s.LoaderPodCluster(
                    k8s_cluster=self.k8s_cluster,
                    loader_cluster_config=LOADER_CLUSTER_CONFIG,
                    loader_cluster_name=self.params.get("k8s_loader_cluster_name") + (f"-{i + 1}" if i else ""),
                    user_prefix=(f"{i + 1}-" if i else "") + self.params.get("user_prefix"),
                    n_nodes=self.params.get("k8s_n_loader_pods_per_cluster") or self.params.get("n_loaders"),
                    params=self.params,
                    node_pool=loader_pool,
                    add_nodes=False,
                ))
            self.loaders = self.loaders_multitenant[0]

        # NOTE: wait for Scylla and loader clusters nodes creation
        self._add_and_wait_for_cluster_nodes_in_parallel(
            self.db_clusters_multitenant + self.loaders_multitenant)

        if self.params.get("n_monitor_nodes") > 0:
            for i in range(self.k8s_cluster.tenants_number):
                self.log.debug("Create monitor for the DB cluster №%s", i + 1)
                self.monitors_multitenant.append(gke.MonitorSetGKE(
                    gce_image=self.params.get("gce_image_monitor"),
                    gce_image_username=self.params.get("gce_image_username"),
                    gce_image_type=self.params.get("gce_root_disk_type_monitor"),
                    gce_image_size=self.params.get('root_disk_size_monitor'),
                    gce_network=self.params.get("gce_network"),
                    gce_instance_type=self.params.get("gce_instance_type_monitor"),
                    gce_n_local_ssd=self.params.get("gce_n_local_ssd_disk_monitor"),
                    gce_datacenter=gce_datacenter,
                    service=services[:1],
                    credentials=self.credentials,
                    user_prefix=(f"{i + 1}-" if i else "") + self.params.get("user_prefix"),
                    n_nodes=self.params.get('n_monitor_nodes'),
                    targets={
                        "db_cluster": self.db_clusters_multitenant[i],
                        "loaders": self.loaders_multitenant[i],
                    },
                    params=self.params,
                    add_nodes=False,
                    monitor_id=self.test_config.test_id() + (f"-{i + 1}" if i else ""),
                ))
            self.monitors = self.monitors_multitenant[0]
            self._add_and_wait_for_cluster_nodes_in_parallel(self.monitors_multitenant)
        else:
            self.monitors = NoMonitorSet()
            self.monitors_multitenant = [self.monitors]

    def get_cluster_k8s_eks(self):
        self.credentials.append(UserRemoteCredentials(key_file=self.params.get('user_credentials_path')))
        regions = self.params.region_names

        ec2_security_group_ids, ec2_subnet_ids = get_ec2_network_configuration(
            regions=regions,
            availability_zones=self.params.get('availability_zone').split(','),
            params=self.params
        )

        services = get_ec2_services(regions)

        common_params = get_common_params(params=self.params, regions=regions, credentials=self.credentials,
                                          services=services)

        monitor_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'root_disk_type': None, 'n_local_ssd': None,
                        'device_mappings': None}
        db_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'root_disk_type': None, 'n_local_ssd': None,
                   'device_mappings': None}

        init_db_info_from_params(db_info, params=self.params, regions=regions)
        init_monitoring_info_from_params(monitor_info, params=self.params, regions=regions)

        self.k8s_cluster = eks.EksCluster(eks_cluster_version=self.params.get("eks_cluster_version"),
                                          ec2_security_group_ids=ec2_security_group_ids,
                                          ec2_subnet_ids=ec2_subnet_ids,
                                          credentials=self.credentials,
                                          ec2_role_arn=self.params.get("eks_role_arn"),
                                          nodegroup_role_arn=self.params.get("eks_nodegroup_role_arn"),
                                          service_ipv4_cidr=self.params.get("eks_service_ipv4_cidr"),
                                          vpc_cni_version=self.params.get("eks_vpc_cni_version"),
                                          user_prefix=self.params.get("user_prefix"),
                                          params=self.params,
                                          region_name=self.params.region_names[0]
                                          )

        self.k8s_cluster.deploy()
        self.k8s_cluster.tune_network()
        self.k8s_cluster.set_nodeselector_for_deployments(
            pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME, namespace="kube-system")

        self.k8s_cluster.deploy_node_pool(
            eks.EksNodePool(
                name=self.k8s_cluster.AUXILIARY_POOL_NAME,
                num_nodes=2,
                instance_type="t3.large",
                # It should have at least 3 vCPU to be able to hold all the pods
                disk_size=40,
                role_arn=self.k8s_cluster.nodegroup_role_arn,
                k8s_cluster=self.k8s_cluster),
            wait_till_ready=False)

        scylla_pool = eks.EksNodePool(
            name=self.k8s_cluster.SCYLLA_POOL_NAME,
            num_nodes=self.params.get("n_db_nodes"),
            instance_type=self.params.get('instance_type_db'),
            role_arn=self.params.get('eks_nodegroup_role_arn'),
            disk_size=self.params.get('root_disk_size_db'),
            k8s_cluster=self.k8s_cluster
        )
        self.k8s_cluster.deploy_node_pool(scylla_pool, wait_till_ready=False)

        loader_pool = None
        if self.params.get("n_loaders"):
            loader_pool = eks.EksNodePool(
                name=self.k8s_cluster.LOADER_POOL_NAME,
                num_nodes=self.params.get("n_loaders"),
                instance_type=self.params.get("instance_type_loader"),
                role_arn=self.params.get('eks_nodegroup_role_arn'),
                disk_size=self.params.get('root_disk_size_monitor'),
                k8s_cluster=self.k8s_cluster)
            self.k8s_cluster.deploy_node_pool(loader_pool, wait_till_ready=False)

        monitor_pool = None
        if self.params.get('k8s_deploy_monitoring'):
            monitor_pool = eks.EksNodePool(
                name=self.k8s_cluster.MONITORING_POOL_NAME,
                num_nodes=1,
                instance_type=self.params.get("instance_type_monitor"),
                role_arn=self.params.get('eks_nodegroup_role_arn'),
                disk_size=self.params.get('root_disk_size_monitor'),
                k8s_cluster=self.k8s_cluster
            )
            self.k8s_cluster.deploy_node_pool(monitor_pool, wait_till_ready=False)
        self.k8s_cluster.wait_all_node_pools_to_be_ready()

        self.k8s_cluster.deploy_cert_manager(pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME)
        self.k8s_cluster.deploy_scylla_operator()
        self.k8s_cluster.prepare_k8s_scylla_nodes(node_pool=scylla_pool)
        if self.params.get('use_mgmt'):
            self.k8s_cluster.deploy_scylla_manager(pool_name=self.k8s_cluster.AUXILIARY_POOL_NAME)

        for i in range(self.k8s_cluster.tenants_number):
            self.db_clusters_multitenant.append(eks.EksScyllaPodCluster(
                k8s_cluster=self.k8s_cluster,
                scylla_cluster_name=self.params.get("k8s_scylla_cluster_name") + (f"-{i + 1}" if i else ""),
                user_prefix=(f"{i + 1}-" if i else "") + self.params.get("user_prefix"),
                n_nodes=self.params.get("k8s_n_scylla_pods_per_cluster") or self.params.get("n_db_nodes"),
                params=self.params,
                node_pool=scylla_pool,
                add_nodes=False,
            ))
        self.db_cluster = self.db_clusters_multitenant[0]

        if self.params.get('k8s_deploy_monitoring'):
            self.k8s_cluster.deploy_monitoring_cluster(
                is_manager_deployed=self.params.get('use_mgmt'),
                node_pool=monitor_pool
            )

        if self.params.get("n_loaders"):
            for i in range(self.k8s_cluster.tenants_number):
                self.loaders_multitenant.append(cluster_k8s.LoaderPodCluster(
                    k8s_cluster=self.k8s_cluster,
                    loader_cluster_config=LOADER_CLUSTER_CONFIG,
                    loader_cluster_name=self.params.get("k8s_loader_cluster_name") + (f"-{i + 1}" if i else ""),
                    user_prefix=(f"{i + 1}-" if i else "") + self.params.get("user_prefix"),
                    n_nodes=self.params.get("k8s_n_loader_pods_per_cluster") or self.params.get("n_loaders"),
                    params=self.params,
                    node_pool=loader_pool,
                    add_nodes=False,
                ))
            self.loaders = self.loaders_multitenant[0]

        # NOTE: wait for Scylla and loader clusters nodes creation
        self._add_and_wait_for_cluster_nodes_in_parallel(
            self.db_clusters_multitenant + self.loaders_multitenant)

        if monitor_info['n_nodes']:
            base_user_prefix = common_params["user_prefix"]
            for i in range(self.k8s_cluster.tenants_number):
                self.log.debug("Create monitor for the DB cluster №%s", i + 1)
                common_params["user_prefix"] = (f"{i + 1}-" if i else "") + base_user_prefix
                self.monitors_multitenant.append(MonitorSetEKS(
                    ec2_ami_id=self.params.get('ami_id_monitor').split(),
                    ec2_ami_username=self.params.get('ami_monitor_user'),
                    ec2_instance_type=monitor_info['type'],
                    ec2_block_device_mappings=monitor_info['device_mappings'],
                    n_nodes=monitor_info['n_nodes'],
                    targets={
                        "db_cluster": self.db_clusters_multitenant[i],
                        "loaders": self.loaders_multitenant[i],
                    },
                    add_nodes=False,
                    monitor_id=self.test_config.test_id() + (f"-{i + 1}" if i else ""),
                    **common_params))
            self.monitors = self.monitors_multitenant[0]
            self._add_and_wait_for_cluster_nodes_in_parallel(self.monitors_multitenant)
        else:
            self.monitors = NoMonitorSet()
            self.monitors_multitenant = [self.monitors]

    def init_resources(self, loader_info=None, db_info=None,
                       monitor_info=None):
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        if loader_info is None:
            loader_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'root_disk_type': None, 'n_local_ssd': None,
                           'device_mappings': None}
        if db_info is None:
            db_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'root_disk_type': None, 'n_local_ssd': None,
                       'device_mappings': None}

        if monitor_info is None:
            monitor_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'root_disk_type': None, 'n_local_ssd': None,
                            'device_mappings': None}

        cluster_backend = self.params.get('cluster_backend')
        if cluster_backend is None:
            cluster_backend = 'aws'

        if cluster_backend in ('aws', 'aws-siren'):
            self.get_cluster_aws(loader_info=loader_info, db_info=db_info,
                                 monitor_info=monitor_info)
        elif cluster_backend in ('gce', 'gce-siren'):
            self.get_cluster_gce(loader_info=loader_info, db_info=db_info,
                                 monitor_info=monitor_info)
        elif cluster_backend == 'docker':
            self.get_cluster_docker()
        elif cluster_backend == 'baremetal':
            self.get_cluster_baremetal()
        elif cluster_backend in ('k8s-local-kind', 'k8s-local-kind-aws', 'k8s-local-kind-gce'):
            self.get_cluster_k8s_local_kind_cluster()
        elif cluster_backend == 'k8s-gke':
            self.get_cluster_k8s_gke()
        elif cluster_backend == 'k8s-eks':
            self.get_cluster_k8s_eks()
        elif cluster_backend == 'azure':
            self.get_cluster_azure(loader_info=loader_info, db_info=db_info,
                                   monitor_info=monitor_info)

        # NOTE: following starts processing of the monitoring inbound events which will be posted
        #       for each of the Grafana instances (may be more than 1 in case of K8S multi-tenant setup)
        start_posting_grafana_annotations()

    def _cs_add_node_flag(self, stress_cmd):
        if '-node' not in stress_cmd:
            if self.test_config.INTRA_NODE_COMM_PUBLIC:
                ip = ','.join(self.db_cluster.get_node_public_ips())
            else:
                ip = self.db_cluster.get_node_private_ips()[0]
            stress_cmd = '%s -node %s' % (stress_cmd, ip)
        return stress_cmd

    def run_stress(self, stress_cmd, duration=None):
        stress_cmd = self._cs_add_node_flag(stress_cmd)
        cs_thread_pool = self.run_stress_thread(stress_cmd=stress_cmd,
                                                duration=duration)
        self.verify_stress_thread(cs_thread_pool=cs_thread_pool)

    # pylint: disable=too-many-arguments,too-many-return-statements
    def run_stress_thread(self, stress_cmd, duration=None, stress_num=1, keyspace_num=1, profile=None, prefix='',
                          round_robin=False, stats_aggregate_cmds=True, keyspace_name=None, use_single_loader=False,
                          stop_test_on_failure=True):
        # We want to prevent situation when stress command starts on not ready cluster (no all nodes are UP).
        # It may cause to stress failure and as result the test will be stopped and failed
        self.db_cluster.wait_for_nodes_up_and_normal(iterations=0, timeout=MAX_TIME_WAIT_FOR_ALL_NODES_UP)

        params = dict(stress_cmd=stress_cmd, duration=duration, stress_num=stress_num, keyspace_num=keyspace_num,
                      keyspace_name=keyspace_name, profile=profile, prefix=prefix, round_robin=round_robin,
                      stats_aggregate_cmds=stats_aggregate_cmds, use_single_loader=use_single_loader)

        if 'cassandra-stress' in stress_cmd:  # cs cmdline might started with JVM_OPTION
            params['stop_test_on_failure'] = stop_test_on_failure
            return self.run_stress_cassandra_thread(**params)
        elif stress_cmd.startswith('scylla-bench'):
            params['stop_test_on_failure'] = stop_test_on_failure
            return self.run_stress_thread_bench(**params)
        elif 'cassandra-harry' in stress_cmd:
            params['stop_test_on_failure'] = stop_test_on_failure
            return self.run_stress_thread_harry(**params)
        elif stress_cmd.startswith('bin/ycsb'):
            return self.run_ycsb_thread(**params)
        elif stress_cmd.startswith('ndbench'):
            return self.run_ndbench_thread(**params)
        elif stress_cmd.startswith('hydra-kcl'):
            return self.run_hydra_kcl_thread(**params)
        elif stress_cmd.startswith('nosqlbench'):
            params['stop_test_on_failure'] = stop_test_on_failure
            return self.run_nosqlbench_thread(**params)
        elif stress_cmd.startswith('table_compare'):
            return self.run_table_compare_thread(**params)
        else:
            raise ValueError(f'Unsupported stress command: "{stress_cmd[:50]}..."')

    # pylint: disable=too-many-arguments
    def run_stress_cassandra_thread(
            self, stress_cmd, duration=None, stress_num=1, keyspace_num=1, profile=None, prefix='', round_robin=False,
            stats_aggregate_cmds=True, keyspace_name=None, stop_test_on_failure=True, **_):
        # stress_cmd = self._cs_add_node_flag(stress_cmd)
        if duration:
            timeout = self.get_duration(duration)
        else:
            timeout = get_timeout_from_stress_cmd(stress_cmd) or self.get_duration(duration)

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, prefix, stresser="cassandra-stress",
                                           aggregate=stats_aggregate_cmds)
        stop_test_on_failure = False if not self.params.get("stop_test_on_stress_failure") else stop_test_on_failure
        cs_thread = CassandraStressThread(loader_set=self.loaders,
                                          stress_cmd=stress_cmd,
                                          timeout=timeout,
                                          stress_num=stress_num,
                                          keyspace_num=keyspace_num,
                                          profile=profile,
                                          node_list=self.db_cluster.nodes,
                                          round_robin=round_robin,
                                          client_encrypt=self.params.get('client_encrypt'),
                                          keyspace_name=keyspace_name,
                                          stop_test_on_failure=stop_test_on_failure).run()
        scylla_encryption_options = self.params.get('scylla_encryption_options')
        if scylla_encryption_options and 'write' in stress_cmd:
            # Configure encryption at-rest for all test tables, sleep a while to wait the workload starts and test tables are created
            time.sleep(60)
            self.alter_test_tables_encryption(scylla_encryption_options=scylla_encryption_options)
            self.db_cluster.wait_for_schema_agreement()
        return cs_thread

    # pylint: disable=too-many-arguments,unused-argument
    def run_stress_thread_bench(self, stress_cmd, duration=None, round_robin=False, stats_aggregate_cmds=True,
                                use_single_loader=False, stop_test_on_failure=True, **_):

        if duration:
            timeout = self.get_duration(duration)
        else:
            timeout = get_timeout_from_stress_cmd(stress_cmd) or self.get_duration(duration)
        stop_test_on_failure = False if not self.params.get("stop_test_on_stress_failure") else stop_test_on_failure

        if self.params.get("client_encrypt") and ' -tls' not in stress_cmd:
            stress_cmd += ' -tls '

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, stresser="scylla-bench", aggregate=stats_aggregate_cmds)
        bench_thread = ScyllaBenchThread(
            stress_cmd, loader_set=self.loaders, timeout=timeout,
            node_list=self.db_cluster.nodes,
            round_robin=round_robin,
            use_single_loader=use_single_loader,
            stop_test_on_failure=stop_test_on_failure,
            credentials=self.db_cluster.get_db_auth()
        )
        bench_thread.run()
        scylla_encryption_options = self.params.get('scylla_encryption_options')
        if scylla_encryption_options and 'write' in stress_cmd:
            # Configure encryption at-rest for all test tables, sleep a while to wait the workload starts and test tables are created
            time.sleep(60)
            self.alter_test_tables_encryption(scylla_encryption_options=scylla_encryption_options)
        return bench_thread

    def run_stress_thread_harry(self, stress_cmd, duration=None, stress_num=1, keyspace_num=1, profile=None, prefix='',
                                # pylint: disable=too-many-arguments,unused-argument
                                round_robin=False, stats_aggregate_cmds=True, keyspace_name=None,
                                use_single_loader=False,
                                stop_test_on_failure=True):  # pylint: disable=too-many-arguments,unused-argument

        timeout = self.get_duration(duration)

        if not self.params.get("stop_test_on_stress_failure"):
            stop_test_on_failure = False

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, stresser="cassandra-harry", aggregate=stats_aggregate_cmds)
        harry_thread = CassandraHarryThread(
            stress_cmd,
            loader_set=self.loaders,
            timeout=timeout,
            node_list=self.db_cluster.nodes,
            round_robin=round_robin,
            use_single_loader=use_single_loader,
            stop_test_on_failure=stop_test_on_failure,
            credentials=self.db_cluster.get_db_auth()
        )
        harry_thread.run()
        return harry_thread

    # pylint: disable=too-many-arguments
    def run_ycsb_thread(self, stress_cmd, duration=None, stress_num=1, prefix='',
                        round_robin=False, stats_aggregate_cmds=True, **_):

        timeout = self.get_duration(duration)

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, prefix, stresser="ycsb", aggregate=stats_aggregate_cmds)

        return YcsbStressThread(loader_set=self.loaders,
                                stress_cmd=stress_cmd,
                                timeout=timeout,
                                stress_num=stress_num,
                                node_list=self.db_cluster.nodes,
                                round_robin=round_robin, params=self.params).run()

    # pylint: disable=too-many-arguments
    def run_hydra_kcl_thread(self, stress_cmd, duration=None, stress_num=1, prefix='',
                             round_robin=False, stats_aggregate_cmds=True, **_):

        timeout = self.get_duration(duration)

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, prefix, stresser="hydra-kcl", aggregate=stats_aggregate_cmds)

        return KclStressThread(loader_set=self.loaders,
                               stress_cmd=stress_cmd,
                               timeout=timeout,
                               stress_num=stress_num,
                               node_list=self.db_cluster.nodes,
                               round_robin=round_robin, params=self.params).run()

    # pylint: disable=too-many-arguments
    def run_nosqlbench_thread(self, stress_cmd, duration=None, stress_num=1, prefix='', round_robin=False,
                              stats_aggregate_cmds=True, stop_test_on_failure=True, **_):

        timeout = self.get_duration(duration)

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, prefix, stresser="nosqlbench", aggregate=stats_aggregate_cmds)

        stop_test_on_failure = False if not self.params.get("stop_test_on_stress_failure") else stop_test_on_failure
        return NoSQLBenchStressThread(
            loader_set=self.loaders,
            stress_cmd=stress_cmd,
            timeout=timeout,
            stress_num=stress_num,
            node_list=self.db_cluster.nodes,
            round_robin=round_robin,
            stop_test_on_failure=stop_test_on_failure,
            params=self.params).run()

    # pylint: disable=too-many-arguments
    def run_table_compare_thread(self, stress_cmd, duration=None, stress_num=1, round_robin=False, **_):

        timeout = self.get_duration(duration)

        return CompareTablesSizesThread(loader_set=self.loaders,
                                        stress_cmd=stress_cmd,
                                        timeout=timeout,
                                        stress_num=stress_num,
                                        node_list=self.db_cluster.nodes,
                                        round_robin=round_robin, params=self.params).run()

    # pylint: disable=too-many-arguments
    def run_ndbench_thread(self, stress_cmd, duration=None, stress_num=1, prefix='',
                           round_robin=False, stats_aggregate_cmds=True, **_):

        timeout = self.get_duration(duration)

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, prefix, stresser="ndbench", aggregate=stats_aggregate_cmds)

        return NdBenchStressThread(loader_set=self.loaders,
                                   stress_cmd=stress_cmd,
                                   timeout=timeout,
                                   stress_num=stress_num,
                                   node_list=self.db_cluster.nodes,
                                   round_robin=round_robin, params=self.params).run()

    # pylint: disable=too-many-arguments
    def run_cdclog_reader_thread(self, stress_cmd, duration=None, stress_num=1, prefix='',
                                 round_robin=False, stats_aggregate_cmds=True, enable_batching=True,
                                 keyspace_name=None, base_table_name=None):
        timeout = self.get_duration(duration)

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, prefix, stresser="cdcreader", aggregate=stats_aggregate_cmds)
        return CDCLogReaderThread(loader_set=self.loaders,
                                  stress_cmd=stress_cmd,
                                  timeout=timeout,
                                  stress_num=stress_num,
                                  node_list=self.db_cluster.nodes,
                                  keyspace_name=keyspace_name,
                                  base_table_name=base_table_name,
                                  enable_batching=enable_batching,
                                  round_robin=round_robin, params=self.params).run()

    def run_gemini(self, cmd, duration=None):
        if duration:
            timeout = self.get_duration(duration)
        else:
            timeout = get_timeout_from_stress_cmd(cmd) or self.get_duration(duration)
        if self.create_stats:
            self.update_stress_cmd_details(cmd, stresser="gemini", aggregate=False)
        return GeminiStressThread(test_cluster=self.db_cluster,
                                  oracle_cluster=self.cs_db_cluster,
                                  loaders=self.loaders,
                                  gemini_cmd=cmd,
                                  timeout=timeout,
                                  outputdir=self.loaders.logdir,
                                  params=self.params).run()

    def kill_stress_thread(self):
        if self.loaders:  # the test can fail on provision step and loaders are still not provisioned
            self.loaders.kill_stress_thread()

    def verify_stress_thread(self, cs_thread_pool):
        if isinstance(cs_thread_pool, dict):
            results = self.get_stress_results_bench(queue=cs_thread_pool)
            errors = []
        else:
            results, errors = cs_thread_pool.verify_results()
        if results and self.create_stats:
            self.update_stress_results(results)
        if not results:
            self.log.warning('There is no stress results, probably stress thread has failed.')
        # Sometimes, we might have an epic error messages list
        # that will make small machines driving the test
        # to run out of memory when writing the XML report. Since
        # the error message is merely informational, let's simply
        # use the last 5 lines for the final error message.
        errors = errors[-5:]
        if errors:
            self.log.warning("cassandra-stress errors on nodes:\n%s", "\n".join(errors))

    def get_stress_results(self, queue, store_results=True) -> list[dict | None]:
        results = queue.get_results()
        if store_results and self.create_stats:
            self.update_stress_results(results)
        return results

    def get_stress_results_bench(self, queue):
        results = queue.get_stress_results_bench()
        if self.create_stats:
            self.update_stress_results(results)
        return results

    def verify_cdclog_reader_results(self, cdcreadstessors_queue, update_es=False):
        results = cdcreadstessors_queue.get_results()
        if not results:
            self.log.warning("There are no cdclog_reader results")
        if self.create_stats and update_es:
            self.log.debug(results)
            self.update_stress_results(results)

        return results

    @staticmethod
    def get_gemini_results(queue):
        return queue.get_gemini_results()

    def verify_gemini_results(self, queue):
        results = queue.get_gemini_results()

        stats = {'results': [], 'errors': {}}
        if not results:
            self.log.error('Gemini results are not found')
            stats['status'] = 'FAILED'
        else:
            result = queue.verify_gemini_results(results)
            stats.update(result)

        if self.create_stats:
            self.update_stress_results(results, calculate_stats=False)
            self.update({'status': stats['status'],
                         'test_details': {'status': stats['status']},
                         'errors': stats['errors']})
        return stats

    def run_fullscan_thread(self, ks_cf='random', interval=1, duration=None):
        """Run thread of cql command select *

        Calculate test duration and timeout interval between
        requests and execute the thread with cqlsh command to
        db node 'select * from ks.cf, where ks and cf are
        random choosen from current configuration'

        Keyword Arguments:
            interval {number} -- interval between requests in min (default: {1})
            duration {int} -- duration of running thread in min (default: {None})
        """
        FullScanThread(
            db_cluster=self.db_cluster,
            ks_cf=ks_cf,
            duration=self.get_duration(duration),
            interval=interval * 60,
            termination_event=self.db_cluster.nemesis_termination_event,
        ).start()

    def run_full_partition_scan_thread(self, duration=None, interval=1, **kwargs):
        """Run thread of cql command select with a clustering key reversed query.

        Calculate test duration and timeout interval between
        requests and execute the thread with cqlsh command to
        db node: select * from ks.cf where pk = ? order by ck desc'

        Keyword Arguments:
            interval {number} -- interval between requests in seconds (default: {1})
            duration {int} -- duration of running thread in min (default: {None})
        """
        FullPartitionScanThread(
            db_cluster=self.db_cluster,
            termination_event=self.db_cluster.nemesis_termination_event,
            duration=self.get_duration(duration),
            interval=interval,
            **kwargs
        ).start()

    @staticmethod
    def is_keyspace_in_cluster(session, keyspace_name):
        query_result = session.execute("SELECT * FROM system_schema.keyspaces;")
        keyspace_list = [row.keyspace_name.lower() for row in query_result.current_rows]
        return keyspace_name.lower() in keyspace_list

    def wait_validate_keyspace_existence(self, session, keyspace_name, timeout=180,
                                         step=5):  # pylint: disable=invalid-name
        text = 'waiting for the keyspace "{}" to be created in the cluster'.format(keyspace_name)
        does_keyspace_exist = wait.wait_for(func=self.is_keyspace_in_cluster, step=step, text=text, timeout=timeout,
                                            session=session, keyspace_name=keyspace_name, throw_exc=False)
        return does_keyspace_exist

    def create_keyspace(self, keyspace_name, replication_factor, replication_strategy=None):
        """
        The default of replication_strategy depends on the type of the replication_factor:
            If it's int, the default of replication_strategy will be 'SimpleStrategy'
            If it's dict, the default of replication_strategy will be 'NetworkTopologyStrategy'

        In the case of NetworkTopologyStrategy, replication_strategy should be a dict that contains the name of
        every dc that the keyspace should be replicated to as keys, and the replication factor of each of those dc
        as values, like so:
        {"dc_name1": 4, "dc_name2": 6, "<dc_name>": <int>...}
        """

        query = 'CREATE KEYSPACE IF NOT EXISTS %s WITH replication={%s}'
        execution_node, validation_node = self.db_cluster.nodes[0], self.db_cluster.nodes[-1]
        with self.db_cluster.cql_connection_patient(execution_node) as session:
            if isinstance(replication_factor, int):
                execution_result = session.execute(
                    query % (keyspace_name, "'class':'{}', 'replication_factor':{}".format(
                        replication_strategy if replication_strategy else "SimpleStrategy",
                        replication_factor)))

            else:
                assert replication_factor, "At least one datacenter/replication_factor pair is needed"
                options = ', '.join(["'{}':{}".format(dc_name, dc_specific_replication_factor) for
                                     dc_name, dc_specific_replication_factor in replication_factor.items()])
                execution_result = session.execute(
                    query % (keyspace_name, "'class':'{}', {}".format(
                        replication_strategy if replication_strategy else "NetworkTopologyStrategy",
                        options)))

        if execution_result:
            self.log.debug("keyspace creation result: {}".format(execution_result.response_future))

        with self.db_cluster.cql_connection_patient(validation_node) as session:
            does_keyspace_exist = self.wait_validate_keyspace_existence(session, keyspace_name)
        return does_keyspace_exist

    def create_table(self, name, key_type="varchar",  # pylint: disable=too-many-arguments,too-many-branches
                     speculative_retry=None, read_repair=None, compression=None,
                     gc_grace=None, columns=None, compaction=None,
                     compact_storage=False, in_memory=False, scylla_encryption_options=None, keyspace_name=None,
                     sstable_size=None):

        # pylint: disable=too-many-locals
        additional_columns = ""
        if columns is not None:
            for key, value in columns.items():
                additional_columns = "%s, %s %s" % (additional_columns, key, value)

        if additional_columns == "":
            query = ('CREATE COLUMNFAMILY IF NOT EXISTS %s (key %s, c varchar, v varchar, '
                     'PRIMARY KEY(key, c)) WITH comment=\'test cf\'' %
                     (name, key_type))
        else:
            query = ('CREATE COLUMNFAMILY IF NOT EXISTS %s (key %s PRIMARY KEY%s) '
                     'WITH comment=\'test cf\'' %
                     (name, key_type, additional_columns))

        if compression is not None:
            query = ('%s AND compression = { \'sstable_compression\': '
                     '\'%sCompressor\' }' % (query, compression))
        else:
            # if a compression option is omitted, C*
            # will default to lz4 compression
            query += ' AND compression = {}'

        if not in_memory and (compaction is not None or sstable_size):
            compaction = compaction or self.params.get('compaction_strategy')
            prefix = ' AND compaction={'
            postfix = '}'
            compaction_clause = " 'class': '%s'" % compaction
            if sstable_size:
                compaction_clause += ", 'sstable_size_in_mb' : '%s'" % sstable_size
            query += prefix + compaction_clause + postfix

        if read_repair is not None:
            query = '%s AND read_repair_chance=%f' % (query, read_repair)
        if gc_grace is not None:
            query = '%s AND gc_grace_seconds=%d' % (query, gc_grace)
        if speculative_retry is not None:
            query = ('%s AND speculative_retry=\'%s\'' %
                     (query, speculative_retry))
        if in_memory:
            query += " AND in_memory=true AND compaction={'class': 'InMemoryCompactionStrategy'}"
        if scylla_encryption_options:
            query = '%s AND scylla_encryption_options=%s' % (query, scylla_encryption_options)
        if compact_storage:
            query += ' AND COMPACT STORAGE'
        self.log.debug('CQL query to execute: {}'.format(query))
        with self.db_cluster.cql_connection_patient(node=self.db_cluster.nodes[0], keyspace=keyspace_name) as session:
            session.execute(query)
        time.sleep(0.2)

    def truncate_cf(self, ks_name, table_name, session):
        try:
            session.execute('TRUNCATE TABLE {0}.{1}'.format(ks_name, table_name))
        except Exception as ex:  # pylint: disable=broad-except
            self.log.debug('Failed to truncate base table {0}.{1}. Error: {2}'.format(ks_name, table_name, str(ex)))

    def create_materialized_view(self, ks_name, base_table_name, mv_name, mv_partition_key, mv_clustering_key, session,
                                 # pylint: disable=too-many-arguments
                                 mv_columns='*', speculative_retry=None, read_repair=None, compression=None,
                                 gc_grace=None, compact_storage=False):

        # pylint: disable=too-many-locals
        mv_columns_str = mv_columns
        if isinstance(mv_columns, list):
            mv_columns_str = ', '.join(c for c in mv_columns)

        where_clause = []
        mv_partition_key = mv_partition_key if isinstance(mv_partition_key, list) else list(mv_partition_key)
        mv_clustering_key = mv_clustering_key if isinstance(mv_clustering_key, list) else list(mv_clustering_key)

        for kc in mv_partition_key + mv_clustering_key:  # pylint: disable=invalid-name
            where_clause.append('{} is not null'.format(kc))

        pk_clause = ', '.join(pk for pk in mv_partition_key)
        cl_clause = ', '.join(cl for cl in mv_clustering_key)

        query = 'CREATE MATERIALIZED VIEW {ks}.{mv_name} AS SELECT {mv_columns} FROM {ks}.{table_name} ' \
                'WHERE {where_clause} PRIMARY KEY ({pk}, {cl}) WITH comment=\'test MV\''.format(ks=ks_name,
                                                                                                mv_name=mv_name,
                                                                                                mv_columns=mv_columns_str,
                                                                                                table_name=base_table_name,
                                                                                                where_clause=' and '.join
                                                                                                (wc for wc in
                                                                                                 where_clause),
                                                                                                pk=pk_clause,
                                                                                                cl=cl_clause)
        if compression is not None:
            query = ('%s AND compression = { \'sstable_compression\': '
                     '\'%sCompressor\' }' % (query, compression))

        if read_repair is not None:
            query = '%s AND read_repair_chance=%f' % (query, read_repair)
        if gc_grace is not None:
            query = '%s AND gc_grace_seconds=%d' % (query, gc_grace)
        if speculative_retry is not None:
            query = ('%s AND speculative_retry=\'%s\'' %
                     (query, speculative_retry))

        if compact_storage:
            query += ' AND COMPACT STORAGE'

        self.log.debug('MV create statement: {}'.format(query))
        session.execute(query)

    def _wait_for_view(self, scylla_cluster, session, key_space, view):
        self.log.debug("Waiting for view {}.{} to finish building...".format(key_space, view))

        def _view_build_finished(live_nodes_amount):
            result = self.rows_to_list(session.execute("SELECT status FROM system_distributed.view_build_status WHERE "
                                                       "keyspace_name='{0}' "
                                                       "AND view_name='{1}'".format(key_space, view)))
            self.log.debug('View build status result: {}'.format(result))
            return len([status for status in result if status[0] == 'SUCCESS']) >= live_nodes_amount

        attempts = 20
        nodes_status = scylla_cluster.get_nodetool_status()
        live_nodes_amount = 0
        for dc in nodes_status.itervalues():
            for ip in dc.itervalues():
                if ip['state'] == 'UN':
                    live_nodes_amount += 1

        while attempts > 0:
            if _view_build_finished(live_nodes_amount):
                return
            time.sleep(3)
            attempts -= 1

        raise Exception("View {}.{} not built".format(key_space, view))

    def _wait_for_view_build_start(self, session, key_space, view, seconds_to_wait=20):

        def _check_build_started():
            result = self.rows_to_list(session.execute("SELECT last_token FROM system.views_builds_in_progress "
                                                       "WHERE keyspace_name='{0}' AND view_name='{1}'".format(key_space,
                                                                                                              view)))
            self.log.debug('View build in progress: {}'.format(result))
            return result != []

        self.log.debug("Ensure view building started.")
        start = time.time()
        while not _check_build_started():
            if time.time() - start > seconds_to_wait:
                raise Exception("View building didn't start in {} seconds".format(seconds_to_wait))

    @staticmethod
    def rows_to_list(rows):
        return [list(row) for row in rows]

    def copy_table(self, node, src_keyspace, src_table, dest_keyspace,  # pylint: disable=too-many-arguments
                   dest_table, columns_list=None, copy_data=False):
        """
        Create table with same structure as <src_keyspace>.<src_table>.
        If columns_list is supplied, the table with create with the columns that in the columns_list
        Copy data from <src_keyspace>.<src_view> to <dest_keyspace>.<dest_table> if copy_data is True
        """
        result = True
        create_statement = "SELECT * FROM system_schema.table where table_name = '%s' " \
                           "and keyspace_name = '%s'" % (src_table, src_keyspace)
        if not self.create_table_as(node, src_keyspace, src_table, dest_keyspace,
                                    dest_table, create_statement, columns_list):
            return False

        if copy_data:
            try:
                result = self.copy_data_between_tables(node, src_keyspace, src_table,
                                                       dest_keyspace, dest_table, columns_list)
            except Exception as error:  # pylint: disable=broad-except
                self.log.error('Copying data from %s to %s failed with error: %s',
                               src_table, dest_table, error)
                return False

        return result

    def copy_view(self, node, src_keyspace, src_view, dest_keyspace,  # pylint: disable=too-many-arguments
                  dest_table, columns_list=None, copy_data=False):
        """
        Create table with same structure as <src_keyspace>.<src_view>.
        If columns_list is supplied, the table with create with the columns that in the columns_list
        Copy data from <src_keyspace>.<src_view> to <dest_keyspace>.<dest_table> if copy_data is True
        """
        result = True
        create_statement = "SELECT * FROM system_schema.views where view_name = '%s' " \
                           "and keyspace_name = '%s'" % (src_view, src_keyspace)
        self.log.debug('Start create table with statement: {}'.format(create_statement))
        if not self.create_table_as(node, src_keyspace, src_view, dest_keyspace,
                                    dest_table, create_statement, columns_list):
            return False
        self.log.debug('Finish create table')
        if copy_data:
            try:
                result = self.copy_data_between_tables(node, src_keyspace, src_view,
                                                       dest_keyspace, dest_table, columns_list)
            except Exception as error:  # pylint: disable=broad-except
                self.log.error('Copying data from %s to %s failed with error %s',
                               src_view, dest_table, error)
                return False

        return result

    def create_table_as(self, node, src_keyspace, src_table,
                        # pylint: disable=too-many-arguments,too-many-locals,inconsistent-return-statements
                        dest_keyspace, dest_table, create_statement,
                        columns_list=None):
        """ Create table with same structure as another table or view
            If columns_list is supplied, the table with create with the columns that in the columns_list
        """
        with self.db_cluster.cql_connection_patient(node) as session:

            result = rows_to_list(session.execute(create_statement))
            if result:
                result = session.execute("SELECT * FROM {keyspace}.{table} LIMIT 1".format(keyspace=src_keyspace,
                                                                                           table=src_table))

                primary_keys = []
                # Create table with table/view structure
                columns = list(zip(result.column_names, [typelist.typename for typelist in result.column_types]))

                # if columns list was supplied, create the table with those columns only
                if columns_list:
                    column_types = []
                    for column in columns_list:
                        column_types.extend(column_type for column_type in columns if column == column_type[0])
                    columns = column_types

                if not columns:
                    self.log.error('Wrong supplied columns list: %s. '
                                   'The columns do not exist in the %s.%s table',
                                   columns, src_keyspace, src_table)
                    return False

                for column in columns_list or result.column_names:
                    column_kind = session.execute("select kind from system_schema.columns where keyspace_name='{ks}' "
                                                  "and table_name='{name}' and column_name='{column}'".format(
                                                      ks=src_keyspace,
                                                      name=src_table,
                                                      column=column))
                    if column_kind.current_rows[0].kind in ['partition_key', 'clustering']:
                        primary_keys.append(column)

                if not primary_keys:
                    primary_keys.append(columns[0][0])

                create_cql = 'create table {keyspace}.{name} ({columns}, PRIMARY KEY ({pk}))' \
                    .format(keyspace=dest_keyspace,
                            name=dest_table,
                            columns=', '.join(['%s %s' % (c[0], c[1]) for c in columns]),
                            pk=', '.join(primary_keys))
                self.log.debug('Create new table with cql: {}'.format(create_cql))
                session.execute(create_cql)
                return True
            return False

    @retrying(n=4, sleep_time=5, message='Fetch all rows', raise_on_exceeded=False)
    def fetch_all_rows(self, session, default_fetch_size, statement, verbose=True):
        """
        ******* Caution *******
        All data from table will be read to the memory
        BE SURE that the builder has enough memory and your dataset will be less then 2Gb.
        """
        if verbose:
            self.log.debug("Fetch all rows by statement: %s", statement)
        session.default_fetch_size = default_fetch_size
        session.default_consistency_level = ConsistencyLevel.QUORUM
        result = session.execute_async(statement)
        fetcher = PageFetcher(result).request_all()
        current_rows = fetcher.all_data()
        if verbose and current_rows:
            dataset_size = sum(sys.getsizeof(e) for e in current_rows[0]) * len(current_rows)
            self.log.debug("Size of fetched rows: %s bytes", dataset_size)
        return current_rows

    def copy_data_between_tables(self, node, src_keyspace, src_table, dest_keyspace,
                                 # pylint: disable=too-many-arguments,too-many-locals
                                 dest_table, columns_list=None):
        """ Copy all data from one table/view to another table
            Structure of the tables has to be same
        """
        self.log.debug('Start copying data')
        with self.db_cluster.cql_connection_patient(node, verbose=False) as session:
            # Copy data from source to the destination table
            statement = "SELECT {columns} FROM {keyspace}.{table}".format(keyspace=src_keyspace,
                                                                          table=src_table,
                                                                          columns=','.join(
                                                                              columns_list) if columns_list else '*')
            # Get table columns list
            result = session.execute(statement + ' LIMIT 1')
            columns = result.column_names

            # Fetch all rows from view / table
            source_table_rows = self.fetch_all_rows(session=session, default_fetch_size=5000, statement=statement)
            if not source_table_rows:
                self.log.error("Can't copy data from %s. Fetch all rows failed, see error above", src_table)
                return False

            # TODO: Temporary function. Will be removed
            self.log.debug('Rows in the {} MV before saving: {}'.format(src_table, len(source_table_rows)))

            insert_statement = session.prepare(
                'insert into {keyspace}.{name} ({columns}) '
                'values ({values})'.format(keyspace=dest_keyspace,
                                           name=dest_table,
                                           columns=', '.join(columns),
                                           values=', '.join(['?' for _ in columns])))

            # Save all rows
            # Workers = Parallel queries = (nodes in cluster) x (cores in node) x 3
            # (from https://www.scylladb.com/2017/02/13/efficient-full-table-scans-with-scylla-1-6/)
            cores = self.db_cluster.nodes[0].cpu_cores
            if not cores:
                # If CPU core didn't find, put 8 as default
                cores = 8
            max_workers = len(self.db_cluster.nodes) * cores * 3

            session.default_consistency_level = ConsistencyLevel.QUORUM

            results = execute_concurrent_with_args(session=session, statement=insert_statement,
                                                   parameters=source_table_rows,
                                                   concurrency=max_workers, results_generator=True)

            try:
                succeeded_rows = sum(1 for (success, result) in results if success)
                if succeeded_rows != len(source_table_rows):
                    self.log.warning('Problem during copying data. Not all rows were inserted. '
                                     'Rows expected to be inserted: %s; '
                                     'Actually inserted rows: %s.',
                                     len(source_table_rows), succeeded_rows)
                    return False
            except Exception as exc:  # pylint: disable=broad-except
                self.log.warning('Problem during copying data: %s', exc)
                return False

            result = session.execute(f"SELECT count(*) FROM {dest_keyspace}.{dest_table}")
            if result:
                if result.current_rows[0].count != len(source_table_rows):
                    self.log.warning('Problem during copying data. '
                                     'Rows in source table: %s; '
                                     'Rows in destination table: %s.',
                                     len(source_table_rows), len(result.current_rows))
                    return False
        self.log.debug('All rows have been copied from %s to %s', src_table, dest_table)
        return True

    def collect_partitions_info(self, table_name, primary_key_column, save_into_file_name):
        # Get and save how many rows in each partition.
        # It may be used for validation data in the end of test
        if not (table_name or primary_key_column):
            self.log.warning('Can\'t collect partitions data. Missed "table name" or "primary key column" info')
            return {}

        get_distinct_partition_keys_cmd = 'select distinct {pk} from {table}'.format(pk=primary_key_column,
                                                                                     table=table_name)
        try:
            out = self.db_cluster.nodes[0].run_cqlsh(cmd=get_distinct_partition_keys_cmd, timeout=600, split=True,
                                                     num_retry_on_failure=5)
        except Exception as exc:  # pylint: disable=broad-except
            self.log.error("Failed to collect partition info. Error details: %s", str(exc))
            return None

        pk_list = sorted([int(pk) for pk in out[3:-3]])

        # Collect data about partitions' rows amount.
        partitions = {}
        partitions_stats_file = os.path.join(self.logdir, save_into_file_name)
        with open(partitions_stats_file, 'a', encoding="utf-8") as stats_file:
            for i in pk_list:
                self.log.debug("Next PK: {}".format(i))
                count_partition_keys_cmd = f'select count(*) from {table_name} where {primary_key_column} = {i}'
                try:
                    out = self.db_cluster.nodes[0].run_cqlsh(cmd=count_partition_keys_cmd, timeout=600, split=True,
                                                             num_retry_on_failure=5)
                except Exception as exc:  # pylint: disable=broad-except
                    self.log.error("Failed to collect partition info. Error details: %s", str(exc))
                    return None

                self.log.debug('Count result: {}'.format(out))
                partitions[i] = out[3] if len(out) > 3 else None
                stats_file.write('{i}:{rows}, '.format(i=i, rows=partitions[i]))
        self.log.info('File with partitions row data: {}'.format(partitions_stats_file))

        return partitions

    def get_tables_id_of_keyspace(self, session, keyspace_name):
        query = "SELECT id FROM system_schema.tables WHERE keyspace_name='{}' ".format(keyspace_name)
        table_id = self.rows_to_list(session.execute(query))
        return table_id[0]

    def get_tables_name_of_keyspace(self, session, keyspace_name):
        query = "SELECT table_name FROM system_schema.tables WHERE keyspace_name='{}' ".format(keyspace_name)
        table_id = self.rows_to_list(session.execute(query))
        return table_id[0]

    def get_truncated_time_from_system_local(self, session):  # pylint: disable=invalid-name
        query = "SELECT truncated_at FROM system.local"
        truncated_time = self.rows_to_list(session.execute(query))
        return truncated_time

    def get_truncated_time_from_system_truncated(self, session, table_id):  # pylint: disable=invalid-name
        query = "SELECT truncated_at FROM system.truncated WHERE table_uuid={}".format(table_id)
        truncated_time = self.rows_to_list(session.execute(query))
        return truncated_time[0]

    def get_describecluster_info(self) -> Optional[ClusterInformation]:  # pylint: disable=too-many-locals
        """
        Runs the 'nodetool describecluster' command on a node.

        Example output:

        nodetool describecluster

        Cluster Information:
             Name: Test Cluster
             Snitch: org.apache.cassandra.locator.SimpleSnitch
             Partitioner: org.apache.cassandra.dht.Murmur3Partitioner
             Schema versions:
                     86a67fc7-1d7c-3dc3-9be9-9c86b27e2506: [172.17.0.2]

        The output is then packaged into a ClusterInformation
        namedtuple, for easier access to the fields.
        """
        node = self.db_cluster.get_node()
        describecluster_output = node.run_nodetool(sub_cmd="describecluster")

        if describecluster_output.ok:
            desc_stdout = describecluster_output.stdout
            name_pattern = re.compile("((?<=Name: )[\w _-]*)")  # pylint: disable=anomalous-backslash-in-string
            snitch_pattern = re.compile("((?<=Snitch: )[\w.]*)")  # pylint: disable=anomalous-backslash-in-string
            partitioner_pattern = re.compile(
                "((?<=Partitioner: )[\w.]*)")  # pylint: disable=anomalous-backslash-in-string
            schema_versions_pattern = re.compile(
                "([a-z0-9-]{36}: \[[\d., ]*\])")  # pylint: disable=anomalous-backslash-in-string

            name = name_pattern.search(desc_stdout).group()
            snitch = snitch_pattern.search(desc_stdout).group()
            partitioner = partitioner_pattern.search(desc_stdout).group()
            schemas = schema_versions_pattern.findall(desc_stdout)
            schema_versions = []
            if schemas:
                for item in schemas:
                    split = item.split(":")
                    schema_id = split[0].strip()
                    node_ips = split[-1].strip()
                    schema_versions.append(SchemaVersion(schema_id=schema_id, node_ips=node_ips))

            return ClusterInformation(
                name=name,
                snitch=snitch,
                partitioner=partitioner,
                schema_versions=schema_versions
            )

        return None

    def get_nemesis_report(self, cluster):
        for current_nemesis in cluster.nemesis:
            with silence(parent=self, name=f"get_nemesis_report(cluster={str(cluster)})"):
                current_nemesis.report()

    @silence()
    def stop_nemesis(self, cluster):  # pylint: disable=no-self-use
        cluster.stop_nemesis(timeout=1800)

    @silence()
    def stop_resources_stop_tasks_threads(self, cluster):  # pylint: disable=no-self-use
        # TODO: this should be run in parallel
        for node in cluster.nodes:
            node.stop_task_threads()
        for node in cluster.nodes:
            with silence(parent=self, name=f'stop_resources_stop_tasks_threads(cluster={str(cluster)})'):
                node.wait_till_tasks_threads_are_stopped()

    @silence()
    def get_backtraces(self, cluster):  # pylint: disable=no-self-use
        cluster.get_backtraces()

    @silence()
    def stop_resources(self):  # pylint: disable=no-self-use
        self.log.debug('Stopping all resources')
        with silence(parent=self, name="Kill Stress Threads"):
            self.kill_stress_thread()

        # Stopping nemesis, using timeout of 30 minutes, since replace/decommission node can take time
        if self.db_cluster:
            self.get_nemesis_report(self.db_cluster)
            self.stop_nemesis(self.db_cluster)
            self.stop_resources_stop_tasks_threads(self.db_cluster)
            self.get_backtraces(self.db_cluster)

        if self.loaders:
            self.get_backtraces(self.loaders)
            self.stop_resources_stop_tasks_threads(self.loaders)

        if self.monitors:
            self.get_backtraces(self.monitors)
            self.stop_resources_stop_tasks_threads(self.monitors)

    @silence()
    def destroy_cluster(self, cluster):  # pylint: disable=no-self-use
        cluster.destroy()

    @silence()
    def set_keep_alive_on_failure(self, cluster):  # pylint: disable=no-self-use
        cluster.set_keep_alive_on_failure()

    @silence()
    def destroy_credentials(self):  # pylint: disable=no-self-use
        if self.credentials is not None:
            for credential in self.credentials:
                credential.destroy()
            self.credentials = []

    @silence()
    def show_alive_threads(self):
        return gather_live_threads_and_dump_to_file(self.left_processes_log)

    @silence()
    def show_alive_processes(self):
        return gather_live_processes_and_dump_to_file(self.left_processes_log)

    @silence()
    def clean_resources(self):
        # pylint: disable=too-many-branches
        if not self.params.get('execute_post_behavior'):
            self.log.info('Resources will continue to run')
            return

        actions_per_cluster_type = get_post_behavior_actions(self.params)
        critical_events = get_testrun_status(self.test_config.test_id(), self.logdir, only_critical=True)
        if self.db_cluster is not None:
            action = actions_per_cluster_type['db_nodes']['action']
            self.log.info("Action for db nodes is %s", action)
            if (action == 'destroy') or (action == 'keep-on-failure' and not critical_events):
                self.destroy_cluster(self.db_cluster)
                self.db_cluster = None
                if self.cs_db_cluster:
                    self.destroy_cluster(self.cs_db_cluster)
            elif action == 'keep-on-failure' and critical_events:
                self.log.info('Critical errors found. Set keep flag for db nodes')
                self.test_config.keep_cluster(node_type='db_nodes', val='keep')
                self.set_keep_alive_on_failure(self.db_cluster)
                if self.cs_db_cluster:
                    self.set_keep_alive_on_failure(self.cs_db_cluster)

        if self.loaders is not None:
            action = actions_per_cluster_type['loader_nodes']['action']
            self.log.info("Action for loader nodes is %s", action)
            if (action == 'destroy') or (action == 'keep-on-failure' and not critical_events):
                self.destroy_cluster(self.loaders)
                self.loaders = None
            elif action == 'keep-on-failure' and critical_events:
                self.log.info('Critical errors found. Set keep flag for loader nodes')
                self.test_config.keep_cluster(node_type='loader_nodes', val='keep')
                self.set_keep_alive_on_failure(self.loaders)

        if self.monitors is not None:
            action = actions_per_cluster_type['monitor_nodes']['action']
            self.log.info("Action for monitor nodes is %s", action)
            if (action == 'destroy') or (action == 'keep-on-failure' and not critical_events):
                self.destroy_cluster(self.monitors)
                self.monitors = None
            elif action == 'keep-on-failure' and critical_events:
                self.log.info('Critical errors found. Set keep flag for monitor nodes')
                self.test_config.keep_cluster(node_type='monitor_nodes', val='keep')
                self.set_keep_alive_on_failure(self.monitors)

        self.destroy_credentials()

    def tearDown(self):
        self.teardown_started = True
        with silence(parent=self, name='Sending test end event'):
            InfoEvent(message="TEST_END").publish()
        self.log.info('TearDown is starting...')
        self.stop_timeout_thread()
        self.stop_event_analyzer()
        self.stop_resources()
        self.get_test_failures()

        # NOTE: running on K8S we need to gather logs otherwise a lot of
        # debugging info is lost
        if self.k8s_cluster:
            self.k8s_cluster.gather_k8s_logs()

        if self.params.get('collect_logs'):
            self.collect_logs()
        self.clean_resources()
        if self.create_stats:
            self.update_test_with_errors()
        self.tag_ami_with_result()
        time.sleep(1)  # Sleep is needed to let final event being saved into files
        self.save_email_data()
        self.destroy_localhost()
        self.send_email()
        self.stop_event_device()
        if self.params.get('collect_logs'):
            self.collect_sct_logs()

        self.finalize_teardown()
        self.argus_finalize_test_run()

        self.log.info('Test ID: {}'.format(self.test_config.test_id()))
        self._check_alive_routines_and_report_them()
        self._check_if_db_log_time_consistency_looks_good()
        self.remove_python_exit_hooks()

    @silence()
    def remove_python_exit_hooks(self):  # pylint: disable=no-self-use
        clear_out_all_exit_hooks()

    @silence()
    def _check_if_db_log_time_consistency_looks_good(self):
        result = DbLogTimeConsistencyAnalyzer.analyze_dir(self.logdir)
        looks_good = True
        for value in result['TOTAL'].values():
            if value > 0:
                looks_good = False
                break
        if looks_good:
            self.log.info('DB logs time consistency is perfect')
        else:
            self.log.error(
                'DB logs time consistency is NOT perfect, details:\n%s',
                yaml.safe_dump(result, sort_keys=False)
            )

    def _check_alive_routines_and_report_them(self):
        threads_alive = self.show_alive_threads()
        processes_alive = self.show_alive_processes()
        if processes_alive or threads_alive:
            self.log.error('Please check %s log to see them', self.left_processes_log)

    def _get_test_result_event(self) -> TestResultEvent:
        return TestResultEvent(
            test_status=self.get_test_status(),
            events=get_events_grouped_by_category(limit=1, _registry=self.events_processes_registry),
        )

    @staticmethod
    def _remove_errors_from_unittest_results(result):
        to_remove = []
        for error in result.errors:
            if error[1] is not None:
                to_remove.append(error)
        for error in to_remove:
            result.errors.remove(error)

    def finalize_teardown(self):
        final_event = self._get_test_result_event()
        if final_event.test_status == 'SUCCESS':
            self.log.info(str(final_event))
            return
        if self._outcome is not None:
            # If there is not self._outcome, it means tests running in non-unittest environment
            self._remove_errors_from_unittest_results(self._outcome)
            self._outcome.errors.append((self, (TestResultEvent, final_event, None)))

    @silence()
    def destroy_localhost(self):
        if self.localhost:
            self.localhost.destroy()

    @silence()
    def stop_event_analyzer(self):  # pylint: disable=no-self-use
        stop_events_analyzer(_registry=self.events_processes_registry)

    @silence()
    def stop_timeout_thread(self):
        if self.timeout_thread:
            self.timeout_thread.cancel()

    @silence()
    def collect_sct_logs(self):
        s3_link = SCTLogCollector(
            [], self.test_config.test_id(), os.path.join(self.logdir, "collected_logs"), params=self.params
        ).collect_logs(self.logdir)
        if s3_link:
            self.log.info(s3_link)
            self.argus_collect_logs({"sct_job_log": s3_link})

            if self.create_stats:
                self.update({'test_details': {'log_files': {'job_log': s3_link}}})

    @silence()
    def stop_event_device(self):  # pylint: disable=no-self-use
        stop_events_device(_registry=self.events_processes_registry)

    @silence()
    def update_test_with_errors(self):
        coredumps = []
        if self.db_cluster:
            coredumps = self.db_cluster.coredumps
        test_events = get_events_grouped_by_category(_registry=self.events_processes_registry)
        self.update_test_details(
            errors=test_events['ERROR'] + test_events['CRITICAL'],
            coredumps=coredumps)

    def populate_data_parallel(self, size_in_gb: int, replication_factor: int = 3, blocking=True, read=False):

        # pylint: disable=too-many-locals
        base_cmd = "cassandra-stress write cl=QUORUM "
        if read:
            base_cmd = "cassandra-stress read cl=ONE "
        stress_fixed_params = f" -schema 'replication(factor={replication_factor}) " \
                              "compaction(strategy=LeveledCompactionStrategy)' " \
                              "-mode cql3 native -rate threads=200 -col 'size=FIXED(1024) n=FIXED(1)' "
        stress_keys = "n="
        population = " -pop seq="

        total_keys = size_in_gb * 1024 * 1024
        n_loaders = int(self.params.get('n_loaders'))
        keys_per_node = total_keys // n_loaders

        write_queue = []
        start = 1
        for i in range(1, n_loaders + 1):
            stress_cmd = base_cmd + stress_keys + str(keys_per_node) + population + str(start) + ".." + \
                str(keys_per_node * i) + stress_fixed_params
            start = keys_per_node * i + 1

            write_queue.append(self.run_stress_thread(stress_cmd=stress_cmd, round_robin=True))
            time.sleep(3)

        if blocking:
            for stress in write_queue:
                self.verify_stress_thread(cs_thread_pool=stress)

        return write_queue

    @log_run_info
    def alter_table_to_in_memory(self, key_space_name="keyspace1", table_name="standard1", node=None):
        if not node:
            node = self.db_cluster.nodes[0]
        compaction_strategy = "%s" % {"class": "InMemoryCompactionStrategy"}
        cql_cmd = "ALTER table {key_space_name}.{table_name} " \
                  "WITH in_memory=true AND compaction={compaction_strategy}".format(key_space_name=key_space_name,
                                                                                    table_name=table_name,
                                                                                    compaction_strategy=compaction_strategy)
        node.run_cqlsh(cql_cmd)

    def alter_table_encryption(self, table, scylla_encryption_options=None, upgradesstables=True):
        """
        Update table encryption
        """
        if not scylla_encryption_options:
            self.log.debug(
                'scylla_encryption_options is not set, skipping to enable encryption at-rest for all test tables')
        else:
            with self.db_cluster.cql_connection_patient(self.db_cluster.nodes[0]) as session:
                query = "ALTER TABLE {table} WITH scylla_encryption_options = {scylla_encryption_options};".format(
                    table=table, scylla_encryption_options=scylla_encryption_options)
                self.log.debug('enable encryption at-rest for table {table}, query:\n\t{query}'.format(**locals()))
                session.execute(query)
            if upgradesstables:
                self.log.debug('upgrade sstables after encryption update')
                for node in self.db_cluster.nodes:
                    node.remoter.run('nodetool upgradesstables', verbose=True, ignore_status=True)

    def disable_table_encryption(self, table, upgradesstables=True):
        self.alter_table_encryption(
            table, scylla_encryption_options="{'key_provider': 'none'}", upgradesstables=upgradesstables)

    def alter_test_tables_encryption(self, scylla_encryption_options=None, upgradesstables=True):
        for table in self.db_cluster.get_non_system_ks_cf_list(self.db_cluster.nodes[0], filter_out_mv=True):
            self.alter_table_encryption(
                table, scylla_encryption_options=scylla_encryption_options, upgradesstables=upgradesstables)

    def get_num_of_hint_files(self, node):
        result = node.remoter.run("sudo find {0.scylla_hints_dir} -name *.log -type f| wc -l".format(self),
                                  verbose=True)
        total_hint_files = int(result.stdout.strip())
        self.log.debug("Number of hint files on '%s': %s.", node.name, total_hint_files)
        return total_hint_files

    def get_num_shards(self, node):
        result = node.remoter.run("sudo ls -1 {0.scylla_hints_dir}| wc -l".format(self), verbose=True)
        return int(result.stdout.strip())

    @retrying(n=3, sleep_time=15, allowed_exceptions=(AssertionError,))
    def hints_sending_in_progress(self):
        query = "sum(rate(scylla_hints_manager_sent{}[1m]))"
        now = time.time()
        # check status of sending hints during last minute range
        results = self.prometheus_db.query(query=query, start=now - 60, end=now)
        self.log.debug("scylla_hints_manager_sent: %s", results)
        assert results, "No results from Prometheus"
        # if all are zeros the result will be False, otherwise we are still sending
        return any((float(v[1]) for v in results[0]["values"]))

    @retrying(n=30, sleep_time=60, allowed_exceptions=(AssertionError, UnexpectedExit, Failure))
    def wait_for_hints_to_be_sent(self, node, num_dest_nodes):
        num_shards = self.get_num_shards(node)
        hints_after_send_completed = num_shards * num_dest_nodes
        # after hints were sent to all nodes, the number of files should be 1 per shard per destination
        assert self.get_num_of_hint_files(node) <= hints_after_send_completed, "Waiting until the number of hint files" \
                                                                               " will be %s." % hints_after_send_completed
        assert self.hints_sending_in_progress() is False, "Waiting until Prometheus hints counter will not change"

    def verify_no_drops_and_errors(self, starting_from):
        q_dropped = "sum(rate(scylla_hints_manager_dropped{}[1m]))"
        q_errors = "sum(rate(scylla_hints_manager_errors{}[1m]))"
        queries_to_check = [q_dropped, q_errors]
        for query in queries_to_check:
            results = self.prometheus_db.query(query=query, start=starting_from, end=time.time())
            err_msg = "There were hint manager %s detected during the test!" % (
                "drops" if "dropped" in query else "errors")
            assert any((float(v[1]) for v in results[0]["values"])) is False, err_msg

    def get_data_set_size(self, cs_cmd):  # pylint: disable=inconsistent-return-statements
        """:returns value of n in stress comand, that is approximation and currently doesn't take in consideration
            column size definitions if they present in the command
        """
        try:
            return int(re.search(r"n=(\d+) ", cs_cmd).group(1))
        except Exception:  # pylint: disable=broad-except
            self.fail("Unable to get data set size from cassandra-stress command: %s" % cs_cmd)
            return None

    @retrying(n=60, sleep_time=60, allowed_exceptions=(AssertionError,))
    def wait_data_dir_reaching(self, size, node):
        query = '(sum(node_filesystem_size_bytes{{mountpoint="{0.scylla_dir}", ' \
                'instance=~"{1.private_ip_address}"}})-sum(node_filesystem_avail_bytes{{mountpoint="{0.scylla_dir}", ' \
                'instance=~"{1.private_ip_address}"}}))'.format(self, node)
        res = self.prometheus_db.query(query=query, start=time.time(), end=time.time())
        assert res, "No results from Prometheus"
        used = int(res[0]["values"][0][1]) / (2 ** 10)
        assert used >= size, f"Waiting for Scylla data dir to reach '{size}', " \
                             f"current size is: '{used}'"

    def check_latency_during_ops(self):
        results_analyzer = LatencyDuringOperationsPerformanceAnalyzer(es_index=self._test_index,
                                                                      es_doc_type=self._es_doc_type,
                                                                      email_recipients=self.params.get(
                                                                          'email_recipients'),
                                                                      events=get_events_grouped_by_category(
                                                                          _registry=self.events_processes_registry))
        with open(self.latency_results_file, encoding="utf-8") as file:
            latency_results = json.load(file)
        self.log.debug('latency_results were loaded from file %s and its result is %s',
                       self.latency_results_file, latency_results)
        if latency_results and self.create_stats:
            latency_results = calculate_latency(latency_results)
            with open(self.latency_results_file, 'w', encoding="utf-8") as file:
                json.dump(latency_results, file)
            self.log.debug('collected latency values are: %s', latency_results)
            self.update({"latency_during_ops": latency_results})

            self.update_test_details()
            results_analyzer.check_regression(test_id=self._test_id, data=latency_results)

    def check_regression(self):
        results_analyzer = PerformanceResultsAnalyzer(es_index=self._test_index,
                                                      es_doc_type=self._es_doc_type,
                                                      email_recipients=self.params.get('email_recipients'),
                                                      events=get_events_grouped_by_category(
                                                          _registry=self.events_processes_registry,
                                                          limit=self.params.get('events_limit_in_email'))
                                                      )
        is_gce = bool(self.params.get('cluster_backend') == 'gce')
        try:
            results_analyzer.check_regression(self._test_id, is_gce,
                                              email_subject_postfix=self.params.get('email_subject_postfix'),
                                              use_wide_query=True)
        except Exception as ex:  # pylint: disable=broad-except
            self.log.exception('Failed to check regression: %s', ex)

    def check_regression_with_baseline(self, subtest_baseline):
        results_analyzer = PerformanceResultsAnalyzer(es_index=self._test_index,
                                                      es_doc_type=self._es_doc_type,
                                                      email_recipients=self.params.get('email_recipients'),
                                                      events=get_events_grouped_by_category(
                                                          _registry=self.events_processes_registry,
                                                          limit=self.params.get('events_limit_in_email'))
                                                      )
        is_gce = bool(self.params.get('cluster_backend') == 'gce')
        try:
            results_analyzer.check_regression_with_subtest_baseline(self._test_id,
                                                                    base_test_id=self.test_config.test_id(),
                                                                    subtest_baseline=subtest_baseline,
                                                                    is_gce=is_gce)
        except Exception as ex:  # pylint: disable=broad-except
            self.log.exception('Failed to check regression: %s', ex)

    def check_regression_multi_baseline(self, subtests_info=None,  # pylint: disable=inconsistent-return-statements
                                        metrics=None, email_subject=None):
        results_analyzer = PerformanceResultsAnalyzer(es_index=self._test_index,
                                                      es_doc_type=self._es_doc_type,
                                                      email_recipients=self.params.get('email_recipients'),
                                                      events=get_events_grouped_by_category(
                                                          _registry=self.events_processes_registry,
                                                          limit=self.params.get('events_limit_in_email'))
                                                      )
        if email_subject is None:
            email_subject = ('Performance Regression Compare Results - {test.test_name} - '
                             '{test.software.scylla_server_any.version.as_string}')
            email_postfix = self.params.get('email_subject_postfix')
            if email_postfix:
                email_subject += ' - ' + email_postfix
        try:
            return results_analyzer.check_regression_multi_baseline(
                self._create_test_id(doc_id_with_timestamp=False),
                metrics=metrics,
                subtests_info=subtests_info,
                subject=email_subject)
        except Exception as ex:  # pylint: disable=broad-except
            self.log.exception('Failed to check regression: %s', ex)
            return False

    def check_specified_stats_regression(self, stats):
        perf_analyzer = SpecifiedStatsPerformanceAnalyzer(es_index=self._test_index,
                                                          es_doc_type=self._es_doc_type,
                                                          email_recipients=self.params.get('email_recipients'),
                                                          events=get_events_grouped_by_category(
                                                              _registry=self.events_processes_registry))
        try:
            perf_analyzer.check_regression(self._test_id, stats)
        except Exception as ex:  # pylint: disable=broad-except
            self.log.exception('Failed to check regression: %s', ex)

    def wait_no_compactions_running(self, n=80, sleep_time=60):  # pylint: disable=invalid-name
        # Wait for up to 80 mins that there are no running compactions
        @retrying(n=n, sleep_time=sleep_time, allowed_exceptions=(AssertionError,))
        def is_compactions_done():
            compaction_query = "sum(scylla_compaction_manager_compactions{})"
            now = time.time()
            results = self.prometheus_db.query(query=compaction_query, start=now - 60, end=now)
            self.log.debug("scylla_compaction_manager_compactions: {results}".format(**locals()))
            assert results or results == [], "No results from Prometheus"
            # if all are zeros the result will be False, otherwise there are still compactions
            if results:
                assert any((float(v[1]) for v in results[0]["values"])) is False, \
                    "Waiting until all compactions settle down"

        is_compactions_done()

    def metric_has_data(self, metric_query, n=80, sleep_time=60, ):  # pylint: disable=invalid-name
        """
        wait for any prometheus metric to have data in it

        example: wait 5 mins for cassandra stress to start writing:

            self.metric_has_data(metric_query='collectd_cassandra_stress_write_gauge{type="ops"}', n=5)

        :param metric_query:
        :param n: number of loop to try
        :param sleep_time: sleep time between each loop
        :return: None
        """

        @retrying(n=n, sleep_time=sleep_time, allowed_exceptions=(AssertionError,))
        def is_metric_has_data():
            now = time.time()
            results = self.prometheus_db.query(query=metric_query, start=now - 60, end=now)
            self.log.debug("metric_has_data: %s", results)
            assert results, "No results from Prometheus"
            if results:
                assert any((float(v[1]) for v in results[0]["values"])) > 0, f"{metric_query} didn't has data in it"

        is_metric_has_data()

    def run_fstrim_on_all_db_nodes(self):
        """
        This function will run fstrim command all db nodes in the cluster to clear any bad state of the disks.
        :return:
        """
        self.db_cluster.fstrim_scylla_disks_on_nodes()

    @silence()
    def collect_logs(self) -> None:
        if not self.params.get("collect_logs"):
            self.log.warning("Collect logs is disabled")
            return

        self.log.info("Start collect logs...")
        logs_dict = {"db_cluster_log": "",
                     "loader_log": "",
                     "monitoring_log": "",
                     "prometheus_data": "",
                     "monitoring_stack": "",
                     "siren_manager": "",
                     }
        storage_dir = os.path.join(self.logdir, "collected_logs")
        os.makedirs(storage_dir, exist_ok=True)
        self.log.info("Storage dir is %s", storage_dir)

        clusters = ({"name": "db_cluster",
                     "nodes": self.db_cluster and self.db_cluster.nodes,
                     "collector": ScyllaLogCollector,
                     "logname": "db_cluster_log", },
                    {"name": "loaders",
                     "nodes": self.loaders and self.loaders.nodes,
                     "collector": LoaderLogCollector,
                     "logname": "loader_log", },
                    {"name": "monitors",
                     "nodes": self.monitors and self.monitors.nodes,
                     "collector": MonitorLogCollector,
                     "logname": "monitoring_log", },
                    {"name": "siren_manager",
                     "nodes": self.db_cluster and self.db_cluster.manager_instance,
                     "collector": SirenManagerLogCollector,
                     "logname": "monitoring_log", },
                    {"name": "k8s_cluster_api",
                     "nodes": [],
                     "collector": KubernetesAPIServerLogCollector,
                     "logname": "k8s_log", },
                    {"name": "k8s_cluster",
                     "nodes": [],
                     "collector": KubernetesLogCollector,
                     "logname": "k8s_log", },
                    )

        for cluster in clusters:
            if not cluster["nodes"]:
                continue
            with silence(parent=self, name=f"Collect and publish {cluster['name']} logs"):
                collector = cluster["collector"](cluster["nodes"], self.test_config.test_id(), storage_dir, self.params)
                if s3_link := collector.collect_logs(self.logdir):
                    self.log.info(s3_link)
                    logs_dict[cluster["logname"]] = s3_link
                else:
                    self.log.warning("There are no logs for %s uploaded", cluster["name"])
        self.argus_collect_logs(logs_dict)

        if self.create_stats:
            with silence(parent=self, name="Publish log links"):
                self.update({"test_details": {"log_files": logs_dict, }, })

        self.log.info("Logs collected. Run command `hydra investigate show-logs %s' to get links",
                      self.test_config.test_id())

    @silence()
    def get_test_failures(self):
        """
            Print to logging in case of failure or error in unittest
            since tearDown can take a while, or even fail on it's own, we want to know fast what the failure/error is.
            applied the idea from
            https://stackoverflow.com/questions/4414234/getting-pythons-unittest-results-in-a-teardown-method/39606065#39606065
            :returns tuple(error, test_failure)
        """
        if hasattr(self, '_outcome'):  # Python 3.4+
            result = self.defaultTestResult()  # these 2 methods have no side effects
            self._feedErrorsToResult(result, self._outcome.errors)  # pylint: disable=no-member
        else:  # Python 3.2 - 3.3 or 3.0 - 3.1 and 2.7
            result = getattr(self, '_outcomeForDoCleanups', self._resultForDoCleanups)  # pylint: disable=no-member
        for error in result.errors + result.failures:
            if len(error) > 1 and error[1]:
                TestFrameworkEvent(
                    source=self.__class__.__name__,
                    source_method=error[0],
                    message=error[1],
                    severity=Severity.ERROR,
                ).publish_or_dump(default_logger=self.log)

    def stop_all_nodes_except_for(self, node):
        self.log.debug("Stopping all nodes except for: {}".format(node.name))

        for c_node in [n for n in self.db_cluster.nodes if n != node]:
            self.log.debug("Stopping node: {}".format(c_node.name))
            c_node.stop_scylla_server()

    def start_all_nodes(self):

        self.log.debug("Starting all nodes")
        # restarting all nodes twice in order to pervent no-seed node issues
        for c_node in self.db_cluster.nodes * 2:
            self.log.debug("Starting node: {} ({})".format(c_node.name, c_node.public_ip_address))
            c_node.start_scylla_server(verify_up=False)
            time.sleep(10)
        self.log.debug("Wait DB is up after all nodes were started")
        for c_node in self.db_cluster.nodes:
            c_node.wait_db_up()

    def start_all_nodes_except_for(self, node):
        self.log.debug("Starting all nodes except for: {}".format(node.name))
        node_list = [n for n in self.db_cluster.nodes if n != node]

        # Start down seed nodes first, if exists
        for c_node in [n for n in node_list if n.is_seed]:
            self.log.debug("Starting seed node: {}".format(c_node.name))
            c_node.start_scylla_server()

        for c_node in [n for n in node_list if not n.is_seed]:
            self.log.debug("Starting non-seed node: {}".format(c_node.name))
            c_node.start_scylla_server()

        node.wait_db_up()

    def get_used_capacity(self, node) -> float:  # pylint: disable=too-many-locals
        # node_filesystem_size_bytes{
        #     mountpoint="/var/lib/scylla", instance=~".*?10.0.79.46.*?"}-node_filesystem_avail_bytes{
        #         mountpoint="/var/lib/scylla", instance=~".*?10.0.79.46.*?"}
        fs_size_metric = 'node_filesystem_size_bytes'
        fs_size_metric_old = 'node_filesystem_size'
        avail_size_metric = 'node_filesystem_avail_bytes'
        avail_size_metric_old = 'node_filesystem_avail'

        instance_filter = f'instance=~".*?{node.private_ip_address}.*?"'

        capacity_query_postfix = f'{{mountpoint="{self.scylla_dir}", {instance_filter}}}'
        filesystem_capacity_query = f'{fs_size_metric}{capacity_query_postfix}'

        used_capacity_query = f'{filesystem_capacity_query}-{avail_size_metric}{capacity_query_postfix}'

        self.log.debug("filesystem_capacity_query: %s", filesystem_capacity_query)

        fs_size_res = self.prometheus_db.query(query=filesystem_capacity_query,
                                               start=int(time.time()) - 5, end=int(time.time()))
        assert fs_size_res, "No results from Prometheus"
        if not fs_size_res[0]:  # if no returned values - try the old metric names.
            filesystem_capacity_query = f'{fs_size_metric_old}{capacity_query_postfix}'
            used_capacity_query = f'{filesystem_capacity_query}-{avail_size_metric_old}{capacity_query_postfix}'
            self.log.debug("filesystem_capacity_query: %s", filesystem_capacity_query)
            fs_size_res = self.prometheus_db.query(query=filesystem_capacity_query, start=int(time.time()) - 5,
                                                   end=int(time.time()))

        assert fs_size_res[0], "Could not resolve capacity query result."
        kb_size = 2 ** 10
        mb_size = kb_size * 1024
        self.log.debug("fs_size_res: {}".format(fs_size_res))
        self.log.debug("used_capacity_query: {}".format(used_capacity_query))

        used_cap_res = self.prometheus_db.query(
            query=used_capacity_query, start=int(time.time()) - 5, end=int(time.time()))
        self.log.debug("used_cap_res: {}".format(used_cap_res))

        assert used_cap_res, "No results from Prometheus"
        used_size_mb = float(used_cap_res[0]["values"][0][1]) / float(mb_size)
        used_size_gb = float(used_size_mb / 1024)
        self.log.debug(
            "The used filesystem capacity on node {} is: {} MB/ {} GB".format(node.public_ip_address, used_size_mb,
                                                                              used_size_gb))
        return used_size_mb

    def print_nodes_used_capacity(self):
        for node in self.db_cluster.nodes:
            used_capacity = self.get_used_capacity(node=node)
            self.log.debug(
                "Node {} ({}) used capacity is: {}".format(node.name, node.private_ip_address, used_capacity))

    def get_nemesises_stats(self):
        nemesis_stats = {}
        if self.create_stats:
            nemesis_stats = self.get_doc_data(key='nemesis')
        else:
            if self.db_cluster:
                for nem in self.db_cluster.nemesis:
                    nemesis_stats.update(nem.stats)
            else:
                self.log.warning("No nemesises as cluster was not created")

        if nemesis_stats:
            for detail in nemesis_stats.values():
                for run in detail.get('runs', []):
                    run['start'] = format_timestamp(float(run['start']))
                    run['end'] = format_timestamp(float(run['end']))
                for failure in detail.get('failures', []):
                    failure['start'] = format_timestamp(float(failure['start']))
                    failure['end'] = format_timestamp(float(failure['end']))
        return nemesis_stats

    @silence()
    def save_email_data(self):
        email_data = {}

        try:
            email_data = self.get_email_data()
            self.argus_collect_screenshots(email_data)
        except Exception as exc:  # pylint: disable=broad-except
            self.log.error("Error while saving email data. Error: %s", exc)

        json_file_path = os.path.join(self.logdir, "email_data.json")

        if email_data:
            email_data["reporter"] = self.email_reporter.__class__.__name__
            self.log.debug('Save email data to file %s', json_file_path)
            self.log.debug('Email data: %s', email_data)
            save_email_data_to_file(email_data, json_file_path)

    def argus_collect_screenshots(self, email_data: dict) -> None:
        screenshot_links = email_data.get("grafana_screenshots", [])
        for link in screenshot_links:
            self.argus_test_run.run_info.results.add_screenshot(link)

        self.argus_test_run.save()

    @silence()
    def send_email(self):
        """Send email with test results on teardown

        The method is used to send email with test results.
        Child class should implement the method get_mail_data
        which return the dict with 2 required fields:
            email_template, email_subject
        """
        send_email = self.params.get('send_email')
        email_results_file = os.path.join(self.logdir, "email_data.json")
        email_data = read_email_data_from_file(email_results_file)

        if get_username() == "jenkins":
            self.log.info("Email will be sent by pipeline stage")
            return

        if send_email and email_data:
            email_data["reporter"] = self.email_reporter.__class__.__name__
            email_data['nodes'] = get_running_instances_for_email_report(self.test_id)
            try:
                if self.email_reporter:
                    self.email_reporter.send_report(email_data)
                else:
                    self.log.warning('Test is not configured to send html reports')

            except Exception as details:  # pylint: disable=broad-except
                self.log.error("Error during sending email: {}".format(details))
        else:
            self.log.warning("Email is not configured: %s or no email data: %s", send_email, email_data)

    def get_email_data(self):  # pylint: disable=no-self-use
        """prepare data to generate and send via email

        Have to return the dict which is used to build the
        html content with email template.
        Required field:
        - email_template: path to file with html template
        - email_subject: subject of email
        have to be implemented in child class.
        """
        return {}

    def all_nodes_scylla_shards(self):
        all_nodes_shards = defaultdict(list)
        for node in self.db_cluster.nodes:
            ipv6 = node.ipv6_ip_address if node.ip_address == node.ipv6_ip_address else ''
            all_nodes_shards['live_nodes'].append({'name': node.name,
                                                   'ip': f"{node.public_ip_address} | {node.private_ip_address}"
                                                         f"{f' | {ipv6}' if ipv6 else ''}",
                                                   'shards': node.scylla_shards})

        all_nodes_shards['dead_nodes'] = [asdict(node) for node in self.db_cluster.dead_nodes_list]

        return all_nodes_shards

    def _get_common_email_data(self) -> Dict[str, Any]:
        """Helper for subclasses which extracts common data for email."""

        if self.db_cluster:
            nodes_shards = self.all_nodes_scylla_shards()
        else:
            nodes_shards = defaultdict(list)

        alive_node = self._get_live_node() or None

        start_time = format_timestamp(self.start_time)
        config_file_name = ";".join(os.path.splitext(os.path.basename(cfg))[0] for cfg in self.params["config_files"])
        test_status = self.get_test_status()
        backend = self.params.get("cluster_backend")
        region_name = self.params.get('region_name') or self.params.get('gce_datacenter')
        build_id = f'#{os.environ.get("BUILD_NUMBER")}' if os.environ.get('BUILD_NUMBER', '') else ''
        scylla_version = alive_node.scylla_version_detailed if alive_node else "N/A"
        kernel_version = alive_node.kernel_version if alive_node else "N/A"

        if backend in ("aws", "aws-siren", "k8s-eks"):
            scylla_instance_type = self.params.get("instance_type_db") or "Unknown"
        elif backend in ("gce", "gce-siren", "k8s-gke"):
            scylla_instance_type = self.params.get("gce_instance_type_db") or "Unknown"
        elif backend in ("azure"):
            scylla_instance_type = self.params.get("azure_instance_type_db") or "Unknown"
        elif backend in ("baremetal", "docker"):
            scylla_instance_type = "N/A"
        else:
            self.log.error("Don't know how to get instance type for the '%s' backend.", backend)
            scylla_instance_type = "N/A"

        job_name = os.environ.get('JOB_NAME').split("/")[-1] if os.environ.get('JOB_NAME') else config_file_name
        restore_monitor_job_base_link = \
            "https://jenkins.scylladb.com/view/QA/job/QA-tools/job/hydra-show-monitor/parambuild/?"
        if build_id:
            sct_branch = os.environ.get("GIT_BRANCH", "master").rsplit("/", maxsplit=1)[-1]
            restore_monitor_job_base_link += f"sct_branch={sct_branch}&"

        return {"backend": backend,
                "build_id": os.environ.get('BUILD_NUMBER', ''),
                "job_url": os.environ.get("BUILD_URL"),
                "end_time": format_timestamp(time.time()),
                "events_summary": self.get_event_summary(),
                "last_events": get_events_grouped_by_category(limit=100, _registry=self.events_processes_registry),
                "nodes": [],
                "number_of_db_nodes": self.params.get('n_db_nodes'),
                "region_name": region_name,
                "scylla_instance_type": scylla_instance_type,
                "scylla_version": scylla_version,
                "live_nodes_shards": nodes_shards.get('live_nodes'),
                "dead_nodes_shards": nodes_shards.get('dead_nodes'),
                "kernel_version": kernel_version,
                "start_time": start_time,
                "subject": f"{test_status}: {os.environ.get('JOB_NAME') or config_file_name}{build_id}: {start_time}",
                "job_name": job_name,
                "config_files": self.params["config_files"],
                "test_id": self.test_id,
                "test_name": self.id(),
                "test_status": test_status,
                "username": get_username(),
                "shard_awareness_driver": self.is_shard_awareness_driver,
                "restore_monitor_job_base_link": restore_monitor_job_base_link}

    @silence()
    def tag_ami_with_result(self):
        if self.params.get('cluster_backend') != 'aws' or not self.params.get('tag_ami_with_result'):
            return
        test_result = 'PASSED' if self.get_test_status() == 'SUCCESS' else 'ERROR'
        job_base_name = os.environ.get('JOB_BASE_NAME', 'UnknownJob')
        ami_id = self.params.get('ami_id_db_scylla').split()[0]
        region_name = self.params.region_names[0]

        tag_ami(ami_id=ami_id, region_name=region_name, tags_dict={"JOB:{}".format(job_base_name): test_result})

    def get_test_results(self, source, severity=None):
        output = []
        for result in getattr(self, '_results', []):
            if result.get('source', None) != source:
                continue
            if severity is not None and severity != result['severity']:
                continue
            output.append(result['message'])
        return output

    def _get_live_node(self) -> Optional[BaseNode]:  # pylint: disable=inconsistent-return-statements
        if not self.db_cluster or not self.db_cluster.nodes:
            self.log.error("Cluster object was not initialized")
            return None
        parallel_obj = ParallelObject(objects=self.db_cluster.nodes)
        parallel_obj_results = parallel_obj.run(self._check_ssh_node_connectivity,
                                                ignore_exceptions=True)

        for result in parallel_obj_results:
            if not result.exc:
                return result.result
        return None

    @staticmethod
    def _check_ssh_node_connectivity(node: BaseNode) -> Optional[BaseNode]:
        node.wait_ssh_up(verbose=False, timeout=50)

        return node

    @property
    def is_shard_awareness_driver(self) -> bool:
        all_events = get_events_grouped_by_category()
        for event_str in all_events["NORMAL"]:
            if "type=ShardAwareDriver" in event_str:
                return True
        return False
