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
import shutil
from collections import defaultdict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import random
import logging
import os
import re
import time
import traceback
import unittest
import unittest.mock
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict, Any
from uuid import uuid4
from functools import wraps, cache
import threading
import signal
import json

import botocore
import yaml
from invoke.exceptions import UnexpectedExit, Failure

from cassandra.concurrent import execute_concurrent_with_args  # pylint: disable=no-name-in-module
from cassandra import ConsistencyLevel
from cassandra.cluster import Session  # pylint: disable=no-name-in-module

from argus.client.sct.client import ArgusSCTClient
from argus.client.base import ArgusClientError
from argus.client.sct.types import Package, EventsInfo, LogLink
from argus.common.enums import TestStatus
from sdcm import nemesis, cluster_docker, cluster_k8s, cluster_baremetal, db_stats, wait
from sdcm.cluster import BaseCluster, NoMonitorSet, SCYLLA_DIR, TestConfig, UserRemoteCredentials, BaseLoaderSet, BaseMonitorSet, \
    BaseScyllaCluster, BaseNode, MINUTE_IN_SEC
from sdcm.cluster_azure import ScyllaAzureCluster, LoaderSetAzure, MonitorSetAzure
from sdcm.cluster_gce import ScyllaGCECluster
from sdcm.cluster_gce import LoaderSetGCE
from sdcm.cluster_gce import MonitorSetGCE
from sdcm.cluster_aws import CassandraAWSCluster
from sdcm.cluster_aws import ScyllaAWSCluster
from sdcm.cluster_aws import LoaderSetAWS
from sdcm.cluster_aws import MonitorSetAWS
from sdcm.cluster_k8s import mini_k8s, gke, eks
from sdcm.cluster_k8s.eks import MonitorSetEKS
from sdcm.cql_stress_cassandra_stress_thread import CqlStressCassandraStressThread
from sdcm.mgmt import get_scylla_manager_tool
from sdcm.provision.aws.capacity_reservation import SCTCapacityReservation
from sdcm.kafka.kafka_cluster import LocalKafkaCluster
from sdcm.kafka.kafka_producer import KafkaProducerThread, KafkaValidatorThread
from sdcm.provision.azure.provisioner import AzureProvisioner
from sdcm.provision.network_configuration import ssh_connection_ip_type
from sdcm.provision.provisioner import provisioner_factory
from sdcm.provision.helpers.certificate import create_ca, update_certificate, cleanup_ssl_config
from sdcm.reporting.elastic_reporter import ElasticRunReporter
from sdcm.reporting.tooling_reporter import PythonDriverReporter
from sdcm.scan_operation_thread import ScanOperationThread
from sdcm.nosql_thread import NoSQLBenchStressThread
from sdcm.scylla_bench_thread import ScyllaBenchThread
from sdcm.cassandra_harry_thread import CassandraHarryThread
from sdcm.teardown_validators import teardown_validators_list
from sdcm.tombstone_gc_verification_thread import TombstoneGcVerificationThread
from sdcm.utils.alternator.consts import NO_LWT_TABLE_NAME
from sdcm.utils.aws_kms import AwsKms
from sdcm.utils.aws_region import AwsRegion
from sdcm.utils.aws_utils import init_monitoring_info_from_params, get_ec2_services, \
    get_common_params, init_db_info_from_params, ec2_ami_get_root_device_name
from sdcm.utils.ci_tools import get_job_name, get_job_url
from sdcm.utils.common import format_timestamp, wait_ami_available, \
    download_dir_from_cloud, get_post_behavior_actions, get_testrun_status, download_encrypt_keys, rows_to_list, \
    make_threads_be_daemonic_by_default, ParallelObject, clear_out_all_exit_hooks, change_default_password, \
    parse_python_thread_command, get_data_dir_path
from sdcm.utils.cql_utils import cql_quote_if_needed
from sdcm.utils.database_query_utils import PartitionsValidationAttributes, fetch_all_rows
from sdcm.utils.features import is_tablets_feature_enabled
from sdcm.utils.get_username import get_username
from sdcm.utils.decorators import log_run_info, retrying, measure_time, optional_stage
from sdcm.utils.git import get_git_commit_id, get_git_status_info
from sdcm.utils.ldap import LDAP_USERS, LDAP_PASSWORD, LDAP_ROLE, LDAP_BASE_OBJECT, \
    LdapConfigurationError, LdapServerType
from sdcm.utils.log import configure_logging, handle_exception
from sdcm.utils.issues import SkipPerIssues
from sdcm.db_stats import PrometheusDBStats
from sdcm.results_analyze import PerformanceResultsAnalyzer, SpecifiedStatsPerformanceAnalyzer, \
    LatencyDuringOperationsPerformanceAnalyzer
from sdcm.sct_config import init_and_verify_sct_config
from sdcm.sct_events import Severity
from sdcm.sct_events.setup import start_events_device, stop_events_device, enable_default_filters
from sdcm.sct_events.system import InfoEvent, TestFrameworkEvent, TestResultEvent, TestTimeoutEvent
from sdcm.sct_events.file_logger import get_events_grouped_by_category, get_logger_event_summary
from sdcm.sct_events.events_analyzer import stop_events_analyzer
from sdcm.sct_events.grafana import start_posting_grafana_annotations
from sdcm.stress_thread import CassandraStressThread, get_timeout_from_stress_cmd
from sdcm.gemini_thread import GeminiStressThread
from sdcm.utils.log_time_consistency import DbLogTimeConsistencyAnalyzer
from sdcm.utils.net import get_my_ip, get_sct_runner_ip
from sdcm.utils.operations_thread import ThreadParams
from sdcm.utils.replication_strategy_utils import LocalReplicationStrategy, NetworkTopologyReplicationStrategy
from sdcm.utils.tablets.common import TabletsConfiguration
from sdcm.utils.threads_and_processes_alive import gather_live_processes_and_dump_to_file, \
    gather_live_threads_and_dump_to_file
from sdcm.utils.version_utils import (
    get_relocatable_pkg_url,
    ComparableScyllaVersion,
)
from sdcm.ycsb_thread import YcsbStressThread
from sdcm.ndbench_thread import NdBenchStressThread
from sdcm.kcl_thread import KclStressThread, CompareTablesSizesThread
from sdcm.stress.latte_thread import LatteStressThread
from sdcm.localhost import LocalHost
from sdcm.cdclog_reader_thread import CDCLogReaderThread
from sdcm.logcollector import (
    KubernetesAPIServerLogCollector,
    KubernetesLogCollector,
    KubernetesMustGatherLogCollector,
    LoaderLogCollector,
    MonitorLogCollector,
    BaseSCTLogCollector,
    PythonSCTLogCollector,
    ScyllaLogCollector,
    SirenManagerLogCollector,
)
from sdcm.send_email import build_reporter, save_email_data_to_file
from sdcm.utils import alternator
from sdcm.utils.profiler import ProfilerFactory
from sdcm.remote import RemoteCmdRunnerBase
from sdcm.utils.gce_utils import get_gce_compute_instances_client
from sdcm.utils.auth_context import temp_authenticator
from sdcm.keystore import KeyStore
from sdcm.utils.latency import calculate_latency, analyze_hdr_percentiles
from sdcm.utils.csrangehistogram import CSHistogramTagTypes, CSWorkloadTypes, make_cs_range_histogram_summary, \
    make_cs_range_histogram_summary_by_interval
from sdcm.utils.raft.common import validate_raft_on_nodes
from sdcm.commit_log_check_thread import CommitLogCheckThread
from sdcm.kafka.kafka_consumer import KafkaCDCReaderThread
from test_lib.compaction import CompactionStrategy


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

PYTHON_THREAD_LIST = (KafkaCDCReaderThread, KafkaProducerThread, KafkaValidatorThread)


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

    def __init__(self, parent=None, name: str = None, verbose: bool = False, raise_error_event=True):
        self.parent = parent
        self.name = name
        self.verbose = verbose
        self.raise_error_event = raise_error_event
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
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
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

    def _store_test_result(self, parent, exc_val, exc_tb, name):
        if not self.raise_error_event:
            return

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
    except Exception:  # pylint: disable=broad-except  # noqa: BLE001
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
    loaders: Union[BaseLoaderSet, LoaderSetAWS, LoaderSetGCE] = None
    db_cluster: Union[BaseCluster, BaseScyllaCluster] = None

    @property
    def k8s_cluster(self):
        return self.k8s_clusters[0] if getattr(self, 'k8s_clusters', None) else None

    def __init__(self, *args, **kwargs):  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
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

        ip_ssh_connections = ssh_connection_ip_type(self.params)
        self.test_config.set_ip_ssh_connections(ip_ssh_connections)
        self._init_test_duration()
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
        if cluster_backend in ('aws', 'k8s-eks'):
            self.test_config.set_multi_region(
                (self.params.get("simulated_regions") or 0) > 1 or len(self.params.region_names) > 1)
        elif cluster_backend in ('gce', 'k8s-gke'):
            self.test_config.set_multi_region(
                (self.params.get("simulated_regions") or 0) > 1 or len(self.params.gce_datacenters) > 1)
        elif cluster_backend == "azure":
            self.test_config.set_multi_region((self.params.get("simulated_regions") or 0) > 1)

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

        self.init_argus_run()
        self.argus_heartbeat_stop_signal = self.start_argus_heartbeat_thread()
        PythonDriverReporter(argus_client=self.test_config.argus_client()).report()
        self.localhost = self._init_localhost()

        if self.params.get("logs_transport") == 'syslog-ng':
            self.test_config.configure_syslogng(self.localhost)

        self.alternator: alternator.api.Alternator = alternator.api.Alternator(sct_params=self.params)
        self.alternator = alternator.api.Alternator(sct_params=self.params)

        if self.params.get("use_ldap"):
            self._init_ldap()

        self.partitions_attrs: PartitionsValidationAttributes | None = self._init_data_validation()
        # Cover multi-tenant configuration. Prevent event device double initiate
        start_events_device(log_dir=self.logdir,
                            _registry=getattr(self, "_registry", None) or self.events_processes_registry)
        enable_default_filters(sct_config=self.params)

        self.skip_test_stages = defaultdict(lambda: False, self.params.get('skip_test_stages') or {})

        time.sleep(0.5)
        InfoEvent(message=f"TEST_START test_id={self.test_config.test_id()}").publish()
        self.bisect_ref_value = None
        self.bisect_result_value = None
        self.stress_cmd = self.params.get('stress_cmd')

    def _init_test_duration(self):
        self._stress_duration: int = self.params.get('stress_duration')
        self._prepare_stress_duration: int = self.params.get('prepare_stress_duration')
        if self._stress_duration:
            self._duration = int(self._prepare_stress_duration) + int(self._stress_duration) + \
                self.test_config.TEST_WARMUP_TEARDOWN

            self.log.info("SCT Test duration: %s; Stress duration: %s; Prepare duration: %s",
                          self._duration, self._stress_duration, self._prepare_stress_duration)
        else:
            self._duration = self.params.get("test_duration")

    def init_argus_run(self):
        try:
            self.test_config.init_argus_client(self.params)
            git_status = get_git_status_info()
            self.test_config.argus_client().submit_sct_run(
                job_name=get_job_name(),
                job_url=get_job_url(),
                started_by=get_username(),
                commit_id=git_status.get('branch.oid', get_git_commit_id()),
                origin_url=git_status.get('upstream.url'),
                branch_name=git_status.get('branch.upstream'),
                sct_config=self.params.dict(),
            )
            self.log.info("Submitted Argus TestRun with test id %s", self.test_config.argus_client().run_id)
            self.test_config.argus_client().set_sct_runner(
                public_ip=get_sct_runner_ip(),
                private_ip=get_my_ip(),
                region="undefined_region",
                backend=self.params.get("cluster_backend"))
            self.log.info("sct_runner info in Argus TestRun is updated")
        except ArgusClientError:
            self.log.error("Failed to submit data to Argus", exc_info=True)

    def start_argus_heartbeat_thread(self) -> threading.Event:
        def send_argus_heartbeat(client: ArgusSCTClient, stop_signal: threading.Event):
            if isinstance(client, unittest.mock.MagicMock):
                return
            fail_count = 0
            while not stop_signal.is_set():
                if fail_count > 5:
                    break
                try:
                    client.sct_heartbeat()
                except Exception:  # pylint: disable=broad-except  # noqa: BLE001
                    self.log.warning("Failed to submit heartbeat to argus, Try #%s", fail_count + 1)
                    fail_count += 1
                time.sleep(30.0)

        thread_stop_signal = threading.Event()
        thread = threading.Thread(name="argus-heartbeat",
                                  target=send_argus_heartbeat,
                                  kwargs={"client": self.test_config.argus_client(), "stop_signal": thread_stop_signal})
        thread.daemon = True
        thread.start()

        return thread_stop_signal

    def argus_update_status(self, status: TestStatus):
        try:
            self.test_config.argus_client().set_sct_run_status(new_status=status)
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error saving test status to Argus", exc_info=True)

    def generate_scylla_server_package(self) -> Package:
        """
            Used for offline tests for tracking scylla versions in Argus.
        """
        scylla_version = self.db_cluster.nodes[0].scylla_version_detailed
        # pylint: disable=line-too-long
        expr = re.compile(
            r'(?P<version>(?P<main>[\w.~]+)-(0.)?(?P<date>[0-9]{8,8}).(?P<commit>\w+).) with build-id (?P<build_id>[\dabcdef]+)')
        version_dict = expr.match(scylla_version).groupdict()

        return Package(
            name="scylla-server",
            version=version_dict.get("main", "#NO_VERSION").replace("~", "."),
            revision_id=version_dict.get("commit", "#NO_COMMIT"),
            build_id=version_dict.get("build_id", "#NO_BUILDID"),
            date=version_dict.get("date", "#NO_DATE"),
        )

    def argus_collect_packages(self):
        try:
            self.log.info("Collecting packages for Argus...")
            packages_to_submit = []
            versions = self.get_scylla_versions()
            kernel_version = self.db_cluster.nodes[0].kernel_version
            kernel_package = Package(name="kernel", date="", version=kernel_version, revision_id="", build_id="")
            packages_to_submit.append(kernel_package)
            for package_name, package_info in versions.items():
                package = Package(name=package_name, date=package_info.get("date", "#NO_DATE"),
                                  version=package_info.get("version", "#NO_VERSION"),
                                  revision_id=package_info.get("commit_id", "#NO_COMMIT"),
                                  build_id=package_info.get("build_id", "#NO_BUILDID"))
                packages_to_submit.append(package)
            if len(versions) == 0:
                packages_to_submit.append(self.generate_scylla_server_package())

            packages_to_submit.extend(self.generate_operator_packages())

            self.log.info("Saving collected packages...")
            self.test_config.argus_client().submit_packages(packages_to_submit)
        except Exception:  # pylint: disable=broad-except
            self.log.error("Unable to collect package versions for Argus - skipping...", exc_info=True)

    def argus_collect_manager_version(self):
        # NOTE: K8S runs manager server in a pod on a non-monitor node.
        #       Moreover, manager version is known in this case by the docker image tags.
        if self.monitors.nodes and self.params.get('use_mgmt') and not self.k8s_clusters:
            manager_tool = get_scylla_manager_tool(manager_node=self.monitors.nodes[0])
            self.log.info("Saving manager version in Argus...")
            # sctool.version output:
            #   [['Client version: 3.2.8-0.20240517.5f324acd2'], ['Server version: 3.2.8-0.20240517.5f324acd2']]
            client_version, server_version, *_ = (version[0].split(":")[1].strip()
                                                  for version in manager_tool.sctool.version)
            manager_packages = [Package(name="scylla-manager-client", date="",
                                        version=client_version, revision_id="", build_id=""),
                                Package(name="scylla-manager-server", date="",
                                        version=server_version, revision_id="", build_id="")]
            self.test_config.argus_client().submit_packages(manager_packages)

    def generate_operator_packages(self) -> list[Package]:
        operator_packages = []
        if self.k8s_clusters:
            operator_helm_chart_version = self.k8s_clusters[0].scylla_operator_chart_version
            operator_packages.append(Package(name="operator-chart", date="",
                                             version=operator_helm_chart_version,
                                             revision_id="", build_id=""))
            operator_image_version = self.k8s_clusters[0].get_operator_image()
            operator_packages.append(Package(name="operator-image", date="",
                                             version=operator_image_version,
                                             revision_id="", build_id=""))

            operator_helm_repo = self.params.get('k8s_scylla_operator_helm_repo')
            operator_packages.append(Package(name="operator-helm-repo", date="",
                                             version=operator_helm_repo,
                                             revision_id="", build_id=""))
        return operator_packages

    def argus_get_scylla_version(self):
        try:
            self.log.info("Collection Scylla version for argus...")
            version_regex = re.compile(r'(([\w.~]+)-(0.)?([0-9]{8,8}).(\w+).)')
            version_str = self.db_cluster.nodes[0].get_scylla_binary_version()
            if version_str and (match := version_regex.match(version_str)):
                version = match.group(2)
                self.test_config.argus_client().update_scylla_version(version=version)
                return
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error getting scylla version for argus", exc_info=True)

        TestFrameworkEvent(
            source=self.__class__.__name__,
            source_method="argus_get_scylla_version",
            message="Failed to get scylla version for argus",
            severity=Severity.ERROR,
        )

    def _is_test_error(self, events: dict[str, list[str]]) -> bool:
        """
            Test whether or not the result of the test can be considered as a TEST_ERROR
            status for Argus. TEST_ERROR status is used for times when the SCT failure is caused by
            either infrastructure or problems within SCT itself.

        """
        def check_error(event: str):
            errors = [
                re.compile(r"SpotTerminationEvent", re.IGNORECASE),
                re.compile(r"ZonalAllocationFailed", re.IGNORECASE),
                re.compile(r"source=[\w]+.SetUp\(\).+exception=403 FORBIDDEN QUOTA_EXCEEDED",
                           re.IGNORECASE | re.DOTALL),
                re.compile(r"source=[\w]+.SetUp\(\).+InsufficientInstanceCapacity", re.IGNORECASE | re.DOTALL),
            ]
            for error in errors:
                if error.search(event):
                    return True
            return False

        return any(check_error(e) for e in [*events.get("CRITICAL"), *events.get("ERROR")])

    def argus_finalize_test_run(self):
        try:
            stat_map = {
                "SUCCESS": TestStatus.PASSED,
                "FAILED": TestStatus.FAILED
            }

            last_events_limit = 100
            last_events = get_events_grouped_by_category(
                limit=last_events_limit, _registry=self.events_processes_registry)
            events_sorted = []
            events_summary = get_logger_event_summary()
            for severity, messages in last_events.items():
                event_category = EventsInfo(
                    severity=severity, total_events=events_summary.get(severity, 0), messages=messages)
                events_sorted.append(event_category)
            self.test_config.argus_client().submit_events(events_sorted)

            test_status = stat_map.get(self.get_test_status(), TestStatus.FAILED)
            if test_status == TestStatus.FAILED and self._is_test_error(last_events):
                test_status = TestStatus.TEST_ERROR
            self.argus_update_status(test_status)

            self.test_config.argus_client().finalize_sct_run()
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error committing test events to Argus", exc_info=True)

    def argus_collect_logs(self, log_links: dict[str, list[str] | str]):
        try:
            logs_to_save = []
            for name, link in log_links.items():
                argus_link = LogLink(log_name=name, log_link=link)
                logs_to_save.append(argus_link)
            self.test_config.argus_client().submit_sct_logs(logs_to_save)
        except Exception:  # pylint: disable=broad-except
            self.log.error("Error saving logs to Argus", exc_info=True)

    def argus_collect_gemini_results(self):
        try:
            # pylint: disable=no-member
            if not hasattr(self, "gemini_results"):
                return

            if self.loaders:
                gemini_version = self.loaders.gemini_version
            else:
                self.log.warning("Failed to get gemini version as loader instance is not created")
                gemini_version = ""

            seed = self.params.get("gemini_seed")
            gemini_command, *_ = self.gemini_results["cmd"]
            if not seed:
                seed_match = re.search(r"--seed (?P<seed>\d+) ", gemini_command)
                if seed_match:
                    seed = seed_match.groupdict().get("seed", -1)
                else:
                    seed = -1

            results = self.gemini_results["results"]
            results = results[0] if len(results) > 0 else None
            if not results or not isinstance(results, dict):
                self.log.warning("Results object is not a dictionary: %s\nReplacing with empty dict.", results)
                results = {}

            self.test_config.argus_client().submit_gemini_results({
                "gemini_command": "\n".join(self.gemini_results["cmd"]),
                "gemini_read_errors": results.get("read_errors", -1),
                "gemini_read_ops": results.get("read_ops", -1),
                "gemini_seed": seed,
                "gemini_status": self.gemini_results["status"],
                "gemini_version": gemini_version,
                "gemini_write_errors": results.get("write_errors", -1),
                "gemini_write_ops": results.get("write_ops", -1),
                "oracle_node_ami_id": self.params.get("ami_id_db_oracle"),
                "oracle_node_instance_type": self.params.get("instance_type_db_oracle"),
                "oracle_node_scylla_version": self.cs_db_cluster.nodes[0].scylla_version if self.cs_db_cluster else "N/A",
                "oracle_nodes_count": self.params.get("n_test_oracle_db_nodes"),
            })
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.warning("Error submitting gemini results to argus", exc_info=True)

    def collect_ssl_conf(self):
        shutil.copytree(Path(get_data_dir_path('ssl_conf')), Path(self.logdir) / 'ssl_conf')

    def _init_data_validation(self):
        if data_validation := self.params.get('data_validation'):
            data_validation_params = yaml.safe_load(data_validation)
            return PartitionsValidationAttributes(tester=self, **data_validation_params)
        return None

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
        min_nodes_dc = min([int(nodes_num) for nodes_num in n_db_nodes.split() if int(nodes_num) > 0])
        with self.db_cluster.cql_connection_patient(self.db_cluster.nodes[0]) as session:
            # In case tablets are enabled, it's better to set RF smaller than dc-nodes-number, so decommission is allowed.
            rf_candidate = max([min_nodes_dc - 1, 1]) if is_tablets_feature_enabled(session) else min_nodes_dc
            # NOTE: use RF=3 at max to avoid problems on big setups
            return min(rf_candidate, 3)

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
    def update_certificate():
        update_certificate()

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
            'kmip_hosts' in append_scylla_yaml
        )

    def prepare_kms_host(self) -> None:
        version_supports_kms = (self.params.is_enterprise and
                                ComparableScyllaVersion(self.params.scylla_version) >= '2023.1.3')
        backend_support_kms = self.params.get('cluster_backend') in ('aws',)
        kms_configured_in_sct = self.params.get('scylla_encryption_options')
        test_uses_oracle = self.params.get("db_type") == "mixed_scylla"
        should_enable_kms = (version_supports_kms and
                             backend_support_kms and
                             not kms_configured_in_sct and
                             not test_uses_oracle and
                             not self.params.get('enterprise_disable_kms'))

        if should_enable_kms:
            self.params['scylla_encryption_options'] = "{ 'cipher_algorithm' : 'AES/ECB/PKCS5Padding', 'secret_key_strength' : 128, 'key_provider': 'KmsKeyProviderFactory', 'kms_host': 'auto'}"  # pylint: disable=line-too-long
        if not (scylla_encryption_options := self.params.get("scylla_encryption_options") or ''):
            return None
        kms_host = (yaml.safe_load(scylla_encryption_options) or {}).get("kms_host") or ''
        if 'auto' not in kms_host:
            return None
        # Create a KMS key alias in each of the regions used by the current test run
        aws_kms = AwsKms(region_names=self.params.region_names)
        alias_name = f"alias/testid-{self.test_config.test_id()}"
        aws_kms.create_alias(kms_key_alias_name=alias_name)

        # Add kms_host with the dynamically created alias to the 'append_scylla_yaml'
        append_scylla_yaml = self.params.get("append_scylla_yaml") or {}
        if "kms_hosts" not in append_scylla_yaml:
            append_scylla_yaml["kms_hosts"] = {}
        append_scylla_yaml["kms_hosts"][kms_host] = {
            'master_key': alias_name,
            'aws_use_ec2_credentials': True,
            # NOTE: the 'aws_region' will be changed per each node separately depending on it's region
            'aws_region': 'auto'
        }
        append_scylla_yaml['user_info_encryption'] = {
            'enabled': True,
            'key_provider': 'KmsKeyProviderFactory',
            'kms_host': kms_host,
        }
        append_scylla_yaml['system_info_encryption'] = {
            'enabled': True,
            'key_provider': 'KmsKeyProviderFactory',
            'kms_host': kms_host,
        }

        if SkipPerIssues("scylladb/scylla-enterprise#4246", self.params):
            # until https://github.com/scylladb/scylla-enterprise/issues/4246 would be solved
            # we should disable the enterprise file streaming feature
            append_scylla_yaml['enable_file_stream'] = False

        self.log.warning("`user_info_encryption` and `system_info_encryption` are configured to use KMS by default")
        self.params["append_scylla_yaml"] = append_scylla_yaml
        return None

    def kafka_configure(self):
        if self.kafka_cluster:
            for connector_config in self.params.get('kafka_connectors'):
                self.kafka_cluster.create_connector(db_cluster=self.db_cluster,
                                                    connector_config=connector_config)

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
        self.kafka_cluster = None
        self.k8s_clusters = []
        self.connections = []
        make_threads_be_daemonic_by_default()

        if (not self.params.get("cluster_backend").startswith("k8s") and
                any([self.params.get('client_encrypt'), self.params.get('server_encrypt')])):
            create_ca(self.localhost)

        # download rpms for update_db_packages
        if self.params.get('update_db_packages'):
            self.download_db_packages()
        if self.is_encrypt_keys_needed:
            self.download_encrypt_keys()
        self.prepare_kms_host()

        self.init_resources()

        if self.k8s_clusters and self.params.get("k8s_use_chaos_mesh"):
            for k8s_cluster in self.k8s_clusters:
                if not k8s_cluster.chaos_mesh.initialized:
                    k8s_cluster.chaos_mesh.initialize()

        def _create_db_clusters():
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
                            # TODO: Replace with strong password generation
                else:
                    self.set_system_auth_rf(db_cluster=db_cluster)

        def _create_loaders():
            if self.loaders and not self.loaders_multitenant:
                self.loaders_multitenant = [self.loaders]
            for loaders in self.loaders_multitenant:
                if loaders:
                    loaders.wait_for_init()

        def _create_gemini_cluster():
            # cs_db_cluster is created in case MIXED_CLUSTER. For example, gemini test
            if self.cs_db_cluster:
                self.init_nodes(db_cluster=self.cs_db_cluster)

        def _create_monitors():
            if self.monitors and not self.monitors_multitenant:
                self.monitors_multitenant = [self.monitors]

            if self.monitors_multitenant:
                monitors_init_in_parallel = ParallelObject(
                    timeout=3600,
                    objects=[[monitor] for monitor in self.monitors_multitenant],
                    num_workers=len(self.monitors_multitenant))
                monitors_init_in_parallel.run(
                    func=(lambda m: m.wait_for_init()),
                    unpack_objects=True, ignore_exceptions=False)

        def _db_post_validation():
            for db_cluster in self.db_clusters_multitenant:
                if db_cluster:
                    db_cluster.validate_seeds_on_all_nodes()
                    validate_raft_on_nodes(nodes=db_cluster.nodes)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_create_db_clusters),
                executor.submit(_create_loaders),
                executor.submit(_create_monitors),
                executor.submit(_create_gemini_cluster)
            ]

            for future in as_completed(futures):
                future.result()

            if self.params.get('use_ldap') and self.params.get('ldap_server_type') == LdapServerType.MS_AD:
                for loaders in self.loaders_multitenant:
                    if loaders:
                        loaders.added_password_suffix = True
                for monitors in self.monitors_multitenant:
                    monitors.added_password_suffix = True

            futures = [
                executor.submit(_db_post_validation),
                executor.submit(self.argus_collect_packages),
                executor.submit(self.argus_get_scylla_version),
                executor.submit(self.argus_collect_manager_version),
            ]

            for db_cluster in self.db_clusters_multitenant:
                if db_cluster:
                    db_cluster.start_kms_key_rotation_thread()

            for future in as_completed(futures):
                future.result()

        if self.create_stats:
            self.create_test_stats()
            # sync test_start_time with ES
            self.start_time = self.get_test_start_time()

        # cancel reuse cluster - for new nodes added during the test
        self.test_config.reuse_cluster(False)
        for monitors in self.monitors_multitenant:
            if monitors and monitors.nodes:
                self.prometheus_db_multitenant.append(
                    PrometheusDBStats(host=monitors.nodes[0].external_address))
        self.prometheus_db = (self.prometheus_db_multitenant or [None])[0]

        self.start_time = time.time()
        self.timeout_thread = self._init_test_timeout_thread()

        if self.params.get('run_commit_log_check_thread'):
            self.run_commit_log_check_thread(self.get_duration(None))

    def set_system_auth_rf(self, db_cluster=None):
        db_cluster = db_cluster or self.db_cluster
        # No need to change system tables when running via scylla-cloud
        # Also, when running a Alternator via scylla-cloud, we don't have CQL access to the cluster
        if self.params.get('db_type') == 'cloud_scylla' or self.params.get("cluster_backend") == "baremetal":
            # TODO: move this skip to siren-tools when possible
            self.log.warning("Skipping this function due this job run from Siren cloud!")
            return
        # no need to change rf if 1 node in cluster (artifact tests)
        if len(db_cluster.nodes) == 1:
            self.log.debug("Skipping change rf for system_auth on single node tests")
            return
        if self.test_config.REUSE_CLUSTER:
            self.log.debug("Skipping change rf of system_auth for reusing cluster")
            return
        if db_cluster.nodes[0].raft.is_consistent_topology_changes_enabled:
            self.log.debug("Skipping change rf of system_auth because with consistent topology auth-v2 is enabled")
            return
        self.set_ks_strategy_to_network_and_rf_according_to_cluster(keyspace="system_auth", db_cluster=db_cluster)

    def set_ks_strategy_to_network_and_rf_according_to_cluster(self, keyspace, db_cluster=None, repair_after_alter=True):
        db_cluster = db_cluster or self.db_cluster
        node = random.choice([node for node in self.db_cluster.nodes if not node.running_nemesis])
        nodes_by_region = self.db_cluster.nodes_by_region()
        dc_by_region = self.db_cluster.get_datacenter_name_per_region(db_cluster.nodes)
        datacenters = {}
        for region in nodes_by_region:
            datacenters.update({dc_by_region[region]: len(nodes_by_region[region])})
        self.log.debug("Number of nodes by datacenter %s", datacenters)
        NetworkTopologyReplicationStrategy(**datacenters).apply(node, keyspace)
        res = node.run_cqlsh(f'DESC KEYSPACE {cql_quote_if_needed(keyspace)}', num_retry_on_failure=3)
        self.log.debug("%s description: %s", keyspace, res.stdout)
        if repair_after_alter:
            self.log.info('repair %s keyspace ...', keyspace)
            for node in self.db_cluster.nodes:
                node.run_nodetool(sub_cmd=f"repair -pr {keyspace}", timeout=MINUTE_IN_SEC * 20)
            self.log.info('repair %s keyspace done', keyspace)

    @optional_stage('prepare_write')
    @cache
    def pre_create_alternator_tables(self):
        node = self.db_cluster.nodes[0]
        if self.params.get('alternator_port'):
            self.log.info("Going to create alternator tables")
            if self.params.get('alternator_enforce_authorization'):
                with self.db_cluster.cql_connection_patient(self.db_cluster.nodes[0]) as session:
                    session.execute("CREATE ROLE %s WITH PASSWORD = %s AND login = true AND superuser = true",
                                    (self.params.get('alternator_access_key_id'),
                                     self.params.get('alternator_secret_access_key')))

            with self.db_cluster.cql_connection_patient(self.db_cluster.nodes[0]) as session:
                # is tablets feature enabled in Scylla configuration.
                tablets_enabled = is_tablets_feature_enabled(session)
            prepare_cmd = self.params.get('prepare_write_cmd')
            stress_cmd = self.params.get('stress_cmd')
            is_ttl_in_workload = any('dynamodb.ttlKey' in str(cmd) for cmd in [prepare_cmd, stress_cmd])

            # Enable tablets with Alternator TTL requires (#16567).
            tablets_support_ttl = not SkipPerIssues(issues="scylladb/scylladb#16567", params=self.params)

            # Enable tablets for Alternator with LWT requires issue: "[Epic] tablets: support LWT #18068"
            tablets_support_lwt = not SkipPerIssues(issues="scylladb/scylladb#18068", params=self.params)

            tablets_supported_alternator_workload = not is_ttl_in_workload or tablets_support_ttl
            schema = self.params.get("dynamodb_primarykey_type")
            self.alternator.create_table(node=node, schema=schema,
                                         tablets_enabled=tablets_enabled and tablets_support_lwt and tablets_supported_alternator_workload)
            self.alternator.create_table(node=node, schema=schema, isolation=alternator.enums.WriteIsolation.FORBID_RMW,
                                         table_name=NO_LWT_TABLE_NAME,
                                         tablets_enabled=tablets_enabled and tablets_supported_alternator_workload)

            if is_ttl_in_workload:
                self.alternator.update_table_ttl(node=node, table_name=alternator.consts.TABLE_NAME)
                self.alternator.update_table_ttl(node=node, table_name=alternator.consts.NO_LWT_TABLE_NAME)

    def get_nemesis_class(self):
        """
        Get a Nemesis class from parameters.

        :return: Nemesis class.
        :rtype: nemesis.Nemesis derived class
        """
        nemesis_threads = []
        list_class_name = self.params.get('nemesis_class_name')
        nemesis_selectors = self.params.get("nemesis_selector")

        if nemesis_selectors and isinstance(nemesis_selectors, str):
            nemesis_selectors = [nemesis_selectors]
        if nemesis_selectors and isinstance(nemesis_selectors[0], str):
            nemesis_selectors = [nemesis_selectors[:]]

        nemesis_class_names = []
        for i, klass in enumerate(list_class_name.split(' ')):
            try:
                nemesis_name, num = klass.strip().split(':')
                nemesis_name = nemesis_name.strip()
                num = int(num.strip())

            except ValueError:
                nemesis_name = klass.split(':')[0]
                num = 1
            for _ in range(num):
                nemesis_class_names.append(nemesis_name)

        for i, nemesis_name in enumerate(nemesis_class_names):
            nemesis_selector = []
            if nemesis_selectors:
                try:
                    nemesis_selector = nemesis_selectors[i % len(nemesis_class_names)]
                except IndexError as details:
                    self.log.warning("Missing nemesis selector. use default. %s", details)

            nemesis_threads.append({'nemesis': getattr(nemesis, nemesis_name),
                                    'nemesis_selector': nemesis_selector})

        self.log.debug("Nemesis threads %s", nemesis_threads)
        return nemesis_threads

    def get_cluster_gce(self, loader_info, db_info, monitor_info):  # noqa: PLR0912, PLR0914
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        if loader_info['n_nodes'] is None:
            n_loader_nodes = self.params.get('n_loaders')
            if isinstance(n_loader_nodes, int):
                loader_info['n_nodes'] = [n_loader_nodes]
            elif isinstance(n_loader_nodes, str):
                loader_info['n_nodes'] = [int(n) for n in n_loader_nodes.split()]
            else:
                self.fail('Unsupported parameter type: {}'.format(type(n_loader_nodes)))
        if loader_info['type'] is None:
            loader_info['type'] = self.params.get('gce_instance_type_loader')
        if loader_info['disk_type'] is None:
            loader_info['disk_type'] = self.params.get('gce_root_disk_type_loader')
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
        if db_info['disk_type'] is None:
            db_info['disk_type'] = self.params.get('gce_root_disk_type_db')
        if db_info['disk_size'] is None:
            db_info['disk_size'] = self.params.get('root_disk_size_db')
        if db_info['n_local_ssd'] is None:
            db_info['n_local_ssd'] = self.params.get('gce_n_local_ssd_disk_db')
        if monitor_info['n_nodes'] is None:
            monitor_info['n_nodes'] = self.params.get('n_monitor_nodes')
        if monitor_info['type'] is None:
            monitor_info['type'] = self.params.get('gce_instance_type_monitor')
        if monitor_info['disk_type'] is None:
            monitor_info['disk_type'] = self.params.get('gce_root_disk_type_monitor')
        if monitor_info['disk_size'] is None:
            monitor_info['disk_size'] = self.params.get('root_disk_size_monitor')
        if monitor_info['n_local_ssd'] is None:
            monitor_info['n_local_ssd'] = self.params.get('gce_n_local_ssd_disk_monitor')

        user_prefix = self.params.get('user_prefix')

        gce_service = get_gce_compute_instances_client()
        gce_datacenter = self.params.get('gce_datacenter').split()
        TEST_LOG.info("Using GCE regions/datacenters: %s", gce_datacenter)

        user_credentials = self.params.get('user_credentials_path')
        self.credentials.append(UserRemoteCredentials(key_file=user_credentials))

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
            nodes_smp = self.params.get('nodes_smp') or []
            self.db_cluster = ScyllaGCECluster(gce_image=self.params.get('gce_image_db').strip(),
                                               gce_image_type=db_info['disk_type'],
                                               gce_image_size=db_info['disk_size'],
                                               gce_n_local_ssd=db_info['n_local_ssd'],
                                               gce_instance_type=db_info['type'],
                                               gce_service=gce_service,
                                               n_nodes=db_info['n_nodes'],
                                               add_disks=cluster_additional_disks,
                                               service_accounts=service_accounts,
                                               nodes_smp=nodes_smp,
                                               **common_params)

        loader_additional_disks = {'pd-ssd': self.params.get('gce_pd_ssd_disk_size_loader')}
        self.loaders = LoaderSetGCE(gce_image=self.params.get('gce_image_loader'),
                                    gce_image_type=loader_info['disk_type'],
                                    gce_image_size=loader_info['disk_size'],
                                    gce_n_local_ssd=loader_info['n_local_ssd'],
                                    gce_instance_type=loader_info['type'],
                                    gce_service=gce_service,
                                    n_nodes=loader_info['n_nodes'],
                                    add_disks=loader_additional_disks,
                                    **common_params)

        if monitor_info['n_nodes'] > 0:
            monitor_additional_disks = {'pd-ssd': self.params.get('gce_pd_ssd_disk_size_monitor')}
            self.monitors = MonitorSetGCE(gce_image=self.params.get('gce_image_monitor').strip(),
                                          gce_image_type=monitor_info['disk_type'],
                                          gce_image_size=monitor_info['disk_size'],
                                          gce_n_local_ssd=monitor_info['n_local_ssd'],
                                          gce_instance_type=monitor_info['type'],
                                          gce_service=gce_service,
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
                                region=region, availability_zone=self.params.get('availability_zone')))
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
        azure_image = self.params.get("azure_image_db").strip()
        user_prefix = self.params.get('user_prefix')
        self.credentials.append(UserRemoteCredentials(key_file=self.params.get('user_credentials_path')))

        common_params = dict(
            credentials=self.credentials,
            user_prefix=user_prefix,
            params=self.params,
            region_names=regions,
        )
        nodes_smp = self.params.get('nodes_smp') or []
        self.db_cluster = ScyllaAzureCluster(image_id=azure_image,
                                             root_disk_size=db_info['disk_size'],
                                             instance_type=db_info['type'],
                                             provisioners=provisioners,
                                             n_nodes=db_info['n_nodes'],
                                             user_name=self.params.get('azure_image_username'),
                                             nodes_smp=nodes_smp,
                                             **common_params)
        self.loaders = LoaderSetAzure(
            image_id=self.params.get('azure_image_loader'),
            root_disk_size=loader_info['disk_size'],
            instance_type="Standard_D2_v4",
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
                                                               region_name=regions[0]),
                    "Ebs": {
                        "VolumeSize": loader_info['disk_size'],
                        "VolumeType": "gp3"
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

        def _get_all_zones_common_params() -> list[dict]:
            all_zones_common_params = []
            aws_region = AwsRegion(region_name=regions[0])
            availability_zones = [zone[-1] for zone in
                                  aws_region.get_availability_zones_for_instance_type(self.params.get('instance_type_db'))]
            for zone in availability_zones:
                all_zones_common_params.append(
                    get_common_params(params=self.params, regions=regions, credentials=self.credentials,
                                      services=services, availability_zone=zone))
            return all_zones_common_params

        common_params = get_common_params(params=self.params, regions=regions, credentials=self.credentials,
                                          services=services)

        def _create_auto_zone_scylla_aws_cluster():
            capacity_errors = []
            for cl_zone_params in _get_all_zones_common_params():
                cl_zone_params.update(_get_instance_params())
                try:
                    nodes_smp = self.params.get('nodes_smp') or []
                    return ScyllaAWSCluster(
                        ec2_ami_id=self.params.get('ami_id_db_scylla').split(),
                        ec2_ami_username=self.params.get('ami_db_scylla_user'),
                        **cl_zone_params, nodes_smp=nodes_smp)
                except botocore.exceptions.ClientError as error:
                    capacity_error_keywards = ['Unsupported', 'InsufficientInstanceCapacity']
                    if any(capacity_error in str(error) for capacity_error in capacity_error_keywards):
                        self.log.warning("Failed creating a Scylla AWS cluster: %s", error)
                        capacity_errors.append(error)
                    else:
                        raise
            raise CriticalTestFailure(f"Failed creating a Scylla AWS cluster: {capacity_errors}")

        def _get_instance_params() -> dict:
            return dict(
                ec2_instance_type=db_info['type'],
                ec2_block_device_mappings=db_info['device_mappings'],
                n_nodes=db_info['n_nodes']
            )

        def create_cluster(db_type='scylla'):
            cl_params = _get_instance_params()
            cl_params.update(common_params)
            nodes_smp = self.params.get('nodes_smp') or []
            if db_type == 'scylla':
                if self.params.get('aws_fallback_to_next_availability_zone'):
                    return _create_auto_zone_scylla_aws_cluster()
                return ScyllaAWSCluster(
                    ec2_ami_id=self.params.get('ami_id_db_scylla').split(),
                    ec2_ami_username=self.params.get('ami_db_scylla_user'),
                    **cl_params,
                    nodes_smp=nodes_smp)
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
                    nodes_smp=nodes_smp,
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

        nodes_smp = self.params.get('nodes_smp') or []
        self.db_cluster = cluster_docker.ScyllaDockerCluster(n_nodes=[self.params.get("n_db_nodes"), ], nodes_smp=nodes_smp,
                                                             **container_node_params, **common_params)
        self.loaders = cluster_docker.LoaderSetDocker(n_nodes=self.params.get("n_loaders"),
                                                      **container_node_params, **common_params)
        self.monitors = cluster_docker.MonitorSetDocker(n_nodes=self.params.get("n_monitor_nodes"),
                                                        targets=dict(db_cluster=self.db_cluster,
                                                                     loaders=self.loaders),
                                                        **common_params)

    def get_cluster_baremetal(self):
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        baremetal_info: cluster_baremetal.BareMetalCredentials = KeyStore(
        ).get_baremetal_config(self.params.get("s3_baremetal_config"))
        user_credentials = self.params.get('user_credentials_path')
        self.credentials.append(UserRemoteCredentials(key_file=user_credentials))
        params = dict(
            n_nodes=len(baremetal_info['db_nodes']['node_list']),
            public_ips=[n['public_ip'] for n in baremetal_info['db_nodes']['node_list']],
            private_ips=[n['private_ip'] for n in baremetal_info['db_nodes']['node_list']],
            user_prefix=self.params.get('user_prefix'),
            credentials=self.credentials,
            ssh_username=baremetal_info['db_nodes']['username'],
            params=self.params,
            nodes_smp=self.params.get('nodes_smp') or []
        )
        self.db_cluster = cluster_baremetal.ScyllaPhysicalCluster(**params)

        params['n_nodes'] = len(baremetal_info['loader_nodes']['node_list'])
        params['public_ips'] = [n['public_ip'] for n in baremetal_info['loader_nodes']['node_list']]
        params['private_ips'] = [n['private_ip'] for n in baremetal_info['loader_nodes']['node_list']]
        params['ssh_username'] = baremetal_info['loader_nodes']['username']
        self.loaders = cluster_baremetal.LoaderSetPhysical(**params)

        params['n_nodes'] = len(baremetal_info['monitor_nodes']['node_list'])
        params['public_ips'] = [n['public_ip'] for n in baremetal_info['monitor_nodes']['node_list']]
        params['private_ips'] = [n['private_ip'] for n in baremetal_info['monitor_nodes']['node_list']]
        params['ssh_username'] = baremetal_info['monitor_nodes']['username']
        self.monitors = cluster_baremetal.MonitorSetPhysical(**params,
                                                             targets=dict(db_cluster=self.db_cluster, loaders=self.loaders),)

    def get_cluster_k8s_local_kind_cluster(self):
        k8s_cluster = mini_k8s.LocalKindCluster(
            software_version=self.params.get("mini_k8s_version"),
            user_prefix=self.params.get("user_prefix"),
            params=self.params,
        )
        self.k8s_clusters.append(k8s_cluster)
        k8s_cluster.deploy()

        auxiliary_pool = mini_k8s.MinimalK8SNodePool(
            k8s_cluster=k8s_cluster,
            name=k8s_cluster.AUXILIARY_POOL_NAME,
            num_nodes=self.params.get("k8s_n_auxiliary_nodes") or 2,
            image_type="fake-image-type",
            instance_type="fake-instance-type")
        k8s_cluster.deploy_node_pool(auxiliary_pool, wait_till_ready=False)
        for namespace in ('kube-system', 'local-path-storage'):
            k8s_cluster.set_nodeselector_for_deployments(
                pool_name=k8s_cluster.AUXILIARY_POOL_NAME, namespace=namespace)

        loader_pool = None
        if self.params.get("n_loaders"):
            loader_pool = mini_k8s.MinimalK8SNodePool(
                k8s_cluster=k8s_cluster,
                name=k8s_cluster.LOADER_POOL_NAME,
                num_nodes=self.params.get("n_loaders"),
                image_type="fake-image-type",
                instance_type="fake-instance-type")
            k8s_cluster.deploy_node_pool(loader_pool, wait_till_ready=False)

        scylla_pool = mini_k8s.MinimalK8SNodePool(
            k8s_cluster=k8s_cluster,
            name=k8s_cluster.SCYLLA_POOL_NAME,
            num_nodes=self.params.get("n_db_nodes"),
            image_type="fake-image-type",
            instance_type="fake-instance-type")
        k8s_cluster.deploy_node_pool(scylla_pool, wait_till_ready=False)
        if self.params.get("k8s_local_volume_provisioner_type") == 'static':
            k8s_cluster.install_static_local_volume_provisioner(node_pools=scylla_pool)
        else:
            k8s_cluster.install_dynamic_local_volume_provisioner(node_pools=scylla_pool)

        k8s_cluster.deploy_cert_manager(pool_name=k8s_cluster.AUXILIARY_POOL_NAME)
        if self.params.get("k8s_enable_sni"):
            k8s_cluster.deploy_ingress_controller(pool_name=k8s_cluster.AUXILIARY_POOL_NAME)
        k8s_cluster.deploy_scylla_operator(pool_name=k8s_cluster.AUXILIARY_POOL_NAME)
        if self.params.get('use_mgmt'):
            k8s_cluster.deploy_scylla_manager(pool_name=k8s_cluster.AUXILIARY_POOL_NAME)

        n_monitor_nodes = self.params.get("k8s_n_monitor_nodes") or self.params.get("n_monitor_nodes")
        if n_monitor_nodes:
            monitor_pool = mini_k8s.MinimalK8SNodePool(
                k8s_cluster=k8s_cluster,
                name=k8s_cluster.MONITORING_POOL_NAME,
                num_nodes=n_monitor_nodes,
                image_type="fake-image-type",
                instance_type="fake-instance-type")
            k8s_cluster.deploy_node_pool(monitor_pool, wait_till_ready=True)
            k8s_cluster.deploy_prometheus_operator()

        nodes_smp = self.params.get('nodes_smp') or []
        self.db_cluster = mini_k8s.LocalMinimalScyllaPodCluster(
            k8s_clusters=[k8s_cluster],
            scylla_cluster_name=self.params.get("k8s_scylla_cluster_name"),
            user_prefix=self.params.get("user_prefix"),
            n_nodes=self.params.get("k8s_n_scylla_pods_per_cluster") or self.params.get("n_db_nodes"),
            params=self.params,
            node_pool_name=scylla_pool.name,
            add_nodes=False,
            nodes_smp=nodes_smp,
        )

        if self.params.get("n_loaders"):
            self.loaders = cluster_k8s.LoaderPodCluster(
                k8s_clusters=[k8s_cluster],
                loader_cluster_name=self.params.get("k8s_loader_cluster_name"),
                user_prefix=self.params.get("user_prefix"),
                n_nodes=self.params.get("k8s_n_loader_pods_per_cluster") or self.params.get("n_loaders"),
                params=self.params,
                node_pool_name=loader_pool.name,
                add_nodes=False,
            )

        self._add_and_wait_for_cluster_nodes_in_parallel([self.db_cluster, self.loaders])

        # Deploy optional K8S-based monitoring
        if self.params.get('k8s_deploy_monitoring'):
            for db_cluster in self.db_clusters_multitenant or [self.db_cluster]:
                cluster_name, namespace = db_cluster.scylla_cluster_name, db_cluster.namespace
                k8s_cluster.deploy_scylla_cluster_monitoring(cluster_name=cluster_name, namespace=namespace)
                k8s_cluster.register_sct_grafana_dashboard(cluster_name=cluster_name, namespace=namespace)

        # Deploy main VM-based monitoring
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

    def get_cluster_kafka(self):
        if kafka_backend := self.params.get('kafka_backend'):
            if kafka_backend == 'localstack':
                self.kafka_cluster = LocalKafkaCluster(params=self.params)
                self.kafka_cluster.start()
            else:
                raise NotImplementedError(f"{kafka_backend=} not implemented")

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

    def get_cluster_k8s_gke(self, n_k8s_clusters: int):
        gce_datacenters = self.params.gce_datacenters
        availability_zones = self.params.get('availability_zone').split(',')
        for _ in range(n_k8s_clusters):
            self.credentials.append(UserRemoteCredentials(key_file=self.params.get('user_credentials_path')))
        for i in range(n_k8s_clusters):
            self.k8s_clusters.append(gke.init_k8s_gke_cluster(
                gce_datacenter=gce_datacenters[i % len(gce_datacenters)],
                availability_zone=availability_zones[i % len(availability_zones)],
                cluster_uuid=(
                    f"{self.test_config.test_id()[:8]}"
                    if n_k8s_clusters < 2 else f"{self.test_config.test_id()[:6]}-{i + 1}"),
                params=self.params,
            ))
        ParallelObject(timeout=7200, num_workers=n_k8s_clusters, objects=self.k8s_clusters).run(
            func=gke.deploy_k8s_gke_cluster, unpack_objects=True, ignore_exceptions=False)

        for i in range(self.k8s_clusters[0].tenants_number):
            nodes_smp = self.params.get('nodes_smp') or []
            self.db_clusters_multitenant.append(gke.GkeScyllaPodCluster(
                k8s_clusters=self.k8s_clusters,
                scylla_cluster_name=self.params.get("k8s_scylla_cluster_name") + (f"-{i + 1}" if i else ""),
                user_prefix=(f"{i + 1}-" if i else "") + self.params.get("user_prefix"),
                n_nodes=self.params.get("k8s_n_scylla_pods_per_cluster") or self.params.get("n_db_nodes"),
                params=deepcopy(self.params),
                node_pool_name=self.k8s_clusters[0].SCYLLA_POOL_NAME,
                add_nodes=False,
                nodes_smp=nodes_smp,
            ))
        self.db_cluster = self.db_clusters_multitenant[0]

        if self.params.get("n_loaders"):
            for i in range(self.k8s_clusters[0].tenants_number):
                self.loaders_multitenant.append(cluster_k8s.LoaderPodCluster(
                    k8s_clusters=self.k8s_clusters,
                    loader_cluster_name=self.params.get("k8s_loader_cluster_name") + (f"-{i + 1}" if i else ""),
                    user_prefix=(f"{i + 1}-" if i else "") + self.params.get("user_prefix"),
                    n_nodes=self.params.get("k8s_n_loader_pods_per_cluster") or self.params.get("n_loaders"),
                    params=self.params,
                    node_pool_name=self.k8s_clusters[0].LOADER_POOL_NAME,
                    add_nodes=False,
                ))
            self.loaders = self.loaders_multitenant[0]

        # NOTE: wait for Scylla and loader clusters nodes creation
        self._add_and_wait_for_cluster_nodes_in_parallel(
            self.db_clusters_multitenant + self.loaders_multitenant)

        # Deploy optional K8S-based monitoring
        if self.params.get('k8s_deploy_monitoring'):
            for k8s_cluster in self.k8s_clusters:
                k8s_cluster.deploy_prometheus_operator()
                for db_cluster in self.db_clusters_multitenant:
                    cluster_name, namespace = db_cluster.scylla_cluster_name, db_cluster.namespace
                    k8s_cluster.deploy_scylla_cluster_monitoring(cluster_name=cluster_name, namespace=namespace)
                    k8s_cluster.register_sct_grafana_dashboard(cluster_name=cluster_name, namespace=namespace)

        # Deploy main VM-based monitoring
        if self.params.get("n_monitor_nodes") > 0:
            for i in range(self.k8s_clusters[0].tenants_number):
                self.log.debug("Create monitor for the DB cluster №%s", i + 1)
                self.monitors_multitenant.append(gke.MonitorSetGKE(
                    gce_image=self.params.get("gce_image_monitor"),
                    gce_image_username=self.params.get("gce_image_username"),
                    gce_image_type=self.params.get("gce_root_disk_type_monitor"),
                    gce_image_size=self.params.get('root_disk_size_monitor'),
                    gce_network=self.params.get("gce_network"),
                    gce_instance_type=self.params.get("gce_instance_type_monitor"),
                    gce_n_local_ssd=self.params.get("gce_n_local_ssd_disk_monitor"),
                    gce_datacenter=gce_datacenters,
                    gce_service=get_gce_compute_instances_client(),
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
                # NOTE: add callback for the monitroing reconfiguration when
                #       Scylla pods of the appropriate Scylla cluster get new IP addresses
                for k8s_cluster in self.k8s_clusters:
                    k8s_cluster.scylla_pods_ip_change_tracker_thread.register_callbacks(
                        callbacks=self.monitors_multitenant[i].reconfigure_scylla_monitoring,
                        namespace=self.db_clusters_multitenant[i].namespace)
            self.monitors = self.monitors_multitenant[0]
            self._add_and_wait_for_cluster_nodes_in_parallel(self.monitors_multitenant)
        else:
            self.monitors = NoMonitorSet()
            self.monitors_multitenant = [self.monitors]

    def get_cluster_k8s_eks(self, n_k8s_clusters: int):  # pylint: disable=too-many-branches
        region_names = self.params.region_names
        availability_zones = self.params.get('availability_zone').split(',')
        for _ in range(n_k8s_clusters):
            self.credentials.append(UserRemoteCredentials(key_file=self.params.get('user_credentials_path')))
        for i in range(n_k8s_clusters):
            self.k8s_clusters.append(eks.init_k8s_eks_cluster(
                region_name=region_names[i % len(region_names)],
                availability_zone=availability_zones[i % len(availability_zones)],
                cluster_uuid=(
                    f"{self.test_config.test_id()[:8]}"
                    if n_k8s_clusters < 2 else f"{self.test_config.test_id()[:6]}-{i + 1}"),
                params=self.params,
                credentials=self.credentials,
            ))
        ParallelObject(timeout=7200, num_workers=n_k8s_clusters, objects=self.k8s_clusters).run(
            func=eks.deploy_k8s_eks_cluster, unpack_objects=True, ignore_exceptions=False)

        for i in range(self.k8s_clusters[0].tenants_number):
            nodes_smp = self.params.get('nodes_smp') or []
            self.db_clusters_multitenant.append(eks.EksScyllaPodCluster(
                k8s_clusters=self.k8s_clusters,
                scylla_cluster_name=self.params.get("k8s_scylla_cluster_name") + (f"-{i + 1}" if i else ""),
                user_prefix=(f"{i + 1}-" if i else "") + self.params.get("user_prefix"),
                n_nodes=self.params.get("k8s_n_scylla_pods_per_cluster") or self.params.get("n_db_nodes"),
                params=deepcopy(self.params),
                node_pool_name=self.k8s_clusters[0].SCYLLA_POOL_NAME,
                add_nodes=False,
                nodes_smp=nodes_smp,
            ))
            if self.params.get('use_mgmt'):
                for k8s_cluster in self.k8s_clusters:
                    k8s_cluster.create_iamserviceaccount_for_s3_access(
                        db_cluster_name=self.db_clusters_multitenant[i].scylla_cluster_name,
                        namespace=self.db_clusters_multitenant[i].namespace)
        self.db_cluster = self.db_clusters_multitenant[0]

        if self.params.get("n_loaders"):
            for i in range(self.k8s_clusters[0].tenants_number):
                self.loaders_multitenant.append(cluster_k8s.LoaderPodCluster(
                    k8s_clusters=self.k8s_clusters,
                    loader_cluster_name=self.params.get("k8s_loader_cluster_name") + (f"-{i + 1}" if i else ""),
                    user_prefix=(f"{i + 1}-" if i else "") + self.params.get("user_prefix"),
                    n_nodes=self.params.get("k8s_n_loader_pods_per_cluster") or self.params.get("n_loaders"),
                    params=self.params,
                    node_pool_name=self.k8s_clusters[0].LOADER_POOL_NAME,
                    add_nodes=False,
                ))
            self.loaders = self.loaders_multitenant[0]

        # NOTE: wait for Scylla and loader clusters nodes creation
        self._add_and_wait_for_cluster_nodes_in_parallel(
            self.db_clusters_multitenant + self.loaders_multitenant)

        # Deploy optional K8S-based monitoring
        if self.params.get('k8s_deploy_monitoring'):
            for k8s_cluster in self.k8s_clusters:
                k8s_cluster.deploy_prometheus_operator()
                for db_cluster in self.db_clusters_multitenant:
                    cluster_name, namespace = db_cluster.scylla_cluster_name, db_cluster.namespace
                    k8s_cluster.deploy_scylla_cluster_monitoring(cluster_name=cluster_name, namespace=namespace)
                    k8s_cluster.register_sct_grafana_dashboard(cluster_name=cluster_name, namespace=namespace)

        # Deploy main VM-based monitoring
        monitor_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'disk_type': None,
                        'n_local_ssd': None, 'device_mappings': None}
        init_monitoring_info_from_params(monitor_info, params=self.params, regions=region_names)
        if monitor_info['n_nodes']:
            common_params = get_common_params(
                params=self.params, regions=region_names, credentials=self.credentials,
                services=get_ec2_services(region_names))
            base_user_prefix = common_params["user_prefix"]
            for i in range(self.k8s_clusters[0].tenants_number):
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
                # NOTE: add callback for the monitroing reconfiguration when
                #       Scylla pods of the appropriate Scylla cluster get new IP addresses
                for k8s_cluster in self.k8s_clusters:
                    k8s_cluster.scylla_pods_ip_change_tracker_thread.register_callbacks(
                        callbacks=self.monitors_multitenant[i].reconfigure_scylla_monitoring,
                        namespace=self.db_clusters_multitenant[i].namespace)
            self.monitors = self.monitors_multitenant[0]
            self._add_and_wait_for_cluster_nodes_in_parallel(self.monitors_multitenant)
        else:
            self.monitors = NoMonitorSet()
            self.monitors_multitenant = [self.monitors]

    def init_resources(self, loader_info=None, db_info=None,
                       monitor_info=None):
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        if loader_info is None:
            loader_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'disk_type': None, 'n_local_ssd': None,
                           'device_mappings': None}
        if db_info is None:
            db_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'disk_type': None, 'n_local_ssd': None,
                       'device_mappings': None}

        if monitor_info is None:
            monitor_info = {'n_nodes': None, 'type': None, 'disk_size': None, 'disk_type': None, 'n_local_ssd': None,
                            'device_mappings': None}

        cluster_backend = self.params.get('cluster_backend')
        if cluster_backend is None:
            cluster_backend = 'aws'

        self.get_cluster_kafka()

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
            self.get_cluster_k8s_gke(n_k8s_clusters=len(self.params.gce_datacenters))
        elif cluster_backend == 'k8s-eks':
            self.get_cluster_k8s_eks(n_k8s_clusters=len(self.params.region_names))
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
    def run_stress_thread(self, stress_cmd, duration=None, stress_num=1, keyspace_num=1, profile=None, prefix='',  # pylint: disable=too-many-arguments  # noqa: PLR0911, PLR0913
                          round_robin=False, stats_aggregate_cmds=True, keyspace_name=None, compaction_strategy='',
                          use_single_loader=False,
                          stop_test_on_failure=True):

        params = dict(stress_cmd=stress_cmd, duration=duration, stress_num=stress_num, keyspace_num=keyspace_num,
                      keyspace_name=keyspace_name, profile=profile, prefix=prefix, round_robin=round_robin,
                      stats_aggregate_cmds=stats_aggregate_cmds, use_single_loader=use_single_loader)

        if 'cql-stress-cassandra-stress' in stress_cmd:
            params['stop_test_on_failure'] = stop_test_on_failure
            params['compaction_strategy'] = compaction_strategy
            return self.run_cql_stress_cassandra_thread(**params)
        elif 'cassandra-stress' in stress_cmd:  # cs cmdline might started with JVM_OPTION
            params['stop_test_on_failure'] = stop_test_on_failure
            params['compaction_strategy'] = compaction_strategy
            return self.run_stress_cassandra_thread(**params)
        elif stress_cmd.startswith('scylla-bench'):
            params['stop_test_on_failure'] = stop_test_on_failure
            return self.run_stress_thread_bench(**params)
        elif 'cassandra-harry' in stress_cmd:
            params['stop_test_on_failure'] = stop_test_on_failure
            return self.run_stress_thread_harry(**params)
        elif stress_cmd.startswith('bin/ycsb'):
            return self.run_ycsb_thread(**params)
        elif stress_cmd.startswith('latte'):
            params['stop_test_on_failure'] = stop_test_on_failure
            return self.run_latte_thread(**params)
        elif stress_cmd.startswith('ndbench'):
            return self.run_ndbench_thread(**params)
        elif stress_cmd.startswith('hydra-kcl'):
            return self.run_hydra_kcl_thread(**params)
        elif stress_cmd.startswith('nosqlbench'):
            params['stop_test_on_failure'] = stop_test_on_failure
            return self.run_nosqlbench_thread(**params)
        elif stress_cmd.startswith('table_compare'):
            return self.run_table_compare_thread(**params)
        elif stress_cmd.startswith('python_thread'):
            return self.run_python_thread(**params)
        else:
            raise ValueError(f'Unsupported stress command: "{stress_cmd[:50]}..."')

    # pylint: disable=too-many-arguments
    def run_stress_cassandra_thread(  # noqa: PLR0913
            self, stress_cmd, duration=None, stress_num=1, keyspace_num=1, profile=None, prefix='', round_robin=False,
            stats_aggregate_cmds=True, keyspace_name=None, compaction_strategy='', stop_test_on_failure=True, params=None, **_):
        # pylint: disable=too-many-locals
        # stress_cmd = self._cs_add_node_flag(stress_cmd)
        if duration:
            timeout = self.get_duration(duration)
            if ' duration' in stress_cmd:
                stress_cmd = re.sub(r'\sduration=\d+[mhd]\s', f' duration={duration}m ', stress_cmd)
        elif self._stress_duration and ' duration=' in stress_cmd:
            timeout = self.get_duration(self._stress_duration)
            stress_cmd = re.sub(r'\sduration=\d+[mhd]\s', f' duration={self._stress_duration}m ', stress_cmd)
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
                                          compaction_strategy=compaction_strategy,
                                          profile=profile,
                                          node_list=self.db_cluster.nodes,
                                          round_robin=round_robin,
                                          client_encrypt=self.params.get('client_encrypt'),
                                          keyspace_name=keyspace_name,
                                          stop_test_on_failure=stop_test_on_failure,
                                          params=params or self.params).run()
        self.alter_test_tables_encryption(stress_command=stress_cmd)
        return cs_thread

    # pylint: disable=too-many-arguments
    def run_cql_stress_cassandra_thread(  # noqa: PLR0913
            self, stress_cmd, duration=None, stress_num=1, keyspace_num=1, profile=None, prefix='', round_robin=False,
            stats_aggregate_cmds=True, keyspace_name=None, compaction_strategy='', stop_test_on_failure=True, params=None, **_):
        # pylint: disable=too-many-locals
        if duration:
            timeout = self.get_duration(duration)
            if ' duration' in stress_cmd:
                stress_cmd = re.sub(r'\sduration=\d+[mhd]\s', f' duration={duration}m ', stress_cmd)
        elif self._stress_duration and ' duration=' in stress_cmd:
            timeout = self.get_duration(self._stress_duration)
            stress_cmd = re.sub(r'\sduration=\d+[mhd]\s', f' duration={self._stress_duration}m ', stress_cmd)
        else:
            timeout = get_timeout_from_stress_cmd(stress_cmd) or self.get_duration(duration)

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, prefix, stresser="cassandra-stress",
                                           aggregate=stats_aggregate_cmds)
        stop_test_on_failure = False if not self.params.get("stop_test_on_stress_failure") else stop_test_on_failure
        cs_thread = CqlStressCassandraStressThread(loader_set=self.loaders,
                                                   stress_cmd=stress_cmd,
                                                   timeout=timeout,
                                                   stress_num=stress_num,
                                                   keyspace_num=keyspace_num,
                                                   compaction_strategy=compaction_strategy,
                                                   profile=profile,
                                                   node_list=self.db_cluster.nodes,
                                                   round_robin=round_robin,
                                                   client_encrypt=self.params.get('client_encrypt'),
                                                   keyspace_name=keyspace_name,
                                                   stop_test_on_failure=stop_test_on_failure,
                                                   params=params or self.params).run()
        self.alter_test_tables_encryption(stress_command=stress_cmd)
        return cs_thread

    # pylint: disable=too-many-arguments,unused-argument
    def run_stress_thread_bench(self, stress_cmd, duration=None, round_robin=False, stats_aggregate_cmds=True,
                                stop_test_on_failure=True, **_):

        if duration:
            timeout = self.get_duration(duration)
        elif self._stress_duration and '-duration=' in stress_cmd:
            timeout = self.get_duration(self._stress_duration)
            stress_cmd = re.sub(r'\s-duration[=\s]+\d+[mhd]+\s*', f' -duration={self._stress_duration}m ', stress_cmd)
        else:
            timeout = get_timeout_from_stress_cmd(stress_cmd) or self.get_duration(duration)
        stop_test_on_failure = False if not self.params.get("stop_test_on_stress_failure") else stop_test_on_failure

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, stresser="scylla-bench", aggregate=stats_aggregate_cmds)
        bench_thread = ScyllaBenchThread(
            stress_cmd, loader_set=self.loaders, timeout=timeout,
            node_list=self.db_cluster.nodes,
            round_robin=round_robin,
            stop_test_on_failure=stop_test_on_failure,
            params=self.params,
        )
        bench_thread.run()

        self.alter_test_tables_encryption(stress_command=stress_cmd)
        return bench_thread

    def run_stress_thread_harry(self, stress_cmd, duration=None,
                                # pylint: disable=too-many-arguments,unused-argument
                                round_robin=False, stats_aggregate_cmds=True,
                                stop_test_on_failure=True, **_):  # pylint: disable=too-many-arguments,unused-argument

        timeout = self.get_duration(duration)

        if not self.params.get("stop_test_on_stress_failure"):
            stop_test_on_failure = False

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, stresser="cassandra-harry", aggregate=stats_aggregate_cmds)
        harry_thread = CassandraHarryThread(
            loader_set=self.loaders,
            stress_cmd=stress_cmd,
            timeout=timeout,
            node_list=self.db_cluster.nodes,
            round_robin=round_robin,
            stop_test_on_failure=stop_test_on_failure,
            params=self.params,
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

    def run_latte_thread(self, stress_cmd, duration=None, stress_num=1, prefix='',
                         round_robin=False, stats_aggregate_cmds=True, stop_test_on_failure=True, **_):
        if duration:
            timeout = self.get_duration(duration)
        elif self._stress_duration and ' --duration' in stress_cmd:
            timeout = self.get_duration(self._stress_duration)
            stress_cmd = re.sub(r'\s--duration\s+\d+[mhd]\s', f' --duration {self._stress_duration}m ', stress_cmd)
        else:
            timeout = get_timeout_from_stress_cmd(stress_cmd) or self.get_duration(duration)

        if self.create_stats:
            self.update_stress_cmd_details(stress_cmd, prefix, stresser="latte", aggregate=stats_aggregate_cmds)

        stop_test_on_failure = False if not self.params.get("stop_test_on_stress_failure") else stop_test_on_failure
        return LatteStressThread(loader_set=self.loaders,
                                 stress_cmd=stress_cmd,
                                 timeout=timeout,
                                 stress_num=stress_num,
                                 node_list=self.db_cluster.nodes,
                                 round_robin=round_robin,
                                 stop_test_on_failure=stop_test_on_failure,
                                 params=self.params).run()

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
        elif self._stress_duration and ' --duration' in cmd:
            timeout = self.get_duration(self._stress_duration)
            cmd = re.sub(r'\s--duration\s+\d+[mhd]\s', f' --duration {self._stress_duration}m ', cmd)
            cmd = re.sub(r'\s--warmup\s+\d+[mhd]\s', f' --warmup {int(self._stress_duration * .2)}m ', cmd)
        else:
            timeout = get_timeout_from_stress_cmd(cmd) or self.get_duration(duration)
        if self.create_stats:
            self.update_stress_cmd_details(cmd, stresser="gemini", aggregate=False)
        return GeminiStressThread(test_cluster=self.db_cluster,
                                  oracle_cluster=self.cs_db_cluster,
                                  loaders=self.loaders,
                                  stress_cmd=cmd,
                                  timeout=timeout,
                                  params=self.params).run()

    # pylint: disable=too-many-arguments
    def run_python_thread(self, stress_cmd, duration=None, **_):
        timeout = self.get_duration(duration)

        options = parse_python_thread_command(stress_cmd)
        klass_thread = next(iter([t for t in PYTHON_THREAD_LIST if options.get('thread') == t.__name__]), None)
        assert klass_thread
        thread = klass_thread(tester=self,
                              timeout=timeout,
                              params=self.params, **options)
        thread.start()
        return thread

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
        return results and not errors

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

    @staticmethod
    def run_fullscan_thread(fullscan_params: ThreadParams, thread_name: str):
        """Run thread of cql command select *

        Calculate test duration and timeout interval between
        requests and execute the thread with cqlsh command to
        db node 'select * from ks.cf, where ks and cf are
        random choosen from current configuration'

        Keyword Arguments:
            fullscan_params: FullScanParams object with params relevant for starting
            the ScanOperationThread
        """
        ScanOperationThread(thread_params=fullscan_params, thread_name=thread_name).start()

    def run_commit_log_check_thread(self, duration):
        CommitLogCheckThread.run(self, duration)

    def run_tombstone_gc_verification_thread(self, duration=None, interval=600, propagation_delay_in_seconds=3600, **kwargs):
        """Run a thread of tombstones gc verification.

        1. Search for repair history.
        2. Go over node sstables, created after last repair.
        3. Run an sstable dump to get sstables data.
        4. Go over the output, searching for tombstones.
        5. If a tombstone creation time (+propagation-delay) is earlier than last repair then ==>
            raise a tombstone-gc-verification error.

        Keyword Arguments:
            interval {number} -- interval between requests in seconds (default: {600})
            duration {int} -- duration of running thread in min (default: {None})
        """
        TombstoneGcVerificationThread(db_cluster=self.db_cluster, duration=self.get_duration(duration),
                                      interval=interval, termination_event=self.db_cluster.nemesis_termination_event,
                                      propagation_delay_in_seconds=propagation_delay_in_seconds, **kwargs).start()

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

    def create_keyspace(self, keyspace_name, replication_factor, tablets_config: Optional[TabletsConfiguration] = None):
        """
        If replication_factor is int, the all DC's will have the same replication factor, if 0, use LocalReplicationStrategy
        If replication_factor is dict, the keys are the DC's names and the values are the replication factor for each DC
        e.g.
        {"dc_name1": 4, "dc_name2": 6, "<dc_name>": <int>...}
        """

        execution_node, validation_node = self.db_cluster.nodes[0], self.db_cluster.nodes[-1]
        with self.db_cluster.cql_connection_patient(execution_node) as session:
            if isinstance(replication_factor, int):
                if replication_factor == 0:
                    replication_strategy = LocalReplicationStrategy()
                else:
                    replication_strategy = NetworkTopologyReplicationStrategy(default_rf=replication_factor)
            else:
                replication_strategy = NetworkTopologyReplicationStrategy(**replication_factor)

            cmd = 'CREATE KEYSPACE IF NOT EXISTS %s WITH replication=%s' % (keyspace_name, str(replication_strategy))
            if tablets_config:
                cmd += ' AND TABLETS = %s' % tablets_config
            execution_result = session.execute(cmd)

        if execution_result:
            self.log.debug("keyspace creation result: {}".format(execution_result.response_future))
        with self.db_cluster.cql_connection_patient(validation_node) as session:
            does_keyspace_exist = self.wait_validate_keyspace_existence(session, keyspace_name)
        return does_keyspace_exist

    def create_table(self, name, key_type="varchar",  # pylint: disable=too-many-arguments,too-many-branches  # noqa: PLR0913
                     speculative_retry=None, read_repair=None, compression=None,
                     gc_grace=None, columns=None, compaction=None,
                     compact_storage=False, scylla_encryption_options=None, keyspace_name=None,
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

        if compaction is not None or sstable_size:
            compaction = compaction or self.params.get('compaction_strategy')
            prefix = ' AND compaction={'
            postfix = '}'
            compaction_clause = " 'class': '%s'" % compaction
            if sstable_size and compaction in [CompactionStrategy.LEVELED.value, CompactionStrategy.INCREMENTAL.value]:
                compaction_clause += ", 'sstable_size_in_mb' : '%s'" % sstable_size
            query += prefix + compaction_clause + postfix

        if read_repair is not None:
            query = '%s AND read_repair_chance=%f' % (query, read_repair)
        if gc_grace is not None:
            query = '%s AND gc_grace_seconds=%d' % (query, gc_grace)
        if speculative_retry is not None:
            query = ('%s AND speculative_retry=\'%s\'' %
                     (query, speculative_retry))
        if scylla_encryption_options:
            query = '%s AND scylla_encryption_options=%s' % (query, scylla_encryption_options)
        if compact_storage:
            query += ' AND COMPACT STORAGE'
        self.log.debug('CQL query to execute: {}'.format(query))
        with self.db_cluster.cql_connection_patient(node=self.db_cluster.nodes[0], keyspace=keyspace_name) as session:
            session.execute(query)
        time.sleep(0.2)

    def truncate_cf(self, ks_name: str, table_name: str, session: Session, truncate_timeout_sec: int | None = None):
        try:
            timeout = f" USING TIMEOUT {truncate_timeout_sec}s" if truncate_timeout_sec else ""
            session.execute('TRUNCATE TABLE {0}.{1}{2}'.format(ks_name, table_name, timeout))
        except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.debug('Failed to truncate base table {0}.{1}. Error: {2}'.format(ks_name, table_name, str(ex)))

    def create_materialized_view(self, ks_name, base_table_name, mv_name, mv_partition_key, mv_clustering_key, session,  # noqa: PLR0913
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
        session.execute(query, timeout=600)

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
            except Exception as error:  # pylint: disable=broad-except  # noqa: BLE001
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
            except Exception as error:  # pylint: disable=broad-except  # noqa: BLE001
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
            source_table_rows = fetch_all_rows(session=session, default_fetch_size=5000, statement=statement)
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
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
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
        SCTCapacityReservation.cancel(self.params)
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

    @silence(name="Save node schema", raise_error_event=False)
    def save_cqlsh_output_in_file(self, node, cmd: str, log_file: str):
        self.log.info("Save command '%s' output in the file. Node %s", cmd, node.name)

        log_file_path = Path(self.logdir) / self.db_cluster.logdir / log_file
        self.log.debug("Schema file path: %s", log_file_path)
        if not (result := node.run_cqlsh(cmd).stdout):
            return

        with open(log_file_path, "w", encoding="utf-8") as res_file:
            res_file.write(result.strip())

    def save_schema(self):
        if self.db_cluster is None:
            self.log.error("Didn't find the nodes in the cluster for saving the schema.")
            return

        self.log.info("Save nodes user schema in the files")
        # Collect schema info from one node only. Not need to collect from every node
        found_live_node = False
        for node in self.db_cluster.nodes:
            if not node._is_node_ready_run_scylla_commands():
                continue
            found_live_node = True
            self.save_cqlsh_output_in_file(node=node, cmd="desc schema", log_file="schema.log")
            self.save_cqlsh_output_in_file(node=node, cmd="select JSON * from system_schema.tables",
                                           log_file="system_schema_tables.log")
            break

        if not found_live_node:
            self.log.error("Didn't find any live node for saving the schema.")

    def tearDown(self):
        self.teardown_started = True
        with silence(parent=self, name='Sending test end event'):
            InfoEvent(message="TEST_END").publish()
        self.save_schema()
        self.log.info("Post test validators are starting...")
        for validator_class in teardown_validators_list:
            validator_class(self.params, self).validate()
        self.log.info('TearDown is starting...')
        self.stop_timeout_thread()
        self.stop_event_analyzer()
        self.stop_resources()
        self.get_test_failures()

        # NOTE: running on K8S we need to gather logs otherwise a lot of
        # debugging info is lost
        if self.k8s_clusters:
            for k8s_cluster in self.k8s_clusters:
                k8s_cluster.gather_k8s_logs_by_operator()
                k8s_cluster.gather_k8s_logs()

        if self.kafka_cluster:
            with silence(parent=self, name='stopping kafka'):
                self.kafka_cluster.stop()
        if self.monitors is not None:
            self.monitors.update_default_time_range(self.start_time, time.time())
        if self.params.get('collect_logs'):
            self.collect_logs()
        self.collect_ssl_conf()
        self.clean_resources()
        if self.create_stats:
            self.update_test_with_errors()
        time.sleep(1)  # Sleep is needed to let final event being saved into files
        self.save_email_data()
        self.argus_collect_gemini_results()
        self.destroy_localhost()
        self.stop_event_device()
        with silence(parent=self, name='Cleaning up SSL config directory'):
            cleanup_ssl_config()

        self.finalize_teardown()
        self.argus_finalize_test_run()
        try:
            ElasticRunReporter().report_run(run_id=self.test_config.test_id(), status=self.get_test_status())
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass
        self.argus_heartbeat_stop_signal.set()

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
        s3_link = []
        for log_collector in [BaseSCTLogCollector, PythonSCTLogCollector]:
            s3_link.extend(log_collector(
                [], self.test_config.test_id(), os.path.join(self.logdir, "collected_logs"), params=self.params
            ).collect_logs(self.logdir))
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
        stress_fixed_params = f" -schema 'replication(strategy=NetworkTopologyStrategy,replication_factor={replication_factor}) " \
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

    def alter_test_tables_encryption(self, upgradesstables=True, stress_command=''):
        """
        Configure encryption at-rest for all test tables if needed,
        sleep a while to wait the workload starts and test tables are created
        """
        append_scylla_yaml = self.params.get("append_scylla_yaml") or {}

        if ((scylla_encryption_options := self.params.get('scylla_encryption_options'))
            and 'write' in stress_command
                and 'user_info_encryption' not in append_scylla_yaml):

            time.sleep(60)  # waiting enough time for stress to start and have the schema created
            for table in self.db_cluster.get_non_system_ks_cf_list(self.db_cluster.nodes[0], filter_out_mv=True):
                self.alter_table_encryption(
                    table, scylla_encryption_options=scylla_encryption_options, upgradesstables=upgradesstables)
            self.db_cluster.wait_for_schema_agreement()

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
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            self.fail("Unable to get data set size from cassandra-stress command: %s" % cs_cmd)
            return None

    def get_c_s_column_definition(self, cs_cmd):  # pylint: disable=inconsistent-return-statements
        """:returns value of -col in stress comand, that is approximation and currently doesn't take in consideration
            column definitions if they present in the command
        """
        try:
            # Example:  -col 'size=FIXED(200) n=FIXED(5)'
            if search_res := re.search(r".* -col ('.*') .*", cs_cmd):
                return search_res.group(1)
            return None
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            self.fail("Unable to get column definition from cassandra-stress command: %s" % cs_cmd)
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
        start_time = self.start_time if not self.create_stats else self._stats["test_details"]["start_time"]
        end_time = time.time()
        analyzer = LatencyDuringOperationsPerformanceAnalyzer
        results_analyzer = analyzer(es_index=self._test_index,
                                    es_doc_type=self._es_doc_type,
                                    email_recipients=self.params.get(
                                        'email_recipients'),
                                    events=get_events_grouped_by_category(
                                        _registry=self.events_processes_registry))
        with open(self.latency_results_file, encoding="utf-8") as file:
            latency_results = json.load(file)
        self.log.debug('latency_results were loaded from file %s and its result is %s',
                       self.latency_results_file, latency_results)
        benchmarks_results = self.db_cluster.get_node_benchmarks_results() if self.db_cluster else {}
        if latency_results and self.create_stats:
            workload = self._test_index.split("-")[-1]
            histogram_total_data = self.get_cs_range_histogram(stress_operation=workload,
                                                               start_time=start_time,
                                                               end_time=end_time)
            histogram_data_by_interval = self.get_cs_range_histogram_by_interval(stress_operation=workload,
                                                                                 start_time=start_time,
                                                                                 end_time=end_time)
            latency_results["summary"] = {"hdr_summary": histogram_total_data,
                                          "hdr": histogram_data_by_interval}
            latency_results = calculate_latency(latency_results)
            latency_results = analyze_hdr_percentiles(latency_results)
            with open(self.latency_results_file, 'w', encoding="utf-8") as file:
                json.dump(latency_results, file)
            self.log.debug('collected latency values are: %s', latency_results)
            self.update({"latency_during_ops": latency_results})
            self.update_test_details()
            try:
                results_analyzer.check_regression(test_id=self._test_id, data=latency_results,
                                                  node_benchmarks=benchmarks_results,
                                                  email_subject_postfix=self.params.get('email_subject_postfix'))
            except Exception as exc:  # noqa: BLE001
                TestFrameworkEvent(
                    message='Failed to check regression',
                    source=self.__class__.__name__,
                    source_method='check_regression',
                    exception=exc
                ).publish_or_dump()

    def check_regression(self):
        results_analyzer = PerformanceResultsAnalyzer(es_index=self._test_index,
                                                      es_doc_type=self._es_doc_type,
                                                      email_recipients=self.params.get('email_recipients'),
                                                      events=get_events_grouped_by_category(
                                                          _registry=self.events_processes_registry,
                                                          limit=self.params.get('events_limit_in_email'))
                                                      )
        is_gce = bool(self.params.get('cluster_backend') == 'gce')
        benchmarks_results = self.db_cluster.get_node_benchmarks_results() if self.db_cluster else {}
        try:
            results_analyzer.check_regression(self._test_id, is_gce,
                                              email_subject_postfix=self.params.get('email_subject_postfix'),
                                              use_wide_query=True,
                                              node_benchmarks=benchmarks_results,
                                              extra_jobs_to_compare=self.params.get('perf_extra_jobs_to_compare'))
        except Exception as exc:  # noqa: BLE001
            TestFrameworkEvent(
                message='Failed to check regression',
                source=self.__class__.__name__,
                source_method='check_regression',
                exception=exc
            ).publish_or_dump()

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
                                                                    is_gce=is_gce,
                                                                    extra_jobs_to_compare=self.params.get('perf_extra_jobs_to_compare'))
        except Exception as exc:  # noqa: BLE001
            TestFrameworkEvent(
                message='Failed to check regression',
                source=self.__class__.__name__,
                source_method='check_regression',
                exception=exc
            ).publish_or_dump()

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
        except Exception as exc:  # noqa: BLE001
            TestFrameworkEvent(
                message='Failed to check regression',
                source=self.__class__.__name__,
                source_method='check_regression',
                exception=exc
            ).publish_or_dump()

    def check_specified_stats_regression(self, stats):
        perf_analyzer = SpecifiedStatsPerformanceAnalyzer(es_index=self._test_index,
                                                          es_doc_type=self._es_doc_type,
                                                          email_recipients=self.params.get('email_recipients'),
                                                          events=get_events_grouped_by_category(
                                                              _registry=self.events_processes_registry))
        try:
            perf_analyzer.check_regression(self._test_id, stats)
        except Exception as exc:  # noqa: BLE001
            TestFrameworkEvent(
                message='Failed to check regression',
                source=self.__class__.__name__,
                source_method='check_regression',
                exception=exc
            ).publish_or_dump()

    @property
    def is_compaction_running(self):
        compaction_query = "sum(scylla_compaction_manager_compactions{})"
        now = time.time()
        results = self.prometheus_db.query(query=compaction_query, start=now - 60, end=now)
        self.log.debug("scylla_compaction_manager_compactions: {results}".format(**locals()))
        assert results or results == [], "No results from Prometheus"
        # if any result values is not zero - there are running compactions.
        return any((float(v[1]) for v in results[0]["values"]))

    def wait_compactions_are_running(self, n=20, sleep_time=60):  # pylint: disable=invalid-name
        # Wait until there are running compactions
        @retrying(n=n, sleep_time=sleep_time, allowed_exceptions=(AssertionError,))
        def _is_compaction_running():
            assert self.is_compaction_running, "Waiting for compaction to start"
        _is_compaction_running()

    @measure_time
    def wait_no_compactions_running(self, n=80, sleep_time=60):  # pylint: disable=invalid-name
        # Wait until there are no running compactions
        @retrying(n=n, sleep_time=sleep_time, allowed_exceptions=(AssertionError,))
        def _is_no_compaction_running():
            assert not self.is_compaction_running, "Waiting until all compactions settle down"
        _is_no_compaction_running()

    def metric_has_data(self, metric_query, n=80, sleep_time=60, ):  # pylint: disable=invalid-name
        """
        wait for any prometheus metric to have data in it

        example: wait 5 mins for cassandra stress to start writing:

            self.metric_has_data(metric_query='sct_cassandra_stress_write_gauge{type="ops"}', n=5)

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
                    {"name": "k8s_cluster_must_gather",
                     "nodes": [],
                     "collector": KubernetesMustGatherLogCollector,
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
        elif self.db_cluster:
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
            self._argus_add_relocatable_pkg(email_data)
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.error("Error while saving email data. Error: %s\nTraceback: %s", exc, traceback.format_exc())

        grafana_screenshots = []
        try:
            grafana_screenshots = self.monitors.get_grafana_screenshots_from_all_monitors(
                self.start_time) if self.monitors else {}
            self.argus_collect_screenshots(grafana_screenshots)
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.exception("Error while collecting screenshots:", exc_info=exc)

        json_file_path = os.path.join(self.logdir, "email_data.json")

        if email_data:
            email_data['grafana_screenshots'] = grafana_screenshots
            email_data["reporter"] = self.email_reporter.__class__.__name__
            self.log.debug('Save email data to file %s', json_file_path)
            self.log.debug('Email data: %s', email_data)
            save_email_data_to_file(email_data, json_file_path)

    def argus_collect_screenshots(self, grafana_screenshots: list) -> None:
        if grafana_screenshots:
            self.test_config.argus_client().submit_screenshots(grafana_screenshots)

    def _argus_add_relocatable_pkg(self, email_data):
        """Adds Package with url to relocatable package in Argus.

        What is relocatable package:
        https://docs.scylladb.com/stable/getting-started/install-scylla/unified-installer.html
        """
        if relocatable_pkg := email_data.get("relocatable_pkg"):
            self.test_config.argus_client().submit_packages(
                [Package(name="relocatable_pkg", date="", version=relocatable_pkg, revision_id="", build_id="")])

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
        config_file_name = ";".join(os.path.splitext(os.path.basename(cfg))[
                                    0] for cfg in self.params.get("config_files"))
        test_status = self.get_test_status()
        backend = self.params.get("cluster_backend")
        build_id = f'#{os.environ.get("BUILD_NUMBER")}' if os.environ.get('BUILD_NUMBER', '') else ''
        scylla_version = alive_node.scylla_version_detailed if alive_node else "N/A"
        kernel_version = alive_node.kernel_version if alive_node else "N/A"

        if backend in ("aws", "aws-siren", "k8s-eks"):
            scylla_instance_type = self.params.get("instance_type_db") or "Unknown"
            region_name = self.params.region_names
        elif backend in ("gce", "gce-siren", "k8s-gke"):
            scylla_instance_type = self.params.get("gce_instance_type_db") or "Unknown"
            region_name = self.params.get("gce_datacenter")
        elif backend in ("azure"):
            scylla_instance_type = self.params.get("azure_instance_type_db") or "Unknown"
            region_name = self.params.get("azure_region_name")
        elif backend in ("baremetal", "docker"):
            scylla_instance_type = "N/A"
            region_name = "N/A"
        else:
            self.log.error("Don't know how to get instance type for the '%s' backend.", backend)
            scylla_instance_type = "N/A"
            region_name = "N/A"

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
                "rack_aware_policy": self.is_rack_aware_policy,
                "restore_monitor_job_base_link": restore_monitor_job_base_link,
                "relocatable_pkg": get_relocatable_pkg_url(scylla_version)}

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

    @property
    def is_rack_aware_policy(self) -> bool:
        all_events = get_events_grouped_by_category()
        for event_str in all_events["NORMAL"]:
            if "type=RackAwarePolicy" in event_str:
                return True
        return False

    def get_cs_range_histogram(self, stress_operation: str,
                               start_time: float, end_time: float,
                               tag_type: CSHistogramTagTypes = CSHistogramTagTypes.LATENCY) -> dict[str, Any]:
        if not self.params["use_hdr_cs_histogram"]:
            return {}
        self.log.info("Build HDR histogram with start time: %s, end time: %s; for operation: %s",
                      start_time, end_time, stress_operation)
        histogram_data = make_cs_range_histogram_summary(
            workload=CSWorkloadTypes(stress_operation),
            base_path=self.loaders.logdir, start_time=start_time, end_time=end_time,
            tag_type=tag_type)
        self.log.info("HDR histogram summary result: %s", histogram_data)
        return histogram_data[0] if histogram_data else {}

    def get_cs_range_histogram_by_interval(
            self, stress_operation: str,
            start_time: float, end_time: float, time_interval: int = 600,
            tag_type: CSHistogramTagTypes = CSHistogramTagTypes.LATENCY) -> list[dict[str, Any]]:
        if not self.params["use_hdr_cs_histogram"]:
            return []
        self.log.info("Build HDR histogram with start time: %s, end time: %s, time interval: %s for operation: %s",
                      start_time, end_time, time_interval, stress_operation)
        return make_cs_range_histogram_summary_by_interval(
            workload=CSWorkloadTypes(stress_operation),
            path=self.loaders.logdir, start_time=start_time, end_time=end_time,
            interval=time_interval, tag_type=tag_type)
