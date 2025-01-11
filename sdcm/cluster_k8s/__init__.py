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

# pylint: disable=too-many-arguments
# pylint: disable=too-many-lines

from __future__ import annotations

import os
import re
import abc
import json
import math
import shutil
import tempfile
import time
import base64
import random
import logging
import traceback
import contextlib
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from difflib import unified_diff
from functools import cached_property, partialmethod, partial
from tempfile import NamedTemporaryFile, TemporaryDirectory
from textwrap import dedent
from threading import Lock, RLock
from typing import Optional, Union, List, Dict, Any, ContextManager, Type, Tuple, Callable

import yaml
import kubernetes as k8s
from kubernetes.client import exceptions as k8s_exceptions
from kubernetes.client import V1ConfigMap
from kubernetes.dynamic.resource import Resource, ResourceField, ResourceInstance, ResourceList, Subresource
import invoke
from invoke.exceptions import CommandTimedOut

from sdcm import sct_abs_path, cluster
from sdcm.cluster import ClusterNodesNotReady
from sdcm.provision.scylla_yaml.scylla_yaml import ScyllaYaml
from sdcm.sct_config import init_and_verify_sct_config
from sdcm.test_config import TestConfig
from sdcm.db_stats import PrometheusDBStats
from sdcm.remote import LOCALRUNNER, NETWORK_EXCEPTIONS
from sdcm.remote.kubernetes_cmd_runner import (
    KubernetesCmdRunner,
    KubernetesPodRunner,
)
from sdcm.coredump import CoredumpExportFileThread
from sdcm.log import SDCMAdapter
from sdcm.mgmt import AnyManagerCluster
from sdcm.sct_events.health import ClusterHealthValidatorEvent
from sdcm.sct_events.system import TestFrameworkEvent
import sdcm.utils.sstable.load_inventory as datasets
from sdcm.utils.adaptive_timeouts import adaptive_timeout, Operations
from sdcm.utils.ci_tools import get_test_name
from sdcm.utils.common import download_from_github, shorten_cluster_name, walk_thru_data
from sdcm.utils.k8s import (
    add_pool_node_affinity,
    convert_cpu_units_to_k8s_value,
    convert_cpu_value_from_k8s_to_units,
    convert_memory_units_to_k8s_value,
    convert_memory_value_from_k8s_to_units,
    get_helm_pool_affinity_values,
    get_pool_affinity_modifiers,
    get_preferred_pod_anti_affinity_values,
    ApiCallRateLimiter,
    JSON_PATCH_TYPE,
    KubernetesOps,
    KUBECTL_TIMEOUT,
    HelmValues,
    ScyllaPodsIPChangeTrackerThread,
    TokenUpdateThread,
)
from sdcm.utils.decorators import log_run_info, retrying
from sdcm.utils.decorators import timeout as timeout_wrapper
from sdcm.utils.k8s.chaos_mesh import ChaosMesh
from sdcm.utils.remote_logger import get_system_logging_thread, CertManagerLogger, ScyllaOperatorLogger, \
    KubectlClusterEventsLogger, ScyllaManagerLogger, KubernetesWrongSchedulingLogger, HaproxyIngressLogger
from sdcm.utils.sstable.load_utils import SstableLoadUtils
from sdcm.utils.version_utils import ComparableScyllaOperatorVersion
from sdcm.wait import wait_for
from sdcm.cluster_k8s.operator_monitoring import ScyllaOperatorLogMonitoring


ANY_KUBERNETES_RESOURCE = Union[  # pylint: disable=invalid-name
    Resource, ResourceField, ResourceInstance, ResourceList, Subresource,
]
NAMESPACE_CREATION_LOCK = Lock()
NODE_INIT_LOCK = Lock()

CERT_MANAGER_TEST_CONFIG = sct_abs_path("sdcm/k8s_configs/cert-manager-test.yaml")
LOADER_POD_CONFIG_PATH = sct_abs_path("sdcm/k8s_configs/loaders/pod.yaml")
LOADER_STS_CONFIG_PATH = sct_abs_path("sdcm/k8s_configs/loaders/sts.yaml")
LOCAL_PROVISIONER_FILE = sct_abs_path("sdcm/k8s_configs/static-local-volume-provisioner.yaml")
LOCAL_MINIO_DIR = sct_abs_path("sdcm/k8s_configs/minio")
INGRESS_CONTROLLER_CONFIG_PATH = sct_abs_path("sdcm/k8s_configs/ingress-controller")
PROMETHEUS_OPERATOR_CONFIG_PATH = sct_abs_path("sdcm/k8s_configs/monitoring/prometheus-operator")
SCYLLA_MONITORING_CONFIG_PATH = sct_abs_path("sdcm/k8s_configs/monitoring/scylladbmonitoring-template.yaml")

SCYLLA_API_VERSION = "scylla.scylladb.com/v1"
SCYLLA_CLUSTER_RESOURCE_KIND = "ScyllaCluster"
DEPLOY_SCYLLA_CLUSTER_DELAY = 15  # seconds
SCYLLA_OPERATOR_NAMESPACE = "scylla-operator"
SCYLLA_MANAGER_NAMESPACE = "scylla-manager"
SCYLLA_NAMESPACE = "scylla"
INGRESS_CONTROLLER_NAMESPACE = "haproxy-controller"
PROMETHEUS_OPERATOR_NAMESPACE = "prometheus-operator"
LOADER_NAMESPACE = "sct-loaders"
MINIO_NAMESPACE = "minio"
SCYLLA_CONFIG_NAME = "scylla-config"
SCYLLA_AGENT_CONFIG_NAME = "scylla-agent-config"

K8S_LOCAL_VOLUME_PROVISIONER_VERSION = "0.3.0"  # without 'v' prefix
SCYLLA_MANAGER_AGENT_VERSION_IN_SCYLLA_MANAGER = "3.2.6"

# NOTE: add custom annotations to a ServiceAccount used by a ScyllaCluster
#       It is needed to make sure that annotations survive operator upgrades
SCYLLA_CLUSTER_SA_ANNOTATION_KEY_PREFIX = "sct-custom-annotation-key-"
SCYLLA_CLUSTER_SA_ANNOTATION_VALUE_PREFIX = "sct-custom-annotation-value-"
SCYLLA_CLUSTER_SA_ANNOTATIONS = {
    f'{SCYLLA_CLUSTER_SA_ANNOTATION_KEY_PREFIX}1': f'{SCYLLA_CLUSTER_SA_ANNOTATION_VALUE_PREFIX}1',
    f'{SCYLLA_CLUSTER_SA_ANNOTATION_KEY_PREFIX}2': f'{SCYLLA_CLUSTER_SA_ANNOTATION_VALUE_PREFIX}2',
}

# Resources that are used by container deployed by scylla-operator on scylla nodes
OPERATOR_CONTAINERS_RESOURCES = {
    'cpu': 0.05,
    'memory': 0.01,
}

# Other common resources which get deployed on each scylla node such as 'kube-proxy'
# EKS: between 130m-260m CPU
# GKE: between 200m-300m CPU and 250Mi RAM
# Above numbers are "explicit" reservations. So, reserve a bit more for other common pods.
COMMON_CONTAINERS_RESOURCES = {
    'cpu': 0.5131,
    'memory': 0.81,
}

SCYLLA_MANAGER_AGENT_RESOURCES = {
    'cpu': 0.04,
    'memory': 0.489,  # 0.489 will give 500Mb as a result
}

LOGGER = logging.getLogger(__name__)


class CloudK8sNodePool(metaclass=abc.ABCMeta):  # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            k8s_cluster: 'KubernetesCluster',
            name: str,
            num_nodes: int,
            instance_type: str,
            image_type: str,
            disk_size: int = None,
            disk_type: str = None,
            labels: dict = None,
            taints: list = None,
            is_deployed: bool = False):
        self.k8s_cluster = k8s_cluster
        self.name = name
        self.num_nodes = int(num_nodes)
        self.instance_type = instance_type
        self.disk_size = disk_size
        self.disk_type = disk_type
        self.image_type = image_type
        self.labels = labels
        self.taints = taints
        self.is_deployed = is_deployed

    @property
    def tags(self):
        return self.k8s_cluster.tags

    @abc.abstractmethod
    def deploy(self):
        pass

    def deploy_and_wait_till_ready(self):
        if not self.is_deployed:
            self.deploy()
        self.wait_for_nodes_readiness()

    @abc.abstractmethod
    def undeploy(self):
        pass

    @abc.abstractmethod
    def resize(self, num_nodes: int):
        pass

    def __str__(self):
        data = [f'name="{self.name}"', *[f'{param}="{value}"' for param, value in self.__dict__.items() if
                                         param not in ['name', 'k8s_cluster']]]
        return f"<{self.__class__.__name__}:{', '.join(data)}>"

    @cached_property
    def affinity_modifiers(self) -> List[Callable]:
        return get_pool_affinity_modifiers(self.pool_label_name, self.name)

    @cached_property
    def helm_affinity_values(self) -> HelmValues:
        return HelmValues(get_helm_pool_affinity_values(self.pool_label_name, self.name))

    @cached_property
    def pool_label_name(self) -> str:
        return self.k8s_cluster.POOL_LABEL_NAME

    @cached_property
    def cpu_and_memory_capacity(self) -> Tuple[float, float]:
        for item in self.k8s_cluster.k8s_core_v1_api.list_node().items:
            if item.metadata.labels.get(self.pool_label_name, '') == self.name:
                capacity = item.status.allocatable
                return convert_cpu_value_from_k8s_to_units(capacity['cpu']), convert_memory_value_from_k8s_to_units(
                    capacity['memory'])
        raise RuntimeError(f"{self.k8s_cluster.region_name}: Can't find any node for pool '{self.name}'")

    @property
    def cpu_capacity(self) -> float:
        return self.cpu_and_memory_capacity[0]

    @property
    def memory_capacity(self) -> float:
        return self.cpu_and_memory_capacity[1]

    @property
    def readiness_timeout(self) -> int:
        return 10 + (10 * self.num_nodes)

    @property
    def nodes(self):
        try:
            return self.k8s_cluster.k8s_core_v1_api.list_node(label_selector=f'{self.pool_label_name}={self.name}')
        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            self.k8s_cluster.log.debug("Failed to get nodes list: %s", str(details))
            return {}

    def wait_for_nodes_readiness(self):
        readiness_timeout = self.readiness_timeout

        @timeout_wrapper(
            message=(
                f"{self.k8s_cluster.region_name}: Wait for {self.num_nodes} node(s)"
                f" in the '{self.name}' pool to be ready..."),
            sleep_time=30,
            timeout=readiness_timeout * 60)
        def wait_nodes_are_ready():
            # To make it more informative in worst case scenario made it repeat 5 times, by readiness_timeout // 5
            result = self.k8s_cluster.kubectl_no_wait(
                f"wait --timeout={self.readiness_timeout // 5}m -l {self.pool_label_name}={self.name} "
                f"--for=condition=Ready node",
                timeout=readiness_timeout // 5 * 60 + 10, verbose=False)
            if result.stdout.count('condition met') != self.num_nodes:
                raise RuntimeError('Not all nodes reported')

        wait_nodes_are_ready()


class KubernetesCluster(metaclass=abc.ABCMeta):  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    AUXILIARY_POOL_NAME = 'auxiliary-pool'
    SCYLLA_POOL_NAME = 'scylla-pool'
    MONITORING_POOL_NAME = 'monitoring-pool'
    LOADER_POOL_NAME = 'loader-pool'
    POOL_LABEL_NAME: str = None
    IS_NODE_TUNING_SUPPORTED: bool = False
    NODE_PREPARE_FILE = None
    NODE_CONFIG_CRD_FILE = None
    TOKEN_UPDATE_NEEDED = True

    api_call_rate_limiter: Optional[ApiCallRateLimiter] = None

    _cert_manager_journal_thread: Optional[CertManagerLogger] = None
    _scylla_manager_journal_thread: Optional[ScyllaManagerLogger] = None
    _scylla_operator_journal_thread: Optional[ScyllaOperatorLogger] = None
    _scylla_operator_scheduling_thread: Optional[KubernetesWrongSchedulingLogger] = None
    _haproxy_ingress_log_thread: Optional[HaproxyIngressLogger] = None
    _scylla_cluster_events_threads: Dict[str, KubectlClusterEventsLogger] = {}
    _scylla_operator_log_monitor_thread: Optional[ScyllaOperatorLogMonitoring] = None
    _token_update_thread: Optional[TokenUpdateThread] = None
    scylla_pods_ip_change_tracker_thread: Optional[ScyllaPodsIPChangeTrackerThread] = None

    pools: Dict[str, CloudK8sNodePool]
    scylla_pods_ip_mapping = {}

    def __init__(self, params: dict, user_prefix: str = '', region_name: str = None, cluster_uuid: str = None):
        self.test_config = TestConfig()
        self.pools = {}
        if cluster_uuid is None:
            self.uuid = self.test_config.test_id()
        else:
            self.uuid = cluster_uuid
        self.region_name = region_name
        self.rack_name = "rack-1"
        self.shortid = str(self.uuid)[:8]
        self.name = f"{user_prefix}-{self.shortid}"
        self.params = params
        self.api_call_rate_limiter = None
        self.k8s_scylla_cluster_name = self.params.get('k8s_scylla_cluster_name')
        self.scylla_config_lock = RLock()
        self.scylla_restart_required = False
        self.scylla_cpu_limit = None
        self.scylla_memory_limit = None
        self.calculated_loader_cpu_limit = None
        self.calculated_loader_memory_limit = None
        self.calculated_loader_affinity_modifiers = []
        self.perf_pods_labels = [
            ('app.kubernetes.io/name', 'scylla-node-config'),
            ('app.kubernetes.io/name', 'node-config'),
            ('scylla-operator.scylladb.com/node-config-job-type', 'Node'),
            ('scylla-operator.scylladb.com/node-config-job-type', 'Containers'),
        ]
        self._scylla_cluster_events_threads = {}
        self.chaos_mesh = ChaosMesh(self)

    # NOTE: Following class attr(s) are defined for consumers of this class
    #       such as 'sdcm.utils.remote_logger.ScyllaOperatorLogger'.
    _scylla_operator_namespace = SCYLLA_OPERATOR_NAMESPACE
    _scylla_manager_namespace = SCYLLA_MANAGER_NAMESPACE

    @cached_property
    def log(self):
        # return logging.getLogger(f"{__name__} | {self.region_name}")
        return SDCMAdapter(LOGGER, extra={'prefix': self.region_name})

    @cached_property
    def tenants_number(self) -> int:
        return self.params.get("k8s_tenants_num") or 1

    @property
    def allowed_labels_on_scylla_node(self):
        # keep pods labels that are allowed to be scheduled on the Scylla node
        return []

    @cached_property
    def cluster_backend(self):
        return self.params.get("cluster_backend")

    @property
    def k8s_server_url(self) -> Optional[str]:
        return None

    @cached_property
    def short_cluster_name(self):
        return shorten_cluster_name(self.name, 40).replace('_', '-')

    kubectl_cmd = partialmethod(KubernetesOps.kubectl_cmd)
    apply_file = partialmethod(KubernetesOps.apply_file)

    def kubectl_no_wait(self, *command, namespace=None, timeout=KUBECTL_TIMEOUT, remoter=None, ignore_status=False,
                        verbose=True):
        return KubernetesOps.kubectl(self, *command, namespace=namespace, timeout=timeout, remoter=remoter,
                                     ignore_status=ignore_status, verbose=verbose)

    def kubectl(self, *command, namespace=None, timeout=KUBECTL_TIMEOUT, remoter=None, ignore_status=False,
                verbose=True):
        if self.api_call_rate_limiter:
            self.api_call_rate_limiter.wait()
        return KubernetesOps.kubectl(self, *command, namespace=namespace, timeout=timeout, remoter=remoter,
                                     ignore_status=ignore_status, verbose=verbose)

    def kubectl_wait(self, *command, namespace=None, timeout=KUBECTL_TIMEOUT, remoter=None, verbose=True):
        """
        We use kubectl wait to wait till all resources get into proper state
        there are two problems with this:
        1. kubectl wait fails when no resource matched criteria
        2. if resources are provisioned gradually, kubectl wait can slip thrue crack when half of the resource
          provisioned and the rest is not even deployed

        This function is to address these problem by wrapping 'kubectl wait' and make it restarted when no resource
        are there to tackle problem #1 and track number of resources it reported and wait+rerun if resource number
        had changed to tackle problem #2
        """
        last_resource_count = -1

        @timeout_wrapper(timeout=timeout, sleep_time=5)
        def wait_body():
            nonlocal last_resource_count
            result = self.kubectl('wait --timeout=1m', *command, namespace=namespace, timeout=timeout, remoter=remoter,
                                  verbose=verbose)
            current_resource_count = result.stdout.count('condition met')
            if current_resource_count != last_resource_count:
                last_resource_count = current_resource_count
                time.sleep(10)
                raise RuntimeError("Retry since matched resource count has changed")
            return result
        return wait_body()

    def kubectl_multi_cmd(self, *command, namespace=None, timeout=KUBECTL_TIMEOUT, remoter=None, ignore_status=False,
                          verbose=True):
        if self.api_call_rate_limiter:
            self.api_call_rate_limiter.wait()
        return KubernetesOps.kubectl_multi_cmd(self, *command, namespace=namespace, timeout=timeout, remoter=remoter,
                                               ignore_status=ignore_status, verbose=verbose)

    @property
    def helm(self):
        if self.api_call_rate_limiter:
            self.api_call_rate_limiter.wait()
        return partial(self.test_config.tester_obj().localhost.helm, self)

    @property
    def helm_install(self):
        if self.api_call_rate_limiter:
            self.api_call_rate_limiter.wait()
        return partial(self.test_config.tester_obj().localhost.helm_install, self)

    @property
    def helm_upgrade(self):
        if self.api_call_rate_limiter:
            self.api_call_rate_limiter.wait()
        return partial(self.test_config.tester_obj().localhost.helm_upgrade, self)

    @cached_property
    def kube_config_dir_path(self):
        _kube_config_dir_path = os.path.join(os.path.expanduser(self.logdir), ".kube")
        os.makedirs(_kube_config_dir_path, exist_ok=True)
        return _kube_config_dir_path

    @cached_property
    def kube_config_path(self):
        return os.path.join(self.kube_config_dir_path, "config")

    @cached_property
    def kubectl_token_path(self):
        return os.path.join(self.kube_config_dir_path, "token")

    @cached_property
    def helm_dir_path(self):
        _helm_dir_path = os.path.join(os.path.dirname(self.kube_config_dir_path), ".helm")
        os.makedirs(_helm_dir_path, exist_ok=True)
        return _helm_dir_path

    def create_namespace(self, namespace: str) -> None:
        self.log.info("Create '%s' namespace", namespace)
        namespaces = yaml.safe_load(self.kubectl("get namespaces -o yaml").stdout)
        if not [ns["metadata"]["name"] for ns in namespaces["items"] if ns["metadata"]["name"] == namespace]:
            self.kubectl(f"create namespace {namespace}")
        else:
            self.log.warning("The '%s' namespace already exists.")

    @cached_property
    def cert_manager_log(self) -> str:
        return os.path.join(self.logdir, "cert_manager.log")

    @cached_property
    def scylla_manager_log(self) -> str:
        return os.path.join(self.logdir, "scylla_manager.log")

    @cached_property
    def haproxy_ingress_log(self) -> str:
        return os.path.join(self.logdir, "haproxy_ingress.log")

    def start_cert_manager_journal_thread(self) -> None:
        self._cert_manager_journal_thread = CertManagerLogger(self, self.cert_manager_log)
        self._cert_manager_journal_thread.start()

    def start_scylla_manager_journal_thread(self):
        self._scylla_manager_journal_thread = ScyllaManagerLogger(self, self.scylla_manager_log)
        self._scylla_manager_journal_thread.start()

    def start_haproxy_ingress_log_thread(self) -> None:
        self._haproxy_ingress_log_thread = HaproxyIngressLogger(self, self.haproxy_ingress_log)
        self._haproxy_ingress_log_thread.start()

    def set_nodeselector_for_deployments(self, pool_name: str,
                                         namespace: str = "kube-system",
                                         selector: str = "") -> None:
        """Sets node selector for deployment objects in a namespace.

        pool_name: any pool name. For example 'auxiliary-pool' in EKS or 'default-pool' in GKE.
        namespace: any namespace to look for deployments.
        selector: any selector to use for filtering deployments in a namespace.
            Example: 'foo=bar' or '' (empty string means 'all').

        Use case: deploying Scylla cluster on EKS or GKE backends we don't want to have
        unexpected pods be placed onto Scylla K8S nodes. Also, we don't want to stick
        to any of deployment names in case it's name or number changes.
        """
        data = {"spec": {"template": {"spec": {"nodeSelector": {
            self.POOL_LABEL_NAME: pool_name,
        }}}}}
        deployment_names = self.kubectl(
            f"get deployments -l '{selector}' --no-headers -o custom-columns=:.metadata.name",
            namespace=namespace).stdout.split()
        for deployment_name in deployment_names:
            self.kubectl(
                f"patch deployments {deployment_name} -p '{json.dumps(data)}'",
                namespace=namespace)

    @log_run_info
    def deploy_cert_manager(self, pool_name: str = None) -> None:
        cert_manager_namespace = "cert-manager"
        if not self.params.get('reuse_cluster'):
            if pool_name is None:
                pool_name = self.AUXILIARY_POOL_NAME

            self.log.info("Deploy cert-manager")
            self.create_namespace(cert_manager_namespace)
            self.log.debug(self.helm("repo add jetstack https://charts.jetstack.io"))

            if pool_name:
                values_dict = get_helm_pool_affinity_values(self.POOL_LABEL_NAME, pool_name)
                values_dict["cainjector"] = {"affinity": values_dict["affinity"]}
                values_dict["webhook"] = {"affinity": values_dict["affinity"]}
                helm_values = HelmValues(values_dict)
            else:
                helm_values = HelmValues()

            helm_values.set('installCRDs', True)

            self.log.debug(self.helm(
                "install cert-manager jetstack/cert-manager"
                f" --version v{self.params.get('k8s_cert_manager_version')}",
                namespace=cert_manager_namespace, values=helm_values))

        self.kubectl_wait("--all --for=condition=Ready pod", namespace="cert-manager",
                          timeout=600)
        wait_for(
            self.check_if_cert_manager_fully_functional,
            text='Waiting for cert-manager to become fully operational',
            timeout=10 * 60,
            step=10,
            throw_exc=True)

        self.start_cert_manager_journal_thread()

    def get_latest_chart_version(self, local_chart_path: str) -> str:
        all_versions = yaml.safe_load(self.helm(
            f"search repo {local_chart_path} --devel --versions -o yaml"))
        assert isinstance(all_versions, list), f"Expected list of data, got: {type(all_versions)}"
        # NOTE: ignore versions like 'v1.8.0-alpha.0' because they refer to the oldest full version
        #       in each 'minor' family.
        current_newest_version, current_newest_version_str = None, ''
        for version_object in all_versions:
            if version_object["version"].count('-') == 1:
                continue
            if ComparableScyllaOperatorVersion(version_object["version"]) > (
                    current_newest_version or '0.0.0'):
                current_newest_version = ComparableScyllaOperatorVersion(version_object["version"])
                current_newest_version_str = version_object["version"]
                continue
        return current_newest_version_str

    @cached_property
    def scylla_operator_chart_version(self):
        self.log.debug(self.helm(
            f"repo add scylla-operator {self.params.get('k8s_scylla_operator_helm_repo')}"))

        # NOTE: 'scylla-operator' and 'scylla-manager' chart versions are always the same.
        #       So, we can reuse one for another.
        chart_version = self.params.get("k8s_scylla_operator_chart_version").strip().lower()
        if chart_version in ("", "latest"):
            chart_version = self.get_latest_chart_version("scylla-operator/scylla-operator")
            self.log.info(
                "Using automatically found following latest scylla-operator chart version: %s",
                chart_version)
        else:
            self.log.info(
                "Using following predefined scylla-operator chart version: %s", chart_version)
        return chart_version

    def get_operator_image(self, repo: str = '', chart_version: str = '') -> str:
        if image := self.params.get("k8s_scylla_operator_docker_image"):
            return image

        # Get 'appVersion' field from the Helm chart which stores image's tag
        repo = repo or self.params.get('k8s_scylla_operator_helm_repo')
        chart_name = "scylla-operator"
        chart_version = chart_version or self.scylla_operator_chart_version
        chart_info = self.helm(
            f"show chart {chart_name} --devel --repo {repo} --version {chart_version}")
        for line in chart_info.split("\n"):
            if line.startswith("appVersion:"):
                # NOTE: 'appVersion' key may have different formats for it's value.
                #       Value may or may not be wrapped in quotes.
                # $ helm show chart scylla-operator --devel --repo %repo% --version v1.6.0-rc.0
                #     apiVersion: v2
                #     appVersion: "1.6"
                #     ...
                #
                # $ helm show chart scylla-operator --devel --repo %repo% --version v1.5.0-rc.0
                #     apiVersion: v2
                #     appVersion: 1.5.0-rc.0
                #     ...
                #
                # Details: https://helm.sh/docs/topics/charts/#the-appversion-field
                found_app_version = line.split(':')[-1].strip().replace('"', '').replace("'", "")
                return f"scylladb/scylla-operator:{found_app_version}"
        raise ValueError(
            f"Cannot get operator image version from the '{chart_name}' chart located at "
            f"'{repo}' having '{chart_version}' version")

    @log_run_info
    def deploy_scylla_manager(self, pool_name: str = None) -> None:
        # Calculate options values which must be set
        #
        # image.tag                  -> self.params.get('mgmt_docker_image').split(':')[-1]
        # controllerImage.repository -> self.params.get(
        #                                   'k8s_scylla_operator_docker_image').split('/')[0]
        # controllerImage.tag        -> self.params.get(
        #                                   'k8s_scylla_operator_docker_image').split(':')[-1]
        if not self.params.get('reuse_cluster'):
            self.log.info("Deploy scylla-manager")

            helm_affinity = get_helm_pool_affinity_values(
                self.POOL_LABEL_NAME, pool_name) if pool_name else {}
            values = HelmValues(**helm_affinity)
            values.set("controllerAffinity", helm_affinity.get("affinity", {}))
            storage_config = {"capacity": "10Gi"}
            if self.cluster_backend == "k8s-eks":
                storage_config["storageClassName"] = "gp3-3k-iops"
            values.set("scylla", {
                "developerMode": True,
                "datacenter": "manager-dc",
                "agentImage": {"tag": SCYLLA_MANAGER_AGENT_VERSION_IN_SCYLLA_MANAGER},
                "racks": [{
                    "name": "manager-rack",
                    "members": 1,
                    "placement": helm_affinity["affinity"],
                    "storage": storage_config,
                    "resources": {
                        "limits": {"cpu": 1, "memory": "200Mi"},
                        "requests": {"cpu": 1, "memory": "200Mi"},
                    },
                }],
            })

            mgmt_docker_image_tag = self.params.get('mgmt_docker_image').split(':')[-1]
            if mgmt_docker_image_tag:
                values.set('image.tag', mgmt_docker_image_tag)

            scylla_operator_repo_base = self.params.get(
                'k8s_scylla_operator_docker_image').rsplit('/', 1)[0]
            if scylla_operator_repo_base:
                values.set('controllerImage.repository', scylla_operator_repo_base)

            scylla_operator_image_tag = self.params.get(
                'k8s_scylla_operator_docker_image').split(':', 1)[-1]
            if scylla_operator_image_tag:
                values.set('controllerImage.tag', scylla_operator_image_tag)

            self.create_namespace(SCYLLA_MANAGER_NAMESPACE)

            # Install and wait for initialization of the Scylla Manager chart
            self.log.debug(self.helm_install(
                target_chart_name="scylla-manager",
                source_chart_name="scylla-operator/scylla-manager",
                version=self.scylla_operator_chart_version,
                use_devel=True,
                values=values,
                namespace=SCYLLA_MANAGER_NAMESPACE,
            ))

        self.kubectl_wait(
            "--all --for=condition=Ready pod",
            namespace=SCYLLA_MANAGER_NAMESPACE,
            timeout=600,
        )
        self.start_scylla_manager_journal_thread()

    def check_if_cert_manager_fully_functional(self) -> bool:
        # Cert-manager readiness status does not guarantee that it is fully operational
        # This function checks it if is operational via deploying ca and issuing certificate
        try:
            self.apply_file(CERT_MANAGER_TEST_CONFIG)
            return True
        finally:
            self.kubectl(f'delete -f {CERT_MANAGER_TEST_CONFIG}', ignore_status=True)

    @property
    def scylla_operator_log(self) -> str:
        return os.path.join(self.logdir, "scylla_operator.log")

    def scylla_cluster_event_log(self, namespace: str = SCYLLA_NAMESPACE) -> str:
        return os.path.join(self.logdir, f"{namespace}_cluster_events.log")

    def start_scylla_operator_journal_thread(self) -> None:
        self._scylla_operator_journal_thread = ScyllaOperatorLogger(self, self.scylla_operator_log)
        self._scylla_operator_journal_thread.start()
        self._scylla_operator_scheduling_thread = KubernetesWrongSchedulingLogger(self, self.scylla_operator_log)
        self._scylla_operator_scheduling_thread.start()
        self._scylla_operator_log_monitor_thread = ScyllaOperatorLogMonitoring(self)
        self._scylla_operator_log_monitor_thread.start()

    def start_scylla_cluster_events_thread(self, namespace: str = SCYLLA_NAMESPACE) -> None:
        self._scylla_cluster_events_threads[namespace] = KubectlClusterEventsLogger(
            self, self.scylla_cluster_event_log(namespace=namespace), namespace=namespace)
        self._scylla_cluster_events_threads[namespace].start()

    @log_run_info
    def deploy_scylla_operator(self, pool_name: str = None) -> None:
        if not self.params.get('reuse_cluster'):
            if pool_name is None:
                pool_name = self.AUXILIARY_POOL_NAME

            affinity_rules = {
                "affinity": get_preferred_pod_anti_affinity_values("scylla-operator"),
                "webhookServerAffinity": get_preferred_pod_anti_affinity_values("webhook-server"),
            }
            if pool_name:
                add_pool_node_affinity(affinity_rules["affinity"], self.POOL_LABEL_NAME, pool_name)
                affinity_rules["webhookServerAffinity"]["nodeAffinity"] = (
                    affinity_rules["affinity"]["nodeAffinity"])
            values = HelmValues(**affinity_rules)
            if ComparableScyllaOperatorVersion(
                    self.scylla_operator_chart_version.split("-")[0]) > "1.3.0":
                # NOTE: following is supported starting with operator-1.4
                values.set("logLevel", 4)

            # Calculate options values which must be set
            #
            # image.repository -> self.params.get('k8s_scylla_operator_docker_image').split('/')[0]
            # image.tag        -> self.params.get('k8s_scylla_operator_docker_image').split(':')[-1]

            scylla_operator_repo_base = self.params.get(
                'k8s_scylla_operator_docker_image').rsplit('/', 1)[0]
            if scylla_operator_repo_base:
                values.set('image.repository', scylla_operator_repo_base)

            scylla_operator_image_tag = self.params.get(
                'k8s_scylla_operator_docker_image').split(':', 1)[-1]
            if scylla_operator_image_tag:
                values.set('image.tag', scylla_operator_image_tag)

            # Install and wait for initialization of the Scylla Operator chart
            self.log.info("Deploy Scylla Operator")
            self.create_namespace(SCYLLA_OPERATOR_NAMESPACE)
            self.log.debug(self.helm_install(
                target_chart_name="scylla-operator",
                source_chart_name="scylla-operator/scylla-operator",
                version=self.scylla_operator_chart_version,
                use_devel=True,
                namespace=SCYLLA_OPERATOR_NAMESPACE,
                values=values
            ))
            scylla_operator_version = self.scylla_operator_chart_version.split("-")[0]
            enable_tls = 'true' if self.params.get('k8s_enable_tls') else 'false'
            if ComparableScyllaOperatorVersion(scylla_operator_version) >= "1.8.0":
                patch_cmd = (
                    'patch deployment scylla-operator --type=json -p=\'[{"op": "add",'
                    ' "path": "/spec/template/spec/containers/0/args/-", '
                    f'"value": "--feature-gates=AutomaticTLSCertificates={enable_tls}" }}]\' ')
                self.kubectl(patch_cmd, namespace=SCYLLA_OPERATOR_NAMESPACE)
            if enable_tls == 'true' and ComparableScyllaOperatorVersion(scylla_operator_version) >= "1.9.0":
                # around 10 keys that need to be cached per cluster
                crypto_key_buffer_size = self.params.get('k8s_tenants_num') * 10
                for flag in (f"--crypto-key-buffer-size-min={crypto_key_buffer_size}",
                             f"--crypto-key-buffer-size-max={crypto_key_buffer_size}"):
                    patch_obj = [{
                        "op": "add",
                        "path": "/spec/template/spec/containers/0/args/-",
                        "value": flag
                    }]
                    patch_cmd = f"patch deployment scylla-operator --type=json -p='{json.dumps(patch_obj)}'"
                    self.kubectl(patch_cmd, namespace=SCYLLA_OPERATOR_NAMESPACE)

            KubernetesOps.wait_for_pods_readiness(
                kluster=self,
                total_pods=lambda pods: pods > 0,
                readiness_timeout=5,
                namespace=SCYLLA_OPERATOR_NAMESPACE
            )

        # Start the Scylla Operator logging thread
        self.start_scylla_operator_journal_thread()

    @log_run_info
    def upgrade_scylla_operator(self, new_helm_repo: str,
                                new_chart_version: str,
                                new_docker_image: str = '') -> None:
        self.log.info(
            "Upgrade Scylla Operator using '%s' helm chart and '%s' docker image\n"
            "Helm repo: %s", new_chart_version, new_docker_image, new_helm_repo)

        local_repo_name = "scylla-operator-upgrade"
        self.log.debug(self.helm(f"repo add {local_repo_name} {new_helm_repo}"))
        self.helm('repo update')

        # Calculate new chart name if it is not specific
        if new_chart_version in ("", "latest"):
            new_chart_version = self.get_latest_chart_version(f"{local_repo_name}/scylla-operator")
            self.log.info(
                "Using automatically found following latest scylla-operator "
                "upgrade chart version: %s", new_chart_version)
        else:
            self.log.info(
                "Using following predefined scylla-operator upgrade chart version: %s",
                new_chart_version)

        # Upgrade CRDs if new chart version is newer than v1.5.0
        # Helm doesn't do CRD 'upgrades', only 'creations'.
        # Details:
        #   https://helm.sh/docs/chart_best_practices/custom_resource_definitions/#some-caveats-and-explanations
        if ComparableScyllaOperatorVersion(new_chart_version.split("-")[0]) > "1.5.0":
            self.log.info("Upgrade Scylla Operator CRDs: START")
            try:
                with TemporaryDirectory() as tmpdir:
                    self.helm(
                        f"pull {local_repo_name}/scylla-operator --devel --untar "
                        f"--version {new_chart_version} --destination {tmpdir}")
                    crd_basedir = os.path.join(tmpdir, 'scylla-operator/crds')
                    for current_file in os.listdir(crd_basedir):
                        if not current_file.endswith((".yaml", ".yml")):
                            continue
                        self.apply_file(
                            os.path.join(crd_basedir, current_file), modifiers=[], envsubst=False)
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                self.log.debug("Upgrade Scylla Operator CRDs: Exception: %s", exc)
            self.log.info("Upgrade Scylla Operator CRDs: END")

        # Get existing scylla-operator helm chart values
        values = HelmValues(json.loads(self.helm(
            "get values scylla-operator -o json", namespace=SCYLLA_OPERATOR_NAMESPACE)))
        if ComparableScyllaOperatorVersion(new_chart_version.split("-")[0]) > "1.3.0":
            # NOTE: following is supported starting with operator-1.4
            values.set("logLevel", 4)

        # NOTE: Apply new image repo if provided or set default one redefining base value
        #       example structure: scylladb/scylla-operator:latest
        values.set('image.repository', new_docker_image.split('/')[0].strip() or 'scylladb')

        # NOTE: Set operator_image_tag even if it is empty, we need to redefine base operator image
        values.set('image.tag', new_docker_image.split(':')[-1].strip())

        # Upgrade Scylla Operator using Helm chart
        self.log.debug(self.helm_upgrade(
            target_chart_name="scylla-operator",
            source_chart_name=f"{local_repo_name}/scylla-operator",
            version=new_chart_version,
            use_devel=True,
            namespace=SCYLLA_OPERATOR_NAMESPACE,
            values=values,
        ))
        if self.params.get('k8s_enable_tls') and ComparableScyllaOperatorVersion(
                self.scylla_operator_chart_version.split("-")[0]) >= "1.8.0":
            patch_cmd = ('patch deployment scylla-operator --type=json -p=\'[{"op": "add",'
                         '"path": "/spec/template/spec/containers/0/args/-", '
                         '"value": "--feature-gates=AutomaticTLSCertificates=true" }]\' ')
            self.kubectl(patch_cmd, namespace=SCYLLA_OPERATOR_NAMESPACE)
        time.sleep(5)
        self.kubectl("rollout status deployment scylla-operator",
                     namespace=SCYLLA_OPERATOR_NAMESPACE)

    def check_scylla_cluster_sa_annotations(self, namespace: str = SCYLLA_NAMESPACE):
        # Make sure that ScyllaCluster ServiceAccount annotations stay unchanged
        raw_sa_data = self.kubectl(
            f"get sa {self.params.get('k8s_scylla_cluster_name')}-member -o yaml",
            namespace=namespace).stdout
        sa_data = yaml.safe_load(raw_sa_data)
        error_msg = (
            "ServiceAccount annotations don't have expected values.\n"
            f"Expected: {SCYLLA_CLUSTER_SA_ANNOTATIONS}\n"
            f"Actual: {sa_data['metadata']['annotations']}"
        )
        for annotation_key, annotation_value in SCYLLA_CLUSTER_SA_ANNOTATIONS.items():
            assert sa_data["metadata"]["annotations"].get(annotation_key) == annotation_value, error_msg

    @log_run_info
    def deploy_minio_s3_backend(self):
        if not self.params.get('reuse_cluster'):
            self.log.info('Deploy minio s3-like backend server')
            self.create_namespace(MINIO_NAMESPACE)
            values = HelmValues({})
            values.set('persistence.size', self.params.get("k8s_minio_storage_size"))
            self.log.debug(self.helm_install(
                target_chart_name="minio",
                source_chart_name=LOCAL_MINIO_DIR,
                namespace=MINIO_NAMESPACE,
                values=values,
            ))

        wait_for(lambda: self.minio_ip_address, text='Waiting for minio pod to popup',
                 timeout=120, throw_exc=True)
        self.kubectl_wait("-l app=minio --for=condition=Ready pod",
                          timeout=600, namespace=MINIO_NAMESPACE)

    def get_scylla_cluster_helm_values(self, cpu_limit, memory_limit, pool_name: str = None,
                                       cluster_name=None) -> HelmValues:
        mgmt_agent_cpu_limit = convert_cpu_units_to_k8s_value(SCYLLA_MANAGER_AGENT_RESOURCES['cpu'])
        mgmt_agent_memory_limit = convert_memory_units_to_k8s_value(
            SCYLLA_MANAGER_AGENT_RESOURCES['memory'])
        if not cluster_name:
            cluster_name = self.params.get('k8s_scylla_cluster_name')
        placement = add_pool_node_affinity({}, self.POOL_LABEL_NAME, pool_name) if pool_name else {}
        pod_affinity_term = {
            "topologyKey": "kubernetes.io/hostname",
            "labelSelector": {"matchExpressions": [
                {"key": "scylla/cluster", "operator": "In", "values": [cluster_name]},
                {"key": "app.kubernetes.io/name", "operator": "In", "values": ["scylla"]},
            ]}
        }
        if self.tenants_number < 2:
            placement["podAntiAffinity"] = {"preferredDuringSchedulingIgnoredDuringExecution": [{
                "weight": 100,
                "podAffinityTerm": pod_affinity_term,
            }]}
        else:
            placement["podAntiAffinity"] = {"requiredDuringSchedulingIgnoredDuringExecution": [
                pod_affinity_term,
            ]}
        placement["tolerations"] = [{
            "key": "role",
            "value": "scylla-clusters",
            "operator": "Equal",
            "effect": "NoSchedule",
        }]

        dns_domains = []
        expose_options = {}
        if ComparableScyllaOperatorVersion(self.scylla_operator_chart_version.split("-")[0]) >= "1.11.0":
            if k8s_db_node_service_type := self.params.get("k8s_db_node_service_type"):
                expose_options["nodeService"] = {"type": k8s_db_node_service_type}
            for broadcast_direction_type in ("node", "client"):
                if ip_type := self.params.get(f"k8s_db_node_to_{broadcast_direction_type}_broadcast_ip_type"):
                    if "broadcastOptions" not in expose_options:
                        expose_options["broadcastOptions"] = {}
                    expose_options["broadcastOptions"][f"{broadcast_direction_type}s"] = {"type": ip_type}
        if self.params.get('k8s_enable_sni') and ComparableScyllaOperatorVersion(
                self.scylla_operator_chart_version.split("-")[0]) >= "1.8.0":
            dns_domains = [f"{cluster_name}.sct.scylladb.com"]
            expose_options = {"cql": {"ingress": {
                "annotations": {
                    "haproxy.org/scale-server-slots": "1",
                    "haproxy.org/ssl-passthrough": "true",
                },
                "disabled": False,
                "ingressClassName": "haproxy",
            }}}
        # NOTE: fs.aio-max-nr's value is defined in the scylla repo here:
        #       dist/common/sysctl.d/99-scylla-aio.conf
        #       Scylla 5.0+ has it as '30000000'
        sysctls = ["fs.aio-max-nr=300000000", ]
        if self.params.get('print_kernel_callstack'):
            sysctls += ["kernel.perf_event_paranoid=0", ]

        scylla_args = self.params.get('append_scylla_args')

        return HelmValues({
            'nameOverride': '',
            'fullnameOverride': cluster_name,
            'scyllaImage': {
                'repository': self.params.get('docker_image'),
                'tag': self.params.get('scylla_version')
            },
            'agentImage': {
                'repository': 'scylladb/scylla-manager-agent',
                'tag': self.params.get('scylla_mgmt_agent_version')
            },
            'serviceAccount': {
                'create': True,
                'annotations': SCYLLA_CLUSTER_SA_ANNOTATIONS,
                'name': f"{cluster_name}-member"
            },
            'alternator': {
                'enabled': self.params.get("k8s_enable_alternator") or False,
                'insecureEnableHTTP': True,
                'insecureDisableAuthorization': not (
                    self.params.get("alternator_enforce_authorization") or False),
                **({"writeIsolation": self.params.get("alternator_write_isolation")}
                   if self.params.get("alternator_write_isolation") else {}),
            },
            'developerMode': False,
            'cpuset': True,
            'hostNetworking': False,
            'automaticOrphanedNodeCleanup': True,
            'sysctls': sysctls,
            'serviceMonitor': {
                'create': False
            },
            'datacenter': self.region_name,
            'scyllaArgs': scylla_args,
            'dnsDomains': dns_domains,
            'exposeOptions': expose_options,
            'racks': [
                {
                    'name': self.rack_name,
                    'scyllaConfig': SCYLLA_CONFIG_NAME,
                    'scyllaAgentConfig': SCYLLA_AGENT_CONFIG_NAME,
                    'members': 0,
                    'storage': {
                        'storageClassName': self.params.get('k8s_scylla_disk_class'),
                        'capacity': f"{self.params.get('k8s_scylla_disk_gi')}Gi"
                    },
                    'resources': {
                        'limits': {
                            'cpu': cpu_limit,
                            'memory': memory_limit
                        },
                        'requests': {
                            'cpu': cpu_limit,
                            'memory': memory_limit
                        },
                    },
                    'agentResources': {
                        'limits': {
                            'cpu': mgmt_agent_cpu_limit,
                            'memory': mgmt_agent_memory_limit,
                        },
                        'requests': {
                            'cpu': mgmt_agent_cpu_limit,
                            'memory': mgmt_agent_memory_limit,
                        },
                    },
                    'placement': placement,
                }
            ]
        })

    def wait_till_cluster_is_operational(self):
        if self.api_call_rate_limiter:
            with self.api_call_rate_limiter.pause:
                self.api_call_rate_limiter.wait_till_api_become_not_operational(self)
                self.api_call_rate_limiter.wait_till_api_become_stable(self)
        self.wait_all_node_pools_to_be_ready()

    def create_scylla_manager_agent_config(self, s3_provider_endpoint: str = None, namespace: str = SCYLLA_NAMESPACE):
        data = {}
        if self.params.get('use_mgmt'):
            data["s3"] = {
                'provider': 'Minio',
                'endpoint': s3_provider_endpoint or self.s3_provider_endpoint,
                'access_key_id': 'minio_access_key',
                'secret_access_key': 'minio_secret_key',
            }

        # Create kubernetes secret that holds scylla manager agent configuration
        self.update_secret_from_data(SCYLLA_AGENT_CONFIG_NAME, namespace, {
            'scylla-manager-agent.yaml': data,
        })

    @cached_property
    def is_performance_tuning_enabled(self):
        return self.params.get('k8s_enable_performance_tuning') and self.IS_NODE_TUNING_SUPPORTED

    def install_static_local_volume_provisioner(
            self, node_pools: list[CloudK8sNodePool] | CloudK8sNodePool) -> None:
        if self.params.get('reuse_cluster'):
            return
        if not isinstance(node_pools, list):
            node_pools = [node_pools]

        self.log.info("Install static local volume provisioner")
        self.apply_file(
            LOCAL_PROVISIONER_FILE,
            modifiers=[affinity_modifier
                       for current_pool in node_pools
                       for affinity_modifier in current_pool.affinity_modifiers],
            envsubst=False)

    def install_dynamic_local_volume_provisioner(
            self, node_pools: list[CloudK8sNodePool] | CloudK8sNodePool) -> None:
        if self.params.get('reuse_cluster'):
            return
        if not isinstance(node_pools, list):
            node_pools = [node_pools]

        self.log.info("Install dynamic local volume provisioner")
        config_modifiers = [affinity_modifier
                            for current_pool in node_pools
                            for affinity_modifier in current_pool.affinity_modifiers]

        def image_modifier(obj):
            if obj["kind"] != "DaemonSet":
                return
            for container_data in obj["spec"]["template"]["spec"]["containers"]:
                if "scylladb/k8s-local-volume-provisioner" in container_data["image"]:
                    container_data["image"] = (
                        f"{container_data['image'].split(':')[0]}:{K8S_LOCAL_VOLUME_PROVISIONER_VERSION}")

        def example_disk_modifier(obj):
            if obj["kind"] != "DaemonSet":
                return
            # NOTE: add a bit more storage to cover XFS filesystem overhead.
            disk_size_kb = int(self.params.get("k8s_scylla_disk_gi") * 1.012 * 1024**2)
            for container_data in obj["spec"]["template"]["spec"]["containers"]:
                if "disk-setup" not in container_data["name"]:
                    continue
                for i, cmd in enumerate(container_data["command"]):
                    if 'seek=' not in cmd:
                        continue
                    # TODO: stop using this modifier when the code in the source repo allows
                    #       to set custom disk size.
                    # NOTE: set the disk size that satisfies our configured values for storage
                    container_data["command"][i] = re.sub(
                        r"seek=\d+", f"seek={disk_size_kb}", container_data["command"][i])

        with TemporaryDirectory() as tmp_dir_name:
            repo_dst_dir = os.path.join(tmp_dir_name, 'dynamic-local-volume-provisioner')

            download_from_github(
                repo='scylladb/k8s-local-volume-provisioner',
                tag=f'tags/v{K8S_LOCAL_VOLUME_PROVISIONER_VERSION}',
                dst_dir=repo_dst_dir)

            self.apply_file(sct_abs_path(f"{repo_dst_dir}/example/storageclass_xfs.yaml"))

            # NOTE: apply example disk setup formatted to the XFS only on local K8S setups
            if "k8s-local" in self.params.get("cluster_backend"):
                path_to_disk_setup_config = sct_abs_path(f"{repo_dst_dir}/example/disk-setup")
                self.apply_file(
                    path_to_disk_setup_config,
                    modifiers=config_modifiers + [example_disk_modifier] + [image_modifier],
                    envsubst=False)
                self.kubectl("rollout status daemonset.apps/xfs-disk-setup",
                             namespace="xfs-disk-setup")

            path_to_csi_driver_config = sct_abs_path(f"{repo_dst_dir}/deploy/kubernetes")
            self.apply_file(
                path_to_csi_driver_config,
                modifiers=config_modifiers + [image_modifier],
                envsubst=False)
            self.kubectl("rollout status daemonset.apps/local-csi-driver",
                         namespace="local-csi-driver")

    @log_run_info
    def prepare_k8s_scylla_nodes(
            self, node_pools: list[CloudK8sNodePool] | CloudK8sNodePool) -> None:
        if not self.NODE_PREPARE_FILE or self.params.get('reuse_cluster'):
            return
        if not isinstance(node_pools, list):
            node_pools = [node_pools]

        def node_config_change_mount_point(obj):
            if obj["kind"] != "NodeConfig":
                return
            # NOTE: this modifier must run only for 'static' local volume provisioner
            obj["spec"]['localDiskSetup']['mounts'][0]['mountPoint'] = '/mnt/raid-disks/disk0'

        def node_setup_for_dynamic_local_volume_provisioner_modifier(obj):
            if obj["kind"] != "DaemonSet":
                return
            for container_data in obj["spec"]["template"]["spec"]["containers"]:
                if container_data["name"] in ("pv-setup", "node-setup"):
                    # NOTE: disable custom node- and pv- setups using dynamic volume provisioner
                    container_data["command"] = ["/bin/bash", "-c", "--"]
                    container_data["args"] = ["while true; do sleep 3600; done"]

        modifiers = [affinity_modifier
                     for current_pool in node_pools
                     for affinity_modifier in current_pool.affinity_modifiers]
        if self.params.get("k8s_local_volume_provisioner_type") != 'static':
            modifiers.append(node_setup_for_dynamic_local_volume_provisioner_modifier)
        self.log.info("Install DaemonSets required by scylla nodes")
        self.apply_file(self.NODE_PREPARE_FILE, modifiers=modifiers, envsubst=False)

        # Tune performance of the Scylla nodes
        if self.is_performance_tuning_enabled:
            self.log.info("Tune K8S nodes dedicated for Scylla")
            self.kubectl_wait(
                "--for condition=established crd/scyllaoperatorconfigs.scylla.scylladb.com")
            self.kubectl_wait("--for condition=established crd/nodeconfigs.scylla.scylladb.com")

            if scylla_utils_docker_image := self.params.get('k8s_scylla_utils_docker_image'):
                self.kubectl(
                    "patch scyllaoperatorconfigs.scylla.scylladb.com cluster --type merge "
                    "-p '{\"spec\":{\"scyllaUtilsImage\":\"%s\"}}'" % scylla_utils_docker_image,
                    ignore_status=False)

            modifiers = [affinity_modifier
                         for current_pool in node_pools
                         for affinity_modifier in current_pool.affinity_modifiers]
        if self.params.get("k8s_local_volume_provisioner_type") == 'static':
            modifiers.append(node_config_change_mount_point)
        self.apply_file(self.NODE_CONFIG_CRD_FILE, modifiers=modifiers, envsubst=False)
        time.sleep(30)

        if self.params.get("k8s_local_volume_provisioner_type") == 'static':
            self.install_static_local_volume_provisioner(node_pools=node_pools)
        else:
            self.install_dynamic_local_volume_provisioner(node_pools=node_pools)

    @log_run_info
    def deploy_ingress_controller(self, pool_name: str = None):
        self.log.info("Create and initialize ingress controller")
        if not self.params.get('reuse_cluster'):
            pool_name = pool_name or self.AUXILIARY_POOL_NAME
            comma_separated_tags = ", ".join([f'{k}={v}' for k, v in self.tags.items()])

            if self.params.get("cluster_backend") == "k8s-eks":
                cpu_limit = 2
                memory_limit = 4
                cpu_request = 1
                memory_request = 2

                cpu_limit = convert_cpu_units_to_k8s_value(cpu_limit)
                memory_limit = convert_memory_units_to_k8s_value(memory_limit)
                cpu_request = convert_cpu_units_to_k8s_value(cpu_request)
                memory_request = convert_memory_units_to_k8s_value(memory_request)
                haproxy_maxconn = 70000
            else:
                # TODO: measure these values in performance testing and update if needed.
                cpu_limit = '400m'
                memory_limit = '200M'
                cpu_request = '100m'
                memory_request = '50M'
                # that's the default, and trigger haproxy auto calculation of it.
                haproxy_maxconn = 0

            environ = {
                "POD_CPU_LIMIT": cpu_limit,
                "POD_MEMORY_LIMIT": memory_limit,
                "POD_CPU_REQUEST": cpu_request,
                "POD_MEMORY_REQUEST": memory_request,
                "HAPROXY_MAXCONN": haproxy_maxconn,
                "TAGS": comma_separated_tags,
            }
            self.apply_file(
                INGRESS_CONTROLLER_CONFIG_PATH,
                modifiers=get_pool_affinity_modifiers(self.POOL_LABEL_NAME, pool_name),
                environ=environ)
        self.kubectl_wait("--all --for=condition=Ready pod",
                          namespace=INGRESS_CONTROLLER_NAMESPACE, timeout=306)
        self.start_haproxy_ingress_log_thread()

    @log_run_info
    def deploy_scylla_cluster(self, node_pool_name: str, namespace: str = SCYLLA_NAMESPACE,
                              cluster_name: str = None, s3_provider_endpoint: str = None) -> None:
        # Calculate cpu and memory limits to occupy all available amounts by scylla pods
        node_pool = self.pools[node_pool_name]
        cpu_limit, memory_limit = node_pool.cpu_and_memory_capacity
        if not self.params.get('k8s_scylla_cpu_limit'):
            # TODO: Remove reduction logic after
            #       https://github.com/scylladb/scylla-operator/issues/384 is fixed
            cpu_limit = int(
                cpu_limit
                - OPERATOR_CONTAINERS_RESOURCES['cpu'] * self.tenants_number
                - COMMON_CONTAINERS_RESOURCES['cpu']
                - SCYLLA_MANAGER_AGENT_RESOURCES['cpu'] * self.tenants_number
            )
            # NOTE: we should use at max 7 from each 8 cores.
            #       i.e 28/32 , 21/24 , 14/16 and 7/8
            new_cpu_limit = math.ceil(cpu_limit / 8) * 7
            if new_cpu_limit < cpu_limit:
                cpu_limit = new_cpu_limit
            cpu_limit = cpu_limit // self.tenants_number or 1
            self.scylla_cpu_limit = convert_cpu_units_to_k8s_value(cpu_limit)
        else:
            self.scylla_cpu_limit = self.params.get('k8s_scylla_cpu_limit')
        if not self.params.get('k8s_scylla_memory_limit'):
            memory_limit = (
                memory_limit
                - OPERATOR_CONTAINERS_RESOURCES['memory'] * self.tenants_number
                - COMMON_CONTAINERS_RESOURCES['memory']
                - SCYLLA_MANAGER_AGENT_RESOURCES['memory'] * self.tenants_number
            )
            memory_limit = memory_limit / self.tenants_number
            self.scylla_memory_limit = convert_memory_units_to_k8s_value(memory_limit)
        else:
            self.scylla_memory_limit = self.params.get('k8s_scylla_memory_limit')

        if self.params.get('reuse_cluster'):
            try:
                self.wait_till_cluster_is_operational()
                self.log.debug("Check Scylla cluster")
                self.kubectl("get scyllaclusters.scylla.scylladb.com", namespace=namespace)
                self.start_scylla_cluster_events_thread()
                return
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "SCT_REUSE_CLUSTER is set, but target scylla cluster is unhealthy") from exc

        self.log.info("Create and initialize a Scylla cluster")
        self.create_scylla_manager_agent_config(s3_provider_endpoint=s3_provider_endpoint, namespace=namespace)

        # Init 'scylla-config' configMap before installation of Scylla to avoid redundant restart
        self.init_scylla_config_map(namespace=namespace)
        self.scylla_restart_required = False

        # Deploy scylla cluster
        self.log.debug(self.helm_install(
            target_chart_name="scylla",
            source_chart_name="scylla-operator/scylla",
            version=self.scylla_operator_chart_version,
            use_devel=True,
            values=self.get_scylla_cluster_helm_values(
                cpu_limit=self.scylla_cpu_limit,
                memory_limit=self.scylla_memory_limit,
                pool_name=node_pool_name,
                cluster_name=cluster_name),
            namespace=namespace,
        ))
        self.wait_till_cluster_is_operational()

        self.log.debug("Check Scylla cluster")
        self.kubectl("get scyllaclusters.scylla.scylladb.com", namespace=namespace)
        self.log.debug(
            "Wait for %d secs before we start to apply changes to the cluster",
            DEPLOY_SCYLLA_CLUSTER_DELAY)
        self.start_scylla_cluster_events_thread(namespace=namespace)

    @cached_property
    def _affinity_modifiers_for_monitoring_resources(self):
        node_pool = self.pools.get(self.MONITORING_POOL_NAME)
        if not node_pool:
            self.log.warning(
                "'Monitoring' node pool was not found."
                " Will schedule 'monitoring' resources creation in the 'Auxiliary' one.")
            node_pool = self.pools.get(self.AUXILIARY_POOL_NAME)
        affinity_modifiers = node_pool.affinity_modifiers if node_pool else None
        if not affinity_modifiers:
            self.log.warning(
                "'Auxiliary' node pool was not found."
                " Will schedule 'monitoring' resources creation without affinity rules.")
        return affinity_modifiers

    @log_run_info
    def deploy_prometheus_operator(self) -> None:
        self.log.info("Deploy Prometheus operator")
        if not self.params.get('reuse_cluster'):
            # NOTE: apply configs on the 'server' side to avoid following error:
            #         The CustomResourceDefinition "prometheuses.monitoring.coreos.com" is invalid:\
            #           metadata.annotations: Too long: must have at most 262144 bytes
            self.apply_file(PROMETHEUS_OPERATOR_CONFIG_PATH,
                            namespace=PROMETHEUS_OPERATOR_NAMESPACE,
                            modifiers=self._affinity_modifiers_for_monitoring_resources,
                            envsubst=False, server_side=True)
            time.sleep(3)
        self.kubectl("rollout status deployment prometheus-operator", namespace=PROMETHEUS_OPERATOR_NAMESPACE)

    def deploy_scylla_cluster_monitoring(self, cluster_name: str, namespace: str,
                                         monitoring_type: str = "Platform") -> None:
        self.log.info(
            "Deploy 'ScyllaDBMonitoring' (type: %s) for the '%s' Scylla cluster in the '%s' namespace",
            monitoring_type, cluster_name, namespace)
        self.apply_file(
            SCYLLA_MONITORING_CONFIG_PATH,
            modifiers=self._affinity_modifiers_for_monitoring_resources,
            environ={
                "SCT_SCYLLA_CLUSTER_NAME": cluster_name,
                "SCT_SCYLLA_CLUSTER_NAMESPACE": namespace,
                "SCT_SCYLLA_CLUSTER_MONITORING_TYPE": monitoring_type,
            })
        for condition in ("Progressing=False", "Degraded=False", "Available=True"):
            self.kubectl_wait(f"--for='condition={condition}' scylladbmonitoring {cluster_name}",
                              namespace=namespace, timeout=600)
        self.kubectl(f"rollout status sts prometheus-{cluster_name}", namespace=namespace)
        self.kubectl(f"rollout status deployment {cluster_name}-grafana", namespace=namespace)

    def delete_scylla_cluster_monitoring(self, namespace: str) -> None:
        self.kubectl("delete --all --wait=true ScyllaDBMonitoring", namespace=namespace, ignore_status=True)
        time.sleep(1)

    def get_grafana_ip(self, cluster_name: str, namespace: str) -> str:
        if self.cluster_backend in ("k8s-eks", "k8s-gke"):
            cmd = (f"get pod -l scylla-operator.scylladb.com/deployment-name={cluster_name}-grafana"
                   " --no-headers -o custom-columns=:.status.podIP")
        else:
            cmd = f"get svc {cluster_name}-grafana --no-headers -o custom-columns=:.spec.clusterIP"
        return self.kubectl(cmd, namespace=namespace).stdout.strip()

    @property
    def grafana_port(self) -> int:
        return 3000

    def get_prometheus_ip(self, cluster_name: str, namespace: str) -> str:
        if self.cluster_backend in ("k8s-eks", "k8s-gke"):
            cmd = f"get pod -l prometheus={cluster_name} --no-headers -o custom-columns=:.status.podIP"
        else:
            cmd = f"get svc {cluster_name}-prometheus --no-headers -o custom-columns=:.spec.clusterIP"
        return self.kubectl(cmd, namespace=namespace).stdout.strip()

    @property
    def prometheus_port(self) -> int:
        return 9090

    def register_sct_grafana_dashboard(self, cluster_name: str, namespace: str) -> str:  # pylint: disable=too-many-locals
        # TODO: make it work for EKS by using ingress LB IP when it is enabled
        sct_dashboard_file = sct_abs_path("data_dir/scylla-dash-per-server-nemesis.master.json")
        sct_dashboard_file_data_str = ""
        with open(sct_dashboard_file, encoding="utf-8") as sct_dashboard_file_obj:
            dashboard_config = yaml.safe_load(sct_dashboard_file_obj)
            dashboard_config["dashboard"]["title"] = dashboard_config["dashboard"]["title"].replace(
                "$test_name", f"{get_test_name()}--{cluster_name}")
            sct_dashboard_file_data_str = json.dumps(dashboard_config)
        grafana_dn = f"{cluster_name}-grafana.{namespace}.svc.cluster.local"
        grafana_ip = self.get_grafana_ip(cluster_name=cluster_name, namespace=namespace)
        grafana_user = base64.b64decode(self.kubectl(
            f"get secret/{cluster_name}-grafana-admin-credentials --template='{{{{ index .data \"username\" }}}}'",
            namespace=namespace).stdout.strip()).decode('utf-8')
        grafana_password = base64.b64decode(self.kubectl(
            f"get secret/{cluster_name}-grafana-admin-credentials --template='{{{{ index .data \"password\" }}}}'",
            namespace=namespace).stdout.strip()).decode('utf-8')
        grafana_cert = base64.b64decode(self.kubectl(
            f"get secret/{cluster_name}-grafana-serving-ca --template='{{{{ index .data \"tls.crt\" }}}}'",
            namespace=namespace).stdout.strip()).decode('utf-8')
        with NamedTemporaryFile(mode='w') as grafana_cert_obj, NamedTemporaryFile(mode='w') as sct_dashboard_obj:
            grafana_cert_obj.write(grafana_cert)
            grafana_cert_obj.flush()
            sct_dashboard_obj.write(sct_dashboard_file_data_str)
            sct_dashboard_obj.flush()
            upload_result = LOCALRUNNER.run(
                f"curl --fail -o /dev/null -w '%{{http_code}}'"
                f" -L 'https://{grafana_dn}:{self.grafana_port}/api/dashboards/db'"
                f" --resolve '{grafana_dn}:{self.grafana_port}:{grafana_ip}'"
                f" --cacert {grafana_cert_obj.name}"
                f" --user '{grafana_user}:{grafana_password}'"
                f" -d @{sct_dashboard_obj.name} -H 'Content-Type: application/json'"
            ).stdout.strip()
        if upload_result != "200":
            self.log.warning(
                "Error uploading SCT dashboard '%s' to the grafana in the '%s' namespace: %s",
                sct_dashboard_file, namespace, upload_result)
        else:
            self.log.info(
                "SCT dashboard '%s' uploaded successfully to the grafana in the '%s' namespace",
                sct_dashboard_file, namespace)
        return upload_result

    @log_run_info
    def deploy_loaders_cluster(self,
                               node_pool_name: str = '',
                               namespace: str = LOADER_NAMESPACE) -> None:
        self.log.info("Create and initialize a loaders cluster in the '%s' namespace", namespace)
        node_pool = self.pools.get(node_pool_name)
        if node_pool:
            self.deploy_node_pool(node_pool)
            cpu_limit, memory_limit = node_pool.cpu_and_memory_capacity
            cpu_limit, memory_limit = cpu_limit - 1, memory_limit - 1
            affinity_modifiers = node_pool.affinity_modifiers
        else:
            cpu_limit = 2
            memory_limit = 4
            affinity_modifiers = []

        self.calculated_loader_cpu_limit = cpu_limit = convert_cpu_units_to_k8s_value(cpu_limit)
        self.calculated_loader_memory_limit = memory_limit = convert_memory_units_to_k8s_value(
            memory_limit)
        self.calculated_loader_affinity_modifiers = affinity_modifiers

    @log_run_info
    def gather_k8s_logs(self) -> None:
        return KubernetesOps.gather_k8s_logs(logdir_path=self.logdir, kubectl=self.kubectl)

    @log_run_info
    def gather_k8s_logs_by_operator(self) -> None:
        return KubernetesOps.gather_k8s_logs_by_operator(kluster=self)

    @property
    def minio_pod(self) -> Resource:
        for pod in KubernetesOps.list_pods(self, namespace=MINIO_NAMESPACE):
            for container in pod.spec.containers:
                for port in container.ports:
                    if hasattr(port, 'container_port') and str(port.container_port) == '9000':
                        return pod
        raise RuntimeError("Can't find minio pod")

    @property
    def minio_ip_address(self) -> str:
        return self.minio_pod.status.pod_ip

    @property
    def s3_provider_endpoint(self) -> str:
        return f"http://{self.minio_ip_address}:9000"

    @cached_property
    def tags(self) -> Dict[str, str]:
        return get_tags_from_params(self.params)

    @cached_property
    def logdir(self) -> str:
        assert '_SCT_TEST_LOGDIR' in os.environ
        logdir = os.path.join(os.environ['_SCT_TEST_LOGDIR'], self.name)
        os.makedirs(logdir, exist_ok=True)
        return logdir

    @property
    def k8s_core_v1_api(self) -> k8s.client.CoreV1Api:
        return KubernetesOps.core_v1_api(self.api_client)

    @property
    def k8s_apps_v1_api(self) -> k8s.client.AppsV1Api:
        return KubernetesOps.apps_v1_api(self.api_client)

    @property
    def k8s_configuration(self) -> k8s.client.Configuration:
        if self.api_call_rate_limiter:
            return self.api_call_rate_limiter.get_k8s_configuration(self)
        return KubernetesOps.create_k8s_configuration(self)

    @property
    def api_client(self) -> k8s.client.ApiClient:
        return self.get_api_client()

    def get_api_client(self) -> k8s.client.ApiClient:
        if self.api_call_rate_limiter:
            return self.api_call_rate_limiter.get_api_client(self.k8s_configuration)
        return KubernetesOps.api_client(self.k8s_configuration)

    @property
    def dynamic_client(self) -> k8s.dynamic.DynamicClient:
        return KubernetesOps.dynamic_client(self.api_client)

    @property
    def scylla_manager_cluster(self) -> 'ManagerPodCluser':
        return ManagerPodCluser(
            k8s_clusters=[self],
            namespace=SCYLLA_MANAGER_NAMESPACE,
            container='scylla-manager',
            cluster_prefix='mgr-',
            node_prefix='mgr-node-',
            params=init_and_verify_sct_config(),
            n_nodes=1
        )

    def create_secret_from_data(self, secret_name: str, namespace: str, data: dict, secret_type: str = 'Opaque'):
        prepared_data = {key: base64.b64encode(json.dumps(value).encode('utf-8')).decode('utf-8')
                         for key, value in data.items()}
        secret = k8s.client.V1Secret(
            api_version="v1",
            data=prepared_data,
            kind="Secret",
            metadata={
                "name": secret_name,
                "namespace": namespace,
            },
            type=secret_type,
        )
        self.k8s_core_v1_api.create_namespaced_secret(namespace=namespace, body=secret)

    def update_secret_from_data(self, secret_name: str, namespace: str, data: dict, secret_type: str = 'Opaque'):
        existing = None
        for secret in self.k8s_core_v1_api.list_namespaced_secret(namespace).items:
            if secret.metadata.name == secret_name:
                existing = secret
                break
        if not existing:
            self.create_secret_from_data(
                secret_name=secret_name, namespace=namespace, data=data, secret_type=secret_type)
            return

        for key, value in data.items():
            existing.data[key] = base64.b64encode(json.dumps(value).encode('utf-8')).decode('utf-8')
        self.k8s_core_v1_api.patch_namespaced_secret(secret_name, namespace, existing)

    def create_secret_from_directory(self, secret_name: str, path: str, namespace: str, secret_type: str = 'generic',
                                     only_files: List[str] = None):
        files = [fname for fname in os.listdir(path) if os.path.isfile(os.path.join(path, fname)) and
                 (not only_files or fname in only_files)]
        cmd = f'create secret {secret_type} {secret_name} ' + \
              ' '.join([f'--from-file={fname}={os.path.join(path, fname)}' for fname in files])
        self.kubectl(cmd, namespace=namespace)

    def patch_kubectl_config(self):
        """
        Patched kubectl config so that it will obtain cloud token from cache file
         that is kept update by token update thread
        """
        self.create_kubectl_config()
        if self.TOKEN_UPDATE_NEEDED:
            self.start_token_update_thread()
            KubernetesOps.patch_kube_config(self, self.kubectl_token_path)
            wait_for(self.check_if_token_is_valid, timeout=120, throw_exc=True)
        self.start_scylla_pods_ip_change_tracker_thread()

    def check_if_token_is_valid(self) -> bool:
        with open(self.kubectl_token_path, mode='rb') as token_file:
            return bool(json.load(token_file))

    def start_token_update_thread(self):
        if os.path.exists(self.kubectl_token_path):
            os.unlink(self.kubectl_token_path)
        self._token_update_thread = self.create_token_update_thread()
        self._token_update_thread.start()
        # Wait till GcloudTokenUpdateThread get tokens and dump them to gcloud_token_path
        wait_for(os.path.exists, timeout=30, step=5, text="Wait for gcloud token", throw_exc=True,
                 path=self.kubectl_token_path)

    def start_scylla_pods_ip_change_tracker_thread(self):
        self.scylla_pods_ip_change_tracker_thread = ScyllaPodsIPChangeTrackerThread(
            self, self.scylla_pods_ip_mapping)
        self.scylla_pods_ip_change_tracker_thread.start()

    def _add_pool(self, pool: CloudK8sNodePool) -> None:
        if pool.name not in self.pools:
            self.pools[pool.name] = pool

    def wait_all_node_pools_to_be_ready(self):
        for node_pool in self.pools.values():
            node_pool.wait_for_nodes_readiness()

    @abc.abstractmethod
    def deploy(self):
        pass

    @abc.abstractmethod
    def create_kubectl_config(self):
        pass

    @abc.abstractmethod
    def create_token_update_thread(self) -> TokenUpdateThread:
        pass

    @abc.abstractmethod
    def deploy_node_pool(self, pool: CloudK8sNodePool, wait_till_ready=True) -> None:
        pass

    def upgrade_kubernetes_platform(self, pod_objects: list[cluster.BaseNode],
                                    use_additional_scylla_nodepool: bool) -> (str, CloudK8sNodePool):
        raise NotImplementedError("Kubernetes upgrade is not implemented on this backend")

    def move_pods_to_new_node_pool(self, pod_objects: list[cluster.BaseNode],
                                   node_pool_name: str, cluster_name: str = "",
                                   cluster_namespace: str = SCYLLA_NAMESPACE,
                                   pod_readiness_timeout_minutes: int = 20):
        cluster_name = cluster_name or self.params.get('k8s_scylla_cluster_name')

        # Update the node affinity rules to match the new nodes
        scylla_cluster_info = yaml.safe_load(self.kubectl(
            f"get scyllaclusters.scylla.scylladb.com {cluster_name} -o yaml",
            namespace=cluster_namespace).stdout)
        racks_info = scylla_cluster_info["spec"]["datacenter"]["racks"]
        total_pods = 0
        for i, rack in enumerate(racks_info):
            rack["placement"]["nodeAffinity"][
                "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0][
                    "matchExpressions"][0]["values"] = [node_pool_name]
            racks_info[i] = rack
            total_pods += int(rack["members"])
        data = {"spec": {"datacenter": {"racks": racks_info}}}
        self.kubectl(
            f"patch scyllaclusters.scylla.scylladb.com {cluster_name} --type merge "
            f"-p '{json.dumps(data)}'",
            namespace=cluster_namespace,
        )

        # Label Scylla pods for replacement to make it be moved to new nodes
        for pod_object in sorted(pod_objects, key=lambda x: x.name, reverse=True):
            old_uid = pod_object.k8s_pod_uid
            pod_object.wait_for_svc()
            pod_object.mark_to_be_replaced()
            pod_object.wait_till_k8s_pod_get_uid(ignore_uid=old_uid)
            self.log.info("Wait for the '%s' pod to be ready", pod_object.name)
            pod_object.wait_for_pod_readiness(
                pod_readiness_timeout_minutes=pod_readiness_timeout_minutes)

    @contextlib.contextmanager
    def scylla_config_map(self, namespace: str = SCYLLA_NAMESPACE) -> dict:
        with self.scylla_config_lock:
            try:
                config_map = self.k8s_core_v1_api.read_namespaced_config_map(
                    name=SCYLLA_CONFIG_NAME, namespace=namespace).data or {}
                exists = True
            except Exception:  # pylint: disable=broad-except  # noqa: BLE001
                config_map = {}
                exists = False
            original_config_map = deepcopy(config_map)
            yield config_map
            if original_config_map == config_map:
                self.log.debug("%s: scylla config map hasn't been changed", self)
                return
            if exists:
                self.k8s_core_v1_api.patch_namespaced_config_map(
                    name=SCYLLA_CONFIG_NAME,
                    namespace=namespace,
                    body=[{"op": "replace", "path": '/data', "value": config_map}]
                )
            else:
                self.k8s_core_v1_api.create_namespaced_config_map(
                    namespace=namespace,
                    body=V1ConfigMap(
                        data=config_map,
                        metadata={'name': SCYLLA_CONFIG_NAME}
                    )
                )

    @contextlib.contextmanager
    def manage_file_in_scylla_config_map(self, filename: str, namespace: str = SCYLLA_NAMESPACE) -> ContextManager:
        """Update scylla.yaml or cassandra-rackdc.properties, k8s way

        Scylla Operator handles file updates using ConfigMap resource
        and we don't need to update it manually on each node.
        Only scylla rollout restart is needed to get it applied.

        Details: https://operator.docs.scylladb.com/master/generic#configure-scylla
        """
        with self.scylla_config_map(namespace=namespace) as scylla_config_map:
            old_data = yaml.safe_load(scylla_config_map.get(filename, "")) or {}
            new_data = deepcopy(old_data)
            yield new_data
            if old_data == new_data:
                self.log.debug("%s: '%s' hasn't been changed", self, filename)
                return
            old_data_as_list = yaml.safe_dump(old_data).splitlines(keepends=True)
            new_data_as_str = yaml.safe_dump(new_data)
            new_data_as_list = new_data_as_str.splitlines(keepends=True)
            diff = "".join(unified_diff(old_data_as_list, new_data_as_list))
            self.log.debug("%s: '%s' has been updated:\n%s", self, filename, diff)
            if not new_data:
                scylla_config_map.pop(filename, None)
            else:
                scylla_config_map[filename] = new_data_as_str
            self.scylla_restart_required = True

    def remote_scylla_yaml(self, namespace: str = SCYLLA_NAMESPACE) -> ContextManager:
        return self.manage_file_in_scylla_config_map(
            filename='scylla.yaml', namespace=namespace)

    def remote_cassandra_rackdc_properties(self, namespace: str = SCYLLA_NAMESPACE) -> ContextManager:
        return self.manage_file_in_scylla_config_map(
            filename='cassandra-rackdc.properties', namespace=namespace)

    def init_scylla_config_map(self, namespace: str = SCYLLA_NAMESPACE, **kwargs):
        # NOTE: operator sets all the scylla options itself based on the configuration of the ScyllaCluster CRD.
        #       It allows to redefine options using 'scylla-config' configMap opject.
        #       So, define some options here by default. It may be extended later if needed.
        with self.remote_scylla_yaml(namespace=namespace) as scylla_yml:
            # Process cluster params
            if self.params.get("experimental_features"):
                scylla_yml["experimental_features"] = self.params.get("experimental_features")
            if self.params.get("hinted_handoff"):
                scylla_yml["hinted_handoff_enabled"] = self.params.get(
                    "hinted_handoff").lower() in ("enabled", "true")
            if self.params.get("endpoint_snitch"):
                scylla_yml["endpoint_snitch"] = self.params.get("endpoint_snitch")

            # Process method kwargs
            if kwargs.get("murmur3_partitioner_ignore_msb_bits"):
                scylla_yml["murmur3_partitioner_ignore_msb_bits"] = int(
                    kwargs.pop("murmur3_partitioner_ignore_msb_bits"))

            self.log.info("K8S SCYLLA_YAML: %s", scylla_yml)
            if kwargs:
                self.log.warning("K8S SCYLLA_YAML, not applied options: %s", kwargs)


class BasePodContainer(cluster.BaseNode):  # pylint: disable=too-many-public-methods
    parent_cluster: PodCluster

    pod_readiness_delay = 30  # seconds
    pod_readiness_timeout = 10  # minutes
    pod_terminate_timeout = 5  # minutes

    def __init__(self, name: str, parent_cluster: PodCluster, node_prefix: str = "node", node_index: int = 1,
                 base_logdir: Optional[str] = None, dc_idx: int = 0, rack=0, shard_num=None):
        self.node_index = node_index
        cluster.BaseNode.__init__(
            self, name=name,
            parent_cluster=parent_cluster,
            base_logdir=base_logdir,
            node_prefix=node_prefix,
            dc_idx=dc_idx,
            rack=rack,
            shard_num=shard_num)
        self.k8s_cluster = self.parent_cluster.k8s_clusters[self.dc_idx]
        self._rack_name = None

    @property
    def node_rack(self) -> str:
        if not self._rack_name:
            if pod := self._pod:
                self._rack_name = pod.metadata.labels.get("scylla/rack", "")

        return self._rack_name

    @cached_property
    def pod_replace_timeout(self) -> int:
        return self.pod_terminate_timeout + self.pod_readiness_timeout

    def configure_remote_logging(self):
        self.k8s_cluster.log.debug("No need to configure remote logging on k8s")

    def wait_for_cloud_init(self):
        pass

    @staticmethod
    def is_docker() -> bool:
        return True

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**super().tags,
                "NodeIndex": str(self.node_index), }

    def _init_remoter(self, ssh_login_info):
        self.remoter = KubernetesCmdRunner(
            kluster=self.k8s_cluster,
            pod_image=self.image,
            pod_name=self.name,
            container=self.parent_cluster.container,
            namespace=self.parent_cluster.namespace)

    def _init_port_mapping(self):
        pass

    @property
    def system_log(self):
        return os.path.join(self.logdir, "system.log")

    @property
    def vm_region(self):
        return self.k8s_cluster.region_name

    @property
    def region(self):
        if self.parent_cluster.params.get('simulated_regions'):
            raise ValueError("K8S backends don't support 'simulated_regions' SCT config")
        return self.vm_region

    def start_journal_thread(self):
        self._journal_thread = get_system_logging_thread(logs_transport="k8s_client",
                                                         node=self,
                                                         target_log_file=self.system_log)
        if self._journal_thread:
            self.k8s_cluster.log.info(
                "Use %s as logging daemon", type(self._journal_thread).__name__)
            self._journal_thread.start()
        else:
            TestFrameworkEvent(source=self.__class__.__name__,
                               source_method="start_journal_thread",
                               message="Got no logging daemon by unknown reason").publish()

    def check_spot_termination(self):
        pass

    @property
    def _pod(self):
        pods = KubernetesOps.list_pods(
            self.k8s_cluster, namespace=self.parent_cluster.namespace,
            field_selector=f"metadata.name={self.name}")
        return pods[0] if pods else None

    @property
    def pod_spec(self):
        if pod := self._pod:
            return pod.spec
        return {}

    @property
    def pod_status(self):
        if pod := self._pod:
            return pod.status
        return None

    @property
    def _node(self):
        return KubernetesOps.get_node(self.k8s_cluster, self.node_name)

    @property
    def _cluster_ip_service(self):
        services = KubernetesOps.list_services(
            self.k8s_cluster, namespace=self.parent_cluster.namespace,
            field_selector=f"metadata.name={self.name}")
        return services[0] if services else None

    @property
    def _svc(self):
        services = KubernetesOps.list_services(
            self.k8s_cluster, namespace=self.parent_cluster.namespace,
            field_selector=f"metadata.name={self.name}")
        return services[0] if services else None

    @property
    def _container_status(self):
        pod_status = self.pod_status
        if pod_status:
            return next((x for x in pod_status.container_statuses if x.name == self.parent_cluster.container), None)
        return None

    @property
    def cql_address(self):
        return self.ip_address

    @property
    def private_ip_address(self) -> Optional[str]:
        if ip := self.k8s_cluster.scylla_pods_ip_mapping.get(
                self.parent_cluster.namespace, {}).get(self.name, {}).get('current_ip'):
            return ip
        return super().private_ip_address

    def _refresh_instance_state(self):
        public_ips = []
        private_ips = []
        if pod_status := self.pod_status:
            public_ips.append(pod_status.host_ip)
            private_ips.append(pod_status.pod_ip)
        if cluster_ip_service := self._cluster_ip_service:
            private_ips.append(cluster_ip_service.spec.cluster_ip)
        return (public_ips or [None, ], private_ips or [None, ])

    @property
    def k8s_pod_uid(self) -> str:
        try:
            return str(self._pod.metadata.uid)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            return ''

    @property
    def k8s_pod_name(self) -> str:
        return str(self._pod.metadata.name)

    def wait_till_k8s_pod_get_uid(self, timeout: int = None, ignore_uid=None, throw_exc=False) -> str:
        """
        Wait till pod get any valid uid.
        If ignore_uid is provided it wait till any valid uid different from ignore_uid
        """
        if timeout is None:
            timeout = self.pod_replace_timeout
        wait_for(lambda: self.k8s_pod_uid and self.k8s_pod_uid != ignore_uid, timeout=timeout,
                 text=f"Wait till host {self} get uid", throw_exc=throw_exc)
        return self.k8s_pod_uid

    def wait_for_k8s_node_readiness(self):
        wait_for(self._wait_for_k8s_node_readiness,
                 text=f"Wait for k8s host {self.node_name} to be ready...",
                 timeout=self.pod_readiness_timeout * 60,
                 step=5,
                 throw_exc=True)

    def _wait_for_k8s_node_readiness(self):
        if self.node_name is None:
            raise RuntimeError(f"Can't find node for pod {self.name}")
        result = self.k8s_cluster.kubectl(
            f"wait node --timeout={self.pod_readiness_timeout // 3}m --for=condition=Ready {self.node_name}",
            namespace=self.parent_cluster.namespace,
            timeout=self.pod_readiness_timeout // 3 * 60 + 10
        )
        if result.stdout.count('condition met') != 1:
            raise RuntimeError('Node is not reported as ready')
        return True

    def wait_for_pod_to_appear(self):
        wait_for(self._wait_for_pod_to_appear,
                 text="Wait for pod to appear...",
                 timeout=self.pod_readiness_timeout * 60,
                 throw_exc=True)

    def _wait_for_pod_to_appear(self):
        if self._pod is None:
            raise RuntimeError('Pod is not there')
        return True

    def wait_for_pod_readiness(self, pod_readiness_timeout_minutes: int = None):
        timeout = pod_readiness_timeout_minutes or self.pod_readiness_timeout
        KubernetesOps.wait_for_pod_readiness(
            kluster=self.k8s_cluster,
            pod_name=self.name,
            namespace=self.parent_cluster.namespace,
            pod_readiness_timeout_minutes=timeout)

    @property
    def image(self) -> str:
        return self._container_status.image

    def _get_ipv6_ip_address(self):
        # NOTE: We don't support IPv6 for k8s-* backends
        return ""

    def restart(self):
        raise NotImplementedError("Not implemented yet")  # TODO: implement this method.

    def hard_reboot(self):
        self.k8s_cluster.kubectl(
            f'delete pod {self.name} --now',
            namespace=self.parent_cluster.namespace,
            timeout=self.pod_terminate_timeout * 60 + 10)

        self.wait_for_pod_readiness()

    def soft_reboot(self):
        # Kubernetes brings pods back to live right after it is deleted
        self.k8s_cluster.kubectl(
            f'delete pod {self.name} --grace-period={self.pod_terminate_timeout * 60}',
            namespace=self.parent_cluster.namespace,
            timeout=self.pod_terminate_timeout * 60 + 10)

        self.wait_for_pod_readiness()

    # On kubernetes there is no stop/start, closest analog of node restart would be soft_restart
    restart = soft_reboot

    @property
    def uptime(self):
        # update -s from inside of docker containers shows docker host uptime
        return datetime.fromtimestamp(int(self.remoter.run('stat -c %Y /proc/1', ignore_status=True).stdout.strip()))

    def start_coredump_thread(self):
        self._coredump_thread = CoredumpExportFileThread(
            self, self._maximum_number_of_cores_to_publish, ['/var/lib/scylla/coredumps'])
        self._coredump_thread.start()

    @cached_property
    def node_name(self) -> str:
        return self._pod.spec.node_name

    @property
    def instance_name(self) -> str:
        return self.node_name

    def terminate_k8s_node(self):
        """
        Delete kubernetes node, which will terminate scylla node that is running on it
        """
        self.k8s_cluster.kubectl(
            f'delete node {self.node_name} --now',
            timeout=self.pod_terminate_timeout * 60 + 10)

    def terminate_k8s_host(self):
        """
        Terminate kubernetes node via cloud API (like GCE/EC2), which will terminate scylla node that is running on it
        """
        raise NotImplementedError("To be overridden in child class")

    @staticmethod
    def is_kubernetes() -> bool:
        return True


class BaseScyllaPodContainer(BasePodContainer):  # pylint: disable=abstract-method,too-many-public-methods
    @cached_property
    def node_type(self) -> str:
        return 'db'

    def restart(self):
        pass

    @property
    def proposed_scylla_yaml(self) -> ScyllaYaml:
        """
        For kubernetes there is no node-specific scylla.yaml, only cluster-wide
        """
        return self.k8s_cluster.proposed_scylla_yaml

    @property
    def network_interfaces(self):
        pass

    parent_cluster: ScyllaPodCluster

    def actual_scylla_yaml(self) -> ContextManager[ScyllaYaml]:
        return super().remote_scylla_yaml()

    def actual_cassandra_rackdc_properties(self) -> ContextManager:
        return super().remote_cassandra_rackdc_properties()

    def remote_scylla_yaml(self) -> ContextManager:
        """
        Scylla Operator handles 'scylla.yaml' file updates using ConfigMap resource
        and we don't need to update it on each node separately.
        """
        return self.k8s_cluster.remote_scylla_yaml()

    def remote_cassandra_rackdc_properties(self) -> ContextManager:
        """
        Scylla Operator handles 'cassandra-rackdc.properties' file updates using ConfigMap resource
        and we don't need to update it on each node separately.
        """
        return self.k8s_cluster.remote_cassandra_rackdc_properties()

    @property
    def verify_up_timeout(self):
        if self.parent_cluster.params.get('k8s_scylla_disk_class') in ['gp2', 'gp3']:
            return super().verify_up_timeout * 2
        else:
            return super().verify_up_timeout

    @cluster.log_run_info
    def start_scylla_server(self, verify_up=True, verify_down=False,
                            timeout=500, verify_up_timeout=None):
        verify_up_timeout = verify_up_timeout or self.verify_up_timeout
        if verify_down:
            self.wait_db_down(timeout=timeout)
        self.remoter.run('sh -c "supervisorctl start scylla || supervisorctl start scylla-server"',
                         timeout=timeout)
        if verify_up:
            self.wait_db_up(timeout=verify_up_timeout)

    @cluster.log_run_info
    @retrying(n=3, sleep_time=5, allowed_exceptions=NETWORK_EXCEPTIONS + (CommandTimedOut, ),
              message="Failed to stop scylla.server, retrying...")
    def stop_scylla_server(self, verify_up=False, verify_down=True, timeout=300,
                           ignore_status=False):
        if verify_up:
            self.wait_db_up(timeout=timeout)
        self.remoter.run('sh -c "supervisorctl stop scylla || supervisorctl stop scylla-server"',
                         timeout=timeout, ignore_status=ignore_status)
        if verify_down:
            self.wait_db_down(timeout=timeout)

    @cluster.log_run_info
    def stop_scylla(self, verify_up=False, verify_down=True, timeout=300):
        self.stop_scylla_server(verify_up=verify_up, verify_down=verify_down, timeout=timeout)

    @property
    def node_name(self) -> str:  # pylint: disable=invalid-overridden-method
        return self._pod.spec.node_name

    def restart_scylla_server(self, verify_up_before=False, verify_up_after=True, timeout=1800,
                              verify_up_timeout=None):
        verify_up_timeout = verify_up_timeout or self.verify_up_timeout
        if verify_up_before:
            self.wait_db_up(timeout=verify_up_timeout)
        self.remoter.run('sh -c "supervisorctl restart scylla || supervisorctl restart scylla-server"',
                         timeout=timeout)
        if verify_up_after:
            self.wait_db_up(timeout=verify_up_timeout)

    @cluster.log_run_info
    def restart_scylla(self, verify_up_before=False, verify_up_after=True, timeout=1800):
        self.restart_scylla_server(
            verify_up_before=verify_up_before, verify_up_after=verify_up_after, timeout=timeout)

    @property
    def scylla_listen_address(self):
        pod_status = self.pod_status
        return pod_status and pod_status.pod_ip

    def init(self) -> None:
        super().init()
        if self.distro.is_rhel_like and not self.remoter.sudo("rpm -q iproute", ignore_status=True).ok:
            # need this because of scylladb/scylla#7560
            # Time to time 'yum install -y iproute' fails, let's download the package and install it afterwards
            self.remoter.sudo('yum install --downloadonly iproute', retry=5)
            self.remoter.sudo("yum install -y iproute")
        self.remoter.sudo('mkdir -p /var/lib/scylla/coredumps', ignore_status=True)

    def drain_k8s_node(self):
        """
        Gracefully terminating kubernetes host and return it back to life.
        It terminates scylla node that is running on it
        """
        self.k8s_cluster.log.info(
            'drain_k8s_node: kubernetes node will be drained, the following is affected :\n' + dedent('''
            GCE instance  -
            K8s node      X  <-
            Scylla Pod    X
            Scylla node   X
            '''))
        k8s_node_name = self.node_name
        self.k8s_cluster.kubectl(
            f'drain {k8s_node_name} -n scylla --ignore-daemonsets --delete-local-data')
        time.sleep(5)
        self.k8s_cluster.kubectl(f'uncordon {k8s_node_name}')

    def _restart_node_with_resharding(self, murmur3_partitioner_ignore_msb_bits: int = 12):
        # Change murmur3_partitioner_ignore_msb_bits parameter to cause resharding.
        self.stop_scylla()
        with self.remote_scylla_yaml() as scylla_yml:
            scylla_yml["murmur3_partitioner_ignore_msb_bits"] = murmur3_partitioner_ignore_msb_bits
        self.soft_reboot()
        search_reshard = self.follow_system_log(patterns=['Reshard', 'Reshap'])
        self.wait_db_up(timeout=self.pod_readiness_timeout * 60)
        return search_reshard

    @property
    def is_seed(self) -> bool:
        try:
            return 'scylla/seed' in self._svc.metadata.labels
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            return False

    @is_seed.setter
    def is_seed(self, value):
        pass

    def mark_to_be_replaced(self, overwrite: bool = False):
        if self.is_seed:
            raise RuntimeError("Scylla-operator does not support seed nodes replacement")
        # Mark pod with label that is going to be picked up by scylla-operator, pod to be removed and reconstructed
        cmd = f'label svc {self.name} scylla/replace=""'
        if overwrite:
            cmd += ' --overwrite'
        self.k8s_cluster.kubectl(cmd, namespace=self.parent_cluster.namespace, ignore_status=True)

    def wait_for_svc(self):
        wait_for(self._wait_for_svc,
                 text=f"Wait for k8s service {self.name} to be ready...",
                 timeout=self.pod_readiness_timeout * 60,
                 throw_exc=True)

    def _wait_for_svc(self):
        self.k8s_cluster.kubectl(
            f"get svc {self.name}", namespace=self.parent_cluster.namespace, verbose=False)
        return True

    def refresh_ip_address(self):
        # Invalidate ip address cache
        old_ip_info = (self.public_ip_address, self.private_ip_address)
        self._private_ip_address_cached = self._public_ip_address_cached = self._ipv6_ip_address_cached = None

        if old_ip_info == (self.public_ip_address, self.private_ip_address):
            return

        self._init_port_mapping()

    @retrying(n=60, sleep_time=10,
              allowed_exceptions=(k8s_exceptions.ApiException, invoke.exceptions.UnexpectedExit),
              message="Failed to run fstrim command...")
    def fstrim_scylla_disks(self):
        # NOTE: to be able to run 'fstrim' command in a pod, it must have direct device mount and
        # appropriate priviledges.
        # Both requirements are satisfied by 'static-local-volume-provisioner' and
        # 'local-csi-driver' pods which provide disks for scylla pods.
        # So, we run this command not on 'scylla' pods but on those
        # ones on each K8S node dedicated for scylla pods.

        if self.parent_cluster.params.get("k8s_local_volume_provisioner_type") == 'static':
            pods_selector = "app=static-local-volume-provisioner"
            scylla_disk_path = "/mnt/raid-disks/disk0/"
            namespace = "default"
        else:
            pods_selector = "app.kubernetes.io/name=local-csi-driver"
            scylla_disk_path = "/mnt/persistent-volumes"
            namespace = "local-csi-driver"
        podnames = self.k8s_cluster.kubectl(
            f"get pod -l {pods_selector} --field-selector spec.nodeName={self.node_name} "
            "-o jsonpath='{range .items[*]}{.metadata.name}'",
            namespace=namespace).stdout.strip().split("\n")
        assert podnames, (
            f"Failed to find pods using '{pods_selector}' selector on '{self.node_name}' node "
            f"in '{namespace}' namespace. Didn't run the 'fstrim' command")
        self.k8s_cluster.kubectl(
            f"exec -ti {podnames[0]} -- sh -c 'fstrim -v {scylla_disk_path}'",
            namespace=namespace)

    @cached_property
    def alternator_ca_bundle_path(self):
        ca_bundle_cmd_output = self.k8s_cluster.kubectl(
            f"get configmap/{self.parent_cluster.scylla_cluster_name}-alternator-local-serving-ca"
            f" --template='{{{{ index .data \"ca-bundle.crt\" }}}}'",
            namespace=self.parent_cluster.namespace, ignore_status=True)
        if ca_bundle_cmd_output.failed:
            self.k8s_cluster.log.warning(
                "Failed to get alternator CA bundle info: %s", ca_bundle_cmd_output.stderr)
            return None
        fd, file_name = tempfile.mkstemp(suffix='.crt')
        os.close(fd)
        ca_bundle_file = Path(file_name)
        ca_bundle_file.write_text(ca_bundle_cmd_output.stdout.strip(), encoding="utf-8")
        return str(ca_bundle_file)

    @cached_property
    def k8s_lb_dns_name(self):
        cluster_name, namespace = self.parent_cluster.scylla_cluster_name, self.parent_cluster.namespace
        return f"{cluster_name}-client.{namespace}.svc"


class LoaderPodContainer(BasePodContainer):
    TEMPLATE_PATH = LOADER_POD_CONFIG_PATH

    @cached_property
    def node_type(self) -> str:
        return 'loader'

    def __init__(self, name: str, parent_cluster: PodCluster,
                 node_prefix: str = "node", node_index: int = 1,
                 base_logdir: Optional[str] = None, dc_idx: int = 0, rack=0):
        self.loader_cluster_name = parent_cluster.loader_cluster_name
        self.loader_name = name
        self.loader_pod_name_template = f"{self.loader_name}-pod"
        super().__init__(
            name=name, parent_cluster=parent_cluster, node_prefix=node_prefix,
            node_index=node_index, base_logdir=base_logdir, dc_idx=dc_idx, rack=rack,
        )

    def init(self):
        environ = {
            "K8S_NAMESPACE": self.parent_cluster.namespace,
            "K8S_LOADER_CLUSTER_NAME": self.loader_cluster_name,
            "K8S_LOADER_NAME": self.loader_name,
            "POD_CPU_LIMIT": self.k8s_cluster.calculated_loader_cpu_limit,
            "POD_MEMORY_LIMIT": self.k8s_cluster.calculated_loader_memory_limit,
        }
        self.remoter = KubernetesPodRunner(
            kluster=self.k8s_cluster,
            template_path=self.TEMPLATE_PATH,
            template_modifiers=list(self.k8s_cluster.calculated_loader_affinity_modifiers),
            pod_name_template=self.loader_pod_name_template,
            namespace=self.parent_cluster.namespace,
            environ=environ,
        )
        self._add_node_to_argus()

    def _init_remoter(self, ssh_login_info):
        pass

    def terminate_k8s_host(self):
        raise NotImplementedError()

    def restart(self):
        raise NotImplementedError()


class LoaderStsContainer(BasePodContainer):
    TEMPLATE_PATH = LOADER_STS_CONFIG_PATH

    @cached_property
    def node_type(self) -> str:
        return 'loader'

    def terminate_k8s_host(self):
        raise NotImplementedError()

    def restart(self):
        raise NotImplementedError()


class PodCluster(cluster.BaseCluster):
    PodContainerClass: Type[BasePodContainer] = BasePodContainer

    def __init__(self,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 k8s_clusters: List[KubernetesCluster],
                 namespace: str = "default",
                 container: Optional[str] = None,
                 cluster_uuid: Optional[str] = None,
                 cluster_prefix: str = "cluster",
                 node_prefix: str = "node",
                 node_type: Optional[str] = None,
                 n_nodes: Union[list, int] = 3,
                 params: Optional[dict] = None,
                 node_pool_name: Optional[str] = '',
                 add_nodes: Optional[bool] = True,
                 nodes_smp: list[int] = []
                 ) -> None:
        self.k8s_clusters = k8s_clusters
        self.namespace = namespace
        self.container = container
        self.node_pool_name = node_pool_name

        super().__init__(cluster_uuid=cluster_uuid,
                         cluster_prefix=cluster_prefix,
                         node_prefix=node_prefix,
                         n_nodes=n_nodes,
                         params=params,
                         region_names=[],
                         node_type=node_type,
                         add_nodes=add_nodes,
                         nodes_smp=nodes_smp)

    def __str__(self):
        return f"{type(self).__name__} {self.name} | Namespace: {self.namespace}"

    @cached_property
    def pool_name(self):
        return self.node_pool_name

    def _create_node(self, node_index: int, pod_name: str, dc_idx: int, rack: int) -> BasePodContainer:
        node = self.PodContainerClass(parent_cluster=self,
                                      name=pod_name,
                                      base_logdir=self.logdir,
                                      node_prefix=self.node_prefix,
                                      node_index=node_index,
                                      dc_idx=dc_idx,
                                      rack=rack,
                                      shard_num=self.get_node_shard_num(node_index))
        # NOTE: use lock to avoid hanging running in a multitenant setup
        with NODE_INIT_LOCK:
            node.init()
        return node

    def add_nodes(self,
                  count: int,
                  ec2_user_data: str = "",
                  dc_idx: int = 0,
                  rack: int = 0,
                  enable_auto_bootstrap: bool = False,
                  instance_type=None,
                  ) -> List[BasePodContainer]:

        # TODO: make it work when we have decommissioned (by nodetool) nodes.
        #       Now it will fail because pod which hosts decommissioned Scylla member is reported
        #       as 'NotReady' and will fail the pod waiter function below.

        # Wait while whole cluster (on all racks) including new nodes are up and running
        current_dc_nodes = [node for node in self.nodes if node.dc_idx == dc_idx]
        expected_dc_nodes_count = len(current_dc_nodes) + count
        self.wait_for_pods_running(
            pods_to_wait=count, total_pods=expected_dc_nodes_count, dc_idx=dc_idx)
        self.wait_for_pods_readiness(
            pods_to_wait=count, total_pods=expected_dc_nodes_count, dc_idx=dc_idx)

        # Register new nodes and return whatever was registered
        k8s_pods = KubernetesOps.list_pods(self.k8s_clusters[dc_idx], namespace=self.namespace)
        nodes = []
        for pod in k8s_pods:
            if not any((x for x in pod.status.container_statuses if x.name == self.container)):
                continue
            is_already_registered = False
            for node in current_dc_nodes:
                if node.name == pod.metadata.name:
                    is_already_registered = True
                    break
            if is_already_registered:
                continue
            # TBD: A rack validation might be needed
            # Register a new node
            node = self._create_node(len(current_dc_nodes), pod.metadata.name, dc_idx, rack)
            nodes.append(node)
            self.nodes.append(node)
        if len(nodes) != count:
            raise RuntimeError(
                f"Requested '{count}' number of nodes to add in the '{dc_idx}' DC,"
                f" while only '{len(nodes)}' new nodes where found")
        return nodes

    def node_setup(self, node, verbose=False, timeout=3600):
        raise NotImplementedError("Derived class must implement 'node_setup' method!")

    def node_startup(self, node, verbose=False, timeout=3600):
        raise NotImplementedError("Derived class must implement 'node_startup' method!")

    def get_node_ips_param(self, public_ip=True):
        raise NotImplementedError("Derived class must implement 'get_node_ips_param' method!")

    def wait_for_init(self, *_, node_list=None, verbose=False, timeout=None, **__):  # pylint: disable=arguments-differ
        raise NotImplementedError("Derived class must implement 'wait_for_init' method!")

    def wait_sts_rollout_restart(self, pods_to_wait: int, dc_idx: int = 0):
        timeout = self.get_nodes_reboot_timeout(pods_to_wait)
        k8s_cluster = self.k8s_clusters[dc_idx]
        for statefulset in KubernetesOps.list_statefulsets(k8s_cluster, namespace=self.namespace):
            k8s_cluster.kubectl(
                f"rollout status statefulset/{statefulset.metadata.name} "
                f"--watch=true --timeout={timeout}m",
                namespace=self.namespace,
                timeout=timeout * 60 + 10)

    def get_nodes_reboot_timeout(self, count) -> Union[float, int]:
        """
        Return readiness timeout (in minutes) for case when nodes are restarted
        sums out readiness and terminate timeouts for given nodes
        """
        return count * self.PodContainerClass.pod_readiness_timeout

    @property
    def pod_selector(self):
        return ''

    @cached_property
    def get_nodes_readiness_delay(self) -> Union[float, int]:
        return self.PodContainerClass.pod_readiness_delay

    def wait_for_pods_readiness(self, pods_to_wait: int, total_pods: int, readiness_timeout: int = None,
                                dc_idx: int = 0):
        KubernetesOps.wait_for_pods_readiness(
            kluster=self.k8s_clusters[dc_idx],
            total_pods=total_pods,
            readiness_timeout=readiness_timeout or self.get_nodes_reboot_timeout(pods_to_wait),
            selector=self.pod_selector,
            namespace=self.namespace
        )

    def wait_for_pods_running(self, pods_to_wait: int, total_pods: int | callable, dc_idx: int = 0):
        KubernetesOps.wait_for_pods_running(
            kluster=self.k8s_clusters[dc_idx],
            total_pods=total_pods,
            timeout=self.get_nodes_reboot_timeout(pods_to_wait),
            selector=self.pod_selector,
            namespace=self.namespace
        )

    def generate_namespace(self, namespace_template: str) -> str:
        # Pick up not used namespace knowing that we may have more than 1 Scylla cluster
        with NAMESPACE_CREATION_LOCK:
            namespaces = self.k8s_clusters[0].kubectl(
                "get namespaces --no-headers -o=custom-columns=:.metadata.name").stdout.split()
            for i in range(1, len(namespaces)):
                candidate_namespace = f"{namespace_template}{'-' + str(i) if i > 1 else ''}"
                if candidate_namespace not in namespaces:
                    # NOTE: the namespaces must match for all the K8S clusters
                    for k8s_cluster in self.k8s_clusters:
                        k8s_cluster.kubectl(f"create namespace {candidate_namespace}")
                    return candidate_namespace
                # TODO: make it work correctly for case with reusage of multi-tenant cluster
                k8s_cluster = self.k8s_clusters[0]
                if k8s_cluster.params.get('reuse_cluster') and k8s_cluster.tenants_number < 2:
                    return namespace_template
        raise RuntimeError("No available namespace was found")


class ScyllaPodCluster(cluster.BaseScyllaCluster, PodCluster):  # pylint: disable=too-many-public-methods
    def __init__(self,
                 k8s_clusters: List[KubernetesCluster],
                 scylla_cluster_name: Optional[str] = None,
                 user_prefix: Optional[str] = None,
                 n_nodes: Union[list, int] = 3,
                 params: Optional[dict] = None,
                 node_pool_name: str = '',
                 add_nodes: bool = True,
                 nodes_smp: list[int] = []
                 ) -> None:
        self.k8s_clusters = k8s_clusters
        self.namespace = self.generate_namespace(namespace_template=SCYLLA_NAMESPACE)
        self.scylla_cluster_name = scylla_cluster_name
        # NOTE: the 'self.k8s_scylla_manager_auth_token' attr is used only in MultiDC setups
        #       and will be updated later with the value from the first region.
        self.k8s_scylla_manager_auth_token = None
        kwargs = {}
        for k8s_cluster in k8s_clusters:
            if not kwargs and params.get('use_mgmt') and k8s_cluster.cluster_backend != "k8s-eks":
                kwargs["s3_provider_endpoint"] = k8s_cluster.s3_provider_endpoint
            k8s_cluster.deploy_scylla_cluster(
                node_pool_name=node_pool_name,
                namespace=self.namespace, cluster_name=self.scylla_cluster_name, **kwargs)
        super().__init__(k8s_clusters=k8s_clusters,
                         namespace=self.namespace,
                         container="scylla",
                         cluster_prefix=cluster.prepend_user_prefix(user_prefix, 'db-cluster'),
                         node_prefix=cluster.prepend_user_prefix(user_prefix, 'db-node'),
                         node_type="scylla-db",
                         n_nodes=n_nodes,
                         params=params,
                         node_pool_name=node_pool_name,
                         add_nodes=add_nodes,
                         nodes_smp=nodes_smp)
        # NOTE: register callbacks for the Scylla pods IP change events
        for k8s_cluster in k8s_clusters:
            if k8s_cluster.scylla_pods_ip_change_tracker_thread:
                k8s_cluster.scylla_pods_ip_change_tracker_thread.register_callbacks(
                    callbacks=self.refresh_scylla_pod_ip_address,
                    namespace=self.namespace,
                    pod_name='__each__',
                    add_pod_name_as_kwarg=True)

    @property
    def scylla_manager_auth_token(self) -> str:
        raise RuntimeError(
            "K8S uses different approach for setting up the scylla manager auth token")

    def refresh_scylla_pod_ip_address(self, pod_name):
        """Designed to be used as a callback for the pod IP change tracker."""
        for node in self.nodes:
            if node.name != pod_name:
                continue
            node.refresh_ip_address()
            break
        else:
            LOGGER.warning(
                "Could not find a node with the '%s' name in '%s' namespace",
                pod_name, self.namespace)

    def get_scylla_args(self) -> str:
        # NOTE: scylla args get appended in K8S differently than in the VM case.
        #       So, we simulate 'empty args' to make the common logic work.
        return ""

    @property
    def pod_selector(self):
        return 'app=scylla'

    def wait_for_nodes_up_and_normal(self, nodes=None, verification_node=None, iterations=None,
                                     sleep_time=None, timeout=None):  # pylint: disable=too-many-arguments
        dc_node_mapping, nodes = {}, (nodes or self.nodes)
        for node in nodes:
            dc_idx = node.dc_idx
            if dc_idx not in dc_node_mapping:
                dc_node_mapping[dc_idx] = []
            dc_node_mapping[dc_idx].append(node)
        for dc_idx, dc_nodes in dc_node_mapping.items():
            self.wait_for_pods_readiness(
                pods_to_wait=len(dc_nodes), total_pods=len([n for n in self.nodes if n.dc_idx == dc_idx]),
                readiness_timeout=timeout, dc_idx=dc_idx)
            self.check_nodes_up_and_normal(nodes=nodes, verification_node=verification_node)

    @timeout_wrapper(timeout=300, sleep_time=3, allowed_exceptions=NETWORK_EXCEPTIONS + (ClusterNodesNotReady,),
                     message="Waiting for nodes to join the cluster")
    def check_nodes_up_and_normal(self, nodes=None, verification_node=None):
        super().check_nodes_up_and_normal(nodes=nodes, verification_node=verification_node)

    @cluster.wait_for_init_wrap
    def wait_for_init(self, *_, node_list=None, verbose=False, timeout=None, wait_for_db_logs=False, **__):  # pylint: disable=arguments-differ
        node_list = node_list if node_list else self.nodes
        self.wait_for_nodes_up_and_normal(nodes=node_list, timeout=timeout)
        if wait_for_db_logs:
            super().wait_for_init(node_list=node_list, check_node_health=False)
        for i, k8s_cluster in enumerate(self.k8s_clusters):
            if k8s_cluster.scylla_restart_required:
                node_list_subset = [node for node in node_list if node.dc_idx == i]
                self.restart_scylla(nodes=node_list_subset)
                self.wait_for_nodes_up_and_normal(nodes=node_list_subset, timeout=timeout)

    def _k8s_scylla_cluster_api(self, dc_idx: int = 0) -> Resource:
        return KubernetesOps.dynamic_api(self.k8s_clusters[dc_idx].dynamic_client,
                                         api_version=SCYLLA_API_VERSION,
                                         kind=SCYLLA_CLUSTER_RESOURCE_KIND)

    @retrying(n=20, sleep_time=3, allowed_exceptions=(k8s_exceptions.ApiException, ),
              message="Failed to update ScyllaCluster's spec...")
    def replace_scylla_cluster_value(self, path: str, value: Any,
                                     dc_idx: int = 0) -> Optional[ANY_KUBERNETES_RESOURCE]:
        self.k8s_clusters[dc_idx].log.debug(
            "Replace `%s' with `%s' in %s's spec", path, value, self.scylla_cluster_name)
        return self._k8s_scylla_cluster_api(dc_idx=dc_idx).patch(
            body=[{"op": "replace", "path": path, "value": value}],
            name=self.scylla_cluster_name,
            namespace=self.namespace,
            content_type=JSON_PATCH_TYPE)

    def get_scylla_cluster_value(self, path: str, dc_idx: int = 0) -> Optional[ANY_KUBERNETES_RESOURCE]:
        """
        Get scylla cluster value from kubernetes API.
        """
        cluster_data = self._k8s_scylla_cluster_api(dc_idx=dc_idx).get(
            namespace=self.namespace, name=self.scylla_cluster_name)
        return walk_thru_data(cluster_data, path)

    def get_scylla_cluster_plain_value(self, path: str, dc_idx: int = 0) -> Union[Dict, List, str, None]:
        """
        Get scylla cluster value from kubernetes API and converts result to basic python data types.
        Use it if you are going to modify the data.
        """
        cluster_data = self._k8s_scylla_cluster_api(dc_idx=dc_idx).get(
            namespace=self.namespace, name=self.scylla_cluster_name).to_dict()
        return walk_thru_data(cluster_data, path)

    def add_scylla_cluster_value(self, path: str, element: Any, dc_idx: int = 0):
        init = self.get_scylla_cluster_value(path, dc_idx=dc_idx) is None
        if path.endswith('/'):
            path = path[0:-1]
        if init:
            # You can't add to empty array, so you need to replace it
            operation = "replace"
            value = [element]
        else:
            operation = "add"
            path = path + "/-"
            value = element
        self._k8s_scylla_cluster_api(dc_idx=dc_idx).patch(
            body=[{"op": operation, "path": path, "value": value}],
            name=self.scylla_cluster_name,
            namespace=self.namespace,
            content_type=JSON_PATCH_TYPE
        )

    def remove_scylla_cluster_value(self, path: str, element_name: str, dc_idx: int = 0):
        element_list = self.get_scylla_cluster_value(path, dc_idx=dc_idx) or []
        found_index = None
        for index, item in enumerate(element_list):
            if item.get("name") == element_name:
                found_index = index
                break
        else:
            raise ValueError(f"{element_name} wasn't found in {path}")

        self._k8s_scylla_cluster_api(dc_idx=dc_idx).patch(
            body=[{"op": "remove", "path": f"{path}/{found_index}"}],
            name=self.scylla_cluster_name,
            namespace=self.namespace,
            content_type=JSON_PATCH_TYPE)

    def scylla_config_map(self, dc_idx: int = 0) -> ContextManager:
        return self.k8s_clusters[dc_idx].scylla_config_map()

    def remote_scylla_yaml(self, dc_idx: int = 0) -> ContextManager:
        return self.k8s_clusters[dc_idx].remote_scylla_yaml()

    def remote_cassandra_rackdc_properties(self, dc_idx: int = 0) -> ContextManager:
        return self.k8s_clusters[dc_idx].remote_cassandra_rackdc_properties()

    def update_seed_provider(self):
        pass

    def install_scylla_manager(self, node):
        pass

    @property
    def seed_nodes_addresses(self):
        return []

    @property
    def seed_nodes(self):
        return []

    @cached_property
    def connection_bundle_file(self) -> Path | None:
        if bundle_file := super().connection_bundle_file:
            return bundle_file

        # TODO: support multiDC case
        k8s_cluster = self.k8s_clusters[0]
        bundle_cmd_output = k8s_cluster.kubectl(
            f"get secret/{self.scylla_cluster_name}-local-cql-connection-configs-admin"
            f" --template='{{{{ index .data \"{self.scylla_cluster_name}.sct.scylladb.com\" }}}}'",
            namespace=self.namespace, ignore_status=True)

        if bundle_cmd_output.failed:
            return None

        fd, file_name = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)
        bundle_file = Path(file_name)
        bundle_file.write_bytes(base64.decodebytes(bytes(bundle_cmd_output.stdout.strip(), encoding='utf-8')))

        lb_external_hostname = k8s_cluster.kubectl(
            "get service/haproxy-kubernetes-ingress "
            "-o jsonpath='{.status.loadBalancer.ingress[0].hostname}'",
            namespace=INGRESS_CONTROLLER_NAMESPACE)

        sni_address = None
        if not (lb_external_hostname.ok and lb_external_hostname.stdout):
            lb_cluster_ip = k8s_cluster.kubectl(
                "get service/haproxy-kubernetes-ingress --template='{{ index .spec.clusterIP }}'",
                namespace=INGRESS_CONTROLLER_NAMESPACE)
            if lb_cluster_ip.ok:
                sni_address = lb_cluster_ip.stdout
        else:
            sni_address = lb_external_hostname.stdout

        if sni_address:
            # TODO: handle the case of multiple datacenters
            # need to get the cluster ip from each k8s cluster
            bundle_yaml = yaml.safe_load(bundle_file.open('r', encoding='utf-8'))
            for _, connection_data in bundle_yaml.get('datacenters', {}).items():
                connection_data['server'] = f'{sni_address.strip()}:9142'
            yaml.dump(bundle_yaml, bundle_file.open('w', encoding='utf-8'))

        return bundle_file

    def node_setup(self, node: BaseScyllaPodContainer, verbose: bool = False, timeout: int = 3600):
        if self.test_config.BACKTRACE_DECODING:
            node.install_scylla_debuginfo()
        self.node_config_setup()

    def node_startup(self, node: BaseScyllaPodContainer, verbose: bool = False, timeout: int = 3600):
        pass

    @cached_property
    def scylla_manager_cluster_name(self):  # pylint: disable=invalid-overridden-method
        return f"{self.namespace}/{self.scylla_cluster_name}"

    @property
    def scylla_manager_node(self):
        # TODO: make sure we deploy scylla manager only on first K8S cluster and reuse for all
        #       by Scylla pod IP addresses.
        return self.k8s_clusters[0].scylla_manager_cluster.nodes[0]

    def get_cluster_manager(self, create_cluster_if_not_exists: bool = False) -> AnyManagerCluster:
        return super().get_cluster_manager(create_cluster_if_not_exists=create_cluster_if_not_exists)

    def create_cluster_manager(self, cluster_name: str, manager_tool=None, host_ip=None):
        self.log.info('Scylla manager should not be manipulated on kubernetes manually')
        self.log.info('Instead of creating new cluster we will wait for 5 minutes till it get registered automatically')
        raise NotImplementedError('Scylla manager should not be manipulated on kubernetes manually')

    def node_config_setup(self,
                          client_encrypt=None):  # pylint: disable=too-many-arguments,invalid-name
        if client_encrypt is None:
            client_encrypt = self.params.get("client_encrypt")

        if client_encrypt:
            raise NotImplementedError("client_encrypt is not supported by k8s-* backends yet")

        if self.get_scylla_args():
            raise NotImplementedError("custom SCYLLA_ARGS is not supported by k8s-* backends yet")

        if self.params.get("server_encrypt"):
            raise NotImplementedError("server_encrypt is not supported by k8s-* backends yet")

        append_scylla_yaml = self.params.get("append_scylla_yaml") or {}

        if append_scylla_yaml:
            unsupported_options = ("system_key_directory", "system_info_encryption", "kmip_hosts", )
            if any(option in append_scylla_yaml for option in unsupported_options):
                raise NotImplementedError(
                    f"{unsupported_options} are not supported in append_scylla_yaml by k8s-* backends yet")

    def validate_seeds_on_all_nodes(self):
        pass

    def set_seeds(self, wait_for_timeout=300, first_only=False):
        assert self.nodes, "DB cluster should have at least 1 node"
        self.nodes[0].is_seed = True

    def _get_rack_nodes(self, rack: int, dc_idx: int) -> list:
        return sorted(
            [node for node in self.nodes if node.rack == rack and node.dc_idx == dc_idx], key=lambda n: n.name)

    def add_nodes(self,  # pylint: disable=too-many-locals,too-many-branches  noqa: PLR0913
                  count: int,
                  ec2_user_data: str = "",
                  # NOTE: 'dc_idx=None' means 'create %count% nodes on each K8S cluster'
                  dc_idx: int = None,
                  rack: int = 0,
                  enable_auto_bootstrap: bool = False,
                  instance_type=None) -> List[BasePodContainer]:
        if dc_idx is None:
            dc_idx = list(range(len(self.k8s_clusters)))
        elif isinstance(dc_idx, int):
            dc_idx = [dc_idx]
        if isinstance(count, str):
            count = count.split(" ")
        else:
            count = [count]
        assert len(count) in (1, len(dc_idx))

        assert instance_type is None, "k8s can't provision different instance types"

        new_nodes = []
        self.log.debug(
            "'%s' configuration was taken for the 'dc_idx': %s",
            "Single-DC" if len(dc_idx) < 2 else "Multi-DC", dc_idx)
        for current_dc_idx in dc_idx:
            node_count_in_dc = count[current_dc_idx] if current_dc_idx < len(count) else count[0]
            self.log.debug(
                "Going to provision '%s' nodes (node_count_in_dc) in DC with '%s' (dc_idx)",
                node_count_in_dc, current_dc_idx)
            self._create_k8s_rack_if_not_exists(rack, dc_idx=current_dc_idx)
            # TODO: 'self._get_rack_nodes(rack)' returns correct number only
            #       when there are no decommissioned, by nodetool, nodes.
            #       Having 1 decommissioned node we do not change node count.
            #       Having 2 decommissioned nodes we will reduce node count.
            current_members = len(self._get_rack_nodes(rack, dc_idx=current_dc_idx))
            # NOTE: update the 'spec.externalSeeds' field only for the very first pod in a second+ region.
            dc_podip_mapping, is_external_seeds_set = {}, False
            if current_dc_idx > 0:
                for node in self.nodes:
                    if node.dc_idx not in dc_podip_mapping:
                        dc_podip_mapping[node.dc_idx] = []
                    dc_podip_mapping[node.dc_idx].append(node.pod_status.pod_ip)
                if not dc_podip_mapping.get(current_dc_idx, []):
                    assert dc_podip_mapping[0], (
                        "Couldn't not find IP addresses of the nodes from the first DC (dc_idx=0)"
                        f" to be used as 'external seeds' for the pods of another DC (dc_idx={current_dc_idx})")
                    self.replace_scylla_cluster_value(
                        "/spec/externalSeeds", dc_podip_mapping[0], dc_idx=current_dc_idx)
                    is_external_seeds_set = True
            total_dc_members = current_members + node_count_in_dc
            self.replace_scylla_cluster_value(
                f"/spec/datacenter/racks/{rack}/members", total_dc_members, dc_idx=current_dc_idx)
            new_nodes.extend(super().add_nodes(
                count=node_count_in_dc, ec2_user_data=ec2_user_data, dc_idx=current_dc_idx, rack=rack,
                enable_auto_bootstrap=enable_auto_bootstrap))
            kubectl = self.k8s_clusters[current_dc_idx].kubectl
            if self.params.get('use_mgmt') and current_dc_idx == 0 and (
                    self.k8s_scylla_manager_auth_token is None):
                self.k8s_scylla_manager_auth_token = base64.b64decode(kubectl(
                    f"get secrets/{self.scylla_cluster_name}-auth-token"
                    " --template='{{ index .data \"auth-token.yaml\" }}'",
                    namespace=self.namespace).stdout.strip()).decode('utf-8').strip()
            elif current_dc_idx > 0 and self.k8s_scylla_manager_auth_token:
                existing_k8s_scylla_manager_auth_token = base64.b64decode(kubectl(
                    f"get secrets/{self.scylla_cluster_name}-auth-token"
                    " --template='{{ index .data \"auth-token.yaml\" }}'",
                    namespace=self.namespace).stdout.strip()).decode('utf-8').strip()
                if existing_k8s_scylla_manager_auth_token != self.k8s_scylla_manager_auth_token:
                    auth_token_base64 = base64.b64encode(
                        self.k8s_scylla_manager_auth_token.encode('utf-8')).decode('utf-8')
                    patch_cmd = (
                        f'patch secret/{self.scylla_cluster_name}-auth-token --type=json -p=\'['
                        ' {"op": "replace", "path": "/data/auth-token.yaml",'
                        f' "value": "{auth_token_base64}"'
                        ' }]\'')
                    kubectl(patch_cmd, namespace=self.namespace)
            # NOTE: remove the 'externalSeeds' values because pod IPs are ephemeral and
            #       we are not going to keep it up-to-date making Scylla pods not try to connect to
            #       some other test run's Scylla pods which may pick up those ephemeral IPs.
            #       Also, avoid redundant Scylla pods roll-outs with each addition of a new node
            #       in second+ regions.
            if current_dc_idx > 0 and is_external_seeds_set:
                self.replace_scylla_cluster_value("/spec/externalSeeds", [], dc_idx=current_dc_idx)
                # NOTE: sleep for some time to avoid concurrency with the 'not-yet-started roll-out'
                #       and 'already-finished-roll-out'. We won't waste time because roll-out takes more than
                #       our small sleep.
                time.sleep(10)
                self.wait_for_pods_running(
                    pods_to_wait=node_count_in_dc, total_pods=total_dc_members, dc_idx=current_dc_idx)
                self.wait_for_pods_readiness(
                    pods_to_wait=node_count_in_dc, total_pods=total_dc_members, dc_idx=current_dc_idx)
        return new_nodes

    def _create_k8s_rack_if_not_exists(self, rack: int, dc_idx: int):
        if self.get_scylla_cluster_value(f'/spec/datacenter/racks/{rack}', dc_idx=dc_idx) is not None:
            return
        # Create new rack of very first rack of the cluster
        new_rack = self.get_scylla_cluster_plain_value('/spec/datacenter/racks/0', dc_idx=dc_idx)
        new_rack['members'] = 0
        new_rack['name'] = f'{new_rack["name"]}-{rack}'
        self.add_scylla_cluster_value('/spec/datacenter/racks', new_rack, dc_idx=dc_idx)

    def _delete_k8s_rack(self, rack: int, dc_idx: int):
        racks = self.get_scylla_cluster_plain_value('/spec/datacenter/racks/', dc_idx=dc_idx)
        if len(racks) == 1:
            return
        racks.pop(rack)
        self.replace_scylla_cluster_value('/spec/datacenter/racks', racks, dc_idx=dc_idx)

    def decommission(self, node: BaseScyllaPodContainer, timeout: int | float = None):
        rack, dc_idx = node.rack, node.dc_idx
        rack_nodes = self._get_rack_nodes(rack, dc_idx=dc_idx)
        assert rack_nodes[-1] == node, "Can withdraw the last node only"
        current_members = len(rack_nodes)

        # NOTE: "scylla_shards" property uses remoter calls, and we save its result before
        # the target scylla node gets killed using kubectl command which precedes the target GCE
        # node deletion using "terminate_node" command.
        scylla_shards = node.scylla_shards

        timeout = timeout or (node.pod_terminate_timeout * 60)
        with adaptive_timeout(operation=Operations.DECOMMISSION, node=node):
            self.replace_scylla_cluster_value(
                f"/spec/datacenter/racks/{rack}/members", current_members - 1, dc_idx=dc_idx)
            self.k8s_clusters[node.dc_idx].kubectl(
                f"wait --timeout={timeout}s --for=delete pod {node.name}",
                namespace=self.namespace,
                timeout=timeout + 10)
        self.terminate_node(node, scylla_shards=scylla_shards)
        shutil.move(node.system_log, os.path.join(
            node.logdir, f"system_{datetime.now().strftime('%y_%m_%d_%H_%M_%S')}.log"))
        if current_members == 1:
            self._delete_k8s_rack(rack, dc_idx=dc_idx)

        if monitors := self.test_config.tester_obj().monitors:
            monitors.reconfigure_scylla_monitoring()

    def upgrade_scylla_cluster(self, new_version: str) -> None:
        self.replace_scylla_cluster_value("/spec/version", new_version)
        new_image = f"{self.params.get('docker_image')}:{new_version}"

        if not self.nodes:
            return True

        @timeout_wrapper(
            timeout=self.nodes[0].pod_replace_timeout * 2 * 60,
            sleep_time=self.PodContainerClass.pod_readiness_delay)
        def wait_till_any_node_get_new_image(nodes_with_old_image: list):
            for node in nodes_with_old_image.copy():
                # NOTE: 'node.image' may be 'docker.io/scylladb/scylla:4.5.3'
                #       as well as 'scylladb/scylla:4.5.3'
                if node.image.endswith(new_image):
                    nodes_with_old_image.remove(node)
                    return True
            raise RuntimeError('No node was upgraded')

        nodes = self.nodes.copy()
        while nodes:
            wait_till_any_node_get_new_image(nodes)

        self.wait_for_pods_readiness(len(self.nodes), len(self.nodes))

    def check_cluster_health(self):
        if self.params.get('k8s_deploy_monitoring'):
            self.check_kubernetes_monitoring_health()
        super().check_cluster_health()

    def check_kubernetes_monitoring_health(self) -> bool:
        # TODO: add grafana checks
        # TODO: make prometheus check be secure
        self.log.debug('Check kubernetes monitoring health')
        with ClusterHealthValidatorEvent() as kmh_event:
            for k8s_cluster in self.k8s_clusters:
                try:
                    prometheus_ip = k8s_cluster.get_prometheus_ip(
                        cluster_name=self.scylla_cluster_name, namespace=self.namespace)
                    PrometheusDBStats(host=prometheus_ip, port=k8s_cluster.prometheus_port, protocol='https')
                    kmh_event.message = "Kubernetes monitoring health checks have successfully been finished"
                except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                    ClusterHealthValidatorEvent.MonitoringStatus(
                        error=f'Failed to connect to K8S prometheus server (namespace={self.namespace}) at '
                              f'{prometheus_ip}:{k8s_cluster.prometheus_port}, due to the: \n'
                              ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    ).publish()
                    kmh_event.message = "Kubernetes monitoring health checks have failed"
                    return False
            return True

    def restart_scylla(self, nodes=None, random_order=False):
        nodes = nodes or self.nodes
        for i, k8s_cluster in enumerate(self.k8s_clusters):
            current_k8s_cluster_nodes = [node for node in nodes if node.dc_idx == i]
            if not current_k8s_cluster_nodes:
                self.log.warning(
                    "No DB nodes specified for the restart in the '%s' K8S cluster", k8s_cluster.name)
                continue
            patch_data = {"spec": {"forceRedeploymentReason": f"Triggered at {time.time()}"}}
            k8s_cluster.kubectl(
                f"patch scyllacluster {self.scylla_cluster_name} --type merge -p '{json.dumps(patch_data)}'",
                namespace=self.namespace)

            # NOTE: sleep for some time to avoid races.
            #       We do not waste time here, because waiting for Scylla pods restart takes minutes.
            time.sleep(10)

            readiness_timeout = self.get_nodes_reboot_timeout(len(current_k8s_cluster_nodes))
            statefulsets = KubernetesOps.list_statefulsets(k8s_cluster, namespace=self.namespace)
            if random_order:
                random.shuffle(statefulsets)
            for statefulset in statefulsets:
                k8s_cluster.kubectl(
                    f"rollout status statefulset/{statefulset.metadata.name} "
                    f"--watch=true --timeout={readiness_timeout}m",
                    namespace=self.namespace,
                    timeout=readiness_timeout * 60 + 10)
            k8s_cluster.scylla_restart_required = False

    def prefill_cluster(self, dataset_name: str):
        test_data = getattr(datasets, dataset_name, None)
        if not test_data:
            raise ValueError(f"Dataset unexpected value: '{dataset_name}'. "
                             "Dataset with this name is not defined in the sdcm.utils.sstable.load_inventory."
                             "Expected values: BIG_SSTABLE_MULTI_COLUMNS_DATA, MULTI_COLUMNS_DATA")

        test_keyspace_name = 'keyspace1'

        node = self.nodes[0]
        test_keyspaces = self.get_test_keyspaces()
        # If 'keyspace1' does not exist, create a schema and load a data.
        create_schema = not (test_keyspace_name in test_keyspaces)  # pylint: disable=superfluous-parens
        if not create_schema:
            # NOTE: if keyspace exists and has data then just exit
            if int(SstableLoadUtils.validate_data_count_after_upload(node=node)) > 0:
                return

        for node in self.nodes:
            with self.cql_connection_exclusive(node) as session:
                kwarg = {"replication_factor": 3,
                         "session": session}

                SstableLoadUtils.upload_sstables(node=node,
                                                 test_data=test_data[0],
                                                 create_schema=create_schema,
                                                 **kwarg)

            SstableLoadUtils.run_refresh(node, test_data=test_data[0])
            if create_schema:
                create_schema = False

        result = SstableLoadUtils.validate_data_count_after_upload(node=node)
        assert int(result) == test_data[0].keys_num, "Data was not inserted"


class ManagerPodCluser(PodCluster):  # pylint: disable=abstract-method
    @property
    def pod_selector(self):
        return 'app.kubernetes.io/name=scylla-manager'


class LoaderPodCluster(cluster.BaseLoaderSet, PodCluster):
    def __init__(self,
                 k8s_clusters: List[KubernetesCluster],
                 loader_cluster_name: Optional[str] = None,
                 user_prefix: Optional[str] = None,
                 n_nodes: Union[list, int] = 3,
                 params: Optional[dict] = None,
                 node_pool_name: str = '',
                 add_nodes: bool = True,
                 ) -> None:

        self.k8s_clusters = k8s_clusters
        self.loader_cluster_name = loader_cluster_name
        self.loader_cluster_created = False
        self.namespace = self.generate_namespace(namespace_template=LOADER_NAMESPACE)

        cluster.BaseLoaderSet.__init__(self, params=params)
        self.k8s_loader_run_type = self.params.get("k8s_loader_run_type")
        if self.k8s_loader_run_type == "static":
            self.PodContainerClass = LoaderStsContainer  # pylint: disable=invalid-name
        elif self.k8s_loader_run_type == "dynamic":
            self.PodContainerClass = LoaderPodContainer  # pylint: disable=invalid-name
        else:
            raise ValueError(
                "'k8s_loader_run_type' has unexpected value: %s" % self.k8s_loader_run_type)
        PodCluster.__init__(self,
                            k8s_clusters=self.k8s_clusters,
                            namespace=self.namespace,
                            container="loader",
                            cluster_prefix=cluster.prepend_user_prefix(user_prefix, "loader-set"),
                            node_prefix=cluster.prepend_user_prefix(user_prefix, "loader-node"),
                            node_type="loader",
                            n_nodes=n_nodes,
                            params=params,
                            node_pool_name=node_pool_name,
                            add_nodes=add_nodes,
                            )

    @property
    def pod_selector(self):
        return 'loader-cluster-name'

    def node_setup(self,
                   node: BasePodContainer,
                   verbose: bool = False,
                   **kwargs) -> None:

        if self.params.get('client_encrypt'):
            node.config_client_encrypt()

    def node_startup(self, node: BasePodContainer, verbose: bool = False, **kwargs) -> None:
        pass

    def _get_docker_image(self):
        if loader_image := self.params.get('stress_image.cassandra-stress'):
            return loader_image
        raise ValueError("No 'stress_image.cassandra-stress' option found in the test configuration")

    def add_nodes(self,
                  count: int,
                  ec2_user_data: str = "",
                  # NOTE: 'dc_idx=None' means 'create %count% nodes on each K8S cluster'
                  dc_idx: int = None,
                  rack: int = 0,
                  enable_auto_bootstrap: bool = False,
                  instance_type=None
                  ) -> List[BasePodContainer]:
        if self.loader_cluster_created:
            raise NotImplementedError(
                "Changing number of nodes in LoaderPodCluster is not supported.")
        if dc_idx is None:
            dc_idx = list(range(len(self.k8s_clusters)))
        elif isinstance(dc_idx, int):
            dc_idx = [dc_idx]
        if isinstance(count, str):
            count = count.split(" ")
        else:
            count = [count]
        assert len(count) in (1, len(dc_idx))
        assert instance_type is None, "k8s can't provision different instance types"

        new_nodes = []
        for current_dc_idx in dc_idx:
            self.k8s_clusters[current_dc_idx].deploy_loaders_cluster(
                node_pool_name=self.node_pool_name, namespace=self.namespace)
            node_count_in_dc = count[current_dc_idx] if current_dc_idx < len(count) else count[0]
            k8s_loader_cluster_name = (
                f"{self.loader_cluster_name}-{self.k8s_clusters[current_dc_idx].region_name}")
            if self.k8s_loader_run_type == "dynamic":
                # TODO: if it is needed to catch coredumps of loader pods then need to create
                #       appropriate daemonset with affinity rules for scheduling on the loader K8S nodes
                for node_index in range(node_count_in_dc):
                    node = self.PodContainerClass(
                        name=f"{k8s_loader_cluster_name}-{node_index}",
                        parent_cluster=self,
                        base_logdir=self.logdir,
                        node_prefix=self.node_prefix,
                        node_index=node_index,
                        dc_idx=current_dc_idx,
                        rack=rack,
                    )
                    node.init()
                    new_nodes.append(node)
                    self.nodes.append(node)
                continue

            self.k8s_clusters[current_dc_idx].apply_file(
                self.PodContainerClass.TEMPLATE_PATH,
                modifiers=self.k8s_clusters[current_dc_idx].calculated_loader_affinity_modifiers,
                environ={
                    "K8S_NAMESPACE": self.namespace,
                    "K8S_LOADER_CLUSTER_NAME": k8s_loader_cluster_name,
                    "DOCKER_IMAGE_WITH_TAG": self._get_docker_image(),
                    "N_LOADERS": node_count_in_dc,
                    "POD_CPU_LIMIT": self.k8s_clusters[current_dc_idx].calculated_loader_cpu_limit,
                    "POD_MEMORY_LIMIT": self.k8s_clusters[current_dc_idx].calculated_loader_memory_limit,
                },
            )
            self.k8s_clusters[current_dc_idx].log.debug(
                "Check the '%s' loaders cluster in the '%s' namespace",
                self.loader_cluster_name, self.namespace)
            self.k8s_clusters[current_dc_idx].kubectl("get statefulset", namespace=self.namespace)
            self.k8s_clusters[current_dc_idx].kubectl("get pods", namespace=self.namespace)
            new_nodes.extend(super().add_nodes(
                count=node_count_in_dc,
                ec2_user_data=ec2_user_data,
                dc_idx=current_dc_idx,
                rack=rack,
                enable_auto_bootstrap=enable_auto_bootstrap))

        self.loader_cluster_created = True
        return new_nodes


def get_tags_from_params(params: dict) -> Dict[str, str]:
    tags = TestConfig().common_tags()
    if params.get("post_behavior_k8s_cluster").startswith("keep"):
        # NOTE: case when 'post_behavior_k8s_cluster' is 'keep' or 'keep-on-failure':
        #       - if TestRun passes then SCT will cleanup resources by itself if needed
        #       - if TestRun fails then cluster will be kept as expected
        tags["keep"] = "alive"
    elif int(params.get("test_duration")) > 660:
        if params.get("cluster_backend") == "k8s-eks":
            # NOTE: set keep:X where X is hours equal to 'duration + 1'
            tags["keep"] = str(params.get("test_duration") // 60 + 1)
        else:
            # NOTE: our GCE/GKE cleanup scripts do not support hour-based 'keep' tag,
            #       so set it as 'alive'.
            LOGGER.warning(
                "The GKE cluster won't be cleaned up by infra clean up scripts because "
                "it has 'keep:alive' tag set due to the too big value in 'test_duration' option. "
                "Please, double-check that GKE cluster gets deleted either by SCT or manually.")
            tags["keep"] = "alive"
    return tags
