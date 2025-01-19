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

"""
Classes that introduce disruption in clusters.
"""
import contextlib
import copy
import datetime
import inspect
import logging
import math
import os
import random
import re
import time
import traceback
import json
import itertools
import enum
from contextlib import ExitStack, contextmanager
from typing import Any, List, Optional, Type, Tuple, Callable, Dict, Set, Union, Iterable
from functools import wraps, partial
from collections import defaultdict, Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import MethodType  # pylint: disable=no-name-in-module

from cassandra import ConsistencyLevel, InvalidRequest
from cassandra.query import SimpleStatement  # pylint: disable=no-name-in-module
from invoke import UnexpectedExit
from elasticsearch.exceptions import ConnectionTimeout as ElasticSearchConnectionTimeout
from argus.common.enums import NemesisStatus

from sdcm import wait
from sdcm.audit import Audit, AuditConfiguration, AuditStore
from sdcm.cluster import (
    BaseCluster,
    BaseNode,
    BaseScyllaCluster,
    DB_LOG_PATTERN_RESHARDING_START,
    DB_LOG_PATTERN_RESHARDING_FINISH,
    MAX_TIME_WAIT_FOR_NEW_NODE_UP,
    MAX_TIME_WAIT_FOR_DECOMMISSION,
    NodeSetupFailed,
    NodeSetupTimeout, HOUR_IN_SEC,
    NodeCleanedAfterDecommissionAborted,
    NodeStayInClusterAfterDecommission,
)
from sdcm.cluster_k8s import (
    KubernetesOps,
    PodCluster,
)
from sdcm.db_stats import PrometheusDBStats
from sdcm.log import SDCMAdapter
from sdcm.logcollector import save_kallsyms_map
from sdcm.mgmt.common import TaskStatus, ScyllaManagerError, get_persistent_snapshots
from sdcm.nemesis_publisher import NemesisElasticSearchPublisher
from sdcm.paths import SCYLLA_YAML_PATH
from sdcm.prometheus import nemesis_metrics_obj
from sdcm.provision.scylla_yaml import SeedProvider
from sdcm.provision.helpers.certificate import update_certificate, TLSAssets
from sdcm.remote.libssh2_client.exceptions import UnexpectedExit as Libssh2UnexpectedExit
from sdcm.sct_events import Severity
from sdcm.sct_events.database import DatabaseLogEvent
from sdcm.sct_events.decorators import raise_event_on_failure
from sdcm.sct_events.filters import DbEventsFilter, EventsSeverityChangerFilter, EventsFilter
from sdcm.sct_events.group_common_events import (
    ignore_alternator_client_errors,
    ignore_no_space_errors,
    ignore_scrub_invalid_errors,
    ignore_stream_mutation_fragments_errors,
    ignore_raft_topology_cmd_failing,
    ignore_ycsb_connection_refused,
    decorate_with_context,
    ignore_reactor_stall_errors,
    ignore_disk_quota_exceeded_errors,
    ignore_raft_transport_failing,
    decorate_with_context_if_issues_open,
    ignore_take_snapshot_failing,
    ignore_ipv6_failure_to_assign,
)
from sdcm.sct_events.health import DataValidatorEvent
from sdcm.sct_events.loaders import CassandraStressLogEvent, ScyllaBenchEvent
from sdcm.sct_events.nemesis import DisruptionEvent
from sdcm.sct_events.system import InfoEvent, CoreDumpEvent
from sdcm.sla.sla_tests import SlaTests
from sdcm.stress_thread import DockerBasedStressThread
from sdcm.utils.aws_kms import AwsKms
from sdcm.utils import cdc
from sdcm.utils.adaptive_timeouts import adaptive_timeout, Operations
from sdcm.utils.common import (get_db_tables, generate_random_string,
                               reach_enospc_on_node, clean_enospc_on_node,
                               parse_nodetool_listsnapshots,
                               update_authenticator, ParallelObject,
                               ParallelObjectResult, sleep_for_percent_of_duration, get_views_of_base_table)
from sdcm.utils.features import is_tablets_feature_enabled
from sdcm.utils.quota import configure_quota_on_node_for_scylla_user_context, is_quota_enabled_on_node, enable_quota_on_node, \
    write_data_to_reach_end_of_quota
from sdcm.utils.compaction_ops import CompactionOps, StartStopCompactionArgs
from sdcm.utils.context_managers import nodetool_context
from sdcm.utils.decorators import retrying, latency_calculator_decorator
from sdcm.utils.decorators import timeout as timeout_decor
from sdcm.utils.decorators import skip_on_capacity_issues
from sdcm.utils.docker_utils import ContainerManager
from sdcm.utils.k8s import (
    convert_cpu_units_to_k8s_value,
    convert_cpu_value_from_k8s_to_units, convert_memory_value_from_k8s_to_units,
)
from sdcm.utils.k8s.chaos_mesh import MemoryStressExperiment, IOFaultChaosExperiment, DiskError, NetworkDelayExperiment, \
    NetworkPacketLossExperiment, NetworkCorruptExperiment, NetworkBandwidthLimitExperiment
from sdcm.utils.ldap import SASLAUTHD_AUTHENTICATOR, LdapServerType
from sdcm.utils.loader_utils import DEFAULT_USER, DEFAULT_USER_PASSWORD, SERVICE_LEVEL_NAME_TEMPLATE
from sdcm.utils.nemesis_utils.indexes import get_random_column_name, create_index, \
    wait_for_index_to_be_built, verify_query_by_index_works, drop_index, get_column_names, \
    wait_for_view_to_be_built, drop_materialized_view, is_cf_a_view
from sdcm.utils.node import build_node_api_command
from sdcm.utils.replication_strategy_utils import temporary_replication_strategy_setter, \
    NetworkTopologyReplicationStrategy, ReplicationStrategy, SimpleReplicationStrategy
from sdcm.utils.sstable.load_utils import SstableLoadUtils
from sdcm.utils.sstable.sstable_utils import SstableUtils
from sdcm.utils.tablets.common import wait_for_tablets_balanced
from sdcm.utils.toppartition_util import NewApiTopPartitionCmd, OldApiTopPartitionCmd
from sdcm.utils.version_utils import (
    MethodVersionNotFound, scylla_versions, ComparableScyllaVersion, get_systemd_version)
from sdcm.utils.raft import Group0MembersNotConsistentWithTokenRingMembersException, TopologyOperations
from sdcm.utils.raft.common import NodeBootstrapAbortManager, get_topology_coordinator_node
from sdcm.utils.issues import SkipPerIssues
from sdcm.wait import wait_for, wait_for_log_lines
from sdcm.exceptions import (
    KillNemesis,
    NoFilesFoundToDestroy,
    NoKeyspaceFound,
    FilesNotCorrupted,
    LogContentNotFound,
    LdapNotRunning,
    TimestampNotFound,
    PartitionNotFound,
    WatcherCallableException,
    UnsupportedNemesis,
    CdcStreamsWasNotUpdated,
    NemesisSubTestFailure,
    AuditLogTestFailure,
    BootstrapStreamErrorFailure,
    QuotaConfigurationFailure,
    NemesisStressFailure,
)
from test_lib.compaction import CompactionStrategy, get_compaction_strategy, get_compaction_random_additional_params, \
    get_gc_mode, GcMode, calculate_allowed_twcs_ttl_borders, get_table_compaction_info
from test_lib.cql_types import CQLTypeBuilder
from test_lib.sla import ServiceLevel, MAX_ALLOWED_SERVICE_LEVELS
from sdcm.utils.topology_ops import FailedDecommissionOperationMonitoring

LOGGER = logging.getLogger(__name__)
# NOTE: following lock is needed in the K8S multitenant case
NEMESIS_LOCK = Lock()
NEMESIS_RUN_INFO = {}
EXCLUSIVE_NEMESIS_NAMES = (
    "disrupt_drain_kubernetes_node_then_replace_scylla_node",
    "disrupt_terminate_kubernetes_host_then_replace_scylla_node",
    "disrupt_drain_kubernetes_node_then_decommission_and_add_scylla_node",
    "disrupt_terminate_kubernetes_host_then_decommission_and_add_scylla_node",
)

NEMESIS_TARGET_SELECTION_LOCK = Lock()
DISRUPT_POOL_PROPERTY_NAME = "target_pool"


class NEMESIS_TARGET_POOLS(enum.Enum):
    data_nodes = "data_nodes"
    zero_nodes = "zero_nodes"
    all_nodes = "nodes"


class DefaultValue:  # pylint: disable=too-few-public-methods
    """
    This is class is intended to be used as default value for the cases when None is not applicable
    """


def target_data_nodes(func: Callable) -> Callable:
    setattr(func, DISRUPT_POOL_PROPERTY_NAME, NEMESIS_TARGET_POOLS.data_nodes)
    return func


def target_zero_nodes(func: Callable) -> Callable:
    setattr(func, DISRUPT_POOL_PROPERTY_NAME, NEMESIS_TARGET_POOLS.zero_nodes)
    return func


def target_all_nodes(func: Callable) -> Callable:
    setattr(func, DISRUPT_POOL_PROPERTY_NAME, NEMESIS_TARGET_POOLS.all_nodes)
    return func


class Nemesis:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    DISRUPT_NAME_PREF: str = "disrupt_"

    # nemesis flags:
    topology_changes: bool = False  # flag that signal that nemesis is changing cluster topology,
    # i.e. adding/removing nodes/data centers
    disruptive: bool = False        # flag that signal that nemesis disrupts node/cluster,
    # i.e reboot,kill, hardreboot, terminate
    run_with_gemini: bool = True    # flag that signal that nemesis runs with gemini tests
    networking: bool = False        # flag that signal that nemesis interact with nemesis,
    # i.e switch off/on network interface, network issues
    kubernetes: bool = False        # flag that signal that nemesis run with k8s cluster
    limited: bool = False           # flag that signal that nemesis are belong to limited set of nemesises
    has_steady_run: bool = False    # flag that signal that nemesis should be run with perf tests with steady run
    schema_changes: bool = False
    config_changes: bool = False
    free_tier_set: bool = False     # nemesis should be run in FreeTierNemesisSet
    manager_operation: bool = False  # flag that signals that the nemesis uses scylla manager
    delete_rows: bool = False  # A flag denotes a nemesis deletes partitions/rows, generating tombstones.
    zero_node_changes: bool = False

    def __init__(self, tester_obj, termination_event, *args, nemesis_selector=None, **kwargs):  # pylint: disable=unused-argument
        for name, member in inspect.getmembers(self, lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
            if not name.startswith(self.DISRUPT_NAME_PREF):
                continue
            is_exclusive = name in EXCLUSIVE_NEMESIS_NAMES
            # add "disrupt_method_wrapper" decorator to all methods are started with "disrupt_"
            setattr(self, name,
                    MethodType(disrupt_method_wrapper(member, is_exclusive=is_exclusive), self))

        # *args -  compatible with CategoricalMonkey
        self.tester = tester_obj  # ClusterTester object
        self.cluster: Union[BaseCluster, BaseScyllaCluster] = tester_obj.db_cluster
        self.loaders = tester_obj.loaders
        self.monitoring_set = tester_obj.monitors
        self.target_node: BaseNode = None
        self.disruptions_list = []
        self.disruptions_cycle: Iterable[Callable] | None = None
        self.termination_event = termination_event
        self.operation_log = []
        self.current_disruption = None
        self.duration_list = []
        self.error_list = []
        self.interval = 60 * self.tester.params.get('nemesis_interval')  # convert from min to sec
        self.start_time = time.time()
        self.stats = {}
        self.nemesis_selector_list = nemesis_selector or []
        # NOTE: 'cluster_index' is set in K8S multitenant case
        if hasattr(self.tester, "cluster_index"):
            tenant_short_name = f"db{self.tester.cluster_index}"
            self.metrics_srv = nemesis_metrics_obj(metric_name_suffix=tenant_short_name)
            logger = logging.getLogger(f"{__name__} | {tenant_short_name}")
        else:
            self.metrics_srv = nemesis_metrics_obj()
            logger = logging.getLogger(__name__)
        self.log = SDCMAdapter(logger, extra={'prefix': str(self)})
        self.task_used_streaming = None
        self.filter_seed = self.cluster.params.get('nemesis_filter_seeds')
        self._random_sequence = None
        self._add_drop_column_max_per_drop = 5
        self._add_drop_column_max_per_add = 5
        self._add_drop_column_max_column_name_size = 10
        self._add_drop_column_max_columns = 200
        self._add_drop_column_columns_info = {}
        self._add_drop_column_target_table = []
        self._add_drop_column_tables_to_ignore = {
            'alternator_usertable': '*',  # Ignore alternator tables
            'ks_truncate': 'counter1',  # Ignore counter table
            'keyspace1': 'counter1',  # Ignore counter table
            # TODO: issue https://github.com/scylladb/scylla/issues/6074. Waiting for dev conclusions
            'cqlstress_lwt_example': '*'  # Ignore LWT user-profile tables
        }
        self.es_publisher = NemesisElasticSearchPublisher(self.tester)
        self._init_num_deletions_factor()
        self._target_node_pool_type = NEMESIS_TARGET_POOLS.data_nodes

    def _init_num_deletions_factor(self):
        # num_deletions_factor is a numeric divisor. It's a factor by which the available-partitions-for-deletion
        # is divided. This is in order to specify choose_partitions_for_delete(partitions_amount) a reasonable number
        # of partitions to delete. We prefer not to delete "too many" partitions at once in a single nemesis.
        # In case 'stress_cmd' has a write-mode and partitions are rewritten, then it is OK to delete all,
        # (so this factor is set to - 1)
        # Example usage: partitions_amount=self.tester.partitions_attrs.non_validated_partitions // self.num_deletions_factor
        self.num_deletions_factor = 5
        if stress_cmds := self.cluster.params.get('stress_cmd'):
            if not isinstance(stress_cmds, list):
                stress_cmds = [stress_cmds]
            for stress_cmd in stress_cmds:
                stress_cmd_splitted = stress_cmd.split()
                # In case background load has writes, we can delete all available partitions,
                # since they are rewritten. Otherwise, we can only delete some of them.
                if 'scylla-bench' in stress_cmd_splitted and '-mode=write' in stress_cmd_splitted:
                    self.num_deletions_factor = 1
                    break

    @classmethod
    def add_disrupt_method(cls, func=None):
        """
        Add disrupt methods to Nemesis class, so those can be randomlly selected by `Nemesis.call_random_disrupt_method`
        or `Nemesis.get_list_of_disrupt_methods_for_nemesis_subclasses`.

        example of usage:
        >>> class AddRemoveDCMonkey(Nemesis):
        >>>    @Nemesis.add_disrupt_method
        >>>    def disrupt_add_remove_dc(self):
        >>>        return 'Worked'
        >>>
        >>>    def disrupt(self):
        >>>        self.disrupt_add_remove_dc()

        :param func: if not None, function was used with parantasis @Nemesis.add_disrupt_method()
        :return: the original function
        """
        if func is None:
            # if func is not defined, return a this function wrapped see https://stackoverflow.com/a/39335652
            return partial(cls.add_disrupt_method)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)  # bind it to Nemesis class
        return func  # returning func means func can still be used normally

    @staticmethod
    @contextmanager
    def run_nemesis(node_list: list['BaseNode'], nemesis_label: str):
        """
        pick a node out of a `node_list`, and mark is as running_nemesis
        for the duration of this context
        """
        with NEMESIS_TARGET_SELECTION_LOCK:
            free_nodes = [node for node in node_list if not node.running_nemesis]
            assert free_nodes, f"couldn't find nodes for running:`{nemesis_label}`, are all nodes running nemesis ?"
            node = random.choice(free_nodes)
            node.running_nemesis = nemesis_label
        try:
            yield node
        finally:
            with NEMESIS_TARGET_SELECTION_LOCK:
                node.running_nemesis = None

    def use_nemesis_seed(self):
        if nemesis_seed := self.tester.params.get("nemesis_seed"):
            random.seed(nemesis_seed)

    def update_stats(self, disrupt, status=True, data=None):
        if not data:
            data = {}
        key = {True: 'runs', False: 'failures'}
        if disrupt not in self.stats:
            self.stats[disrupt] = {'runs': [], 'failures': [], 'cnt': 0}
        self.stats[disrupt][key[status]].append(data)
        self.stats[disrupt]['cnt'] += 1
        self.log.debug('Update nemesis info with: %s', data)
        if self.tester.create_stats:
            self.tester.update({'nemesis': self.stats})
        if self.es_publisher:
            self.es_publisher.publish(disrupt_name=disrupt, status=status, data=data)

    def publish_event(self, disrupt, status=True, data=None):
        if not data:
            data = {}
        data['node'] = self.target_node
        severity = Severity.NORMAL if status else Severity.ERROR
        DisruptionEvent(nemesis_name=disrupt, severity=severity, **data).publish()

    def switch_target_node(self, node: BaseNode):
        with NEMESIS_TARGET_SELECTION_LOCK:
            self.target_node.running_nemesis = None
            self.target_node = None
            node.running_nemesis = self.current_disruption
            self.target_node = node

    def set_current_running_nemesis(self, node):
        with NEMESIS_TARGET_SELECTION_LOCK:
            node.running_nemesis = self.current_disruption

    @staticmethod
    def unset_current_running_nemesis(node):
        if node is not None:
            with NEMESIS_TARGET_SELECTION_LOCK:
                node.running_nemesis = None

    def set_target_node_pool_type(self, pool_type: NEMESIS_TARGET_POOLS = NEMESIS_TARGET_POOLS.data_nodes):
        """Set pool type to choose nodes for target node """
        self._target_node_pool_type = pool_type

    def _get_target_nodes(
            self,
            is_seed: Optional[Union[bool, DefaultValue]] = DefaultValue,
            dc_idx: Optional[int] = None,
            rack: Optional[int] = None) -> list:
        """
        Filters and return nodes from target node pool that has no running nemesis on them
        Target node pool could be set with method 'set_target_node_pool' to use
        only data_nodes, zero_token nodes or any set of nodes.
        It can filter node by following criteria: is_seed, dc_idx, rack
        Same mechanism works for other parameters, if multiple criteria provided it will return nodes
        that match all of them.
        if is_seed is None - it will ignore seed status of the nodes
        if is_seed is True - it will pick only seed nodes
        if is_seed is False - it will pick only non-seed nodes
        if is_seed is DefaultValue - if self.filter_seed is True it act as if is_seed=False,
          otherwise it will act as if is_seed is None
        """
        if is_seed is DefaultValue:
            is_seed = False if self.filter_seed else None
        self.log.debug("Target node pool type: %s", self._target_node_pool_type)
        nodes = [node for node in getattr(self.cluster, self._target_node_pool_type.value) if not node.running_nemesis]
        if is_seed is not None:
            nodes = [node for node in nodes if node.is_seed == is_seed]
        if dc_idx is not None:
            nodes = [node for node in nodes if node.dc_idx == dc_idx]
        if rack is not None:
            nodes = [node for node in nodes if node.rack == rack]
        return nodes

    def set_target_node(self, dc_idx: Optional[int] = None, rack: Optional[int] = None,
                        is_seed: Union[bool, DefaultValue, None] = DefaultValue,
                        allow_only_last_node_in_rack: bool = False, current_disruption=None):
        """Set a Scylla node as target node.

        if is_seed is None - it will ignore seed status of the nodes
        if is_seed is True - it will pick only seed nodes
        if is_seed is False - it will pick only non-seed nodes
        if is_seed is DefaultValue - if self.filter_seed is True it act as if is_seed=False,
          otherwise it will act as if is_seed is None
        """
        # first give up the current target node
        self.unset_current_running_nemesis(self.target_node)

        with NEMESIS_TARGET_SELECTION_LOCK:
            nodes = self._get_target_nodes(is_seed=is_seed, dc_idx=dc_idx, rack=rack)
            if not nodes:
                dc_str = '' if dc_idx is None else f'dc {dc_idx} '
                rack_str = '' if rack is None else f'rack {rack} '
                raise UnsupportedNemesis(
                    f"Can't allocate node from {dc_str}{rack_str}to run nemesis on")
            if allow_only_last_node_in_rack:
                self.target_node = nodes[-1]
            else:
                self.target_node = random.choice(nodes)

            if current_disruption:
                self.target_node.running_nemesis = current_disruption
                self.set_current_disruption(current_disruption)
            elif self.current_disruption:
                self.target_node.running_nemesis = self.current_disruption
            else:
                raise ValueError("current_disruption is not set")
            self.log.info('Current Target: %s with running nemesis: %s',
                          self.target_node, self.target_node.running_nemesis)

    @raise_event_on_failure
    def run(self, interval=None, cycles_count: int = -1):
        self.es_publisher.create_es_connection()
        if interval:
            self.interval = interval * 60
        self.log.info('Interval: %s s', self.interval)
        try:
            while not self.termination_event.is_set():
                if cycles_count == 0:
                    self.log.info("Defined number of Nemesis cycles completed. Stopping Nemesis thread.")
                    break
                cycles_count -= 1
                cur_interval = self.interval
                try:
                    self.disrupt()
                except (UnsupportedNemesis, MethodVersionNotFound) as exc:
                    self.log.warning("Skipping unsupported nemesis: %s", exc)
                    cur_interval = 0
                finally:
                    self.unset_current_running_nemesis(self.target_node)
                    self.termination_event.wait(timeout=cur_interval)
        except KillNemesis:
            self.log.debug("Nemesis thread [%s] stopped by KillNemesis", id(self))

    def report(self):
        if self.duration_list:
            avg_duration = sum(self.duration_list) / len(self.duration_list)
        else:
            avg_duration = 0

        self.log.info('Report')
        self.log.info('DB Version: %s', getattr(self.cluster.nodes[0], "scylla_version", "n/a"))
        self.log.info('Interval: %s s', self.interval)
        self.log.info('Average duration: %s s', avg_duration)
        self.log.info('Total execution time: %s s', int(time.time() - self.start_time))
        self.log.info('Times executed: %s', len(self.duration_list))
        self.log.info('Unexpected errors: %s', len(self.error_list))
        self.log.info('Operation log:')
        for operation in self.operation_log:
            self.log.info(operation)

    # pylint: disable=too-many-arguments,unused-argument
    def get_list_of_methods_compatible_with_backend(
            self,
            disruptive: Optional[bool] = None,
            run_with_gemini: Optional[bool] = None,
            networking: Optional[bool] = None,
            limited: Optional[bool] = None,
            topology_changes: Optional[bool] = None,
            schema_changes: Optional[bool] = None,
            config_changes: Optional[bool] = None,
            free_tier_set: Optional[bool] = None,
            manager_operation: Optional[bool] = None,
            zero_node_changes: Optional[bool] = None,
    ) -> List[str]:
        return self.get_list_of_methods_by_flags(
            disruptive=disruptive,
            run_with_gemini=run_with_gemini,
            networking=networking,
            kubernetes=self._is_it_on_kubernetes() or None,
            limited=limited,
            topology_changes=topology_changes,
            schema_changes=schema_changes,
            config_changes=config_changes,
            free_tier_set=free_tier_set,
            manager_operation=manager_operation,
            zero_node_changes=zero_node_changes
        )

    def _is_it_on_kubernetes(self) -> bool:
        return isinstance(getattr(self.tester, "db_cluster", None), PodCluster)

    # pylint: disable=too-many-arguments,unused-argument
    def get_list_of_methods_by_flags(  # pylint: disable=too-many-locals  # noqa: PLR0913
            self,
            disruptive: Optional[bool] = None,
            run_with_gemini: Optional[bool] = None,
            networking: Optional[bool] = None,
            kubernetes: Optional[bool] = None,
            limited: Optional[bool] = None,
            topology_changes: Optional[bool] = None,
            schema_changes: Optional[bool] = None,
            config_changes: Optional[bool] = None,
            free_tier_set: Optional[bool] = None,
            sla: Optional[bool] = None,
            manager_operation: Optional[bool] = None,
            zero_node_changes: Optional[bool] = None,
    ) -> List[str]:
        subclasses_list = self._get_subclasses(
            disruptive=disruptive,
            run_with_gemini=run_with_gemini,
            networking=networking,
            kubernetes=kubernetes,
            limited=limited,
            topology_changes=topology_changes,
            schema_changes=schema_changes,
            config_changes=config_changes,
            free_tier_set=free_tier_set,
            sla=sla,
            manager_operation=manager_operation,
        )
        disrupt_methods_list = []
        for subclass in subclasses_list:
            method_name = re.search(
                r'self\.(?P<method_name>disrupt_[A-Za-z_]+?)\(.*\)', inspect.getsource(subclass), flags=re.MULTILINE)
            if method_name:
                disrupt_methods_list.append(method_name.group('method_name'))
        self.log.debug("Gathered subclass methods: {}".format(disrupt_methods_list))
        return disrupt_methods_list

    def get_list_of_subclasses_by_property_name(self, list_of_properties_to_include):
        flags = {flag_name.strip('!'): not flag_name.startswith(
            '!') for flag_name in list_of_properties_to_include}
        subclasses_list = self._get_subclasses(**flags)
        return subclasses_list

    def get_list_of_disrupt_methods(self, subclasses_list, export_properties=False):
        disrupt_methods_objects_list = []
        disrupt_methods_names_list = []
        nemesis_classes = []
        all_methods_with_properties = []
        for subclass in subclasses_list:
            properties_list = []
            per_method_properties = {}
            method_name = re.search(
                r'self\.(?P<method_name>disrupt_[0-9A-Za-z_]+?)\(.*\)', inspect.getsource(subclass), flags=re.MULTILINE)
            for attribute in subclass.__dict__.keys():
                if attribute[:2] != '__':
                    value = getattr(subclass, attribute)
                    if not callable(value):
                        properties_list.append(f"{attribute} = {value}")
            if method_name:
                method_name_str = method_name.group('method_name')
                disrupt_methods_names_list.append(method_name_str)
                nemesis_classes.append(subclass.__name__)
                if export_properties:
                    per_method_properties[method_name_str] = properties_list
                    all_methods_with_properties.append(per_method_properties)
                    all_methods_with_properties = sorted(all_methods_with_properties, key=lambda d: list(d.keys()))
        nemesis_classes.sort()
        self.log.debug("list of matching disrupions: {}".format(disrupt_methods_names_list))
        for _ in disrupt_methods_names_list:
            disrupt_methods_objects_list = [attr[1] for attr in inspect.getmembers(self) if
                                            attr[0] in disrupt_methods_names_list and callable(attr[1])]
        return disrupt_methods_objects_list, all_methods_with_properties, nemesis_classes

    @classmethod
    def _get_subclasses(cls, **flags) -> List[Type['Nemesis']]:
        tmp = Nemesis.__subclasses__()
        subclasses = []
        while tmp:
            for nemesis in tmp.copy():
                subclasses.append(nemesis)
                tmp.remove(nemesis)
                tmp.extend(nemesis.__subclasses__())
        return cls._get_subclasses_from_list(subclasses, **flags)

    @staticmethod
    def _get_subclasses_from_list(
            list_of_nemesis: List[Type['Nemesis']],
            **flags) -> List[Type['Nemesis']]:
        """
        It apply 'and' logic to filter,
            if any value in the filter does not match what nemeses have,
            nemeses will be filtered out.
        """
        nemesis_subclasses = []
        nemesis_to_exclude = COMPLEX_NEMESIS + DEPRECATED_LIST_OF_NEMESISES
        for nemesis in list_of_nemesis:
            if nemesis in nemesis_to_exclude:
                continue
            matches = True
            for filter_name, filter_value in flags.items():
                if filter_value is None:
                    continue
                attr = getattr(nemesis, filter_name, False)
                if attr != filter_value:
                    matches = False
                    break
            if not matches:
                continue
            nemesis_subclasses.append(nemesis)
        return nemesis_subclasses

    def __str__(self):
        try:
            return str(self.__class__).split("'")[1]
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            return str(self.__class__)

    def _kill_scylla_daemon(self):
        with EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                         event_class=CassandraStressLogEvent,
                                         regex=".*Connection reset by peer.*",
                                         extra_time_to_expiration=30):
            self.log.info('Kill all scylla processes in %s', self.target_node)
            self.target_node.remoter.sudo("pkill -9 scylla", ignore_status=True)

            # Wait for the process to be down before waiting for service to be restarted
            self.target_node.wait_db_down(check_interval=2)

            # Let's wait for the target Node to have their services re-started
            self.log.info('Waiting scylla services to be restarted after we killed them...')
            self.target_node.wait_db_up(timeout=14400)
            if (self.cluster.params.get('use_mgmt')
                    and SkipPerIssues('scylladb/scylla-manager#2813', params=self.tester.params)):
                # Workaround for https://github.com/scylladb/scylla-manager/issues/2813
                # When scylla take too long time to bring api port up
                #  scylla-manager-agent fails to start and never go up
                self.target_node.start_service(service_name='scylla-manager-agent', timeout=600, ignore_status=True)
            self.log.info('Waiting JMX services to be restarted after we killed them...')
            self.target_node.wait_jmx_up()
        self.cluster.wait_for_schema_agreement()

    @decorate_with_context(ignore_raft_topology_cmd_failing)
    @target_all_nodes
    def disrupt_stop_wait_start_scylla_server(self, sleep_time=300):  # pylint: disable=invalid-name
        self.target_node.stop_scylla_server(verify_up=False, verify_down=True)
        self.log.info("Sleep for %s seconds", sleep_time)
        time.sleep(sleep_time)
        if self._is_it_on_kubernetes():
            # Kubernetes brings node up automatically, no need to start it up
            self.target_node.wait_node_fully_start(timeout=sleep_time)
            return
        self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    @decorate_with_context(ignore_ycsb_connection_refused)
    def disrupt_stop_start_scylla_server(self):  # pylint: disable=invalid-name
        self.target_node.stop_scylla_server(verify_up=False, verify_down=True)
        self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    @staticmethod
    def _handle_start_stop_compaction_results(trigger_and_watcher_futures: dict[str, ParallelObjectResult],
                                              allow_trigger_exceptions: bool = True):
        """
        Handle the results from running in parallel a compaction
        trigger and watcher / stop-compaction functions.
        """
        if trigger_and_watcher_futures["trigger"].exc:
            LOGGER.warning("Trigger function failed with:\n%s", trigger_and_watcher_futures["trigger"].exc)

        if trigger_and_watcher_futures["watcher"].exc:
            raise WatcherCallableException from trigger_and_watcher_futures["watcher"].exc
        return trigger_and_watcher_futures

    def _get_random_non_system_ks_cf(self, filter_empty_tables: bool = True) -> tuple[str, str] | None:
        """
        Get a random (ks, cf) from the target node.
        The get_non_system_ks_cf_list returns a list of strings
        in the format 'keyspace_name.cf_name' so we need to
        call split() on the output to get keyspace_name and
        cf_name separately.
        """
        if ks_cf_list := self.cluster.get_non_system_ks_cf_list(
                db_node=self.target_node, filter_empty_tables=filter_empty_tables):
            return random.choice(ks_cf_list).split(".")
        else:
            return None, None

    def _prepare_start_stop_compaction(self) -> StartStopCompactionArgs:
        ks, cf = self._get_random_non_system_ks_cf()

        if not ks or not cf:
            raise UnsupportedNemesis("No non-system keyspaces to run the nemesis with. Skipping.")

        return StartStopCompactionArgs(
            keyspace=ks,
            columnfamily=cf,
            timeout=360,
            target_node=self.target_node,
            compaction_ops=CompactionOps(cluster=self.cluster, node=self.target_node)
        )

    def disrupt_start_stop_major_compaction(self):
        """
        Start and stop a major compaction on a non-system columnfamily.

        1. Query the target node for a non-system cf. If no cf is
        found, skip the nemesis.

        In parallel:
        2.1 Start a major compaction using a REST API call (via a Remoter
        curl call on the target node).
        2.2. Watch for the major compaction to show up in logs.
        3. Stop the running major compaction by issuing a <nodetool stop
        COMPACTION> command after the watcher picks up the compaction
        started.

        The REST API call starting the compaction may not return, since
        it returns on completing the compaction task (which we attempt
        to interrupt). We ignore such timeouts and validate only the
        success of the command in (4).
        """
        compaction_args = self._prepare_start_stop_compaction()
        trigger_func = partial(compaction_args.compaction_ops.trigger_major_compaction,
                               keyspace=compaction_args.keyspace,
                               cf=compaction_args.columnfamily)
        watch_func = partial(compaction_args.compaction_ops.stop_on_user_compaction_logged,
                             node=compaction_args.target_node,
                             mark=compaction_args.target_node.mark_log(),
                             watch_for="User initiated compaction started on behalf of",
                             timeout=compaction_args.timeout,
                             stop_func=compaction_args.compaction_ops.stop_major_compaction)

        results = ParallelObject.run_named_tasks_in_parallel(
            tasks={"trigger": trigger_func, "watcher": watch_func},
            timeout=compaction_args.timeout + 5,
            ignore_exceptions=True
        )

        self._handle_start_stop_compaction_results(
            trigger_and_watcher_futures=results,
            allow_trigger_exceptions=True
        )

    def clear_snapshots(self) -> None:
        self.log.info("Clear snapshots if there are some of them")
        result = self.target_node.run_nodetool("clearsnapshot")
        self.log.debug(result)

    def disrupt_start_stop_scrub_compaction(self):
        """
        Start and stop a scrub compaction on a non-system columnfamily.

        1. Query the target node for a non-system cf. If no cf is found,
        skip the nemesis.

        In parallel:
        2.1 Start a scrub compaction using a REST API call (via a Remoter
        curl call on the target node).
        2.2. Watch for the scrub compaction to show up in logs.
        3. Stop the running scrub compaction by issuing a <nodetool stop
        SCRUB> command after the watcher picks up the compaction started.

        The REST API call starting the compaction may not return, since it
        returns on completing the compaction task (which we attempt to
        interrupt). We ignore such timeouts and validate only the success
        of the command in (4).
        """
        compaction_args = self._prepare_start_stop_compaction()
        trigger_func = partial(compaction_args.compaction_ops.trigger_scrub_compaction)
        watch_func = partial(compaction_args.compaction_ops.stop_on_user_compaction_logged,
                             node=compaction_args.target_node,
                             mark=compaction_args.target_node.mark_log(),
                             watch_for="Scrubbing",
                             timeout=compaction_args.timeout,
                             stop_func=compaction_args.compaction_ops.stop_scrub_compaction)

        results = ParallelObject.run_named_tasks_in_parallel(
            tasks={"trigger": trigger_func, "watcher": watch_func},
            timeout=compaction_args.timeout + 5,
            ignore_exceptions=True
        )

        self._handle_start_stop_compaction_results(
            trigger_and_watcher_futures=results,
            allow_trigger_exceptions=True
        )

        self.clear_snapshots()

    def disrupt_start_stop_cleanup_compaction(self):
        """
        Start and stop a cleanup compaction on a non-system columnfamily.

        1. Query the target node for a non-system cf. If no cf is found,
        skip the nemesis.

        In parallel:
        2.1 Start a cleanup compaction using a REST API call (via a Remoter
        curl call on the target node).
        2.2. Watch for the cleanup compaction to show up in logs.
        3. Stop the running cleanup compaction by issuing a <nodetool stop
        CLEANUP> command after the watcher picks up the compaction started.

        The REST API call starting the compaction may not return, since it
        returns on  completing the compaction task (which we attempt to
        interrupt). We ignore such timeouts and validate only the success
        of the command in (4).
        """
        compaction_args = self._prepare_start_stop_compaction()
        trigger_func = partial(compaction_args.compaction_ops.trigger_cleanup_compaction, timeout=600)
        watch_func = partial(compaction_args.compaction_ops.stop_on_user_compaction_logged,
                             node=compaction_args.target_node,
                             mark=compaction_args.target_node.mark_log(),
                             timeout=compaction_args.timeout,
                             watch_for="Cleaning",
                             stop_func=compaction_args.compaction_ops.stop_cleanup_compaction)

        results = ParallelObject.run_named_tasks_in_parallel(
            tasks={"trigger": trigger_func, "watcher": watch_func},
            timeout=compaction_args.timeout + 5,
            ignore_exceptions=True
        )

        self._handle_start_stop_compaction_results(
            trigger_and_watcher_futures=results,
            allow_trigger_exceptions=True
        )

    def disrupt_start_stop_validation_compaction(self):
        """
        Start and stop a validation compaction on a non-system columnfamily.

        1. Query the target node for a non-system cf. If no cf is found,
        skip the nemesis.

        In parallel:
        2.1 Start a validation compaction using a REST API call (via a
        Remoter curl call on the target node).
        2.2. Watch for the validation compaction to show up in logs.
        3. Stop the running validation compaction by issuing a <nodetool
        stop SCRUB> command after the watcher picks up the compaction
        started.

        The REST API call starting the compaction may not return, since
        it returns on completing the compaction task (which we attempt
        to interrupt). We ignore such timeouts and validate only the
        success of the command in (4).
        """
        compaction_args = self._prepare_start_stop_compaction()
        trigger_func = partial(compaction_args.compaction_ops.trigger_validation_compaction)
        watch_func = partial(compaction_args.compaction_ops.stop_on_user_compaction_logged,
                             node=compaction_args.target_node,
                             mark=compaction_args.target_node.mark_log(),
                             watch_for="Scrubbing ",
                             timeout=compaction_args.timeout,
                             stop_func=compaction_args.compaction_ops.stop_validation_compaction)

        results = ParallelObject.run_named_tasks_in_parallel(
            tasks={"trigger": trigger_func, "watcher": watch_func},
            timeout=compaction_args.timeout + 5,
            ignore_exceptions=True
        )

        self._handle_start_stop_compaction_results(
            trigger_and_watcher_futures=results,
            allow_trigger_exceptions=True
        )

    # This nemesis should be run with "private" ip_ssh_connections till the issue #665 is not fixed

    def disrupt_restart_then_repair_node(self):  # pylint: disable=invalid-name
        with DbEventsFilter(db_event=DatabaseLogEvent.DATABASE_ERROR,
                            line="Can't find a column family with UUID", node=self.target_node), \
            DbEventsFilter(db_event=DatabaseLogEvent.BACKTRACE,
                           line="Can't find a column family with UUID", node=self.target_node):
            self.target_node.restart()

        self.target_node.wait_node_fully_start(timeout=28800)  # 8 hours
        self.repair_nodetool_repair()

    def disrupt_resetlocalschema(self):  # pylint: disable=invalid-name
        rlocal_schema_res = self.target_node.follow_system_log(patterns=["schema_tables - Schema version changed to"])
        self.target_node.run_nodetool("resetlocalschema")

        assert wait_for(
            func=lambda: list(rlocal_schema_res),
            timeout=30,
            text="Waiting for schema version being recalculated",
            throw_exc=False,
        ), "Schema version has not been recalculated"

        # Check schema version on the nodes will be preformed after nemesis by ClusterHealthChecker
        # Waiting 60 sec: this time is defined by Tomasz
        self.log.debug("Sleep for 60 sec: the other nodes should pull new version")
        time.sleep(60)

    def disrupt_hard_reboot_node(self):
        self.reboot_node(target_node=self.target_node, hard=True)
        self.target_node.wait_node_fully_start()

    def disrupt_multiple_hard_reboot_node(self) -> None:
        cdc_expected_error_patterns = [
            "cdc - Could not update CDC description table with generation",
        ]

        # If this messages appeared, CDC successfully updated generation.
        cdc_success_msg_patterns = [
            "streams description table already updated",
            "CDC description table successfully updated with generation",
        ]

        num_of_reboots = random.randint(2, 10)
        InfoEvent(message=f'MultipleHardRebootNode {self.target_node}').publish()
        for i in range(num_of_reboots):
            self.log.debug("Rebooting %s out of %s times", i + 1, num_of_reboots)
            cdc_expected_error = self.target_node.follow_system_log(patterns=cdc_expected_error_patterns)
            cdc_success_msg = self.target_node.follow_system_log(patterns=cdc_success_msg_patterns)
            self.reboot_node(target_node=self.target_node, hard=True)
            if random.choice([True, False]):
                self.log.info('Waiting scylla services to start after node reboot')
                self.target_node.wait_db_up()
            else:
                self.log.info('Waiting JMX services to start after node reboot')
                self.target_node.wait_jmx_up()
            self.cluster.wait_for_nodes_up_and_normal(nodes=[self.target_node])
            found_cdc_error = list(cdc_expected_error)
            if found_cdc_error:
                # if cdc error message "cdc - Could not update CDC description..."
                # was found in log during reboot, but after that success messages:
                # "streams description updated" or "CDC desc table updated" were not
                # found in logs, raise the exception to fail the nemesis.
                self._check_for_cdc_success_messages(found_cdc_error, cdc_success_msg)

            sleep_time = random.randint(0, 100)
            self.log.info("Sleep %s seconds after hard reboot and service-up for node %s", sleep_time, self.target_node)
            time.sleep(sleep_time)

    @staticmethod
    @retrying(n=5, sleep_time=25, allowed_exceptions=(CdcStreamsWasNotUpdated,))
    def _check_for_cdc_success_messages(found_cdc_error, cdc_success_msg):
        found_success_info = list(cdc_success_msg)
        if not found_success_info:
            raise CdcStreamsWasNotUpdated(
                f"After '{found_cdc_error[0]}', messages '{' or '.join(cdc_success_msg)}' were not found")

    def disrupt_soft_reboot_node(self):
        self.reboot_node(target_node=self.target_node, hard=False)
        self.target_node.wait_node_fully_start()

    @decorate_with_context(ignore_ycsb_connection_refused)
    def disrupt_rolling_restart_cluster(self, random_order=False):
        self.cluster.restart_scylla(random_order=random_order)

    def disrupt_switch_between_password_authenticator_and_saslauthd_authenticator_and_back(self):
        """
        If prepare_saslauthd is enabled, saslauthd and ldap environment will be prepared for
        using SaslauthdAuthenticator. We have same account (cassandra) for SaslauthdAuthenticator
        and PasswordAuthenticator, so it can be toggled smoothly without effect to c-s workloads.

        It's only support to switch between PasswordAuthenticator and SaslauthdAuthenticator,
        the authenticator will be reset back in the end of nemesis.
        """
        if not self.cluster.params.get('prepare_saslauthd'):
            raise UnsupportedNemesis("SaslauthdAuthenticator can't work without saslauthd environment")
        if not self.cluster.nodes[0].is_enterprise:
            raise UnsupportedNemesis("SaslauthdAuthenticator is only supported by Scylla Enterprise")

        with self.target_node.remote_scylla_yaml() as scylla_yml:
            orig_auth = scylla_yml.authenticator
            if orig_auth == SASLAUTHD_AUTHENTICATOR:
                opposite_auth = 'PasswordAuthenticator'
            elif orig_auth == 'PasswordAuthenticator':
                opposite_auth = SASLAUTHD_AUTHENTICATOR
            else:
                raise UnsupportedNemesis(
                    'This nemesis only supports to switch between SaslauthdAuthenticator and PasswordAuthenticator')
        update_authenticator(self.cluster.nodes, opposite_auth)
        try:
            # Run connect a new session after authenticator switch, and run a short workload
            self._prepare_test_table(ks='keyspace_for_authenticator_switch', table='standard1')
        finally:
            # Wait 2 mins to let the workloads run with new Authenticator,
            # then switch Authenticator back to original
            time.sleep(120)
            update_authenticator(self.cluster.nodes, orig_auth)
            # Run connect a new session after authenticator switch, drop the test keyspace
            with self.cluster.cql_connection_patient(self.target_node) as session:
                session.execute('DROP KEYSPACE keyspace_for_authenticator_switch')

    @decorate_with_context(ignore_ycsb_connection_refused)
    def disrupt_rolling_config_change_internode_compression(self):
        def get_internode_compression_new_value_randomly(current_compression):
            self.log.debug(f"Current compression is {current_compression}")
            values = ['dc', 'all', None]
            values_to_toggle = list(filter(lambda value: value != current_compression, values))
            return random.choice(values_to_toggle)

        if self._is_it_on_kubernetes():
            # NOTE: on K8S update of 'scylla.yaml' and 'cassandra-rackdc.properties' files is done
            #       via update of the single reused place and serial restart of Scylla pods.
            raise UnsupportedNemesis(
                "This logic will be covered by an operator functional test. Skipping.")
        with self.target_node.remote_scylla_yaml() as scylla_yaml:
            current = scylla_yaml.internode_compression
        new_value = get_internode_compression_new_value_randomly(current)
        for node in self.cluster.nodes:
            self.log.debug(f"Changing {node} inter node compression to {new_value}")
            with node.remote_scylla_yaml() as scylla_yaml:
                scylla_yaml.internode_compression = new_value
            self.log.info(f"Restarting node {node}")
            node.restart_scylla_server()

    @decorate_with_context(ignore_ycsb_connection_refused)
    def disrupt_restart_with_resharding(self):
        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis(
                "Not supported on K8S. "
                "Run 'disrupt_nodetool_flush_and_reshard_on_kubernetes' instead")

        # If tablets in use, skipping resharding since it is not supported.
        with self.cluster.cql_connection_patient(self.target_node) as session:
            if is_tablets_feature_enabled(session=session):
                if SkipPerIssues('https://github.com/scylladb/scylladb/issues/16739', params=self.tester.params):
                    raise UnsupportedNemesis('https://github.com/scylladb/scylladb/issues/16739')

        murmur3_partitioner_ignore_msb_bits = 15  # pylint: disable=invalid-name
        self.log.info(f'Restart node with resharding. New murmur3_partitioner_ignore_msb_bits value: '
                      f'{murmur3_partitioner_ignore_msb_bits}')
        self.target_node.restart_node_with_resharding(
            murmur3_partitioner_ignore_msb_bits=murmur3_partitioner_ignore_msb_bits)
        self.target_node.wait_node_fully_start()

        # Wait 5 minutes our before return back the default value
        self.log.debug(
            'Wait 5 minutes our before return murmur3_partitioner_ignore_msb_bits back the default value (12)')
        time.sleep(360)
        self.log.info('Set back murmur3_partitioner_ignore_msb_bits value to 12')
        self.target_node.restart_node_with_resharding()

    def replace_full_file_name_to_prefix(self, one_file, ks_cf_for_destroy):
        # The file name like: /var/lib/scylla/data/scylla_bench/test-f60e4f30c98f11e98d46000000000002/mc-220-big-Data.db
        # or /var/lib/scylla/data/scylla_bench/test-f60e4f30c98f11e98d46000000000002/mc-3g6x_0sic_4r1eo23d0mkrb3fs2l-big-Data.db
        # For corruption we need to remove all files that their names are started from "mc-220-" or "mc-3g6x_0sic_4r1eo23d0mkrb3fs2l-"
        # (MC format)
        # Old format: "system-truncated-ka-" (system-truncated-ka-7-Data.db)
        # Search for these prefixes
        file_name = os.path.basename(one_file)

        try:
            file_name_template = re.search(r"([^-]+-[^-]+)-", file_name).group(1)
        except Exception as error:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.debug('File name "{file_name}" is not as expected for Scylla data files. '
                           'Search files for "{ks_cf_for_destroy}" table'.format(file_name=file_name,
                                                                                 ks_cf_for_destroy=ks_cf_for_destroy))
            self.log.debug('Error: {}'.format(error))
            return ""

        file_for_destroy = one_file.replace(file_name, file_name_template + '-*')
        self.log.debug('Selected files for destroy: {}'.format(file_for_destroy))
        return file_for_destroy

    @retrying(n=10, allowed_exceptions=(NoKeyspaceFound, NoFilesFoundToDestroy))
    def _choose_file_for_destroy(self, ks_cfs, return_one_file=True):
        file_for_destroy = ''
        all_files = []

        ks_cf_for_destroy = random.choice(ks_cfs)  # expected value as: 'keyspace1.standard1'

        ks_cf_for_destroy = ks_cf_for_destroy.replace('.', '/')
        files = self.target_node.remoter.sudo("find /var/lib/scylla/data/%s-* -maxdepth 1 -type f"
                                              % ks_cf_for_destroy, verbose=False)
        if files.stderr:
            raise NoFilesFoundToDestroy(
                'Failed to get data files for destroy in {}. Error: {}'.format(ks_cf_for_destroy,
                                                                               files.stderr))

        for one_file in files.stdout.split():
            if not one_file or '/' not in one_file:
                continue

            if not (file_for_destroy := self.replace_full_file_name_to_prefix(one_file, ks_cf_for_destroy)):
                continue

            self.log.debug('Selected files for destroy: {}'.format(file_for_destroy))
            if file_for_destroy:
                if return_one_file:
                    break
                all_files.append(file_for_destroy)

        if not file_for_destroy:
            raise NoFilesFoundToDestroy('Data file for destroy is not found in {}'.format(ks_cf_for_destroy))

        return file_for_destroy if return_one_file else all_files

    def get_all_sstables(self, tables: list[str], node: BaseNode = None):
        """
        :param tables: list of tables. Format of name: <keyspace_name.table_name>
        :param node: get SStables from the node
        """
        node = node or self.target_node

        sstables = []
        for ks_cf in tables:
            sstable_util = SstableUtils(db_node=node, ks_cf=ks_cf)
            ks_cf_sstables = sstable_util.get_sstables()
            sstables.extend(ks_cf_sstables)

        self.log.debug("All Sstables are: %s", sstables)

        return sstables

    @decorate_with_context([ignore_ycsb_connection_refused, ignore_raft_topology_cmd_failing])
    def _destroy_data_and_restart_scylla(self, keyspaces_for_destroy: list = None, sstables_to_destroy_perc: int = 50):  # pylint: disable=too-many-statements
        tables = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node, filter_empty_tables=False,
                                                        filter_by_keyspace=keyspaces_for_destroy)
        if not tables:
            raise UnsupportedNemesis(
                'Non-system keyspace and table are not found. The nemesis can\'t be run')

        self.log.debug("Chosen tables: %s", tables)

        # Stop scylla service before deleting sstables to avoid partial deletion of files that are under compaction
        self.target_node.stop_scylla_server(verify_up=False, verify_down=True)

        try:
            # Remove data files
            if not (all_files_to_destroy := self.get_all_sstables(tables=tables, node=self.target_node)):
                raise UnsupportedNemesis(
                    'SStables for destroy are not found. The nemesis can\'t be run')

            # How many SStables are going to be deleted
            sstables_amount_to_destroy = int(len(all_files_to_destroy) * sstables_to_destroy_perc / 100)
            self.log.debug("SStables amount to destroy (%s percent of all SStables): %s", sstables_to_destroy_perc,
                           sstables_amount_to_destroy)

            while sstables_amount_to_destroy > 0:
                file_for_destroy = random.choice(all_files_to_destroy)
                if not (file_group_for_destroy := self.replace_full_file_name_to_prefix(one_file=file_for_destroy,
                                                                                        ks_cf_for_destroy=tables)):
                    continue

                result = self.target_node.remoter.sudo('rm -f %s' % file_group_for_destroy)
                if result.stderr:
                    raise FilesNotCorrupted(
                        'Files were not removed. The nemesis can\'t be run. Error: {}'.format(result))
                all_files_to_destroy.remove(file_for_destroy)
                sstables_amount_to_destroy -= 1
                self.log.debug('Files {} were destroyed'.format(file_for_destroy))

        finally:
            self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    def disrupt(self):
        raise NotImplementedError('Derived classes must implement disrupt()')

    def get_disrupt_name(self):
        return self.current_disruption.split()[0]

    def get_class_name(self):
        return self.__class__.__name__.replace('Monkey', '')

    def set_current_disruption(self, label=None):
        self.log.debug('Set current_disruption -> %s', label)
        self.current_disruption = label

    def disrupt_destroy_data_then_repair(self):  # pylint: disable=invalid-name
        self._destroy_data_and_restart_scylla()
        # try to save the node
        self.repair_nodetool_repair()

    def disrupt_destroy_data_then_rebuild(self):  # pylint: disable=invalid-name
        self._destroy_data_and_restart_scylla()
        # try to save the node
        self.repair_nodetool_rebuild()

    @decorate_with_context(ignore_ycsb_connection_refused)
    def disrupt_nodetool_drain(self):
        result = self.target_node.run_nodetool("drain", timeout=15*60, coredump_on_timeout=True)
        self.target_node.run_nodetool("status", ignore_status=True, verbose=True,
                                      warning_event_on_exception=(Exception,))

        if result is not None:
            # workaround for issue #7332: don't interrupt test and don't raise exception
            # for UnexpectedExit, Failure and CommandTimedOut if "scylla-server stop" failed
            # or scylla-server was stopped gracefully.
            self.target_node.stop_scylla_server(verify_up=False, verify_down=True, ignore_status=True)
            self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    def disrupt_ldap_connection_toggle(self):
        if not self.cluster.params.get('use_ldap_authorization'):
            raise UnsupportedNemesis('Cluster is not configured to run with LDAP authorization, hence skipping')
        if not self.target_node.is_enterprise:
            raise UnsupportedNemesis('Cluster is not enterprise. LDAP is supported only for enterprise. Skipping')
        if not self.cluster.params.get('ldap_server_type') == LdapServerType.OPENLDAP:
            raise UnsupportedNemesis('This nemesis is supported only for open-Ldap. Skipping')

        self.log.info('Will now pause the LDAP container')
        ContainerManager.pause_container(self.tester.localhost, 'ldap')
        self.log.info('Will now sleep 180 seconds')
        time.sleep(180)
        self.log.info('Will now resume the LDAP container')
        ContainerManager.unpause_container(self.tester.localhost, 'ldap')
        self.log.info('finished with nemesis')

    def disrupt_disable_enable_ldap_authorization(self):
        if not self.cluster.params.get('use_ldap_authorization'):
            raise UnsupportedNemesis('Cluster is not configured to run with LDAP authorization, hence skipping')
        if not self.target_node.is_enterprise:
            raise UnsupportedNemesis('Cluster is not enterprise. LDAP is supported only for enterprise. Skipping')

        ldap_config = {'role_manager': '',
                       'ldap_url_template': '',
                       'ldap_attr_role': '',
                       'ldap_bind_dn': '',
                       'ldap_bind_passwd': ''}

        def remove_ldap_configuration_from_node(node):
            with node.remote_scylla_yaml() as scylla_yaml:
                for key in ldap_config:
                    ldap_config[key] = getattr(scylla_yaml, key)
                    setattr(scylla_yaml, key, None)
            node.restart_scylla_server()

        if not ContainerManager.is_running(self.tester.localhost, 'ldap'):
            raise LdapNotRunning("LDAP server was supposed to be running, but it is not")

        InfoEvent(message='Disable LDAP Authorization Configuration').publish()
        for node in self.cluster.nodes:
            remove_ldap_configuration_from_node(node)
        self.log.info('Will now pause the LDAP container')
        ContainerManager.pause_container(self.tester.localhost, 'ldap')

        self.log.debug('Will wait few minutes with LDAP disabled, before re-enabling it')
        time.sleep(600)

        def add_ldap_configuration_to_node(node):
            node.refresh_ip_address()
            with node.remote_scylla_yaml() as scylla_yaml:
                scylla_yaml.update(ldap_config)
            node.restart_scylla_server()

        self.log.info('Will now resume the LDAP container')
        ContainerManager.unpause_container(self.tester.localhost, 'ldap')
        for node in self.cluster.nodes:
            add_ldap_configuration_to_node(node)

        if not ContainerManager.is_running(self.tester.localhost, 'ldap'):
            raise LdapNotRunning("LDAP server was supposed to be running, but it is not")

    def _replace_cluster_node(self, old_node_ip: str | None = None, host_id: str | None = None,
                              timeout: int | float = MAX_TIME_WAIT_FOR_NEW_NODE_UP, rack=0, is_zero_node: bool = False) -> BaseNode:
        """When old_node_ip or host_id are not None then replacement node procedure is initiated"""
        # TODO: make it work on K8S when we have decommissioned (by nodetool) nodes.
        #       Now it will fail because pod which hosts decommissioned Scylla member is reported
        #       as 'NotReady' and will fail the pod waiter function.
        self.log.info("Adding new node to cluster...")
        InfoEvent(message='StartEvent - Adding new node to cluster').publish()
        if is_zero_node:
            new_node = skip_on_capacity_issues(self.cluster.add_nodes)(
                count=1, dc_idx=self.target_node.dc_idx, enable_auto_bootstrap=True, rack=rack, is_zero_node=is_zero_node)[0]
        else:
            new_node = skip_on_capacity_issues(self.cluster.add_nodes)(
                count=1, dc_idx=self.target_node.dc_idx, enable_auto_bootstrap=True, rack=rack)[0]
        self.monitoring_set.reconfigure_scylla_monitoring()
        self.set_current_running_nemesis(node=new_node)  # prevent to run nemesis on new node when running in parallel

        # since we need this logic before starting a node, and in `use_preinstalled_scylla: false` case
        # scylla is not yet installed or target node was terminated, we should use an alive node without nemesis for version,
        # it should be up and with scylla executable available
        verification_node = random.choice([
            node for node in self.cluster.nodes
            if node not in (new_node, self.target_node) and not node.running_nemesis])

        if verification_node.is_replacement_by_host_id_supported:
            new_node.replacement_host_id = host_id
        else:
            new_node.replacement_node_ip = old_node_ip
        try:
            with adaptive_timeout(Operations.NEW_NODE, node=self.cluster.data_nodes[0], timeout=timeout):
                self.cluster.wait_for_init(node_list=[new_node], timeout=timeout, check_node_health=False)
            self.cluster.clean_replacement_node_options(new_node)
            self.cluster.set_seeds()
            self.cluster.update_seed_provider()
        except (NodeSetupFailed, NodeSetupTimeout):
            self.log.warning("TestConfig of the '%s' failed, removing it from list of nodes" % new_node)
            self.cluster.nodes.remove(new_node)
            self.log.warning("Node will not be terminated. Please terminate manually!!!")
            raise
        self.cluster.wait_for_nodes_up_and_normal(nodes=[new_node])
        new_node.wait_node_fully_start()
        InfoEvent(message="FinishEvent - New Node is up and normal").publish()
        return new_node

    def _add_and_init_new_cluster_nodes(self, count, timeout=MAX_TIME_WAIT_FOR_NEW_NODE_UP, rack=None, instance_type: str = None, is_zero_node: bool = False) -> list[BaseNode]:
        if rack is None and self._is_it_on_kubernetes():
            rack = 0
        self.log.info("Adding %s new nodes to cluster...", count)
        InfoEvent(message=f'StartEvent - Adding {count} new nodes to cluster').publish()
        add_node_func_args = {"count": count,
                              "dc_idx": self.target_node.dc_idx,
                              "enable_auto_bootstrap": True,
                              "rack": rack,
                              "instance_type": instance_type
                              }
        if is_zero_node:
            instance_type = self.cluster.params.get("zero_token_instance_type_db") or instance_type
            add_node_func_args.update({"is_zero_node": is_zero_node, "instance_type": instance_type})

        new_nodes = skip_on_capacity_issues(self.cluster.add_nodes)(**add_node_func_args)
        self.monitoring_set.reconfigure_scylla_monitoring()
        for new_node in new_nodes:
            self.set_current_running_nemesis(node=new_node)
        try:
            with adaptive_timeout(Operations.NEW_NODE, node=self.cluster.data_nodes[0], timeout=timeout):
                self.cluster.wait_for_init(node_list=new_nodes, timeout=timeout, check_node_health=False)
            self.cluster.set_seeds()
            self.cluster.update_seed_provider()
        except (NodeSetupFailed, NodeSetupTimeout):
            self.log.warning("TestConfig of the '%s' failed, removing them from list of nodes" % new_nodes)
            for node in new_nodes:
                self.cluster.nodes.remove(node)
            self.log.warning("Nodes will not be terminated. Please terminate manually!!!")
            raise
        for new_node in new_nodes:
            new_node.wait_native_transport()
        self.cluster.wait_for_nodes_up_and_normal(nodes=new_nodes)
        for new_node in new_nodes:
            self.unset_current_running_nemesis(node=new_node)
        InfoEvent(message="FinishEvent - New Nodes are up and normal").publish()
        return new_nodes

    @decorate_with_context([ignore_ycsb_connection_refused, ignore_raft_topology_cmd_failing])
    def _terminate_cluster_node(self, node):
        self.cluster.terminate_node(node)
        self.monitoring_set.reconfigure_scylla_monitoring()

    def _nodetool_decommission(self, add_node=True):
        if self._is_it_on_kubernetes():
            self.unset_current_running_nemesis(self.target_node)
            self.set_target_node(allow_only_last_node_in_rack=True)

        target_is_seed = self.target_node.is_seed
        dc_topology_rf_change = self.cluster.decommission(self.target_node)
        new_node = None
        if add_node:
            # When adding node after decommission the node is declared as up only after it completed bootstrapping,
            # increasing the timeout for now
            new_node = self._add_and_init_new_cluster_nodes(count=1, rack=self.target_node.rack)[0]
            # after decomission and add_node, the left nodes have data that isn't part of their tokens anymore.
            # In order to eliminate cases that we miss a "data loss" bug because of it, we cleanup this data.
            # This fix important when just user profile is run in the test and "keyspace1" doesn't exist.
            if new_node.is_seed != target_is_seed:
                new_node.set_seed_flag(target_is_seed)
                self.cluster.update_seed_provider()
            if dc_topology_rf_change:
                dc_topology_rf_change.revert_to_original_keyspaces_rf(node_to_wait_for_balance=new_node)
            try:
                self.nodetool_cleanup_on_all_nodes_parallel()
            finally:
                self.unset_current_running_nemesis(new_node)
        return new_node

    def disrupt_nodetool_decommission(self, add_node=True):
        return self._nodetool_decommission(add_node=add_node)

    def disrupt_nodetool_seed_decommission(self, add_node=True):
        if len(self.cluster.seed_nodes) < 2:
            raise UnsupportedNemesis("To running seed decommission the cluster must contains at least 2 seed nodes")

        if not self.target_node.is_seed:
            self.target_node = random.choice(self.cluster.seed_nodes)
        self.target_node.set_seed_flag(False)
        self.cluster.update_seed_provider()

        new_seed_node = self._nodetool_decommission(add_node=add_node)
        if new_seed_node and not new_seed_node.is_seed:
            new_seed_node.set_seed_flag(True)
            self.cluster.update_seed_provider()

    @latency_calculator_decorator(legend="Terminate node and wait before adding new node")
    def _terminate_and_wait(self, target_node, sleep_time=300):
        self._terminate_cluster_node(target_node)
        time.sleep(sleep_time)  # Sleeping for 5 mins to let the cluster live with a missing node for a while

    @latency_calculator_decorator(legend="Replace a node in cluster with new one")
    def replace_node(self, old_node_ip: str, host_id: str, rack: int = 0, is_zero_node: bool = False) -> BaseNode:
        return self._replace_cluster_node(old_node_ip, host_id, rack=rack, is_zero_node=is_zero_node)

    def _verify_resharding_on_k8s(self, cpus, dc_idx):
        nodes_data = []
        for node in (n for n in self.cluster.nodes[::-1] if n.dc_idx == dc_idx):
            liveness_probe_failures = node.follow_system_log(
                patterns=["healthz probe: can't connect to JMX"])
            resharding_start = node.follow_system_log(patterns=[DB_LOG_PATTERN_RESHARDING_START])
            resharding_finish = node.follow_system_log(patterns=[DB_LOG_PATTERN_RESHARDING_FINISH])
            nodes_data.append((node, liveness_probe_failures, resharding_start, resharding_finish))

        self.log.info(
            "Update the cpu count to '%s' CPUs to make Scylla start "
            "the resharding process on all the nodes 1 by 1", cpus)
        # TODO: properly pick up the rack. For now it assumes we have only one.
        self.tester.db_cluster.replace_scylla_cluster_value(
            "/spec/datacenter/racks/0/resources", {
                "limits": {"cpu": cpus, "memory": self.cluster.k8s_clusters[dc_idx].scylla_memory_limit},
                "requests": {"cpu": cpus, "memory": self.cluster.k8s_clusters[dc_idx].scylla_memory_limit},
            }, dc_idx=dc_idx)

        # Wait for the start of the resharding.
        # In K8S it starts from the last node of a rack and then goes to previous ones.
        # One resharding with 100Gb+ may take about 3-4 minutes. So, set 5 minutes timeout per node.
        for node, liveness_probe_failures, resharding_start, resharding_finish in nodes_data:
            assert wait.wait_for(
                func=lambda: list(resharding_start),  # pylint: disable=cell-var-from-loop
                step=1, timeout=300, throw_exc=False,
                text=f"Waiting for the start of resharding on the '{node.name}' node.",
            ), f"Start of resharding hasn't been detected on the '{node.name}' node."
            resharding_started = time.time()
            self.log.debug("Resharding has been started on the '%s' node.", node.name)

            # Wait for the end of resharding
            assert wait.wait_for(
                func=lambda: list(resharding_finish),  # pylint: disable=cell-var-from-loop
                step=3, timeout=1800, throw_exc=False,
                text=f"Waiting for the finish of resharding on the '{node.name}' node.",
            ), f"Finish of the resharding hasn't been detected on the '{node.name}' node."
            self.log.debug("Resharding has been finished successfully on the '%s' node.", node.name)

            # Calculate the time spent for resharding. We need to have it be bigger than 2minutes
            # because it is the timeout of the liveness probe for Scylla pods.
            resharding_time = time.time() - resharding_started
            if resharding_time < 120:
                self.log.warning(
                    "Resharding was too fast - '%s's (<120s) on the '%s' node. "
                    "So, nemesis didn't cover the case.",
                    resharding_time, node.name)
            else:
                self.log.info(
                    "Resharding took '%s's on the '%s' node. It is enough to cover the case.",
                    resharding_time, node.name)

            # Check that liveness probe didn't report any errors
            # https://github.com/scylladb/scylla-operator/issues/894
            liveness_probe_failures_return = list(liveness_probe_failures)
            assert not liveness_probe_failures_return, (
                f"There are liveness probe failures: {liveness_probe_failures_return}")

        self.log.info("Resharding has successfully ended on whole Scylla cluster.")

    def disrupt_nodetool_flush_and_reshard_on_kubernetes(self):
        """Covers https://github.com/scylladb/scylla-operator/issues/894"""
        # NOTE: To check resharding we don't need to trigger it on all the nodes,
        #       so, pick up only one K8S cluster and only if it is EKS.
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('It is supported only on kubernetes')
        # If tablets in use, skipping resharding since it is not supported.
        with self.cluster.cql_connection_patient(self.target_node) as session:
            if is_tablets_feature_enabled(session=session):
                if SkipPerIssues('https://github.com/scylladb/scylladb/issues/16739', params=self.tester.params):
                    raise UnsupportedNemesis('https://github.com/scylladb/scylladb/issues/16739')

        dc_idx = 0
        for node in self.cluster.nodes:
            if hasattr(node.k8s_cluster, 'eks_cluster_version') and node.scylla_shards >= 7:
                dc_idx = node.dc_idx

                # Calculate new value for the CPU cores dedicated for Scylla pods
                current_cpus = convert_cpu_value_from_k8s_to_units(node.k8s_cluster.scylla_cpu_limit)
                new_cpus = convert_cpu_units_to_k8s_value(current_cpus + (1 if current_cpus <= 1 else -1))
                break
        else:
            # NOTE: bug https://github.com/scylladb/scylla-operator/issues/1077 reproduces better
            #       on slower machines with smaller amount number of cores.
            #       So, allow it to run only on fast K8S-EKS backend having at least 7 cores per pod.
            raise UnsupportedNemesis("https://github.com/scylladb/scylla-operator/issues/1077")

        # Run 'nodetool flush' command
        self.target_node.run_nodetool("flush -- keyspace1")

        try:
            # Change number of CPUs dedicated for Scylla pods
            # and make sure that the resharding process begins and finishes
            self._verify_resharding_on_k8s(new_cpus, dc_idx)
        finally:
            # Return the cpu count back and wait for the resharding begin and finish
            self._verify_resharding_on_k8s(current_cpus, dc_idx)

    def disrupt_drain_kubernetes_node_then_replace_scylla_node(self):
        self._disrupt_kubernetes_then_replace_scylla_node('drain_k8s_node')

    def disrupt_terminate_kubernetes_host_then_replace_scylla_node(self):
        if (not self.cluster.params.get("k8s_enable_sni")
                and SkipPerIssues('https://github.com/scylladb/scylla-operator/issues/1124', params=self.tester.params)):
            raise UnsupportedNemesis("https://github.com/scylladb/scylla-operator/issues/1124")
        self._disrupt_kubernetes_then_replace_scylla_node('terminate_k8s_host')  # pylint: disable=unreachable

    def _disrupt_kubernetes_then_replace_scylla_node(self, disruption_method):
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('Supported only on kubernetes')
        node = self.target_node
        InfoEvent(f'Running {disruption_method} on K8S node that hosts {node} scylla pod').publish()
        old_uid = node.k8s_pod_uid

        neighbour_scylla_pods = self._get_neighbour_scylla_pods(scylla_pod=node)

        self.log.info(
            "Running '%s' method on the '%s' K8S node that hosts '%s' target pod (uid=%s)"
            "and '%s' neighbour pods.",
            disruption_method, node.pod_spec.node_name, node, old_uid,
            [f"{neighbour_scylla_pod.metadata.namespace}/{neighbour_scylla_pod.metadata.name}"
             for neighbour_scylla_pod in neighbour_scylla_pods])
        getattr(node, disruption_method)()
        node.wait_till_k8s_pod_get_uid(ignore_uid=old_uid)
        old_uid = node.k8s_pod_uid

        self.log.info('Mark %s (uid=%s) to be replaced', node, old_uid)
        node.wait_for_svc()
        with DbEventsFilter(
                db_event=DatabaseLogEvent.DATABASE_ERROR,
                # NOTE: ignore following expected error messages:
                #       'init - Startup failed: seastar::sleep_aborted (Sleep is aborted)'
                #       'init - Startup failed: seastar::gate_closed_exception (gate closed)'
                line="init - Startup failed: seastar"):
            node.mark_to_be_replaced()
            self._kubernetes_wait_till_node_up_after_been_recreated(node, old_uid=old_uid)

        # NOTE: wait for all other neighbour pods become ready
        for neighbour_scylla_pod in neighbour_scylla_pods:
            KubernetesOps.wait_for_pod_readiness(
                kluster=node.k8s_cluster,
                pod_name=neighbour_scylla_pod.metadata.name,
                namespace=neighbour_scylla_pod.metadata.namespace,
                # TODO: calculate timeout based on the data size and load
                pod_readiness_timeout_minutes=30)

    def disrupt_drain_kubernetes_node_then_decommission_and_add_scylla_node(self):
        self._disrupt_kubernetes_then_decommission_and_add_scylla_node('drain_k8s_node')

    def disrupt_terminate_kubernetes_host_then_decommission_and_add_scylla_node(self):
        self._disrupt_kubernetes_then_decommission_and_add_scylla_node('terminate_k8s_host')

    def _disrupt_kubernetes_then_decommission_and_add_scylla_node(self, disruption_method):
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('Supported only on kubernetes')
        self.set_target_node(
            # NOTE: pick up a new target node only in the same region to reduce possible confusion
            #       during the debug process in case of a failure.
            dc_idx=self.target_node.dc_idx,
            rack=random.choice(list(self.cluster.racks)), allow_only_last_node_in_rack=True)
        node = self.target_node

        neighbour_scylla_pods = self._get_neighbour_scylla_pods(scylla_pod=node)
        InfoEvent(f'Running {disruption_method} on K8S node that hosts {self.target_node}').publish()
        self.log.info(
            "Running '%s' method on the '%s' K8S node that hosts '%s' target pod "
            "and '%s' neighbour pods.",
            disruption_method, node.pod_spec.node_name, node,
            [f"{neighbour_scylla_pod.metadata.namespace}/{neighbour_scylla_pod.metadata.name}"
             for neighbour_scylla_pod in neighbour_scylla_pods])
        getattr(node, disruption_method)()

        self.log.info('Decommission %s', node)
        dc_topology_rf_change = self.cluster.decommission(node, timeout=MAX_TIME_WAIT_FOR_DECOMMISSION)

        new_node = self.add_new_nodes(count=1, rack=node.rack)[0]
        if dc_topology_rf_change:
            dc_topology_rf_change.revert_to_original_keyspaces_rf(node_to_wait_for_balance=new_node)
        self.unset_current_running_nemesis(new_node)

        # NOTE: wait for all other neighbour pods become ready
        for neighbour_scylla_pod in neighbour_scylla_pods:
            KubernetesOps.wait_for_pod_readiness(
                kluster=node.k8s_cluster,
                pod_name=neighbour_scylla_pod.metadata.name,
                namespace=neighbour_scylla_pod.metadata.namespace,
                # TODO: calculate timeout based on the data size and load
                pod_readiness_timeout_minutes=30)

    def _get_neighbour_scylla_pods(self, scylla_pod):
        if self.tester.params.get('k8s_tenants_num') < 2:
            return []
        matched_pods = KubernetesOps.list_pods(
            scylla_pod.k8s_cluster, namespace=None,
            field_selector=f"spec.nodeName={scylla_pod.pod_spec.node_name}",
            label_selector="app.kubernetes.io/name=scylla")
        return [matched_pod
                for matched_pod in matched_pods
                if scylla_pod.name != matched_pod.metadata.name]

    def disrupt_replace_scylla_node_on_kubernetes(self):
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('OperatorNodeReplace is supported only on kubernetes')
        old_uid = self.target_node.k8s_pod_uid
        self.log.info('TerminateNode %s', self.target_node)
        self.log.info('Mark %s to be replaced', self.target_node)
        self.target_node.wait_for_svc()
        self.target_node.mark_to_be_replaced()
        self._kubernetes_wait_till_node_up_after_been_recreated(self.target_node, old_uid=old_uid)

    def _kubernetes_wait_till_node_up_after_been_recreated(self, node, old_uid=None):
        node.wait_till_k8s_pod_get_uid(ignore_uid=old_uid)
        self.log.info('Wait till %s is ready', node)
        node.wait_for_pod_readiness()

    @target_all_nodes
    def disrupt_terminate_and_replace_node(self):  # pylint: disable=invalid-name
        self._terminate_and_replace_node()

    def _terminate_and_replace_node(self):
        def get_node_state(node_ip: str) -> List["str"] | None:
            """Gets node state by IP address from nodetool status response"""
            status = self.cluster.get_nodetool_status()
            states = [val['state'] for dc in status.values() for ip, val in dc.items() if ip == node_ip]
            return states[0] if states else None

        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis(
                'Use "disrupt_terminate_kubernetes_host_then_replace_scylla_node" '
                'instead this one for K8S')
        # using "Replace a Dead Node" procedure from http://docs.scylladb.com/procedures/replace_dead_node/
        old_node_ip = self.target_node.ip_address
        host_id = self.target_node.host_id
        is_old_node_seed = self.target_node.is_seed
        InfoEvent(message='StartEvent - Terminate node and wait 5 minutes').publish()
        self._terminate_and_wait(target_node=self.target_node)
        assert get_node_state(old_node_ip) == "DN", "Removed node state should be DN"
        InfoEvent(message='FinishEvent - target_node was terminated').publish()
        new_node = self.replace_node(old_node_ip, host_id, rack=self.target_node.rack,
                                     is_zero_node=self.target_node._is_zero_token_node)
        try:
            if new_node.get_scylla_config_param("enable_repair_based_node_ops") == 'false':
                InfoEvent(message='StartEvent - Run repair on new node').publish()
                self.repair_nodetool_repair(new_node)
                InfoEvent(message='FinishEvent - Finished running repair on new node').publish()

            # wait until node gives up on the old node, the default timeout is `ring_delay_ms: 300000`
            # scylla: [shard 0] gossip - FatClient 10.0.22.115 has been silent for 30000ms, removing from gossip
            @retrying(n=20, sleep_time=20, allowed_exceptions=(AssertionError,))
            def wait_for_old_node_to_removed():
                state = get_node_state(old_node_ip)
                if old_node_ip == new_node.ip_address:
                    assert state == "UN", \
                        f"New node with the same IP as removed one should be in UN state but was: {state}"
                else:
                    assert state is None, \
                        f"Old node should have been removed from status but it wasn't. State was: {state}"

            wait_for_old_node_to_removed()

        finally:
            self.unset_current_running_nemesis(new_node)
            if is_old_node_seed:
                new_node.set_seed_flag(True)
                self.cluster.update_seed_provider()

    @decorate_with_context([ignore_ycsb_connection_refused, ignore_raft_topology_cmd_failing])
    @target_all_nodes
    def disrupt_kill_scylla(self):
        self._kill_scylla_daemon()

    def disrupt_no_corrupt_repair(self):

        if SkipPerIssues("https://github.com/scylladb/scylladb/issues/18059", self.tester.params):
            raise UnsupportedNemesis('Disabled due to https://github.com/scylladb/scylladb/issues/18059 not fixed yet')

        # prepare test tables and fill test data
        for i in range(10):
            self.log.debug('Prepare test tables if they do not exist')
            self._prepare_test_table(ks=f'drop_table_during_repair_ks_{i}', table='standard1')
            self.cluster.wait_for_schema_agreement()

        self.log.debug("Start repair target_node in background")
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='NodeToolRepairThread') as thread_pool:
            thread = thread_pool.submit(self.repair_nodetool_repair)
            try:
                # drop test tables one by one during repair
                for i in range(10):
                    time.sleep(random.randint(0, 300))
                    with self.cluster.cql_connection_patient(self.target_node, connect_timeout=600) as session:
                        session.execute(SimpleStatement(
                            f'DROP TABLE drop_table_during_repair_ks_{i}.standard1'), timeout=300)
            finally:
                thread.result()

    def _major_compaction(self):
        with adaptive_timeout(Operations.MAJOR_COMPACT, self.target_node, timeout=8000):
            self.target_node.run_nodetool("compact")

    def disrupt_major_compaction(self):
        self._major_compaction()

    @target_data_nodes
    def disrupt_load_and_stream(self):
        # Checking the columns number of keyspace1.standard1
        self.log.debug('Prepare keyspace1.standard1 if it does not exist')
        self._prepare_test_table(ks='keyspace1', table='standard1')
        column_num = SstableLoadUtils.calculate_columns_count_in_table(self.target_node)

        # Run load-and-stream test on regular standard1 table of cassandra-stress.
        if column_num < 5:
            raise UnsupportedNemesis("Schema doesn't match the snapshot, not uploading")

        test_data = SstableLoadUtils.get_load_test_data_inventory(column_num, big_sstable=False, load_and_stream=True)

        result = self.target_node.run_nodetool(sub_cmd="cfstats", args="keyspace1.standard1")

        if result is not None and result.exit_status == 0:
            map_files_to_node = SstableLoadUtils.distribute_test_files_to_cluster_nodes(nodes=self.cluster.data_nodes,
                                                                                        test_data=test_data)
            for sstables_info, load_on_node in map_files_to_node:
                SstableLoadUtils.upload_sstables(load_on_node, test_data=sstables_info, table_name="standard1")
                # NOTE: on K8S logs may appear with a delay, so add a bigger timeout for it.
                #       See https://github.com/scylladb/scylla-cluster-tests/issues/6314
                kwargs = {"start_timeout": 1800, "end_timeout": 1800} if self._is_it_on_kubernetes() else {}
                SstableLoadUtils.run_load_and_stream(load_on_node, **kwargs)

    # pylint: disable=too-many-statements
    @target_data_nodes
    def disrupt_nodetool_refresh(self, big_sstable: bool = False):
        # Checking the columns number of keyspace1.standard1
        self.log.debug('Prepare keyspace1.standard1 if it does not exist')
        self._prepare_test_table(ks='keyspace1', table='standard1')
        column_num = SstableLoadUtils.calculate_columns_count_in_table(self.target_node)

        # Note: when issue #6617 is fixed, we can try to load snapshot (cols=5) to a table (1 < cols < 5),
        #       expect that refresh will fail (no serious db error).
        if 1 < column_num < 5:
            raise UnsupportedNemesis("Schema doesn't match the snapshot, not uploading")

        test_data = SstableLoadUtils.get_load_test_data_inventory(column_num, big_sstable=big_sstable,
                                                                  load_and_stream=False)

        result = self.target_node.run_nodetool(sub_cmd="cfstats", args="keyspace1.standard1")

        if result is not None and result.exit_status == 0:
            key = '0x32373131364f334f3830'
            # Check one special key before refresh, we will verify refresh by query in the end
            # Note: we can't DELETE the key before refresh, otherwise the old sstable won't be loaded
            #       TRUNCATE can be used the clean the table, but we can't do it for keyspace1.standard1
            query_verify = f"SELECT * FROM keyspace1.standard1 WHERE key={key}"
            result = self.target_node.run_cqlsh(query_verify)
            if '(0 rows)' in result.stdout:
                self.log.debug('Key %s does not exist before refresh', key)
            else:
                self.log.debug('Key %s already exists before refresh', key)

            # Executing rolling refresh one by one
            shards_num = self.cluster.data_nodes[0].scylla_shards
            for node in self.cluster.data_nodes:
                SstableLoadUtils.upload_sstables(node, test_data=test_data[0], table_name="standard1",
                                                 is_cloud_cluster=self.cluster.params.get("db_type") == 'cloud_scylla')
                system_log_follower = SstableLoadUtils.run_refresh(node, test_data=test_data[0])
                # NOTE: resharding happens only if we have more than 1 core.
                #       We may have 1 core in a K8S multitenant setup.
                # If tablets in use, skipping resharding validation since it doesn't work the same as vnodes
                with self.cluster.cql_connection_patient(self.cluster.data_nodes[0]) as session:
                    if shards_num > 1 and not is_tablets_feature_enabled(session=session):
                        SstableLoadUtils.validate_resharding_after_refresh(
                            node=node, system_log_follower=system_log_follower)

            # Verify that the special key is loaded by SELECT query
            result = self.target_node.run_cqlsh(query_verify)
            assert '(1 rows)' in result.stdout, f'The key {key} is not loaded by `nodetool refresh`'

    def _k8s_fake_enospc_error(self, node):  # pylint: disable=no-self-use
        """Fakes ENOSPC error for scylla container (for /var/lib/scylla dir) using chaos-mesh without filling up disk."""

        if SkipPerIssues("https://github.com/scylladb/scylla-cluster-tests/issues/6327", params=self.tester.params):
            raise UnsupportedNemesis("https://github.com/scylladb/scylla-cluster-tests/issues/6327")

        if not node.k8s_cluster.chaos_mesh.initialized:  # pylint: disable=unreachable
            raise UnsupportedNemesis(
                "Chaos Mesh is not installed. Set 'k8s_use_chaos_mesh' config option to 'true'")
        no_space_errors_in_log = node.follow_system_log(patterns=['No space left on device'])
        try:
            experiment = IOFaultChaosExperiment(node, duration="300s", error=DiskError.NO_SPACE_LEFT_ON_DEVICE, error_probability=100,
                                                methods=["write", "flush"],
                                                volume_path="/var/lib/scylla")
            experiment.start()
            experiment.wait_until_finished()
            # wait some time before restarting to prevent supervisorctl error.
            time.sleep(30)
        finally:
            no_space_errors = list(no_space_errors_in_log)
            node.restart_scylla_server(verify_up_after=True)
            assert no_space_errors, "There are no 'No space left on device' errors in db log during enospc disruption."

    @target_data_nodes
    def disrupt_nodetool_enospc(self, sleep_time=30, all_nodes=False):

        if all_nodes:
            nodes = self.cluster.data_nodes
            InfoEvent('Enospc test on {}'.format([n.name for n in nodes])).publish()
        else:
            nodes = [self.target_node]

        for node in nodes:
            with ignore_no_space_errors(node=node):
                if self._is_it_on_kubernetes():
                    self._k8s_fake_enospc_error(node)
                else:
                    result = node.remoter.run('cat /proc/mounts')
                    if '/var/lib/scylla' not in result.stdout:
                        self.log.error("Scylla doesn't use an individual storage, skip enospc test")
                        continue

                    try:
                        reach_enospc_on_node(target_node=node)
                    finally:
                        clean_enospc_on_node(target_node=node, sleep_time=sleep_time)

    @target_data_nodes
    def disrupt_end_of_quota_nemesis(self, sleep_time=30):
        """
        Nemesis flow
        ---------------------
        1. Enable quota on the node.
        2. Check disk usage.
        3. Define quota size same as disk usage + 3GB.
        4. Approach end of quota in a loop with fallocate file that takes 90% of quota size each iteration.
        5. Wait for end of quota message to appear in the log - "Disk quota exceeded" / Except I/O Error if happens.
        6. Remove the sparse files.
        7. Remove the quota by setting it to 0.
        8. Restart scylla server.
        9. Verify scylla is up.
        """
        # Temporary disable due to https://github.com/scylladb/scylla-enterprise/issues/3736
        if SkipPerIssues('https://github.com/scylladb/scylla-enterprise/issues/3736', self.tester.params):
            raise UnsupportedNemesis('Disabled due to https://github.com/scylladb/scylla-enterprise/issues/3736')

        node = self.target_node
        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis("Skipping nemesis for kubernetes")

        result = node.remoter.run('cat /proc/mounts')
        if '/var/lib/scylla' not in result.stdout:
            raise UnsupportedNemesis("Scylla doesn't use an individual storage, skip end of quota test")

        enable_quota_on_node(node)
        quota_enabled = is_quota_enabled_on_node(node)
        if not quota_enabled:
            raise QuotaConfigurationFailure("Failed to configure quota on the cluster")

        with ignore_disk_quota_exceeded_errors(node):
            with configure_quota_on_node_for_scylla_user_context(node) as quota_size:
                write_data_to_reach_end_of_quota(node, quota_size)
            LOGGER.debug('Sleep 15 seconds before restart scylla-server')
            time.sleep(15)
            node.restart_scylla_server()
            node.wait_db_up()

    def disrupt_remove_service_level_while_load(self):
        # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
        if not self.cluster.params.get('sla'):
            raise UnsupportedNemesis("SLA nemesis can be run during SLA test only")

        if not self.cluster.nodes[0].is_enterprise:
            raise UnsupportedNemesis("SLA feature is only supported by Scylla Enterprise")

        if not self.cluster.params.get('authenticator'):
            raise UnsupportedNemesis("SLA feature can't work without authenticator")

        if not getattr(self.tester, "roles", None):
            raise UnsupportedNemesis('This nemesis is supported with Service Level and role are pre-defined')

        role = self.tester.roles[0]

        with self.cluster.cql_connection_patient(node=self.cluster.nodes[0], user=DEFAULT_USER,
                                                 password=DEFAULT_USER_PASSWORD) as session:
            self.log.info("Drop service level %s", role.attached_service_level_name)
            removed_shares = role.attached_service_level.shares
            role.attached_service_level.session = session
            role.session = session
            try:
                role.attached_service_level.drop(if_exists=False)
                time.sleep(300)  # let load to run without service level for 5 minutes
            finally:
                role.attach_service_level(
                    ServiceLevel(session=session,
                                 name=SERVICE_LEVEL_NAME_TEMPLATE % (removed_shares, random.randint(0, 10)),
                                 shares=removed_shares).create())

    def _deprecated_disrupt_stop_start(self):
        # TODO: We don't support fully stopping the AMI instance anymore
        # TODO: This nemesis has to be rewritten to just stop/start scylla server
        self.log.info('StopStart %s', self.target_node)
        self.target_node.restart()

    def call_random_disrupt_method(self, disrupt_methods=None, predefined_sequence=False):
        # pylint: disable=too-many-branches

        if disrupt_methods is None:
            disrupt_methods = [attr[1] for attr in inspect.getmembers(self) if
                               attr[0].startswith('disrupt_') and
                               callable(attr[1])]
        else:
            disrupt_methods = [attr[1] for attr in inspect.getmembers(self) if
                               attr[0] in disrupt_methods and
                               callable(attr[1])]
        if not disrupt_methods:
            self.log.warning("No monkey to run")
            return
        if not predefined_sequence:
            disrupt_method = random.choice(disrupt_methods)
        else:
            if not self._random_sequence:
                # Generate random sequence, every method has same chance to be called.
                # Here we use multiple original methods list, it will increase the chance
                # to call same method continuously.
                #
                # Adjust the rate according to the test duration. Try to call more unique
                # methods and don't wait to long time to meet the balance if the test
                # duration is short.
                test_duration = self.cluster.params.get('test_duration')
                if test_duration < 600:  # less than 10 hours
                    rate = 1
                elif test_duration < 4320:  # less than 3 days
                    rate = 2
                else:
                    rate = 3
                multiple_disrupt_methods = disrupt_methods * rate
                random.shuffle(multiple_disrupt_methods)
                self._random_sequence = multiple_disrupt_methods
            # consume the random sequence
            disrupt_method = self._random_sequence.pop()

        self.execute_disrupt_method(disrupt_method)

    def execute_disrupt_method(self, disrupt_method):
        disrupt_method_name = disrupt_method.__name__.replace('disrupt_', '')
        self.metrics_srv.event_start(disrupt_method_name)
        try:
            disrupt_method()
        finally:
            self.metrics_srv.event_stop(disrupt_method_name)

    def build_list_of_disruptions_to_execute(self, nemesis_selector=None, nemesis_multiply_factor=1):
        """
        Builds the list of disruptions that should be excuted during a test.

        nemesis_selector: should be retrived from the test yaml by using the "nemesis_selector".
        Here it kept for future usages and unit testing ability.
        more about nemesis_selector behaviour in sct_config.py

        nemesis_multiply_factor: should be retrived from the test yaml by using the "nemesis_multiply_factor".
        Here it kept for future usages and unit testing ability.
        more about nemesis_selector behaviour in sct_config.py
        """
        nemesis_selector = nemesis_selector or self.nemesis_selector_list
        nemesis_multiply_factor = self.cluster.params.get('nemesis_multiply_factor') or nemesis_multiply_factor
        if nemesis_selector:
            subclasses = self.get_list_of_subclasses_by_property_name(
                list_of_properties_to_include=nemesis_selector)
            if subclasses:
                disruptions, _, _ = self.get_list_of_disrupt_methods(subclasses_list=subclasses)
            else:
                disruptions = []
        else:
            disruptions = [attr[1] for attr in inspect.getmembers(self)
                           if attr[0].startswith('disrupt_') and callable(attr[1])]

        if nemesis_multiply_factor:
            disruptions = disruptions * nemesis_multiply_factor

        self.disruptions_list.extend(disruptions)
        return self.disruptions_list

    @property
    def nemesis_selector_list(self) -> list:
        if self._nemesis_selector_list:
            return self._nemesis_selector_list

        nemesis_selector = self.cluster.params.get('nemesis_selector') or []
        if self.cluster.params.get('nemesis_exclude_disabled'):
            nemesis_selector.append('!disabled')
        self._nemesis_selector_list = nemesis_selector
        return self._nemesis_selector_list

    @nemesis_selector_list.setter
    def nemesis_selector_list(self, value: list):
        self._nemesis_selector_list = value
        if value and self.cluster.params.get('nemesis_exclude_disabled'):
            self._nemesis_selector_list.append('!disabled')

    @property
    def _disruption_list_names(self):
        return [nemesis.__name__ for nemesis in self.disruptions_list]

    def shuffle_list_of_disruptions(self):
        if self.cluster.params.get('nemesis_seed'):
            nemesis_seed = self.cluster.params.get('nemesis_seed')
        else:
            nemesis_seed = random.randint(0, 1000)
            self.log.info(f'nemesis_seed generated for this test is {nemesis_seed}')
        self.log.debug(f'nemesis_seed to be used is {nemesis_seed}')

        self.log.debug(f"nemesis stack BEFORE SHUFFLE is {self._disruption_list_names}")
        random.Random(nemesis_seed).shuffle(self.disruptions_list)
        self.log.info(f"List of Nemesis to execute: {self._disruption_list_names}")

    def call_next_nemesis(self):
        assert self.disruptions_list, "no nemesis were selected"
        if not self.disruptions_cycle:
            self.disruptions_cycle = itertools.cycle(self.disruptions_list)
        self.log.debug(f'Selecting the next nemesis out of stack {self._disruption_list_names[10:]}')
        self.execute_disrupt_method(disrupt_method=next(self.disruptions_cycle))

    @latency_calculator_decorator(legend="Run repair process with nodetool repair")
    def repair_nodetool_repair(self, node=None, publish_event=True):
        node = node if node else self.target_node
        with adaptive_timeout(Operations.REPAIR, node, timeout=HOUR_IN_SEC * 48):
            node.run_nodetool(sub_cmd="repair", publish_event=publish_event)

    def repair_nodetool_rebuild(self):
        with adaptive_timeout(Operations.REBUILD, self.target_node, timeout=HOUR_IN_SEC * 48):
            self.target_node.run_nodetool('rebuild', long_running=True, retry=0)

    def nodetool_cleanup_on_all_nodes_parallel(self):
        # Inner disrupt function for ParallelObject
        def _nodetool_cleanup(node):
            InfoEvent('NodetoolCleanupMonkey %s' % node).publish()
            with adaptive_timeout(Operations.CLEANUP, node, timeout=HOUR_IN_SEC * 48):
                node.run_nodetool(sub_cmd="cleanup", long_running=True, retry=0)

        parallel_objects = ParallelObject(self.cluster.nodes, num_workers=min(
            32, len(self.cluster.nodes)), timeout=HOUR_IN_SEC * 48)
        parallel_objects.run(_nodetool_cleanup)

    def disrupt_nodetool_cleanup(self):
        self.nodetool_cleanup_on_all_nodes_parallel()

    def _prepare_test_table(self, ks='keyspace1', table=None):
        ks_cfs = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node)
        table_exist = f'{ks}.{table}' in ks_cfs if table else True

        test_keyspaces = self.cluster.get_test_keyspaces()
        # if keyspace or table doesn't exist, create it by cassandra-stress
        if ks not in test_keyspaces or not table_exist:
            stress_cmd = "cassandra-stress write n=400000 cl=QUORUM -mode native cql3 " \
                         f"-schema 'replication(strategy=NetworkTopologyStrategy," \
                         f"replication_factor={self.tester.reliable_replication_factor})' -log interval=5"
            cs_thread = self.tester.run_stress_thread(
                stress_cmd=stress_cmd, keyspace_name=ks, stop_test_on_failure=False, round_robin=True)
            cs_thread.verify_results()
            self.stop_nemesis_on_stress_errors(cs_thread)

    def stop_nemesis_on_stress_errors(self, stress_thread: DockerBasedStressThread) -> None:
        # Some implementations of stress threads override logic of the base class method
        # DockerBasedStressThread.get_results() and filter out 'events' portion of a result (e.g. c-s stress thread).
        # To retrieve all results of a thread we need to call the base class method directly
        stress_results = super(stress_thread.__class__, stress_thread).get_results()
        node_errors = {}
        for node, _, event in stress_results:
            if event.errors:
                node_errors.setdefault(node.name, []).extend(event.errors)

        if len(node_errors) == len(stress_results):  # stop only if stress command failed on all loaders
            errors_str = ''.join(f"  on node '{node_name}': {errors}\n" for node_name, errors in node_errors.items())
            raise NemesisStressFailure(
                f"Aborting '{self.__class__.__name__}' nemesis as '{stress_thread.stress_cmd}' stress command failed "
                f"with the following errors:\n{errors_str}")

    @scylla_versions(("5.2.rc0", None), ("2023.1.rc0", None))
    def _truncate_cmd_timeout_suffix(self, truncate_timeout):  # pylint: disable=no-self-use
        # NOTE: 'self' is used by the 'scylla_versions' decorator
        return f' USING TIMEOUT {int(truncate_timeout)}s'

    @scylla_versions((None, "5.1"), (None, "2022.2"))
    def _truncate_cmd_timeout_suffix(self, truncate_timeout):  # pylint: disable=no-self-use
        # NOTE: 'self' is used by the 'scylla_versions' decorator
        return ''

    def disrupt_drop(self):
        keyspace_drop = 'ks_drop'
        table = 'standard1'

        self._prepare_test_table(ks=keyspace_drop)

        # do the actual drop
        with self.cluster.cql_connection_patient(self.target_node, keyspace=keyspace_drop) as session:
            session.execute(f"DROP TABLE {table};")

    def disrupt_drop_perf(self):
        sleep_time_between_ops = self.cluster.params.get('nemesis_sequence_sleep_between_ops')
        if not self.has_steady_run and sleep_time_between_ops:
            self.steady_state_latency()
            self.has_steady_run = True

        latency_calculator_decorator(legend="Drop Table")(self.disrupt_drop)()

    def disrupt_truncate(self):
        keyspace_truncate = 'ks_truncate'
        table = 'standard1'

        self._prepare_test_table(ks=keyspace_truncate)

        # In order to workaround issue #4924 when truncate timeouts, we try to flush before truncate.
        with adaptive_timeout(Operations.FLUSH, self.target_node, timeout=HOUR_IN_SEC * 2):
            self.target_node.run_nodetool("flush")
        # do the actual truncation
        truncate_timeout = 600
        truncate_cmd_timeout_suffix = self._truncate_cmd_timeout_suffix(truncate_timeout)
        self.target_node.run_cqlsh(
            cmd=f'TRUNCATE {keyspace_truncate}.{table}{truncate_cmd_timeout_suffix}',
            timeout=truncate_timeout)

    def disrupt_truncate_perf(self):
        sleep_time_between_ops = self.cluster.params.get('nemesis_sequence_sleep_between_ops')
        if not self.has_steady_run and sleep_time_between_ops:
            self.steady_state_latency()
            self.has_steady_run = True

        latency_calculator_decorator(legend="Truncate Table")(self.disrupt_truncate)()

    def disrupt_truncate_large_partition(self):
        """
        Introduced a new truncate nemesis, it truncates a large-partition table,
        it's used to cover one improvement of compaction.
        The increase frequency of checking abortion is very useful for truncate.
        """
        if (SkipPerIssues(issues="https://github.com/scylladb/scylladb/issues/20356",
                          params=self.tester.params)
                and self.tester.params.get("use_zero_nodes")):
            raise UnsupportedNemesis("Unsupported nemesis due to scylladb/scylladb#20356")
        ks_name = 'ks_truncate_large_partition'
        table = 'test_table'
        stress_cmd = "scylla-bench -workload=sequential -mode=write -replication-factor=3 -partition-count=10 " + \
                     "-clustering-row-count=5555 -clustering-row-size=uniform:10..20 -concurrency=10 " + \
                     "-connection-count=10 -consistency-level=quorum -rows-per-request=10 -timeout=60s " + \
                     f"-keyspace {ks_name} -table {table}"
        bench_thread = self.tester.run_stress_thread(
            stress_cmd=stress_cmd, stop_test_on_failure=False)
        self.tester.verify_stress_thread(bench_thread)
        self.stop_nemesis_on_stress_errors(bench_thread)

        # In order to workaround issue #4924 when truncate timeouts, we try to flush before truncate.
        with adaptive_timeout(Operations.FLUSH, self.target_node, timeout=HOUR_IN_SEC * 2):
            self.target_node.run_nodetool("flush")
        # do the actual truncation
        truncate_timeout = 600
        truncate_cmd_timeout_suffix = self._truncate_cmd_timeout_suffix(truncate_timeout)
        self.target_node.run_cqlsh(
            cmd=f'TRUNCATE {ks_name}.{table}{truncate_cmd_timeout_suffix}',
            timeout=truncate_timeout)

    def _modify_table_property(self, name, val, filter_out_table_with_counter=False, keyspace_table=None):
        disruption_name = "".join([p.strip().capitalize() for p in name.split("_")])
        InfoEvent('ModifyTableProperties%s %s' % (disruption_name, self.target_node)).publish()

        if not keyspace_table:
            self.use_nemesis_seed()

            ks_cfs = self.cluster.get_non_system_ks_cf_list(
                db_node=self.target_node, filter_out_table_with_counter=filter_out_table_with_counter,
                filter_out_mv=True)  # not allowed to modify MV

            keyspace_table = random.choice(ks_cfs) if ks_cfs else ks_cfs

        if not keyspace_table:
            raise UnsupportedNemesis(
                'Non-system keyspace and table are not found. ModifyTableProperties nemesis can\'t be run')

        cmd = "ALTER TABLE {keyspace_table} WITH {name} = {val};".format(
            keyspace_table=keyspace_table, name=name, val=val)
        self.log.debug('_modify_table_property: %s', cmd)
        with self.cluster.cql_connection_patient(self.target_node) as session:
            session.execute(cmd)

    def _get_all_tables_with_no_compact_storage(self, tables_to_skip=None):
        """
        Return all tables with no "COMPACT STORAGE" in table code
        :param tables_to_skip: Dict of keyspaces/tables to be excluded from results.
            Examples:
                {'ks': 'table'} - will skip ks.table
                {'ks': '*'} - will skip any tables from keyspace "ks"
                {'*': 'table1,table2'} - will skip table1 and table2 from any keyspace
        :return: Dict of with keyspaces as key and list of table names as value
        """
        keyspaces = []
        output = {}
        if tables_to_skip is None:
            tables_to_skip = {}
        to_be_skipped_default = tables_to_skip.get('*', '').split(',')
        with self.cluster.cql_connection_patient(self.target_node) as session:
            query_result = session.execute('SELECT keyspace_name FROM system_schema.keyspaces;')
            for result_rows in query_result:
                keyspaces.extend([row.lower()
                                 for row in result_rows if not row.lower().startswith(("system", "audit"))])
            for ks in keyspaces:
                to_be_skipped = tables_to_skip.get(ks, None)
                if to_be_skipped is None:
                    to_be_skipped = to_be_skipped_default
                elif to_be_skipped == '*':
                    continue
                elif to_be_skipped == '':
                    to_be_skipped = []
                else:
                    to_be_skipped = to_be_skipped.split(',') + to_be_skipped_default
                tables = get_db_tables(session=session,
                                       keyspace_name=ks,
                                       node=self.target_node,
                                       with_compact_storage=False)
                if to_be_skipped:
                    tables = [table for table in tables if table not in to_be_skipped]
                if not tables:
                    continue
                output[ks] = tables
        return output

    def _add_drop_column_get_target_table(self, stored_target_table: list):
        current_tables = self._get_all_tables_with_no_compact_storage(self._add_drop_column_tables_to_ignore)
        if stored_target_table:
            if stored_target_table[1] in current_tables.get(stored_target_table[0], []):
                return stored_target_table
        if not current_tables:
            return None
        ks_name = next(iter(current_tables.keys()))
        table_name = current_tables[ks_name][0]
        return [ks_name, table_name]

    @staticmethod
    def _add_drop_column_get_added_columns_info(target_table: list, added_fields):
        ks = added_fields.get(target_table[0], None)
        if ks is None:
            output = {'column_names': {}, 'column_types': {}}
            added_fields[target_table[0]] = {target_table[1]: output}
            return output
        table = ks.get(target_table[1], None)
        if table is not None:
            return table
        ks[target_table[1]] = output = {'column_names': {}, 'column_types': {}}
        return output

    @staticmethod
    def _random_column_name(avoid_names=None, max_name_size=5):
        if avoid_names is None:
            avoid_names = []
        while True:
            column_name = generate_random_string(max_name_size)
            if column_name not in avoid_names:
                break
        return column_name

    def _add_drop_column_generate_columns_to_drop(self, added_columns_info):  # pylint: disable=too-many-branches
        drop = []
        columns_to_drop = min(len(added_columns_info['column_names']) + 1, self._add_drop_column_max_per_drop + 1)
        if columns_to_drop > 1:
            columns_to_drop = random.randrange(1, columns_to_drop)
        for _ in range(columns_to_drop):
            choice = [n for n in added_columns_info['column_names'] if n not in drop]
            if choice:
                column_name = random.choice(choice)
                drop.append(column_name)
        return drop

    def _add_drop_column_run_cql_query(self, cmd, ks,
                                       consistency_level=ConsistencyLevel.ALL):  # pylint: disable=too-many-branches
        try:
            with self.cluster.cql_connection_patient(self.target_node, keyspace=ks) as session:
                session.default_consistency_level = consistency_level
                session.execute(cmd)
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.debug(f"Add/Remove Column Nemesis: CQL query '{cmd}' execution has failed with error '{str(exc)}'")
            return False
        return True

    def _add_drop_column_generate_columns_to_add(self, added_columns_info):
        add = []
        columns_to_add = min(
            self._add_drop_column_max_columns - len(added_columns_info['column_names']),
            self._add_drop_column_max_per_add
        )
        if columns_to_add > 1:
            columns_to_add = random.randrange(1, columns_to_add)
        for _ in range(columns_to_add):
            new_column_name = self._random_column_name(added_columns_info['column_names'].keys(),
                                                       self._add_drop_column_max_column_name_size)
            new_column_type = CQLTypeBuilder.get_random(added_columns_info['column_types'], allow_levels=10,
                                                        avoid_types=['counter'], forget_on_exhaust=True)
            if new_column_type is None:
                continue
            add.append([new_column_name, new_column_type])
        return add

    def _add_drop_column(self, drop=True, add=True):  # pylint: disable=too-many-branches
        self._add_drop_column_target_table = self._add_drop_column_get_target_table(
            self._add_drop_column_target_table)
        if self._add_drop_column_target_table is None:
            return
        added_columns_info = self._add_drop_column_get_added_columns_info(self._add_drop_column_target_table,
                                                                          self._add_drop_column_columns_info)
        added_columns_info.get('column_names', None)
        if not added_columns_info['column_names']:
            drop = False
        if drop:
            drop = self._add_drop_column_generate_columns_to_drop(added_columns_info)
        if add:
            add = self._add_drop_column_generate_columns_to_add(added_columns_info)
        if not add and not drop:
            return
        # TBD: Scylla does not support DROP and ADD in the same statement
        if drop:
            cmd = f"ALTER TABLE {self._add_drop_column_target_table[1]} DROP ( {', '.join(drop)} );"
            if self._add_drop_column_run_cql_query(cmd, self._add_drop_column_target_table[0]):
                for column_name in drop:
                    column_type = added_columns_info['column_names'][column_name]
                    del added_columns_info['column_names'][column_name]
        if add:
            cmd = f"ALTER TABLE {self._add_drop_column_target_table[1]} " \
                  f"ADD ( {', '.join(['%s %s' % (col[0], col[1]) for col in add])} );"
            if self._add_drop_column_run_cql_query(cmd, self._add_drop_column_target_table[0]):
                for column_name, column_type in add:
                    added_columns_info['column_names'][column_name] = column_type
                    column_type.remember_variant(added_columns_info['column_types'])

    def _add_drop_column_run_in_cycle(self):
        start_time = time.time()
        end_time = start_time + 600
        while time.time() < end_time:
            self._add_drop_column()

    def verify_initial_inputs_for_delete_nemesis(self):
        test_keyspaces = self.cluster.get_test_keyspaces()

        if 'scylla_bench' not in test_keyspaces:
            raise UnsupportedNemesis("This nemesis can run on scylla_bench test only")

        if not (self.tester.partitions_attrs and self.tester.partitions_attrs.max_partitions_in_test_table):
            raise UnsupportedNemesis(
                'This nemesis expects "max_partitions_in_test_table" sub-parameter of data_validation to be set')

    # pylint: disable=too-many-locals
    def choose_partitions_for_delete(self, partitions_amount, ks_cf, with_clustering_key_data=False,
                                     exclude_partitions=None):
        """
        :type partitions_amount: int
        :type ks_cf: str
        :type with_clustering_key_data: bool
        :type exclude_partitions: list
        :return: defaultdict
        """
        if not exclude_partitions:
            exclude_partitions = []

        # In the large partition tests part of partitions are used for data validation. We do not want to delete a data in those partitions
        # to prevent scylla-bench command failure
        partitions_attrs = self.tester.partitions_attrs
        if partitions_attrs.partition_range_with_data_validation:
            partitions_amount = min(partitions_amount, partitions_attrs.non_validated_partitions)
            available_partitions_for_deletion = list(
                range(partitions_attrs.partition_end_range + 1, partitions_attrs.max_partitions_in_test_table))
        else:
            available_partitions_for_deletion = list(range(partitions_attrs.max_partitions_in_test_table))
        self.log.debug(f"Partitions amount for delete : {partitions_amount}")

        partitions_for_delete = defaultdict(list)
        with self.cluster.cql_connection_patient(self.target_node, connect_timeout=300) as session:
            session.default_consistency_level = ConsistencyLevel.ONE

            while available_partitions_for_deletion and len(partitions_for_delete) < partitions_amount:
                random_index = random.randint(0, len(available_partitions_for_deletion) - 1)
                partition_key = available_partitions_for_deletion.pop(random_index)

                if exclude_partitions and partition_key in exclude_partitions:
                    continue

                # Get the max cl value in the partition.
                cmd = f"select ck from {ks_cf} where pk={partition_key} order by ck desc limit 1"
                try:
                    result = session.execute(SimpleStatement(cmd, fetch_size=1), timeout=300)
                except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                    self.log.error(str(exc))
                    continue

                if not result:
                    continue

                if not with_clustering_key_data:
                    partitions_for_delete[partition_key] = []
                    continue

                # Suppose that min ck value is 0 in the partition
                partitions_for_delete[partition_key].extend([0, result[0].ck])

                if None in partitions_for_delete[partition_key]:
                    partitions_for_delete.pop(partition_key)

        self.log.debug(f'Partitions for delete: {partitions_for_delete}')
        return partitions_for_delete

    def get_random_timestamp_from_partition(self, ks_cf, pkey, partition_percentage=0.25) -> tuple[int, int]:
        """
            Get a write timestamp from a "pivot" row inside a single partition.
            partition_percentage controls where the "pivot" is, default is 25% of the partition size (total number of rows)
            Returns timestamp and the clustering key value as tuple
        """
        with self.cluster.cql_connection_patient(node=self.target_node) as session:
            number_of_rows = session.execute(
                SimpleStatement(f"select count(ck) from {ks_cf} where pk = {pkey}")).one().system_count_ck
            fetch_limit = max(math.ceil(number_of_rows * partition_percentage), 11)
            self.log.debug(
                "[%s_using_timestamp] Partition size: %s, fetching up to %s",
                self.get_disrupt_name(),
                number_of_rows,
                fetch_limit
            )
            partition = session.execute(SimpleStatement(
                f"select pk, ck from {ks_cf} where pk = {pkey} limit {fetch_limit}")).all()
            delete_mark = partition[-1].ck
            timestamp = session.execute(
                SimpleStatement(f"select writetime(v) from {ks_cf} where pk = {pkey} and ck = {delete_mark}")).one().writetime_v
            if not timestamp:
                message = f"Unable to get writetime for row (pk = {pkey}, ck = {delete_mark})"
                self.log.error(message)
                raise TimestampNotFound(message)

            return timestamp, delete_mark

    def run_deletions(self, queries, ks_cf):
        for cmd in queries:
            self.log.debug(f'delete query: {cmd}')
            with self.cluster.cql_connection_patient(self.target_node, connect_timeout=300) as session:
                session.execute(SimpleStatement(cmd, consistency_level=ConsistencyLevel.QUORUM), timeout=3600)

        self.target_node.run_nodetool('flush', args=ks_cf.replace('.', ' '))

    def _verify_using_timestamp_deletions(self, ks_cf: str, verification_queries: list[tuple[int, int, int]]):
        mv_not_configured = False
        mv_table_name = ".".join([ks_cf.split(sep=".")[0], "view_test"])
        with self.cluster.cql_connection_patient(self.target_node, connect_timeout=300) as session:
            # pylint: disable=invalid-name
            for pk, ck, ts in verification_queries:
                result = session.execute(SimpleStatement(
                    f"SELECT pk, ck, writetime(v) FROM {ks_cf} WHERE pk = {pk} AND ck = {ck}")).one()
                assert not result or result.writetime_v != ts, \
                    f"USING TIMESTAMP: deletion failed for ({pk}, {ck}), row still exists, timestamp used: {ts}"
                if not mv_not_configured:
                    try:
                        result = session.execute(
                            SimpleStatement(
                                f"SELECT pk, ck, writetime(v) FROM {mv_table_name} WHERE pk = {pk} AND ck = {ck}")).one()
                        assert not result or result.writetime_v != ts, f"USING TIMESTAMP: deletion failed for ({pk}, {ck}), " \
                            f"row still exists in MV (!!!), timestamp used: {ts}"
                    except InvalidRequest:
                        mv_not_configured = True

    def delete_half_partition(self, ks_cf):
        self.log.debug('Delete by range - half of partition')

        # Select half of partitions because we need available partitions in the next step: delete_range_in_few_partitions module
        partitions_amount = self.tester.partitions_attrs.non_validated_partitions / 2
        self.log.debug('delete_half_partition.partitions_amount: %s', partitions_amount)
        partitions_for_delete = self.choose_partitions_for_delete(partitions_amount=partitions_amount,
                                                                  ks_cf=ks_cf,
                                                                  with_clustering_key_data=True)
        if not partitions_for_delete:
            raise UnsupportedNemesis('Not found partitions for delete. Nemesis can not be run')

        queries = []
        for pkey, ckey in partitions_for_delete.items():
            queries.append(f"delete from {ks_cf} where pk = {pkey} and ck > {int(ckey[1] / 2)}")
        self.run_deletions(queries=queries, ks_cf=ks_cf)
        return partitions_for_delete

    def delete_by_range_using_timestamp(self, ks_cf: str):
        self.log.debug('Delete by range - using timestamp')

        partitions_for_delete = self.choose_partitions_for_delete(
            partitions_amount=self.tester.partitions_attrs.non_validated_partitions // self.num_deletions_factor, ks_cf=ks_cf,
            with_clustering_key_data=False)
        if not partitions_for_delete:
            message = "Unable to find partitions to delete"
            self.log.error(message)
            raise PartitionNotFound(message)

        queries = []
        verification_queries = []
        partition_percentage = random.randint(25, 75) / 100
        for pkey, _ in partitions_for_delete.items():
            self.log.debug("Using USING TIMESTAMP clause in the deletion for this partition: %s", pkey)
            timestamp, clustering_key = self.get_random_timestamp_from_partition(
                ks_cf=ks_cf, pkey=pkey, partition_percentage=partition_percentage)
            queries.append(f"delete from {ks_cf} using timestamp {timestamp} where pk = {pkey}")
            verification_queries.append([pkey, clustering_key, timestamp])

        self.run_deletions(queries=queries, ks_cf=ks_cf)
        self._verify_using_timestamp_deletions(ks_cf=ks_cf, verification_queries=verification_queries)

        return partitions_for_delete

    def delete_range_in_few_partitions(self, ks_cf, partitions_for_exclude_dict):
        self.log.debug('Delete same range in the few partitions')

        partitions_for_exclude = list(partitions_for_exclude_dict.keys())
        partitions_for_delete = self.choose_partitions_for_delete(
            partitions_amount=self.tester.partitions_attrs.non_validated_partitions // self.num_deletions_factor, ks_cf=ks_cf,
            with_clustering_key_data=True,
            exclude_partitions=partitions_for_exclude)
        if not partitions_for_delete:
            raise Exception("No partitions for deletion found. Cannot execute a range deletion.")

        # Choose same "ck" values that exists for all partitions
        # min_clustering_key - the biggest from min(ck) value for all selected partitions
        # max_clustering_key - the smallest from max(ck) value for all selected partitions
        min_clustering_key = max([v[0] for v in partitions_for_delete.values()])
        max_clustering_key = min([v[1] for v in partitions_for_delete.values()])
        clustering_keys = []
        if max_clustering_key > min_clustering_key:
            third_ck = int((max_clustering_key - min_clustering_key) / 3)
            clustering_keys = range(min_clustering_key + third_ck, max_clustering_key - third_ck)

        if not clustering_keys:
            clustering_keys = range(min_clustering_key, max_clustering_key)

        queries = []
        for pkey in partitions_for_delete.keys():
            queries.append(f"delete from {ks_cf} where pk = {pkey} and ck >= {clustering_keys[0]} "
                           f"and ck <= {clustering_keys[-1]}")

        self.run_deletions(queries=queries, ks_cf=ks_cf)

        return list(partitions_for_delete.keys()) + partitions_for_exclude

    def disrupt_delete_10_full_partitions(self):
        """
        Delete few partitions in the table with large partitions
        """
        self.verify_initial_inputs_for_delete_nemesis()

        ks_cf = 'scylla_bench.test'
        partitions_for_delete = self.choose_partitions_for_delete(10, ks_cf)

        if not partitions_for_delete:
            raise UnsupportedNemesis('Not found partitions for delete. Nemesis can not be run')

        queries = []
        for partition_key in partitions_for_delete.keys():
            queries.append(f"delete from {ks_cf} where pk = {partition_key}")

        self.run_deletions(queries=queries, ks_cf=ks_cf)

    def disrupt_delete_overlapping_row_ranges(self):
        """
        Delete several overlapping row ranges in the table with large partitions.
        """
        self.verify_initial_inputs_for_delete_nemesis()
        ks_cf = 'scylla_bench.test'
        partitions_for_delete = self.choose_partitions_for_delete(
            partitions_amount=self.tester.partitions_attrs.non_validated_partitions // self.num_deletions_factor, ks_cf=ks_cf,
            with_clustering_key_data=True)
        if not partitions_for_delete:
            self.log.error('No partitions for delete found!')
            raise UnsupportedNemesis("DeleteOverlappingRowRangesMonkey: No partitions for delete found!")

        self.log.debug('Delete random row ranges in few partitions')
        queries = []
        for pkey, ckey in partitions_for_delete.items():
            for _ in range(random.randint(3, 20)):  # Get a random number of ranges to delete.
                min_ck = random.randint(0, ckey[1])  # Generate a random range of rows to delete in a single partition.
                max_ck = random.randint(min_ck, ckey[1])
                queries.append(f"delete from {ks_cf} where pk = {pkey} and ck > {min_ck} and ck < {max_ck}")
        self.run_deletions(queries=queries, ks_cf=ks_cf)

    def disrupt_delete_by_rows_range(self):
        """
        Delete few partitions in the table with large partitions
        """
        self.verify_initial_inputs_for_delete_nemesis()

        ks_cf = 'scylla_bench.test'
        # Step-1: delete_half_partition or delete_by_range_using_timestamp
        if random.random() > 0.5:
            partitions_for_exclude = self.delete_half_partition(ks_cf)
        else:
            partitions_for_exclude = self.delete_by_range_using_timestamp(ks_cf)
        # Step-2: delete_range_in_few_partitions
        self.delete_range_in_few_partitions(ks_cf, partitions_for_exclude)

    def disrupt_add_drop_column(self):
        """
        It searches for a table that allow add/drop columns (non compact storage table)
        If there is no such table it draw an error and quit
        It keeps tracking what columns where added and never drops column that were added by someone else.
        """
        self.log.debug("AddDropColumnMonkey: Started")
        self._add_drop_column_target_table = self._add_drop_column_get_target_table(
            self._add_drop_column_target_table)
        if self._add_drop_column_target_table is None:
            raise UnsupportedNemesis("AddDropColumnMonkey: can't find table to run on")
        InfoEvent(f'AddDropColumnMonkey table {".".join(self._add_drop_column_target_table)}').publish()
        self._add_drop_column_run_in_cycle()

    def modify_table_comment(self):
        # default: comment = ''
        prop_val = generate_random_string(24)
        self._modify_table_property(name="comment", val=f"'{prop_val}'")

    def modify_table_gc_grace_time(self):
        """
            The number of seconds after data is marked with a tombstone (deletion marker)
            before it is eligible for garbage-collection.
            default: gc_grace_seconds = 864000
        """
        self._modify_table_property(name="gc_grace_seconds", val=random.randint(216000, 864000))

    def modify_table_caching(self):
        """
           Caching optimizes the use of cache memory by a table without manual tuning.
           Cassandra weighs the cached data by size and access frequency.
           default: caching = {'keys': 'ALL', 'rows_per_partition': 'ALL'}
        """
        prop_val = dict(
            keys=random.choice(["NONE", "ALL"]),
            rows_per_partition=random.choice(["NONE", "ALL", random.randint(1, 10000)])
        )
        self._modify_table_property(name="caching", val=str(prop_val))

    def modify_table_bloom_filter_fp_chance(self):  # pylint: disable=invalid-name
        """
            The Bloom filter sets the false-positive probability for SSTable Bloom filters.
            When a client requests data, Cassandra uses the Bloom filter to check if the row
            exists before doing disk I/O. Bloom filter property value ranges from 0 to 1.0.
            Lower Bloom filter property probabilities result in larger Bloom filters that use more memory.
            The effects of the minimum and maximum values:
                0: Enables the unmodified, effectively the largest possible, Bloom filter.
                1.0: Disables the Bloom filter.
            default: bloom_filter_fp_chance = 0.01
        """
        self._modify_table_property(name="bloom_filter_fp_chance", val=random.random() / 2)

    def toggle_table_gc_mode(self):
        """
            Alters a non-system table tombstone_gc_mode option.
            Choose the alternate option of 'repair' / 'timeout'
             (*) The other available values of 'disabled' / 'immediate' are not tested by
             this nemesis since not applicable to a longevity test.
        """
        if SkipPerIssues("https://github.com/scylladb/scylla-enterprise/issues/4082", self.tester.params):
            raise UnsupportedNemesis('Disabled due to https://github.com/scylladb/scylla-enterprise/issues/4082')

        # This nemesis can not be run on table with RF = 1:
        #   ConfigurationException: tombstone_gc option with mode = repair not supported for table with RF one or local replication strategy
        # We do not run tests with local strategy ({'class': 'org.apache.cassandra.locator.LocalStrategy'}), so I do not add this filter
        if not (all_ks_cfs := self.cluster.get_non_system_ks_cf_list(db_node=self.target_node,
                                                                     filter_func=self.cluster.is_ks_rf_one)):
            raise UnsupportedNemesis(
                'Any table with RF != 1 is not found. disrupt_toggle_table_gc_mode nemesis can\'t run')

        keyspace_table = random.choice(all_ks_cfs)
        keyspace, table = keyspace_table.split('.')
        if get_gc_mode(node=self.target_node, keyspace=keyspace, table=table) != GcMode.REPAIR:
            new_gc_mode = GcMode.REPAIR
        else:
            new_gc_mode = GcMode.TIMEOUT
        new_gc_mode_as_dict = {'mode': new_gc_mode.value}

        alter_command_prefix = 'ALTER TABLE ' if not is_cf_a_view(
            node=self.target_node, ks=keyspace, cf=table) else 'ALTER MATERIALIZED VIEW '
        cmd = alter_command_prefix + f" {keyspace_table} WITH tombstone_gc = {new_gc_mode_as_dict};"
        self.log.info("Alter GC mode query to execute: %s", cmd)
        self.target_node.run_cqlsh(cmd)

    def toggle_table_ics(self):  # pylint: disable=too-many-locals
        """
            Alters a non-system table compaction strategy from ICS to any-other and vise versa.
        """
        if SkipPerIssues("https://github.com/scylladb/scylla-enterprise/issues/4082", self.tester.params):
            raise UnsupportedNemesis('Disabled due to https://github.com/scylladb/scylla-enterprise/issues/4082')

        all_ks_cfs = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node)

        if not all_ks_cfs:
            raise UnsupportedNemesis(
                'Non-system keyspace and table are not found. toggle_tables_ics nemesis can\'t run')

        keyspace_table = random.choice(all_ks_cfs)
        keyspace, table = keyspace_table.split('.')
        cur_compaction_strategy = get_compaction_strategy(node=self.target_node, keyspace=keyspace,
                                                          table=table)
        if cur_compaction_strategy != CompactionStrategy.INCREMENTAL:
            new_compaction_strategy = CompactionStrategy.INCREMENTAL
        else:
            new_compaction_strategy = random.choice([strategy for strategy in list(
                CompactionStrategy) if strategy != CompactionStrategy.INCREMENTAL])
        new_compaction_strategy_as_dict = {'class': new_compaction_strategy.value}

        if new_compaction_strategy in [CompactionStrategy.INCREMENTAL, CompactionStrategy.SIZE_TIERED]:
            for param in get_compaction_random_additional_params(new_compaction_strategy):
                new_compaction_strategy_as_dict.update(param)
        alter_command_prefix = 'ALTER TABLE ' if not is_cf_a_view(
            node=self.target_node, ks=keyspace, cf=table) else 'ALTER MATERIALIZED VIEW '
        cmd = alter_command_prefix + \
            " {keyspace_table} WITH compaction = {new_compaction_strategy_as_dict};".format(**locals())
        self.log.debug("Toggle table ICS query to execute: {}".format(cmd))
        try:
            self.target_node.run_cqlsh(cmd)
        except (UnexpectedExit, Libssh2UnexpectedExit) as unexpected_exit:
            if "Unable to find compaction strategy" in str(unexpected_exit):
                err_msg = "for this nemesis to work, you need ICS supported scylla version."
                raise UnsupportedNemesis(err_msg) from unexpected_exit
            raise unexpected_exit

        self.cluster.wait_for_schema_agreement()

    def modify_table_compaction(self):
        """
            The compaction property defines the compaction strategy class for this table.
            default: compaction = {
                'class': 'SizeTieredCompactionStrategy'
                'bucket_high': 1.5,
                'bucket_low': 0.5,
                'min_sstable_size': 50,
                'min_threshold': 4,
                'max_threshold': 32,
            }
        """
        strategies = [
            {
                'class': 'SizeTieredCompactionStrategy',
                'bucket_high': 1.5,
                'bucket_low': 0.5,
                'min_sstable_size': 50,
                'min_threshold': 4,
                'max_threshold': 32,
            },
            {
                'class': 'LeveledCompactionStrategy',
                'sstable_size_in_mb': 160,
            },
            {
                'class': 'TimeWindowCompactionStrategy',
                'compaction_window_unit': 'DAYS',
                'compaction_window_size': 1,
                'expired_sstable_check_frequency_seconds': 600,
                'min_threshold': 4,
                'max_threshold': 32,
            },
        ]
        prop_val = random.choice(strategies)
        # max allowed TTL - 49 days (4300000) (to be compatible with default TWCS settings)
        if prop_val['class'] == 'TimeWindowCompactionStrategy':
            self._modify_table_property(name="default_time_to_live", val=str(4300000))

        self._modify_table_property(name="compaction", val=str(prop_val))

    def modify_table_compression(self):
        """
            The compression algorithm. Valid values are LZ4Compressor, SnappyCompressor, DeflateCompressor and
            ZstdCompressor
            default: compression = {}
        """
        algos = ("",  # no compression
                 "LZ4Compressor",
                 "SnappyCompressor",
                 "DeflateCompressor",
                 "ZstdCompressor")
        algo = random.choice(algos)
        prop_val = {"sstable_compression": algo}
        if algo:
            prop_val["chunk_length_kb"] = random.choice(["4K", "64KB", "128KB"])
            prop_val["crc_check_chance"] = random.random()
        self._modify_table_property(name="compression", val=str(prop_val))

    def modify_table_crc_check_chance(self):
        """
            default: crc_check_chance = 1.0
        """
        self._modify_table_property(name="crc_check_chance", val=random.random())

    def modify_table_dclocal_read_repair_chance(self):  # pylint: disable=invalid-name
        """
            The probability that a successful read operation triggers a read repair.
            Unlike the repair controlled by read_repair_chance, this repair is limited to
            replicas in the same DC as the coordinator. The value must be between 0 and 1
            default: dclocal_read_repair_chance = 0.1
        """
        self._modify_table_property(name="dclocal_read_repair_chance", val=random.choice([0, 0.2, 0.5, 0.9]))

    def modify_table_default_time_to_live(self):  # pylint: disable=invalid-name
        """
            The value of this property is a number of seconds. If it is set, Cassandra applies a
            default TTL marker to each column in the table, set to this value. When the table TTL
            is exceeded, Cassandra tombstones the table.
            This nemesis selects random table, check if it has TimeWindowCompactionStrategy applied
            and calculate possible default time to live, if no - sets random values in allowed range.
            default: default_time_to_live = 0
        """
        # Select table without columns with "counter" type for this nemesis - issue #1037:
        #    Modify_table nemesis chooses first non-system table, and modify default_time_to_live of it.
        #    But table with counters doesn't support this

        # max allowed TTL - 49 days (4300000) (to be compatible with default TWCS settings)

        default_min_ttl = 864000  # 10 days in seconds
        default_max_ttl = 4300000

        ks_cfs = self.cluster.get_non_system_ks_cf_list(
            db_node=self.target_node, filter_out_table_with_counter=True,
            filter_out_mv=True)

        keyspace_table = random.choice(ks_cfs) if ks_cfs else ks_cfs
        keyspace, table = keyspace_table.split('.')
        compaction_strategy = get_compaction_strategy(node=self.target_node, keyspace=keyspace, table=table)

        if compaction_strategy == CompactionStrategy.TIME_WINDOW:
            with self.cluster.cql_connection_patient(self.target_node) as session:
                LOGGER.debug(f'Getting data from Scylla node: {self.target_node}, table: {keyspace_table}')
                compaction_properties = get_table_compaction_info(
                    keyspace=keyspace, table=table, session=session
                )
            default_min_ttl, default_max_ttl = calculate_allowed_twcs_ttl_borders(
                compaction_properties, default_min_ttl
            )

        value = random.randint(default_min_ttl, default_max_ttl)

        InfoEvent(f'New default time to live to be set: {value}, for table: {keyspace_table}').publish()
        self._modify_table_property(name="default_time_to_live", val=value,
                                    filter_out_table_with_counter=True, keyspace_table=keyspace_table)

    def modify_table_max_index_interval(self):
        """
            If the total memory usage of all index summaries reaches this value, Cassandra decreases
            the index summaries for the coldest SSTables to the maximum set by max_index_interval.
            The max_index_interval is the sparsest possible sampling in relation to memory pressure.
            default: max_index_interval = 2048
        """
        self._modify_table_property(name="max_index_interval", val=random.choice([1024, 4096, 8192]))

    def modify_table_min_index_interval(self):
        """
            The minimum gap between index entries in the index summary. A lower min_index_interval
            means the index summary contains more entries from the index, which allows Cassandra
            to search fewer index entries to execute a read. A larger index summary may also use
            more memory. The value for min_index_interval is the densest possible sampling of the index.
            default: min_index_interval = 128
        """
        self._modify_table_property(name="min_index_interval", val=random.choice([128, 256, 512]))

    def modify_table_memtable_flush_period_in_ms(self):  # pylint: disable=invalid-name
        """
            The number of milliseconds before Cassandra flushes memtables associated with this table.
            default: memtable_flush_period_in_ms = 0
        """
        self._modify_table_property(name="memtable_flush_period_in_ms",
                                    val=random.choice([0, random.randint(60000, 200000)]))

    def modify_table_read_repair_chance(self):
        """
            The probability that a successful read operation will trigger a read repair.of read repairs
            being invoked. Unlike the repair controlled by dc_local_read_repair_chance, this repair is
            not limited to replicas in the same DC as the coordinator. The value must be between 0 and 1
            default: read_repair_chance = 0.0
        """
        self._modify_table_property(name="read_repair_chance", val=random.choice([0, 0.2, 0.5, 0.9]))

    def modify_table_speculative_retry(self):
        """
            Use the speculative retry property to configure rapid read protection. In a normal read,
            Cassandra sends data requests to just enough replica nodes to satisfy the consistency
            level. In rapid read protection, Cassandra sends out extra read requests to other replicas,
            even after the consistency level has been met. The speculative retry property specifies
            the trigger for these extra read requests.
                ALWAYS: Send extra read requests to all other replicas after every read.
                Xpercentile: Cassandra constantly tracks each table's typical read latency (in milliseconds).
                             If you set speculative retry to Xpercentile, Cassandra sends redundant read
                             requests if the coordinator has not received a response after X percent of the
                             table's typical latency time.
                Nms: Send extra read requests to all other replicas if the coordinator node has not received
                     any responses within N milliseconds.
                NONE: Do not send extra read requests after any read.
            default: speculative_retry = '99.0PERCENTILE';
        """
        options = ("'ALWAYS'",
                   "'%spercentile'" % random.randint(95, 99),
                   "'%sms'" % random.randint(300, 1000))
        self._modify_table_property(name="speculative_retry", val=random.choice(options))

    def modify_table_twcs_window_size(self):
        """ Change window size for tables with TWCS

            After window size of TWCS changed, tables should be
            reshaped. Process should not bring write amplification
            if size of sstables in timewindow is differs significantly
        """
        self.use_nemesis_seed()

        def set_new_twcs_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
            """ Recommended number of sstables for twcs is 20 - 30
                if number of sstables more than 32, sstables are picked up
                in bucket by 32

                if number of sstables more than 100, then 99 sstables contain 1 time window
                and 100th sstables will contain more than 1 sstable
                default value for twcs_max_window_count = 50
                number_of_sstables = ttl / unit size ~=~ 100 min / 1 min = 100 sstables

                example of settings param:
                    {"name": ks_cf,
                     "compaction":  {'class': 'TimeWindowCompactionStrategy',
                                     'compaction_window_size': '1',
                                     'compaction_window_unit': 'MINUTES'},
                     "gc": 864000,
                     "dttl": 86400}

            """
            # It is allowed that compaction strategy has only "class" defined, and "compaction_window_size" and "compaction_window_unit"
            # are not defined.
            current_unit = settings["compaction"].get("compaction_window_unit") or "DAYS"  # default value
            current_size = int(settings["compaction"].get("compaction_window_size") or 1)  # default value
            multiplier = 3600 if current_unit in ["DAYS", "HOURS"] else 60
            expected_sstable_number = 35

            if current_unit == "DAYS":
                current_size = current_size + 1
            elif current_unit == "HOURS":
                if (current_size // 24) > 2:
                    current_unit = "DAYS"
                    current_size = 3
                else:
                    current_size += 10
            elif (current_size // 60) > 10:
                current_unit = "HOURS"
                current_size = 11
            else:
                current_size += 35

            settings["gc"] = current_size * multiplier * expected_sstable_number // 2
            settings["dttl"] = current_size * multiplier * expected_sstable_number
            settings["compaction"]["compaction_window_unit"] = current_unit
            settings["compaction"]["compaction_window_size"] = current_size

            return settings

        all_ks_cs_with_twcs = self.cluster.get_all_tables_with_twcs(self.target_node)
        self.log.debug("All tables with TWCS %s", all_ks_cs_with_twcs)

        if not all_ks_cs_with_twcs:
            raise UnsupportedNemesis('No table found with TWCS')

        target_ks_cs_with_settings = random.choice(all_ks_cs_with_twcs)

        ks_cs_settings = set_new_twcs_settings(target_ks_cs_with_settings)
        keyspace, table = ks_cs_settings["name"].split(".")

        num_sstables_before_change = len(self.target_node.get_list_of_sstables(keyspace, table, suffix="-Data.db"))

        self.log.debug("New TWCS settings: %s", str(ks_cs_settings))
        self._modify_table_property(
            name="compaction", val=ks_cs_settings["compaction"], keyspace_table=ks_cs_settings["name"])
        self._modify_table_property(name="default_time_to_live",
                                    val=ks_cs_settings["dttl"], keyspace_table=ks_cs_settings["name"])
        self._modify_table_property(name="gc_grace_seconds",
                                    val=ks_cs_settings["gc"], keyspace_table=ks_cs_settings["name"])

        self.cluster.wait_for_schema_agreement()
        # wait timeout  equal 2% of test duration for generating sstables with timewindow settings
        sleep_timeout = int(0.02 * self.tester.params["test_duration"])
        time.sleep(sleep_timeout)

        self.target_node.stop_scylla()

        reshape_twcs_records = self.target_node.follow_system_log(
            patterns=["need reshape. Starting reshape process",
                      "Reshaping",
                      f"Reshape {ks_cs_settings['name']} .* Reshaped"])

        self.target_node.start_scylla()

        reshape_twcs_records = list(reshape_twcs_records)
        if not reshape_twcs_records:
            self.log.warning("Log message with sstables for reshape was not found. Autocompaction already"
                             "compact sstables by timewindows")
        self.log.debug("Reshape log %s", reshape_twcs_records)

        num_sstables_after_change = len(self.target_node.get_list_of_sstables(keyspace, table, suffix="-Data.db"))

        self.log.info("Number of sstables before: %s and after %s change twcs settings",
                      num_sstables_before_change, num_sstables_after_change)
        if num_sstables_before_change > num_sstables_after_change:
            self.log.error("Number of sstables after change settings larger than before")
        # run major compaction on all nodes
        # to reshape sstables on other nodes
        for node in self.cluster.data_nodes:
            num_sstables_before_change = len(node.get_list_of_sstables(keyspace, table, suffix="-Data.db"))
            node.run_nodetool("compact", args=f"{keyspace} {table}")
            num_sstables_after_change = len(node.get_list_of_sstables(keyspace, table, suffix="-Data.db"))
            self.log.info("Number of sstables before: %s and after %s change twcs settings on node: %s",
                          num_sstables_before_change, num_sstables_after_change, node.name)
            if num_sstables_before_change > num_sstables_after_change:
                self.log.error("Number of sstables after change settings larger than before")

    def disrupt_toggle_table_ics(self):
        self.toggle_table_ics()

    def disrupt_toggle_table_gc_mode(self):
        self.toggle_table_gc_mode()

    def disrupt_modify_table(self):
        # randomly select and run one of disrupt_modify_table* methods
        disrupt_func_name = random.choice([dm for dm in dir(self) if dm.startswith("modify_table")])
        disrupt_func = getattr(self, disrupt_func_name)
        disrupt_func()

    @target_data_nodes
    def disrupt_mgmt_backup_specific_keyspaces(self):
        self._mgmt_backup(backup_specific_tables=True)

    @target_data_nodes
    def disrupt_mgmt_backup(self):
        self._mgmt_backup(backup_specific_tables=False)

    @target_data_nodes
    def disrupt_mgmt_restore(self):
        def get_total_scylla_partition_size():
            result = self.cluster.data_nodes[0].remoter.run("df -k | grep /var/lib/scylla")  # Size in KB
            free_space_size = int(result.stdout.split()[1]) / 1024 ** 2  # Converting to GB
            return free_space_size

        def choose_snapshot(snapshots_dict, region: str):
            snapshot_groups_by_size = snapshots_dict["snapshots_sizes"]
            total_partition_size = get_total_scylla_partition_size()
            all_snapshot_sizes = sorted(list(snapshot_groups_by_size.keys()), reverse=True)
            fitting_snapshot_sizes = [size for size in all_snapshot_sizes if total_partition_size / size >= 20]
            if self.tester.test_duration < 1000:
                # Since verifying the restored data takes a long time, the nemesis limits the size of the restored
                # backup based on the test duration
                fitting_snapshot_sizes = [size for size in fitting_snapshot_sizes if size < 50]
            # The restore should not take more than 5% of the space total space in /var/lib/scylla
            assert fitting_snapshot_sizes, "There's not enough space for any snapshot restoration"

            self.use_nemesis_seed()
            chosen_snapshot_size = random.choice(fitting_snapshot_sizes)
            all_snapshots_per_region = snapshot_groups_by_size[chosen_snapshot_size]["snapshots"][region]

            if self.cluster.nodes[0].is_enterprise:
                snapshot_tag = random.choice(list(all_snapshots_per_region.keys()))
            else:
                oss_snapshots = [snapshot_key for snapshot_key, snapshot_value in all_snapshots_per_region.items() if
                                 snapshot_value['scylla_product'] == "oss"]
                snapshot_tag = random.choice(oss_snapshots)

            snapshot_info = all_snapshots_per_region[snapshot_tag]
            snapshot_info.update({"expected_timeout": snapshot_groups_by_size[chosen_snapshot_size]["expected_timeout"],
                                  "number_of_rows": snapshot_groups_by_size[chosen_snapshot_size]["number_of_rows"]})
            return snapshot_tag, snapshot_info

        def execute_data_validation_thread(command_template, keyspace_name, number_of_rows):
            stress_queue = []
            number_of_loaders = self.tester.params.get("n_loaders")
            rows_per_loader = int(number_of_rows / number_of_loaders)
            for loader_index in range(number_of_loaders):
                stress_command = command_template.format(num_of_rows=rows_per_loader,
                                                         keyspace_name=keyspace_name,
                                                         sequence_start=rows_per_loader * loader_index + 1,
                                                         sequence_end=rows_per_loader * (loader_index + 1))
                read_thread = self.tester.run_stress_thread(stress_cmd=stress_command, round_robin=True,
                                                            stop_test_on_failure=False)
                stress_queue.append(read_thread)
            return stress_queue

        skip_issues = [
            "https://github.com/scylladb/scylla-manager/issues/3829",
            "https://github.com/scylladb/scylla-manager/issues/4049"
        ]
        is_multi_dc = (len(self.cluster.params.region_names) > 1) or ((self.params.get("simulated_regions") or 0) > 1)
        if SkipPerIssues(skip_issues, params=self.tester.params) and is_multi_dc:
            raise UnsupportedNemesis("MultiDC cluster configuration is not supported by this nemesis")

        if not (self.cluster.params.get('use_mgmt') or self.cluster.params.get('use_cloud_manager')):
            raise UnsupportedNemesis('Scylla-manager configuration is not defined!')
        if self.cluster.params.get('cluster_backend') not in ('aws', 'k8s-eks'):
            raise UnsupportedNemesis("The restore test only supports 'AWS' and 'K8S-EKS' backends.")

        mgr_cluster = self.cluster.get_cluster_manager()
        cluster_backend = self.cluster.params.get('cluster_backend')
        if cluster_backend == 'k8s-eks':
            cluster_backend = 'aws'

        persistent_manager_snapshots_dict = get_persistent_snapshots()
        region = next(iter(self.cluster.params.region_names), '')
        target_bucket = persistent_manager_snapshots_dict[cluster_backend]["bucket"].format(region=region)
        chosen_snapshot_tag, chosen_snapshot_info = (
            choose_snapshot(snapshots_dict=persistent_manager_snapshots_dict[cluster_backend], region=region)
        )

        self.log.info("Restoring the keyspace %s", chosen_snapshot_info["keyspace_name"])
        location_list = [f"{self.cluster.params.get('backup_bucket_backend')}:{target_bucket}"]
        test_keyspaces = [keyspace.replace('"', '') for keyspace in self.cluster.get_test_keyspaces()]
        # Keyspace names that start with a digit are surrounded by quotation marks in the output of a describe query
        if chosen_snapshot_info["keyspace_name"] not in test_keyspaces:
            self.log.info("Restoring the schema of the keyspace '%s'",
                          chosen_snapshot_info["keyspace_name"])
            restore_task = mgr_cluster.create_restore_task(restore_schema=True, location_list=location_list,
                                                           snapshot_tag=chosen_snapshot_tag)

            restore_task.wait_and_get_final_status(step=10,
                                                   timeout=6*60)  # giving 6 minutes to restore the schema
            assert restore_task.status == TaskStatus.DONE, \
                f'Schema restoration of {chosen_snapshot_tag} has failed!'

            with ignore_ycsb_connection_refused():
                self.cluster.restart_scylla()  # After schema restoration, you should restart the nodes

            # TODO: Bring it back after the implementation of https://github.com/scylladb/scylla-manager/issues/4049
            # which will unblock schema restore into a different DC. For now, we can restore schema only within one DC.
            # According to https://github.com/scylladb/scylla-manager/issues/4041#issuecomment-2565489699, the step
            # below is not needed if restoring the schema within one DC.
            #
            # self.tester.set_ks_strategy_to_network_and_rf_according_to_cluster(
            #    keyspace=chosen_snapshot_info["keyspace_name"], repair_after_alter=False)

        restore_task = mgr_cluster.create_restore_task(restore_data=True,
                                                       location_list=location_list,
                                                       snapshot_tag=chosen_snapshot_tag)
        restore_task.wait_and_get_final_status(step=30, timeout=chosen_snapshot_info["expected_timeout"])
        assert restore_task.status == TaskStatus.DONE, f'Data restoration of {chosen_snapshot_tag} has failed!'

        confirmation_stress_template = (
            persistent_manager_snapshots_dict)[cluster_backend]["confirmation_stress_template"]
        stress_queue = execute_data_validation_thread(command_template=confirmation_stress_template,
                                                      keyspace_name=chosen_snapshot_info["keyspace_name"],
                                                      number_of_rows=chosen_snapshot_info["number_of_rows"])
        for stress in stress_queue:
            is_passed = self.tester.verify_stress_thread(cs_thread_pool=stress)
            assert is_passed, (
                "Data verification stress command, triggered by the 'mgmt_restore' nemesis, has failed")

    def _delete_existing_backups(self, mgr_cluster):
        deleted_tasks = []
        existing_backup_tasks = mgr_cluster.backup_task_list
        for backup_task in existing_backup_tasks:
            if backup_task.status in [TaskStatus.NEW, TaskStatus.RUNNING, TaskStatus.STARTING, TaskStatus.ERROR]:
                deleted_tasks.append(backup_task.id)
                mgr_cluster.delete_task(backup_task)
        if deleted_tasks:
            self.log.warning("Deleted the following backup tasks before the nemesis starts: %s",
                             ", ".join(deleted_tasks))

    @decorate_with_context_if_issues_open(
        ignore_take_snapshot_failing,
        issue_refs=['https://github.com/scylladb/scylla-manager/issues/3389'])
    def _mgmt_backup(self, backup_specific_tables):
        if not self.cluster.params.get('use_mgmt') and not self.cluster.params.get('use_cloud_manager'):
            raise UnsupportedNemesis('Scylla-manager configuration is not defined!')
        mgr_cluster = self.cluster.get_cluster_manager()
        if self.cluster.params.get('use_cloud_manager'):
            auto_backup_task = mgr_cluster.backup_task_list[0]
            #  An example of the auto generated backup task of cloud manager is:
            #  │ backup/8e40b8b4-5394-42d3-9884-a4ce8ab69687
            #  │ --dc 'AWS_US_EAST_1' -L AWS_US_EAST_1:s3:scylla-cloud-backup-9952-10120-4q4w4d --retention 14
            #  --rate-limit AWS_US_EAST_1:100 --snapshot-parallel '<nil>' --upload-parallel '<nil>'
            #  │ 06 Jun 21 18:10:05 UTC (+1d)  │ NEW
            location = auto_backup_task.get_task_info_dict()["location"]
        else:
            if not self.cluster.params.get('backup_bucket_location'):
                raise UnsupportedNemesis('backup bucket location configuration is not defined!')

            backup_bucket_backend = self.cluster.params.get("backup_bucket_backend")
            region = next(iter(self.cluster.params.region_names), '')
            backup_bucket_location = self.cluster.params.get("backup_bucket_location").format(region=region)
            location = f"{backup_bucket_backend}:{backup_bucket_location.split()[0]}"
        self._delete_existing_backups(mgr_cluster)
        if backup_specific_tables:
            non_test_keyspaces = self.cluster.get_test_keyspaces()
            mgr_task = mgr_cluster.create_backup_task(location_list=[location, ], keyspace_list=non_test_keyspaces)
        else:
            mgr_task = mgr_cluster.create_backup_task(location_list=[location, ])

        assert mgr_task is not None, "Backup task wasn't created"

        status = mgr_task.wait_and_get_final_status(timeout=54000, step=5, only_final=True)
        if status == TaskStatus.DONE:
            self.log.info("Task: %s is done.", mgr_task.id)
        elif status in (TaskStatus.ERROR, TaskStatus.ERROR_FINAL):
            assert False, f'Backup task {mgr_task.id} failed'
        else:
            mgr_task.stop()
            assert False, f'Backup task {mgr_task.id} timed out - while on status {status}'

    def disrupt_mgmt_repair_cli(self):
        if not self.cluster.params.get('use_mgmt') and not self.cluster.params.get('use_cloud_manager'):
            raise UnsupportedNemesis('Scylla-manager configuration is not defined!')
        self._mgmt_repair_cli()

    def disrupt_mgmt_corrupt_then_repair(self):
        if not self.cluster.params.get('use_mgmt') and not self.cluster.params.get('use_cloud_manager'):
            raise UnsupportedNemesis('Scylla-manager configuration is not defined!')
        self._destroy_data_and_restart_scylla()
        self._mgmt_repair_cli()

    @latency_calculator_decorator(legend="Scylla-Manger repair")
    def _mgmt_repair_cli(self):
        mgr_cluster = self.cluster.get_cluster_manager()
        mgr_task = mgr_cluster.create_repair_task()
        task_final_status = mgr_task.wait_and_get_final_status(timeout=86400)  # timeout is 24 hours
        if task_final_status != TaskStatus.DONE:
            progress_full_string = mgr_task.progress_string(
                parse_table_res=False, is_verify_errorless_result=True).stdout
            if task_final_status != TaskStatus.ERROR_FINAL:
                mgr_task.stop()
            raise ScyllaManagerError(
                f'Task: {mgr_task.id} final status is: {str(task_final_status)}.\nTask progress string: '
                f'{progress_full_string}')
        self.log.info('Task: {} is done.'.format(mgr_task.id))

    def disrupt_abort_repair(self):
        """
        Start repair target_node in background, then try to abort the repair streaming.
        """
        self.log.debug("Start repair target_node in background")

        @raise_event_on_failure
        def silenced_nodetool_repair_to_fail():
            try:
                self.target_node.run_nodetool("repair", verbose=True,
                                              warning_event_on_exception=(UnexpectedExit, Libssh2UnexpectedExit),
                                              error_message="Repair failed as expected. ",
                                              publish_event=False,
                                              long_running=True, retry=0)
            except (UnexpectedExit, Libssh2UnexpectedExit):
                self.log.info('Repair failed as expected')
            except Exception:
                self.log.error('Repair failed due to the unknown error')
                raise

        def repair_streaming_exists():
            path = '/storage_service/active_repair/'
            active_repair_cmd = build_node_api_command(path_url=path)
            result = self.target_node.remoter.run(active_repair_cmd)
            active_repairs = re.match(r".*\[(\d)+\].*", result.stdout)
            if active_repairs:
                self.log.debug("Found '%s' active repairs", active_repairs.group(1))
                return True
            return False

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='NodeToolRepairThread') as thread_pool:
            thread = thread_pool.submit(silenced_nodetool_repair_to_fail)
            wait.wait_for(func=repair_streaming_exists,
                          timeout=300,
                          step=1,
                          throw_exc=True,
                          text='Wait for repair starts')

            self.log.debug("Abort repair streaming by storage_service/force_terminate_repair API")

            with DbEventsFilter(db_event=DatabaseLogEvent.DATABASE_ERROR,
                                line="repair's stream failed: streaming::stream_exception",
                                node=self.target_node), \
                DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                               line="Can not find stream_manager",
                               node=self.target_node), \
                DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                               line="is aborted",
                               node=self.target_node), \
                DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                               line="Failed to repair",
                               node=self.target_node):
                self.target_node.remoter.run(
                    "curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json'"
                    " http://127.0.0.1:10000/storage_service/force_terminate_repair"
                )
                thread.result(timeout=120)
                time.sleep(10)  # to make sure all failed logs/events, are ignored correctly

        self.log.debug("Execute a complete repair for target node")
        self.repair_nodetool_repair()

    @target_data_nodes
    def disrupt_validate_hh_short_downtime(self):  # pylint: disable=invalid-name
        """
            Validates that hinted handoff mechanism works: there were no drops and errors
            during short stop of one of the nodes in cluster
        """
        # NOTE: enable back when 'https://github.com/scylladb/scylladb/issues/8136' issue is fixed
        if SkipPerIssues('https://github.com/scylladb/scylladb/issues/8136', params=self.tester.params):
            raise UnsupportedNemesis('https://github.com/scylladb/scylladb/issues/8136')

        if self.cluster.params.get('hinted_handoff') == 'disabled':
            raise UnsupportedNemesis('For this nemesis to work, `hinted_handoff` needs to be set to `enabled`')

        start_time = time.time()
        self.target_node.stop_scylla()
        time.sleep(10)
        self.target_node.start_scylla()

        # Wait until all other nodes see the target node as UN
        # Only then we can expect that hint sending started on all nodes
        def target_node_reported_un_by_others():
            for node in self.cluster.data_nodes:
                if node is not self.target_node:
                    self.cluster.check_nodes_up_and_normal(nodes=[self.target_node], verification_node=node)
            return True

        wait.wait_for(func=target_node_reported_un_by_others,
                      timeout=300,
                      step=5,
                      throw_exc=True,
                      text='Wait for target_node to be seen as UN by others')

        time.sleep(120)  # Wait to complete hints sending
        assert self.tester.hints_sending_in_progress() is False, "Hints are sent too slow"
        self.tester.verify_no_drops_and_errors(starting_from=start_time)

    def _validate_snapshot(self, nodetool_cmd: str, snapshot_content: namedtuple):
        """
            The snapshot may be taken for a few options:
            - for all keyspaces (with all their tables) - nodetool snapshot cmd without parameters:
                nodetool snapshot
            - for one keyspace (with all its tables) - nodetool snapshot cmd with "-ks" parameter:
                nodetool snapshot -ks system
            - for a few keyspaces (with all their tables) - nodetool snapshot cmd with "-ks" parameter:
                nodetool snapshot -ks system, system_schema
            - for one keyspace and few tables - nodetool snapshot cmd with "-cf" parameter like:
                nodetool snapshot test -cf cf1,cf2. In this case the snapshot will be taken for table and its MVs

            By parsing of nodetool_cmd is recognized with type of snapshot was created.
            This function check if all expected keyspace/tables are in the snapshot
        """
        snapshot_params = nodetool_cmd.split()

        def is_virtual_tables_get_snapshot():
            """
            scylla commit https://github.com/scylladb/scylladb/commit/24589cf00cf8f1fae0b19a2ac1bd7b637061301a
            has stopped creating snapshots for virtual tables.
            hence we need to filter them out when compare tables to snapshot content.
            """
            if self.target_node.is_enterprise:
                return ComparableScyllaVersion(self.target_node.scylla_version) >= "2024.3.0-dev"
            else:
                return ComparableScyllaVersion(self.target_node.scylla_version) >= "6.3.0-dev"

        filter_func = self.cluster.is_table_has_no_sstables if is_virtual_tables_get_snapshot() else None
        ks_cf = self.cluster.get_any_ks_cf_list(
            db_node=self.target_node, filter_empty_tables=False, filter_func=filter_func)
        # remove quotes from keyspace or column family, since output of `nodetool listsnapshots` isn't returning them quoted
        ks_cf = [k_c.replace('"', '') for k_c in ks_cf]
        keyspace_table = []
        if len(snapshot_params) > 1:
            if snapshot_params[1] == "-kc":
                # Example: snapshot -kc cqlstress_lwt_example
                for ks in snapshot_params[2].split(','):
                    keyspace_table.extend([k_c.split('.') for k_c in ks_cf if k_c.startswith(f"{ks}.")])
            else:
                # Example: snapshot cqlstress_lwt_example -cf blogposts_update_one_column_lwt_indicator_expect,blogposts
                keyspace = snapshot_params[1]
                with self.cluster.cql_connection_patient(self.cluster.nodes[0]) as session:
                    for cf in snapshot_params[3].split(','):
                        keyspace_table.extend([[keyspace, cf]])
                        for view in get_views_of_base_table(keyspace_name=keyspace, base_table_name=cf, session=session):
                            keyspace_table.extend([[keyspace, view]])
        else:
            # Example: snapshot
            keyspace_table.extend([k_c.split('.') for k_c in ks_cf])

        snapshot_content_list = [[elem.keyspace_name, elem.table_name] for elem in snapshot_content]
        if sorted(keyspace_table) != sorted(snapshot_content_list):
            raise AssertionError(f"Snapshot content not as expected. \n"
                                 f"Expected content: {sorted(keyspace_table)} \n "
                                 f"Actual snapshot content: {sorted(snapshot_content_list)}")

    @target_data_nodes
    def disrupt_snapshot_operations(self):  # pylint: disable=too-many-statements
        """
        Extend this nemesis to run 'nodetool snapshot' more options including multiple tables.
        Random choose between:
        - create snapshot of all keyspaces
        - create snapshot of one keyspace
        - create snapshot of few keyspaces
        - create snapshot of few tables, including MVs
        """

        def _full_snapshot():
            self.log.info("Take all keyspaces snapshot")
            return 'snapshot'

        def _ks_snapshot(one_ks):
            self.log.info(f"Take {'one keyspace' if one_ks else 'few keyspaces'} snapshot")
            # Prefer to take snapshot of test keyspace. Try to find it
            ks_cf = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node)
            if not ks_cf:
                # If test table wasn't found - take system keyspace snapshot
                ks_cf = self.cluster.get_any_ks_cf_list(db_node=self.target_node)

            if one_ks:
                keyspace_table = random.choice(ks_cf)
                keyspace, _ = keyspace_table.split('.')
            else:
                keyspaces = {ks.split('.')[0] for ks in ks_cf}
                if len(keyspaces) == 1:
                    ks_cf = self.cluster.get_any_ks_cf_list(db_node=self.target_node)
                    keyspaces = {ks.split('.')[0] for ks in ks_cf}
                keyspace = ','.join(keyspaces)

            if not keyspace:
                # it's unexpected situation. We have system keyspaces, so the keyspace(s) should be found always.
                # If not - raise exception to investigate the issue
                raise Exception(f'Something wrong happened, not empty keyspace(s) was not'
                                f'found in the list:\n{ks_cf}')

            return f'snapshot -kc {keyspace}'

        def _few_tables():
            def get_ks_with_few_tables(keyspace_table):
                ks_occurrences = Counter([ks.split('.')[0] for ks in keyspace_table])
                repeated_ks = [ks for ks, count in ks_occurrences.items() if count > 1]
                return repeated_ks

            self.log.info("Take few tables snapshot")
            # Prefer to take snapshot of test table. Try to find it
            ks_cf = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node, filter_out_mv=True)

            if not ks_cf or len(ks_cf) == 1:
                # If test table wasn't found - take system table snapshot
                ks_cf = self.cluster.get_any_ks_cf_list(db_node=self.target_node, filter_out_mv=True)

            ks_with_few_tables = get_ks_with_few_tables(ks_cf)
            if not ks_with_few_tables:
                # If non-system keyspace with few tables wasn't found - take system table snapshot
                ks_cf = self.cluster.get_any_ks_cf_list(db_node=self.target_node, filter_out_mv=True)
                ks_with_few_tables = get_ks_with_few_tables(ks_cf)

            if not ks_with_few_tables:
                self.log.warning('Keyspace with few tables has not been found')
                return _full_snapshot()

            selected_keyspace = random.choice(ks_with_few_tables)
            tables = ','.join([k_c.split('.')[1] for k_c in ks_cf if k_c.startswith(f"{selected_keyspace}.")])

            if not tables:
                # it's unexpected situation. We have system keyspaces with few tables, so the keyspaces should be found.
                # If not - raise exception to investigate the issue
                raise Exception(f'Something wrong happened, tables for "{selected_keyspace}" keyspace were not'
                                f'found in the list:\n{ks_cf}')

            return f'snapshot {selected_keyspace} -cf {tables}'

        functions_map = [(_few_tables,), (_full_snapshot,), (_ks_snapshot, True), (_ks_snapshot, False)]
        snapshot_option = random.choice(functions_map)

        try:
            nodetool_cmd = snapshot_option[0]() if len(snapshot_option) == 1 else snapshot_option[0](snapshot_option[1])
            if not nodetool_cmd:
                raise ValueError("Failed to get nodetool command.")
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            raise ValueError(f"Failed to get nodetool command. Error: {exc}") from exc

        self.log.debug(f'Take snapshot with command: {nodetool_cmd}')
        result = self.target_node.run_nodetool(nodetool_cmd)
        self.log.debug(result)
        if "snapshot name" not in result.stdout:
            raise Exception(f"Snapshot name wasn't found in {nodetool_cmd} output:\n{result.stdout}")

        snapshot_name = re.findall(r'(\d+)', result.stdout.split("snapshot name")[1])[0]
        result = self.target_node.run_nodetool('listsnapshots')
        self.log.debug(result)
        if snapshot_name in result.stdout:
            self.log.info('Snapshot %s created' % snapshot_name)
            snapshots_content = parse_nodetool_listsnapshots(listsnapshots_output=result.stdout)
            snapshot_content = snapshots_content.get(snapshot_name)
            self._validate_snapshot(nodetool_cmd=nodetool_cmd, snapshot_content=snapshot_content)
            self.log.info('Snapshot %s validated successfully' % snapshot_name)
        else:
            raise Exception(f"Snapshot {snapshot_name} wasn't found in: \n{result.stdout}")

        self.clear_snapshots()

    # NOTE: '2023.1.rc0' is set in advance, not guaranteed to match when appears
    @scylla_versions(("4.6.rc0", None), ("2023.1.rc0", None))
    def disrupt_show_toppartitions(self):
        # NOTE: new API is supported only starting with Scylla 4.6
        #       In Scylla 4.5 it exists but disabled.
        return self._disrupt_show_toppartitions(allow_new_api=True)

    @scylla_versions(("4.3.rc1", "4.5"), ("2020.1.rc0", "2022.1"))
    def disrupt_show_toppartitions(self):  # pylint: disable=function-redefined
        return self._disrupt_show_toppartitions(allow_new_api=False)

    def _disrupt_show_toppartitions(self, allow_new_api: bool):
        self.log.debug(
            "Running 'disrupt_show_toppartitions' method using %s API.",
            "new and old" if allow_new_api else "old")
        ks_cf_list = self.cluster.get_any_ks_cf_list(self.target_node)
        if not ks_cf_list:
            raise UnsupportedNemesis('User-defined Keyspace and ColumnFamily are not found.')

        top_partition_api_cmds = [OldApiTopPartitionCmd]
        if allow_new_api:
            top_partition_api_cmds.append(NewApiTopPartitionCmd)
        for top_partition_api_cmd in top_partition_api_cmds:
            top_partition_api = top_partition_api_cmd(ks_cf_list)
            # workaround for issue #4519
            self.target_node.run_nodetool('cfstats')
            top_partition_api.generate_cmd_arg_values()
            result = self.target_node.run_nodetool(
                sub_cmd='toppartitions', args=top_partition_api.get_cmd_args())
            top_partition_api.verify_output(result.stdout)

    def get_rate_limit_for_network_disruption(self) -> Optional[str]:
        if not self.monitoring_set.nodes:
            return None

        # get the last 10min avg network bandwidth used, and limit  30% to 70% of it
        prometheus_stats = PrometheusDBStats(host=self.monitoring_set.nodes[0].external_address)
        # If test runs with 2 network interfaces configuration, "node_network_receive_bytes_total" will be reported on device that
        # broadcast_address is configured on it
        query = 'avg(node_network_receive_bytes_total{instance=~".*?%s.*?", device="%s"})' % \
                (self.target_node.ip_address, self.target_node.scylla_network_configuration.device)
        now = time.time()
        results = prometheus_stats.query(query=query, start=now - 600, end=now)
        assert results, "no results for node_network_receive_bytes_total metric in Prometheus "
        received_bytes_over_time = [float(avg_rate) for _, avg_rate in results[0]["values"]]
        avg_bitrate_per_node = (received_bytes_over_time[-1] - received_bytes_over_time[0]) / 600
        avg_mpbs_per_node = avg_bitrate_per_node / 1024 / 1024

        if avg_mpbs_per_node > 10:
            min_limit = int(round(avg_mpbs_per_node * 0.30))
            max_limit = int(round(avg_mpbs_per_node * 0.70))
            rate_limit_suffix = "mbps"
        else:
            avg_kbps_per_node = avg_bitrate_per_node / 1024
            min_limit = int(round(avg_kbps_per_node * 0.30))
            max_limit = int(round(avg_kbps_per_node * 0.70))
            rate_limit_suffix = "kbps"

        return "{}{}".format(random.randrange(min_limit, max_limit), rate_limit_suffix)

    def _disrupt_network_random_interruptions_k8s(self, list_of_timeout_options):
        interruptions = ["delay", "loss", "corrupt"]
        rate_limit: Optional[str] = self.get_rate_limit_for_network_disruption()
        if not rate_limit:
            self.log.warning("NetworkRandomInterruption won't limit network bandwidth due to lack of monitoring nodes.")
        else:
            interruptions.append("rate")
        duration = f"{random.choice(list_of_timeout_options)}s"
        match random.choice(interruptions):
            case "delay":
                delay_in_msecs = random.randrange(50, 300)
                jitter = delay_in_msecs * 0.2
                LOGGER.info("Delaying network - duration: %s delay: %s ms",
                            duration, delay_in_msecs)
                experiment = NetworkDelayExperiment(self.target_node, duration,
                                                    f"{delay_in_msecs}ms", correlation=20, jitter=f"{jitter}ms")
            case "loss":
                loss_percentage = random.randrange(1, 15)
                LOGGER.info("Dropping network packets - duration: %s loss_percentage: %s%%",
                            duration, loss_percentage)
                experiment = NetworkPacketLossExperiment(
                    self.target_node, duration, loss_percentage, correlation=20)
            case "corrupt":
                corrupt_percentage = random.randrange(1, 15)
                LOGGER.info("Corrupting network packets - duration: %s corrupt_percentage: %s%%",
                            duration, corrupt_percentage)
                experiment = NetworkCorruptExperiment(self.target_node, duration,
                                                      corrupt_percentage, correlation=20)
            case "rate":
                rate, suffix = rate_limit[:-4], rate_limit[-4:]
                limit_base = int(rate) * 1024 * (1024 if suffix == "mbps" else 1)
                limit = limit_base * 20
                LOGGER.info("Limiting network bandwidth - duration: %s rate: %s limit: %s",
                            duration, rate_limit, limit)
                experiment = NetworkBandwidthLimitExperiment(
                    self.target_node, duration, rate=rate_limit, limit=limit, buffer=10000)
        experiment.start()
        experiment.wait_until_finished()
        self.cluster.wait_all_nodes_un()

    def disrupt_network_random_interruptions(self):  # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        list_of_timeout_options = [10, 60, 120, 300, 500]
        if self._is_it_on_kubernetes():
            self._disrupt_network_random_interruptions_k8s(list_of_timeout_options)
            return

        if not self.cluster.extra_network_interface:
            raise UnsupportedNemesis("for this nemesis to work, you need to set `extra_network_interface: True`")

        if not self.target_node.install_traffic_control():
            raise UnsupportedNemesis("Traffic control package not installed on system")

        if not self.target_node.scylla_network_configuration.device:
            raise ValueError("The network device name is not recognized")

        rate_limit: Optional[str] = self.get_rate_limit_for_network_disruption()
        if not rate_limit:
            self.log.warning("NetworkRandomInterruption won't limit network bandwidth due to lack of monitoring nodes.")

        # random packet loss - between 1% - 15%
        loss_percentage = random.randrange(1, 15)

        # random packet corruption - between 1% - 15%
        corrupt_percentage = random.randrange(1, 15)

        # random packet delay - between 1s - 30s
        delay_in_secs = random.randrange(1, 30)

        list_of_tc_options = [
            ("NetworkRandomInterruption_{}pct_loss".format(loss_percentage), "--loss {}%".format(loss_percentage)),
            ("NetworkRandomInterruption_{}pct_corrupt".format(corrupt_percentage),
             "--corrupt {}%".format(corrupt_percentage)),
            ("NetworkRandomInterruption_{}sec_delay".format(delay_in_secs),
             "--delay {}s --delay-distro 500ms".format(delay_in_secs))]
        if rate_limit:
            list_of_tc_options.append(
                ("NetworkRandomInterruption_{}_limit".format(rate_limit), "--rate {}".format(rate_limit)))

        option_name, selected_option = random.choice(list_of_tc_options)
        wait_time = random.choice(list_of_timeout_options)

        InfoEvent(option_name).publish()
        self.log.debug("NetworkRandomInterruption: [%s] for %dsec", selected_option, wait_time)
        self.target_node.traffic_control(None)
        try:
            self.target_node.traffic_control(selected_option)
            time.sleep(wait_time)
        finally:
            self.target_node.traffic_control(None)
            self.cluster.wait_all_nodes_un()

    def _disrupt_network_block_k8s(self, list_of_timeout_options):
        duration = f"{random.choice(list_of_timeout_options)}s"
        experiment = NetworkPacketLossExperiment(self.target_node, duration, probability=100)
        experiment.start()
        experiment.wait_until_finished()
        time.sleep(15)
        self.cluster.wait_all_nodes_un()

    def disrupt_network_block(self):
        list_of_timeout_options = [10, 60, 120, 300, 500]
        if self._is_it_on_kubernetes():
            self._disrupt_network_block_k8s(list_of_timeout_options)
            return

        if not self.cluster.extra_network_interface:
            raise UnsupportedNemesis("for this nemesis to work, you need to set `extra_network_interface: True`")

        if not self.target_node.install_traffic_control():
            raise UnsupportedNemesis("Traffic control package not installed on system")

        systemd_version = get_systemd_version(
            self.target_node.remoter.run("systemctl --version", ignore_status=True).stdout)
        if systemd_version < 256:
            context_manager = EventsSeverityChangerFilter(
                new_severity=Severity.WARNING, event_class=CoreDumpEvent, regex=r".*executable=.*networkd.*",
                extra_time_to_expiration=60)
        else:
            context_manager = contextlib.nullcontext()

        selected_option = "--loss 100%"
        wait_time = random.choice(list_of_timeout_options)
        self.log.debug("BlockNetwork: [%s] for %dsec", selected_option, wait_time)
        with context_manager:
            self.target_node.traffic_control(None)
            try:
                self.target_node.traffic_control(selected_option)
                time.sleep(wait_time)
            finally:
                self.target_node.traffic_control(None)
                self.cluster.wait_all_nodes_un()

    @target_data_nodes
    def disrupt_remove_node_then_add_node(self):  # pylint: disable=too-many-branches
        """
        https://docs.scylladb.com/operating-scylla/procedures/cluster-management/remove_node/

        1. Terminate node
        2. Run full repair
        3. Nodetool removenode, if removenode rejected, because removing node is UN in gossiper,
           repeat operation in 5 second
        4. Add new node
        5. Run nodetool cleanup (on each node) for each keyspace
        """
        if self.cluster.params.get("db_type") == 'cloud_scylla':
            raise UnsupportedNemesis("Skipping this nemesis due the replace node option that supported by Cloud "
                                     "is tested by CloudReplaceNonResponsiveNode nemesis")
        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis("On K8S nodes get removed differently. Skipping.")

        node_to_remove = self.target_node
        up_normal_nodes = self.cluster.get_nodes_up_and_normal(verification_node=node_to_remove)
        # node_to_remove must be different than node
        verification_node = random.choice([n for n in self.cluster.nodes if n is not node_to_remove])

        # node_to_remove is single/last seed in cluster, before
        # it will be terminated, choose new seed node
        num_of_seed_nodes = len(self.cluster.seed_nodes)
        if node_to_remove.is_seed and num_of_seed_nodes < 2:
            new_seed_node = random.choice([n for n in self.cluster.nodes if n is not node_to_remove])
            new_seed_node.set_seed_flag(True)
            self.cluster.update_seed_provider()

        # get node's host_id
        removed_node_status = self.cluster.get_node_status_dictionary(
            ip_address=node_to_remove.ip_address, verification_node=verification_node)
        assert removed_node_status is not None, "failed to get host_id using nodetool status"
        host_id = removed_node_status["host_id"]

        with ignore_ycsb_connection_refused():
            # node stop and make sure its "DN"
            node_to_remove.stop_scylla_server(verify_up=True, verify_down=True)

            # terminate node
            self._terminate_cluster_node(node_to_remove)

        @retrying(n=3, sleep_time=5, message="Removing node from cluster...")
        def remove_node():
            removenode_reject_msg = r"Rejected removenode operation.*the node being removed is alive"
            # nodetool removenode 'host_id'
            rnd_node = random.choice([n for n in self.cluster.nodes if n is not self.target_node])
            self.log.info("Running removenode command on {}, Removing node with the following host_id: {}"
                          .format(rnd_node.ip_address, host_id))
            with adaptive_timeout(Operations.REMOVE_NODE, rnd_node, timeout=HOUR_IN_SEC * 48):
                res = rnd_node.run_nodetool("removenode {}".format(
                    host_id), ignore_status=True, verbose=True, long_running=True, retry=0)
            if res.failed and re.match(removenode_reject_msg, res.stdout + res.stderr):
                raise Exception(f"Removenode was rejected {res.stdout}\n{res.stderr}")

            return res.exit_status

        # full cluster repair
        up_normal_nodes.remove(node_to_remove)
        # Repairing will result in a best effort repair due to the terminated node,
        # and as a result requires ignoring repair errors
        with DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                            line="failed to repair"):
            for node in up_normal_nodes:
                try:
                    self.repair_nodetool_repair(node=node, publish_event=False)
                except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
                    self.log.error(f"failed to execute repair command "
                                   f"on node {node} due to the following error: {str(details)}")

            # WORKAROUND: adding here the continuation of the nemesis to avoid the late filter messages above failing
            # the entire nemesis.

            exit_status = remove_node()

            # if remove node command failed by any reason,
            # we will remove the terminated node from
            # dead_nodes_list, so the health validator terminate the job
            if exit_status != 0:
                self.log.error(f"nodetool removenode command exited with status {exit_status}")
                # check difference between group0 and token ring,
                garbage_host_ids = verification_node.raft.get_diff_group0_token_ring_members()
                self.log.debug("Difference between token ring and group0 is %s", garbage_host_ids)
                if garbage_host_ids:
                    # if difference found, clean garbage and continue
                    verification_node.raft.clean_group0_garbage()
                else:
                    # group0 and token ring are consistent. Removenode failed by meanigfull reason.
                    # remove node from dead_nodes list to raise critical issue by HealthValidator
                    self.log.debug(
                        f"Remove failed node {node_to_remove} from dead node list {self.cluster.dead_nodes_list}")
                    node = next((n for n in self.cluster.dead_nodes_list if n.ip_address ==
                                node_to_remove.ip_address), None)
                    if node:
                        self.cluster.dead_nodes_list.remove(node)
                    else:
                        self.log.debug(f"Node {node.name} with ip {node.ip_address} was not found in dead_nodes_list")

            # verify node is removed by nodetool status
            removed_node_status = self.cluster.get_node_status_dictionary(
                ip_address=node_to_remove.ip_address, verification_node=verification_node)
            assert removed_node_status is None, \
                "Node was not removed properly (Node status:{})".format(removed_node_status)

            # add new node
            new_node = self._add_and_init_new_cluster_nodes(count=1, rack=self.target_node.rack)[0]
            # in case the removed node was not last seed.
            if node_to_remove.is_seed and num_of_seed_nodes > 1:
                new_node.set_seed_flag(True)
                self.cluster.update_seed_provider()
            # after add_node, the left nodes have data that isn't part of their tokens anymore.
            # In order to eliminate cases that we miss a "data loss" bug because of it, we cleanup this data.
            # This fix important when just user profile is run in the test and "keyspace1" doesn't exist.
            try:
                self.nodetool_cleanup_on_all_nodes_parallel()
            finally:
                self.unset_current_running_nemesis(new_node)

    def disrupt_network_reject_inter_node_communication(self):
        """
        Generates random firewall rule to drop/reject packets for inter-node communications, port 7000 and 7001
        """

        # Temporary disable due to  https://github.com/scylladb/scylla/issues/6522
        if SkipPerIssues('https://github.com/scylladb/scylladb/issues/6522', self.tester.params):
            raise UnsupportedNemesis('https://github.com/scylladb/scylladb/issues/6522')

        name = 'RejectInterNodeNetwork'

        self._install_iptables()

        textual_matching_rule, matching_rule = self._iptables_randomly_get_random_matching_rule()
        textual_pkt_action, pkt_action = self._iptables_randomly_get_disrupting_target()
        wait_time = random.choice([10, 60, 120, 300, 500])

        InfoEvent(f'{name} {textual_matching_rule} that belongs to '
                  'inter node communication connections (port=7000 and 7001) will be'
                  f' {textual_pkt_action} for {wait_time}s').publish()

        # because of https://github.com/scylladb/scylla/issues/5802, we ignore YCSB client errors here
        with ignore_alternator_client_errors():
            return self._run_commands_wait_and_cleanup(
                self.target_node,
                name=name,
                start_commands=[
                    f'sudo iptables -t filter -A INPUT -p tcp --dport 7000 {matching_rule} -j {pkt_action}',
                    f'sudo iptables -t filter -A INPUT -p tcp --dport 7001 {matching_rule} -j {pkt_action}'
                ],
                cleanup_commands=[
                    f'sudo iptables -t filter -D INPUT -p tcp --dport 7000 {matching_rule} -j {pkt_action}',
                    f'sudo iptables -t filter -D INPUT -p tcp --dport 7001 {matching_rule} -j {pkt_action}'
                ],
                wait_time=wait_time
            )

    def disrupt_network_reject_node_exporter(self):
        """
        Generates random firewall rule to drop/reject packets for node exporter connections, port 9100
        """
        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis("Not implemented for the K8S backend.")
        name = 'RejectNodeExporterNetwork'

        self._install_iptables()

        textual_matching_rule, matching_rule = self._iptables_randomly_get_random_matching_rule()
        textual_pkt_action, pkt_action = self._iptables_randomly_get_disrupting_target()
        wait_time = random.choice([10, 60, 120, 300, 500])

        InfoEvent(f'{name} {textual_matching_rule} that belongs to '
                  f'node-exporter(port=9100) connections will be {textual_pkt_action} for {wait_time}s').publish()

        return self._run_commands_wait_and_cleanup(
            self.target_node,
            name=name,
            start_commands=[f'sudo iptables -t filter -A INPUT -p tcp --dport 9100 {matching_rule} -j {pkt_action}'],
            cleanup_commands=[f'sudo iptables -t filter -D INPUT -p tcp --dport 9100 {matching_rule} -j {pkt_action}'],
            wait_time=wait_time
        )

    def disrupt_network_reject_thrift(self):
        """
        Generates random firewall rule to drop/reject packets for thrift connections, port 9100
        """
        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis("Not implemented for the K8S backend.")
        name = 'RejectThriftNetwork'

        self._install_iptables()

        textual_matching_rule, matching_rule = self._iptables_randomly_get_random_matching_rule()
        textual_pkt_action, pkt_action = self._iptables_randomly_get_disrupting_target()
        wait_time = random.choice([10, 60, 120, 300, 500])

        InfoEvent(f'{name} {textual_matching_rule} that belongs to '
                  f'Thrift(port=9160) connections will be {textual_pkt_action} for {wait_time}s').publish()

        return self._run_commands_wait_and_cleanup(
            self.target_node,
            name=name,
            start_commands=[f'sudo iptables -t filter -A INPUT -p tcp --dport 9160 {matching_rule} -j {pkt_action}'],
            cleanup_commands=[f'sudo iptables -t filter -D INPUT -p tcp --dport 9160 {matching_rule} -j {pkt_action}'],
            wait_time=wait_time
        )

    def _install_iptables(self) -> None:
        if self.target_node.distro.is_ubuntu:  # iptables is missing in a minimized Ubuntu installation
            self.target_node.install_package("iptables")

    @staticmethod
    def _iptables_randomly_get_random_matching_rule():
        """
        Randomly generates iptables matching rule to match packets in random manner
        """
        match_type = random.choice(
            ['statistic', 'statistic', 'statistic', 'limit', 'limit', 'limit', '']
            #  Make no matching rule less probable
        )

        if match_type == 'statistic':
            mode = random.choice(['random', 'nth'])
            if match_type == 'random':
                probability = random.choice(['0.0001', '0.001', '0.01', '0.1', '0.3', '0.6', '0.8', '0.9'])
                return f'randomly chosen packet with {probability} probability', \
                       f'-m statistic --mode {mode} --probability {probability}'
            elif match_type == 'nth':
                every = random.choice(['2', '4', '8', '16', '32', '64', '128'])
                return f'every {every} packet', \
                       f'-m statistic --mode {mode} --every {every} --packet 0'
        elif match_type == 'limit':
            period = random.choice(['second', 'minute'])
            pkts_per_period = random.choice({
                'second': [1, 5, 10],
                'minute': [2, 10, 40, 80]
            }.get(period))
            return f'string of {pkts_per_period} very first packets every {period}', \
                   f'-m limit --limit {pkts_per_period}/{period}'
        elif match_type == 'connbytes':
            bytes_from = random.choice(['100', '200', '400', '800', '1600', '3200', '6400', '12800', '1280000'])
            return f'every packet from connection that total byte counter exceeds {bytes_from}', \
                   f'-m connbytes --connbytes-mode bytes --connbytes-dir both --connbytes {bytes_from}'
        return 'every packet', ''

    @staticmethod
    def _iptables_randomly_get_disrupting_target():
        """
        Randomly generates iptables target that can cause disruption
        """
        target_type = random.choice(['REJECT', 'DROP'])
        if target_type == 'REJECT':
            reject_with = random.choice([
                'icmp-net-unreachable',
                'icmp-host-unreachable',
                'icmp-port-unreachable',
                'icmp-proto-unreachable',
                'icmp-net-prohibited',
                'icmp-host-prohibited',
                'icmp-admin-prohibited'
            ])
            return f'rejected with {reject_with}', \
                   f'{target_type} --reject-with {reject_with}'
        return 'dropped', f'{target_type}'

    def _run_commands_wait_and_cleanup(  # pylint: disable=too-many-arguments
            self, node, name: str, start_commands: List[str],
            cleanup_commands: List[str] = None, wait_time: int = 0):
        """
        Runs command/commands on target node wait and run cleanup commands
            :param node: target node
            :param name: Name of Nemesis for logging
            :param start_commands: commands to run on start
            :param cleanup_commands: commands to run to cleanup
            :param wait_time: waiting time
        """
        cmd_executed = {}
        if cleanup_commands is None:
            cleanup_commands = []
        for cmd_num, cmd in enumerate(start_commands):
            try:
                node.remoter.run(cmd)
                self.log.debug(f"{name}: executed: {cmd}")
                cmd_executed[cmd_num] = True
                if wait_time:
                    time.sleep(wait_time)
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                cmd_executed[cmd_num] = False
                self.log.error(
                    f"{name}: failed to execute start command "
                    f"{cmd} on node {node} due to the following error: {str(exc)}")
        if not cmd_executed:
            return
        for cmd_num, cmd in enumerate(cleanup_commands):
            try:
                node.remoter.run(cmd)
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                self.log.debug(f"{name}: failed to execute cleanup command "
                               f"{cmd} on node {node} due to the following error: {str(exc)}")

    @decorate_with_context([
        ignore_ycsb_connection_refused,
    ])
    @decorate_with_context_if_issues_open(
        ignore_ipv6_failure_to_assign,
        issue_refs=['https://github.com/scylladb/scylladb/issues/20387'])
    def reboot_node(self, target_node, hard=True, verify_ssh=True):
        target_node.reboot(hard=hard, verify_ssh=verify_ssh)
        if self.tester.params.get('print_kernel_callstack'):
            save_kallsyms_map(node=target_node)

    def disrupt_network_start_stop_interface(self):  # pylint: disable=invalid-name
        if not self.cluster.extra_network_interface:
            raise UnsupportedNemesis("for this nemesis to work, you need to set `extra_network_interface: True`")

        list_of_timeout_options = [10, 60, 120, 300, 500]
        wait_time = random.choice(list_of_timeout_options)
        self.log.debug("Taking down eth1 for %dsec", wait_time)

        try:
            self.target_node.stop_network_interface()
            time.sleep(wait_time)
        finally:
            self.target_node.start_network_interface()
            self.cluster.wait_all_nodes_un()

    def _call_disrupt_func_after_expression_logged(self,
                                                   log_follower: Iterable[str],
                                                   disrupt_func: Callable,
                                                   disrupt_func_kwargs: dict = None,
                                                   sleep: int = 1,
                                                   delay: int = 10,
                                                   timeout: int = 600):
        """
        This method watches the target node logs for an expression
        with a <sleep> time step. Once the expression is found it
        will call the callable <disrupt_func> with <disrupt_func_kwargs> keyword
        arguments after a <delay>.
        """
        start_time = time.time()

        with DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                            line="This node was decommissioned and will not rejoin",
                            node=self.target_node), \
            DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                           line="Fail to send STREAM_MUTATION_DONE",
                           node=self.target_node), \
            DbEventsFilter(db_event=DatabaseLogEvent.DATABASE_ERROR,
                           line="streaming::stream_exception",
                           node=self.target_node), \
            DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                           line="got error in row level repair",
                           node=self.target_node):
            while time.time() - start_time < timeout:
                if list(log_follower):
                    time.sleep(delay)
                    disrupt_func(**disrupt_func_kwargs)
                    break
                time.sleep(sleep)

    def start_and_interrupt_decommission_streaming(self):
        """
        1. Start to decommission the target node using <nodetool decommission>.
        2. Stop the decommission with a hard reboot once unbootstrapping starts.
        3. Verify that the node was not decommissioned completely or add
        a new node in its place in case it was.
        4. Trigger a rebuild on the decommissioned node.
        """
        def decommission_post_action():
            """
            Verify that the decommission ocurred, was interrupted and
            is still in the cluster ip list. If that is not the case,
            add a new node to the cluster (to keep the desired number
            of nodes) and raise a DecommissionNotStopped exception.
            """
            target_is_seed = self.target_node.is_seed
            try:
                self.cluster.verify_decommission(self.target_node)
            except NodeStayInClusterAfterDecommission:
                self.log.debug('The decommission of target node is successfully interrupted')
                return None
            except NodeCleanedAfterDecommissionAborted:
                self.log.debug("Decommission aborted, Group0 was cleaned successfully. New node will be added")
            except Group0MembersNotConsistentWithTokenRingMembersException as exc:
                self.log.error("Cluster state could be not predictable due to ghost members in raft group0: %s", exc)
                raise
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                self.log.error('Unexpected exception raised in checking decommission status: %s', exc)

            self.log.info('Decommission might complete before stopping it. Re-add a new node')
            new_node = self._add_and_init_new_cluster_nodes(count=1, rack=self.target_node.rack)[0]
            if new_node.is_seed != target_is_seed:
                new_node.set_seed_flag(target_is_seed)
                self.cluster.update_seed_provider()
            self.unset_current_running_nemesis(new_node)
            return new_node

        terminate_pattern = self.target_node.raft.get_random_log_message(operation=TopologyOperations.DECOMMISSION,
                                                                         seed=self.tester.params.get("nemesis_seed"))
        self.log.debug("Reboot node after log message: '%s'", terminate_pattern.log_message)

        nodetool_decommission_timeout = terminate_pattern.timeout + 600

        log_follower = self.target_node.follow_system_log(patterns=[terminate_pattern.log_message])

        trigger = partial(self.target_node.run_nodetool,
                          sub_cmd="decommission",
                          timeout=nodetool_decommission_timeout,
                          warning_event_on_exception=(Exception,),
                          long_running=True, retry=0)

        watcher = partial(
            self._call_disrupt_func_after_expression_logged,
            log_follower=log_follower,
            disrupt_func=self.reboot_node,
            disrupt_func_kwargs={"target_node": self.target_node, "hard": True, "verify_ssh": True},
            delay=0
        )
        full_operations_timeout = nodetool_decommission_timeout + 600
        with contextlib.ExitStack() as stack:
            for expected_start_failed_context in self.target_node.raft.get_severity_change_filters_scylla_start_failed(
                    terminate_pattern.timeout):
                stack.enter_context(expected_start_failed_context)
            with ignore_stream_mutation_fragments_errors(), ignore_raft_topology_cmd_failing(), \
                self.run_nemesis(node_list=self.cluster.data_nodes, nemesis_label="DecommissionStreamingErr") as verification_node, \
                FailedDecommissionOperationMonitoring(target_node=self.target_node,
                                                      verification_node=verification_node,
                                                      timeout=full_operations_timeout):

                ParallelObject(objects=[trigger, watcher], timeout=full_operations_timeout).call_objects()
            if new_node := decommission_post_action():
                new_node.wait_node_fully_start()
                new_node.run_nodetool("rebuild", long_running=True, retry=0)
            else:
                self.target_node.wait_node_fully_start()
                self.target_node.run_nodetool(sub_cmd="rebuild", long_running=True, retry=0)

    def start_and_interrupt_repair_streaming(self):
        """
        1. Destroy some data on the target node and restart it.
        2. Start a repair on the target node using <nodetool repair>.
        3. Stop it with a hard reboot once the repair starts.
        4. Trigger a rebuild on the target node after the reboot.
        """
        self._destroy_data_and_restart_scylla()
        trigger = partial(
            self.target_node.run_nodetool, sub_cmd="repair", warning_event_on_exception=(Exception,), long_running=True, retry=0, timeout=600,
        )
        log_follower = self.target_node.follow_system_log(patterns=["Repair 1 out of"])
        timeout = 1200
        if self.cluster.params.get('cluster_backend') == 'azure':
            timeout += 1200  # Azure reboot can take up to 20min to initiate
        watcher = partial(
            self._call_disrupt_func_after_expression_logged,
            log_follower=log_follower,
            disrupt_func=self.reboot_node,
            disrupt_func_kwargs={"target_node": self.target_node, "hard": True, "verify_ssh": True},
            delay=1
        )
        ParallelObject(objects=[trigger, watcher], timeout=timeout).call_objects()

        self.target_node.wait_node_fully_start()

        with adaptive_timeout(Operations.REBUILD, self.target_node, timeout=HOUR_IN_SEC * 48):
            self.target_node.run_nodetool("rebuild", long_running=True, retry=0)

    def start_and_interrupt_rebuild_streaming(self):
        """
        1. Destroy some data on the target node and restart it.
        2. Start a rebuild on the target node using <nodetool rebuild>.
        3. Stop it with a hard reboot once the rebuild starts.
        4. Trigger another rebuild after the hard reboot on the target node.
        """
        self._destroy_data_and_restart_scylla()

        timeout = 1800 if self._is_it_on_kubernetes() else 400
        if self.cluster.params.get('cluster_backend') == 'azure':
            timeout += 1200  # Azure reboot can take up to 20min to initiate

        trigger = partial(
            self.target_node.run_nodetool, sub_cmd="rebuild", warning_event_on_exception=(Exception,), long_running=True, retry=0, timeout=timeout//2
        )
        log_follower = self.target_node.follow_system_log(patterns=["rebuild from dc:"])

        watcher = partial(
            self._call_disrupt_func_after_expression_logged,
            log_follower=log_follower,
            disrupt_func=self.reboot_node,
            disrupt_func_kwargs={"target_node": self.target_node, "hard": True, "verify_ssh": True},
            timeout=timeout,
            delay=1
        )
        ParallelObject(objects=[trigger, watcher], timeout=timeout + 60).call_objects()
        self.target_node.wait_node_fully_start(timeout=300)
        with adaptive_timeout(Operations.REBUILD, self.target_node, timeout=HOUR_IN_SEC * 48):
            self.target_node.run_nodetool("rebuild", long_running=True, retry=0)

    def disrupt_decommission_streaming_err(self):
        """
        Stop decommission in middle to trigger some streaming fails, then rebuild the data on the node.
        If the node is decommissioned unexpectedly, need to re-add a new node to cluster.
        """
        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis(
                "This nemesis logic is not compatible with K8S approach "
                "for handling Scylla member's decommissioning.")

        with ignore_stream_mutation_fragments_errors(), ignore_raft_topology_cmd_failing():
            self.start_and_interrupt_decommission_streaming()

    def disrupt_rebuild_streaming_err(self):
        """
        Stop rebuild in middle to trigger some streaming fails, then rebuild the data on the node.
        """
        with ignore_stream_mutation_fragments_errors(), ignore_raft_topology_cmd_failing():
            self.start_and_interrupt_rebuild_streaming()

    def disrupt_repair_streaming_err(self):
        """
        Stop repair in middle to trigger some streaming fails, then rebuild the data on the node.
        """
        with ignore_raft_topology_cmd_failing():
            self.start_and_interrupt_repair_streaming()

    def _corrupt_data_file(self):
        """Randomly corrupt data file by dd"""
        ks_cfs = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node)
        if not ks_cfs:
            raise UnsupportedNemesis(
                'Non-system keyspace and table are not found. Nemesis can\'t be run')

        # Corrupt data file
        data_file_pattern = self._choose_file_for_destroy(ks_cfs)
        res = self.target_node.remoter.run('sudo find {}-Data.db'.format(data_file_pattern))
        for sstable_file in res.stdout.split():
            self.target_node.remoter.run('sudo dd if=/dev/urandom of={} count=1024'.format(sstable_file))
            self.log.debug('File {} was corrupted by dd'.format(sstable_file))

    def disrupt_corrupt_then_scrub(self):
        """
        Try to rebuild the sstables of all test keyspaces by scrub, the corrupted partitions
        will be skipped.
        """
        self.log.debug("Rebuild sstables by scrub with `--skip-corrupted`, corrupted partitions will be skipped.")
        with ignore_scrub_invalid_errors(), adaptive_timeout(Operations.SCRUB, self.target_node, timeout=HOUR_IN_SEC * 48):
            for ks in self.cluster.get_test_keyspaces():
                self.target_node.run_nodetool("scrub", args=f"--skip-corrupted {ks}")

        self.clear_snapshots()

    @latency_calculator_decorator(legend="Adding new nodes")
    def add_new_nodes(self, count, rack=None, instance_type: str = None) -> list[BaseNode]:
        nodes = self._add_and_init_new_cluster_nodes(count, rack=rack, instance_type=instance_type)
        wait_for_tablets_balanced(nodes[0])
        return nodes

    @latency_calculator_decorator(legend="Decommission nodes: remove nodes from cluster")
    def decommission_nodes(self, nodes):

        def _decommission(node):
            try:
                InfoEvent(f'StartEvent - ShrinkCluster started decommissioning a node {node}').publish()
                self.cluster.decommission(node)
                InfoEvent(f'FinishEvent - ShrinkCluster has done decommissioning a node {node}').publish()
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                InfoEvent(f'FinishEvent - ShrinkCluster failed decommissioning a node {self.target_node} with error '
                          f'{str(exc)}').publish()
                raise

        num_workers = None if (self.cluster.parallel_node_operations and nodes[0].raft.is_enabled) else 1
        parallel_obj = ParallelObject(objects=nodes, timeout=MAX_TIME_WAIT_FOR_DECOMMISSION, num_workers=num_workers)
        InfoEvent(f'StartEvent - ShrinkCluster started decommissioning {len(nodes)} nodes').publish()
        parallel_obj.run(_decommission, ignore_exceptions=False, unpack_objects=True)
        self.monitoring_set.reconfigure_scylla_monitoring()
        InfoEvent(f'FinishEvent - ShrinkCluster has done decommissioning {len(nodes)} nodes').publish()

    def _decommission_nodes(self, nodes_number, rack, is_seed: Optional[Union[bool, DefaultValue]] = DefaultValue,
                            dc_idx: Optional[int] = None, exact_nodes: list[BaseNode] | None = None):
        nodes_to_decommission = []
        if exact_nodes:
            nodes_to_decommission = exact_nodes
            for node in exact_nodes:
                self.set_current_running_nemesis(node=node)
        else:
            for idx in range(nodes_number):
                if self._is_it_on_kubernetes():
                    if rack is None and self._is_it_on_kubernetes():
                        rack = 0
                    self.set_target_node_pool_type(NEMESIS_TARGET_POOLS.data_nodes)
                    self.set_target_node(rack=rack, is_seed=is_seed, allow_only_last_node_in_rack=True)
                else:
                    rack_idx = rack if rack is not None else idx % self.cluster.racks_count
                    # if rack is not specified, round-robin racks
                    self.set_target_node(is_seed=is_seed, dc_idx=dc_idx, rack=rack_idx)
                nodes_to_decommission.append(self.target_node)
                self.target_node = None  # otherwise node.running_nemesis will be taken off the node by self.set_target_node
        try:
            if self.cluster.parallel_node_operations:
                self.decommission_nodes(nodes_to_decommission)
            else:
                for node in nodes_to_decommission:
                    self.decommission_nodes([node])
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            InfoEvent(f'FinishEvent - ShrinkCluster failed decommissioning a node {self.target_node} with error '
                      f'{str(exc)}').publish()

    @latency_calculator_decorator(legend="Doubling cluster load")
    def _double_cluster_load(self, duration: int) -> None:
        duration = 30
        self.log.info("Doubling the load on the cluster for %s minutes", duration)
        stress_queue = self.tester.run_stress_thread(
            stress_cmd=self.tester.stress_cmd, stress_num=1, stats_aggregate_cmds=False, duration=duration)
        results = self.tester.get_stress_results(queue=stress_queue, store_results=False)
        self.stop_nemesis_on_stress_errors(stress_queue)
        self.log.info(f"Double load results: {results}")

    @target_data_nodes
    def disrupt_grow_cluster(self):
        sleep_time_between_ops = self.cluster.params.get('nemesis_sequence_sleep_between_ops')
        if not self.has_steady_run and sleep_time_between_ops:
            self.steady_state_latency()
            self.has_steady_run = True
        self._grow_cluster(rack=None)

    @target_data_nodes
    def disrupt_grow_fill(self):
        sleep_time_between_ops = self.cluster.params.get('nemesis_sequence_sleep_between_ops')
        if not self.has_steady_run and sleep_time_between_ops:
            self.steady_state_latency()
            self.has_steady_run = True

        stress_cmds = self.cluster.params.get('stress_cmd')
        for stress_cmd in stress_cmds:
            self._grow_cluster()
            self.tester.wait_no_compactions_running(n=240)
            write_thread = self.tester.run_stress_thread(
                stress_cmd=stress_cmd, stop_test_on_failure=False, stats_aggregate_cmds=False)
            self.tester.verify_stress_thread(write_thread)
            self.log.info("Finish cluster fill")
            time.sleep(self.interval)

    @target_data_nodes
    def disrupt_grow_shrink_cluster(self):
        sleep_time_between_ops = self.cluster.params.get('nemesis_sequence_sleep_between_ops')
        if not self.has_steady_run and sleep_time_between_ops:
            self.steady_state_latency()
            self.has_steady_run = True
        new_nodes = self._grow_cluster(rack=None)

        # pass on the exact nodes only if we have specific types for them
        new_nodes = new_nodes if self.tester.params.get('nemesis_grow_shrink_instance_type') else None
        if duration := self.tester.params.get('nemesis_double_load_during_grow_shrink_duration'):
            self._double_cluster_load(duration)
        self._shrink_cluster(rack=None, new_nodes=new_nodes)

    @target_data_nodes
    def disrupt_grow_shrink_datacenter(self):
        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis("Operator doesn't support multi-DC yet. Skipping.")
        if self.cluster.test_config.MULTI_REGION:
            raise UnsupportedNemesis(
                "grow_shring_datacenter skipped for multi-dc scenario (https://github.com/scylladb/scylla-cluster-tests/issues/5369)")
        InfoEvent(message='Starting Grow Shrink DC Nemesis').publish()
        sleep_time_between_ops = self.cluster.params.get('nemesis_sequence_sleep_between_ops')
        sleep_time_between_ops = sleep_time_between_ops if sleep_time_between_ops else 10
        sleep_time_between_ops = sleep_time_between_ops * 60
        if not self.has_steady_run and sleep_time_between_ops:
            self.steady_state_latency()
            self.has_steady_run = True

        # Switch to NetworkTopologyReplicationStrategy using the existing DC
        node = self.cluster.data_nodes[0]
        system_keyspaces = ["system_distributed", "system_traces"]
        if not node.raft.is_consistent_topology_changes_enabled:  # auth-v2 is used when consistent topology is enabled
            system_keyspaces.insert(0, "system_auth")
        self._switch_to_network_replication_strategy(self.cluster.get_test_keyspaces() + system_keyspaces)

        # create a new dc
        InfoEvent(message='New DC').publish()
        nodes_on_new_dc = []
        initial_dc_nodes = self.cluster.params.get('n_db_nodes')
        for _ in range(initial_dc_nodes):
            nodes_on_new_dc += [self._add_new_node_in_new_dc()]
        time.sleep(sleep_time_between_ops)

        # reconfigure keyspaces
        InfoEvent(message='Update replication strategy').publish()
        datacenters = list(self.tester.db_cluster.get_nodetool_status().keys())
        new_dc_list = [dc for dc in datacenters if dc.endswith("_nemesis_dc")]
        assert new_dc_list, "new datacenter was not registered"
        new_dc_name = new_dc_list[0]

        for keyspace in system_keyspaces + self.cluster.get_test_keyspaces():
            strategy = ReplicationStrategy.get(node, keyspace)
            assert isinstance(
                strategy, NetworkTopologyReplicationStrategy), "Should have been already switched to NetworkStrategy"
            for rf in range(1, initial_dc_nodes+1):
                strategy.replication_factors_per_dc.update({new_dc_name: rf})
                strategy.apply(node, keyspace)

        InfoEvent(message='execute rebuild on new datacenter').publish()
        for new_node in nodes_on_new_dc:
            with wait_for_log_lines(node=new_node, start_line_patterns=["rebuild.*started with keyspaces=", "Rebuild starts"],
                                    end_line_patterns=["rebuild.*finished with keyspaces=", "Rebuild succeeded"],
                                    start_timeout=60, end_timeout=600):
                new_node.run_nodetool(sub_cmd=f"rebuild -- {datacenters[0]}", long_running=True, retry=0)
        InfoEvent(message='Running full cluster repair on each data node').publish()
        for cluster_node in self.cluster.data_nodes:
            cluster_node.run_nodetool(sub_cmd="repair -pr", publish_event=True)

        # add nodes to each dc
        InfoEvent(message='Grow both DCs').publish()
        add_nodes_number = self.tester.params.get('nemesis_add_node_cnt')
        self._grow_cluster()
        for _ in range(add_nodes_number):
            nodes_on_new_dc += [self._add_new_node_in_new_dc()]
        time.sleep(sleep_time_between_ops)

        # remove the new dc
        InfoEvent(message='Remove DC').publish()
        for node in nodes_on_new_dc:
            self.cluster.decommission(node)

    # NOTE: version limitation is caused by the following:
    #       - https://github.com/scylladb/scylla-enterprise/issues/3211
    #       - https://github.com/scylladb/scylladb/issues/14184
    @scylla_versions(("5.2.7", None), ("2023.1.1", None))
    def disrupt_grow_shrink_new_rack(self):
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis("Adding new rack is not supported for non-k8s Scylla clusters")
        rack = max(self.cluster.racks) + 1
        self._grow_cluster(rack)
        self._shrink_cluster(rack)

    def _grow_cluster(self, rack=None):
        if rack is None and self._is_it_on_kubernetes():
            rack = 0
        add_nodes_number = self.tester.params.get('nemesis_add_node_cnt')
        InfoEvent(message=f"Start grow cluster by {add_nodes_number} data nodes").publish()
        new_nodes = []
        if self.cluster.parallel_node_operations:
            new_nodes = self.add_new_nodes(count=add_nodes_number, rack=rack,
                                           instance_type=self.tester.params.get('nemesis_grow_shrink_instance_type'))
        else:
            for idx in range(add_nodes_number):
                # if rack is not specified, round-robin racks to spread nodes evenly
                rack_idx = rack if rack is not None else idx % self.cluster.racks_count
                new_nodes += self.add_new_nodes(count=1, rack=rack_idx,
                                                instance_type=self.tester.params.get('nemesis_grow_shrink_instance_type'))
        self.log.info("Finish cluster grow")
        time.sleep(self.interval)
        return new_nodes

    def _shrink_cluster(self, rack=None, new_nodes: list[BaseNode] | None = None):
        add_nodes_number = self.tester.params.get('nemesis_add_node_cnt')
        InfoEvent(message=f'Start shrink cluster by {add_nodes_number} nodes').publish()
        # Check that number of nodes is enough for decommission:
        self.log.debug("Current target_node %s, is zero_node: %s, dc_idx: %s", self.target_node.name,
                       self.target_node._is_zero_token_node, self.target_node.dc_idx)
        cur_num_nodes_in_dc = len([n for n in self.cluster.data_nodes if n.dc_idx == self.target_node.dc_idx])
        initial_db_size = self.tester.params.get("n_db_nodes")
        if self._is_it_on_kubernetes():
            initial_db_size = self.tester.params.get("k8s_n_scylla_pods_per_cluster") or initial_db_size

        if isinstance(initial_db_size, int):
            decommission_nodes_number = min(cur_num_nodes_in_dc - initial_db_size, add_nodes_number)
        else:
            initial_db_size_in_dc = int(initial_db_size.split(" ")[self.target_node.dc_idx])
            decommission_nodes_number = min(cur_num_nodes_in_dc - initial_db_size_in_dc, add_nodes_number)

        if decommission_nodes_number < 1:
            error = "Not enough nodes for decommission"
            self.log.warning("Shrink cluster skipped. Error: %s", error)
            raise Exception(error)

        self.log.info("Start shrink cluster by %s nodes", decommission_nodes_number)
        # Currently on kubernetes first two nodes of each rack are getting seed status
        # Because of such behavior only way to get them decommission is to enable decommissioning
        # TBD: After https://github.com/scylladb/scylla-operator/issues/292 is fixed remove is_seed parameter
        self._decommission_nodes(
            decommission_nodes_number,
            rack,
            is_seed=None if self._is_it_on_kubernetes() else DefaultValue,
            dc_idx=self.target_node.dc_idx,
            exact_nodes=new_nodes,
        )
        num_of_nodes = len(self.cluster.data_nodes)
        self.log.info("Cluster shrink finished. Current number of data nodes %s", num_of_nodes)
        InfoEvent(message=f'Cluster shrink finished. Current number of data nodes {num_of_nodes}').publish()

    # TODO: add support for the 'LocalFileSystemKeyProviderFactory' and 'KmipKeyProviderFactory' key providers
    # TODO: add encryption for a table with large partitions?

    def disrupt_enable_disable_table_encryption_aws_kms_provider_without_rotation(self):
        self._enable_disable_table_encryption(
            enable_kms_key_rotation=False,
            additional_scylla_encryption_options={'key_provider': 'KmsKeyProviderFactory'})

    def disrupt_enable_disable_table_encryption_aws_kms_provider_with_rotation(self):
        self._enable_disable_table_encryption(
            enable_kms_key_rotation=True,
            additional_scylla_encryption_options={'key_provider': 'KmsKeyProviderFactory'})

    @decorate_with_context(ignore_ycsb_connection_refused)
    @scylla_versions(("2023.1.1-dev", None))
    @target_data_nodes
    def _enable_disable_table_encryption(self, enable_kms_key_rotation, additional_scylla_encryption_options=None):  # noqa: PLR0914
        if self.cluster.params.get("cluster_backend") != "aws":
            raise UnsupportedNemesis("This nemesis is supported only on the AWS cluster backend")

        scylla_encryption_options = {'cipher_algorithm': 'AES/ECB/PKCS5Padding', 'secret_key_strength': 128}
        scylla_encryption_options |= additional_scylla_encryption_options or {}
        aws_kms, kms_key_alias_name = None, None

        # Handle AWS KMS specific parts
        if additional_scylla_encryption_options and additional_scylla_encryption_options.get(
                'key_provider', 'N/A') == 'KmsKeyProviderFactory':
            kms_host_name = "kms-host"
            kms_key_alias_name = f"alias/testid-{self.cluster.test_config.test_id()}"
            scylla_encryption_options |= {'kms_host': kms_host_name}
            aws_kms = AwsKms(region_names=self.cluster.params.region_names)
            aws_kms.create_alias(kms_key_alias_name)
            for node in self.cluster.nodes:
                is_restart_needed = False
                with node.remote_scylla_yaml() as scylla_yml:
                    if not scylla_yml.kms_hosts:
                        scylla_yml.kms_hosts = {}
                    if kms_host_name not in scylla_yml.kms_hosts:
                        scylla_yml.kms_hosts[kms_host_name] = {
                            'master_key': kms_key_alias_name,
                            'aws_region': node.region,
                            'aws_use_ec2_credentials': True,
                        }
                        is_restart_needed = True
                if is_restart_needed:
                    node.restart_scylla()

        # Create table with encryption
        keyspace_name, table_name = self.cluster.get_test_keyspaces()[0], 'tmp_encrypted_table'
        self.log.info("Create '%s.%s' table with server encryption", keyspace_name, table_name)
        with self.cluster.cql_connection_patient(self.target_node, keyspace=keyspace_name) as session:
            # NOTE: scylla-bench expects following table structure:
            #       (pk bigint, ck bigint, v blob, PRIMARY KEY(pk, ck)) WITH compression = { }
            create_table_query_cmd = (
                f"CREATE TABLE IF NOT EXISTS {table_name}"
                " (pk bigint, ck bigint, v blob, PRIMARY KEY (pk, ck))"
                " WITH compression = { } AND read_repair_chance=0.0"
                f" AND compaction = {{ 'class' : '{self.cluster.params.get('compaction_strategy')}' }}"
                f" AND scylla_encryption_options = {scylla_encryption_options};")
            session.execute(create_table_query_cmd)

        def upgrade_sstables(nodes):
            for node in nodes:
                self.log.info("Upgradesstables on the '%s' node for the new encrypted table", node.name)
                # NOTE: 'flush' is needed in case there are no sstables yet
                node.remoter.run(f'nodetool flush -- {keyspace_name} {table_name}', verbose=True)
                # NOTE: 'flush' is needed for system_schema, to make sure the new table info
                # is on disk, `scylla sstable` reads only from disk
                node.remoter.run('nodetool flush -- system_schema', verbose=True)
                time.sleep(2)
                node.remoter.run(f'nodetool upgradesstables -a -- {keyspace_name} {table_name}', verbose=True)

        @retrying(n=4, sleep_time=30, allowed_exceptions=(AssertionError, ))
        def check_encryption_fact(sstable_util_instance, expected_bool_value):
            sstable_util_instance.check_that_sstables_are_encrypted(expected_bool_value=expected_bool_value)

        def run_write_scylla_bench_load(write_cmd):
            # NOTE: 'scylla-bench' runs 'truncate' operation when 'validate-data' is used in addition
            #       to the 'write' mode. So it may cause racy following loader error:
            #
            #       Error during truncate: seastar::rpc::remote_verb_error (filesystem error: \
            #         link failed: No such file or directory
            #       It also may cause the 'sstable - Error while linking SSTable' error messages in DB logs
            with EventsSeverityChangerFilter(
                    new_severity=Severity.WARNING, event_class=ScyllaBenchEvent, extra_time_to_expiration=30,
                    regex=r".*Error during truncate: seastar::rpc::remote_verb_error \(filesystem error.*"
            ), EventsSeverityChangerFilter(
                    new_severity=Severity.WARNING, event_class=DatabaseLogEvent, extra_time_to_expiration=30,
                    regex=".*sstable - Error while linking SSTable.*filesystem error: stat failed: No such file or directory.*"):
                write_thread = self.tester.run_stress_thread(stress_cmd=write_cmd, stop_test_on_failure=False)
                self.tester.verify_stress_thread(write_thread)
                self.stop_nemesis_on_stress_errors(write_thread)

        try:
            for i in range(2 if (aws_kms and kms_key_alias_name and enable_kms_key_rotation) else 1):
                # Write data
                write_cmd = (
                    "scylla-bench -mode=write -workload=sequential -consistency-level=all -replication-factor=3"
                    " -partition-count=50 -clustering-row-count=100 -clustering-row-size=uniform:75..125"
                    f" -keyspace {keyspace_name} -table {table_name} -timeout=120s -validate-data")
                run_write_scylla_bench_load(write_cmd)
                upgrade_sstables(self.cluster.data_nodes)

                # Read data
                read_cmd = (
                    "scylla-bench -mode=read -workload=sequential -consistency-level=all -replication-factor=3"
                    " -partition-count=50 -clustering-row-count=100 -clustering-row-size=uniform:75..125"
                    f" -keyspace {keyspace_name} -table {table_name} -timeout=120s -validate-data"
                    " -iterations=1 -concurrency=10 -connection-count=10 -rows-per-request=10")
                read_thread = self.tester.run_stress_thread(stress_cmd=read_cmd, stop_test_on_failure=False)
                self.tester.verify_stress_thread(read_thread)
                self.stop_nemesis_on_stress_errors(read_thread)

                # Rotate KMS key
                if enable_kms_key_rotation and aws_kms and kms_key_alias_name and i == 0:
                    aws_kms.rotate_kms_key(kms_key_alias_name)

            # Check that sstables of that table are really encrypted
            sstable_util = SstableUtils(db_node=self.target_node, ks_cf=f"{keyspace_name}.{table_name}")
            check_encryption_fact(sstable_util, True)

            with self.target_node.remote_scylla_yaml() as scylla_yaml:
                user_info_encryption_enabled = scylla_yaml.user_info_encryption \
                    and scylla_yaml.user_info_encryption.get('enabled', False)

            # if encryption is enabled by default, we currently can't disable it
            if not user_info_encryption_enabled:
                # Disable encryption for the encrypted table
                self.log.info("Disable encryption for the '%s.%s' table", keyspace_name, table_name)
                with self.cluster.cql_connection_patient(self.target_node, keyspace=keyspace_name) as session:
                    query = f"ALTER TABLE {table_name} WITH scylla_encryption_options = {{'key_provider': 'none'}};"
                    session.execute(query)
                upgrade_sstables(self.cluster.nodes)

                # ReRead data
                read_thread2 = self.tester.run_stress_thread(stress_cmd=read_cmd, stop_test_on_failure=False)
                self.tester.verify_stress_thread(read_thread2)
                self.stop_nemesis_on_stress_errors(read_thread2)

                # ReWrite data making the sstables be rewritten
                run_write_scylla_bench_load(write_cmd)
                upgrade_sstables(self.cluster.nodes)

                # ReRead data
                read_thread3 = self.tester.run_stress_thread(stress_cmd=read_cmd, stop_test_on_failure=False)
                self.tester.verify_stress_thread(read_thread3)
                self.stop_nemesis_on_stress_errors(read_thread3)

                # Check that sstables of that table are not encrypted anymore
                check_encryption_fact(sstable_util, False)
        finally:
            # Delete table
            self.log.info("Delete '%s.%s' table with server encryption", keyspace_name, table_name)
            with self.cluster.cql_connection_patient(self.target_node, keyspace=keyspace_name) as session:
                session.execute(f"DROP TABLE {table_name};")

    def disrupt_hot_reloading_internode_certificate(self):
        """
        https://github.com/scylladb/scylla/issues/6067
        Scylla has the ability to hot reload SSL certificates.
        This test will create and reload new certificates for the inter node communication.
        """
        if not self.cluster.params.get('server_encrypt'):
            raise UnsupportedNemesis('Server Encryption is not enabled, hence skipping')

        @timeout_decor(timeout=600, allowed_exceptions=(LogContentNotFound, ))
        def check_ssl_reload_log(node_system_log):
            if not list(node_system_log):
                raise LogContentNotFound('Reload SSL message not found in node log')
            return True

        ssl_files_location = json.loads(
            self.target_node.get_scylla_config_param("server_encryption_options"))["certificate"]
        in_place_crt = self.target_node.remoter.run(f"cat {ssl_files_location}",
                                                    ignore_status=True).stdout
        node_system_logs = {}

        if SkipPerIssues('https://github.com/scylladb/scylladb/issues/7909', params=self.tester.params):
            # TBD: To be removed after https://github.com/scylladb/scylla/issues/7909#issuecomment-758062545 is resolved
            context_manager = DbEventsFilter(
                db_event=DatabaseLogEvent.DATABASE_ERROR,
                line="error GnuTLS:-34, Base64 decoding error")
        else:
            context_manager = contextlib.nullcontext()

        with context_manager:
            for node in self.cluster.nodes:
                node_system_logs[node] = node.follow_system_log(
                    patterns=[f'messaging_service - Reloaded {{"{ssl_files_location}"}}'])
                update_certificate(node)
                node.remoter.send_files(src=str(node.ssl_conf_dir / TLSAssets.DB_CERT), dst='/tmp')
                node.remoter.run(f"sudo cp -f /tmp/{TLSAssets.DB_CERT} {ssl_files_location}")
                new_crt = node.remoter.run(f"cat {ssl_files_location}").stdout
                if in_place_crt == new_crt:
                    raise Exception('The CRT file was not replaced')

            for node in self.cluster.nodes:
                if not check_ssl_reload_log(node_system_logs[node]):
                    raise Exception('SSL auto Reload did not happen')

    @latency_calculator_decorator
    def steady_state_latency(self, sleep_time=None):
        if not sleep_time:
            sleep_time = self.cluster.params.get('nemesis_interval') * 60
        InfoEvent(message=f'StartEvent - start a sleep of {sleep_time} as Steady State').publish()
        time.sleep(sleep_time)
        InfoEvent(message='FinishEvent - Steady State sleep has been finished').publish()

    def disrupt_run_unique_sequence(self):
        sleep_time_between_ops = self.cluster.params.get('nemesis_sequence_sleep_between_ops')
        sleep_time_between_ops = sleep_time_between_ops if sleep_time_between_ops else 8
        sleep_time_between_ops = sleep_time_between_ops * 60
        if not self.has_steady_run:
            self.steady_state_latency()
            self.has_steady_run = True
        InfoEvent(message='StartEvent - start a repair by ScyllaManager').publish()
        if self.cluster.params.get('use_mgmt') or self.cluster.params.get('use_cloud_manager'):
            self._mgmt_repair_cli()
            InfoEvent(message='FinishEvent - Manager repair has finished').publish()
        else:
            InfoEvent(message='FinishEvent - Manager repair was Skipped').publish()
        time.sleep(sleep_time_between_ops)
        InfoEvent(message='Starting grow disruption').publish()
        self._grow_cluster(rack=None)
        InfoEvent(message='Finished grow disruption').publish()
        time.sleep(sleep_time_between_ops)
        InfoEvent(message='Starting terminate_and_replace disruption').publish()
        self._terminate_and_replace_node()
        InfoEvent(message='Finished terminate_and_replace disruption').publish()
        time.sleep(sleep_time_between_ops)
        InfoEvent(message='Starting shrink disruption').publish()
        self._shrink_cluster(rack=None)
        InfoEvent(message='Starting shrink disruption').publish()

    def _k8s_disrupt_memory_stress(self):
        """Uses chaos-mesh experiment based on https://github.com/chaos-mesh/memStress"""
        if not self.target_node.k8s_cluster.chaos_mesh.initialized:
            raise UnsupportedNemesis(
                "Chaos Mesh is not installed. Set 'k8s_use_chaos_mesh' config option to 'true'")
        memory_limit = self.target_node.k8s_cluster.scylla_memory_limit
        # If a container's memory usage increases too quickly the OOM killer is invoked
        # so reduce ramp to ~2GB/s: time_to_reach = memory (in GB) /2
        time_to_reach_secs = int(convert_memory_value_from_k8s_to_units(memory_limit)/2)
        duration = 100 + time_to_reach_secs
        self.log.info('Try to allocate 90% available memory')
        experiment = MemoryStressExperiment(pod=self.target_node, duration=f"{duration}s",
                                            workers=1, size="90%", time_to_reach=f"{time_to_reach_secs}s")
        experiment.start()
        experiment.wait_until_finished()

        self.log.info('Try to allocate 100% total memory')
        experiment = MemoryStressExperiment(pod=self.target_node, duration=f"{duration}s",
                                            workers=1, size="100%", time_to_reach=f"{time_to_reach_secs}s")
        experiment.start()
        experiment.wait_until_finished()

    @decorate_with_context(ignore_reactor_stall_errors)
    def disrupt_memory_stress(self):
        """
        Try to run stress-ng to preempt allocated memory of scylla process,
        we don't monitor swap usage in /proc/$scylla_pid/status, just make sure
        no coredump, serious db error occur during the heavy load of memory.

        """
        if self._is_it_on_kubernetes():
            self._k8s_disrupt_memory_stress()
            return

        if SkipPerIssues('scylladb/scylladb#11807', params=self.tester.params):
            # since we might get into uncontrolled situations with this nemesis
            # see https://github.com/scylladb/scylla-cluster-tests/issues/6928
            raise UnsupportedNemesis("Disabled cause of https://github.com/scylladb/scylla-cluster-tests/issues/6928")

        if self.target_node.distro.is_rhel_like:
            self.target_node.install_epel()
            self.target_node.remoter.sudo('yum install -y stress-ng')
        elif self.target_node.distro.is_ubuntu:
            self.target_node.remoter.sudo('apt-get -y install stress-ng')
        else:
            raise UnsupportedNemesis(f"{self.target_node.distro} OS not supported!")

        self.log.info('Try to allocate 90% total memory, the allocated memory will be swaped out')
        self.target_node.remoter.run(
            "stress-ng --vm-bytes $(awk '/MemTotal/{printf \"%d\\n\", $2 * 0.9;}' < /proc/meminfo)k --vm-keep -m 1 -t 100")

    def disrupt_toggle_cdc_feature_properties_on_table(self):
        """Manipulate cdc feature settings

        Find table with CDC enabled (skip nemesis if not found).
        Randomly select on CDC parameter.
        Toggle the selected parameter state (True/False) or select a random value for TTL

        """
        ks_tables_with_cdc = self.cluster.get_all_tables_with_cdc(self.target_node)
        self.log.debug(f"Found next tables with enabled cdc feature: {ks_tables_with_cdc}")

        if not ks_tables_with_cdc:
            raise UnsupportedNemesis("CDC is not enabled on any table. Skipping")

        ks, table = random.choice(ks_tables_with_cdc).split(".")

        self.log.debug(f"Get table {ks}.{table} cdc extension state")
        with self.cluster.cql_connection_patient(node=self.target_node) as session:
            cdc_settings = cdc.options.get_table_cdc_properties(session, ks, table)
        self.log.debug(f"table {ks}.{table} cdc extension state: {cdc_settings}")

        self.log.debug("Choose random cdc property to toggle")
        cdc_property = random.choice(cdc.options.get_cdc_settings_names())
        self.log.debug(f"Next cdc property will be changed {cdc_property}")

        cdc_settings[cdc_property] = cdc.options.toggle_cdc_property(cdc_property, cdc_settings[cdc_property])
        self.log.debug(f"New table cdc extension state: {cdc_settings}")

        self.log.info(f"Apply new cdc settigs for table {ks}.{table}: {cdc_settings}")
        self._alter_table_with_cdc_properties(ks, table, cdc_settings)
        self.log.debug(f"Verify new cdc settings on table {ks}.{table}")
        self._verify_cdc_feature_status(ks, table, cdc_settings)
        InfoEvent(f"{ks}.{table} have new cdc settings {cdc_settings}").publish()

    def disrupt_run_cdcstressor_tool(self):
        ks_tables_with_cdc = self.cluster.get_all_tables_with_cdc(self.target_node)
        if not ks_tables_with_cdc:
            self.log.warning("CDC is not enabled on any table. Skipping")
            raise UnsupportedNemesis("CDC is not enabled on any table. Skipping")

        ks, table = random.choice(ks_tables_with_cdc).split(".")
        self._run_cdc_stressor_tool(ks, table)

    def _run_cdc_stressor_tool(self, ks, table):
        cdc_stressor_cmd = self.tester.params.get("stress_cdclog_reader_cmd")

        if " -duration" not in cdc_stressor_cmd:
            read_time = random.randint(5, 20)
            cdc_stressor_cmd += f" -duration {read_time}m "

        cdc_reader_thread = self.tester.run_cdclog_reader_thread(stress_cmd=cdc_stressor_cmd,
                                                                 stress_num=1,
                                                                 keyspace_name=ks,
                                                                 base_table_name=table)

        self.tester.verify_cdclog_reader_results(cdc_reader_thread, update_es=False)

    def _alter_table_with_cdc_properties(self, keyspace: str, table: str, cdc_settings: dict) -> None:
        """Alter base table with cdc properties

        Build valid query and alter table with cdc properties
        :param keyspace: keyspace name
        :type keyspace: str
        :param table: base table name
        :type table: str
        :param cdc_enabled: is cdc enabled for base table, defaults to True
        :type cdc_enabled: bool, optional
        :param preimage: is preimage enabled for base table, defaults to False
        :type preimage: bool, optional
        :param postimage: is postimage enabled for base table, defaults to False
        :type postimage: bool, optional
        :param ttl: set ttl for scylla_cdc_log table, defaults to None
        :type ttl: Optional[int], optional
        """
        cmd = f"ALTER TABLE {keyspace}.{table} WITH cdc = {cdc_settings};"
        self.log.debug(f"Alter command: {cmd}")
        with self.cluster.cql_connection_patient(self.target_node) as session:
            session.execute(cmd)
        # wait applying cdc configuration
        time.sleep(30)

    def _verify_cdc_feature_status(self, keyspace: str, table: str, cdc_settings: dict) -> None:

        output = self.target_node.run_cqlsh(f"desc keyspace {keyspace}")
        self.log.debug(output.stdout)

        with self.cluster.cql_connection_patient(node=self.target_node) as session:
            actual_cdc_settings = cdc.options.get_table_cdc_properties(session, keyspace, table)

        if not cdc_settings["enabled"]:
            assert actual_cdc_settings["enabled"] is False, \
                f"CDC options was not disabled. Current: {actual_cdc_settings} expected: {cdc_settings}"
        else:
            assert actual_cdc_settings == cdc_settings, \
                f"CDC extension settings are differs. Current: {actual_cdc_settings} expected: {cdc_settings}"

    @latency_calculator_decorator(legend="Adding new nodes in new DC")
    def _add_new_node_in_new_dc(self, is_zero_node=False) -> BaseNode:
        if is_zero_node:
            new_node = skip_on_capacity_issues(self.cluster.add_nodes)(
                1, dc_idx=0, enable_auto_bootstrap=True, is_zero_node=is_zero_node)[0]  # add node
        else:
            new_node = skip_on_capacity_issues(self.cluster.add_nodes)(
                1, dc_idx=0, enable_auto_bootstrap=True)[0]  # add node
        with new_node.remote_scylla_yaml() as scylla_yml:
            scylla_yml.rpc_address = new_node.ip_address
            scylla_yml.seed_provider = [SeedProvider(class_name='org.apache.cassandra.locator.SimpleSeedProvider',
                                                     parameters=[{"seeds": self.tester.db_cluster.seed_nodes_addresses}])]
            endpoint_snitch = self.cluster.params.get("endpoint_snitch") or ""
            if endpoint_snitch.endswith("GossipingPropertyFileSnitch"):
                rackdc_value = {"dc": "add_remove_nemesis_dc"}
            else:
                rackdc_value = {"dc_suffix": "_nemesis_dc"}
        with new_node.remote_cassandra_rackdc_properties() as properties_file:
            properties_file.update(**rackdc_value)
        self.cluster.wait_for_init(node_list=[new_node], timeout=900,
                                   check_node_health=False)
        new_node.wait_node_fully_start()
        self.monitoring_set.reconfigure_scylla_monitoring()
        return new_node

    def _write_read_data_to_multi_dc_keyspace(self, datacenters: List[str]) -> None:
        InfoEvent(message='Writing and reading data with new dc').publish()
        write_cmd = f"cassandra-stress write no-warmup cl=ALL n=10000 -schema 'keyspace=keyspace_new_dc " \
                    f"replication(strategy=NetworkTopologyStrategy,{datacenters[0]}=3,{datacenters[1]}=1) " \
                    f"compression=LZ4Compressor compaction(strategy=SizeTieredCompactionStrategy)' " \
                    f"-mode cql3 native compression=lz4 -rate threads=5 -pop seq=1..10000 -log interval=5"
        write_thread = self.tester.run_stress_thread(stress_cmd=write_cmd, round_robin=True, stop_test_on_failure=False)
        self.tester.verify_stress_thread(cs_thread_pool=write_thread)
        self.stop_nemesis_on_stress_errors(write_thread)
        self._verify_multi_dc_keyspace_data(consistency_level="ALL")

    def _verify_multi_dc_keyspace_data(self, consistency_level: str = "ALL"):
        read_cmd = f"cassandra-stress read no-warmup cl={consistency_level} n=10000 -schema 'keyspace=keyspace_new_dc " \
                   f"compression=LZ4Compressor' -mode cql3 native compression=lz4 -rate threads=5 " \
                   f"-pop seq=1..10000 -log interval=5"
        read_thread = self.tester.run_stress_thread(stress_cmd=read_cmd, round_robin=True, stop_test_on_failure=False)
        self.tester.verify_stress_thread(cs_thread_pool=read_thread)
        self.stop_nemesis_on_stress_errors(read_thread)

    def _switch_to_network_replication_strategy(self, keyspaces: List[str]) -> None:
        """Switches replication strategy to NetworkTopology for given keyspaces.
        """
        node = self.cluster.data_nodes[0]
        nodes_by_region = self.tester.db_cluster.nodes_by_region(nodes=self.tester.db_cluster.data_nodes)
        region = list(nodes_by_region.keys())[0]
        dc_name = self.tester.db_cluster.get_nodetool_info(nodes_by_region[region][0])['Data Center']
        for keyspace in keyspaces:
            replication_strategy = ReplicationStrategy.get(node, keyspace)
            if not isinstance(replication_strategy, SimpleReplicationStrategy):
                # no need to switch as already is NetworkTopology
                continue
            self.log.info(f"Switching replication strategy to Network for '{keyspace}' keyspace")
            if keyspace == "system_auth" and replication_strategy.replication_factors[0] != len(nodes_by_region[region]):
                self.log.warning(f"system_auth keyspace is not replicated on all nodes "
                                 f"({replication_strategy.replication_factors[0]}/{len(nodes_by_region[region])}).")
            network_replication = NetworkTopologyReplicationStrategy(
                **{dc_name: replication_strategy.replication_factors[0]})  # pylint: disable=protected-access
            network_replication.apply(node, keyspace)

    def disrupt_add_remove_dc(self) -> None:
        if self._is_it_on_kubernetes():
            raise UnsupportedNemesis("Operator doesn't support multi-DC yet. Skipping.")
        if self.cluster.test_config.MULTI_REGION:
            raise UnsupportedNemesis(
                "add_remove_dc skipped for multi-dc scenario (https://github.com/scylladb/scylla-cluster-tests/issues/5369)")
        InfoEvent(message='Starting New DC Nemesis').publish()
        node = self.cluster.data_nodes[0]
        system_keyspaces = ["system_distributed", "system_traces"]
        if not node.raft.is_consistent_topology_changes_enabled:  # auth-v2 is used when consistent topology is enabled
            system_keyspaces.insert(0, "system_auth")
        self._switch_to_network_replication_strategy(self.cluster.get_test_keyspaces() + system_keyspaces)
        datacenters = list(self.tester.db_cluster.get_nodetool_status().keys())
        self.tester.create_keyspace("keyspace_new_dc", replication_factor={
                                    datacenters[0]: min(3, len(self.cluster.data_nodes))})
        node_added = False
        with ExitStack() as context_manager:
            def finalizer(exc_type, *_):
                # in case of test end/killed, leave the cleanup alone
                if exc_type is not KillNemesis:
                    with self.cluster.cql_connection_patient(node) as session:
                        session.execute('DROP KEYSPACE IF EXISTS keyspace_new_dc')
                    if node_added:
                        self.cluster.decommission(new_node)
            context_manager.push(finalizer)

            with temporary_replication_strategy_setter(node) as replication_strategy_setter:
                new_node = self._add_new_node_in_new_dc()
                node_added = True
                status = self.tester.db_cluster.get_nodetool_status()
                new_dc_list = [dc for dc in list(status.keys()) if dc.endswith("_nemesis_dc")]
                assert new_dc_list, "new datacenter was not registered"
                # Mark a new node as "running nemesis" to prevent it be marked as "target node" by parallel nemesis.
                # This new node should not be unset as running nemesis because the node will be terminated in the end of nemesis
                # and removed from the list of nodes
                self.set_current_running_nemesis(new_node)
                new_dc_name = new_dc_list[0]
                for keyspace in system_keyspaces + ["keyspace_new_dc"]:
                    strategy = ReplicationStrategy.get(node, keyspace)
                    assert isinstance(strategy, NetworkTopologyReplicationStrategy), \
                        "Should have been already switched to NetworkStrategy"
                    strategy.replication_factors_per_dc.update({new_dc_name: 1})  # pylint: disable=protected-access
                    replication_strategy_setter(**{keyspace: strategy})
                InfoEvent(message='execute rebuild on new datacenter').publish()
                with wait_for_log_lines(node=new_node, start_line_patterns=["rebuild.*started with keyspaces=", "Rebuild starts"],
                                        end_line_patterns=["rebuild.*finished with keyspaces=", "Rebuild succeeded"],
                                        start_timeout=60, end_timeout=600):
                    new_node.run_nodetool(sub_cmd=f"rebuild -- {datacenters[0]}", long_running=True, retry=0)
                InfoEvent(message='Running full cluster repair on each data node').publish()
                for cluster_node in self.cluster.data_nodes:
                    cluster_node.run_nodetool(sub_cmd="repair -pr", publish_event=True)
                datacenters = list(self.tester.db_cluster.get_nodetool_status().keys())
                self._write_read_data_to_multi_dc_keyspace(datacenters)

            self.cluster.decommission(new_node)
            node_added = False

            datacenters = list(self.tester.db_cluster.get_nodetool_status().keys())
            assert not [dc for dc in datacenters if dc.endswith("_nemesis_dc")], "new datacenter was not unregistered"
            self._verify_multi_dc_keyspace_data(consistency_level="QUORUM")

    def get_cassandra_stress_write_cmds(self):
        write_cmds = self.tester.params.get("prepare_write_cmd")
        if not write_cmds:
            return None

        if not isinstance(write_cmds, list):
            write_cmds = [write_cmds]

        stress_cmds = [cmd for cmd in write_cmds if " profile=" not in cmd and " n=" in cmd]
        if not stress_cmds:
            return None

        return stress_cmds

    def get_cassandra_stress_definition(self, stress_cmds, default_data_set_size=5000000):
        stress_cmd = stress_cmds[0]
        self.log.debug("Stress commands to get cassandra_stress_definition: %s", stress_cmd)
        column_definition = self.tester.get_c_s_column_definition(cs_cmd=stress_cmd)
        # Set small amount rows 5000000 when can not recognize a real dataset size, suppose that this amount should be
        # inserted in any case
        data_set_size = self.tester.get_data_set_size(stress_cmd) or default_data_set_size

        self.log.debug("Dataset size for nemesis: %s", data_set_size)
        self.log.debug("Cassandra-stress column definition for nemesis: %s", column_definition)
        return column_definition, data_set_size

    def disrupt_sla_increase_shares_during_load(self):
        # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
        if not self.cluster.params.get('sla'):
            raise UnsupportedNemesis("SLA nemesis can be run during SLA test only")

        if not self.cluster.nodes[0].is_enterprise:
            raise UnsupportedNemesis("SLA feature is only supported by Scylla Enterprise")

        if not self.cluster.params.get('authenticator'):
            raise UnsupportedNemesis("SLA feature can't work without authenticator")

        if not (stress_cmds := self.get_cassandra_stress_write_cmds()):
            raise UnsupportedNemesis("SLA nemesis needs cassandra-stress default table 'keyspace1.standard1' is created"
                                     " and prefilled")

        column_definition, dataset_size = self.get_cassandra_stress_definition(stress_cmds)

        prometheus_stats = PrometheusDBStats(host=self.monitoring_set.nodes[0].external_address)
        sla_tests = SlaTests()
        error_events = sla_tests.test_increase_shares_during_load(tester=self.tester,
                                                                  prometheus_stats=prometheus_stats,
                                                                  num_of_partitions=dataset_size,
                                                                  cassandra_stress_column_definition=column_definition)
        self.format_error_for_sla_test_and_raise(error_events=error_events)

    def disrupt_sla_decrease_shares_during_load(self):
        # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
        if not self.cluster.params.get('sla'):
            raise UnsupportedNemesis("SLA nemesis can be run during SLA test only")

        if not self.cluster.nodes[0].is_enterprise:
            raise UnsupportedNemesis("SLA feature is only supported by Scylla Enterprise")

        if not self.cluster.params.get('authenticator'):
            raise UnsupportedNemesis("SLA feature can't work without authenticator")

        if not (stress_cmds := self.get_cassandra_stress_write_cmds()):
            raise UnsupportedNemesis("SLA nemesis needs cassandra-stress default table 'keyspace1.standard1' is created"
                                     " and prefilled")

        column_definition, dataset_size = self.get_cassandra_stress_definition(stress_cmds)

        prometheus_stats = PrometheusDBStats(host=self.monitoring_set.nodes[0].external_address)
        sla_tests = SlaTests()
        error_events = sla_tests.test_decrease_shares_during_load(tester=self.tester,
                                                                  prometheus_stats=prometheus_stats,
                                                                  num_of_partitions=dataset_size,
                                                                  cassandra_stress_column_definition=column_definition)
        self.format_error_for_sla_test_and_raise(error_events=error_events)

    def disrupt_replace_service_level_using_detach_during_load(self):
        # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
        if not self.cluster.params.get('sla'):
            raise UnsupportedNemesis("SLA nemesis can be run during SLA test only")

        if not self.cluster.nodes[0].is_enterprise:
            raise UnsupportedNemesis("SLA feature is only supported by Scylla Enterprise")

        if not self.cluster.params.get('authenticator'):
            raise UnsupportedNemesis("SLA feature can't work without authenticator")

        if not (stress_cmds := self.get_cassandra_stress_write_cmds()):
            raise UnsupportedNemesis("SLA nemesis needs cassandra-stress default table 'keyspace1.standard1' is created"
                                     " and prefilled")

        column_definition, dataset_size = self.get_cassandra_stress_definition(stress_cmds)

        prometheus_stats = PrometheusDBStats(host=self.monitoring_set.nodes[0].external_address)
        sla_tests = SlaTests()
        error_events = sla_tests.test_replace_service_level_using_detach_during_load(
            tester=self.tester,
            prometheus_stats=prometheus_stats,
            num_of_partitions=dataset_size,
            cassandra_stress_column_definition=column_definition)
        self.format_error_for_sla_test_and_raise(error_events=error_events)

    def disrupt_replace_service_level_using_drop_during_load(self):
        # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
        if not self.cluster.params.get('sla'):
            raise UnsupportedNemesis("SLA nemesis can be run during SLA test only")

        if not self.cluster.nodes[0].is_enterprise:
            raise UnsupportedNemesis("SLA feature is only supported by Scylla Enterprise")

        if not self.cluster.params.get('authenticator'):
            raise UnsupportedNemesis("SLA feature can't work without authenticator")

        if not (stress_cmds := self.get_cassandra_stress_write_cmds()):
            raise UnsupportedNemesis("SLA nemesis needs cassandra-stress default table 'keyspace1.standard1' is created"
                                     " and prefilled")

        column_definition, dataset_size = self.get_cassandra_stress_definition(stress_cmds)

        prometheus_stats = PrometheusDBStats(host=self.monitoring_set.nodes[0].external_address)
        sla_tests = SlaTests()
        error_events = sla_tests.test_replace_service_level_using_drop_during_load(
            tester=self.tester,
            prometheus_stats=prometheus_stats,
            num_of_partitions=dataset_size,
            cassandra_stress_column_definition=column_definition)

        self.format_error_for_sla_test_and_raise(error_events=error_events)

    def disrupt_increase_shares_by_attach_another_sl_during_load(self):
        # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
        if not self.cluster.params.get('sla'):
            raise UnsupportedNemesis("SLA nemesis can be run during SLA test only")

        if not self.cluster.nodes[0].is_enterprise:
            raise UnsupportedNemesis("SLA feature is only supported by Scylla Enterprise")

        if not self.cluster.params.get('authenticator'):
            raise UnsupportedNemesis("SLA feature can't work without authenticator")

        if not (stress_cmds := self.get_cassandra_stress_write_cmds()):
            raise UnsupportedNemesis("SLA nemesis needs cassandra-stress default table 'keyspace1.standard1' is created"
                                     " and prefilled")

        column_definition, dataset_size = self.get_cassandra_stress_definition(stress_cmds)

        prometheus_stats = PrometheusDBStats(host=self.monitoring_set.nodes[0].external_address)
        sla_tests = SlaTests()
        error_events = sla_tests.test_increase_shares_by_attach_another_sl_during_load(
            tester=self.tester,
            prometheus_stats=prometheus_stats,
            num_of_partitions=dataset_size,
            cassandra_stress_column_definition=column_definition)

        self.format_error_for_sla_test_and_raise(error_events=error_events)

    def disrupt_maximum_allowed_sls_with_max_shares_during_load(self):
        # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
        if not self.cluster.params.get('sla'):
            raise UnsupportedNemesis("SLA nemesis can be run during SLA test only")

        if not self.cluster.nodes[0].is_enterprise:
            raise UnsupportedNemesis("SLA feature is only supported by Scylla Enterprise")

        if not self.cluster.params.get('authenticator'):
            raise UnsupportedNemesis("SLA feature can't work without authenticator")

        if not (stress_cmds := self.get_cassandra_stress_write_cmds()):
            raise UnsupportedNemesis("SLA nemesis needs cassandra-stress default table 'keyspace1.standard1' is created"
                                     " and prefilled")

        column_definition, dataset_size = self.get_cassandra_stress_definition(stress_cmds)

        prometheus_stats = PrometheusDBStats(host=self.monitoring_set.nodes[0].external_address)
        sla_tests = SlaTests()
        with self.cluster.cql_connection_patient(node=self.cluster.nodes[0],
                                                 user=DEFAULT_USER,
                                                 password=DEFAULT_USER_PASSWORD) as session:
            # Get amount of existing service levels plus default "cassandra"
            created_service_levels = len(ServiceLevel(session=session, name="dummy").list_all_service_levels()) + 1

        error_events = sla_tests.test_maximum_allowed_sls_with_max_shares_during_load(
            tester=self.tester,
            prometheus_stats=prometheus_stats,
            num_of_partitions=dataset_size,
            cassandra_stress_column_definition=column_definition,
            service_levels_amount=MAX_ALLOWED_SERVICE_LEVELS-created_service_levels)

        self.format_error_for_sla_test_and_raise(error_events=error_events)

    @staticmethod
    def format_error_for_sla_test_and_raise(error_events):
        if any(error_events):
            raise NemesisSubTestFailure("\n".join(f"Step: {error.step}. Error:\n - {error}"
                                                  for error in error_events if error)
                                        )

    def disrupt_create_index(self):
        """
        Create index on a random column (regular or static) of a table with the most number of partitions and wait until it gets build.
        Then verify it can be used in a query. Finally, drop the index.
        """
        with self.cluster.cql_connection_patient(self.target_node, connect_timeout=300) as session:

            ks_cf_list = self.cluster.get_non_system_ks_cf_list(self.target_node, filter_out_mv=True)
            if not ks_cf_list:
                raise UnsupportedNemesis("No table found to create index on")
            ks, cf = random.choice(ks_cf_list).split('.')
            column = get_random_column_name(session, ks, cf)
            if not column:
                raise UnsupportedNemesis("No column found to create index on")
            try:
                index_name = create_index(session, ks, cf, column)
            except InvalidRequest as exc:
                LOGGER.warning(exc)
                raise UnsupportedNemesis(  # pylint: disable=raise-missing-from
                    "Tried to create already existing index. See log for details")
            try:
                with adaptive_timeout(operation=Operations.CREATE_INDEX, node=self.target_node, timeout=14400) as timeout:
                    wait_for_index_to_be_built(self.target_node, ks, index_name, timeout=timeout * 2)
                verify_query_by_index_works(session, ks, cf, column)
                sleep_for_percent_of_duration(self.tester.test_duration * 60, percent=1,
                                              min_duration=300, max_duration=2400)
            finally:
                drop_index(session, ks, index_name)

    @target_data_nodes
    def disrupt_add_remove_mv(self):
        """
        Create a Materialized view on an existing table while a node is down.
        Take node up and run a repair.
        Verify the MV can be used in a query.
        Finally, drop the MV.
        """

        free_nodes = [node for node in self.cluster.data_nodes if not node.running_nemesis]
        if not free_nodes:
            raise UnsupportedNemesis("Not enough free nodes for nemesis. Skipping.")
        cql_query_executor_node = random.choice(free_nodes)
        self.set_current_running_nemesis(cql_query_executor_node)
        try:
            ks_cfs = self.cluster.get_non_system_ks_cf_list(db_node=cql_query_executor_node,
                                                            filter_empty_tables=True, filter_out_mv=True)
            if not ks_cfs:
                raise UnsupportedNemesis(
                    'Non-system keyspace and table are not found. nemesis can\'t be run')
            ks_name, base_table_name = random.choice(ks_cfs).split('.')
            view_name = f'{base_table_name}_view'
            with self.cluster.cql_connection_patient(node=cql_query_executor_node, connect_timeout=600) as session:
                primary_key_columns = get_column_names(
                    session=session, ks=ks_name, cf=base_table_name, is_primary_key=True)
                # selecting a supported column for creating a materialized-view (not a collection type).
                column = get_random_column_name(session=session, ks=ks_name,
                                                cf=base_table_name, filter_out_collections=True)
                if not column:
                    raise UnsupportedNemesis(
                        'A supported column for creating MV is not found. nemesis can\'t run')
                column = f'"{column}"'
                self.log.info("Stopping Scylla on node %s", self.target_node.name)
                self.target_node.stop_scylla()
                InfoEvent(message=f'Create a materialized-view for table {ks_name}.{base_table_name}').publish()
                try:
                    with EventsFilter(event_class=DatabaseLogEvent,
                                      regex='.*Error applying view update.*',
                                      extra_time_to_expiration=180):
                        self.tester.create_materialized_view(ks_name, base_table_name, view_name, [column],
                                                             primary_key_columns, session,
                                                             mv_columns=[column] + primary_key_columns)
                except Exception as error:  # pylint: disable=broad-except
                    self.log.warning('Failed creating a materialized view: %s', error)
                    self.target_node.start_scylla()
                    raise
                try:
                    self.log.info("Starting Scylla on node %s", self.target_node.name)
                    self.target_node.start_scylla()
                    self.target_node.run_nodetool(sub_cmd="repair -pr")
                    with adaptive_timeout(operation=Operations.CREATE_MV, node=self.target_node, timeout=14400) as timeout:
                        wait_for_view_to_be_built(self.target_node, ks_name, view_name, timeout=timeout * 2)
                    session.execute(SimpleStatement(f'SELECT * FROM {ks_name}.{view_name} limit 1', fetch_size=10))
                    sleep_for_percent_of_duration(self.tester.test_duration * 60, percent=1,
                                                  min_duration=300, max_duration=2400)
                finally:
                    drop_materialized_view(session, ks_name, view_name)
        finally:
            self.unset_current_running_nemesis(cql_query_executor_node)

    def disrupt_toggle_audit_syslog(self):
        self._disrupt_toggle_audit(store="syslog")

    def _disrupt_toggle_audit(self, store: AuditStore):
        """
            Enable audit log with all categories and user keyspaces (if audit already enabled, disable it and finish the Nemesis),
            verify audit log content,
            reduce categories by excluding DML and QUERY,
            verify DDL are logged in audit log correctly. Leaves audit log enabled this way.
        """
        if not self.target_node.is_enterprise:
            raise UnsupportedNemesis("Auditing feature is only supported by Scylla Enterprise")

        if store == "syslog" and self._is_it_on_kubernetes():
            # generally syslog is not supported on K8S because of different log line format
            raise UnsupportedNemesis("syslog store is not supported on Kubernetes scylladb/scylla-operator#1299")

        audit = Audit(self.cluster)

        if audit.is_enabled():
            audit.disable()
            raise UnsupportedNemesis("Audit was enabled -> disabling it")

        audit_keyspace = "audit_keyspace"
        keyspaces_for_audit = [audit_keyspace]
        InfoEvent(f"Enabling full audit for keyspaces: {keyspaces_for_audit}").publish()
        audit_config = AuditConfiguration(
            store=store,
            categories=["DCL", "DDL", "AUTH", "ADMIN", "DML", "QUERY"],
            keyspaces=keyspaces_for_audit,
            tables=[],
        )
        try:
            audit.configure(audit_config)
            keyspace_name = keyspaces_for_audit[0]
            errors = []
            audit_start = datetime.datetime.now() - datetime.timedelta(seconds=5)
            InfoEvent(message='Writing/Reading data from audited keyspace').publish()
            write_cmd = f"cassandra-stress write no-warmup cl=ONE n=1000 -schema" \
                        f" 'replication(strategy=NetworkTopologyStrategy,replication_factor=3)" \
                        f" keyspace={audit_keyspace}' -mode cql3 native -rate 'threads=1 throttle=1000/s'" \
                        f" -pop seq=1..1000 -col 'n=FIXED(1) size=FIXED(128)' -log interval=5"
            write_thread = self.tester.run_stress_thread(
                stress_cmd=write_cmd, round_robin=True, stop_test_on_failure=False)
            self.tester.verify_stress_thread(cs_thread_pool=write_thread)
            self.stop_nemesis_on_stress_errors(write_thread)
            read_cmd = f"cassandra-stress read no-warmup cl=ONE n=1000 " \
                       f" -schema 'replication(strategy=NetworkTopologyStrategy,replication_factor=3)" \
                       f" keyspace={audit_keyspace}' -mode cql3 native -rate 'threads=1 throttle=1000/s'" \
                       f" -pop seq=1..1000 -col 'n=FIXED(1) size=FIXED(128)' -log interval=5"
            read_thread = self.tester.run_stress_thread(
                stress_cmd=read_cmd, round_robin=True, stop_test_on_failure=False)
            self.tester.verify_stress_thread(cs_thread_pool=read_thread)
            self.stop_nemesis_on_stress_errors(read_thread)
            InfoEvent(message='Verifying Audit table contents').publish()
            rows = audit.get_audit_log(from_datetime=audit_start, category="DML", limit_rows=1500)
            # filter out USE keyspace rows due to https://github.com/scylladb/scylla-enterprise/issues/3169
            rows = [row for row in rows if not row.operation.startswith("USE")]
            if len(rows) != 1000:
                errors.append(f"Audit log for DML contains {len(rows)} rows while should contain 1000 rows")
                for row in rows:
                    LOGGER.error("DML audit log row: %s", row)
            rows = audit.get_audit_log(from_datetime=audit_start, category="QUERY", limit_rows=1500)
            if len(rows) != 1000:
                errors.append(f"Audit log for QUERY contains {len(rows)} rows while should contain 1000 rows")
                for row in rows:
                    LOGGER.error("QUERY audit log row: %s", row)
        except Exception as ex:  # pylint: disable=broad-except
            LOGGER.error("Exception while testing full audit: %s", ex)
            audit_config.categories = ["DCL", "DDL", "AUTH", "ADMIN"]
            audit.configure(audit_config)
            raise

        InfoEvent("Reducing audit categories and setting back audited keyspaces").publish()

        audit_config.categories = ["DCL", "DDL", "AUTH", "ADMIN"]
        audit.configure(audit_config)
        table_name = "audit_cf"
        audit_start = datetime.datetime.now() - datetime.timedelta(seconds=5)
        with self.cluster.cql_connection_patient(node=self.target_node) as session:
            query = f"CREATE TABLE IF NOT EXISTS {keyspace_name}.{table_name} (id int PRIMARY KEY, value timestamp)"
            session.execute(query)
            self.cluster.wait_for_schema_agreement()
            audit_rows = audit.get_audit_log(from_datetime=audit_start, category="DDL", operation=query, limit_rows=10)
            if not audit_rows:
                errors.append("Audit log is empty while should contain executed DDL (create table) operation")

            audit_start = datetime.datetime.now() - datetime.timedelta(seconds=5)
            query = f"ALTER TABLE {keyspace_name}.{table_name} WITH read_repair_chance = 0.0"
            session.execute(query)
            self.cluster.wait_for_schema_agreement()
            audit_rows = audit.get_audit_log(from_datetime=audit_start, category="DDL", operation=query, limit_rows=10)
            if not audit_rows:
                errors.append("Audit log is empty while should contain executed DDL (alter table) operation")

            audit_start = datetime.datetime.now() - datetime.timedelta(seconds=5)
            query = f"DROP TABLE {keyspace_name}.{table_name}"
            session.execute(query, timeout=300)
            self.cluster.wait_for_schema_agreement()
            audit_rows = audit.get_audit_log(from_datetime=audit_start, category="DDL", operation=query, limit_rows=10)
            if not audit_rows:
                errors.append("Audit log is empty while should contain executed DDL (drop table) operation")

        if errors:
            raise AuditLogTestFailure("\n".join(errors))

    @target_data_nodes
    def disrupt_bootstrap_streaming_error(self):
        """Abort bootstrap process at different point

        During bootstrap new node stream data from token ring
        If bootstrap is aborted, according to Failed topology
        changes manual, the process could be restarted and node
        should be added to cluster successfully. Nemesis terminate
        the bootstrap streaming base on found log messages and then
        remove ghost group0 members and restart bootstrap.

        If node was not added anyway, clean it from cluster
        and return the cluster to initial state(by num of nodes)
        """
        self.cluster.wait_all_nodes_un()

        new_node: BaseNode = skip_on_capacity_issues(self.cluster.add_nodes)(
            count=1, dc_idx=self.target_node.dc_idx, enable_auto_bootstrap=True, rack=self.target_node.rack)[0]
        self.monitoring_set.reconfigure_scylla_monitoring()
        self.set_current_running_nemesis(node=new_node)  # prevent to run nemesis on new node when running in parallel

        terminate_pattern = self.target_node.raft.get_random_log_message(operation=TopologyOperations.BOOTSTRAP,
                                                                         seed=self.tester.params.get("nemesis_seed"))

        bootstrapabortmanager = NodeBootstrapAbortManager(bootstrap_node=new_node, verification_node=self.target_node)

        with ignore_stream_mutation_fragments_errors(), ignore_raft_topology_cmd_failing(), ignore_raft_transport_failing():
            bootstrapabortmanager.run_bootstrap_and_abort_with_action(
                terminate_pattern, abort_action=new_node.stop_scylla)
            bootstrapabortmanager.clean_and_restart_bootstrap_after_abort()

            if not new_node.db_up() or (new_node.db_up() and not self.target_node.raft.is_cluster_topology_consistent()):
                bootstrapabortmanager.clean_unbootstrapped_node()
                raise BootstrapStreamErrorFailure(f"Node {new_node.name} failed to bootstrap. See log for more details")

        if new_node.db_up() and self.target_node.raft.is_cluster_topology_consistent():
            self.log.info("Wait 5 minutes with new topology")
            time.sleep(300)
            self.log.info("Decommission added node")
            decommission_timeout = 7200
            monitoring_decommission_timeout = decommission_timeout + 100
            un_nodes = self.cluster.get_nodes_up_and_normal()
            with Nemesis.run_nemesis(node_list=un_nodes, nemesis_label="BootstrapStreaminError") as verification_node, \
                    FailedDecommissionOperationMonitoring(target_node=new_node, verification_node=verification_node,
                                                          timeout=monitoring_decommission_timeout):

                self.cluster.decommission(new_node, timeout=decommission_timeout)

    def disrupt_disable_binary_gossip_execute_major_compaction(self):
        with nodetool_context(node=self.target_node, start_command="disablebinary", end_command="enablebinary"):
            self.target_node.run_nodetool("statusbinary")
            self.target_node.run_nodetool("status")
            time.sleep(5)
            with nodetool_context(node=self.target_node, start_command="disablegossip", end_command="enablegossip"):
                self.target_node.run_nodetool("statusgossip")
                self.target_node.run_nodetool("status")
                time.sleep(30)
                self._major_compaction()
        self.target_node.run_nodetool("statusgossip")
        self.target_node.run_nodetool("statusbinary")
        time.sleep(30)
        try:
            self.cluster.wait_for_nodes_up_and_normal(nodes=[self.target_node])
            self.target_node.run_cqlsh(
                "SELECT * FROM system_schema.keyspaces;", num_retry_on_failure=20, retry_interval=3)
        except Exception:  # pylint: disable=broad-except
            # NOTE: restart the target node because it was the remedy for the problems with CQL workability
            self.log.warning("'%s' node will be restarted to make the CQL work again", self.target_node)
            self.target_node.restart_scylla_server()
            raise

    @target_all_nodes
    def disrupt_grow_shrink_zero_nodes(self):
        """"Add/remove znodes to same dc where target node. The target node could be any node"""
        if not self.cluster.params.get('use_zero_nodes'):
            raise UnsupportedNemesis("The zero tokens support is not enabled")

        duration_with_znode = 300
        new_znode = self._add_and_init_new_cluster_nodes(count=1, is_zero_node=True)[0]
        self.log.debug("Run with zero-token node %s for %ds", new_znode.name, duration_with_znode)
        time.sleep(duration_with_znode)
        znode = random.choice([node for node in self.cluster.zero_nodes if node.dc_idx == self.target_node.dc_idx])
        self.decommission_nodes(nodes=[znode])

    @target_all_nodes
    def disrupt_serial_restart_elected_topology_coordinator(self):
        """ Serial restart of elected topology coordinator node,
        should trigger new coordinator node election
        """
        if not self.target_node.raft.is_consistent_topology_changes_enabled:
            raise UnsupportedNemesis("Consistent topology changes feature is disabled")

        self.use_nemesis_seed()
        num_of_restarts = random.randint(1, len(self.cluster.nodes))
        self.log.debug("Number of serial restart of topology coordinator: %s", num_of_restarts)
        election_wait_timeout = random.choice([5, 10, 15])
        self.log.debug("Wait new topology coordinator election timeout: %s", election_wait_timeout)
        for num_of_restart in range(num_of_restarts):
            with self.run_nemesis(node_list=self.cluster.nodes, nemesis_label="search coordinator") as verification_node:
                coordinator_node = get_topology_coordinator_node(verification_node)
            if coordinator_node != self.target_node and coordinator_node.running_nemesis:
                raise UnsupportedNemesis(
                    f"Coordinator node is busy with {coordinator_node.running_nemesis}, Coordinator node was restarted: {num_of_restart}")
            elif coordinator_node != self.target_node:
                self.switch_target_node(coordinator_node)
            self.log.debug("Coordinator node: %s, %s", coordinator_node, coordinator_node.name)
            self.target_node.stop_scylla()
            self.log.debug("Wait random timeout %s to new coordinator will be elected", election_wait_timeout)
            time.sleep(election_wait_timeout)
            with self.run_nemesis(node_list=self.cluster.nodes,
                                  nemesis_label="search coordinator") as verification_node:
                new_coordinator_node = get_topology_coordinator_node(verification_node)
            self.log.debug("New coordinator node: %s, %s", new_coordinator_node, new_coordinator_node.name)
            self.target_node.start_scylla()
            assert self.target_node != new_coordinator_node, \
                f"New coordinator node was not elected while old one {coordinator_node.name} was stopped"


def disrupt_method_wrapper(method, is_exclusive=False):  # pylint: disable=too-many-statements  # noqa: PLR0915
    """
    Log time elapsed for method to run

    :param method: Remote method to wrap.
    :return: Wrapped method.
    """

    def argus_create_nemesis_info(nemesis: Nemesis, class_name: str, method_name: str, start_time: int | float) -> bool:
        try:
            argus_client = nemesis.target_node.test_config.argus_client()
            argus_client.submit_nemesis(
                name=method_name,
                class_name=class_name,
                start_time=int(start_time),
                target_name=nemesis.target_node.name,
                target_ip=nemesis.target_node.public_ip_address,
                target_shards=nemesis.target_node.scylla_shards,
            )
            return True
        except Exception:  # pylint: disable=broad-except
            nemesis.log.error("Error creating nemesis information in Argus", exc_info=True)
        return False

    def argus_finalize_nemesis_info(nemesis: Nemesis, method_name: str, start_time: int, nemesis_event: DisruptionEvent):
        if not isinstance(start_time, int):
            start_time = int(start_time)
        try:
            argus_client = nemesis.cluster.test_config.argus_client()
            if nemesis_event.severity == Severity.ERROR:
                argus_client.finalize_nemesis(name=method_name, start_time=start_time,
                                              status=NemesisStatus.FAILED, message=nemesis_event.full_traceback)
            elif nemesis_event.is_skipped:
                argus_client.finalize_nemesis(name=method_name, start_time=start_time,
                                              status=NemesisStatus.SKIPPED, message=nemesis_event.skip_reason)
            else:
                argus_client.finalize_nemesis(name=method_name, start_time=start_time,
                                              status=NemesisStatus.SUCCEEDED, message="")
        except Exception:  # pylint: disable=broad-except
            nemesis.log.error("Error finalizing nemesis information in Argus", exc_info=True)

    def get_nemesis_status(nemesis_event: DisruptionEvent) -> str:
        if nemesis_event.severity == Severity.ERROR:
            return NemesisStatus.FAILED
        if nemesis_event.is_skipped:
            return NemesisStatus.SKIPPED
        return NemesisStatus.SUCCEEDED

    def data_validation_prints(args):
        try:
            if hasattr(args[0].tester, 'data_validator') and args[0].tester.data_validator:
                if not (keyspace := args[0].tester.data_validator.keyspace_name):
                    DataValidatorEvent.DataValidator(severity=Severity.NORMAL,
                                                     message="Failed fo get keyspace name. Data validator is disabled."
                                                     ).publish()
                    return

                with args[0].cluster.cql_connection_patient(args[0].cluster.nodes[0], keyspace=keyspace) as session:
                    args[0].tester.data_validator.validate_range_not_expected_to_change(session, during_nemesis=True)
                    args[0].tester.data_validator.validate_range_expected_to_change(session, during_nemesis=True)
                    args[0].tester.data_validator.validate_deleted_rows(session, during_nemesis=True)
        except Exception as err:  # pylint: disable=broad-except  # noqa: BLE001
            args[0].log.debug(f'Data validator error: {err}')

    @wraps(method)
    def wrapper(*args, **kwargs):  # pylint: disable=too-many-statements  # noqa: PLR0914, PLR0915
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        method_name = method.__name__
        target_pool_type = getattr(method, DISRUPT_POOL_PROPERTY_NAME, NEMESIS_TARGET_POOLS.data_nodes)
        nemesis_run_info_key = f"{id(args[0])}--{method_name}"
        try:
            NEMESIS_LOCK.acquire()  # pylint: disable=consider-using-with
            if not is_exclusive:
                NEMESIS_RUN_INFO[nemesis_run_info_key] = "Running"
                NEMESIS_LOCK.release()
            else:
                while NEMESIS_RUN_INFO:
                    # NOTE: exclusive nemesis will wait before the end of all other ones
                    time.sleep(10)

            args[0].cluster.check_cluster_health()
            num_data_nodes_before = len(args[0].cluster.data_nodes)
            num_zero_nodes_before = len(args[0].cluster.zero_nodes)
            start_time = time.time()

            current_disruption = "".join(p.capitalize() for p in method_name.replace("disrupt_", "").split("_"))
            args[0].set_target_node_pool_type(target_pool_type)
            args[0].set_target_node(current_disruption=current_disruption)
            start_msg = (f"Started disruption {method_name} ({current_disruption} nemesis) on the target node "
                         f"'{str(args[0].target_node)}'")
            args[0].log.debug("{start_symbol} {msg} {start_symbol}".format(start_symbol='>' * 12, msg=start_msg))
            args[0].cluster.log_message(
                "{start_symbol} {msg} {start_symbol}".format(start_symbol='=' * 12, msg=start_msg))

            class_name = args[0].get_class_name()
            if class_name.find('Chaos') < 0:
                args[0].metrics_srv.event_start(class_name)
            result = None
            status = True
            # pylint: disable=protected-access

            log_info = {
                'operation': args[0].current_disruption,
                'start': int(start_time),
                'end': 0,
                'duration': 0,
                'node': str(args[0].target_node),
                'subtype': 'end',
            }
            # TODO: Temporary print. Will be removed later
            data_validation_prints(args=args)

            with DisruptionEvent(nemesis_name=args[0].get_disrupt_name(),
                                 node=args[0].target_node, publish_event=True) as nemesis_event:
                nemesis_info = argus_create_nemesis_info(nemesis=args[0], class_name=class_name,
                                                         method_name=method_name, start_time=start_time)
                try:
                    result = method(*args[1:], **kwargs)
                except (UnsupportedNemesis, MethodVersionNotFound) as exp:
                    skip_reason = str(exp)
                    log_info.update({'subtype': 'skipped', 'skip_reason': skip_reason})
                    nemesis_event.skip(skip_reason=skip_reason)
                    raise
                except KillNemesis:
                    if args[0].tester.get_event_summary().get('CRITICAL', 0):
                        error_sting = "Killed by tearDown - test fail"
                        nemesis_event.add_error([error_sting])
                        nemesis_event.full_traceback = traceback.format_exc()
                        nemesis_event.severity = Severity.ERROR
                        log_info.update({'error': error_sting, 'full_traceback': traceback.format_exc()})
                        status = False
                    else:
                        skip_reason = "Killed by tearDown - test success"
                        log_info.update({'subtype': 'skipped', 'skip_reason': skip_reason})
                        nemesis_event.skip(skip_reason=skip_reason)
                    raise
                except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
                    nemesis_event.add_error([str(details)])
                    nemesis_event.full_traceback = traceback.format_exc()
                    nemesis_event.severity = Severity.ERROR
                    args[0].error_list.append(str(details))
                    args[0].log.error('Unhandled exception in method %s', method, exc_info=True)
                    log_info.update({'error': str(details), 'full_traceback': traceback.format_exc()})
                    status = False
                finally:
                    end_time = time.time()
                    time_elapsed = int(end_time - start_time)
                    log_info.update({
                        'end': int(end_time),
                        'duration': time_elapsed,
                    })
                    args[0].duration_list.append(time_elapsed)
                    args[0].operation_log.append(copy.deepcopy(log_info))
                    args[0].log.debug('%s duration -> %s s', args[0].current_disruption, time_elapsed)

                    if class_name.find('Chaos') < 0:
                        args[0].metrics_srv.event_stop(class_name)
                    disrupt = args[0].get_disrupt_name()
                    del log_info['operation']

                    try:  # So that the nemesis thread won't stop due to elasticsearch failure
                        args[0].update_stats(disrupt, status, log_info)
                    except ElasticSearchConnectionTimeout as err:
                        args[0].log.warning(f"Connection timed out when attempting to update elasticsearch statistics:\n"
                                            f"{err}")
                    except Exception as err:  # pylint: disable=broad-except  # noqa: BLE001
                        args[0].log.warning(f"Unexpected error when attempting to update elasticsearch statistics:\n"
                                            f"{err}")
                    args[0].log.info(f"log_info: {log_info}")
                    nemesis_event.duration = time_elapsed

                    if nemesis_info:
                        argus_finalize_nemesis_info(nemesis=args[0], method_name=method_name, start_time=int(
                            start_time), nemesis_event=nemesis_event)

                    end_msg = (f"Finished disruption {method_name} ({current_disruption} nemesis) with status "
                               f"'{get_nemesis_status(nemesis_event)}'")
                    args[0].log.debug("{end_symbol} {msg} {end_symbol}".format(end_symbol='<' * 12, msg=end_msg))
                    args[0].cluster.log_message(
                        "{end_symbol} {msg} {end_symbol}".format(end_symbol='=' * 12, msg=end_msg))

            args[0].cluster.check_cluster_health()
            num_data_nodes_after = len(args[0].cluster.data_nodes)
            num_zero_nodes_after = len(args[0].cluster.zero_nodes)
            if num_data_nodes_before != num_data_nodes_after:
                args[0].log.error('num data nodes before %s and data nodes after %s does not match' %
                                  (num_data_nodes_before, num_data_nodes_after))
            if args[0].cluster.params.get("use_zero_nodes") and num_zero_nodes_before != num_zero_nodes_after:
                args[0].log.error('num zero nodes before %s and zero nodes after %s does not match' %
                                  (num_zero_nodes_before, num_zero_nodes_after))
            # TODO: Temporary print. Will be removed later
            data_validation_prints(args=args)
        finally:
            if is_exclusive:
                # NOTE: sleep the nemesis interval here because the next one is already
                #       ready to start right after the lock gets released.
                if args[0].tester.params.get('k8s_tenants_num') > 1:
                    args[0].log.debug(
                        "Exclusive nemesis: Sleep for '%s' seconds",
                        args[0].interval)
                    time.sleep(args[0].interval)
                NEMESIS_LOCK.release()
            else:
                # NOTE: the key may be absent if a nemesis which waits for a lock release
                #       gets killed/aborted. So, use safe 'pop' call with the default 'None' value.
                NEMESIS_RUN_INFO.pop(nemesis_run_info_key, None)

            args[0].set_target_node_pool_type(NEMESIS_TARGET_POOLS.data_nodes)

        return result

    return wrapper


class SslHotReloadingNemesis(Nemesis):
    disruptive = False
    config_changes = True

    def disrupt(self):
        self.disrupt_hot_reloading_internode_certificate()


class PauseLdapNemesis(Nemesis):
    disruptive = False
    limited = True

    def disrupt(self):
        self.disrupt_ldap_connection_toggle()


class ToggleLdapConfiguration(Nemesis):
    disruptive = True
    limited = True

    def disrupt(self):
        self.disrupt_disable_enable_ldap_authorization()


class NoOpMonkey(Nemesis):
    kubernetes = True

    def disrupt(self):
        time.sleep(300)


class AddRemoveDcNemesis(Nemesis):

    disruptive = True
    kubernetes = False
    run_with_gemini = False
    limited = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_add_remove_dc()


class GrowClusterNemesis(Nemesis):
    disruptive = True
    kubernetes = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_grow_cluster()


class GrowFillNemesis(Nemesis):
    disruptive = True
    kubernetes = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_grow_fill()


class GrowShrinkClusterNemesis(Nemesis):
    disruptive = True
    kubernetes = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_grow_shrink_cluster()


class GrowShrinkDatacenterNemesis(Nemesis):
    disruptive = True
    kubernetes = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_grow_shrink_datacenter()


class AddRemoveRackNemesis(Nemesis):
    disruptive = True
    kubernetes = True
    config_changes = True

    def disrupt(self):
        self.disrupt_grow_shrink_new_rack()


class StopWaitStartMonkey(Nemesis):
    disruptive = True
    kubernetes = True
    limited = True
    zero_node_changes = True

    def disrupt(self):
        self.disrupt_stop_wait_start_scylla_server(600)


class StopStartMonkey(Nemesis):
    disruptive = True
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_stop_start_scylla_server()


class EnableDisableTableEncryptionAwsKmsProviderWithRotationMonkey(Nemesis):
    disruptive = True
    kubernetes = False  # Enable it when EKS SCT code starts supporting the KMS service

    def disrupt(self):
        self.disrupt_enable_disable_table_encryption_aws_kms_provider_with_rotation()


class EnableDisableTableEncryptionAwsKmsProviderWithoutRotationMonkey(Nemesis):
    disruptive = True
    kubernetes = False  # Enable it when EKS SCT code starts supporting the KMS service

    def disrupt(self):
        self.disrupt_enable_disable_table_encryption_aws_kms_provider_without_rotation()


class EnableDisableTableEncryptionAwsKmsProviderMonkey(Nemesis):
    disruptive = True
    kubernetes = False  # Enable it when EKS SCT code starts supporting the KMS service

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = [
            'disrupt_enable_disable_table_encryption_aws_kms_provider_without_rotation',
            'disrupt_enable_disable_table_encryption_aws_kms_provider_with_rotation',
        ]

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list, predefined_sequence=True)


class RestartThenRepairNodeMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_restart_then_repair_node()


class MultipleHardRebootNodeMonkey(Nemesis):
    disruptive = True
    kubernetes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_multiple_hard_reboot_node()


class HardRebootNodeMonkey(Nemesis):
    disruptive = True
    kubernetes = True
    limited = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_hard_reboot_node()


class SoftRebootNodeMonkey(Nemesis):
    disruptive = True
    kubernetes = True
    limited = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_soft_reboot_node()


class DrainerMonkey(Nemesis):
    disruptive = True
    kubernetes = True
    limited = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_nodetool_drain()


class CorruptThenRepairMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_destroy_data_then_repair()


class CorruptThenRebuildMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_destroy_data_then_rebuild()


class DecommissionMonkey(Nemesis):
    disruptive = True
    limited = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_nodetool_decommission()


class DecommissionSeedNode(Nemesis):
    disruptive = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_nodetool_seed_decommission()


class NoCorruptRepairMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_no_corrupt_repair()


class MajorCompactionMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_major_compaction()


class RefreshMonkey(Nemesis):
    disruptive = False
    run_with_gemini = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_nodetool_refresh(big_sstable=False)


class LoadAndStreamMonkey(Nemesis):
    disruptive = False
    run_with_gemini = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_load_and_stream()


class RefreshBigMonkey(Nemesis):
    disruptive = False
    run_with_gemini = False
    kubernetes = True

    def disrupt(self):
        self.disrupt_nodetool_refresh(big_sstable=True)


class RemoveServiceLevelMonkey(Nemesis):
    disruptive = True
    sla = True

    def disrupt(self):
        self.disrupt_remove_service_level_while_load()


class EnospcMonkey(Nemesis):
    disruptive = True
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_nodetool_enospc()


class EnospcAllNodesMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_nodetool_enospc(all_nodes=True)


class NodeToolCleanupMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_nodetool_cleanup()


class DropMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_drop()


class DropPerfMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_drop_perf()


class TruncateMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_truncate()


class TruncatePerfMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_truncate_perf()


class TruncateLargeParititionMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_truncate_large_partition()


class DeleteByPartitionsMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    free_tier_set = True
    delete_rows = True

    def disrupt(self):
        self.disrupt_delete_10_full_partitions()


class DeleteByRowsRangeMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    free_tier_set = True
    delete_rows = True

    def disrupt(self):
        self.disrupt_delete_by_rows_range()


class DeleteOverlappingRowRangesMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    free_tier_set = True
    delete_rows = True

    def disrupt(self):
        self.disrupt_delete_overlapping_row_ranges()


class ChaosMonkey(Nemesis):

    def disrupt(self):
        self.call_random_disrupt_method()


class CategoricalMonkey(Nemesis):
    """Randomly picks disruptions to execute using the given categorical distribution.

    Each disruption is assigned a weight. The probability that a disruption D with weight W
    will be executed is W / T, where T is the sum of weights of all disruptions.

    The distribution is passed into the monkey's constructor as a dictionary.
    Keys in the dictionary are names of the disruption methods (from the `Nemesis` class)
    e.g. `disrupt_hard_reboot_node`. The value for each key is the weight of this disruption.
    You can omit the ``disrupt_'' prefix from the key, e.g. `hard_reboot_node`.

    A default weight can be passed; it will be assigned to each disruption that is not listed.
    In particular if the default weight is 0 then the unlisted disruptions won't be executed.
    """

    @staticmethod
    def get_disruption_distribution(dist: dict, default_weight: float) -> Tuple[List[Callable], List[float]]:
        def is_nonnegative_number(val):
            try:
                val = float(val)
            except ValueError:
                return False
            else:
                return val >= 0

        def prefixed(pref: str, val: str) -> str:
            if val.startswith(pref):
                return val
            return pref + val

        all_methods = CategoricalMonkey.get_disrupt_methods()

        population: List[Callable] = []
        weights: List[float] = []
        listed_methods: Set[str] = set()

        for _name, _weight in dist.items():
            name = str(_name)
            prefixed_name = prefixed('disrupt_', name)
            if prefixed_name not in all_methods:
                raise ValueError(f"'{name}' is not a valid disruption. All methods: {all_methods.keys()}")

            if not is_nonnegative_number(_weight):
                raise ValueError("Each disruption weight must be a non-negative number."
                                 " '{weight}' is not a valid weight.")

            weight = float(_weight)
            if weight > 0:
                population.append(all_methods[prefixed_name])
                weights.append(weight)
            listed_methods.add(prefixed_name)

        if default_weight > 0:
            for method_name, method in all_methods.items():
                if method_name not in listed_methods:
                    population.append(method)
                    weights.append(default_weight)

        if not population:
            raise ValueError("There must be at least one disruption with a positive weight.")

        return population, weights

    @staticmethod
    def get_disrupt_methods() -> Dict[str, Callable]:
        return {attr[0]: attr[1] for attr in inspect.getmembers(CategoricalMonkey) if
                attr[0].startswith('disrupt_') and
                callable(attr[1])}

    def __init__(self, tester_obj, termination_event, dist: dict, *args, default_weight: float = 1, **kwargs):
        super().__init__(tester_obj, termination_event, *args, **kwargs)
        self.disruption_distribution = CategoricalMonkey.get_disruption_distribution(dist, default_weight)

    def disrupt(self):
        self._random_disrupt()

    def _random_disrupt(self):
        population, weights = self.disruption_distribution
        assert len(population) == len(weights) and population

        method = random.choices(population, weights=weights)[0]
        bound_method = method.__get__(self, CategoricalMonkey)
        self.execute_disrupt_method(bound_method)


class LimitedChaosMonkey(Nemesis):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(limited=True)

    def disrupt(self):
        # Limit the nemesis scope:
        #  - NodeToolCleanupMonkey
        #  - DecommissionMonkey
        #  - DrainerMonkey
        #  - RefreshMonkey
        #  - StopStartMonkey
        #  - MajorCompactionMonkey
        #  - ModifyTableMonkey
        #  - EnospcMonkey
        #  - StopWaitStartMonkey
        #  - HardRebootNodeMonkey
        #  - SoftRebootNodeMonkey
        #  - TruncateMonkey
        #  - TopPartitions
        #  - MgmtCorruptThenRepair
        #  - MgmtRepair
        #  - NoCorruptRepairMonkey
        #  - SnapshotOperations
        #  - AbortRepairMonkey
        #  - MgmtBackup
        #  - MgmtBackupSpecificKeyspaces
        #  - AddDropColumnMonkey
        #  - PauseLdapNemesis
        #  - ToggleLdapConfiguration
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


CLOUD_LIMITED_CHAOS_MONKEY = ['disrupt_nodetool_cleanup',
                              'disrupt_nodetool_drain', 'disrupt_nodetool_refresh',
                              'disrupt_stop_start_scylla_server', 'disrupt_major_compaction',
                              'disrupt_modify_table', 'disrupt_nodetool_enospc',
                              'disrupt_stop_wait_start_scylla_server',
                              'disrupt_soft_reboot_node',
                              'disrupt_truncate']


class ScyllaCloudLimitedChaosMonkey(Nemesis):

    def disrupt(self):
        # Limit the nemesis scope to only one relevant to scylla cloud, where we defined we don't have AWS api access:
        self.call_random_disrupt_method(disrupt_methods=CLOUD_LIMITED_CHAOS_MONKEY)


class AllMonkey(Nemesis):

    def disrupt(self):
        self.call_random_disrupt_method(predefined_sequence=True)


class MdcChaosMonkey(Nemesis):

    def disrupt(self):
        self.call_random_disrupt_method(
            disrupt_methods=['disrupt_destroy_data_then_repair', 'disrupt_no_corrupt_repair',
                             'disrupt_nodetool_decommission'])


class UpgradeNemesis(Nemesis):

    # # upgrade a single node
    # def upgrade_node(self, node):
    #     repo_file = self.cluster.params.get('repo_file', None,  'scylla.repo.upgrade')
    #     new_version = self.cluster.params.get('new_version', None,  '')
    #     upgrade_node_packages = self.cluster.params.get('upgrade_node_packages')
    #     self.log.info('Upgrading a Node')
    #
    #     # We assume that if update_db_packages is not empty we install packages from there.
    #     # In this case we don't use upgrade based on repo_file(ignored sudo yum update scylla...)
    #     orig_ver = node.remoter.run('rpm -qa scylla-server')
    #     if upgrade_node_packages:
    #         # update_scylla_packages
    #         node.remoter.send_files(upgrade_node_packages, '/tmp/scylla', verbose=True)
    #         # node.remoter.run('sudo yum update -y --skip-broken', connect_timeout=900)
    #         node.remoter.run('sudo yum install python34-PyYAML -y')
    #         # replace the packages
    #         node.remoter.run('rpm -qa scylla\*')
    #         node.run_nodetool("snapshot")
    #         # update *development* packages
    #         node.remoter.run('sudo rpm -UvhR --oldpackage /tmp/scylla/*development*', ignore_status=True)
    #         # and all the rest
    #         node.remoter.run('sudo rpm -URvh --replacefiles /tmp/scylla/*.rpm | true')
    #         node.remoter.run('rpm -qa scylla\*')
    #     elif repo_file:
    #         scylla_repo = get_data_dir_path(repo_file)
    #         node.remoter.send_files(scylla_repo, '/tmp/scylla.repo', verbose=True)
    #         node.remoter.run('sudo cp /etc/yum.repos.d/scylla.repo ~/scylla.repo-backup')
    #         node.remoter.run('sudo cp /tmp/scylla.repo /etc/yum.repos.d/scylla.repo')
    #         # backup the data
    #         node.remoter.run('sudo cp /etc/scylla/scylla.yaml /etc/scylla/scylla.yaml-backup')
    #         node.run_nodetool("snapshot")
    #         node.remoter.run('sudo chown root.root /etc/yum.repos.d/scylla.repo')
    #         node.remoter.run('sudo chmod 644 /etc/yum.repos.d/scylla.repo')
    #         node.remoter.run('sudo yum clean all')
    #         ver_suffix = '-{}'.format(new_version) if new_version else ''
    #         node.remoter.run('sudo yum install scylla{0} scylla-server{0} scylla-jmx{0} scylla-tools{0}'
    #                          ' scylla-conf{0} scylla-kernel-conf{0} scylla-debuginfo{0} -y'.format(ver_suffix))
    #     # flush all memtables to SSTables
    #     node.run_nodetool("drain", timeout=3600, coredump_on_timeout=True)
    #     node.remoter.run('sudo systemctl restart scylla-server.service')
    #     node.wait_db_up(verbose=True)
    #     new_ver = node.remoter.run('rpm -qa scylla-server')
    #     if orig_ver == new_ver:
    #         self.log.error('scylla-server version isn\'t changed')

    def disrupt(self):
        self.log.info('Upgrade Nemesis begin')
        # get the number of nodes
        nodes_num = len(self.cluster.nodes)
        # prepare an array containing the indexes
        indexes = list(range(nodes_num))
        # shuffle it so we will upgrade the nodes in a
        # random order
        random.shuffle(indexes)

        # upgrade all the nodes in random order
        for i in indexes:
            node = self.cluster.nodes[i]
            self.upgrade_node(node)  # pylint: disable=no-member

        self.log.info('Upgrade Nemesis end')


class UpgradeNemesisOneNode(UpgradeNemesis):

    def disrupt(self):
        self.log.info('UpgradeNemesisOneNode begin')
        self.upgrade_node(self.cluster.node_to_upgrade)  # pylint: disable=no-member

        self.log.info('UpgradeNemesisOneNode end')


class RollbackNemesis(Nemesis):

    def rollback_node(self, node):
        self.log.info('Rollbacking a Node')
        orig_ver = node.remoter.run('rpm -qa scylla-server')
        node.remoter.run('sudo cp ~/scylla.repo-backup /etc/yum.repos.d/scylla.repo')
        # backup the data
        node.run_nodetool("snapshot")
        node.remoter.run('sudo chown root.root /etc/yum.repos.d/scylla.repo')
        node.remoter.run('sudo chmod 644 /etc/yum.repos.d/scylla.repo')
        node.remoter.run('sudo yum clean all')
        node.remoter.run(
            'sudo yum downgrade scylla scylla-server scylla-jmx scylla-tools scylla-conf scylla-kernel-conf scylla-debuginfo -y')
        # flush all memtables to SSTables
        node.run_nodetool("drain", timeout=15*60, coredump_on_timeout=True)
        node.remoter.run('sudo cp {0}-backup {0}'.format(SCYLLA_YAML_PATH))
        node.remoter.run('sudo systemctl restart scylla-server.service')
        node.wait_db_up(verbose=True)
        new_ver = node.remoter.run('rpm -qa scylla-server')
        self.log.debug('original scylla-server version is %s, latest: %s' % (orig_ver, new_ver))
        if orig_ver == new_ver:
            raise ValueError('scylla-server version isn\'t changed')

    def disrupt(self):
        self.log.info('Rollback Nemesis begin')
        # get the number of nodes
        nodes_num = len(self.cluster.nodes)
        # prepare an array containing the indexes
        indexes = list(range(nodes_num))
        # shuffle it so we will rollback the nodes in a
        # random order
        random.shuffle(indexes)

        # rollback all the nodes in random order
        for i in indexes:
            node = self.cluster.nodes[i]
            self.rollback_node(node)

        self.log.info('Rollback Nemesis end')


class ModifyTableMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True
    schema_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_modify_table()


class AddDropColumnMonkey(Nemesis):
    disruptive = False
    run_with_gemini = False
    networking = False
    kubernetes = True
    limited = True
    schema_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_add_drop_column()


class ToggleTableIcsMonkey(Nemesis):
    kubernetes = True
    schema_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_toggle_table_ics()


class ToggleGcModeMonkey(Nemesis):
    kubernetes = True
    disruptive = False
    schema_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_toggle_table_gc_mode()


class MgmtBackup(Nemesis):
    manager_operation = True
    disruptive = False
    limited = True

    def disrupt(self):
        self.disrupt_mgmt_backup()


class MgmtBackupSpecificKeyspaces(Nemesis):
    manager_operation = True
    disruptive = False
    limited = True

    def disrupt(self):
        self.disrupt_mgmt_backup_specific_keyspaces()


class MgmtRestore(Nemesis):
    manager_operation = True
    disruptive = True
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_mgmt_restore()


class MgmtRepair(Nemesis):
    manager_operation = True
    disruptive = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.log.info('disrupt_mgmt_repair_cli Nemesis begin')
        self.disrupt_mgmt_repair_cli()
        self.log.info('disrupt_mgmt_repair_cli Nemesis end')
        # For Manager APIs test, use: self.disrupt_mgmt_repair_api()


class MgmtCorruptThenRepair(Nemesis):
    manager_operation = True
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_mgmt_corrupt_then_repair()


class AbortRepairMonkey(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_abort_repair()


class NodeTerminateAndReplace(Nemesis):
    disruptive = True
    # It should not be run on kubernetes, since it is a manual procedure
    # While on kubernetes we put it all on scylla-operator
    kubernetes = False
    topology_changes = True
    zero_node_changes = True

    def disrupt(self):
        self.disrupt_terminate_and_replace_node()


class DrainKubernetesNodeThenReplaceScyllaNode(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_drain_kubernetes_node_then_replace_scylla_node()


class TerminateKubernetesHostThenReplaceScyllaNode(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_terminate_kubernetes_host_then_replace_scylla_node()


class DisruptKubernetesNodeThenReplaceScyllaNode(Nemesis):
    disruptive = True
    kubernetes = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = [
            'disrupt_drain_kubernetes_node_then_replace_scylla_node',
            'disrupt_terminate_kubernetes_host_then_replace_scylla_node',
        ]

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class DrainKubernetesNodeThenDecommissionAndAddScyllaNode(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_drain_kubernetes_node_then_decommission_and_add_scylla_node()


class TerminateKubernetesHostThenDecommissionAndAddScyllaNode(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_terminate_kubernetes_host_then_decommission_and_add_scylla_node()


class DisruptKubernetesNodeThenDecommissionAndAddScyllaNode(Nemesis):
    disruptive = True
    kubernetes = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = [
            'disrupt_drain_kubernetes_node_then_decommission_and_add_scylla_node',
            'disrupt_terminate_kubernetes_host_then_decommission_and_add_scylla_node',
        ]

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class K8sSetMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = [
            'disrupt_drain_kubernetes_node_then_replace_scylla_node',
            'disrupt_terminate_kubernetes_host_then_replace_scylla_node',
            'disrupt_drain_kubernetes_node_then_decommission_and_add_scylla_node',
            'disrupt_terminate_kubernetes_host_then_decommission_and_add_scylla_node',
        ]

    def disrupt(self):
        self.call_random_disrupt_method(
            disrupt_methods=self.disrupt_methods_list, predefined_sequence=True)


class OperatorNodeReplace(Nemesis):
    disruptive = True
    kubernetes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_replace_scylla_node_on_kubernetes()


class OperatorNodetoolFlushAndReshard(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_nodetool_flush_and_reshard_on_kubernetes()


class ScyllaKillMonkey(Nemesis):
    disruptive = True
    kubernetes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_kill_scylla()


class ValidateHintedHandoffShortDowntime(Nemesis):
    disruptive = True
    kubernetes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_validate_hh_short_downtime()


class SnapshotOperations(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_snapshot_operations()


class NodeRestartWithResharding(Nemesis):
    disruptive = True
    kubernetes = True
    topology_changes = True
    config_changes = True

    def disrupt(self):
        self.disrupt_restart_with_resharding()


class ClusterRollingRestart(Nemesis):
    disruptive = True
    kubernetes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_rolling_restart_cluster(random_order=False)


class RollingRestartConfigChangeInternodeCompression(Nemesis):
    disruptive = True
    full_cluster_restart = True
    config_changes = True

    def disrupt(self):
        self.disrupt_rolling_config_change_internode_compression()


class ClusterRollingRestartRandomOrder(Nemesis):
    disruptive = True
    kubernetes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_rolling_restart_cluster(random_order=True)


class SwitchBetweenPasswordAuthAndSaslauthdAuth(Nemesis):
    disruptive = True  # the nemesis has rolling restart
    config_changes = True

    def disrupt(self):
        self.disrupt_switch_between_password_authenticator_and_saslauthd_authenticator_and_back()


class TopPartitions(Nemesis):
    disruptive = False
    kubernetes = True
    limited = True

    def disrupt(self):
        self.disrupt_show_toppartitions()


class RandomInterruptionNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False
    kubernetes = True

    def disrupt(self):
        self.disrupt_network_random_interruptions()


class BlockNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False
    kubernetes = True

    def disrupt(self):
        self.disrupt_network_block()


class RejectInterNodeNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False
    free_tier_set = True

    def disrupt(self):
        self.disrupt_network_reject_inter_node_communication()


class RejectNodeExporterNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    def disrupt(self):
        self.disrupt_network_reject_node_exporter()


class RejectThriftNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    def disrupt(self):
        self.disrupt_network_reject_thrift()


class StopStartInterfacesNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    def disrupt(self):
        self.disrupt_network_start_stop_interface()


class DisruptiveMonkey(Nemesis):
    # Limit the nemesis scope:
    #  - ValidateHintedHandoffShortDowntime
    #  - CorruptThenRepairMonkey
    #  - CorruptThenRebuildMonkey
    #  - RestartThenRepairNodeMonkey
    #  - StopStartMonkey
    #  - MultipleHardRebootNodeMonkey
    #  - HardRebootNodeMonkey
    #  - SoftRebootNodeMonkey
    #  - StopWaitStartMonkey
    #  - NodeTerminateAndReplace
    #  - EnospcMonkey
    #  - DecommissionMonkey
    #  - NodeRestartWithResharding
    #  - DrainerMonkey

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(disruptive=True)

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class NonDisruptiveMonkey(Nemesis):
    # Limit the nemesis scope:
    #  - NodeToolCleanupMonkey
    #  - SnapshotOperations
    #  - RefreshMonkey
    #  - RefreshBigMonkey -
    #  - NoCorruptRepairMonkey
    #  - MgmtRepair

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(disruptive=False)

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class NetworkMonkey(Nemesis):
    # Limit the nemesis scope:
    #  - RandomInterruptionNetworkMonkey
    #  - StopStartInterfacesNetworkMonkey
    #  - BlockNetworkMonkey
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(networking=True)

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class GeminiChaosMonkey(Nemesis):
    # Limit the nemesis scope to use with gemini
    # - StopStartMonkey
    # - RestartThenRepairNodeMonkey
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(run_with_gemini=True)

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class GeminiNonDisruptiveChaosMonkey(Nemesis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        run_with_gemini = set(self.get_list_of_methods_compatible_with_backend(run_with_gemini=True))
        non_disruptive = set(self.get_list_of_methods_compatible_with_backend(disruptive=False))
        self.disrupt_methods_list = run_with_gemini.intersection(non_disruptive)

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class ScyllaOperatorBasicOperationsMonkey(Nemesis):
    """
    Selected number of nemesis that is focused on scylla-operator functionality
    """
    disruptive = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disrupt_methods_list = [
            'disrupt_nodetool_flush_and_reshard_on_kubernetes',
            'disrupt_rolling_restart_cluster',
            'disrupt_grow_shrink_cluster',
            'disrupt_grow_shrink_new_rack',
            'disrupt_stop_start_scylla_server',
            'disrupt_drain_kubernetes_node_then_replace_scylla_node',
            'disrupt_terminate_kubernetes_host_then_replace_scylla_node',
            'disrupt_drain_kubernetes_node_then_decommission_and_add_scylla_node',
            'disrupt_terminate_kubernetes_host_then_decommission_and_add_scylla_node',
            'disrupt_replace_scylla_node_on_kubernetes',
            'disrupt_mgmt_corrupt_then_repair',
            'disrupt_mgmt_repair_cli',
            'disrupt_mgmt_backup_specific_keyspaces',
            'disrupt_mgmt_backup',
        ]

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list, predefined_sequence=True)


class NemesisSequence(Nemesis):
    disruptive = True
    networking = False
    run_with_gemini = False

    def disrupt(self):
        self.disrupt_run_unique_sequence()


class TerminateAndRemoveNodeMonkey(Nemesis):
    """Remove a Node from a Scylla Cluster (Down Scale)"""
    disruptive = True
    # It should not be run on kubernetes, since it is a manual procedure
    # While on kubernetes we put it all on scylla-operator
    kubernetes = False
    topology_changes = True

    def disrupt(self):
        self.disrupt_remove_node_then_add_node()


class SisyphusMonkey(Nemesis):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_list_of_disruptions_to_execute()
        self.shuffle_list_of_disruptions()

    def disrupt(self):
        self.call_next_nemesis()


class ToggleCDCMonkey(Nemesis):
    disruptive = False
    schema_changes = True
    config_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_toggle_cdc_feature_properties_on_table()


class CDCStressorMonkey(Nemesis):
    disruptive = False
    free_tier_set = True

    def disrupt(self):
        self.disrupt_run_cdcstressor_tool()


class DecommissionStreamingErrMonkey(Nemesis):

    disruptive = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_decommission_streaming_err()


class RebuildStreamingErrMonkey(Nemesis):

    disruptive = True

    def disrupt(self):
        self.disrupt_rebuild_streaming_err()


class RepairStreamingErrMonkey(Nemesis):

    disruptive = True

    def disrupt(self):
        self.disrupt_repair_streaming_err()


DEPRECATED_LIST_OF_NEMESISES = [UpgradeNemesis, UpgradeNemesisOneNode, RollbackNemesis]

COMPLEX_NEMESIS = [NoOpMonkey, ChaosMonkey,
                   LimitedChaosMonkey,
                   ScyllaCloudLimitedChaosMonkey,
                   AllMonkey, MdcChaosMonkey,
                   DisruptiveMonkey, NonDisruptiveMonkey, GeminiNonDisruptiveChaosMonkey,
                   GeminiChaosMonkey, NetworkMonkey, SisyphusMonkey,
                   DisruptKubernetesNodeThenReplaceScyllaNode,
                   DisruptKubernetesNodeThenDecommissionAndAddScyllaNode,
                   CategoricalMonkey]


class CorruptThenScrubMonkey(Nemesis):
    disruptive = False

    def disrupt(self):
        self.disrupt_corrupt_then_scrub()


class MemoryStressMonkey(Nemesis):
    disruptive = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_memory_stress()


class ResetLocalSchemaMonkey(Nemesis):
    disruptive = False
    config_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_resetlocalschema()


class StartStopMajorCompaction(Nemesis):
    disruptive = False

    def disrupt(self):
        self.disrupt_start_stop_major_compaction()


class StartStopScrubCompaction(Nemesis):
    disruptive = False

    def disrupt(self):
        self.disrupt_start_stop_scrub_compaction()


class StartStopCleanupCompaction(Nemesis):
    disruptive = False

    def disrupt(self):
        self.disrupt_start_stop_cleanup_compaction()


class StartStopValidationCompaction(Nemesis):
    disruptive = False

    def disrupt(self):
        self.disrupt_start_stop_validation_compaction()


class FreeTierSetMonkey(SisyphusMonkey):
    """Nemesis set for testing Scylla Cloud free tier.

    Disruptions that can be caused by random failures and user actions and human operator:
    - doesn't include any topology changes (scale up/down, decommission)
    - doesn't include manager backup/repairs
    - doesn't include k8s nodes related (i.e. one title as exclusive)"""

    def __init__(self, *args, **kwargs):
        # skip SisyphusMonkey __init__ to not repeat build disruption logic, but still we want to run Nemesis class __init__
        super(SisyphusMonkey, self).__init__(*args, **kwargs)  # pylint: disable=bad-super-call
        self.build_list_of_disruptions_to_execute(nemesis_selector=['free_tier_set'])
        self.shuffle_list_of_disruptions()


class SlaIncreaseSharesDuringLoad(Nemesis):
    disruptive = False
    sla = True

    def disrupt(self):
        self.disrupt_sla_increase_shares_during_load()


class SlaDecreaseSharesDuringLoad(Nemesis):
    disruptive = False
    sla = True

    def disrupt(self):
        self.disrupt_sla_decrease_shares_during_load()


class SlaReplaceUsingDetachDuringLoad(Nemesis):
    # TODO: This SLA nemesis uses binary disable/enable workaround that in a test with parallel nemeses can cause to the errors and
    #  failures that is not a problem of Scylla. The option "disruptive" was set to True to prevent irrelevant failures. Should be changed
    #  to False when the issue https://github.com/scylladb/scylla-enterprise/issues/2572 will be fixed.
    disruptive = True
    sla = True

    def disrupt(self):
        self.disrupt_replace_service_level_using_detach_during_load()


class SlaReplaceUsingDropDuringLoad(Nemesis):
    # TODO: This SLA nemesis uses binary disable/enable workaround that in a test with parallel nemeses can cause to the errors and
    #  failures that is not a problem of Scylla. The option "disruptive" was set to True to prevent irrelevant failures. Should be changed
    #  to False when the issue https://github.com/scylladb/scylla-enterprise/issues/2572 will be fixed.
    disruptive = True
    sla = True

    def disrupt(self):
        self.disrupt_replace_service_level_using_drop_during_load()


class SlaIncreaseSharesByAttachAnotherSlDuringLoad(Nemesis):
    # TODO: This SLA nemesis uses binary disable/enable workaround that in a test with parallel nemeses can cause to the errors and
    #  failures that is not a problem of Scylla. The option "disruptive" was set to True to prevent irrelevant failures. Should be changed
    #  to False when the issue https://github.com/scylladb/scylla-enterprise/issues/2572 will be fixed.
    disruptive = True
    sla = True

    def disrupt(self):
        self.disrupt_increase_shares_by_attach_another_sl_during_load()


class SlaMaximumAllowedSlsWithMaxSharesDuringLoad(Nemesis):
    disruptive = False
    sla = True

    def disrupt(self):
        self.disrupt_maximum_allowed_sls_with_max_shares_during_load()


class SlaNemeses(Nemesis):
    disruptive = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_list_of_disruptions_to_execute()
        self.disrupt_methods_list = self.get_list_of_methods_by_flags(sla=True)
        self.shuffle_list_of_disruptions()

    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class CreateIndexNemesis(Nemesis):

    disruptive = False
    schema_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_create_index()


class AddRemoveMvNemesis(Nemesis):

    disruptive = True
    schema_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_add_remove_mv()


class ToggleAuditNemesisSyslog(Nemesis):
    disruptive = True
    schema_changes = True
    config_changes = True
    free_tier_set = True

    def disrupt(self):
        self.disrupt_toggle_audit_syslog()


class BootstrapStreamingErrorNemesis(Nemesis):

    disruptive = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_bootstrap_streaming_error()


class DisableBinaryGossipExecuteMajorCompaction(Nemesis):
    disruptive = True
    kubernetes = True

    def disrupt(self):
        self.disrupt_disable_binary_gossip_execute_major_compaction()


class EndOfQuotaNemesis(Nemesis):
    disruptive = True
    config_changes = True

    def disrupt(self):
        self.disrupt_end_of_quota_nemesis()


class GrowShrinkZeroTokenNode(Nemesis):

    disruptive = True
    schema_changes = False
    free_tier_set = False
    zero_node_changes = True

    def disrupt(self):
        self.disrupt_grow_shrink_zero_nodes()


class ZeroTokenSetMonkey(SisyphusMonkey):
    """Nemesis set for testing Scylla with configured zero nodes

    Disruptions that can be caused by random failures and user actions with
    zero node configured
    """

    def __init__(self, *args, **kwargs):
        super(SisyphusMonkey, self).__init__(*args, **kwargs)  # pylint: disable=bad-super-call
        self.use_all_nodes_as_target = True
        self.build_list_of_disruptions_to_execute(nemesis_selector=['zero_node_changes'])
        self.shuffle_list_of_disruptions()


class SerialRestartOfElectedTopologyCoordinatorNemesis(Nemesis):

    disruptive = True
    topology_changes = True

    def disrupt(self):
        self.disrupt_serial_restart_elected_topology_coordinator()
