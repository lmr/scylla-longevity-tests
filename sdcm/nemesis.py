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
import inspect
import logging
import random
import time
import datetime
import threading
import os
import re
import traceback
import json
from typing import List, Optional, TypedDict, Type, Callable, Tuple, Dict, Set, Union
from functools import wraps, partial
from collections import OrderedDict, defaultdict, Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor

from invoke import UnexpectedExit
from cassandra import ConsistencyLevel

from sdcm.cluster import SCYLLA_YAML_PATH, NodeSetupTimeout, NodeSetupFailed, ClusterNodesNotReady
from sdcm.mgmt import TaskStatus
from sdcm.utils.common import remote_get_file, get_db_tables, generate_random_string, \
    update_certificates, reach_enospc_on_node, clean_enospc_on_node, parse_nodetool_listsnapshots
from sdcm.utils.decorators import retrying, timeout
from sdcm.utils import cdc
from sdcm.utils.decorators import retrying, latency_calculator_decorator
from sdcm.wait import wait_for
from sdcm.log import SDCMAdapter
from sdcm.keystore import KeyStore
from sdcm.prometheus import nemesis_metrics_obj
from sdcm import wait
from sdcm.sct_events import Severity
from sdcm.sct_events.base import LogEvent
from sdcm.sct_events.system import InfoEvent
from sdcm.sct_events.filters import DbEventsFilter, EventsSeverityChangerFilter
from sdcm.sct_events.loaders import CassandraStressLogEvent
from sdcm.sct_events.nemesis import DisruptionEvent
from sdcm.sct_events.database import DatabaseLogEvent
from sdcm.sct_events.decorators import raise_event_on_failure
from sdcm.sct_events.group_common_events import ignore_alternator_client_errors, ignore_no_space_errors
from sdcm.db_stats import PrometheusDBStats
from sdcm.remote.libssh2_client.exceptions import UnexpectedExit as Libssh2UnexpectedExit
from sdcm.cluster_k8s import PodCluster
from sdcm.cluster_k8s.minikube import MinikubeScyllaPodCluster
from sdcm.cluster_k8s.gke import GkeScyllaPodCluster
from sdcm.nemesis_publisher import NemesisElasticSearchPublisher
from test_lib.compaction import CompactionStrategy, get_compaction_strategy, get_compaction_random_additional_params
from test_lib.cql_types import CQLTypeBuilder


class NoFilesFoundToDestroy(Exception):
    pass


class NoKeyspaceFound(Exception):
    pass


class FilesNotCorrupted(Exception):
    pass


class LogContentNotFound(Exception):
    pass


class UnsupportedNemesis(Exception):
    """ raised from within a nemesis execution to skip this nemesis"""


class NoMandatoryParameter(Exception):
    """ raised from within a nemesis execution to skip this nemesis"""


class DefaultValue:
    """
    This is class is intended to be used as default value for the cases when None is not applicable
    """
    pass


class Nemesis:  # pylint: disable=too-many-instance-attributes,too-many-public-methods

    disruptive = False
    run_with_gemini = True
    networking = False
    kubernetes = False
    MINUTE_IN_SEC = 60
    HOUR_IN_SEC = 60 * MINUTE_IN_SEC
    disruptions_list = []
    has_steady_run = False

    def __init__(self, tester_obj, termination_event):
        self.tester = tester_obj  # ClusterTester object
        self.cluster = tester_obj.db_cluster
        self.loaders = tester_obj.loaders
        self.monitoring_set = tester_obj.monitors
        self.target_node = None
        logger = logging.getLogger(__name__)
        self.log = SDCMAdapter(logger, extra={'prefix': str(self)})
        self.termination_event = termination_event
        self.operation_log = []
        self.current_disruption = None
        self.duration_list = []
        self.error_list = []
        self.interval = 60 * self.tester.params.get('nemesis_interval')  # convert from min to sec
        self.start_time = time.time()
        self.stats = {}
        self.metrics_srv = nemesis_metrics_obj()
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
            'scylla_bench': '*',  # Ignore scylla-bench tables
            'alternator_usertable': '*',  # Ignore alternator tables
            'ks_truncate': 'counter1',  # Ignore counter table
            'keyspace1': 'counter1',  # Ignore counter table
            # TODO: issue https://github.com/scylladb/scylla/issues/6074. Waiting for dev conclusions
            'cqlstress_lwt_example': '*'  # Ignore LWT user-profile tables
        }
        self.es_publisher = NemesisElasticSearchPublisher(self.tester)

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
        DisruptionEvent(type=disrupt, status=status, **data).publish()

    def set_current_running_nemesis(self, node):
        node.running_nemesis = self.__class__.__name__

    def unset_current_running_nemesis(self, node):
        if node is not None:
            node.running_nemesis = None

    def _get_target_nodes(
            self,
            is_seed: Optional[Union[bool, DefaultValue]] = DefaultValue,
            dc_idx: Optional[int] = None,
            rack: Optional[int] = None) -> list:
        """
        Filters and return nodes in the cluster that has no running nemesis on them
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
        nodes = [node for node in self.cluster.nodes if not node.running_nemesis]
        if is_seed is not None:
            nodes = [node for node in nodes if node.is_seed == is_seed]
        if dc_idx is not None:
            nodes = [node for node in nodes if node.dc_idx == dc_idx]
        if rack is not None:
            nodes = [node for node in nodes if node.rack == rack]
        return nodes

    def set_target_node(self, dc_idx: Optional[int] = None, rack: Optional[int] = None,
                        is_seed: Union[bool, DefaultValue, None] = DefaultValue):
        """Randomly pick a  node from list of the nodes that fit the filter and set it as target node
        if is_seed is None - it will ignore seed status of the nodes
        if is_seed is True - it will pick only seed nodes
        if is_seed is False - it will pick only non-seed nodes
        if is_seed is DefaultValue - if self.filter_seed is True it act as if is_seed=False,
          otherwise it will act as if is_seed is None
        """
        self.unset_current_running_nemesis(self.target_node)
        nodes = self._get_target_nodes(is_seed=is_seed, dc_idx=dc_idx, rack=rack)
        if not nodes:
            raise UnsupportedNemesis("Can't allocate node to run nemesis on")
        # Set name of nemesis, which is going to run on target node
        self.target_node = random.choice(nodes)
        self.set_current_running_nemesis(node=self.target_node)
        self.log.info('Current Target: %s with running nemesis: %s', self.target_node, self.target_node.running_nemesis)

    def set_last_node_as_target(self, dc_idx: Optional[int] = None, rack: Optional[int] = None,
                                is_seed: Union[bool, DefaultValue, None] = DefaultValue):
        """Pick last node from list of the nodes that fit the filter and set it as target node
        if is_seed is None - it will ignore seed status of the nodes
        if is_seed is True - it will pick only seed nodes
        if is_seed is False - it will pick only non-seed nodes
        if is_seed is DefaultValue - if self.filter_seed is True it act as if is_seed=False,
          otherwise it will act as if is_seed is None
        """
        self.unset_current_running_nemesis(self.target_node)
        target_nodes = self._get_target_nodes(is_seed=is_seed, dc_idx=dc_idx, rack=rack)
        if not target_nodes:
            dc_str = '' if dc_idx is None else f'dc {dc_idx} '
            rack_str = '' if rack is None else f'rack {rack} '
            raise UnsupportedNemesis(f"Can't allocate node from {dc_str}{rack_str}to run nemesis on")
        self.target_node = target_nodes[-1]
        self.set_current_running_nemesis(node=self.target_node)
        self.log.info('Current Target: %s with running nemesis: %s', self.target_node, self.target_node.running_nemesis)

    @raise_event_on_failure
    def run(self, interval=None):
        self.es_publisher.create_es_connection()
        if interval:
            self.interval = interval * 60
        self.log.info('Interval: %s s', self.interval)
        while not self.termination_event.is_set():
            cur_interval = self.interval
            self.set_target_node()
            self._set_current_disruption(report=False)
            try:
                self.disrupt()
            except UnsupportedNemesis:
                cur_interval = 0
            finally:
                self.unset_current_running_nemesis(self.target_node)
                self.termination_event.wait(timeout=cur_interval)

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

    def get_list_of_methods_compatible_with_backend(
            self,
            disruptive: Optional[bool] = None,
            run_with_gemini: Optional[bool] = None,
            networking: Optional[bool] = None) -> List[str]:
        return self.get_list_of_methods_by_flags(
            disruptive=disruptive,
            run_with_gemini=run_with_gemini,
            networking=networking,
            kubernetes=True if self._is_it_on_kubernetes() else None
        )

    def _is_it_on_kubernetes(self) -> bool:
        return isinstance(getattr(self.tester, "db_cluster", None), PodCluster)

    def get_list_of_methods_by_flags(
            self,
            disruptive: Optional[bool] = None,
            run_with_gemini: Optional[bool] = None,
            networking: Optional[bool] = None,
            kubernetes: Optional[bool] = None) -> List[str]:
        attributes = locals()
        flags = {flag_name: attributes[flag_name] for flag_name in
                 ['disruptive', 'run_with_gemini', 'networking', 'kubernetes'] if attributes[flag_name] is not None}
        subclasses_list = self._get_subclasses(**flags)
        disrupt_methods_list = []
        for subclass in subclasses_list:
            method_name = re.search(
                r'self\.(?P<method_name>disrupt_[A-Za-z_]+?)\(.*\)', inspect.getsource(subclass), flags=re.MULTILINE)
            if method_name:
                disrupt_methods_list.append(method_name.group('method_name'))
        self.log.debug("Gathered subclass methods: {}".format(disrupt_methods_list))
        return disrupt_methods_list

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
                if getattr(nemesis, filter_name) != filter_value:
                    matches = False
                    break
            if not matches:
                continue
            nemesis_subclasses.append(nemesis)
        return nemesis_subclasses

    def __str__(self):
        try:
            return str(self.__class__).split("'")[1]
        except Exception:  # pylint: disable=broad-except
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
            self.log.info('Waiting JMX services to be restarted after we killed them...')
            self.target_node.wait_jmx_up()
        self.cluster.wait_for_schema_agreement()

    def disrupt_stop_wait_start_scylla_server(self, sleep_time=300):  # pylint: disable=invalid-name
        self._set_current_disruption('StopWaitStartService %s' % self.target_node)
        self.target_node.stop_scylla_server(verify_up=False, verify_down=True)
        self.log.info("Sleep for %s seconds", sleep_time)
        time.sleep(sleep_time)
        if self._is_it_on_kubernetes():
            # Kubernetes brings node up automatically, no need to start it up
            self.target_node.wait_db_up(timeout=sleep_time)
            return
        self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    def disrupt_stop_start_scylla_server(self):  # pylint: disable=invalid-name
        self._set_current_disruption('StopStartService %s' % self.target_node)
        self.target_node.stop_scylla_server(verify_up=False, verify_down=True)
        self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    # This nemesis should be run with "private" ip_ssh_connections till the issue #665 is not fixed
    def disrupt_restart_then_repair_node(self):  # pylint: disable=invalid-name
        self._set_current_disruption('RestartThenRepairNode %s' % self.target_node)
        # Task https://trello.com/c/llRuLIOJ/2110-add-dbeventfilter-for-nosuchcolumnfamily-error
        # If this error happens during the first boot with the missing disk this issue is expected and it's not an issue
        with DbEventsFilter(db_event=DatabaseLogEvent.DATABASE_ERROR,
                            line="Can't find a column family with UUID", node=self.target_node):
            self.target_node.restart()

        self.log.info('Waiting scylla services to start after node restart')
        self.target_node.wait_db_up(timeout=28800)  # 8 hours
        self.log.info('Waiting JMX services to start after node restart')
        self.target_node.wait_jmx_up()
        self.repair_nodetool_repair()

    def disrupt_hard_reboot_node(self):
        self._set_current_disruption('HardRebootNode %s' % self.target_node)
        self.target_node.reboot(hard=True)
        self.log.info('Waiting scylla services to start after node reboot')
        self.target_node.wait_db_up()
        self.log.info('Waiting JMX services to start after node reboot')
        self.target_node.wait_jmx_up()
        self.cluster.wait_for_nodes_up_and_normal(nodes=[self.target_node])

    def disrupt_multiple_hard_reboot_node(self):  # pylint: disable=invalid-name
        num_of_reboots = random.randint(2, 10)
        for i in range(num_of_reboots):
            self._set_current_disruption('MultipleHardRebootNode %s' % self.target_node)
            self.log.debug("Rebooting {} out of {} times".format(i + 1, num_of_reboots))
            self.target_node.reboot(hard=True)
            if random.choice([True, False]):
                self.log.info('Waiting scylla services to start after node reboot')
                self.target_node.wait_db_up()
            else:
                self.log.info('Waiting JMX services to start after node reboot')
                self.target_node.wait_jmx_up()
            self.cluster.wait_for_nodes_up_and_normal(nodes=[self.target_node])
            sleep_time = random.randint(0, 100)
            self.log.info(
                'Sleep {} seconds after hard reboot and service-up for node {}'.format(sleep_time, self.target_node))
            time.sleep(sleep_time)

    def disrupt_soft_reboot_node(self):
        self._set_current_disruption('SoftRebootNode %s' % self.target_node)
        self.target_node.reboot(hard=False)
        self.log.info('Waiting scylla services to start after node reboot')
        self.target_node.wait_db_up()
        self.log.info('Waiting JMX services to start after node reboot')
        self.target_node.wait_jmx_up()
        self.cluster.wait_for_nodes_up_and_normal(nodes=[self.target_node])

    def disrupt_rolling_restart_cluster(self):
        self._set_current_disruption('RollingRestartCluster %s' % self.target_node)
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('RollingRestartCluster is supported only for kubernetes backends')
        self.cluster.rollout_restart()

    def disrupt_restart_with_resharding(self):
        self._set_current_disruption('RestartNodeWithResharding %s' % self.target_node)
        murmur3_partitioner_ignore_msb_bits = 15  # pylint: disable=invalid-name
        self.log.info(f'Restart node with resharding. New murmur3_partitioner_ignore_msb_bits value: '
                      f'{murmur3_partitioner_ignore_msb_bits}')
        self.target_node.restart_node_with_resharding(
            murmur3_partitioner_ignore_msb_bits=murmur3_partitioner_ignore_msb_bits)
        self.log.info('Waiting scylla services to start after node restart')
        self.target_node.wait_db_up()
        self.log.info('Waiting JMX services to start after node restart')
        self.target_node.wait_jmx_up()

        # Wait 5 minutes our before return back the default value
        self.log.debug(
            'Wait 5 minutes our before return murmur3_partitioner_ignore_msb_bits back the default value (12)')
        time.sleep(360)
        self.log.info('Set back murmur3_partitioner_ignore_msb_bits value to 12')
        self.target_node.restart_node_with_resharding()

    @retrying(n=10, allowed_exceptions=(NoKeyspaceFound, NoFilesFoundToDestroy))
    def _choose_file_for_destroy(self, ks_cfs):
        file_for_destroy = ''

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
            file_name = os.path.basename(one_file)
            # The file name like: /var/lib/scylla/data/scylla_bench/test-f60e4f30c98f11e98d46000000000002/mc-220-big-Data.db
            # For corruption we need to remove all files that their names are started from "mc-220-" (MC format)
            # Old format: "system-truncated-ka-" (system-truncated-ka-7-Data.db)
            # Search for "<digit>-" substring

            try:
                file_name_template = re.search(r"(.*-\d+)-", file_name).group(1)
            except Exception as error:  # pylint: disable=broad-except
                self.log.debug('File name "{file_name}" is not as expected for Scylla data files. '
                               'Search files for "{ks_cf_for_destroy}" table'.format(file_name=file_name,
                                                                                     ks_cf_for_destroy=ks_cf_for_destroy))
                self.log.debug('Error: {}'.format(error))
                continue

            file_for_destroy = one_file.replace(file_name, file_name_template + '-*')
            self.log.debug('Selected files for destroy: {}'.format(file_for_destroy))
            if file_for_destroy:
                break

        if not file_for_destroy:
            raise NoFilesFoundToDestroy('Data file for destroy is not found in {}'.format(ks_cf_for_destroy))

        return file_for_destroy

    def _destroy_data_and_restart_scylla(self):

        ks_cfs = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node)
        if not ks_cfs:
            raise UnsupportedNemesis(
                'Non-system keyspace and table are not found. CorruptThenRepair nemesis can\'t be run')

        # Stop scylla service before deleting sstables to avoid partial deletion of files that are under compaction
        self.target_node.stop_scylla_server(verify_up=False, verify_down=True)

        try:
            # Remove 5 data files
            for _ in range(5):
                file_for_destroy = self._choose_file_for_destroy(ks_cfs)

                result = self.target_node.remoter.sudo('rm -f %s' % file_for_destroy)
                if result.stderr:
                    raise FilesNotCorrupted('Files were not corrupted. CorruptThenRepair nemesis can\'t be run. '
                                            'Error: {}'.format(result))
                self.log.debug('Files {} were destroyed'.format(file_for_destroy))

        finally:
            self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    def disrupt(self):
        raise NotImplementedError('Derived classes must implement disrupt()')

    def get_disrupt_name(self):
        return self.current_disruption.split()[0]

    def get_class_name(self):
        return self.__class__.__name__.replace('Monkey', '')

    def _set_current_disruption(self, label=None, report=True, node=None):
        current_node = node if node else self.target_node

        if not label:
            label = "%s on target node %s" % (self.__class__.__name__, current_node)
        self.log.debug('Set current_disruption -> %s', label)
        self.current_disruption = label
        self.log.info(label)
        if report:
            DisruptionEvent(
                type=self.get_disrupt_name(),
                subtype="start",
                status=True,
                node=current_node
            ).publish()

    def disrupt_destroy_data_then_repair(self):  # pylint: disable=invalid-name
        self._set_current_disruption('CorruptThenRepair %s' % self.target_node)
        self._destroy_data_and_restart_scylla()
        # try to save the node
        self.repair_nodetool_repair()

    def disrupt_destroy_data_then_rebuild(self):  # pylint: disable=invalid-name
        self._set_current_disruption('CorruptThenRebuild %s' % self.target_node)
        self._destroy_data_and_restart_scylla()
        # try to save the node
        self.repair_nodetool_rebuild()

    def disrupt_nodetool_drain(self):
        self._set_current_disruption('Drainer %s' % self.target_node)
        result = self.target_node.run_nodetool("drain", timeout=3600, coredump_on_timeout=True)
        self.target_node.run_nodetool("status", ignore_status=True, verbose=True)

        if result is not None:
            # workaround for issue #7332: don't interrupt test and don't raise exception
            # for UnexpectedExit, Failure and CommandTimedOut if "scylla-server stop" failed
            # or scylla-server was stopped gracefully.
            self.target_node.stop_scylla_server(verify_up=False, verify_down=True, ignore_status=True)
            self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    @retrying(n=3, sleep_time=60, allowed_exceptions=(NodeSetupFailed, NodeSetupTimeout))
    def _add_and_init_new_cluster_node(self, old_node_ip=None, timeout=HOUR_IN_SEC * 6, rack=0):
        """When old_node_private_ip is not None replacement node procedure is initiated"""
        self.log.info("Adding new node to cluster...")
        InfoEvent(message='StartEvent - Adding new node to cluster').publish()
        new_node = self.cluster.add_nodes(
            count=1, dc_idx=self.target_node.dc_idx, enable_auto_bootstrap=True, rack=rack)[0]
        self.monitoring_set.reconfigure_scylla_monitoring()
        self.set_current_running_nemesis(node=new_node)  # prevent to run nemesis on new node when running in parallel
        new_node.replacement_node_ip = old_node_ip
        try:
            self.cluster.wait_for_init(node_list=[new_node], timeout=timeout, check_node_health=False)
            self.cluster.clean_replacement_node_ip(new_node)
        except (NodeSetupFailed, NodeSetupTimeout):
            self.log.warning("Setup of the '%s' failed, removing it from list of nodes" % new_node)
            self.cluster.nodes.remove(new_node)
            self.log.warning("Node will not be terminated. Please terminate manually!!!")
            raise
        self.cluster.wait_for_nodes_up_and_normal(nodes=[new_node])
        InfoEvent(message="FinishEvent - New Node is up and normal").publish()
        return new_node

    def _get_kubernetes_node_break_methods(self):
        if isinstance(self.cluster, GkeScyllaPodCluster):
            return ['drain_k8s_node', 'terminate_k8s_host', 'terminate_k8s_node']
        raise UnsupportedNemesis("Only GkeScyllaPodCluster is supported")

    def _terminate_cluster_node(self, node):
        self.cluster.terminate_node(node)
        self.monitoring_set.reconfigure_scylla_monitoring()

    def disrupt_nodetool_decommission(self, add_node=True, disruption_name=None):
        if self._is_it_on_kubernetes() and disruption_name is None:
            self.set_last_node_as_target()
        self._set_current_disruption(f"{disruption_name or 'Decommission'} {self.target_node}")
        self.cluster.decommission(self.target_node)
        new_node = None
        if add_node:
            # When adding node after decommission the node is declared as up only after it completed bootstrapping,
            # increasing the timeout for now
            new_node = self._add_and_init_new_cluster_node(rack=self.target_node.rack)
            # after decomission and add_node, the left nodes have data that isn't part of their tokens anymore.
            # In order to eliminate cases that we miss a "data loss" bug because of it, we cleanup this data.
            # This fix important when just user profile is run in the test and "keyspace1" doesn't exist.
            try:
                test_keyspaces = self.cluster.get_test_keyspaces()
                for node in self.cluster.nodes:
                    for keyspace in test_keyspaces:
                        node.run_nodetool(sub_cmd='cleanup', args=keyspace)
            finally:
                self.unset_current_running_nemesis(new_node)
        return new_node

    def disrupt_nodetool_seed_decommission(self, add_node=True):
        if len(self.cluster.seed_nodes) < 2:
            raise UnsupportedNemesis("To running seed decommission the cluster must contains at least 2 seed nodes")

        if not self.target_node.is_seed:
            self.target_node = random.choice(self.cluster.seed_nodes)
        self.target_node.is_seed = False
        self.cluster.update_seed_provider()

        new_seed_node = self.disrupt_nodetool_decommission(add_node=add_node, disruption_name="SeedDecommission")
        if new_seed_node and not new_seed_node.is_seed:
            new_seed_node.is_seed = True
            self.cluster.update_seed_provider()

    @latency_calculator_decorator
    def _terminate_and_wait(self, target_node, sleep_time=300):
        self._terminate_cluster_node(target_node)
        time.sleep(sleep_time)  # Sleeping for 5 mins to let the cluster live with a missing node for a while

    @latency_calculator_decorator
    def _init_new_node_with_latency_calculation(self, old_node_ip, rack=0):
        return self._add_and_init_new_cluster_node(old_node_ip, rack=rack)

    @latency_calculator_decorator
    def run_nodetool_repair_on_new_node(self, node):
        self.repair_nodetool_repair(node)

    def disrupt_terminate_and_recover_node_kubernetes(self):  # pylint: disable=invalid-name
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('OperatorNodeTerminateAndRecover is supported only on kubernetes')
        for node_terminate_method_name in self._get_kubernetes_node_break_methods():
            self.set_target_node()
            self._set_current_disruption(
                f'OperatorNodeTerminateAndRecover ({node_terminate_method_name}) {self.target_node}')
            self._disrupt_terminate_and_recover_node_kubernetes(self.target_node, node_terminate_method_name)

    def disrupt_terminate_and_replace_node_kubernetes(self):  # pylint: disable=invalid-name
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('OperatorNodeTerminateAndReplace is supported only on kubernetes')
        for node_terminate_method_name in self._get_kubernetes_node_break_methods():
            self.set_target_node()
            self._set_current_disruption(
                f'OperatorNodeTerminateAndReplace ({node_terminate_method_name}) {self.target_node}')
            self._disrupt_terminate_and_replace_node_kubernetes(self.target_node, node_terminate_method_name)

    def disrupt_terminate_decommission_add_node_kubernetes(self):  # pylint: disable=invalid-name
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('OperatorNodeTerminateDecommissionAdd is supported only on kubernetes')
        for node_terminate_method_name in self._get_kubernetes_node_break_methods():
            self.set_last_node_as_target(rack=random.choice(list(self.cluster.racks)))
            self._set_current_disruption(
                f'OperatorNodeTerminateDecommissionAdd ({node_terminate_method_name}) {self.target_node}')
            self._disrupt_terminate_decommission_add_node_kubernetes(self.target_node, node_terminate_method_name)

    def disrupt_replace_node_kubernetes(self):
        self._set_current_disruption('OperatorNodeReplace %s' % self.target_node)
        if not self._is_it_on_kubernetes():
            raise UnsupportedNemesis('OperatorNodeReplace is supported only on kubernetes')
        old_uid = self.target_node.k8s_pod_uid
        self.log.info(f'TerminateNode {self.target_node}')
        self.log.info(f'Mark {self.target_node} to be removed')
        self.target_node.mark_to_be_replaced()
        self._kubernetes_wait_till_node_up_after_been_recreated(self.target_node, old_uid=old_uid)

    def _disrupt_terminate_and_recover_node_kubernetes(self, node, node_terminate_method_name):  # pylint: disable=invalid-name
        old_uid = node.k8s_pod_uid
        node_terminate_method = getattr(node, node_terminate_method_name)
        node_terminate_method()
        self._kubernetes_wait_till_node_up_after_been_recreated(node, old_uid=old_uid)

    def _disrupt_terminate_decommission_add_node_kubernetes(self, node, node_terminate_method_name):  # pylint: disable=invalid-name
        self.log.info(f'Terminate {node}')
        node_terminate_method = getattr(node, node_terminate_method_name)
        node_terminate_method()
        self.log.info(f'Decommission {node}')
        self.cluster.decommission(node)
        self.add_new_node(rack=node.rack)

    def _disrupt_terminate_and_replace_node_kubernetes(self, node, node_terminate_method_name):  # pylint: disable=invalid-name
        old_uid = node.k8s_pod_uid
        self.log.info(f'TerminateNode {node}')
        node_terminate_method = getattr(node, node_terminate_method_name)
        node_terminate_method()
        self.log.info(f'Mark {node} to be removed')
        node.mark_to_be_replaced()
        self._kubernetes_wait_till_node_up_after_been_recreated(node, old_uid=old_uid)

    def _kubernetes_wait_till_node_up_after_been_recreated(self, node, old_uid=None):
        node.wait_k8s_pod_uid(old_uid=old_uid)
        self.log.info(f'Wait till {node} is ready')
        node.wait_for_pod_readiness()
        self.log.info(f'{node} is ready, updating ip address and monitoring')
        node.refresh_ip_address()
        self.monitoring_set.reconfigure_scylla_monitoring()

    def disrupt_terminate_and_replace_node(self):  # pylint: disable=invalid-name
        # using "Replace a Dead Node" procedure from http://docs.scylladb.com/procedures/replace_dead_node/
        self._set_current_disruption('TerminateAndReplaceNode %s' % self.target_node)
        old_node_ip = self.target_node.ip_address
        InfoEvent(message='StartEvent - Terminate node and wait 5 minutes').publish()
        self._terminate_and_wait(target_node=self.target_node)
        InfoEvent(message='FinishEvent - target_node was terminated').publish()
        new_node = self._add_and_init_new_cluster_node(old_node_ip, rack=self.target_node.rack)
        try:
            if new_node.get_scylla_config_param("enable_repair_based_node_ops") == 'false':
                InfoEvent(message='StartEvent - Run repair on new node').publish()
                self.run_nodetool_repair_on_new_node(new_node)
                InfoEvent(message='FinishEvent - Finished running repair on new node').publish()

            # wait until node gives up on the old node, the default timeout is `ring_delay_ms: 300000`
            # scylla: [shard 0] gossip - FatClient 10.0.22.115 has been silent for 30000ms, removing from gossip
            @retrying(n=20, sleep_time=20, allowed_exceptions=(AssertionError,))
            def wait_for_old_node_to_removed():
                status = self.cluster.get_nodetool_status(verification_node=new_node)
                nodes_ips = [ip for dc in status.values() for ip in dc.keys()]
                assert old_node_ip not in nodes_ips, f"'{old_node_ip}' shouldn't be in: {nodes_ips}"

            wait_for_old_node_to_removed()

        finally:
            self.unset_current_running_nemesis(new_node)

    def disrupt_kill_scylla(self):
        self._set_current_disruption('ScyllaKillMonkey %s' % self.target_node)
        self._kill_scylla_daemon()

    def disrupt_no_corrupt_repair(self):
        self._set_current_disruption('NoCorruptRepair %s' % self.target_node)

        # prepare test tables and fill test data
        for i in range(10):
            self.log.debug('Prepare test tables if they do not exist')
            self._prepare_test_table(ks=f'drop_table_during_repair_ks_{i}', table='standard1')

        self.log.debug("Start repair target_node in background")
        thread1 = threading.Thread(target=self.repair_nodetool_repair, name='NodeToolRepairThread', daemon=True)
        thread1.start()

        # drop test tables one by one during repair
        for i in range(10):
            time.sleep(random.randint(0, 300))
            with self.cluster.cql_connection_patient(self.target_node) as session:
                session.execute(f'DROP TABLE drop_table_during_repair_ks_{i}.standard1')
        thread1.join(timeout=120)

    def disrupt_major_compaction(self):
        self._set_current_disruption('MajorCompaction %s' % self.target_node)
        self.target_node.run_nodetool("compact")

    def disrupt_nodetool_refresh(self, big_sstable: bool = False):
        self._set_current_disruption('Refresh keyspace1.standard1 on {}'.format(self.target_node.name))

        # Checking the columns number of keyspace1.standard1
        self.log.debug('Prepare keyspace1.standard1 if it does not exist')
        self._prepare_test_table(ks='keyspace1', table='standard1')
        query_cmd = "SELECT * FROM keyspace1.standard1 LIMIT 1"
        result = self.target_node.run_cqlsh(query_cmd)
        col_num = len(re.findall("(\| C\d+)", result.stdout))

        if col_num == 1:
            # Use special schema (one column) for refresh before https://github.com/scylladb/scylla/issues/6617 is fixed
            if big_sstable:
                sstable_url = 'https://s3.amazonaws.com/scylla-qa-team/refresh_nemesis_c0/keyspace1.standard1.100M.tar.gz'
                sstable_file = '/tmp/keyspace1.standard1.100M.tar.gz'
                sstable_md5 = 'e4f6addc2db8e9af3a906953288ef676'
                keys_num = 1001000
            else:
                sstable_url = 'https://s3.amazonaws.com/scylla-qa-team/refresh_nemesis_c0/keyspace1.standard1.tar.gz'
                sstable_file = "/tmp/keyspace1.standard1.tar.gz"
                sstable_md5 = 'c4aee10691fa6343a786f52663e7f758'
                keys_num = 1000
        elif col_num >= 5:
            # The snapshot has 5 columns, the snapshot (col=5) can be loaded to table (col > 5).
            # they rest columns will be filled to 'null'.
            if big_sstable:
                # 100G, the big file will be saved to GCE image
                # Fixme: It's very slow and unstable to download 100G files from S3 to GCE instances,
                #        currently we actually uploaded a small file (3.6 K) to S3.
                #        We had a solution to save the file in GCE image, it requires bigger boot disk.
                #        In my old test, the instance init is easy to fail. We can try to use a
                #        split shared disk to save the 100GB file.
                # 100M (500000 rows)
                sstable_url = 'https://s3.amazonaws.com/scylla-qa-team/refresh_nemesis/keyspace1.standard1.100M.tar.gz'
                sstable_file = '/tmp/keyspace1.standard1.100M.tar.gz'
                sstable_md5 = '9c5dd19cfc78052323995198b0817270'
                keys_num = 501000
            else:
                sstable_url = 'https://s3.amazonaws.com/scylla-qa-team/refresh_nemesis/keyspace1.standard1.tar.gz'
                sstable_file = "/tmp/keyspace1.standard1.tar.gz"
                sstable_md5 = 'c033a3649a1aec3ba9b81c446c6eecfd'
                keys_num = 1000
        else:
            # Note: when issue #6617 is fixed, we can try to load snapshot (cols=5) to a table (1 < cols < 5),
            #       expect that refresh will fail (no serious db error).
            raise UnsupportedNemesis("Schema doesn't match the snapshot, not uploading")

        result = self.target_node.run_nodetool(sub_cmd="cfstats", args="keyspace1.standard1")

        def do_refresh(node):
            key_store = KeyStore()
            creds = key_store.get_scylladb_upload_credentials()
            # Download the sstable files from S3
            remote_get_file(node.remoter, sstable_url, sstable_file,
                            hash_expected=sstable_md5, retries=2,
                            user_agent=creds['user_agent'])
            result = node.remoter.sudo("ls -t /var/lib/scylla/data/keyspace1/")
            upload_dir = result.stdout.split()[0]
            if node.is_docker():
                node.remoter.run(f'tar xvfz {sstable_file} -C /var/lib/scylla/data/keyspace1/{upload_dir}/upload/')
            else:
                node.remoter.sudo(
                    f'tar xvfz {sstable_file} -C /var/lib/scylla/data/keyspace1/{upload_dir}/upload/', user='scylla')

            # Scylla Enterprise 2019.1 doesn't support to load schema.cql and manifest.json, let's remove them
            node.remoter.sudo(f'rm -f /var/lib/scylla/data/keyspace1/{upload_dir}/upload/schema.cql')
            node.remoter.sudo(f'rm -f /var/lib/scylla/data/keyspace1/{upload_dir}/upload/manifest.json')
            self.log.debug(f'Loading {keys_num} keys to {node.name} by refresh')

            # Resharding of the loaded sstable files is performed before they are moved from upload to the main folder.
            # So we need to validate that resharded files are placed in the "upload" folder before moving.
            # Find the compaction output that reported about the resharding

            search_reshard = node.follow_system_log(patterns=['Resharded.*\[/'])
            node.run_nodetool(sub_cmd="refresh", args="-- keyspace1 standard1")
            resharding_done = list(search_reshard)
            return resharding_done

        def validate_resharding_after_refresh(resharding_done):
            """
            # Validate that files after resharding were saved in the "upload" folder.
            # Example of compaction output:

            #   scylla[6653]:  [shard 0] compaction - [Reshard keyspace1.standard1 3cad4140-f8c3-11ea-acb1-000000000002]
            #   Resharded 1 sstables to [
            #   /var/lib/scylla/data/keyspace1/standard1-9fbed8d0f8c211ea9bb1000000000000/upload/md-9-big-Data.db:level=0,
            #   /var/lib/scylla/data/keyspace1/standard1-9fbed8d0f8c211ea9bb1000000000000/upload/md-10-big-Data.db:level=0,
            #   /var/lib/scylla/data/keyspace1/standard1-9fbed8d0f8c211ea9bb1000000000000/upload/md-11-big-Data.db:level=0,
            #   /var/lib/scylla/data/keyspace1/standard1-9fbed8d0f8c211ea9bb1000000000000/upload/md-12-big-Data.db:level=0,
            #   /var/lib/scylla/data/keyspace1/standard1-9fbed8d0f8c211ea9bb1000000000000/upload/md-13-big-Data.db:level=0,
            #   /var/lib/scylla/data/keyspace1/standard1-9fbed8d0f8c211ea9bb1000000000000/upload/md-22-big-Data.db:level=0,
            #   /var/lib/scylla/data/keyspace1/standard1-9fbed8d0f8c211ea9bb1000000000000/upload/md-15-big-Data.db:level=0,
            #   /var/lib/scylla/data/keyspace1/standard1-9fbed8d0f8c211ea9bb1000000000000/upload/md-16-big-Data.db:level=0,
            #   ]. 91MB to 92MB (~100% of original) in 5009ms = 18MB/s. ~370176 total partitions merged to 370150
            """
            assert resharding_done, "Resharding wasn't run"
            self.log.debug(f"Found resharding: {resharding_done}")

            for line in resharding_done:
                # Find all files that were created after resharding
                for one_file in re.findall(r"(/var/.*?),", line, re.IGNORECASE):
                    # The file path have to include "upload" folder
                    assert '/upload/' in one_file, "Loaded file was resharded not in 'upload' folder"

        if result is not None and result.exit_status == 0:
            # Check one special key before refresh, we will verify refresh by query in the end
            # Note: we can't DELETE the key before refresh, otherwise the old sstable won't be loaded
            #       TRUNCATE can be used the clean the table, but we can't do it for keyspace1.standard1
            query_verify = "SELECT * FROM keyspace1.standard1 WHERE key=0x32373131364f334f3830"
            result = self.target_node.run_cqlsh(query_verify)
            if '(0 rows)' in result.stdout:
                self.log.debug('Key 0x32373131364f334f3830 does not exist before refresh')
            else:
                self.log.debug('Key 0x32373131364f334f3830 already exists before refresh')

            # Executing rolling refresh one by one
            for node in self.cluster.nodes:
                resharding_done = do_refresh(node)
                validate_resharding_after_refresh(resharding_done)

            # Verify that the special key is loaded by SELECT query
            result = self.target_node.run_cqlsh(query_verify)
            assert '(1 rows)' in result.stdout, 'The key is not loaded by `nodetool refresh`'

    def disrupt_nodetool_enospc(self, sleep_time=30, all_nodes=False):
        if isinstance(self.cluster, MinikubeScyllaPodCluster):
            # Minikube cluster has shared file system, it is shared not only between cluster nodes
            # but also between kubernetes services too, which make kubernetes inoperational once enospc is reached
            raise UnsupportedNemesis('disrupt_nodetool_enospc is not supported on minikube cluster')

        if all_nodes:
            nodes = self.cluster.nodes
        else:
            nodes = [self.target_node]
        self._set_current_disruption('Enospc test on {}'.format([n.name for n in nodes]))

        for node in nodes:
            with ignore_no_space_errors(node=node):

                result = node.remoter.run('cat /proc/mounts')
                if '/var/lib/scylla' not in result.stdout:
                    self.log.error("Scylla doesn't use an individual storage, skip enospc test")
                    continue

                try:
                    reach_enospc_on_node(target_node=node)
                finally:
                    clean_enospc_on_node(target_node=node, sleep_time=sleep_time)

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
        self.log.info(">>>>>>>>>>>>>Started random_disrupt_method %s" % disrupt_method_name)
        self.metrics_srv.event_start(disrupt_method_name)
        try:
            disrupt_method()
        except Exception as exc:  # pylint: disable=broad-except
            error_msg = "Exception in random_disrupt_method %s: %s", disrupt_method_name, exc
            self.log.error(error_msg)
            self.error_list.append(error_msg)
            raise
        else:
            self.log.info("<<<<<<<<<<<<<Finished random_disrupt_method %s" % disrupt_method_name)
        finally:
            self.metrics_srv.event_stop(disrupt_method_name)

    def get_all_disrupt_methods(self):
        all_disruptions = [attr[1] for attr in inspect.getmembers(self)
                           if attr[0].startswith('disrupt_') and callable(attr[1])]
        self.disruptions_list.extend(all_disruptions)

    def shuffle_list_of_disruptions(self):
        if self.cluster.params.get('nemesis_seed'):
            nemesis_seed = self.cluster.params.get('nemesis_seed')
        else:
            nemesis_seed = random.randint(0, 1000)
            self.log.info(f'nemesis_seed generated for this test is {nemesis_seed}')
        self.log.debug(f'nemesis_seed to be used is {nemesis_seed}')

        self.log.debug(f"nemesis stack BEFORE SHUFFLE is {self.disruptions_list}")
        random.Random(nemesis_seed).shuffle(self.disruptions_list)
        self.log.debug(f"nemesis stack AFTER SHUFFLE is {self.disruptions_list}")

    def call_next_nemesis(self):
        self.log.info(f'Selecting the next nemesis out of stack {self.disruptions_list}')
        if self.disruptions_list is not None:
            self.execute_disrupt_method(disrupt_method=self.disruptions_list.pop())
            self.log.info(f'Remaining nemesis to execute {self.disruptions_list}')
        else:
            self.log.info('Nemesis stack is empty - setting termination_event')
            self.termination_event.set()

    @latency_calculator_decorator
    def repair_nodetool_repair(self, node=None):
        node = node if node else self.target_node
        node.run_nodetool("repair")

    def repair_nodetool_rebuild(self):
        self.target_node.run_nodetool('rebuild')

    def disrupt_nodetool_cleanup(self):
        # This fix important when just user profile is run in the test and "keyspace1" doesn't exist.
        test_keyspaces = self.cluster.get_test_keyspaces()
        for node in self.cluster.nodes:
            self._set_current_disruption('NodetoolCleanupMonkey %s' % node, node=node)
            for keyspace in test_keyspaces:
                node.run_nodetool(sub_cmd="cleanup", args=keyspace)

    def _prepare_test_table(self, ks='keyspace1', table=None):
        ks_cfs = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node)
        table_exist = f'{ks}.{table}' in ks_cfs if table else True

        test_keyspaces = self.cluster.get_test_keyspaces()
        # if keyspace or table doesn't exist, create it by cassandra-stress
        if ks not in test_keyspaces or not table_exist:
            stress_cmd = "cassandra-stress write n=400000 cl=QUORUM -port jmx=6868 -mode native cql3 " \
                         f"-schema 'replication(factor={self.tester.reliable_replication_factor})' -log interval=5"
            cs_thread = self.tester.run_stress_thread(
                stress_cmd=stress_cmd, keyspace_name=ks, stop_test_on_failure=False)
            cs_thread.verify_results()

    def disrupt_truncate(self):
        self._set_current_disruption('TruncateMonkey {}'.format(self.target_node))

        keyspace_truncate = 'ks_truncate'
        table = 'standard1'

        self._prepare_test_table(ks=keyspace_truncate)

        # In order to workaround issue #4924 when truncate timeouts, we try to flush before truncate.
        self.target_node.run_nodetool("flush")
        # do the actual truncation
        self.target_node.run_cqlsh(cmd='TRUNCATE {}.{}'.format(keyspace_truncate, table), timeout=120)

    def disrupt_truncate_large_partition(self):
        """
        Introduced a new truncate nemesis, it truncates a large-partition table,
        it's used to cover one improvement of compaction.
        The increase frequency of checking abortion is very useful for truncate.
        """
        self._set_current_disruption('TruncateMonkeyLargePartition {}'.format(self.target_node))

        ks_name = 'ks_truncate_large_partition'
        table = 'test_table'
        stress_cmd = "scylla-bench -workload=sequential -mode=write -replication-factor=3 -partition-count=10 " + \
                     "-clustering-row-count=5555 -clustering-row-size=uniform:10..20 -concurrency=10 " + \
                     "-connection-count=10 -consistency-level=quorum -rows-per-request=10 -timeout=60s " + \
                     f"-keyspace {ks_name} -table {table}"
        bench_thread = self.tester.run_stress_thread(
            stress_cmd=stress_cmd, stop_test_on_failure=False)
        self.tester.verify_stress_thread(bench_thread)

        # In order to workaround issue #4924 when truncate timeouts, we try to flush before truncate.
        self.target_node.run_nodetool("flush")
        # do the actual truncation
        self.target_node.run_cqlsh(cmd='TRUNCATE {}.{}'.format(ks_name, table), timeout=120)

    def _modify_table_property(self, name, val, filter_out_table_with_counter=False, modify_all_tables=False):
        disruption_name = "".join([p.strip().capitalize() for p in name.split("_")])
        self._set_current_disruption('ModifyTableProperties%s %s' % (disruption_name, self.target_node))

        ks_cfs = self.cluster.get_non_system_ks_cf_list(
            db_node=self.target_node, filter_out_table_with_counter=filter_out_table_with_counter,
            filter_out_mv=True)  # not allowed to modify MV

        keyspace_table = random.choice(ks_cfs) if ks_cfs else ks_cfs
        if not keyspace_table:
            raise UnsupportedNemesis(
                'Non-system keyspace and table are not found. ModifyTableProperties nemesis can\'t be run')

        cmd = "ALTER TABLE {keyspace_table} WITH {name} = {val};".format(
            keyspace_table=keyspace_table, name=name, val=val)
        self.log.info('_modify_table_property: {}'.format(cmd))
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
        with self.cluster.cql_connection_patient(self.tester.db_cluster.nodes[0]) as session:
            query_result = session.execute('SELECT keyspace_name FROM system_schema.keyspaces;')
            for result_rows in query_result:
                keyspaces.extend([row.lower() for row in result_rows if not row.lower().startswith("system")])
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
                tables = get_db_tables(session, ks, with_compact_storage=False)
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
        except Exception as exc:  # pylint: disable=broad-except
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

        max_partitions_in_test_table = self.cluster.params.get('max_partitions_in_test_table')

        if not max_partitions_in_test_table:
            raise UnsupportedNemesis(
                'This nemesis expects mandatory "max_partitions_in_test_table" parameter to be set')

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

        max_partitions_in_test_table = self.cluster.params.get('max_partitions_in_test_table')
        partition_range_with_data_validation = self.cluster.params.get('partition_range_with_data_validation')

        if partition_range_with_data_validation and '-' in partition_range_with_data_validation:
            partition_range_splitted = partition_range_with_data_validation.split('-')
            exclude_partitions.extend(
                [i for i in range(int(partition_range_splitted[0]),
                                  int(partition_range_splitted[1]))])  # pylint: disable=unnecessary-comprehension

        partitions_for_delete = defaultdict(list)
        with self.cluster.cql_connection_patient(self.target_node, connect_timeout=300) as session:
            session.default_consistency_level = ConsistencyLevel.ONE

            for partition_key in [i * 2 + 50 for i in range(max_partitions_in_test_table)]:
                if len(partitions_for_delete) == partitions_amount:
                    break

                if exclude_partitions and partition_key in exclude_partitions:
                    continue

                # The scylla_bench.test table is created WITH CLUSTERING ORDER BY (ck DESC).
                # So first returned value is max cl value in the partition
                cmd = f"select ck from {ks_cf} where pk={partition_key} limit 1"
                try:
                    result = session.execute(cmd, timeout=300)
                except Exception as exc:  # pylint: disable=broad-except
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

    def run_deletions(self, queries, ks_cf):
        for cmd in queries:
            self.log.debug(f'delete query: {cmd}')
            with self.cluster.cql_connection_patient(self.target_node, connect_timeout=300) as session:
                session.execute(cmd, timeout=3600)

        self.target_node.run_nodetool('flush', args=ks_cf.replace('.', ' '))

    def delete_half_partition(self, ks_cf):
        self.log.debug('Delete by range - half of partition')

        partitions_for_delete = self.choose_partitions_for_delete(10, ks_cf, with_clustering_key_data=True)
        if not partitions_for_delete:
            self.log.error('Not found partitions for delete')
            return partitions_for_delete

        queries = []
        for pkey, ckey in partitions_for_delete.items():
            queries.append(f"delete from {ks_cf} where pk = {pkey} and ck > {int(ckey[1] / 2)}")
        self.run_deletions(queries=queries, ks_cf=ks_cf)

        return partitions_for_delete

    def delete_range_in_few_partitions(self, ks_cf, partitions_for_exclude_dict):
        self.log.debug('Delete same range in the few partitions')

        partitions_for_exclude = list(partitions_for_exclude_dict.keys())
        partitions_for_delete = self.choose_partitions_for_delete(10, ks_cf, with_clustering_key_data=True,
                                                                  exclude_partitions=partitions_for_exclude)
        if not partitions_for_delete:
            self.log.error('Not found partitions for delete')
            return []

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
        self._set_current_disruption('DeleteByPartitionsMonkey {}'.format(self.target_node))

        ks_cf = 'scylla_bench.test'
        partitions_for_delete = self.choose_partitions_for_delete(10, ks_cf)

        if not partitions_for_delete:
            self.log.error('Not found partitions for delete')
            return

        queries = []
        for partition_key in partitions_for_delete.keys():
            queries.append(f"delete from {ks_cf} where pk = {partition_key}")

        self.run_deletions(queries=queries, ks_cf=ks_cf)

    def disrupt_delete_by_rows_range(self):
        """
        Delete few partitions in the table with large partitions
        """
        self.verify_initial_inputs_for_delete_nemesis()
        self._set_current_disruption('DeleteByRowsRangeMonkey {}'.format(self.target_node))

        ks_cf = 'scylla_bench.test'
        partitions_for_exclude = self.delete_half_partition(ks_cf)

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
        self._set_current_disruption(f'AddDropColumnMonkey table {".".join(self._add_drop_column_target_table)}')
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

    def toggle_table_ics(self):
        """
            Alters a non-system table compaction strategy from ICS to any-other and vise versa.
        """
        list_additional_params = get_compaction_random_additional_params()
        all_ks_cfs = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node)
        non_mview_ks_cfs = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node, filter_out_mv=True)

        if not all_ks_cfs:
            raise UnsupportedNemesis(
                'Non-system keyspace and table are not found. toggle_tables_ics nemesis can\'t run')

        mview_ks_cfs = list(set(all_ks_cfs) - set(non_mview_ks_cfs))
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
            for param in list_additional_params:
                new_compaction_strategy_as_dict.update(param)
        alter_command_prefix = 'ALTER TABLE ' if keyspace_table not in mview_ks_cfs else 'ALTER MATERIALIZED VIEW '
        cmd = alter_command_prefix + \
            " {keyspace_table} WITH compaction = {new_compaction_strategy_as_dict};".format(**locals())
        self.log.debug("Toggle table ICS query to execute: {}".format(cmd))
        try:
            self.target_node.run_cqlsh(cmd)
        except (UnexpectedExit, Libssh2UnexpectedExit) as unexpected_exit:
            if "Unable to find compaction strategy" in str(unexpected_exit):
                raise UnsupportedNemesis("for this nemesis to work, you need ICS supported scylla version.")
            raise unexpected_exit

        self.cluster.wait_for_schema_agreement()

    def modify_table_compaction(self):
        """
            The compaction property defines the compaction strategy class for this table.
            default: compaction = {'class': 'SizeTieredCompactionStrategy'}
        """
        # TODO: Sub-properties for each of compaction strategies should also be tested
        strategies = ("SizeTieredCompactionStrategy", "DateTieredCompactionStrategy",
                      "TimeWindowCompactionStrategy", "LeveledCompactionStrategy")
        prop_val = {"class": random.choice(strategies)}
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
            prop_val["chunk_length_kb"] = random.choice(["4K", "64KB", "1M"])
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
            default: default_time_to_live = 0
        """
        # Select table without columns with "counter" type for this nemesis - issue #1037:
        #    Modify_table nemesis chooses first non-system table, and modify default_time_to_live of it.
        #    But table with counters doesn't support this
        self._modify_table_property(name="default_time_to_live", val=random.randint(864000, 630720000),
                                    filter_out_table_with_counter=True)  # max allowed TTL - 20 years (630720000)

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
        self._modify_table_property(name="memtable_flush_period_in_ms", val=random.randint(1, 5000))

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
                   "'%spercentile'" % random.randint(1, 99),
                   "'%sms'" % random.randint(1, 1000))
        self._modify_table_property(name="speculative_retry", val=random.choice(options))

    def disrupt_toggle_table_ics(self):
        self._set_current_disruption('ToggleTableICS')
        self.toggle_table_ics()

    def disrupt_modify_table(self):
        # randomly select and run one of disrupt_modify_table* methods
        disrupt_func_name = random.choice([dm for dm in dir(self) if dm.startswith("modify_table")])
        disrupt_func = getattr(self, disrupt_func_name)
        disrupt_func()

    def disrupt_mgmt_backup_specific_keyspaces(self):
        self._set_current_disruption('ManagementBackupWithSpecificKeyspaces')
        self._mgmt_backup(backup_specific_tables=True)

    def disrupt_mgmt_backup(self):
        self._set_current_disruption('ManagementBackup')
        self._mgmt_backup(backup_specific_tables=False)

    def _mgmt_backup(self, backup_specific_tables):
        if not (self.cluster.params.get('use_mgmt') or
                self.cluster.params.get('use_cloud_manager')):
            raise UnsupportedNemesis('Scylla-manager configuration is not defined!')

        mgr_cluster = self.cluster.get_cluster_manager()
        cluster_backend = self.cluster.params.get('cluster_backend')
        backup_bucket_location = self.cluster.get_backup_buckets_name()
        if cluster_backend == 'aws':
            location_list = [f"s3:{backup_bucket_location.split()[0]}"]
        elif cluster_backend == 'gce':  # For cloud it should be "elif 'gce' in cluster_backend:
            location_list = [f"gcs:{backup_bucket_location}"]
        elif cluster_backend == 'aws-siren':
            location_list = None
        else:
            raise ValueError(f"cluster_backend={cluster_backend} not supported in ManagementBackup")
        if backup_specific_tables:
            non_test_keyspaces = self.cluster.get_test_keyspaces()
            mgr_task = mgr_cluster.create_backup_task(location_list=location_list, keyspace_list=non_test_keyspaces)
        else:
            mgr_task = mgr_cluster.create_backup_task(location_list=location_list)

        succeeded, status = mgr_task.wait_for_task_done_status(timeout=54000)
        if succeeded and status == TaskStatus.DONE:
            self.log.info('Task: {} is done.'.format(mgr_task.id))
        if not succeeded:
            if status == TaskStatus.ERROR:
                assert succeeded, f'Backup task {mgr_task.id} failed'
            else:
                mgr_task.stop()
                assert succeeded, f'Backup task {mgr_task.id} timed out - while on status {status}'

    @latency_calculator_decorator
    def disrupt_mgmt_repair_cli(self):
        self._set_current_disruption('ManagementRepair')
        if not self.cluster.params.get('use_mgmt') and not self.cluster.params.get('use_cloud_manager'):
            raise UnsupportedNemesis('Scylla-manager configuration is not defined!')
        mgr_cluster = self.cluster.get_cluster_manager()
        mgr_task = mgr_cluster.create_repair_task()
        task_final_status = mgr_task.wait_and_get_final_status(timeout=86400)  # timeout is 24 hours
        assert task_final_status == TaskStatus.DONE, 'Task: {} final status is: {}.'.format(
            mgr_task.id, str(mgr_task.status))
        self.log.info('Task: {} is done.'.format(mgr_task.id))

    def disrupt_abort_repair(self):
        """
        Start repair target_node in background, then try to abort the repair streaming.
        """
        self._set_current_disruption('AbortRepairMonkey')
        self.log.debug("Start repair target_node in background")

        @raise_event_on_failure
        def silenced_nodetool_repair_to_fail():
            try:
                self.target_node.run_nodetool("repair", verbose=False)
            except (UnexpectedExit, Libssh2UnexpectedExit):
                self.log.info('Repair failed as expected')
            except Exception:
                self.log.error('Repair failed due to the unknown error')
                raise
            else:
                raise RuntimeError('This repair should fail due to the being aborted')

        def repair_streaming_exists():
            active_repair_cmd = 'curl -s -X GET --header "Content-Type: application/json" --header ' \
                                '"Accept: application/json" "http://127.0.0.1:10000/storage_service/active_repair/"'
            result = self.target_node.remoter.run(active_repair_cmd)
            active_repairs = re.match(r".*\[(\d)+\].*", result.stdout)
            if active_repairs:
                self.log.debug("Found '%s' active repairs", active_repairs.group(1))
                return True
            return False

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='NodeToolRepairThread') as tp:
            thread = tp.submit(silenced_nodetool_repair_to_fail)
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
    # Temporary disable due to https://trello.com/c/Ru0T9Nmu/1239-fix-validatehintedhandoff-nemesis
    # TODO: Bentsi to fix this nemesis or investigate if it's a real scylla issue.
    # TODO: Bentsi to fix this nemesis or investigate if it's a real scylla issue.

    def disable_disrupt_validate_hh_short_downtime(self):  # pylint: disable=invalid-name
        """
            Validates that hinted handoff mechanism works: there were no drops and errors
            during short stop of one of the nodes in cluster
        """
        self._set_current_disruption("ValidateHintedHandoffShortDowntime")
        if self.cluster.params.get('hinted_handoff') == 'disabled':
            raise UnsupportedNemesis('For this nemesis to work, `hinted_handoff` needs to be set to `enabled`')

        start_time = time.time()
        self.target_node.stop_scylla()
        time.sleep(10)
        self.target_node.start_scylla()

        # Wait until all other nodes see the target node as UN
        # Only then we can expect that hint sending started on all nodes
        def target_node_reported_un_by_others():
            for node in self.cluster.nodes:
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
        - for all keyspaces (with all their tables) - nodetool snapshot cmd without parameters: nodetool snapshot
        - for one kespace (with all its tables) - nodetool snapshot cmd with "-ks" parameter: nodetool snapshot -ks system
        - for a few keyspaces (with all their tables) - nodetool snapshot cmd with "-ks" parameter: nodetool snapshot -ks system, system_schema
        - for one kespace and few tables - nodetool snapshot cmd with "-cf" parameter like: nodetool snapshot test -cf cf1,cf2

        By parsing of nodetool_cmd is recognized with type of snapshot was created.
        This function check if all expected keyspace/tables are in the snapshot
        """
        snapshot_params = nodetool_cmd.split()
        ks_cf = self.cluster.get_any_ks_cf_list(db_node=self.target_node, filter_empty_tables=False)
        keyspace_table = []
        if len(snapshot_params) > 1:
            if snapshot_params[1] == "-kc":
                for ks in snapshot_params[2].split(','):
                    keyspace_table.extend([k_c.split('.') for k_c in ks_cf if k_c.startswith(f"{ks}.")])
            else:
                keyspace = snapshot_params[1]
                keyspace_table.extend([[keyspace, cf] for cf in snapshot_params[3].split(',')])
        else:
            keyspace_table.extend([k_c.split('.') for k_c in ks_cf])

        snapshot_content_list = [[elem.keyspace_name, elem.table_name] for elem in snapshot_content]
        if sorted(keyspace_table) != sorted(snapshot_content_list):
            raise AssertionError(f"Snapshot content not as expected. \n"
                                 f"Expected content: {sorted(keyspace_table)} \n "
                                 f"Actual snapshot content: {sorted(snapshot_content_list)}")

    def disrupt_snapshot_operations(self):
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
                keyspaces = list(set([ks.split('.')[0] for ks in ks_cf]))
                if len(keyspaces) == 1:
                    ks_cf = self.cluster.get_any_ks_cf_list(db_node=self.target_node)
                    keyspaces = list(set([ks.split('.')[0] for ks in ks_cf]))
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
            ks_cf = self.cluster.get_non_system_ks_cf_list(db_node=self.target_node)

            if not ks_cf or len(ks_cf) == 1:
                # If test table wasn't found - take system table snapshot
                ks_cf = self.cluster.get_any_ks_cf_list(db_node=self.target_node)

            ks_with_few_tables = get_ks_with_few_tables(ks_cf)
            if not ks_with_few_tables:
                # If non-system keyspace with few tables wasn't found - take system table snapshot
                ks_cf = self.cluster.get_any_ks_cf_list(db_node=self.target_node)
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

        self._set_current_disruption('SnapshotOperations')
        functions_map = [(_few_tables,), (_full_snapshot,), (_ks_snapshot, True), (_ks_snapshot, False)]
        snapshot_option = random.choice(functions_map)

        try:
            nodetool_cmd = snapshot_option[0]() if len(snapshot_option) == 1 else snapshot_option[0](snapshot_option[1])
            if not nodetool_cmd:
                raise ValueError(f"Failed to get nodetool command.")
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Failed to get nodetool command. Error: {exc}")

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

        result = self.target_node.run_nodetool('clearsnapshot')
        self.log.debug(result)

    def disrupt_show_toppartitions(self):
        def _parse_toppartitions_output(output):
            """parsing output of toppartitions

            input format stored in output parameter:
            WRITES Sampler:
              Cardinality: ~10 (15 capacity)
              Top 10 partitions:
                Partition     Count       +/-
                9        11         0
                0         1         0
                1         1         0

            READS Sampler:
              Cardinality: ~10 (256 capacity)
              Top 3 partitions:
                Partition     Count       +/-
                0         3         0
                1         3         0
                2         3         0
                3         2         0

            return Dict:
            {
                'READS': {
                    'toppartitions': '10',
                    'partitions': OrderedDict('0': {'count': '1', 'margin': '0'},
                                              '1': {'count': '1', 'margin': '0'},
                                              '2': {'count': '1', 'margin': '0'}),
                    'cardinality': '10',
                    'capacity': '256',
                },
                'WRITES': {
                    'toppartitions': '10',
                    'partitions': OrderedDict('10': {'count': '1', 'margin': '0'},
                                              '11': {'count': '1', 'margin': '0'},
                                              '21': {'count': '1', 'margin': '0'}),
                    'cardinality': '10',
                    'capacity': '256',
                    'sampler': 'WRITES'
                }
            }


            Arguments:
                output {str} -- stdout of nodetool topparitions command

            Returns:
                dict -- result of parsing
            """

            pattern1 = r"(?P<sampler>[A-Z]+)\sSampler:\W+Cardinality:\s~(?P<cardinality>[0-9]+)\s\((?P<capacity>[0-9]+)\scapacity\)\W+Top\s(?P<toppartitions>[0-9]+)\spartitions:"
            pattern2 = r"(?P<partition>[\w:]+)\s+(?P<count>[\d]+)\s+(?P<margin>[\d]+)"
            toppartitions = {}
            for out in output.strip().split('\n\n'):
                partition = OrderedDict()
                sampler_data = re.search(pattern1, out, re.MULTILINE)
                assert sampler_data, f"Pattern:{pattern1} are not matched on string:\n {out}"
                sampler_data = sampler_data.groupdict()
                partitions = re.findall(pattern2, out, re.MULTILINE)
                self.log.debug(f"Next list of top partitions are found {partitions}")
                for val in partitions:
                    partition.update({val[0]: {'count': val[1], 'margin': val[2]}})
                sampler_data.update({'partitions': partition})
                toppartitions[sampler_data.pop('sampler')] = sampler_data
            return toppartitions

        def generate_random_parameters_values():  # pylint: disable=invalid-name
            ks_cf_list = self.cluster.get_non_system_ks_cf_list(self.target_node)
            if not ks_cf_list:
                raise UnsupportedNemesis('User-defined Keyspace and ColumnFamily are not found.')
            try:
                ks, cf = random.choice(ks_cf_list).split('.')
            except IndexError as details:
                self.log.error('User-defined Keyspace and ColumnFamily are not found %s.', ks_cf_list)
                self.log.debug('Error during choosing keyspace and columnfamily %s', details)
                raise Exception('User-defined Keyspace and ColumnFamily are not found. \n{}'.format(details))
            return {
                'toppartition': str(random.randint(5, 20)),
                'samplers': random.choice(['writes', 'reads', 'writes,reads']),
                'capacity': str(random.randint(100, 1024)),
                'ks': ks,
                'cf': cf,
                'duration': str(random.randint(1000, 10000))
            }

        self._set_current_disruption("ShowTopPartitions")
        result = self.target_node.run_nodetool(sub_cmd='help', args='toppartitions')
        if 'Unknown command toppartitions' in result.stdout:
            raise UnsupportedNemesis("nodetool doesn't support toppartitions")

        # workaround for issue #4519
        self.target_node.run_nodetool('cfstats')

        args = generate_random_parameters_values()
        sub_cmd_args = "{ks} {cf} {duration} -s {capacity} -k {toppartition} -a {samplers}".format(**args)

        result = self.target_node.run_nodetool(sub_cmd='toppartitions', args=sub_cmd_args)

        toppartition_result = _parse_toppartitions_output(result.stdout)
        for sampler in args['samplers'].split(','):
            sampler = sampler.upper()
            self.tester.assertIn(sampler, toppartition_result,
                                 msg="{} sampler not found in result".format(sampler))
            self.tester.assertTrue(toppartition_result[sampler]['toppartitions'] == args['toppartition'],
                                   msg="Wrong expected and actual top partitions number for {} sampler".format(sampler))
            self.tester.assertTrue(toppartition_result[sampler]['capacity'] == args['capacity'],
                                   msg="Wrong expected and actual capacity number for {} sampler".format(sampler))
            self.tester.assertLessEqual(len(toppartition_result[sampler]['partitions'].keys()),
                                        int(args['toppartition']),
                                        msg="Wrong number of requested and expected toppartitions for {} sampler".format(
                                            sampler))

    def get_rate_limit_for_network_disruption(self) -> Optional[str]:
        if not self.monitoring_set.nodes:
            return None

        # get the last 10min avg network bandwidth used, and limit  30% to 70% of it
        prometheus_stats = PrometheusDBStats(host=self.monitoring_set.nodes[0].external_address)
        query = 'avg(node_network_receive_bytes_total{instance=~".*?%s.*?", device="eth0"})' % \
                self.target_node.ip_address
        now = time.time()
        results = prometheus_stats.query(query=query, start=now - 600, end=now)
        assert results, "no results for node_network_receive_bytes_total metric in Prometheus "
        avg_bitrate_per_node = max([float(avg_rate) for _, avg_rate in results[0]["values"]])
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

    def disrupt_network_random_interruptions(self):  # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        self._set_current_disruption('NetworkRandomInterruption')
        if not self.cluster.extra_network_interface:
            raise UnsupportedNemesis("for this nemesis to work, you need to set `extra_network_interface: True`")

        if not self.target_node.install_traffic_control():
            raise UnsupportedNemesis("Traffic control package not installed on system")

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

        list_of_timeout_options = [10, 60, 120, 300, 500]
        option_name, selected_option = random.choice(list_of_tc_options)
        wait_time = random.choice(list_of_timeout_options)

        self._set_current_disruption(option_name)
        self.log.debug("NetworkRandomInterruption: [%s] for %dsec", selected_option, wait_time)
        self.target_node.traffic_control(None)
        try:
            self.target_node.traffic_control(selected_option)
            time.sleep(wait_time)
        finally:
            self.target_node.traffic_control(None)
            self._wait_all_nodes_un()

    def disrupt_network_block(self):
        self._set_current_disruption('BlockNetwork')
        if not self.cluster.extra_network_interface:
            raise UnsupportedNemesis("for this nemesis to work, you need to set `extra_network_interface: True`")

        if not self.target_node.install_traffic_control():
            raise UnsupportedNemesis("Traffic control package not installed on system")

        selected_option = "--loss 100%"
        list_of_timeout_options = [10, 60, 120, 300, 500]
        wait_time = random.choice(list_of_timeout_options)
        self.log.debug("BlockNetwork: [%s] for %dsec", selected_option, wait_time)
        self.target_node.traffic_control(None)
        try:
            self.target_node.traffic_control(selected_option)
            time.sleep(wait_time)
        finally:
            self.target_node.traffic_control(None)
            self._wait_all_nodes_un()

    @retrying(n=15, sleep_time=5, allowed_exceptions=ClusterNodesNotReady)
    def _wait_all_nodes_un(self):
        for node in self.cluster.nodes:
            self.cluster.check_nodes_up_and_normal(verification_node=node)

    def disrupt_remove_node_then_add_node(self):
        """
        https://docs.scylladb.com/operating-scylla/procedures/cluster-management/remove_node/

        1.Terminate node
        2.Run full repair
        3.Nodetool removenode
        4.Add new node
        """
        if self.cluster.params.get("cluster_backend") == 'aws-siren':
            raise UnsupportedNemesis("Skipping this nemesis due this job run from Siren cloud with 2019 version!")

        self._set_current_disruption('TerminateAndRemoveNodeMonkey')
        node_to_remove = self.target_node
        up_normal_nodes = self.cluster.get_nodes_up_and_normal(verification_node=node_to_remove)
        # node_to_remove must be different than node
        verification_node = random.choice([n for n in self.cluster.nodes if n is not node_to_remove])

        # node_to_remove must not be the only seed in cluster
        num_of_seed_nodes = len(self.cluster.seed_nodes)
        if node_to_remove.is_seed and num_of_seed_nodes < 2:
            raise UnsupportedNemesis("Removing the only seed node is not yet supported")

        # get node's host_id
        removed_node_status = self.cluster.get_node_status_dictionary(
            ip_address=node_to_remove.ip_address, verification_node=verification_node)
        assert removed_node_status is not None, "failed to get host_id using nodetool status"
        host_id = removed_node_status["host_id"]

        # node stop and make sure its "DN"
        node_to_remove.stop_scylla_server(verify_up=True, verify_down=True)

        # terminate node
        self._terminate_cluster_node(node_to_remove)

        # full cluster repair
        up_normal_nodes.remove(node_to_remove)
        # Repairing the first node will result in a best effort repair due to the terminated node,
        # and as a result requires ignoring repair errors
        first_node_to_repair = up_normal_nodes[0]
        with DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                            line="failed to repair",
                            node=first_node_to_repair):
            try:
                self.repair_nodetool_repair(node=first_node_to_repair)
            except Exception as details:  # pylint: disable=broad-except
                self.log.error(f"failed to execute repair command "
                               f"on node {first_node_to_repair} due to the following error: {str(details)}")
        for node in up_normal_nodes[1:]:
            try:
                self.repair_nodetool_repair(node=node)
            except Exception as details:  # pylint: disable=broad-except
                self.log.error(f"failed to execute repair command "
                               f"on node {node} due to the following error: {str(details)}")

        def remove_node():
            # nodetool removenode 'host_id'
            rnd_node = random.choice([n for n in self.cluster.nodes if n is not self.target_node])
            self.log.info("Running removenode command on {}, Removing node with the following host_id: {}"
                          .format(rnd_node.ip_address, host_id))
            res = rnd_node.run_nodetool("removenode {}".format(host_id), ignore_status=True, verbose=True)
            return res.exit_status

        exit_status = remove_node()
        assert exit_status == 0, "nodetool removenode command exited with status {}".format(exit_status)

        # verify node is removed by nodetool status
        removed_node_status = self.cluster.get_node_status_dictionary(
            ip_address=node_to_remove.ip_address, verification_node=verification_node)
        assert removed_node_status is None, "Node was not removed properly (Node status:{})".format(removed_node_status)

        # add new node
        new_node = self._add_and_init_new_cluster_node(rack=self.target_node.rack)
        # in case the removed node was a seed
        if node_to_remove.is_seed:
            new_node.is_seed = True
            self.cluster.update_seed_provider()
        # after add_node, the left nodes have data that isn't part of their tokens anymore.
        # In order to eliminate cases that we miss a "data loss" bug because of it, we cleanup this data.
        # This fix important when just user profile is run in the test and "keyspace1" doesn't exist.
        try:
            test_keyspaces = self.cluster.get_test_keyspaces()
            for node in self.cluster.nodes:
                for keyspace in test_keyspaces:
                    node.run_nodetool(sub_cmd='cleanup', args=keyspace)
        finally:
            self.unset_current_running_nemesis(new_node)

    # Temporary disable due to  https://github.com/scylladb/scylla/issues/6522
    def _disrupt_network_reject_inter_node_communication(self):
        """
        Generates random firewall rule to drop/reject packets for inter-node communications, port 7000 and 7001
        """
        name = 'RejectInterNodeNetwork'
        textual_matching_rule, matching_rule = self._iptables_randomly_get_random_matching_rule()
        textual_pkt_action, pkt_action = self._iptables_randomly_get_disrupting_target()
        wait_time = random.choice([10, 60, 120, 300, 500])
        self._set_current_disruption(f'{name} {textual_matching_rule} that belongs to '
                                     'inter node communication connections (port=7000 and 7001) will be'
                                     f' {textual_pkt_action} for {wait_time}s')

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
        name = 'RejectNodeExporterNetwork'
        textual_matching_rule, matching_rule = self._iptables_randomly_get_random_matching_rule()
        textual_pkt_action, pkt_action = self._iptables_randomly_get_disrupting_target()
        wait_time = random.choice([10, 60, 120, 300, 500])
        self._set_current_disruption(f'{name} {textual_matching_rule} that belongs to '
                                     f'node-exporter(port=9100) connections will be {textual_pkt_action} for {wait_time}s')
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
        name = 'RejectThriftNetwork'
        textual_matching_rule, matching_rule = self._iptables_randomly_get_random_matching_rule()
        textual_pkt_action, pkt_action = self._iptables_randomly_get_disrupting_target()
        wait_time = random.choice([10, 60, 120, 300, 500])
        self._set_current_disruption(f'{name} {textual_matching_rule} that belongs to '
                                     f'Thrift(port=9160) connections will be {textual_pkt_action} for {wait_time}s')
        return self._run_commands_wait_and_cleanup(
            self.target_node,
            name=name,
            start_commands=[f'sudo iptables -t filter -A INPUT -p tcp --dport 9160 {matching_rule} -j {pkt_action}'],
            cleanup_commands=[f'sudo iptables -t filter -D INPUT -p tcp --dport 9160 {matching_rule} -j {pkt_action}'],
            wait_time=wait_time
        )

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
        return f'dropped', \
               f'{target_type}'

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
            except Exception as exc:  # pylint: disable=broad-except
                cmd_executed[cmd_num] = False
                self.log.error(
                    f"{name}: failed to execute start command "
                    f"{cmd} on node {node} due to the following error: {str(exc)}")
        if not cmd_executed:
            return
        for cmd_num, cmd in enumerate(cleanup_commands):
            try:
                node.remoter.run(cmd)
            except Exception as exc:  # pylint: disable=broad-except
                self.log.debug(f"{name}: failed to execute cleanup command "
                               f"{cmd} on node {node} due to the following error: {str(exc)}")

    def disrupt_network_start_stop_interface(self):  # pylint: disable=invalid-name
        self._set_current_disruption('StopStartNetworkInterfaces')
        if not self.cluster.extra_network_interface:
            raise UnsupportedNemesis("for this nemesis to work, you need to set `extra_network_interface: True`")

        list_of_timeout_options = [10, 60, 120, 300, 500]
        wait_time = random.choice(list_of_timeout_options)
        self.log.debug("Taking down eth1 for %dsec", wait_time)

        try:
            self.target_node.remoter.run("sudo /sbin/ifdown eth1")
            time.sleep(wait_time)
        finally:
            self.target_node.remoter.run("sudo /sbin/ifup eth1")
            self._wait_all_nodes_un()

    def break_streaming_task_and_rebuild(self, task='decommission'):  # pylint: disable=too-many-statements
        """
        Stop streaming task in middle and rebuild the data on the node.
        """

        def decommission_post_action():
            decommission_done = list(self.target_node.follow_system_log(
                patterns=['DECOMMISSIONING: done'], start_from_beginning=True))

            ips = []
            seed_nodes = [node for node in self.cluster.nodes if node.is_seed]
            status = self.cluster.get_nodetool_status(verification_node=seed_nodes[0])
            self.log.debug(status)
            for dc_info in status.values():
                try:
                    ips.extend(dc_info.keys())
                except Exception:  # pylint: disable=broad-except
                    self.log.debug(dc_info)

            if self.target_node.ip_address not in ips or decommission_done:
                self.log.error(
                    'The target node is decommission unexpectedly, decommission might complete before stopping it. Re-add a new node')
                self._terminate_cluster_node(self.target_node)
                new_node = self._add_and_init_new_cluster_node(rack=self.target_node.rack)
                self.unset_current_running_nemesis(new_node)
                return new_node
            return None

        def streaming_task_thread(nodetool_task='rebuild'):
            """
            Execute some nodetool command to start persistent streaming
            task: decommission | rebuild | repair
            """
            if nodetool_task in ['repair', 'rebuild']:
                self._destroy_data_and_restart_scylla()

            try:
                self.target_node.run_nodetool(nodetool_task)
            except Exception:  # pylint: disable=broad-except
                self.log.debug('%s is stopped' % nodetool_task)

        self.task_used_streaming = None
        streaming_logs_stream = self.target_node.follow_system_log(
            patterns=["range_streamer - Unbootstrap starts|range_streamer - Rebuild starts"])
        repair_logs_stream = self.target_node.follow_system_log(patterns=['repair - Repair 1 out of'])
        streaming_error_logs_stream = self.target_node.follow_system_log(patterns=['streaming.*err'])

        streaming_thread = threading.Thread(target=streaming_task_thread, kwargs={'nodetool_task': task},
                                            name='StreamingThread', daemon=True)
        streaming_thread.start()

        def is_streaming_started():
            streaming_logs = list(streaming_logs_stream)
            self.log.debug(streaming_logs)
            if streaming_logs:
                self.task_used_streaming = True

            # In latest master, repair always won't use streaming
            repair_logs = list(repair_logs_stream)
            self.log.debug(repair_logs)
            return len(streaming_logs) > 0 or len(repair_logs) > 0

        wait.wait_for(func=is_streaming_started, timeout=600, step=1,
                      text='Wait for streaming starts', throw_exc=False)
        self.log.debug('wait for random between 10s to 10m')
        time.sleep(random.randint(10, 600))

        self.log.debug('Interrupt the task by hard reboot')

        with DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                            line="This node was decommissioned and will not rejoin",
                            node=self.target_node), \
            DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR,
                           line="Fail to send STREAM_MUTATION_DONE",
                           node=self.target_node):
            self.target_node.reboot(hard=True, verify_ssh=True)
            streaming_thread.join(60)

        new_node = None
        if task == 'decommission':
            new_node = decommission_post_action()

        if self.task_used_streaming:
            err = list(streaming_error_logs_stream)
            self.log.debug(err)
        else:
            self.log.debug(
                'Streaming is not used. In latest Scylla, it is optional to use streaming for rebuild and decommission, and repair will not use streaming.')
        self.log.info('Recover the target node by a final rebuild')
        if new_node:
            new_node.wait_db_up(verbose=True, timeout=300)
            new_node.run_nodetool('rebuild')
        else:
            self.target_node.wait_db_up(verbose=True, timeout=300)
            self.target_node.run_nodetool('rebuild')

    def disrupt_decommission_streaming_err(self):
        """
        Stop decommission in middle to trigger some streaming fails, then rebuild the data on the node.
        If the node is decommission unexpectedly, need to re-add a new node to cluster.
        """
        self._set_current_disruption('DecommissionStreamingErr')
        self.break_streaming_task_and_rebuild(task='decommission')

    def disrupt_rebuild_streaming_err(self):
        """
        Stop rebuild in middle to trigger some streaming fails, then rebuild the data on the node.
        """
        self._set_current_disruption('RebuildStreamingErr')
        self.break_streaming_task_and_rebuild(task='rebuild')

    def disrupt_repair_streaming_err(self):
        """
        Stop repair in middle to trigger some streaming fails, then rebuild the data on the node.
        """
        self._set_current_disruption('RepairStreamingErr')
        self.break_streaming_task_and_rebuild(task='repair')

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
            result = self.target_node.remoter.run(
                'sudo dd if=/dev/urandom of={} count=1024'.format(sstable_file))
            self.log.debug('File {} was corrupted by dd'.format(sstable_file))

    def disable_disrupt_corrupt_then_scrub(self):

        # Flush data to sstable
        self.target_node.run_nodetool("flush")

        self.log.debug("Randomly corrupt sstable file")
        self._corrupt_data_file()

        self.log.debug("Rebuild sstable file by scrub, corrupted data file will be skipped.")
        for ks_cf in self.cluster.get_non_system_ks_cf_list(self.target_node):
            self.target_node.run_nodetool("scrub", args=f"{ks_cf.split('.')[0]}")

        self.log.debug(
            'Refreshing the cache by restart the node, and verify the rebuild sstable can be loaded successfully.')
        self.target_node.stop_scylla_server(verify_up=False, verify_down=True)
        self.target_node.start_scylla_server(verify_up=True, verify_down=False)

    @latency_calculator_decorator
    def add_new_node(self, rack=0):
        return self._add_and_init_new_cluster_node(rack=rack)

    @latency_calculator_decorator
    def decommission_node(self, node):
        self.cluster.decommission(node)

    def decommission_nodes(self, add_nodes_number, rack, is_seed: Optional[Union[bool, DefaultValue]] = DefaultValue):
        for _ in range(add_nodes_number):
            if self._is_it_on_kubernetes():
                self.set_last_node_as_target(rack=rack, is_seed=is_seed)
            else:
                self.set_target_node(is_seed=is_seed)
            self.log.info("Next node will be removed %s", self.target_node)
            try:
                InfoEvent(message='StartEvent - ShrinkCluster started decommissioning a node').publish()
                self.decommission_node(self.target_node)
            finally:
                InfoEvent(message='FinishEvent - ShrinkCluster has done decommissioning a node').publish()

    def disrupt_grow_shrink_cluster(self):
        self._disrupt_grow_shrink_cluster(rack=0)

    def disrupt_grow_shrink_new_rack(self):
        self._disrupt_grow_shrink_cluster(rack=max(self.cluster.racks) + 1)

    def _disrupt_grow_shrink_cluster(self, rack=0):
        if rack > 0:
            if not self._is_it_on_kubernetes():
                raise UnsupportedNemesis("SCT rack functionality is implemented only on kubernetes")
            self.log.info("Rack deletion is not supported on kubernetes yet. "
                          "Please see https://github.com/scylladb/scylla-operator/issues/287")
        add_nodes_number = self.tester.params.get('nemesis_add_node_cnt')
        self.unset_current_running_nemesis(self.target_node)
        self._set_current_disruption("GrowCluster")
        self.log.info("Start grow cluster on %s nodes", add_nodes_number)
        InfoEvent(message=f'GrowCluster - Add New node to {rack} rack').publish()
        for _ in range(add_nodes_number):
            added_node = self.add_new_node(rack=rack)
            self.unset_current_running_nemesis(added_node)
            InfoEvent(message='GrowCluster - Done adding New node').publish()
            self.log.info("New node added %s", added_node.name)
        self.log.info("Finish cluster grow")
        time.sleep(self.interval)

        self._set_current_disruption("ShrinkCluster")
        self.log.info("Start shrink cluster on %s nodes", add_nodes_number)
        # Currently on kubernetes first two nodes of each rack are getting seed status
        # Because of such behavior only way to get them decommission is to enable decommissioning
        # TBD: After https://github.com/scylladb/scylla-operator/issues/292 is fixed remove is_seed parameter
        self.decommission_nodes(add_nodes_number, rack, is_seed=None if self._is_it_on_kubernetes() else DefaultValue)
        self.log.info("Finish cluster shrink. Current number of nodes %s", len(self.cluster.nodes))

    def disrupt_hot_reloading_internode_certificate(self):
        """
        https://github.com/scylladb/scylla/issues/6067
        Scylla has the ability to hot reload SSL certificates.
        This test will create and reload new certificates for the inter node communication.
        """
        self._set_current_disruption('ServerSslHotReloadingNemesis')
        if not self.cluster.params.get('server_encrypt'):
            raise UnsupportedNemesis('Server Encryption is not enabled, hence skipping')

        @timeout(timeout=20, allowed_exceptions=(LogContentNotFound, ))
        def check_ssl_reload_log(node_system_log):
            if not list(node_system_log):
                raise LogContentNotFound('Reload SSL message not found in node log')
            return True

        ssl_files_location = json.loads(
            self.target_node.get_scylla_config_param("server_encryption_options"))["certificate"]
        in_place_crt = self.target_node.remoter.run(f"cat {ssl_files_location}",
                                                    ignore_status=True).stdout
        update_certificates()
        node_system_logs = {}
        with DbEventsFilter(
                db_event=DatabaseLogEvent.DATABASE_ERROR,
                line="error GnuTLS:-34, Base64 decoding error"):
            # TBD: To be removed after https://github.com/scylladb/scylla/issues/7909#issuecomment-758062545 is resolved
            for node in self.cluster.nodes:
                node_system_logs[node] = node.follow_system_log(
                    patterns=f'messaging_service - Reloaded {{{ssl_files_location}}}')
                node.remoter.send_files(src='data_dir/ssl_conf/db.crt', dst='/tmp')
                node.remoter.run(f"sudo cp -f /tmp/db.crt {ssl_files_location}")
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
        self.disrupt_mgmt_repair_cli()
        InfoEvent(message='FinishEvent - Manager repair has finished').publish()
        time.sleep(sleep_time_between_ops)
        InfoEvent(message='Starting terminate_and_replace disruption').publish()
        self.disrupt_terminate_and_replace_node()
        InfoEvent(message='Finished terminate_and_replace disruption').publish()
        time.sleep(sleep_time_between_ops)
        InfoEvent(message='Starting grow_shrink disruption').publish()
        self.disrupt_grow_shrink_cluster()
        InfoEvent(message="Finished grow_shrink disruption").publish()

    def disrupt_memory_stress(self):
        """
        Try to run stress-ng to preempt allocated memory of scylla process,
        we don't monitor swap usage in /proc/$scylla_pid/status, just make sure
        no coredump, serious db error occur during the heavy load of memory.
        """
        self._set_current_disruption(f'MemoryStress on {self.target_node}')
        self.target_node.install_epel()
        self.target_node.remoter.run('sudo yum install -y stress-ng')

        self.log.info('Try to allocate 120% available memory, the allocated memory will be swaped out')
        self.target_node.remoter.run(
            "stress-ng --vm-bytes $(awk '/MemAvailable/{printf \"%d\\n\", $2 * 1.2;}' < /proc/meminfo)k --vm-keep -m 1 -t 100")

        self.log.info('Try to allocate 90% total memory, the allocated memory will be swaped out')
        self.target_node.remoter.run(
            "stress-ng --vm-bytes $(awk '/MemTotal/{printf \"%d\\n\", $2 * 0.9;}' < /proc/meminfo)k --vm-keep -m 1 -t 100")

    def disrupt_toggle_cdc_feature_properties_on_table(self):
        """Manipulate cdc feature settings

        Find table with CDC enabled (skip nemesis if not found).
        Randomly select on CDC parameter.
        Toggle the selected parameter state (True/False) or select a random value for TTL

        """
        self._set_current_disruption("ToggleCDCProperties")

        ks_tables_with_cdc = self.cluster.get_all_tables_with_cdc(self.target_node)
        self.log.debug(f"Found next tables with enabled cdc feature: {ks_tables_with_cdc}")

        if not ks_tables_with_cdc:
            self.log.warning("CDC is not enabled on any table. Skipping")
            UnsupportedNemesis("CDC is not enabled on any table. Skipping")
            return

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
        self._set_current_disruption(label="RunCDCStressorTool")

        ks_tables_with_cdc = self.cluster.get_all_tables_with_cdc(self.target_node)
        if not ks_tables_with_cdc:
            self.log.warning("CDC is not enabled on any table. Skipping")
            UnsupportedNemesis("CDC is not enabled on any table. Skipping")
            return
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
            assert (actual_cdc_settings["enabled"] is False,
                    f"CDC options was not disabled. Current: {actual_cdc_settings} expected: {cdc_settings}")
        else:
            assert (actual_cdc_settings == cdc_settings,
                    f"CDC extension settings are differs. Current: {actual_cdc_settings} expected: {cdc_settings}")


def log_time_elapsed_and_status(method):  # pylint: disable=too-many-statements
    """
    Log time elapsed for method to run

    :param method: Remote method to wrap.
    :return: Wrapped method.
    """

    def data_validation_prints(args):
        try:
            if hasattr(args[0].tester, 'data_validator') and args[0].tester.data_validator:
                with args[0].cluster.cql_connection_patient(
                        args[0].cluster.nodes[0], keyspace=args[0].tester.data_validator.keyspace_name) as session:
                    args[0].tester.data_validator.validate_range_not_expected_to_change(session, during_nemesis=True)
                    args[0].tester.data_validator.validate_range_expected_to_change(session, during_nemesis=True)
                    args[0].tester.data_validator.validate_deleted_rows(session, during_nemesis=True)
        except Exception as err:  # pylint: disable=broad-except
            args[0].log.debug(f'Data validator error: {err}')

    def wrapper(*args, **kwargs):  # pylint: disable=too-many-statements
        # pylint: disable=too-many-locals
        args[0].cluster.check_cluster_health()
        num_nodes_before = len(args[0].cluster.nodes)
        start_time = time.time()
        args[0].log.debug('Start disruption at `%s`', datetime.datetime.fromtimestamp(start_time))
        class_name = args[0].get_class_name()
        if class_name.find('Chaos') < 0:
            args[0].metrics_srv.event_start(class_name)
        result = None
        status = True
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

        try:
            result = method(*args, **kwargs)
        except UnsupportedNemesis as exp:
            log_info.update({'subtype': 'skipped', 'skip_reason': str(exp)})
            logging.info("skipping %s", args[0].get_disrupt_name())
            logging.info("cause of: %s", str(exp))
            raise

        except Exception as details:  # pylint: disable=broad-except
            details = str(details)
            args[0].error_list.append(details)
            args[0].log.error('Unhandled exception in method %s', method, exc_info=True)
            log_info.update({'error': details, 'full_traceback': traceback.format_exc()})
            status = False
        finally:
            end_time = time.time()
            time_elapsed = int(end_time - start_time)
            log_info.update({
                'end': int(end_time),
                'duration': time_elapsed,
            })
            args[0].duration_list.append(time_elapsed)
            args[0].operation_log.append(log_info)
            args[0].log.debug('%s duration -> %s s', args[0].current_disruption, time_elapsed)

            if class_name.find('Chaos') < 0:
                args[0].metrics_srv.event_stop(class_name)
            disrupt = args[0].get_disrupt_name()
            del log_info['operation']

            args[0].update_stats(disrupt, status, log_info)
            DisruptionEvent(type=disrupt, status=status, **log_info).publish()
            args[0].cluster.check_cluster_health()
            num_nodes_after = len(args[0].cluster.nodes)
            if num_nodes_before != num_nodes_after:
                args[0].log.error('num nodes before %s and nodes after %s does not match' %
                                  (num_nodes_before, num_nodes_after))
            # TODO: Temporary print. Will be removed later
            data_validation_prints(args=args)
        return result

    return wrapper


class SslHotReloadingNemesis(Nemesis):
    disruptive = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_hot_reloading_internode_certificate()


class NoOpMonkey(Nemesis):
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        time.sleep(300)


class GrowShrinkClusterNemesis(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_grow_shrink_cluster()


class AddRemoveRackNemesis(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_grow_shrink_new_rack()


class StopWaitStartMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_stop_wait_start_scylla_server(600)


class StopStartMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_stop_start_scylla_server()


class RestartThenRepairNodeMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_restart_then_repair_node()


class MultipleHardRebootNodeMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_multiple_hard_reboot_node()


class HardRebootNodeMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_hard_reboot_node()


class SoftRebootNodeMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_soft_reboot_node()


class DrainerMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_nodetool_drain()


class CorruptThenRepairMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_destroy_data_then_repair()


class CorruptThenRebuildMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_destroy_data_then_rebuild()


class DecommissionMonkey(Nemesis):
    disruptive = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_nodetool_decommission()


class DecommissionSeedNode(Nemesis):
    disruptive = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_nodetool_seed_decommission()


class NoCorruptRepairMonkey(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_no_corrupt_repair()


class MajorCompactionMonkey(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_major_compaction()


class RefreshMonkey(Nemesis):
    disruptive = False
    run_with_gemini = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_nodetool_refresh(big_sstable=False)


class RefreshBigMonkey(Nemesis):
    disruptive = False
    run_with_gemini = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_nodetool_refresh(big_sstable=True)


class EnospcMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_nodetool_enospc()


class EnospcAllNodesMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_nodetool_enospc(all_nodes=True)


class NodeToolCleanupMonkey(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_nodetool_cleanup()


class TruncateMonkey(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_truncate()


class TruncateLargeParititionMonkey(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_truncate_large_partition()


class DeleteByPartitionsMonkey(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_delete_10_full_partitions()


class DeleteByRowsRangeMonkey(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_delete_by_rows_range()


class ChaosMonkey(Nemesis):

    @log_time_elapsed_and_status
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

        for name, weight in dist.items():
            name = str(name)
            prefixed_name = prefixed('disrupt_', name)
            if prefixed_name not in all_methods:
                raise ValueError(f"'{name}' is not a valid disruption")

            if not is_nonnegative_number(weight):
                raise ValueError("Each disruption weight must be a non-negative number."
                                 " '{weight}' is not a valid weight.")

            weight = float(weight)
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

    def __init__(self, tester_obj, termination_event, dist: dict, default_weight: float = 1):
        super(CategoricalMonkey, self).__init__(tester_obj, termination_event)
        self.disruption_distribution = CategoricalMonkey.get_disruption_distribution(dist, default_weight)

    @log_time_elapsed_and_status
    def disrupt(self):
        self._random_disrupt()

    def _random_disrupt(self):
        population, weights = self.disruption_distribution
        assert len(population) == len(weights) and population

        method = random.choices(population, weights=weights)[0]
        bound_method = method.__get__(self, CategoricalMonkey)
        self.execute_disrupt_method(bound_method)


class LimitedChaosMonkey(Nemesis):

    @log_time_elapsed_and_status
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
        #  - HardRebootMonkey
        #  - SoftRebootMonkey
        #  - TruncateMonkey
        #  - ToppartitionsMonkey
        #  - MgmtRepair
        #  - NoCorruptRepairMonkey
        #  - SnapshotOperations
        #  - AbortRepairMonkey
        #  - MgmtBackup
        #  - MgmtBackupSpecificKeyspaces
        #  - AddDropColumnMonkey
        self.call_random_disrupt_method(disrupt_methods=['disrupt_nodetool_cleanup', 'disrupt_nodetool_decommission',
                                                         'disrupt_nodetool_drain', 'disrupt_nodetool_refresh',
                                                         'disrupt_stop_start_scylla_server', 'disrupt_major_compaction',
                                                         'disrupt_modify_table', 'disrupt_nodetool_enospc',
                                                         'disrupt_stop_wait_start_scylla_server',
                                                         'disrupt_hard_reboot_node', 'disrupt_soft_reboot_node',
                                                         'disrupt_truncate', 'disrupt_show_toppartitions',
                                                         'disrupt_mgmt_repair_cli', 'disrupt_no_corrupt_repair',
                                                         'disrupt_snapshot_operations', 'disrupt_abort_repair',
                                                         'disrupt_mgmt_backup',
                                                         'disrupt_mgmt_backup_specific_keyspaces',
                                                         'disrupt_add_drop_column'])


class ScyllaCloudLimitedChaosMonkey(Nemesis):

    @log_time_elapsed_and_status
    def disrupt(self):
        # Limit the nemesis scope to only one relevant to scylla cloud, where we defined we don't have AWS api access:
        self.call_random_disrupt_method(disrupt_methods=['disrupt_nodetool_cleanup',
                                                         'disrupt_nodetool_drain', 'disrupt_nodetool_refresh',
                                                         'disrupt_stop_start_scylla_server', 'disrupt_major_compaction',
                                                         'disrupt_modify_table', 'disrupt_nodetool_enospc',
                                                         'disrupt_stop_wait_start_scylla_server',
                                                         'disrupt_soft_reboot_node',
                                                         'disrupt_truncate'])


class AllMonkey(Nemesis):

    @log_time_elapsed_and_status
    def disrupt(self):
        self.call_random_disrupt_method(predefined_sequence=True)


class MdcChaosMonkey(Nemesis):

    @log_time_elapsed_and_status
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

    @log_time_elapsed_and_status
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

    @log_time_elapsed_and_status
    def disrupt(self):
        self.log.info('UpgradeNemesisOneNode begin')
        self.upgrade_node(self.cluster.node_to_upgrade)  # pylint: disable=no-member

        self.log.info('UpgradeNemesisOneNode end')


class RollbackNemesis(Nemesis):

    # rollback a single node
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
        node.run_nodetool("drain", timeout=3600, coredump_on_timeout=True)
        node.remoter.run('sudo cp {0}-backup {0}'.format(SCYLLA_YAML_PATH))
        node.remoter.run('sudo systemctl restart scylla-server.service')
        node.wait_db_up(verbose=True)
        new_ver = node.remoter.run('rpm -qa scylla-server')
        self.log.debug('original scylla-server version is %s, latest: %s' % (orig_ver, new_ver))
        if orig_ver == new_ver:
            raise ValueError('scylla-server version isn\'t changed')

    @log_time_elapsed_and_status
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

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_modify_table()


class AddDropColumnMonkey(Nemesis):
    disruptive = False
    run_with_gemini = False
    networking = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_add_drop_column()


class ToggleTableIcsMonkey(Nemesis):
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_toggle_table_ics()


class MgmtBackup(Nemesis):
    disruptive = False

    def __init__(self, *args, **kwargs):
        super(MgmtBackup, self).__init__(*args, **kwargs)
        self.filter_seed = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_mgmt_backup()


class MgmtBackupSpecificKeyspaces(Nemesis):
    disruptive = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_mgmt_backup_specific_keyspaces()


class MgmtRepair(Nemesis):
    disruptive = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.log.info('disrupt_mgmt_repair_cli Nemesis begin')
        self.disrupt_mgmt_repair_cli()
        self.log.info('disrupt_mgmt_repair_cli Nemesis end')
        # For Manager APIs test, use: self.disrupt_mgmt_repair_api()


class AbortRepairMonkey(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_abort_repair()


class NodeTerminateAndReplace(Nemesis):
    disruptive = True
    # It should not be run on kubernetes, since it is a manual procedure
    # While on kubernetes we put it all on scylla-operator
    kubernetes = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_terminate_and_replace_node()


class OperatorNodeTerminateAndRecover(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_terminate_and_recover_node_kubernetes()


class OperatorNodeTerminateAndReplace(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_terminate_and_replace_node_kubernetes()


class OperatorNodeTerminateDecommissionAdd(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_terminate_decommission_add_node_kubernetes()


class OperatorNodeReplace(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_replace_node_kubernetes()


class ScyllaKillMonkey(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_kill_scylla()


class ValidateHintedHandoffShortDowntime(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disable_disrupt_validate_hh_short_downtime()


class SnapshotOperations(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_snapshot_operations()


class NodeRestartWithResharding(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_restart_with_resharding()


class ClusterRollingRestart(Nemesis):
    disruptive = True
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_rolling_restart_cluster()


class TopPartitions(Nemesis):
    disruptive = False
    kubernetes = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_show_toppartitions()


class RandomInterruptionNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_network_random_interruptions()


class BlockNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_network_block()


class RejectInterNodeNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self._disrupt_network_reject_inter_node_communication()


class RejectNodeExporterNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_network_reject_node_exporter()


class RejectThriftNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_network_reject_thrift()


class StopStartInterfacesNetworkMonkey(Nemesis):
    disruptive = True
    networking = True
    run_with_gemini = False

    @log_time_elapsed_and_status
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
        super(DisruptiveMonkey, self).__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(disruptive=True)

    @log_time_elapsed_and_status
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
        super(NonDisruptiveMonkey, self).__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(disruptive=False)

    @log_time_elapsed_and_status
    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class NetworkMonkey(Nemesis):
    # Limit the nemesis scope:
    #  - RandomInterruptionNetworkMonkey
    #  - StopStartInterfacesNetworkMonkey
    #  - BlockNetworkMonkey
    def __init__(self, *args, **kwargs):
        super(NetworkMonkey, self).__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(networking=True)

    @log_time_elapsed_and_status
    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class GeminiChaosMonkey(Nemesis):
    # Limit the nemesis scope to use with gemini
    # - StopStartMonkey
    # - RestartThenRepairNodeMonkey
    def __init__(self, *args, **kwargs):
        super(GeminiChaosMonkey, self).__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend(run_with_gemini=True)

    @log_time_elapsed_and_status
    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class GeminiNonDisruptiveChaosMonkey(Nemesis):
    def __init__(self, *args, **kwargs):
        super(GeminiNonDisruptiveChaosMonkey, self).__init__(*args, **kwargs)
        run_with_gemini = set(self.get_list_of_methods_compatible_with_backend(run_with_gemini=True))
        non_disruptive = set(self.get_list_of_methods_compatible_with_backend(disruptive=False))
        self.disrupt_methods_list = run_with_gemini.intersection(non_disruptive)

    @log_time_elapsed_and_status
    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list)


class KubernetesScyllaOperatorMonkey(Nemesis):
    # All Nemesis that could be run on kubernetes backend
    disruptive = True

    def __init__(self, *args, **kwargs):
        super(KubernetesScyllaOperatorMonkey, self).__init__(*args, **kwargs)
        self.disrupt_methods_list = self.get_list_of_methods_compatible_with_backend()

    @log_time_elapsed_and_status
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
            'disrupt_rolling_restart_cluster',
            'disrupt_grow_shrink_cluster',
            'disrupt_grow_shrink_new_rack',
            'disrupt_stop_start_scylla_server',
            'disrupt_terminate_and_replace_node_kubernetes',
            'disrupt_terminate_and_recover_node_kubernetes',
            'disrupt_replace_node_kubernetes',
        ]

    @log_time_elapsed_and_status
    def disrupt(self):
        self.call_random_disrupt_method(disrupt_methods=self.disrupt_methods_list, predefined_sequence=True)


class NemesisSequence(Nemesis):
    disruptive = True
    networking = False
    run_with_gemini = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_run_unique_sequence()


class TerminateAndRemoveNodeMonkey(Nemesis):
    """Remove a Node from a Scylla Cluster (Down Scale)"""
    disruptive = False
    # It should not be run on kubernetes, since it is a manual procedure
    # While on kubernetes we put it all on scylla-operator
    kubernetes = False

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_remove_node_then_add_node()


class SisyphusMonkey(Nemesis):

    def __init__(self, *args, **kwargs):
        super(SisyphusMonkey, self).__init__(*args, **kwargs)
        self.get_all_disrupt_methods()
        self.shuffle_list_of_disruptions()

    @log_time_elapsed_and_status
    def disrupt(self):
        self.call_next_nemesis()


class ToggleCDCMonkey(Nemesis):
    disruptive = False

    def disrupt(self):
        self.disrupt_toggle_cdc_feature_properties_on_table()


class CDCStressorMonkey(Nemesis):
    disruptive = False

    def disrupt(self):
        self.disrupt_run_cdcstressor_tool()


class DecommissionStreamingErrMonkey(Nemesis):

    disruptive = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_decommission_streaming_err()


class RebuildStreamingErrMonkey(Nemesis):

    disruptive = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_rebuild_streaming_err()


class RepairStreamingErrMonkey(Nemesis):

    disruptive = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_repair_streaming_err()


DEPRECATED_LIST_OF_NEMESISES = [UpgradeNemesis, UpgradeNemesisOneNode, RollbackNemesis]

COMPLEX_NEMESIS = [NoOpMonkey, ChaosMonkey,
                   LimitedChaosMonkey,
                   ScyllaCloudLimitedChaosMonkey,
                   AllMonkey, MdcChaosMonkey,
                   DisruptiveMonkey, NonDisruptiveMonkey, GeminiNonDisruptiveChaosMonkey,
                   GeminiChaosMonkey, NetworkMonkey, KubernetesScyllaOperatorMonkey, SisyphusMonkey,
                   CategoricalMonkey]


# TODO: https://trello.com/c/vwedwZK2/1881-corrupt-the-scrub-fails
# class CorruptThenScrubMonkey(Nemesis):
#     disruptive = False
#
#     @log_time_elapsed_and_status
#     def disrupt(self):
#         self.disable_disrupt_corrupt_then_scrub()


class MemoryStressMonkey(Nemesis):
    disruptive = True

    @log_time_elapsed_and_status
    def disrupt(self):
        self.disrupt_memory_stress()
