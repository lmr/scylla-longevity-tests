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

import re
import datetime
import time
import os
import platform
import logging
import json
import urllib.parse

from textwrap import dedent
from math import sqrt
from typing import Optional
from functools import cached_property
from collections import defaultdict

import yaml
import requests

from sdcm.es import ES
from sdcm.test_config import TestConfig
from sdcm.utils.common import normalize_ipv6_url, get_ami_tags
from sdcm.utils.git import get_git_commit_id
from sdcm.utils.decorators import retrying
from sdcm.sct_events.system import ElasticsearchEvent
from sdcm.utils.ci_tools import get_job_name, get_job_url

LOGGER = logging.getLogger(__name__)

FS_SIZE_METRIC = 'node_filesystem_size_bytes'
FS_SIZE_METRIC_OLD = 'node_filesystem_size'
AVAIL_SIZE_METRIC = 'node_filesystem_avail_bytes'
AVAIL_SIZE_METRIC_OLD = 'node_filesystem_avail'
KB_SIZE = 2 ** 10
MB_SIZE = KB_SIZE * 1024
GB_SIZE = MB_SIZE * 1024
SCYLLA_DIR = "/var/lib/scylla"


class CassandraStressCmdParseError(Exception):
    def __init__(self, cmd, ex):
        super().__init__()
        self.command = cmd
        self.exception = repr(ex)

    def __str__(self):
        return dedent("""
            Stress command: '{0.command}'
            Error: {0.exception}""".format(self))

    def __repr__(self):
        return self.__str__()


def stddev(lst):
    mean = float(sum(lst)) / len(lst)
    return sqrt(sum((x - mean)**2 for x in lst) / len(lst))


def get_stress_cmd_params(cmd):
    """
    Parsing cassandra stress command
    :param cmd: stress cmd
    :return: dict with params
    """

    # pylint: disable=too-many-branches
    cmd_params = {
        "raw_cmd": cmd
    }
    try:
        cmd = cmd.strip().split('cassandra-stress')[1].strip()
        if cmd.split(' ')[0] in ['read', 'write', 'mixed', 'counter_write', 'user']:
            cmd_params['command'] = cmd.split(' ')[0]
            if 'no-warmup' in cmd:
                cmd_params['no-warmup'] = True

            match = re.search(r'(cl\s?=\s?\w+)', cmd)
            if match:
                cmd_params['cl'] = match.group(0).split('=')[1].strip()

            match = re.search(r'(duration\s?=\s?\w+)', cmd)
            if match:
                cmd_params['duration'] = match.group(0).split('=')[1].strip()

            match = re.search(r'( n\s?=\s?\w+)', cmd)
            if match:
                cmd_params['n'] = match.group(0).split('=')[1].strip()
            match = re.search(r'profile=(\S+)\s+', cmd)
            if match:
                cmd_params['profile'] = match.group(1).strip()
                match = re.search(r'ops(\S+)\s+', cmd)
                if match:
                    cmd_params['ops'] = match.group(1).split('=')[0].strip('(')

            for temp in cmd.split(' -')[1:]:
                try:
                    key, value = temp.split(" ", 1)
                except ValueError as ex:
                    LOGGER.warning("%s:%s", temp, ex)
                else:
                    cmd_params[key] = value.strip().replace("'", "")
            if 'rate' in cmd_params:
                # split rate section on separate items
                if 'threads' in cmd_params['rate']:
                    cmd_params['rate threads'] = \
                        re.search(r'(threads\s?=\s?(\w+))', cmd_params['rate']).group(2)
                if 'throttle' in cmd_params['rate']:
                    cmd_params['throttle threads'] =\
                        re.search(r'(throttle\s?=\s?(\w+))', cmd_params['rate']).group(2)
                if 'fixed' in cmd_params['rate']:
                    cmd_params['fixed threads'] =\
                        re.search(r'(fixed\s?=\s?(\w+))', cmd_params['rate']).group(2)
                del cmd_params['rate']

        return cmd_params
    except Exception as ex:
        raise CassandraStressCmdParseError(cmd=cmd, ex=ex) from None


def get_stress_bench_cmd_params(cmd):
    """
    Parsing bench stress command
    :param cmd: stress cmd
    :return: dict with params
    """
    cmd = cmd.strip().split('scylla-bench')[1].strip()
    cmd_params = {}
    for key in ['partition-count', 'clustering-row-count', 'clustering-row-size', 'mode',
                'workload', 'concurrency', 'max-rate', 'connection-count', 'replication-factor',
                'timeout', 'client-compression', 'duration']:
        match = re.search(r'(-' + key + r'\s+([^-| ]+))', cmd)
        if match:
            cmd_params[key] = match.group(2).strip()
    return cmd_params


def get_stress_harry_cmd_params(cmd):
    """
    Parsing cassandra-harry stress command
    :param cmd: stress cmd
    :return: dict with params
    """
    cmd = cmd.split('cassandra-harry')[1].strip()
    cmd_params = {}
    for key in ['run-time', 'run-time-unit']:
        match = re.search(fr'(-{key}\s+([^-| ]+))', cmd)
        if match:
            cmd_params[key] = match.group(2).strip()
    return cmd_params


def get_ycsb_cmd_params(cmd):
    """
    Parsing ycsb command
    :param cmd: stress cmd
    :return: dict with params
    """

    cmd_params = {
        "raw_cmd": cmd
    }

    threads_regex = re.compile(r'-threads\s*(.*?)[\s$]')
    key_value_regex = re.compile(r"-p\s.*?(?P<key>.*?)=(?P<value>.*?)(\s|$)")
    for match in key_value_regex.finditer(cmd):
        match_dict = match.groupdict()
        cmd_params[match_dict['key']] = match_dict['value']
    match = threads_regex.search(cmd)
    if match:
        cmd_params['threads'] = match.group(1)
    return cmd_params


def get_raw_cmd_params(cmd):
    cmd = cmd.strip()
    cmd_params = {
        'raw_cmd': cmd
    }
    return cmd_params


def get_gemini_cmd_params(cmd):
    return get_raw_cmd_params(cmd)


def get_cdcreader_cmd_params(cmd):
    return get_raw_cmd_params(cmd)


class PrometheusDBStats:
    def __init__(self, host, port=9090, protocol='http', alternator=None):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.range_query_url = "{}://{}:{}/api/v1/query_range?query=".format(
            protocol, normalize_ipv6_url(host), port)
        self.config = self.get_configuration()
        self.alternator = alternator

    @property
    def scylla_scrape_interval(self):
        return int(self.config["scrape_configs"]["scylla"]["scrape_interval"][:-1])

    @retrying(n=5, sleep_time=7, allowed_exceptions=(requests.ConnectionError, requests.HTTPError))
    def request(self, url, post=False):
        kwargs = {}
        if self.protocol == 'https':
            kwargs['verify'] = False
        if post:
            response = requests.post(url, **kwargs)
        else:
            response = requests.get(url, **kwargs)
        response.raise_for_status()

        result = json.loads(response.content)
        LOGGER.debug("Response from Prometheus server: %s", str(result)[:200])
        if result["status"] == "success":
            return result
        else:
            LOGGER.error("Prometheus returned error: %s", result)
        return None

    def get_configuration(self):
        result = self.request(url="{}://{}:{}/api/v1/status/config".format(
            self.protocol, normalize_ipv6_url(self.host), self.port))
        configs = yaml.safe_load(result["data"]["yaml"])
        LOGGER.debug("Parsed Prometheus configs: %s", configs)
        new_scrape_configs = {}
        for conf in configs["scrape_configs"]:
            new_scrape_configs[conf["job_name"]] = conf
        configs["scrape_configs"] = new_scrape_configs
        return configs

    def query(self, query, start, end, scrap_metrics_step=None):
        """
        :param start: time=<rfc3339 | unix_timestamp>: Start timestamp.
        :param end: time=<rfc3339 | unix_timestamp>: End timestamp.
        :param scrap_metrics_step is the granularity of data requested from Prometheus DB
        :param query:
        :return: {
                  metric: { },
                  values: [[linux_timestamp1, value1], [linux_timestamp2, value2]...[linux_timestampN, valueN]]
                 }
        """
        url = "{}://{}:{}/api/v1/query_range?query=".format(self.protocol, normalize_ipv6_url(self.host), self.port)
        if not scrap_metrics_step:
            scrap_metrics_step = self.scylla_scrape_interval
        _query = "{url}{query}&start={start}&end={end}&step={scrap_metrics_step}".format(
            url=url, query=query, start=start, end=end, scrap_metrics_step=scrap_metrics_step)
        LOGGER.debug("Query to PrometheusDB: %s", _query)
        result = self.request(url=_query)
        if result:
            return result["data"]["result"]
        else:
            LOGGER.error("Prometheus query unsuccessful!")
            return []

    @staticmethod
    def _check_start_end_time(start_time, end_time):
        if end_time - start_time < 120:
            LOGGER.warning("Time difference too low to make a query [start_time: %s, end_time: %s", start_time,
                           end_time)
            return False
        return True

    def _get_query_values(self, query, start_time, end_time, scrap_metrics_step=None):
        results = self.query(query=query, start=start_time, end=end_time, scrap_metrics_step=scrap_metrics_step)
        if results:
            return results[0]["values"]
        else:
            return []

    def get_throughput(self, start_time, end_time, scrap_metrics_step=None):
        """
        Get Scylla throughput (ops/second) from PrometheusDB

        :return: list of tuples (unix time, op/s)
        """
        if not self._check_start_end_time(start_time, end_time):
            return []
        # the query is taken from the Grafana Dashborad definition
        if self.alternator:
            query = "sum(irate(scylla_alternator_operation{}[30s]))"
        else:
            query = "sum(irate(scylla_transport_requests_served{}[30s]))"
        return self._get_query_values(query, start_time, end_time, scrap_metrics_step=scrap_metrics_step)

    def get_scylla_reactor_utilization(self, start_time, end_time, scrap_metrics_step=None, instance=None):
        """
        Get Scylla CPU (avg) from PrometheusDB

        :return: list of tuples (unix time, op/s)
        """
        if not self._check_start_end_time(start_time, end_time):
            return []

        instance_filter = f'instance="{instance}"' if instance else ""
        query = "avg(scylla_reactor_utilization{%s})" % instance_filter
        res = self._get_query_values(query, start_time, end_time, scrap_metrics_step=scrap_metrics_step)
        if res:
            res = [float(value[1]) for value in res]
            return sum(res) / len(res)
        else:
            return res

    def get_scylla_scheduler_runtime_ms(self, start_time, end_time, node_ip, irate_sample_sec='30s'):
        """
        Get Scylla CPU scheduler runtime from PrometheusDB

        :return: list of tuples (unix time, op/s)
        """
        if not self._check_start_end_time(start_time, end_time):
            return {}
        # the query is taken from the Grafana Dashborad definition
        query = 'avg(irate(scylla_scheduler_runtime_ms{group=~"sl:.*", instance="%s"}  [%s] )) ' \
            'by (group, instance)' % (node_ip, irate_sample_sec)
        results = self.query(query=query, start=start_time, end=end_time)
        res = defaultdict(dict)
        for item in results:
            res[item['metric']['instance']].update({item['metric']['group']:
                                                    [float(runtime[1]) for runtime in item['values']]})
        return res

    def get_scylla_scheduler_shares_per_sla(self, start_time, end_time, node_ip):  # pylint: disable=invalid-name
        """
        Get scylla_scheduler_shares from PrometheusDB

        :return: list of tuples (unix time, op/s)
        """
        if not self._check_start_end_time(start_time, end_time):
            return {}
        # the query is taken from the Grafana Dashborad definition
        query = 'avg(scylla_scheduler_shares{group=~"sl:.*", instance="%s"} ) by (group, instance)' % node_ip
        results = self.query(query=query, start=start_time, end=end_time)
        res = {}
        for item in results:
            try:
                res[item['metric']['group']] = {int(i[1]) for i in item['values']}
            except Exception as error:  # pylint: disable=broad-except
                # average value may be returned not integer. Ignore it
                LOGGER.error("Failed to analyze results of query: %s\nResults: %s\nError: %s", query, results, error)
        return res

    def get_scylla_storage_proxy_replica_cross_shard_ops(self, start_time, end_time):
        query = """sum(irate(scylla_storage_proxy_replica_cross_shard_ops{instance=~".+[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}.+",shard=~"[0-9]+"}[1m])) by (dc)"""  # pylint: disable=line-too-long
        query = urllib.parse.quote(query)
        results = self.query(query, start_time, end_time)
        cross_shards_ops_per_node_shard_by_dc = []
        for res in results:
            cross_shards_ops_per_node_shard_by_dc.extend([float(v) for _, v in res["values"]])

        return cross_shards_ops_per_node_shard_by_dc

    def get_latency(self, start_time, end_time, latency_type, scrap_metrics_step=None):
        """latency values are returned in microseconds"""
        assert latency_type in ["read", "write"]
        if not self._check_start_end_time(start_time, end_time):
            return []
        query = "histogram_quantile(0.99, sum(rate(scylla_storage_proxy_" \
            "coordinator_%s_latency_bucket{}[30s])) by (le))" % latency_type
        return self._get_query_values(query, start_time, end_time, scrap_metrics_step=scrap_metrics_step)

    def get_latency_read_99(self, start_time, end_time, scrap_metrics_step=None):
        return self.get_latency(start_time, end_time, latency_type="read",
                                scrap_metrics_step=scrap_metrics_step)

    def get_latency_write_99(self, start_time, end_time, scrap_metrics_step=None):
        return self.get_latency(start_time, end_time, latency_type="write",
                                scrap_metrics_step=scrap_metrics_step)

    def create_snapshot(self):
        url = "http://{}:{}/api/v1/admin/tsdb/snapshot".format(normalize_ipv6_url(self.host), self.port)
        result = self.request(url, True)
        LOGGER.debug('Request result: {}'.format(result))
        return result

    @staticmethod
    def generate_node_capacity_query_postfix(node):
        return f'{{mountpoint="{SCYLLA_DIR}", instance=~".*?{node.private_ip_address}.*?"}}'

    def get_used_capacity_gb(self, node):  # pylint: disable=too-many-locals
        #  example: node_filesystem_size_bytes{mountpoint="/var/lib/scylla",
        #  instance=~".*?10.0.79.46.*?"}-node_filesystem_avail_bytes{mountpoint="/var/lib/scylla",
        #  instance=~".*?10.0.79.46.*?"}
        node_capacity_query_postfix = self.generate_node_capacity_query_postfix(node)
        filesystem_capacity_query = f'{FS_SIZE_METRIC}{node_capacity_query_postfix}'
        used_capacity_query = f'{filesystem_capacity_query}-{AVAIL_SIZE_METRIC}{node_capacity_query_postfix}'
        LOGGER.debug("filesystem_capacity_query: %s", filesystem_capacity_query)

        fs_size_res = self.query(query=filesystem_capacity_query, start=int(time.time()) - 5,
                                 end=int(time.time()))
        if not fs_size_res:
            LOGGER.warning("No results from Prometheus query: %s", filesystem_capacity_query)
            return 0
            # assert fs_size_res, "No results from Prometheus"
        if not fs_size_res[0]:  # if no returned values - try the old metric names.
            filesystem_capacity_query = f'{FS_SIZE_METRIC_OLD}{node_capacity_query_postfix}'
            used_capacity_query = f'{filesystem_capacity_query}-{AVAIL_SIZE_METRIC_OLD}{node_capacity_query_postfix}'
            LOGGER.debug("filesystem_capacity_query: %s", filesystem_capacity_query)
            fs_size_res = self.query(query=filesystem_capacity_query, start=int(time.time()) - 5,
                                     end=int(time.time()))

        assert fs_size_res[0], "Could not resolve capacity query result."
        LOGGER.debug("used_capacity_query: %s", used_capacity_query)
        used_cap_res = self.query(query=used_capacity_query, start=int(time.time()) - 5,
                                  end=int(time.time()))
        assert used_cap_res, "No results from Prometheus"
        used_size_mb = float(used_cap_res[0]["values"][0][1]) / float(MB_SIZE)
        used_size_gb = round(float(used_size_mb / 1024), 2)
        LOGGER.debug("The used filesystem capacity on node %s is: %s MB/ %s GB",
                     node.public_ip_address, used_size_mb, used_size_gb)
        return used_size_gb

    def get_filesystem_total_size_gb(self, node) -> float:
        """
        :param node:
        :return: FS size in GB
        """
        node_capacity_query_postfix = self.generate_node_capacity_query_postfix(node)
        filesystem_capacity_query = f'{FS_SIZE_METRIC}{node_capacity_query_postfix}'
        fs_size_res = self.query(query=filesystem_capacity_query, start=int(time.time()) - 5,
                                 end=int(time.time()))
        assert fs_size_res, "No results from Prometheus"
        if not fs_size_res[0]:  # if no returned values - try the old metric names.
            filesystem_capacity_query = f'{FS_SIZE_METRIC_OLD}{node_capacity_query_postfix}'
            LOGGER.debug("filesystem_capacity_query: %s", filesystem_capacity_query)
            fs_size_res = self.query(query=filesystem_capacity_query, start=int(time.time()) - 5,
                                     end=int(time.time()))
        assert fs_size_res[0], "Could not resolve capacity query result."
        LOGGER.debug("fs_size_res: %s", fs_size_res)
        fs_size_gb = float(fs_size_res[0]["values"][0][1]) / float(GB_SIZE)
        return fs_size_gb


class Stats:
    """Create and update a document in Elasticsearch."""

    def __init__(self, *args, **kwargs):
        self._test_index = kwargs.get("test_index")
        self._test_id = kwargs.get("test_id")
        self._es_doc_type = "test_stats"
        self._stats = {}
        self.test_config = TestConfig()

        # For using this class as a base for TestStatsMixin.
        if not self._test_id:
            super().__init__(*args, **kwargs)

    @cached_property
    def elasticsearch(self) -> Optional[ES]:
        try:
            return ES()
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to create ES connection (doc_id=%s)", self._test_id)
            ElasticsearchEvent(doc_id=self._test_id, error=str(exc)).publish()
            return None

    def create(self) -> None:
        if not self.elasticsearch:
            LOGGER.error("Failed to create test stats: ES connection is not created (doc_id=%s)", self._test_id)
            return
        try:
            self.elasticsearch.create_doc(
                index=self._test_index,
                doc_type=self._es_doc_type,
                doc_id=self._test_id,
                body=self._stats,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to create test stats (doc_id=%s)", self._test_id)
            ElasticsearchEvent(doc_id=self._test_id, error=str(exc)).publish()

    def update(self, data: dict) -> None:
        if not self.elasticsearch:
            LOGGER.error("Failed to update test stats: ES connection is not created (doc_id=%s)", self._test_id)
            return
        try:
            self.elasticsearch.update_doc(
                index=self._test_index,
                doc_type=self._es_doc_type,
                doc_id=self._test_id,
                body=data,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to update test stats (doc_id=%s)", self._test_id)
            ElasticsearchEvent(doc_id=self._test_id, error=str(exc)).publish()

    def exists(self) -> Optional[bool]:
        if not self.elasticsearch:
            LOGGER.error("Failed to check for test stats existence: ES connection is not created (doc_id=%s)",
                         self._test_id)
            return None
        try:
            return self.elasticsearch.exists(
                index=self._test_index,
                doc_type=self._es_doc_type,
                id=self._test_id,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to check for test stats existence (doc_id=%s)", self._test_id)
            ElasticsearchEvent(doc_id=self._test_id, error=str(exc)).publish()
        return None


class TestStatsMixin(Stats):
    """
    This mixin is responsible for saving test details and statistics in database.
    """
    KEYS = {
        'test_details': {},
        'setup_details': {},
        'versions': {},
        'results': {},
        'nemesis': {},
        'errors': [],
        'coredumps': {},
    }
    PROMETHEUS_STATS = ('throughput', 'latency_read_99', 'latency_write_99')
    PROMETHEUS_STATS_UNITS = {'throughput': "op/s", 'latency_read_99': "us", 'latency_write_99': "us"}
    STRESS_STATS = ('op rate', 'latency mean', 'latency 99th percentile')
    STRESS_STATS_TOTAL = ('op rate', 'Total errors')

    def _create_test_id(self, doc_id_with_timestamp=False):
        """Return doc_id equal unified test-id

        Generate doc_id for ES document as unified global test-id
        if doc_id_with_timestamp is true, create ES
        document with global test_id + timestamp

        :param doc_id_with_timestamp: add timestamp to test_id , defaults to False
        :type doc_id_with_timestamp: bool, optional
        :returns: doc_id for document in ES
        :rtype: {str}
        """
        # avoid cyclic-decencies between cluster and db_stats
        doc_id = self.test_config.test_id()
        # NOTE: 'cluster_index' is needed in case of multi-tenant perf test on K8S
        if hasattr(self, 'cluster_index'):
            doc_id += f"--{self.cluster_index}"
        if doc_id_with_timestamp:
            doc_id += "_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        return doc_id

    def _init_stats(self):
        return {k: v.copy() for k, v in self.KEYS.items()}

    def get_scylla_versions(self):
        versions = {}
        try:
            node = self.db_cluster.nodes[0]
            if node.scylla_version_detailed and 'build-id' in node.scylla_version_detailed:
                build_id = node.scylla_version_detailed.split('with build-id ')[-1].strip()
            else:
                build_id = node.get_scylla_build_id()
            if node.distro.is_rhel_like:
                version_cmd = 'rpm -qa |grep scylla'
            else:
                version_cmd = "dpkg -l |grep scylla|awk '{print $2 \"-\" $3}'"
            versions_output = node.remoter.run(version_cmd).stdout.splitlines()
            for line in versions_output:
                for package in ['scylla-jmx', 'scylla-server', 'scylla-tools', 'scylla-enterprise-jmx',
                                'scylla-enterprise-server', 'scylla-enterprise-tools']:
                    match = re.search(r'(%s-([\w.~]+)-(0.)?([0-9]{8,8}).(\w+).)' % package, line)
                    if match:
                        versions[package.replace('-enterprise', '')] = {'version': match.group(2),
                                                                        'date': match.group(4),
                                                                        'commit_id': match.group(5),
                                                                        'build_id': build_id if build_id else ''}
        except Exception as ex:  # pylint: disable=broad-except
            LOGGER.error('Failed getting scylla versions: %s', ex)

        # append Scylla build mode in scylla_versions
        versions['scylla-server']['build_mode'] = node.scylla_build_mode

        return versions

    def get_setup_details(self):
        exclude_details = ['send_email', 'email_recipients', 'es_url', 'es_password', 'reuse_cluster']
        setup_details = {}
        is_gce = self.params.get('cluster_backend') == 'gce'

        test_params = self.params.items()

        for key, value in test_params:
            if key in exclude_details or (isinstance(key, str) and key.startswith('stress_cmd')):  # pylint: disable=no-else-continue
                continue
            else:
                if is_gce and key in \
                        ['instance_type_loader',  # pylint: disable=no-else-continue
                         'instance_type_monitor',
                         'instance_type_db']:
                    # exclude these params from gce run
                    continue
                elif key == 'n_db_nodes' and isinstance(value, str) and re.search(r'\s', value):  # multidc
                    setup_details['n_db_nodes'] = sum([int(i) for i in value.split()])
                else:
                    setup_details[key] = value

        if self.params.get('cluster_backend') == 'aws':
            setup_details["ami_tags_db_scylla"] = []
            region_names = self.params.region_names
            ami_list = self.params.get('ami_id_db_scylla').split()
            for ami_id, region_name in zip(ami_list, region_names):
                tags = get_ami_tags(ami_id, region_name)
                setup_details["ami_tags_db_scylla"].append(tags)

        new_scylla_packages = self.params.get('update_db_packages')
        setup_details['packages_updated'] = bool(new_scylla_packages and os.listdir(new_scylla_packages))
        setup_details['cpu_platform'] = 'UNKNOWN'
        if is_gce and self.db_cluster:
            if hasattr(self.db_cluster.nodes[0]._instance, "cpu_platform"):  # pylint: disable=protected-access
                setup_details['cpu_platform'] = self.db_cluster.nodes[0]._instance.cpu_platform  # pylint: disable=protected-access
            else:
                setup_details['cpu_platform'] = self.db_cluster.nodes[0]._instance.extra.get(  # pylint: disable=protected-access
                    'cpuPlatform', 'UNKNOWN')

        setup_details['db_cluster_node_details'] = {}
        setup_details['sysctl_output'] = []
        return setup_details

    def get_test_details(self):
        test_details = {}
        test_details['sct_git_commit'] = get_git_commit_id()
        test_details['job_name'] = get_job_name()
        test_details['job_url'] = get_job_url()
        test_details['start_host'] = platform.node()
        test_details['test_duration'] = self.params.get(key='test_duration')
        test_details['start_time'] = int(time.time())
        test_details['grafana_snapshots'] = []
        test_details['grafana_screenshots'] = []
        test_details['grafana_annotations'] = []
        test_details['prometheus_data'] = ""
        test_details['test_id'] = self.test_config.test_id()
        test_details['log_files'] = {}
        return test_details

    def create_test_stats(self, sub_type=None, specific_tested_stats=None,  # pylint: disable=too-many-arguments
                          doc_id_with_timestamp=False, append_sub_test_to_name=True, test_name=None, test_index=None):

        if not self.create_stats:
            return
        if test_index:
            self._test_index = test_index
        elif not self._test_index:
            self._test_index = self.__class__.__name__.lower()
        self._test_id = self._create_test_id(doc_id_with_timestamp)
        self._stats = self._init_stats()
        self._stats['setup_details'] = self.get_setup_details()
        self._stats['versions'] = self.get_scylla_versions()
        self._stats['test_details'] = self.get_test_details()
        test_name = test_name if test_name else self.id()
        if sub_type:
            self._stats['test_details']['sub_type'] = sub_type
        if sub_type and append_sub_test_to_name:
            self._stats['test_details']['test_name'] = '{}_{}'.format(test_name, sub_type)
        else:
            self._stats['test_details']['test_name'] = test_name
        for stat in self.PROMETHEUS_STATS:
            self._stats['results'][stat] = {}
        if specific_tested_stats:
            self._stats['results'].update(specific_tested_stats)
            self.log.info("Creating specific tested stats of: {}".format(specific_tested_stats))
        self.create()

    def update_stress_cmd_details(self, cmd, prefix='', stresser="cassandra-stress", aggregate=True):
        section = '{prefix}{stresser}'.format(prefix=prefix, stresser=stresser)
        if section not in self._stats['test_details']:
            self._stats['test_details'][section] = [] if aggregate else {}
        if stresser == "cassandra-stress":
            cmd_params = get_stress_cmd_params(cmd)
        elif stresser == "scylla-bench":
            cmd_params = get_stress_bench_cmd_params(cmd)
        elif stresser == "cassandra-harry":
            cmd_params = get_stress_harry_cmd_params(cmd)
        elif stresser == 'ycsb':
            cmd_params = get_ycsb_cmd_params(cmd)
        elif stresser in ['gemini', 'ndbench']:
            cmd_params = get_raw_cmd_params(cmd)
        elif stresser == 'gemini':
            cmd_params = get_gemini_cmd_params(cmd)
        elif stresser == 'cdcreader':
            cmd_params = get_cdcreader_cmd_params(cmd)
        else:
            cmd_params = None
            self.log.warning("Unknown stresser: %s" % stresser)
        if cmd_params:
            if aggregate:
                self._stats['test_details'][section].append(cmd_params)
            else:
                self._stats['test_details'][section].update(cmd_params)
            self.update(dict(test_details=self._stats['test_details']))

    def _calc_stats(self, ps_results):
        try:
            if not ps_results or len(ps_results) <= 3:
                self.log.error("Not enough data from Prometheus: %s" % ps_results)
                return {}
            stat = {}
            ops_per_sec = [float(val) for _, val in ps_results if val.lower() != "nan"]  # float("nan") is not a number
            stat["max"] = max(ops_per_sec)
            # filter all values that are less than 1% of max
            ops_filtered = [x for x in ops_per_sec if x >= stat["max"] * 0.01]
            stat["min"] = min(ops_filtered)
            stat["avg"] = float(sum(ops_filtered)) / len(ops_filtered)
            stat["stdev"] = stddev(ops_filtered)
            self.log.debug("Stats: %s", stat)
            return stat
        except Exception as ex:  # pylint: disable=broad-except
            self.log.error("Exception when calculating PrometheusDB stats: %s" % ex)
            return {}

    @retrying(n=5, sleep_time=0, message="Retrying on getting prometheus stats")
    def get_prometheus_stats(self, alternator=False, scrap_metrics_step=None):
        self.log.info("Calculating throughput stats from PrometheusDB...")
        prometheus_db_stats = PrometheusDBStats(host=self.monitors.nodes[0].external_address,
                                                alternator=alternator)
        offset = 120  # 2 minutes offset
        start = int(self._stats["test_details"]["start_time"] + offset)
        end = int(time.time() - offset)
        prometheus_stats = {}
        for stat in self.PROMETHEUS_STATS:
            stat_calc_func = getattr(prometheus_db_stats, "get_" + stat)
            prometheus_stats[stat] = self._calc_stats(ps_results=stat_calc_func(start_time=start, end_time=end,
                                                                                scrap_metrics_step=scrap_metrics_step))
        self._stats['results'].update(prometheus_stats)
        return prometheus_stats

    def update_hdrhistograms(self, histogram_name, histogram_data):
        if "histograms" not in self._stats['results']:
            self._stats['results']['histograms'] = {}
        if histogram_name not in self._stats['results']['histograms']:
            self._stats['results']['histograms'][histogram_name] = [histogram_data]
        else:
            self._stats['results']['histograms'][histogram_name].extend(histogram_data)

        self.update(dict(results=self._stats['results']))

    def update_stress_results(self, results, calculate_stats=True):
        if 'stats' not in self._stats['results']:
            self._stats['results']['stats'] = results
        else:
            self._stats['results']['stats'].extend(results)
        if calculate_stats:
            self.calculate_stats_average()
            self.calculate_stats_total()
        self.update(dict(results=self._stats['results']))

    def _convert_stat(self, stat, stress_result):
        if stat not in stress_result or stress_result[stat] == 'NaN':
            self.log.warning("Stress stat not found: '%s'", stat)
            return 0
        try:
            return float(stress_result[stat])
        except Exception as details:  # pylint: disable=broad-except
            self.log.warning("Error in conversion of '%s' for stat '%s': '%s'"
                             "Discarding stat." % (stress_result[stat], stat, details))
        return 0

    def _calc_stat_total(self, stat):
        total = 0
        for stress_result in self._stats['results']['stats']:
            stat_val = self._convert_stat(stat=stat, stress_result=stress_result)
            if not stat_val:
                return 0  # discarding all stat results completely if one of the results is bad
            total += stat_val
        return total

    def calculate_stats_average(self):
        # calculate average stats
        average_stats = {}
        for stat in self.STRESS_STATS:
            average_stats[stat] = ''  # default
            total = self._calc_stat_total(stat=stat)
            if total:
                average_stats[stat] = round(total / len(self._stats['results']['stats']), 1)
        self._stats['results']['stats_average'] = average_stats

    def calculate_stats_total(self):
        total_stats = {}
        for stat in self.STRESS_STATS_TOTAL:
            total_stats[stat] = ''  # default
            total = self._calc_stat_total(stat=stat)
            if total:
                total_stats[stat] = total
        self._stats['results']['stats_total'] = total_stats

    # pylint: disable=too-many-arguments,too-many-locals
    def update_test_details(self, errors=None, coredumps=None, scylla_conf=False, extra_stats=None, alternator=False,
                            scrap_metrics_step=None):
        if not self.create_stats:
            return

        if not self._stats:
            self.log.error("Stats was not initialized. Could be error during TestConfig")
            return

        if extra_stats is None:
            extra_stats = {}

        update_data = {**extra_stats}
        update_data["test_details"] = test_details = self._stats.setdefault("test_details", {})
        update_data["setup_details"] = setup_details = self._stats.setdefault("setup_details", {})
        update_data["status"] = self._stats["status"] = self.status

        if errors:
            update_data["errors"] = errors

        if coredumps:
            update_data["coredumps"] = coredumps

        for key, value in extra_stats.items():
            self.log.debug("extra stats -- k: %s v: %s", key, value)
        self._stats["results"].update(extra_stats)

        test_details["time_completed"] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        if self.monitors and self.monitors.nodes:
            if self.params.get("store_perf_results"):
                update_data["results"] = self.get_prometheus_stats(alternator=alternator,
                                                                   scrap_metrics_step=scrap_metrics_step)
            grafana_dataset = self.monitors.get_grafana_screenshot_and_snapshot(test_details["start_time"])
            test_details.update({"grafana_screenshots": grafana_dataset.get("screenshots", []),
                                 "grafana_snapshots": grafana_dataset.get("snapshots", []),
                                 "grafana_annotations": self.monitors.upload_annotations_to_s3(),
                                 "prometheus_data": self.monitors.download_monitor_data(), })

        if self.db_cluster and self.db_cluster.nodes:
            node = self.db_cluster.nodes[0]

            def output(cmd):
                result = node.remoter.sudo(cmd, ignore_status=True, verbose=True)
                if result.ok:
                    return result.stdout.strip()
                self.log.error("Failed to run `%s' on %s", cmd, node)
                return "<< failed to get >>"

            setup_details["db_cluster_node_details"] = {
                "cpu_model": output("awk -F: '/^model name/{print $2; exit}' /proc/cpuinfo"),
                "cpu_mhz": output("awk -F: '/^cpu MHz/{print $2; exit}' /proc/cpuinfo"),
                "cache_size": output("awk -F: '/^cache size/{print $2; exit}' /proc/cpuinfo"),
                "flags": output("awk -F: '/^flags/{print $2; exit}' /proc/cpuinfo"),
                "sys_info": output("uname -a"),
            }
            if scylla_conf and "scylla_args" not in setup_details:
                scylla_server_conf = f"/etc/{'sysconfig' if node.distro.is_rhel_like else 'default'}/scylla-server"
                setup_details.update({"scylla_args": output(f"grep ^SCYLLA_ARGS {scylla_server_conf}"), })
                if node.has_seastar_conf:
                    setup_details.update({"seastar.conf": output("cat /etc/scylla.d/seastar.conf"), })
                else:
                    setup_details.update({"io_conf": output("grep -v ^# /etc/scylla.d/io.conf"),
                                      "cpuset_conf": output("grep -v ^# /etc/scylla.d/cpuset.conf"), })
            sysctl_excludes = (
                'net.bridge', 'net.ipv', 'net.netfilter', 'kernel.sched_', 'sunrpc',
            )
            for node in self.db_cluster.nodes:
                result = node.get_sysctl_properties()
                if result:
                    # NOTE: exclude huge sets of not needed data to avoid following errors in ES:
                    #       elasticsearch.exceptions.RequestError: RequestError(
                    #           400,
                    #           'illegal_argument_exception',
                    #           'Limit of total fields [1000] has been exceeded')
                    setup_details["sysctl_output"].append(
                        {key: value for key, value in result.items()
                         if not any(tag in key for tag in sysctl_excludes)})

        self.update(update_data)

    def get_doc_data(self, key) -> Optional[dict]:
        if self.create_stats and self._test_index and self._test_id:
            if not self.elasticsearch:
                LOGGER.error("Failed to get test stats: ES connection is not created (doc_id=%s)", self._test_id)
                return None
            try:
                result = self.elasticsearch.get_doc(
                    index=self._test_index,
                    doc_type=self._es_doc_type,
                    doc_id=self._test_id,
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("Failed to get test stats (doc_id=%s)", self._test_id)
                ElasticsearchEvent(doc_id=self._test_id, error=str(exc)).publish()
            else:
                return result["_source"].get(key)
        return None

    def get_test_start_time(self):
        return self._stats['test_details'].get('start_time', None)
