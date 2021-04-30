# pylint: disable=too-many-lines

import os
import math
import pprint
import logging
import collections

from datetime import datetime
from sortedcontainers import SortedDict

import jinja2

from sdcm.es import ES
from sdcm.db_stats import TestStatsMixin
from sdcm.send_email import Email
from sdcm.sct_events.file_logger import get_events_grouped_by_category, get_logger_event_summary
from sdcm.utils.es_queries import query_filter, QueryFilter, PerformanceFilterYCSB, PerformanceFilterScyllaBench, \
    PerformanceFilterCS, CDCQueryFilterCS
from test_lib.utils import MagicList, get_data_by_path
from .test import TestResultClass


LOGGER = logging.getLogger(__name__)
PP = pprint.PrettyPrinter(indent=2)


class BaseResultsAnalyzer:  # pylint: disable=too-many-instance-attributes
    def __init__(self, es_index, es_doc_type, send_email=False, email_recipients=(),
                 email_template_fp="", query_limit=1000, logger=None, events=None):  # pylint: disable=too-many-arguments
        self._es = ES()
        self._conf = self._es._conf  # pylint: disable=protected-access
        self._es_index = es_index
        self._es_doc_type = es_doc_type
        self._limit = query_limit
        self._send_email = send_email
        self._email_recipients = email_recipients
        self._email_template_fp = email_template_fp
        self.log = logger if logger else LOGGER
        self._events = events

    def get_all(self):
        """
        Get all the test results in json format
        """
        return self._es.search(index=self._es_index, size=self._limit)  # pylint: disable=unexpected-keyword-arg

    def get_test_by_id(self, test_id):
        """
        Get test results by test id
        :param test_id: test id created by performance test
        :return: test results in json format
        """
        if not self._es.exists(index=self._es_index, doc_type=self._es_doc_type, id=test_id):
            self.log.error('Test results not found: {}'.format(test_id))
            return None
        return self._es.get(index=self._es_index, doc_type=self._es_doc_type, id=test_id)

    @staticmethod
    def _get_grafana_snapshot(test_doc):
        grafana_snapshots = test_doc['_source']['test_details'].get('grafana_snapshots')
        if grafana_snapshots and isinstance(grafana_snapshots, list):
            return grafana_snapshots
        elif grafana_snapshots and isinstance(grafana_snapshots, str):
            return [grafana_snapshots]
        else:
            return []

    @staticmethod
    def _get_grafana_screenshot(test_doc):
        grafana_screenshots = test_doc['_source']['test_details'].get('grafana_screenshot')
        if not grafana_screenshots:
            grafana_screenshots = test_doc['_source']['test_details'].get('grafana_screenshots')

        if grafana_screenshots and isinstance(grafana_screenshots, list):
            return grafana_screenshots
        elif grafana_screenshots and isinstance(grafana_screenshots, str):
            return [grafana_screenshots]
        else:
            return []

    @staticmethod
    def _get_setup_details(test_doc, is_gce):
        setup_details = {'cluster_backend': test_doc['_source']['setup_details'].get('cluster_backend')}
        if setup_details['cluster_backend'] == "aws":
            setup_details['ami_id_db_scylla'] = test_doc['_source']['setup_details']['ami_id_db_scylla']
            setup_details['region_name'] = test_doc['_source']['setup_details']['region_name']
        for setup_param in QueryFilter(test_doc, is_gce).setup_instance_params():
            setup_details.update(
                [(setup_param.replace('gce_', ''), test_doc['_source']['setup_details'].get(setup_param))])
        return setup_details

    def _test_version(self, test_doc):
        if test_doc['_source'].get('versions'):
            for value in ('scylla-server', 'scylla-enterprise-server'):
                key = test_doc['_source']['versions'].get(value)
                if key:
                    return key

        self.log.error('Scylla version is not found for test %s', test_doc['_id'])
        return None

    def get_events(self):
        last_events = dict()
        events_summary = dict()

        if not self._events:
            return [last_events, events_summary]

        last_events['CRITICAL'] = self._events.get("CRITICAL", [])
        events_summary['CRITICAL'] = len(last_events['CRITICAL'])

        last_events['ERROR'] = self._events.get("ERROR", [])
        events_summary['ERROR'] = len(last_events['ERROR'])

        last_events['WARNING'] = self._events.get("WARNING", [])
        events_summary['WARNING'] = len(last_events['WARNING'])
        return [last_events, events_summary]

    def render_to_html(self, results, html_file_path="", template=None):
        """
        Render analysis results to html template
        :param results: results dictionary
        :param html_file_path: Boolean, whether to save html file on disk
        :param template: If not None, define the template to use, default to `self._email_template_fp`
        :return: html string
        """
        email_template_fp = template if template else self._email_template_fp
        self.log.info("Rendering results to html using '%s' template...", email_template_fp)
        loader = jinja2.FileSystemLoader(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../report_templates'))
        print(os.path.dirname(os.path.abspath(__file__)))
        env = jinja2.Environment(loader=loader, autoescape=True, extensions=[
                                 'jinja2.ext.loopcontrols', 'jinja2.ext.do'])
        template = env.get_template(email_template_fp)
        html = template.render(results)
        self.log.info("Results has been rendered to html")
        if html_file_path:
            with open(html_file_path, "w") as html_file:
                html_file.write(html)
            self.log.info("HTML report saved to '%s'.", html_file_path)
        return html

    def send_email(self, subject, content, html=True, files=()):
        if self._send_email and self._email_recipients:
            self.log.debug('Send email to {}'.format(self._email_recipients))
            email = Email()
            email.send(subject, content, html=html, recipients=self._email_recipients, files=files)
        else:
            self.log.warning("Won't send email (send_email: %s, recipients: %s)",
                             self._send_email, self._email_recipients)

    def gen_kibana_dashboard_url(self, dashboard_path=""):
        return "%s/%s" % (self._conf.get('kibana_url'), dashboard_path)


class LatencyDuringOperationsPerformanceAnalyzer(BaseResultsAnalyzer):
    """
    Get latency during operations performance analyzer
    """

    def __init__(self, es_index, es_doc_type, send_email, email_recipients, logger=None, events=None):   # pylint: disable=too-many-arguments
        super(LatencyDuringOperationsPerformanceAnalyzer, self).__init__(
            es_index=es_index,
            es_doc_type=es_doc_type,
            send_email=send_email,
            email_recipients=email_recipients,
            email_template_fp="results_latency_during_ops.html",
            logger=logger,
            events=events)

    def check_regression(self, test_id, data, is_gce=False):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        doc = self.get_test_by_id(test_id)
        full_test_name = doc["_source"]["test_details"]["test_name"]
        test_name = full_test_name.split('.')[-1]  # Example: longevity_test.py:LongevityTest.test_custom_time
        test_start_time = datetime.utcfromtimestamp(float(doc["_source"]["test_details"]["start_time"]))
        test_version_info = self._test_version(doc)
        test_version = '.'.join([val for val in test_version_info.values()])

        last_events, events_summary = self.get_events()

        subject = f'Performance Regression Compare Results (latency during operations) -' \
                  f' {test_name} - {test_version} - {str(test_start_time)}'
        results = dict(
            events_summary=events_summary,
            last_events=last_events,
            stats=data,
            test_name=full_test_name,
            test_start_time=str(test_start_time),
            test_id=doc['_source']['test_details'].get('test_id', doc["_id"]),
            test_version=test_version,
            setup_details=self._get_setup_details(doc, is_gce),
            grafana_snapshots=self._get_grafana_snapshot(doc),
            grafana_screenshots=self._get_grafana_screenshot(doc),
            job_url=doc['_source']['test_details'].get('job_url', ""),
        )
        html = self.render_to_html(results=results)
        self.send_email(subject=subject, content=html)


class SpecifiedStatsPerformanceAnalyzer(BaseResultsAnalyzer):
    """
    Get specified performance test results from elasticsearch DB and analyze it to find a regression
    """

    def __init__(self, es_index, es_doc_type, send_email, email_recipients, logger=None, events=None):   # pylint: disable=too-many-arguments
        super(SpecifiedStatsPerformanceAnalyzer, self).__init__(
            es_index=es_index,
            es_doc_type=es_doc_type,
            send_email=send_email,
            email_recipients=email_recipients,
            email_template_fp="",
            logger=logger,
            events=events)

    def _test_stats(self, test_doc):
        # check if stats exists
        if 'results' not in test_doc['_source']:
            self.log.error('Cannot find the field: results for test id: {}!'.format(test_doc['_id']))
            return None
        return test_doc['_source']['results']

    def check_regression(self, test_id, stats):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        """
        Get test results by id, filter similar results and calculate DB values for each version,
        then compare with max-allowed in the tested version (and report all the found versions).
        :param test_id: test id created by performance test
        :param is_gce: is gce instance
        """
        allowed_deviation = 1.20
        # get test res
        doc = self.get_test_by_id(test_id)
        if not doc:
            self.log.error('Cannot find test by id: {}!'.format(test_id))
            return False

        test_stats = self._test_stats(doc)
        if not test_stats:
            self.log.debug("Could not find test statistics, regression check is skipped")
            return False
        es_base_path = 'hits.hits'
        es_source_path = es_base_path + '._source'
        filter_path = ['.'.join([es_base_path, '_id']),
                       '.'.join([es_source_path, 'results', 'throughput']),
                       '.'.join([es_source_path, 'versions'])]

        for stat in stats.keys():  # Add all requested specific-stats to be retrieved from ES DB.
            stat_path = '.'.join([es_source_path, stat])
            filter_path.append(stat_path)

        tests_filtered = self._es.search(index=self._es_index, filter_path=filter_path, size=self._limit)
        self.log.debug("Filtered tests found are: {}".format(tests_filtered))

        if not tests_filtered:
            self.log.info('Cannot find tests with the same parameters as {}'.format(test_id))
            return False
        cur_test_version = None
        tested_params = stats.keys()
        group_by_version = dict()
        # repair_runtime result example:
        # { u'_id': u'20190303-105120-405065',
        #   u'_index': u'performanceregressionrowlevelrepairtest',
        #   u'_source': { u'coredumps': { },
        #                 u'errors': { },
        #                 u'nemesis': { },
        #                 u'repair_runtime': 11.847206830978394,
        #                 u'results': { u'latency_read_99': { },
        #                               u'latency_write_99': { },
        #                               u'repair_runtime': -1,
        #                               u'throughput': { }},
        #
        # # Find the average results for each version per tested param (stats)
        for tag_row in tests_filtered['hits']['hits']:
            if '_source' not in tag_row:  # non-valid record?
                self.log.error('Skip non-valid test: %s', tag_row['_id'])
                continue
            if not tag_row['_source']['versions'] or 'scylla-server' not in tag_row['_source']['versions']:
                continue
            version_info = tag_row['_source']['versions']['scylla-server']
            version = version_info['version']
            self.log.debug("version_info={} version={}".format(version_info, version))

            if tag_row['_id'] == test_id:  # save the current test values
                cur_test_version = version
                continue

            if not version:
                continue

            # group_by_version example:
            #   { '2.3.0':
            #       { 'repair_runtime': [12.345, 13.031]},
            #     '3.1.0': { 'repair_runtime': [5.341]}
            #   }
            if version not in group_by_version:
                group_by_version[version] = dict()
                for param in tested_params:
                    if param in tag_row['_source']:
                        group_by_version[version][param] = [tag_row['_source'][param]]
            else:
                for param in tested_params:
                    if param in tag_row['_source']:
                        if param not in group_by_version[version]:
                            group_by_version[version][param] = [tag_row['_source'][param]]
                        else:
                            group_by_version[version][param].append(tag_row['_source'][param])

            self.log.debug("group_by_version={}".format(group_by_version))

        if not cur_test_version:
            raise ValueError("Could not retrieve current test details from database")
        for param in tested_params:
            if param in group_by_version[cur_test_version]:
                cur_test_param_result = stats[param]
                list_param_stats = group_by_version[cur_test_version][param]
                param_avg = sum(list_param_stats) / float(len(list_param_stats))
                deviation_limit = param_avg * allowed_deviation
                self.log.info(
                    "Performance result for: {} is: {}. (average statistics deviation limit is: {}".format(param,
                                                                                                           cur_test_param_result,
                                                                                                           deviation_limit))
                for version in group_by_version:
                    if param in group_by_version[version]:
                        list_param_results = group_by_version[version][param]
                        version_avg = sum(list_param_results) / float(len(list_param_results))
                        self.log.info("Performance average of {} results for: {} on version: {} is: {}".format(
                            len(list_param_results), param, version, version_avg))
                assert float(
                    cur_test_param_result) < deviation_limit, "Current test performance for: {} exceeds allowed deviation ({})".format(
                    param, deviation_limit)
        return True


class PerformanceResultsAnalyzer(BaseResultsAnalyzer):
    """
    Get performance test results from elasticsearch DB and analyze it to find a regression
    """

    PARAMS = TestStatsMixin.STRESS_STATS

    def __init__(self, es_index, es_doc_type, send_email, email_recipients, logger=None, events=None):  # pylint: disable=too-many-arguments
        super(PerformanceResultsAnalyzer, self).__init__(
            es_index=es_index,
            es_doc_type=es_doc_type,
            send_email=send_email,
            email_recipients=email_recipients,
            email_template_fp="results_performance.html",
            logger=logger,
            events=events
        )

    @staticmethod
    def _remove_non_stat_keys(stats):
        for non_stat_key in ['loader_idx', 'cpu_idx', 'keyspace_idx']:
            if non_stat_key in stats:
                del stats[non_stat_key]
        return stats

    def _test_stats(self, test_doc):
        # check if stats exists
        if 'results' not in test_doc['_source'] or 'stats_average' not in test_doc['_source']['results'] or \
                'stats_total' not in test_doc['_source']['results']:
            self.log.error('Cannot find one of the fields: results, results.stats_average, '
                           'results.stats_total for test id: {}!'.format(test_doc['_id']))
            return None
        stats_average = self._remove_non_stat_keys(test_doc['_source']['results']['stats_average'])
        stats_total = test_doc['_source']['results']['stats_total']
        if not stats_average or not stats_total or any([stats_average[k] == '' for k in self.PARAMS]):
            self.log.error('Cannot find average/total results for test: {}!'.format(test_doc['_id']))
            return None
        # replace average by total value for op rate
        stats_average['op rate'] = stats_total['op rate']
        return stats_average

    def _get_best_value(self, key, val1, val2):
        if key == self.PARAMS[0]:  # op rate
            return val1 if val1 > val2 else val2
        return val1 if val2 == 0 or val1 < val2 else val2  # latency

    @staticmethod
    def _query_filter(test_doc, is_gce):
        if test_doc['_source']['test_details'].get('scylla-bench'):
            return PerformanceFilterScyllaBench(test_doc, is_gce)()
        elif test_doc['_source']['test_details'].get('ycsb'):
            return PerformanceFilterYCSB(test_doc, is_gce)()
        elif "cdc" in test_doc['_source']['test_details'].get('sub_type'):
            return CDCQueryFilterCS(test_doc, is_gce)()
        else:
            return PerformanceFilterCS(test_doc, is_gce)()

    def cmp(self, src, dst, version_dst, best_test_id):
        """
        Compare current test results with the best results
        :param src: current test results
        :param dst: previous best test results
        :param version_dst: scylla server version to compare with
        :param best_test_id: the best results test id(for each parameter)
        :return: dictionary with compare calculation results
        """
        cmp_res = dict(version_dst=version_dst, res=dict())
        for param in self.PARAMS:
            param_key_name = param.replace(' ', '_')
            status = 'Progress'
            try:
                delta = src[param] - dst[param]
                change_perc = int(math.fabs(delta) * 100 / dst[param])
                best_id = best_test_id[param]
                if (param.startswith('latency') and delta > 0) or (param == 'op rate' and delta < 0):
                    status = 'Regression'
                if change_perc == 0:
                    status = "Difference"
                cmp_res['res'][param_key_name] = dict(percent='{}%'.format(change_perc),
                                                      val=src[param],
                                                      best_val=dst[param],
                                                      best_id=best_id,
                                                      status=status)
            except TypeError:
                self.log.exception('Failed to compare {} results: {} vs {}, version {}'.format(
                    param, src[param], dst[param], version_dst))
        return cmp_res

    def check_regression(self, test_id, is_gce=False, email_subject_postfix=None, use_wide_query=False, lastyear=False):
        """
        Get test results by id, filter similar results and calculate max values for each version,
        then compare with max in the test version and all the found versions.
        Save the analysis in log and send by email.
        :param test_id: test id created by performance test
        :param is_gce: is gce instance
        :return: True/False
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements

        # get test res
        doc = self.get_test_by_id(test_id)
        if not doc:
            self.log.error('Cannot find test by id: {}!'.format(test_id))
            return False
        self.log.debug(PP.pformat(doc))

        test_stats = self._test_stats(doc)
        if not test_stats:
            return False

        # filter tests
        query = query_filter(doc, is_gce, use_wide_query, lastyear)
        if not query:
            return False
        self.log.debug("Query to ES: %s", query)
        filter_path = ['hits.hits._id',
                       'hits.hits._source.results.stats_average',
                       'hits.hits._source.results.stats_total',
                       'hits.hits._source.results.throughput',
                       'hits.hits._source.versions']
        tests_filtered = self._es.search(index=self._es_index, q=query, filter_path=filter_path,  # pylint: disable=unexpected-keyword-arg
                                         size=self._limit, request_timeout=30)

        if not tests_filtered:
            self.log.info('Cannot find tests with the same parameters as {}'.format(test_id))
            return False
        # get the best res for all versions of this job
        group_by_version = dict()
        # Example:
        # group_by_version = {
        #     "2.3.rc1": {
        #         "tests": {  # SortedDict(),
        #             "20180726": {
        #                 "latency 99th percentile": 10.3,
        #                 "op rate": 15034.3
        #                 #...
        #             }
        #         },
        #
        #         "stats_best": {
        #             "op rate": 0,
        #             "latency mean": 0,
        #         },
        #         "best_test_id": {
        #             "op rate": {"commit": 9b4a0a287", "date": "2020.02.02"},
        #             "latency mean": {"commit": 9b4a0a287", "date": "2020.02.02"},
        #
        #         }
        #     }
        # }
        # Find best results for each version
        for row in tests_filtered['hits']['hits']:
            if row['_id'] == test_id:  # filter the current test
                continue
            if '_source' not in row:  # non-valid record?
                self.log.error('Skip non-valid test: %s', row['_id'])
                continue
            version_info = self._test_version(row)
            if not version_info:
                continue
            version = version_info['version']
            curr_test_stats = self._test_stats(row)
            if not curr_test_stats:
                continue

            formated_version_date = datetime.strptime(version_info['date'], "%Y%m%d").strftime("%Y-%m-%d")
            version_info_data = {"commit": version_info['commit_id'], "date": formated_version_date}

            if version not in group_by_version:
                group_by_version[version] = dict(tests=SortedDict(), stats_best=dict(), best_test_id=dict())
                group_by_version[version]['stats_best'] = {k: 0 for k in self.PARAMS}
                group_by_version[version]['best_test_id'] = {
                    k: version_info_data for k in self.PARAMS}
            group_by_version[version]['tests'][version_info['date']] = {
                "test_stats": curr_test_stats,
                "version": {k: version_info_data for k in self.PARAMS}
            }
            old_best = group_by_version[version]['stats_best']
            group_by_version[version]['stats_best'] =\
                {k: self._get_best_value(k, curr_test_stats[k], old_best[k])
                 for k in self.PARAMS if k in curr_test_stats and k in old_best}
            # replace best test id if best value changed
            for k in self.PARAMS:
                if k in curr_test_stats and k in old_best and\
                        group_by_version[version]['stats_best'][k] == curr_test_stats[k]:
                    group_by_version[version]['best_test_id'][k] = version_info_data
        res_list = list()
        # compare with the best in the test version and all the previous versions
        test_version_info = self._test_version(doc)
        test_version = test_version_info['version']

        for version in group_by_version:
            if version == test_version and not group_by_version[test_version]['tests']:
                self.log.info('No previous tests in the current version {} to compare'.format(test_version))
                continue
            cmp_res = self.cmp(test_stats,
                               group_by_version[version]['stats_best'],
                               version,
                               group_by_version[version]['best_test_id'])
            _, latest_version_test = group_by_version[version]["tests"].peekitem(index=-1)
            latest_res = self.cmp(test_stats,
                                  latest_version_test["test_stats"],
                                  version,
                                  latest_version_test["version"])
            res_list.append({"best": cmp_res, "last": latest_res})
        if not res_list:
            self.log.info('No test results to compare with')
            return False
        # send results by email
        full_test_name = doc["_source"]["test_details"]["test_name"]
        test_start_time = datetime.utcfromtimestamp(float(doc["_source"]["test_details"]["start_time"]))
        cassandra_stress = doc['_source']['test_details'].get('cassandra-stress')
        ycsb = doc['_source']['test_details'].get('ycsb')
        dashboard_path = "app/kibana#/dashboard/03414b70-0e89-11e9-a976-2fe0f5890cd0?_g=()"

        last_events, events_summary = self.get_events()

        ebs = doc["_source"]["setup_details"].get("data_device")
        ebs_type = doc["_source"]["setup_details"].get("aws_ebs_volume_type") if ebs == "attached" else None
        results = dict(test_name=full_test_name,
                       test_start_time=str(test_start_time),
                       test_version=test_version_info,
                       res_list=res_list,
                       setup_details=self._get_setup_details(doc, is_gce),
                       prometheus_stats={stat: doc["_source"]["results"].get(
                           stat, {}) for stat in TestStatsMixin.PROMETHEUS_STATS},
                       prometheus_stats_units=TestStatsMixin.PROMETHEUS_STATS_UNITS,
                       grafana_snapshots=self._get_grafana_snapshot(doc),
                       grafana_screenshots=self._get_grafana_screenshot(doc),
                       cs_raw_cmd=cassandra_stress.get('raw_cmd', "") if cassandra_stress else "",
                       ycsb_raw_cmd=ycsb.get('raw_cmd', "") if ycsb else "",
                       job_url=doc['_source']['test_details'].get('job_url', ""),
                       kibana_url=self.gen_kibana_dashboard_url(dashboard_path),
                       events_summary=events_summary,
                       last_events=last_events)
        self.log.debug('Regression analysis:')
        self.log.debug(PP.pformat(results))
        test_name = full_test_name.split('.')[-1]  # Example: longevity_test.py:LongevityTest.test_custom_time
        subject = f'Performance Regression Compare Results - {test_name} - {test_version} - {str(test_start_time)}'
        if ycsb:
            subject = f'(Alternator) Performance Regression - {test_name} - {test_version} - {str(test_start_time)}'
        if ebs:
            subject = f'{subject} (ebs volume type {ebs_type})'
        if email_subject_postfix:
            subject = f'{subject} {email_subject_postfix}'
        html = self.render_to_html(results)
        self.send_email(subject, html)

        return True

    def check_regression_with_subtest_baseline(self, test_id, base_test_id, subtest_baseline, is_gce=False):
        """
        Get test results by id, filter similar results and calculate max values for each version,
        then compare with max in the test version and all the found versions.
        Save the analysis in log and send by email.
        :param test_id: test id created by performance test
        :param base_test_id: current test id
        :param subtest_baseline: which of the subtest is the baseline
        :param is_gce: is gce instance
        :return: True/False
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements

        doc = self.get_test_by_id(test_id)
        if not doc:
            self.log.error('Cannot find test by id: {}!'.format(test_id))
            return False
        self.log.debug(PP.pformat(doc))

        test_stats = self._test_stats(doc)
        if not test_stats:
            return False

        # filter tests
        query = self._query_filter(doc, is_gce)
        self.log.debug(query)
        if not query:
            return False
        self.log.debug("Query to ES: %s", query)
        filter_path = ['hits.hits._id',
                       'hits.hits._source.results.stats_average',
                       'hits.hits._source.results.stats_total',
                       'hits.hits._source.results.throughput',
                       'hits.hits._source.results',
                       'hits.hits._source.versions',
                       'hits.hits._source.test_details']
        tests_filtered = self._es.search(index=self._es_index, q=query, filter_path=filter_path,
                                         # pylint: disable=unexpected-keyword-arg
                                         size=self._limit)

        if not tests_filtered:
            self.log.info('Cannot find tests with the same parameters as {}'.format(test_id))
            return False
        # get the best res for all versions of this job
        group_by_version_sub_type = SortedDict()
        # Example:
        # group_by_type = {
        #     "version": {
        #           "sub_type": {
        #               "tests": {  # SortedDict(),
        #                   "20180726": {
        #                       "latency 99th percentile": 10.3,
        #                       "op rate": 15034.3
        #                       #...
        #                   }
        #               },
        #
        #               "stats_best": {
        #                   "op rate": 0,
        #                   "latency mean": 0,
        #               },
        #               "best_test_id": {
        #                   "op rate": "9b4a0a287",
        #                   "latency mean": "9b4a0a287",
        #
        #               }
        #           }
        #      }
        # }
        # Find best results for each version

        current_tests = dict()
        grafana_snapshots = dict()
        grafana_screenshots = dict()
        for row in tests_filtered['hits']['hits']:
            if '_source' not in row:  # non-valid record?
                self.log.error('Skip non-valid test: %s', row['_id'])
                continue
            version_info = self._test_version(row)
            version = version_info['version']
            version_info['date'] = datetime.strptime(version_info['date'], "%Y%m%d").strftime("%Y-%m-%d")
            if not version:
                self.log.error('Skip with wrong version %s', row['_id'])
                continue
            if "results" not in row["_source"]:
                self.log.error('Skip with no results %s', row['_id'])
                continue
            sub_type = row["_source"]['test_details'].get('sub_type')
            curr_test_stats = self._test_stats(row)
            if not curr_test_stats or not sub_type:
                self.log.error('Skip with no test stats %s', row['_id'])
                continue
            if base_test_id in row["_id"] and sub_type not in current_tests:
                current_tests[sub_type] = dict()
                current_tests[sub_type]['stats'] = curr_test_stats
                current_tests[sub_type]['version'] = version_info
                current_tests[sub_type]['best_test_id'] = {
                    k: f"#{version_info['commit_id']}, {version_info['date']}" for k in self.PARAMS}
                current_tests[sub_type]['results'] = row['_source']['results']
                grafana_snapshots[sub_type] = self._get_grafana_snapshot(row)
                grafana_screenshots[sub_type] = self._get_grafana_screenshot(row)

                self.log.info('Added current test results %s. Check next', row['_id'])
                continue
            if version not in group_by_version_sub_type:
                group_by_version_sub_type[version] = dict()

            if sub_type not in group_by_version_sub_type[version]:
                group_by_version_sub_type[version][sub_type] = dict(
                    tests=SortedDict(), stats_best=dict(), best_test_id=dict())
                group_by_version_sub_type[version][sub_type]['stats_best'] = {k: 0 for k in self.PARAMS}
                group_by_version_sub_type[version][sub_type]['best_test_id'] = {
                    k: f"#{version_info['commit_id']}, {version_info['date']}" for k in self.PARAMS}

            group_by_version_sub_type[version][sub_type]['tests'][version_info['date']] = curr_test_stats
            old_best = group_by_version_sub_type[version][sub_type]['stats_best']
            group_by_version_sub_type[version][sub_type]['stats_best'] = \
                {k: self._get_best_value(k, curr_test_stats[k], old_best[k])
                 for k in self.PARAMS if k in curr_test_stats and k in old_best}
            # replace best test id if best value changed
            for k in self.PARAMS:
                if k in curr_test_stats and k in old_best and \
                        group_by_version_sub_type[version][sub_type]['stats_best'][k] == curr_test_stats[k]:
                    group_by_version_sub_type[version][sub_type]['best_test_id'][
                        k] = f"#{version_info['commit_id']}, {version_info['date']}"

        current_res_list = list()
        versions_res_list = list()

        test_version_info = self._test_version(doc)
        test_version_info['date'] = datetime.strptime(test_version_info['date'], "%Y%m%d").strftime("%Y-%m-%d")
        test_version = test_version_info['version']
        base_line = current_tests.get(subtest_baseline)
        for sub_type in current_tests:
            if not current_tests[sub_type] or sub_type == subtest_baseline:
                self.log.info(f'No tests with {subtest_baseline} in the current run {test_version} to compare')
                continue
            cmp_res = self.cmp(current_tests[sub_type]['stats'],
                               base_line['stats'],
                               sub_type,
                               current_tests[sub_type]['best_test_id'])
            current_res_list.append(cmp_res)

        if not current_res_list:
            self.log.info('No test results to compare with')
            return False
        current_prometheus_stats = SortedDict()
        for sub_type in current_tests:
            current_prometheus_stats[sub_type] = {stat: current_tests[sub_type]["results"].get(
                stat, {}) for stat in TestStatsMixin.PROMETHEUS_STATS}

        for version in group_by_version_sub_type:
            cmp_res = dict()
            for sub_type in group_by_version_sub_type[version]:
                if not group_by_version_sub_type[version][sub_type]['tests']:
                    self.log.info('No previous tests in the current version {} to compare'.format(test_version))
                    continue
                if sub_type not in current_tests:
                    continue
                cmp_res[sub_type] = self.cmp(group_by_version_sub_type[version][sub_type]['stats_best'],
                                             current_tests[sub_type]['stats'],
                                             version,
                                             group_by_version_sub_type[version][sub_type]['best_test_id'])
            versions_res_list.append({version: cmp_res})

        # send results by email
        full_test_name = doc["_source"]["test_details"]["test_name"]
        test_start_time = datetime.utcfromtimestamp(float(doc["_source"]["test_details"]["start_time"]))
        cassandra_stress = doc['_source']['test_details'].get('cassandra-stress')
        ycsb = doc['_source']['test_details'].get('ycsb')
        dashboard_path = "app/kibana#/dashboard/03414b70-0e89-11e9-a976-2fe0f5890cd0?_g=()"
        last_events, events_summary = self.get_events()
        results = dict(test_name=full_test_name,
                       test_start_time=str(test_start_time),
                       test_version=test_version_info,
                       base_line=base_line,
                       res_list=current_res_list,
                       ver_res_list=versions_res_list,
                       setup_details=self._get_setup_details(doc, is_gce),
                       prometheus_stats=current_prometheus_stats,
                       prometheus_stats_units=TestStatsMixin.PROMETHEUS_STATS_UNITS,
                       grafana_snapshots=grafana_snapshots,
                       grafana_screenshots=grafana_screenshots,
                       cs_raw_cmd=cassandra_stress.get('raw_cmd', "") if cassandra_stress else "",
                       ycsb_raw_cmd=ycsb.get('raw_cmd', "") if ycsb else "",
                       job_url=doc['_source']['test_details'].get('job_url', ""),
                       kibana_url=self.gen_kibana_dashboard_url(dashboard_path),
                       baseline_type=subtest_baseline,
                       events_summary=events_summary,
                       last_events=last_events)
        self.log.debug('Regression analysis:')
        self.log.debug(PP.pformat(results))
        test_name = full_test_name.split('.', 1)[1]  # Example: longevity_test.py:LongevityTest.test_custom_time
        subject = f'Performance Regression Compare Results - {test_name} - {test_version} - {str(test_start_time)}'
        if ycsb:
            subject = f'(Alternator) Performance Regression - {test_name} - {test_version} - {str(test_start_time)}'
        html = self.render_to_html(results, template='results_performance_baseline.html')
        self.send_email(subject, html)

        return True

    def get_test_instance_by_id(self, test_id):
        rp_main_test = TestResultClass.get_by_test_id(test_id, self._es_index)
        if rp_main_test:
            rp_main_test = rp_main_test[0]
        if not rp_main_test or not rp_main_test.is_valid():
            self.log.error('Cannot find main_test by id: {}!'.format(test_id))
            return None
        return rp_main_test

    @staticmethod
    def _add_remarks_to_test(test, remarks, tests_info):
        if isinstance(remarks, str):
            remarks = [remarks]
        if test not in tests_info:
            tests_info[test] = {'remarks': remarks}
        elif 'remarks' not in tests_info[test]:
            tests_info[test]['remarks'] = remarks
        else:
            tests_info[test]['remarks'].extend(remarks)

    @staticmethod
    def _add_best_for_info(test, subtest, metric_path, tests_info):
        subtest_name = subtest.subtest_name
        if test not in tests_info:
            tests_info[test] = {'best_for': {subtest_name: {metric_path: True}}}
        elif 'best_for' not in tests_info[test]:
            tests_info[test]['best_for'] = {subtest_name: {metric_path: True}}
        elif subtest_name not in tests_info[test]['best_for']:
            tests_info[test]['best_for'][subtest_name] = {metric_path: True}
        else:
            tests_info[test]['best_for'][subtest_name][metric_path] = True

    def _mark_best_tests(self, prior_subtests, metrics, tests_info, main_test_id):
        main_tests_by_id = MagicList(tests_info.keys()).group_by('test_id')
        for _, prior_tests in prior_subtests.items():
            prior_tests = MagicList(
                [prior_test for prior_test in prior_tests if prior_test.main_test_id != main_test_id])
            if not prior_tests:
                continue
            for metric_path in metrics:
                # Getting main test of the subtest that showed best score in this metric
                sorted_subtests = prior_tests.sort_by(f'{metric_path}.betterness')
                if not sorted_subtests:
                    continue
                best_subtest = sorted_subtests[-1]
                # Find subtests from any other tests
                self._add_best_for_info(
                    main_tests_by_id[best_subtest.main_test_id][0],
                    best_subtest,
                    metric_path,
                    tests_info
                )

    @staticmethod
    def _clean_up_not_marked_runs(prior_subtests, tests_info):
        main_tests_by_id = MagicList(tests_info.keys()).group_by('test_id')
        for _, prior_tests in prior_subtests.items():
            for prior_test in [prior_test for prior_test in prior_tests
                               if not main_tests_by_id.get(prior_test.main_test_id, False)]:
                prior_tests.remove(prior_test)

    def _find_versions_to_compare_with(self, test, prior_main_tests, tests_info):
        current_test_version_int = test.software.scylla_server_any.version.as_int
        current_test_version_major_int = test.software.scylla_server_any.version.major_as_int

        grouped_by_version = prior_main_tests.group_by(
            data_path='software.scylla_server_any.version.major_as_int',
            sort_keys=-1,
            group_values={
                'data_path': 'software.scylla_server_any.version.as_int',
                'sort_keys': -1
            }
        )

        for version_major_int, prior_tests_grouped in grouped_by_version.items():
            if version_major_int == current_test_version_major_int:
                for version_int, prior_tests_grouped2 in prior_tests_grouped.items():
                    if version_int == current_test_version_int:
                        if prior_tests_grouped2:
                            self._add_remarks_to_test(prior_tests_grouped2[0], 'Latest<br>(same version)', tests_info)
                    elif version_int < current_test_version_int:
                        if prior_tests_grouped2:
                            self._add_remarks_to_test(prior_tests_grouped2[0], [], tests_info)
            elif version_major_int != current_test_version_major_int:
                limit = 2
                for version_int, prior_tests_grouped2 in prior_tests_grouped.items():
                    if limit < 0:
                        break
                    if prior_tests_grouped2:
                        self._add_remarks_to_test(prior_tests_grouped2[0], [], tests_info)
                        limit -= 1

    @staticmethod
    def _get_prior_tests_for_subtests(subtests: list):
        output = collections.OrderedDict()
        for subtest in subtests:
            prior_tests = MagicList(
                [prior_sub_test for prior_sub_test in subtest.get_prior_tests()
                 if prior_sub_test.metrics and prior_sub_test.metrics.is_valid()
                 ])
            output[subtest] = prior_tests
        return output

    @staticmethod
    def _cleanup_not_complete_main_tests(prior_main_tests: list, prior_subtests: dict, expected_subtests_count):  # pylint: disable=too-many-branches
        is_test_complete = {}
        for subtest, prior_tests in prior_subtests.items():
            for prior_test_id, _ in prior_tests.group_by('main_test_id').items():
                if prior_test_id not in is_test_complete:
                    is_test_complete[prior_test_id] = [subtest.subtest_name]
                else:
                    is_test_complete[prior_test_id].append(subtest.subtest_name)

        for prior_test_id in list(is_test_complete.keys()):
            is_test_complete[prior_test_id] = len(is_test_complete[prior_test_id]) >= expected_subtests_count
        to_delete = []
        for num, prior_main_test in enumerate(prior_main_tests):
            if not is_test_complete.get(prior_main_test.test_id, False):
                to_delete.append(num)
        for num in sorted(to_delete, reverse=True):
            prior_main_tests.pop(num)

        for _, prior_tests in prior_subtests.items():
            to_delete = []
            for num, prior_subtest in enumerate(prior_tests):
                if not is_test_complete.get(prior_subtest.main_test_id, False):
                    to_delete.append(num)
            if to_delete:
                for num in sorted(to_delete, reverse=True):
                    prior_tests.pop(num)

    def check_regression_multi_baseline(
            self,
            test_id,
            subtests_info: list = None,
            metrics: list = None,
            email_subject: str = None,
    ):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """
        Build regression report for subtests.
        test_id: Main test id
        subtests_info: List of subtest information.
        metrics: List of metrics paths,
        email_subject: email subject
        Returns True if report was generated ans issued, otherwise - False

        subtest_info example: [{
            'name': 'INSERT IF NOT EXISTS',
            'subtest_name': 'lwt-insert-not-exists',
            'baseline': 'lwt-insert-std'
        }, {
            'name': 'INSERT',
            'subtest_name': 'lwt-insert-std',
        }]
        'subtest_name' - should be same as you provide to sub_type argument of test.create_test_stats when creating stats for the subcase
        'name' - Textual representation for reports
        'baseline' - A subtest_name of baseline test, set it to None or do not add it of subtest does not have baseline.

        Subtest that has baseline will be compared to baseline subtest of the same test and result will be showed next to subtest value.

        metrics example: ['metrics.cs_metrics.latency_mean','metrics.cs_metrics.throughput']

        Metric path is path with starting point of TestDataClass that points to metric value.
        Look at sdcm/utils/es_test.py:TestDataClass for more details
        """
        if metrics is None:
            metrics = [
                'metrics.cs_metrics.latency_mean',
                'metrics.cs_metrics.throughput'
            ]
        rp_main_test = self.get_test_instance_by_id(test_id)

        if not rp_main_test:
            self.log.error('Cannot find test with id: {}!'.format(test_id))
            return False

        if email_subject is None:
            email_subject = 'Performance Regression Compare Results - {test.test_name} - ' \
                            '{test.software.scylla_server_any.version.as_string}'.format(test=rp_main_test)
        else:
            email_subject = email_subject.format(test=rp_main_test)

        rp_main_tests_to_compare = collections.OrderedDict()
        rp_metric_info = {}

        self._add_remarks_to_test(rp_main_test, 'Current', rp_main_tests_to_compare)

        # Get prior main tests and clean it up from tests that has bad start_time or
        # ran on scylla version newer than this test
        prior_main_tests = MagicList([
            prior_test for prior_test in rp_main_test.get_prior_tests() if
            prior_test.is_valid() and prior_test.start_time and prior_test.start_time.is_valid() and
            prior_test.test_id != rp_main_test.test_id
        ])

        if not prior_main_tests:
            self.log.error('Cannot find prior runs for test with id: {}!'.format(test_id))
            return False

        # Get all subtests of the current main test and sort them by subtest name
        rp_subtests_of_current_test = MagicList([
            subtest for subtest in rp_main_test.get_subtests() if
            subtest.is_valid() and subtest.complete_time and subtest.complete_time.is_valid()
        ]).sort_by('subtest_name')

        if not rp_subtests_of_current_test:
            self.log.error('Cannot find subtests for test id: {}!'.format(test_id))
            return False

        if not subtests_info:
            subtests_info = [
                {
                    'name': test.subtest_name,
                    'subtest_name': test.subtest_name,
                } for test in rp_subtests_of_current_test]
        else:
            new_subtests_info = []
            for subtest_info in subtests_info:
                for subtest in rp_subtests_of_current_test:
                    if subtest_info['subtest_name'] == subtest.subtest_name:
                        new_subtests_info.append(subtest_info)
                        break
            subtests_info = new_subtests_info
            del new_subtests_info
        rp_subtests_info = {subtest_info['subtest_name']: subtest_info for subtest_info in subtests_info}
        rp_subtests_of_current_test = MagicList(
            [test for test in rp_subtests_of_current_test if test.subtest_name in rp_subtests_info]
        )

        prior_subtests = self._get_prior_tests_for_subtests(rp_subtests_of_current_test)
        self._cleanup_not_complete_main_tests(prior_main_tests, prior_subtests, len(rp_subtests_of_current_test) // 2)
        self._find_versions_to_compare_with(rp_main_test, prior_main_tests, rp_main_tests_to_compare)
        self._clean_up_not_marked_runs(prior_subtests, rp_main_tests_to_compare)
        self._mark_best_tests(prior_subtests, metrics, rp_main_tests_to_compare, rp_main_test.test_id)

        for metric_path in metrics:
            rp_metric_info[metric_path] = rp_main_test.get_metric_class(metric_path)

        rp_metrics_table = collections.OrderedDict()
        prior_tests_by_main_test_id = {subtest.subtest_name: prior_tests.group_by(
            'main_test_id') for subtest, prior_tests in prior_subtests.items()}

        subtests_groups = {}

        for subtest_info in rp_subtests_info.values():
            groups = subtest_info.get('groups', [''])
            for group in groups:
                if group not in subtests_groups:
                    subtests_groups[group] = MagicList()
                subtests_groups[group].append(subtest_info)

        tmp_subtest_by_name = rp_subtests_of_current_test.group_by('subtest_name')

        for group_name, group_substest_infos in subtests_groups.items():  # pylint: disable=too-many-nested-blocks
            rp_metrics_table[group_name] = rp_metrics_table_l1 = collections.OrderedDict()
            grouped_subtests = []
            subtest_info_grouped_by_baseline = group_substest_infos.group_by('baseline', '', sort_keys=1)
            no_baselines = subtest_info_grouped_by_baseline.get('', None)
            if no_baselines is None:
                no_baselines = MagicList()
            else:
                no_baselines = subtest_info_grouped_by_baseline.pop('')
            for baseline_subtest_name, subtest_infos in subtest_info_grouped_by_baseline.items():
                no_baselines.remove_where('subtest_name', baseline_subtest_name)
                if baseline_subtest_name in tmp_subtest_by_name and tmp_subtest_by_name[baseline_subtest_name]:
                    grouped_subtests.append(tmp_subtest_by_name[baseline_subtest_name][0])
                for subtest_info in MagicList(subtest_infos).sort_by('name'):
                    if subtest_info['subtest_name'] in tmp_subtest_by_name \
                            and tmp_subtest_by_name[subtest_info['subtest_name']]:
                        grouped_subtests.append(tmp_subtest_by_name[subtest_info['subtest_name']][0])
            for subtest_info in no_baselines:
                if subtest_info['subtest_name'] in tmp_subtest_by_name \
                        and tmp_subtest_by_name[subtest_info['subtest_name']]:
                    grouped_subtests.append(tmp_subtest_by_name[subtest_info['subtest_name']][0])
            del subtest_info_grouped_by_baseline

            for subtest in grouped_subtests:
                prior_tests = prior_subtests.get(subtest)
                baseline_subtest_name = rp_subtests_info.get(subtest.subtest_name, {}).get('baseline', None)
                baseline_prior_tests = None
                if baseline_subtest_name:
                    baseline_prior_tests = prior_tests_by_main_test_id.get(baseline_subtest_name, None)

                grouped_by_main_test_id = prior_tests.group_by('main_test_id')
                rp_metrics_table_l1[subtest.subtest_name] = rp_metrics_table_l2 = {
                    'total_metrics': 0,
                    'data': collections.OrderedDict()
                }
                for metric_path in metrics:
                    rp_metrics_table_l2['data'][metric_path] = rp_metrics_table_l3 = {
                        'relative': collections.OrderedDict(),
                        'absolute': collections.OrderedDict()
                    }
                    metric_bucket_main = rp_metrics_table_l3['absolute']
                    metric_bucket_baseline = rp_metrics_table_l3['relative']
                    number_of_metrics = 0
                    if not baseline_prior_tests:
                        del rp_metrics_table_l3['relative']
                    else:
                        holds_any_value = False
                        for main_test, _ in rp_main_tests_to_compare.items():
                            prior_subtest = grouped_by_main_test_id.get(main_test.test_id, [])
                            if not prior_subtest:
                                metric_bucket_baseline[main_test] = None
                                continue
                            baseline_subtest = baseline_prior_tests.get(main_test.test_id, [])
                            if not baseline_subtest:
                                metric_bucket_baseline[main_test] = None
                                continue
                            baseline_metric = get_data_by_path(baseline_subtest[0], metric_path)
                            metric_instance = get_data_by_path(prior_subtest[0], metric_path)
                            if baseline_metric is None or metric_instance is None:
                                metric_bucket_baseline[main_test] = None
                                continue
                            metric_bucket_baseline[main_test] = baseline_metric.rate(
                                metric_instance,
                                name=f'{metric_instance.name}<br>baseline diff'
                            )
                            holds_any_value = True
                        if not holds_any_value:
                            del rp_metrics_table_l3['relative']
                        else:
                            number_of_metrics += 1

                    holds_any_value = False
                    for main_test, _ in rp_main_tests_to_compare.items():
                        prior_subtest = grouped_by_main_test_id.get(main_test.test_id, [])
                        if not prior_subtest:
                            metric_bucket_main[main_test] = None
                            continue
                        metric_instance = get_data_by_path(prior_subtest[0], metric_path, None)
                        if metric_instance is None:
                            metric_bucket_main[main_test] = None
                            continue
                        metric_bucket_main[main_test] = metric_instance
                        holds_any_value = True
                    if not holds_any_value:
                        del rp_metrics_table_l2['data'][metric_path]
                        number_of_metrics = 0
                    else:
                        number_of_metrics += 1
                    rp_metrics_table_l2['total_metrics'] += number_of_metrics

        last_events, events_summary = self.get_events()
        results = dict(
            current_main_test=rp_main_test,
            results=rp_metrics_table,
            metric_info=rp_metric_info,
            tests_info=rp_main_tests_to_compare,
            subtests=rp_subtests_of_current_test,
            subtests_info=rp_subtests_info,
            events_summary=events_summary,
            last_events=last_events
        )
        if len(rp_metric_info) == 1:
            html = self.render_to_html(results, template='results_performance_multi_baseline_single_metric.html')
        else:
            html = self.render_to_html(results, template='results_performance_multi_baseline.html')
        self.send_email(email_subject, html)
        return True
