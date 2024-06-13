import logging

from datetime import datetime, timedelta


LOGGER = logging.getLogger(__name__)


class QueryFilter:
    """
    Definition of query filtering parameters
    """
    SETUP_PARAMS = ['n_db_nodes', 'n_loaders', 'n_monitor_nodes']
    SETUP_INSTANCE_PARAMS = ['instance_type_db', 'instance_type_loader', 'instance_type_monitor', 'append_scylla_args']

    def __init__(self, test_doc, is_gce=False, use_wide_query=False, lastyear=False):
        self.test_doc = test_doc
        self.test_name = test_doc["_source"]["test_details"]["test_name"]
        self.is_gce = is_gce
        self.date_re = '/.*/'
        self.use_wide_query = use_wide_query
        self.lastyear = lastyear

    def setup_instance_parameters(self):
        return ['gce_' + param for param in self.SETUP_INSTANCE_PARAMS] if self.is_gce else self.SETUP_INSTANCE_PARAMS

    def filter_setup_details(self):
        setup_details = ''
        for param in self.SETUP_PARAMS + self.setup_instance_parameters():
            if setup_details:
                setup_details += ' AND '
            setup_details += 'setup_details.{}: "{}"'.format(param, self.test_doc['_source']['setup_details'][param])
        return setup_details

    def filter_test_details(self):
        test_details = 'test_details.job_name:\"{}\" '.format(
            self.test_doc['_source']['test_details']['job_name'].split('/')[0])
        test_details += self.test_cmd_details()
        test_details += ' AND test_details.time_completed: {}'.format(self.date_re)
        test_details += ' AND test_details.test_name: {}'.format(self.test_name.replace(":", r"\:"))
        return test_details

    @staticmethod
    def filter_test_for_last_year():
        year_ago = (datetime.today() - timedelta(days=365)).date().strftime("%Y%m%d")
        return f"versions.scylla-server.date:{{{year_ago} TO *}}"

    def test_cmd_details(self):
        raise NotImplementedError('Derived classes must implement this method.')

    def filter_by_dashboard_query(self):
        if "throughput" in self.test_doc['_source']['test_details']['job_name']:
            test_type = r'results.stats.op\ rate:*'
        elif "latency" in self.test_doc['_source']['test_details']['job_name']:
            test_type = r'results.stats.latency\ 99th\ percentile:*'
        else:
            test_type = ""
        return test_type

    def build_query(self):
        query = [self.filter_test_details()]

        if not self.use_wide_query:
            query.append(self.filter_setup_details())
        else:
            query.append(self.filter_by_dashboard_query())

        if self.lastyear:
            query.append(self.filter_test_for_last_year())

        return " AND ".join([q for q in query if q])

    def __call__(self, *args, **kwargs):
        try:
            return self.build_query()
        except KeyError:
            LOGGER.exception('Expected parameters for filtering are not found , test {}'.format(self.test_doc['_id']))
        return None


class PerformanceQueryFilter(QueryFilter):
    ALLOWED_PERFORMANCE_JOBS = [
        {
            "job_folder": "Perf-Regression",
            "job_prefix": ["scylla-release-perf-regression"]
        },
        {
            "job_folder": "scylla-master",
            "job_prefix": ["scylla-master-perf-regression"]
        },

    ]

    def filter_test_details(self):
        test_details = self.build_filter_job_name()
        if not self.use_wide_query:
            test_details += self.test_cmd_details()
            test_details += ' AND test_details.time_completed: {}'.format(self.date_re)
        test_details += ' AND {}'.format(self.build_filter_test_name())
        return test_details

    def build_filter_job_name(self):
        """Prepare filter to search by job name

        Select performance tests if job name is located in folder
        MAIN_JOB_FOLDERS, of job name starts from MAIN_JOB_FOLDER item( this
        is mostly doing for collecting previous tests)

        For other job filter documents only with exact same job name
        :returns: [description]
        :rtype: {[type]}
        """
        job_name = self.test_doc['_source']['test_details']['job_name'].split("/")

        def get_query_filter_by_job_folder(job_item):
            filter_query = ""
            if job_name[0] and job_name[0] in job_item['job_folder']:
                base_job_name = '\/'.join(job_name[1:])  # pylint: disable=anomalous-backslash-in-string
                if self.use_wide_query:
                    filter_query = rf'test_details.job_name.keyword: {job_name[0]}\/{base_job_name}'
                else:
                    filter_query = rf'(test_details.job_name.keyword: {job_name[0]}\/{base_job_name} OR \
                                       test_details.job_name.keyword: {base_job_name}) '
            return filter_query

        def get_query_filter_by_job_prefix(job_item):
            filter_query = ""
            for job_prefix in job_item['job_prefix']:
                if not job_name[0].startswith(job_prefix):
                    continue
                base_job_name = job_name[0]
                if self.use_wide_query:
                    filter_query = rf'test_details.job_name.keyword: {job_item["job_folder"]}\/{base_job_name}'
                else:
                    filter_query = rf'(test_details.job_name.keyword: {job_item["job_folder"]}\/{base_job_name} OR \
                                       test_details.job_name.keyword: {base_job_name}) '
                break
            return filter_query

        job_filter_query = ""
        for job_item in self.ALLOWED_PERFORMANCE_JOBS:
            job_filter_query = get_query_filter_by_job_folder(job_item)
            if not job_filter_query:
                job_filter_query = get_query_filter_by_job_prefix(job_item)
            if job_filter_query:
                break

        if not job_filter_query:
            full_job_name = self.test_doc['_source']['test_details']['job_name']
            if '/' in full_job_name:
                full_job_name = f'"{full_job_name}"'
            job_filter_query = r'test_details.job_name.keyword: {} '.format(full_job_name)

        return job_filter_query

    def build_filter_test_name(self):
        new_test_name = ""
        avocado_test_name = ""
        if ".py:" in self.test_name:
            new_test_name = self.test_name.replace(".py:", ".")
            avocado_test_name = self.test_name.replace(":", r"\:")
        else:
            avocado_test_name = self.test_name.replace(".", r".py\:", 1)
            new_test_name = self.test_name
        if self.use_wide_query:
            return f"test_details.test_name.keyword: {new_test_name}"
        else:
            return f" (test_details.test_name.keyword: {new_test_name} \
                       OR test_details.test_name.keyword: {avocado_test_name})"

    def test_cmd_details(self):
        raise NotImplementedError('Derived classes must implement this method.')


class QueryFilterCS(QueryFilter):
    _CMD = ('cassandra-stress',)
    _PRELOAD_CMD = ('preload-cassandra-stress',)
    _PARAMS = ('command', 'cl', 'rate threads', 'schema', 'mode', 'pop', 'duration')
    _PROFILE_PARAMS = ('command', 'profile', 'ops', 'rate threads', 'duration')

    def test_details_params(self):
        return self._CMD + self._PRELOAD_CMD if \
            self.test_doc['_source']['test_details'].get(self._PRELOAD_CMD[0]) else self._CMD

    def cs_params(self):
        return self._PROFILE_PARAMS if self.test_name.endswith('profiles') else self._PARAMS

    def test_cmd_details(self):
        test_details = ""
        for cassandra_stress in self.test_details_params():
            for param in self.cs_params():
                if param == 'rate threads':
                    test_details += r' AND test_details.{}.rate\ threads: {}'.format(
                        cassandra_stress, self.test_doc['_source']['test_details'][cassandra_stress][param])
                elif param == 'duration' and cassandra_stress.startswith('preload'):
                    continue
                else:
                    if param not in self.test_doc['_source']['test_details'][cassandra_stress]:
                        continue
                    param_val = self.test_doc['_source']['test_details'][cassandra_stress][param]
                    if param in ['profile', 'ops']:
                        param_val = "\"{}\"".format(param_val)
                    test_details += ' AND test_details.{}.{}: {}'.format(cassandra_stress, param, param_val)
        return test_details


class PerformanceFilterCS(QueryFilterCS, PerformanceQueryFilter):
    pass


class QueryFilterScyllaBench(QueryFilter):
    _CMD = ('scylla-bench',)
    _PARAMS = ('mode', 'workload', 'partition-count', 'clustering-row-count', 'concurrency', 'connection-count',
               'replication-factor', 'duration')

    def test_cmd_details(self):
        test_details = ['AND test_details.{}.{}: {}'.format(
            cmd, param, self.test_doc['_source']['test_details'][cmd][param])
            for param in self._PARAMS for cmd in self._CMD]

        return ' '.join(test_details)


class QueryFilterYCSB(QueryFilter):
    _YCSB_CMD = ('ycsb', )
    _YCSB_PARAMS = ('fieldcount', 'fieldlength', 'readproportion', 'insertproportion', 'recordcount', 'operationcount')

    def test_details_params(self):
        return self._YCSB_CMD

    def cs_params(self):
        return self._YCSB_PARAMS

    def test_cmd_details(self):
        test_details = ""
        for ycsb in self._YCSB_CMD:
            for param in self._YCSB_PARAMS:
                param_val = self.test_doc['_source']['test_details'][ycsb].get(param)
                if not param_val:
                    continue
                test_details += ' AND test_details.{}.{}: {}'.format(ycsb, param, param_val)
        return test_details


class PerformanceFilterYCSB(QueryFilterYCSB, PerformanceQueryFilter):
    pass


class PerformanceFilterScyllaBench(QueryFilterScyllaBench, PerformanceQueryFilter):
    pass


class CDCQueryFilter(QueryFilter):
    def filter_test_details(self):
        test_details = 'test_details.job_name: \"{}\" '.format(
            self.test_doc['_source']['test_details']['job_name'].split('/')[0])
        test_details += self.test_cmd_details()
        test_details += ' AND test_details.time_completed: {}'.format(self.date_re)
        test_details += r' AND test_details.sub_type: cdc* '
        return test_details

    def test_cmd_details(self):
        pass


class CDCQueryFilterCS(QueryFilterCS, CDCQueryFilter):
    def cs_params(self):
        return self._PROFILE_PARAMS if 'profiles' in self.test_name else self._PARAMS


class LatencyWithNemesisQueryFilter(QueryFilterCS, PerformanceQueryFilter):

    def filter_by_dashboard_query(self):
        return "latency_during_ops: *"
