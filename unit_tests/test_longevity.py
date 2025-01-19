from longevity_test import LongevityTest
from unit_tests.test_utils_common import DummyDbCluster, DummyNode
from unit_tests.test_cluster import DummyDbCluster

import pytest
import threading
from unittest.mock import MagicMock

LongevityTest = pytest.mark.skip(reason="we don't need to run those tests")(LongevityTest)


@pytest.mark.sct_config(files='test-cases/scale/longevity-5000-tables.yaml')
def test_test_user_batch_custom_time(params):

    class DummyLongevityTest(LongevityTest):
        def _init_params(self):
            self.params = params

        def start_argus_heartbeat_thread(self):
            # prevent from heartbeat thread to start
            # because it can be left running after the test
            # and break other tests
            return threading.Event()

        def _pre_create_templated_user_schema(self, *args, **kwargs):
            pass

        def init_resources(self):
            pass

        def argus_collect_manager_version(self):
            pass

        def _run_all_stress_cmds(self, stress_queue, params):
            for _ in range(len(params['stress_cmd'])):
                m = MagicMock()
                m.verify_results.return_value = ('', [])
                stress_queue.append(m)

    test = DummyLongevityTest()
    test.setUp()
    node = DummyNode(name='test_node',
                     parent_cluster=None,
                     ssh_login_info=dict(key_file='~/.ssh/scylla-test'))
    node.parent_cluster = DummyDbCluster([node], params=params)
    node.parent_cluster.nemesis_termination_event = threading.Event()
    node.parent_cluster.nemesis = []
    node.parent_cluster.nemesis_threads = []
    test.db_cluster = node.parent_cluster
    test.monitors = MagicMock()
    test.test_user_batch_custom_time()
