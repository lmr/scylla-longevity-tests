"""
Unit tests for scan_operation_thread.py
contains 3 tests that check scan operations behavior in different conditions

test_scan_positive - positive scenario
test_scan_negative_operation_timed_out - getting operation_timed_out in scan execution
test_scan_negative_exception - getting operation_timed_out in scan execution (with and without nemesis)
"""
from pathlib import Path
import os
from threading import Event
from importlib import reload
from unittest.mock import MagicMock, patch
import pytest
from cassandra import OperationTimedOut, ReadTimeout

# from sdcm.utils.operations_thread import ThreadParams
from unit_tests.test_cluster import DummyDbCluster, DummyNode
from sdcm.utils.decorators import retrying, Retry
import sdcm.scan_operation_thread
from sdcm.scan_operation_thread import ScanOperationThread, ThreadParams, PrometheusDBStats


def mock_retrying_decorator(*args, **kwargs):  # pylint: disable=unused-argument
    """Decorate by doing nothing."""
    return retrying(1, 1, allowed_exceptions=(Retry, ))


with patch('sdcm.utils.decorators.retrying', mock_retrying_decorator):
    reload(sdcm.scan_operation_thread)

DEFAULT_PARAMS = {
    'termination_event': Event(),
    'user': 'sla_role_name',
    'user_password': 'sla_role_password',
    'duration': 10,
    'interval': 0,
    'validate_data': True
}


class DBCluster(DummyDbCluster):  # pylint: disable=abstract-method
    # pylint: disable=super-init-not-called
    def __init__(self, connection_mock, nodes, params):
        super().__init__(nodes, params=params)
        self.connection_mock = connection_mock
        self.params = {'nemesis_seed': 1}

    def get_non_system_ks_cf_list(*args, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-method-argument
        return ["test", "a.b"]

    def cql_connection_patient(self, *args, **kwargs):
        # pylint: disable=unused-argument
        return self.connection_mock


def get_event_log_file(events):
    if (log_file := Path(events.temp_dir, "events_log", "events.log")).exists():
        return log_file.read_text(encoding="utf-8").rstrip().split('\n')
    return ""


@pytest.fixture(scope='function', autouse=True)
def cleanup_event_log_file(events):
    with open(os.path.join(events.temp_dir, "events_log", "events.log"), 'r+', encoding="utf-8") as file:
        file.truncate(0)


@pytest.fixture(scope='module', autouse=True)
def mock_get_partition_keys():
    with patch('sdcm.scan_operation_thread.get_partition_keys'):
        yield


@pytest.fixture(scope='module')
def node():
    return DummyNode(name='test_node',
                     parent_cluster=None,
                     ssh_login_info=dict(key_file='~/.ssh/scylla-test'))


class MockCqlConnectionPatient(MagicMock):
    def execute_async(*args, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-method-argument
        class MockFuture:
            # pylint: disable=too-few-public-methods
            has_more_pages = False

            def add_callbacks(self, callback, errback):
                # pylint: disable=unused-argument
                # pylint: disable=no-self-use
                callback([MagicMock()])
        return MockFuture()

    events = ["Dispatching forward_request to 1 endpoints"]


@pytest.fixture(scope='module', name="cluster")
def new_cluster(node):  # pylint: disable=redefined-outer-name
    db_cluster = DBCluster(MockCqlConnectionPatient(), [node], {})
    node.parent_cluster = db_cluster

    def tester_obj():
        class Monitors:
            def __getattribute__(self, item):
                if item not in "external_address":
                    return self
                else:
                    return "test"

            def __getitem__(self, item):
                return self
        return Monitors()

    db_cluster.test_config.tester_obj = tester_obj
    return db_cluster


@pytest.mark.parametrize("mode", ['table', 'partition', 'aggregate'])
def test_scan_positive(mode, events, cluster):  # pylint: disable=redefined-outer-name
    default_params = ThreadParams(
        db_cluster=cluster,
        ks_cf='a.b',
        mode=mode,
        **DEFAULT_PARAMS
    )
    with patch.object(PrometheusDBStats, '__init__', return_value=None):
        with patch.object(PrometheusDBStats, 'query', return_value=[{'values': [[0, '1'], [1, '2']]}]):
            with events.wait_for_n_events(events.get_events_logger(), count=2, timeout=10):
                ScanOperationThread(default_params)._run_next_operation()  # pylint: disable=protected-access
            all_events = get_event_log_file(events)
            assert "Severity.NORMAL" in all_events[0] and "period_type=begin" in all_events[0]
            assert "Severity.NORMAL" in all_events[1] and "period_type=end" in all_events[1]
            if mode == "aggregate":
                assert "MockCqlConnectionPatient" in all_events[1]


def test_negative_prometheus_validation_error(events, cluster):
    default_params = ThreadParams(
        db_cluster=cluster,
        ks_cf='a.b',
        mode="aggregate",
        **DEFAULT_PARAMS
    )
    with patch.object(PrometheusDBStats, '__init__', return_value=None):
        with patch.object(PrometheusDBStats, 'query', return_value=[{'values': [[0, '1'], [1, '1']]}]):
            with events.wait_for_n_events(events.get_events_logger(), count=2, timeout=2):
                ScanOperationThread(default_params)._run_next_operation()  # pylint: disable=protected-access
            all_events = get_event_log_file(events)
            assert "Severity.NORMAL" in all_events[0] and "period_type=begin" in all_events[0]
            assert "Severity.ERROR" in all_events[1] and "period_type=end" in all_events[
                1] and "Fullscan failed - 'mapreduce_service_requests_dispatched_to_other_nodes' was not triggered" in all_events[1]


class ExecuteOperationTimedOutMockCqlConnectionPatient(MockCqlConnectionPatient):
    def execute(*args, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-method-argument
        raise OperationTimedOut("timeout")


class ExecuteAsyncOperationTimedOutMockCqlConnectionPatient(MockCqlConnectionPatient):
    def execute_async(*args, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-method-argument
        raise OperationTimedOut("timeout")


@pytest.mark.parametrize(("mode", 'severity', 'timeout', 'execute_mock'),
                         [['partition', 'WARNING', 0, 'execute_async'],
                          ['aggregate', 'ERROR', 60*30, 'execute'],
                          ['table', 'WARNING', 0, 'execute']])
def test_scan_negative_operation_timed_out(mode, severity, timeout, execute_mock, events, node):
    # pylint: disable=redefined-outer-name
    # pylint: disable=too-many-arguments
    if execute_mock == 'execute_async':
        connection = ExecuteAsyncOperationTimedOutMockCqlConnectionPatient()
    else:
        connection = ExecuteOperationTimedOutMockCqlConnectionPatient()
    db_cluster = DBCluster(connection, [node], {})
    node.parent_cluster = db_cluster
    default_params = ThreadParams(
        db_cluster=db_cluster,
        ks_cf='a.b',
        mode=mode,
        full_scan_aggregates_operation_limit=timeout,
        full_scan_operation_limit=timeout,
        **DEFAULT_PARAMS
    )
    with events.wait_for_n_events(events.get_events_logger(), count=2, timeout=10):
        ScanOperationThread(default_params)._run_next_operation()  # pylint: disable=protected-access
    all_events = get_event_log_file(events)
    assert "Severity.NORMAL" in all_events[0] and "period_type=begin" in all_events[0]
    assert f"Severity.{severity}" in all_events[1] and "period_type=end" in all_events[1]


class ExecuteReadTimeoutMockCqlConnectionPatient1(MockCqlConnectionPatient):
    def execute(*args, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-method-argument
        raise ReadTimeout("Operation timed out")


class ExecuteReadTimeoutMockCqlConnectionPatient2(MockCqlConnectionPatient):
    def execute(*args, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-method-argument
        raise ReadTimeout("some another reason")


@pytest.mark.parametrize(('execute_mock', "expected_message"),
                         [[ExecuteReadTimeoutMockCqlConnectionPatient1, "operation failed due to operation timed out"],
                          [ExecuteReadTimeoutMockCqlConnectionPatient2, "operation failed, ReadTimeout error"]])
def test_scan_negative_read_timedout(execute_mock, expected_message, events, node):
    # pylint: disable=redefined-outer-name
    # pylint: disable=too-many-arguments

    connection = execute_mock()
    db_cluster = DBCluster(connection, [node], {})
    node.parent_cluster = db_cluster
    default_params = ThreadParams(
        db_cluster=db_cluster,
        ks_cf='a.b',
        mode='aggregate',
        full_scan_aggregates_operation_limit=60*30,
        full_scan_operation_limit=300,
        **DEFAULT_PARAMS
    )
    with events.wait_for_n_events(events.get_events_logger(), count=2, timeout=10):
        ScanOperationThread(default_params)._run_next_operation()  # pylint: disable=protected-access
    all_events = get_event_log_file(events)
    assert "Severity.NORMAL" in all_events[0] and "period_type=begin" in all_events[0]
    assert "Severity.ERROR" in all_events[1] and "period_type=end" in all_events[1]
    assert expected_message in all_events[1]


class ExecuteExceptionMockCqlConnectionPatient(MockCqlConnectionPatient):
    def execute(*args, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-method-argument
        raise Exception("Exception")


class ExecuteAsyncExceptionMockCqlConnectionPatient(MockCqlConnectionPatient):
    def execute_async(*args, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-method-argument
        raise Exception("Exception")


@pytest.mark.parametrize(('mode', 'execute_mock'), [
    ['partition', 'execute_async'],
    ['aggregate', 'execute'],
    ['table', 'execute']])
def test_scan_negative_exception(mode, execute_mock, events, node):
    # pylint: disable=redefined-outer-name
    # pylint: disable=too-many-arguments
    if execute_mock == 'execute_async':
        connection = ExecuteAsyncExceptionMockCqlConnectionPatient()
    else:
        connection = ExecuteExceptionMockCqlConnectionPatient()
    db_cluster = DBCluster(connection, [node], {})
    node.parent_cluster = db_cluster
    default_params = ThreadParams(
        db_cluster=db_cluster,
        ks_cf='a.b',
        mode=mode,
        ** DEFAULT_PARAMS
    )
    with events.wait_for_n_events(events.get_events_logger(), count=2, timeout=10):
        ScanOperationThread(default_params)._run_next_operation()  # pylint: disable=protected-access
    all_events = get_event_log_file(events)
    assert "Severity.NORMAL" in all_events[0] and "period_type=begin" in all_events[0]
    assert f"Severity.ERROR" in all_events[1] and "period_type=end" in all_events[1]
