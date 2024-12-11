from collections import namedtuple
import pytest

from sdcm.utils.replication_strategy_utils import temporary_replication_strategy_setter, \
    SimpleReplicationStrategy, NetworkTopologyReplicationStrategy, ReplicationStrategy, LocalReplicationStrategy


class TestReplicationStrategies:

    def test_can_create_simple_replication_strategy(self):  # pylint: disable=no-self-use
        strategy = SimpleReplicationStrategy(replication_factor=3)
        assert str(strategy) == "{'class': 'SimpleStrategy', 'replication_factor': 3}"

    def test_can_create_network_topology_replication_strategy(self):  # pylint: disable=no-self-use
        strategy = NetworkTopologyReplicationStrategy(dc1=3, dc2=8)
        assert str(strategy) == "{'class': 'NetworkTopologyStrategy', 'dc1': 3, 'dc2': 8}"

    def test_can_create_network_topology_replication_strategy_with_default_rf(self):  # pylint: disable=no-self-use
        strategy = NetworkTopologyReplicationStrategy(2, dc1=3, dc2=8)
        assert str(strategy) == "{'class': 'NetworkTopologyStrategy', 'replication_factor': 2, 'dc1': 3, 'dc2': 8}"

    def test_can_create_network_topology_replication_strategy_only_with_default_rf(self):  # pylint: disable=no-self-use
        strategy = NetworkTopologyReplicationStrategy(3)
        assert str(strategy) == "{'class': 'NetworkTopologyStrategy', 'replication_factor': 3}"

    def test_can_create_simple_replication_strategy_from_string(self):  # pylint: disable=no-self-use
        strategy = ReplicationStrategy.from_string(
            "REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 4}")
        assert isinstance(strategy, SimpleReplicationStrategy)
        assert str(strategy) == "{'class': 'SimpleStrategy', 'replication_factor': 4}"

        # test regex match is case insensitive and white spaces insensitive
        strategy = ReplicationStrategy.from_string(
            "replication = {'class': 'SimpleStrategy', 'replication_factor': 4}")
        assert isinstance(strategy, SimpleReplicationStrategy)
        assert str(strategy) == "{'class': 'SimpleStrategy', 'replication_factor': 4}"

    def test_can_create_network_topology_replication_strategy_from_string(self):  # pylint: disable=no-self-use
        strategy = ReplicationStrategy.from_string(
            "REPLICATION = { 'class' : 'NetworkTopologyStrategy', 'DC1' : 2, 'DC2': 8}")
        assert isinstance(strategy, NetworkTopologyReplicationStrategy)
        assert str(strategy) == "{'class': 'NetworkTopologyStrategy', 'DC1': 2, 'DC2': 8}"

    def test_can_create_network_topology_replication_strategy_from_string_with_replication_factor(self):  # pylint: disable=no-self-use
        strategy = ReplicationStrategy.from_string(
            "REPLICATION = { 'class' : 'NetworkTopologyStrategy', 'replication_factor' : 2}")
        assert isinstance(strategy, NetworkTopologyReplicationStrategy)
        assert str(strategy) == "{'class': 'NetworkTopologyStrategy', 'replication_factor': 2}"

    def test_get_replication_startegy_from_string_with_few_curly_braces(self):  # pylint: disable=no-self-use
        strategy = ReplicationStrategy.from_string(
            "replication = {'class': 'org.apache.cassandra.locator.SimpleStrategy', 'replication_factor': '1'} "
            "AND durable_writes = true AND tablets = {'enabled': false}")
        assert str(strategy) == "{'class': 'SimpleStrategy', 'replication_factor': 1}"

    def test_cannot_create_network_topology_replication_strategy_without_replication_factor(self):  # pylint: disable=no-self-use
        with pytest.raises(ValueError):
            NetworkTopologyReplicationStrategy()

    def test_can_create_local_replication_strategy(self):  # pylint: disable=no-self-use
        strategy = LocalReplicationStrategy()
        assert str(strategy) == "{'class': 'LocalStrategy'}"


class Cluster:  # pylint: disable=unused-argument,too-few-public-methods
    class Session:
        @staticmethod
        def execute(cql, timeout=None):
            if 'some error' in cql:
                raise AttributeError("found some error")
            print(cql)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    @staticmethod
    def cql_connection_patient(node):
        return Cluster.Session()


class Node():  # pylint: disable=too-few-public-methods

    def __init__(self):
        self.parent_cluster = Cluster()

    def run_cqlsh(self, cql):  # pylint: disable=no-self-use
        if 'some error' in cql:
            raise AttributeError("found some error")
        print(cql)
        ret = namedtuple("Result", 'stdout')
        ret.stdout = f"\n dd replication = {SimpleReplicationStrategy(4)}"
        return ret


class TestReplicationStrategySetter:

    def test_temporary_replication_strategy_setter_rolls_back_on_exit(self, capsys):  # pylint: disable=no-self-use
        with temporary_replication_strategy_setter(node=Node()) as replication_setter:
            replication_setter(
                ks=SimpleReplicationStrategy(3),
                ks2=NetworkTopologyReplicationStrategy(dc1=3, dc2=8)
            )
            replication_setter(ks=NetworkTopologyReplicationStrategy(dc1=8, dc2=9))
        out = iter(capsys.readouterr().out.splitlines())
        assert next(out) == "describe ks"
        assert next(out) == f"ALTER KEYSPACE ks WITH replication = {SimpleReplicationStrategy(3)}"
        assert next(out) == "describe ks2"
        assert next(out) == f"ALTER KEYSPACE ks2 WITH replication = {NetworkTopologyReplicationStrategy(dc1=3, dc2=8)}"
        assert next(out) == f"ALTER KEYSPACE ks WITH replication = {NetworkTopologyReplicationStrategy(dc1=8, dc2=9)}"
        # rollback validation
        assert next(out) == f"ALTER KEYSPACE ks WITH replication = {SimpleReplicationStrategy(4)}"
        assert next(out) == f"ALTER KEYSPACE ks2 WITH replication = {SimpleReplicationStrategy(4)}"
        with pytest.raises(StopIteration):
            # shouldn't do anything else
            next(out)

    def test_temporary_replication_strategy_setter_rolls_back_on_failure(self, capsys):  # pylint: disable=no-self-use
        with pytest.raises(AttributeError), temporary_replication_strategy_setter(node=Node()) as replication_setter:
            replication_setter(
                keyspace=SimpleReplicationStrategy(3), keyspace_x='some error',
                keyspace2=NetworkTopologyReplicationStrategy(dc1=3, dc2=8)
            )
        out = iter(capsys.readouterr().out.splitlines())
        assert next(out) == "describe keyspace"
        assert next(out) == f"ALTER KEYSPACE keyspace WITH replication = {SimpleReplicationStrategy(3)}"
        assert next(out) == "describe keyspace_x"
        # rollback validation
        assert next(out) == f"ALTER KEYSPACE keyspace WITH replication = {SimpleReplicationStrategy(4)}"
        assert next(out) == f"ALTER KEYSPACE keyspace_x WITH replication = {SimpleReplicationStrategy(4)}"
        with pytest.raises(StopIteration):
            # shouldn't do anything else
            next(out)
