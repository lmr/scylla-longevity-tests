import contextlib
import logging
import random
import json

from enum import Enum
from abc import ABC, abstractmethod
from typing import NamedTuple, Mapping, Iterable, Any, Generator

from sdcm.sct_events.database import DatabaseLogEvent
from sdcm.sct_events.filters import EventsSeverityChangerFilter
from sdcm.sct_events.health import ClusterHealthValidatorEvent
from sdcm.sct_events import Severity
from sdcm.utils.features import is_consistent_topology_changes_feature_enabled, is_consistent_cluster_management_feature_enabled
from sdcm.utils.health_checker import HealthEventsGenerator
from sdcm.wait import wait_for
from sdcm.rest.raft_api import RaftApi
from sdcm.exceptions import ReadBarrierErrorException



LOGGER = logging.getLogger(__name__)
RAFT_DEFAULT_SCYLLA_VERSION = "5.5.0-dev"


class Group0MembersNotConsistentWithTokenRingMembersException(Exception):
    pass


class LogPosition(Enum):
    BEGIN = 0
    END = 1


class TopologyOperations(Enum):
    DECOMMISSION = "decommission"
    BOOTSTRAP = "bootstrap"


class MessagePosition(NamedTuple):
    log_message: str
    position: LogPosition


class MessageTimeout(NamedTuple):
    log_message: str
    timeout: int


BACKEND_TIMEOUTS: dict[str, Mapping[LogPosition, int]] = {
    "aws": {LogPosition.BEGIN: 300, LogPosition.END: 3600},
    "gce": {LogPosition.BEGIN: 300, LogPosition.END: 3600},
    "azure": {LogPosition.BEGIN: 1200, LogPosition.END: 7200},
}

ABORT_DECOMMISSION_LOG_PATTERNS: Iterable[MessagePosition] = [
    MessagePosition("api - decommission", LogPosition.BEGIN),
    MessagePosition("DECOMMISSIONING: unbootstrap starts", LogPosition.BEGIN),
    MessagePosition("DECOMMISSIONING: unbootstrap done", LogPosition.END),
    MessagePosition("becoming a group 0 non-voter", LogPosition.END),
    MessagePosition("became a group 0 non-voter", LogPosition.END),
    MessagePosition("leaving token ring", LogPosition.END),
    MessagePosition("left token ring", LogPosition.END),
    MessagePosition("raft_topology - decommission: waiting for completion", LogPosition.BEGIN),
    MessagePosition("repair - decommission_with_repair", LogPosition.END)
]

ABORT_BOOTSTRAP_LOG_PATTERNS: Iterable[MessagePosition] = [
    MessagePosition("Starting to bootstrap", LogPosition.BEGIN),
    MessagePosition("setup_group0: joining group 0", LogPosition.BEGIN),
    MessagePosition("setup_group0: successfully joined group 0", LogPosition.BEGIN),
    MessagePosition("setup_group0: the cluster is ready to use Raft. Finishing", LogPosition.BEGIN),
    MessagePosition("entering BOOTSTRAP mode", LogPosition.BEGIN),
    MessagePosition("storage_service - Bootstrap completed", LogPosition.END),
    MessagePosition("finish_setup_after_join: becoming a voter in the group 0 configuration", LogPosition.END),
    MessagePosition("raft_group0 - finish_setup_after_join: group 0 ID present, loading server info", LogPosition.END),
    MessagePosition("became a group 0 voter", LogPosition.END),
]


class RaftFeatureOperations(ABC):
    _node: "BaseNode"  # noqa: F821
    TOPOLOGY_OPERATION_LOG_PATTERNS: dict[TopologyOperations, Iterable[MessagePosition]]
    message_iter: Iterable | None = None

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """
                    Please do not use this method in simple statements like:
                        if not current_node.raft.is_enabled
                    move logic into separate methods of RaftFeature/NoRaft
                    classes instead
                """
        ...

    @property
    @abstractmethod
    def is_consistent_topology_changes_enabled(self) -> bool:
        ...

    @abstractmethod
    def get_status(self) -> str:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def get_group0_members(self) -> list[str]:
        ...

    @abstractmethod
    def get_group0_non_voters(self) -> list[dict[str, str]]:
        ...

    @abstractmethod
    def clean_group0_garbage(self, raise_exception: bool = False) -> None:
        ...

    def check_group0_tokenring_consistency(
            self, group0_members: list[dict[str, str]],
            tokenring_members: list[dict[str, str]]) -> [HealthEventsGenerator | None]:
        ...

    def get_message_waiting_timeout(self, message_position: MessagePosition) -> MessageTimeout:
        backend = self._node.parent_cluster.params["cluster_backend"]
        if backend not in BACKEND_TIMEOUTS:
            backend = "aws"
        return MessageTimeout(message_position.log_message,
                              BACKEND_TIMEOUTS[backend][message_position.position])

    def get_random_log_message(self, operation: TopologyOperations, seed: int | None = None):
        random.seed(seed)
        log_patterns = self.TOPOLOGY_OPERATION_LOG_PATTERNS.get(operation)
        log_pattern = random.choice(log_patterns)
        return self.get_message_waiting_timeout(log_pattern)

    def get_all_messages_timeouts(self, operation: TopologyOperations):
        log_patterns = self.TOPOLOGY_OPERATION_LOG_PATTERNS.get(operation)
        return list(map(self.get_message_waiting_timeout, log_patterns))

    def call_read_barrier(self):
        ...


class RaftFeature(RaftFeatureOperations):
    TOPOLOGY_OPERATION_LOG_PATTERNS: dict[TopologyOperations, Iterable[MessagePosition]] = {
        TopologyOperations.DECOMMISSION: ABORT_DECOMMISSION_LOG_PATTERNS,
        TopologyOperations.BOOTSTRAP: ABORT_BOOTSTRAP_LOG_PATTERNS,
    }

    def __init__(self, node: "BaseNode") -> None:  # noqa: F821
        super().__init__()
        self._node = node

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def is_consistent_topology_changes_enabled(self) -> bool:
        """Check whether CONSISTENT_TOPOLOGY_CHANGES feature is enabled"""
        with self._node.parent_cluster.cql_connection_patient(node=self._node) as session:
            return is_consistent_topology_changes_feature_enabled(session)

    def get_status(self) -> str:
        """ get raft status """
        with self._node.parent_cluster.cql_connection_patient_exclusive(node=self._node) as session:
            row = session.execute("SELECT * FROM system.scylla_local WHERE key = 'group0_upgrade_state'").one()
            return getattr(row, 'value', "")

    def is_ready(self) -> bool:
        """check if raft running

        According to https://docs.scylladb.com/branch-5.2/architecture/raft.html
        """
        state = self.get_status()
        return state == "use_post_raft_procedures"

    def get_group0_members(self) -> list[dict[str, str]]:
        LOGGER.debug("Get group0 members")
        group0_members = []
        try:
            with self._node.parent_cluster.cql_connection_patient_exclusive(node=self._node) as session:
                raft_group0_id = self.get_group0_id(session)
                assert raft_group0_id, "Group0 id was not found"
                rows = session.execute(f"select server_id, can_vote from system.raft_state  \
                                        where group_id = {raft_group0_id} and disposition = 'CURRENT'").all()
                for row in rows:
                    group0_members.append({"host_id": str(row.server_id),
                                           "voter": row.can_vote})
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            err_msg = f"Get group0 members failed with error: {exc}"
            LOGGER.error(err_msg)

        LOGGER.debug("Group0 members: %s", group0_members)
        return group0_members

    def get_group0_non_voters(self) -> list[str]:
        LOGGER.debug("Get group0 members in status non-voter")
        # Add host id which cann't vote after decommission was aborted because it is fast rebooted / terminated")
        return [member['host_id'] for member in self.get_group0_members() if not member['voter']]

    def get_diff_group0_token_ring_members(self) -> list[str]:
        LOGGER.debug("Get diff group0 from token ring")
        group0_members = self.get_group0_members()
        group0_members_ids = {member["host_id"] for member in group0_members}
        token_ring_members = self._node.get_token_ring_members()
        token_ring_member_ids = {member["host_id"] for member in token_ring_members}
        LOGGER.debug("Token rings members ids: %s", token_ring_member_ids)
        LOGGER.debug("Group0 members ids: %s", group0_members_ids)

        diff = list(group0_members_ids - token_ring_member_ids)
        LOGGER.debug("Group0 differs from token ring: %s", diff)
        return diff

    def clean_group0_garbage(self, raise_exception: bool = False) -> None:
        def is_node_down(removing_host_id: str) -> bool:
            node_status = get_node_status_from_system_by(verification_node=self._node,
                                                         host_id=removing_host_id)
            return not node_status.get("up", False)

        def is_node_in_bootstrapping_status(removing_host_id: str) -> bool:
            node_status = get_node_status_from_system_by(verification_node=self._node,
                                                         host_id=removing_host_id)
            return node_status.get("status", "") == "BOOTSTRAPPING"

        LOGGER.debug("Clean group0 non-voter's members")
        host_ids = self.get_diff_group0_token_ring_members()
        if not host_ids:
            LOGGER.debug("Node could return to token ring but not yet bootstrap")
            host_ids = self.get_group0_non_voters()
        attempt = 3
        while host_ids:
            removing_host_id = host_ids.pop(0)
            wait_for(func=is_node_down, step=5, timeout=60, throw_exc=False,
                     text=f"Waiting node with {removing_host_id} marked down", removing_host_id=removing_host_id)
            wait_for(func=lambda: not is_node_in_bootstrapping_status(), step=5, timeout=120, throw_exc=False,
                     text=f"Waiting node with {removing_host_id} doesn't have status BOOTSTRAPPING", removing_host_id=removing_host_id)

            ignore_dead_nodes_opt = f"--ignore-dead-nodes {','.join(host_ids)}" if host_ids else ""

            result = self._node.run_nodetool(f"removenode {removing_host_id} {ignore_dead_nodes_opt}",
                                             ignore_status=True,
                                             verbose=True,
                                             retry=3)
            if not result.ok:
                LOGGER.error("Removenode with host_id %s failed with %s",
                             removing_host_id, result.stdout + result.stderr)
                LOGGER.debug("Return host id to pool for remove")
                host_ids.append(removing_host_id)
                attempt -= 1
            if not host_ids or attempt < 1:
                break

        missing_host_ids = self.get_diff_group0_token_ring_members() or self.get_group0_non_voters()
        if missing_host_ids:
            token_ring_members = self._node.get_token_ring_members()
            group0_members = self.get_group0_members()
            error_msg = (f"Token ring {token_ring_members} and group0 {group0_members} are differs on: {missing_host_ids}"
                         f" or/and has non-voter member {missing_host_ids}")
            LOGGER.error(error_msg)
            if raise_exception:
                raise Group0MembersNotConsistentWithTokenRingMembersException(error_msg)
        LOGGER.debug("Group0 cleaned")

    @staticmethod
    def get_severity_change_filters_scylla_start_failed(timeout: int | None = None) -> tuple:
        return (
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.DATABASE_ERROR,
                                        regex=r".*storage_service - decommission.*Operation failed",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.DATABASE_ERROR,
                                        regex=r".*node_ops - decommission.*seastar::sleep_aborted",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.DATABASE_ERROR,
                                        regex=r".*This node was decommissioned and will not rejoin the ring",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.RUNTIME_ERROR,
                                        regex=r".*This node was decommissioned and will not rejoin the ring",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.RUNTIME_ERROR,
                                        regex=r".*Startup failed: std::runtime_error \(the topology coordinator rejected request to join the cluster",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.DATABASE_ERROR,
                                        regex=r".*gossip - is_safe_for_restart.*status=LEFT",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent,
                                        regex=".*init - Startup failed.*"),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.DATABASE_ERROR,
                                        regex=r".*node_ops - bootstrap.*Operation failed.*seastar::abort_requested_exception",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.DATABASE_ERROR,
                                        regex=r".*raft - .* failed with: raft::transport_error .* connection is closed",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.DATABASE_ERROR,
                                        regex=r".*raft - .*Transferring snapshot to.*Request is aborted by a caller",
                                        extra_time_to_expiration=timeout),
            EventsSeverityChangerFilter(new_severity=Severity.WARNING,
                                        event_class=DatabaseLogEvent.DATABASE_ERROR,
                                        regex=r".*raft.*applier fiber stopped because of the error.*abort requested",
                                        extra_time_to_expiration=timeout),
        )

    def is_cluster_topology_consistent(self) -> bool:
        group0_ids = [member["host_id"] for member in self.get_group0_members()]
        LOGGER.debug("Group0 member ids %s", group0_ids)
        token_ring_ids = [member["host_id"] for member in self._node.get_token_ring_members()]
        LOGGER.debug("Token ring member ids: %s", token_ring_ids)
        diff = set(group0_ids) - set(token_ring_ids) or set(token_ring_ids) - set(group0_ids)
        LOGGER.debug("Difference between group0 and token ring: %s", diff)
        num_of_nodes = len(self._node.parent_cluster.nodes)
        LOGGER.debug("Number of nodes in sct cluster %s", num_of_nodes)
        non_voters_ids = self.get_group0_non_voters()

        return not diff and not non_voters_ids and len(group0_ids) == len(token_ring_ids) == num_of_nodes

    def check_group0_tokenring_consistency(
            self, group0_members: list[dict[str, str]],
            tokenring_members: list[dict[str, str]]) -> HealthEventsGenerator:
        LOGGER.debug("Check group0 and token ring consistency on node %s (host_id=%s)...",
                     self._node.name, self._node.host_id)
        token_ring_node_ids = [member["host_id"] for member in tokenring_members]
        for member in group0_members:
            if member["voter"] and member["host_id"] in token_ring_node_ids:
                continue
            error_message = f"Node {self._node.name} has group0 member with host_id {member['host_id']} with " \
                f"can_vote {member['voter']} and " \
                f"presents in token ring {member['host_id'] in token_ring_node_ids}. " \
                f"Inconsistency between group0: {group0_members} " \
                f"and tokenring: {tokenring_members}"
            LOGGER.error(error_message)
            yield ClusterHealthValidatorEvent.Group0TokenRingInconsistency(
                severity=Severity.ERROR,
                node=self._node.name,
                error=error_message,
            )
        LOGGER.debug("Group0 and token-ring are consistent on node %s (host_id=%s)...",
                     self._node.name, self._node.host_id)

    def get_group0_id(self, session) -> str:
        try:
            row = session.execute("select value from system.scylla_local where key = 'raft_group0_id'").one()
            return row.value
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            err_msg = f"Get group0 members failed with error: {exc}"
            LOGGER.error(err_msg)
            return ""

    def call_read_barrier(self, timeout: int = 60):
        """ Wait until the node applies all previously committed Raft entries

        Any schema/topology changes are committed with Raft group0 on node. Before
        change is written to node, raft group0 checks that all previous schema/
        topology changes were applied. Raft group0 triggers read_barrier on node, and
        node applies all previous changes(commits) before new schema/operation
        write will be applied. After read barrier finished, it guarantees that
        node has all schema/topology changes, which was done in cluster before
        read_barrier started on node.
        """
        with self._node.parent_cluster.cql_connection_patient_exclusive(node=self._node) as session:
            raft_group0_id = self.get_group0_id(session)
            assert raft_group0_id, "Group0 id was not found"
        api = RaftApi(self._node)
        result = api.read_barrier(group_id=raft_group0_id, timeout=timeout)
        LOGGER.debug("Api response %s", result)
        if not result:
            return
        status = json.loads(result)
        raise ReadBarrierErrorException(f"Error code: {status['code']}, Error message: {status['message']}")


class NoRaft(RaftFeatureOperations):
    TOPOLOGY_OPERATION_LOG_PATTERNS = {
        TopologyOperations.DECOMMISSION: [
            MessagePosition("DECOMMISSIONING: unbootstrap starts", LogPosition.BEGIN)],
        TopologyOperations.BOOTSTRAP: [
            MessagePosition("entering BOOTSTRAP mode", LogPosition.BEGIN)
        ]
    }

    def __init__(self, node: "BaseNode") -> None:  # noqa: F821
        super().__init__()
        self._node = node

    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def is_consistent_topology_changes_enabled(self) -> bool:
        return False

    def get_status(self) -> str:
        return ""

    def is_ready(self) -> bool:
        return False

    def get_group0_members(self) -> list[dict[str, str]]:
        return []

    def get_group0_non_voters(self) -> list[str]:
        return []

    def clean_group0_garbage(self, raise_exception: bool = False) -> None:
        return

    def get_diff_group0_token_ring_members(self) -> list[str]:
        return []

    @staticmethod
    def get_severity_change_filters_scylla_start_failed(timeout: int = None) -> tuple:
        return (contextlib.nullcontext(),)

    def is_cluster_topology_consistent(self) -> bool:
        token_ring_ids = [member["host_id"] for member in self._node.get_token_ring_members()]
        LOGGER.debug("Token ring member ids: %s", token_ring_ids)
        num_of_nodes = len(self._node.parent_cluster.nodes)
        LOGGER.debug("Number of nodes in sct cluster %s", num_of_nodes)

        return len(token_ring_ids) == num_of_nodes

    def check_group0_tokenring_consistency(
            self, group0_members: list[dict[str, str]],
            tokenring_members: list[dict[str, str]]) -> Generator[None, None, None]:
        LOGGER.debug("Raft feature is disabled on node %s (host_id=%s)", self._node.name, self._node.host_id)

        yield None

    def call_read_barrier(self):
        ...


def get_raft_mode(node) -> RaftFeature | NoRaft:
    with node.parent_cluster.cql_connection_patient(node) as session:
        return RaftFeature(node) if is_consistent_cluster_management_feature_enabled(session) else NoRaft(node)


def get_node_status_from_system_by(verification_node: "BaseNode", *, ip_address: str = "", host_id: str = "") -> dict[str, Any]:  # noqa: F821
    """Get node status from system.cluster_status table

    The table contains actual information about nodes statuses in cluster
    updating by raft. it is faster to get required node state by ip or hostid
    from this table
    """
    query = "select peer, host_id, status, up from system.cluster_status"
    if ip_address:
        query += f" where peer = '{ip_address}'"
    elif host_id:
        query += f" where host_id={host_id} ALLOW FILTERING"
    else:
        LOGGER.warning("Ip address or host id were not provided")
        return {}

    with verification_node.parent_cluster.cql_connection_patient(node=verification_node) as session:
        session.default_timeout = 300
        results = session.execute(query)
        row = results.one()
        if not row:
            return {}
        node_status = {"ip_address": row.peer, "host_id": str(
            row.host_id), "state": row.status, "up": row.up}
        LOGGER.debug("Node status: %s", node_status)
        return node_status


__all__ = ["get_raft_mode",
           "get_node_status_from_system_by",
           "Group0MembersNotConsistentWithTokenRingMembersException", "RestApiError",
           "TopologyOperations"]
