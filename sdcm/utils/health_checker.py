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

import time
import logging
from typing import Generator

from sdcm.sct_events import Severity
from sdcm.sct_events.health import ClusterHealthValidatorEvent


CHECK_NODE_HEALTH_RETRIES = 10
CHECK_NODE_HEALTH_RETRY_DELAY = 15

LOGGER = logging.getLogger(__name__)

# Every health check function returns a generator of health events which should be published by a consumer:
#
#   >>> for event in check_fn():
#   ...     event.publish()
#
# It's done this way to be able add a retry mechanism for the cluster health validation.
HealthEventsGenerator = Generator[ClusterHealthValidatorEvent, None, None]


def check_nodes_status(nodes_status: dict, current_node, removed_nodes_list=()) -> HealthEventsGenerator:
    node_type = 'target' if current_node.running_nemesis else 'regular'
    if not nodes_status:
        LOGGER.warning("Node status info is not available. Search for the warning above")
        return
    LOGGER.debug("Status for %s node %s", node_type, current_node.name)
    for node, node_properties in nodes_status.items():
        if node_properties['status'] != "UN":
            LOGGER.debug("All nodes that have been removed up until this point: %s", str(removed_nodes_list))
            is_target = current_node.print_node_running_nemesis(node.ip_address)
            yield ClusterHealthValidatorEvent.NodeStatus(
                severity=Severity.CRITICAL,
                node=current_node.name,
                error=f"Current node {current_node}. "
                      f"Node {node}{is_target} status is {node_properties['status']}",
            )


def check_nulls_in_peers(gossip_info, peers_details, current_node) -> HealthEventsGenerator:
    """
    This validation is added to recreate the issue: https://github.com/scylladb/scylla/issues/4652
    Found scenario described in the https://github.com/scylladb/scylla/issues/6397
    """
    if not gossip_info:
        LOGGER.warning("Gossip info is not available. Search for the warning above")
        return

    for node, node_info in peers_details.items():
        if all(value != "null" for value in node_info.values()):
            continue

        is_target = current_node.print_node_running_nemesis(node.ip_address)
        message = f"Current node {current_node}. Found nulls in system.peers for " \
                  f"node {node}{is_target} with status {gossip_info.get(node, {}).get('status', 'n/a')} : " \
                  f"{peers_details[node]}"

        # By Asias request: https://github.com/scylladb/scylla/issues/6397#issuecomment-666893877
        LOGGER.debug("Print all columns from system.peers for peer %s", node)
        with current_node.parent_cluster.cql_connection_patient_exclusive(current_node) as session:
            result = session.execute(f"select * from system.peers where peer = '{node.ip_address}'")
            LOGGER.debug(result.one()._asdict())
        if node in gossip_info and gossip_info[node]['status'] not in current_node.GOSSIP_STATUSES_FILTER_OUT:
            yield ClusterHealthValidatorEvent.NodePeersNulls(
                severity=Severity.ERROR,
                node=current_node.name,
                error=message,
            )
        else:
            # Issue https://github.com/scylladb/scylla/issues/6397 - Should the info about decommissioned node
            # be kept in the system.peers?
            LOGGER.warning(message)


def check_node_status_in_gossip_and_nodetool_status(gossip_info, nodes_status, current_node) -> HealthEventsGenerator:
    if not nodes_status:
        LOGGER.warning("Node status info is not available. Search for the warning above. "
                       "Verify node status can't be performed")
        return

    for node, node_info in gossip_info.items():
        is_target = current_node.print_node_running_nemesis(node)
        if node not in nodes_status:
            if node_info['status'] not in current_node.GOSSIP_STATUSES_FILTER_OUT:
                LOGGER.debug("Gossip info: %s\nnodetool.status info: %s", gossip_info, nodes_status)
                yield ClusterHealthValidatorEvent.NodeStatus(
                    severity=Severity.ERROR,
                    node=current_node.name,
                    error=f"Current node {current_node}. The node {node}{is_target} "
                          f"exists in the gossip but doesn't exist in the nodetool.status",
                )
            continue

        if (node_info['status'] == 'NORMAL' and nodes_status[node]['status'] != 'UN') or \
                (node_info['status'] != 'NORMAL' and nodes_status[node]['status'] == 'UN'):
            LOGGER.debug("Gossip info: %s\nnodetool.status info: %s", gossip_info, nodes_status)
            yield ClusterHealthValidatorEvent.NodeStatus(
                severity=Severity.ERROR,
                node=current_node.name,
                error=f"Current node {current_node}. Wrong node status. "
                      f"Node {node}{is_target} status in nodetool.status is "
                      f"{nodes_status[node]['status']}, but status in gossip {node_info['status']}",
            )

    # Validate that all nodes in nodetool.status exist in gossip
    not_in_gossip = list(set(nodes_status.keys()) - set(gossip_info.keys()))
    for node in not_in_gossip:
        if nodes_status[node]['status'] == 'UN':
            is_target = current_node.print_node_running_nemesis(node.ip_address)
            LOGGER.debug("Gossip info: %s\nnodetool.status info: %s", gossip_info, nodes_status)
            yield ClusterHealthValidatorEvent.NodeSchemaVersion(
                severity=Severity.ERROR,
                node=current_node.name,
                error=f"Current node {current_node}. "
                      f"Node {node}{is_target} exists in the nodetool.status but missed in gossip.",
            )


def check_schema_version(gossip_info, peers_details, nodes_status, current_node) -> HealthEventsGenerator:
    if nodes_status and len(nodes_status.keys()) == 1:
        LOGGER.debug('There is one node only in the cluster. No peers data. '
                     'Verify schema version can\'t be performed')
        return

    if not peers_details:
        LOGGER.warning("SYSTEM.PEERS info is not availble. Search for the warning above. "
                       "Verify schema version can\'t be performed")
        return

    if not gossip_info:
        LOGGER.warning("Gossip info is not availble. Search for the warning above. "
                       "Verify schema version can\'t be performed")
        return

    debug_message = f"Gossip info: {gossip_info}\nSYSTEM.PEERS info: {peers_details}"
    # Validate schema version
    for node, node_info in gossip_info.items():
        # SYSTEM.PEERS table includes peers of the current node, so the node itcurrent_node doesn't exist in the list
        if current_node == node:
            continue

        # Can't validate the schema version if the node is not in NORMAL status
        if node_info['status'] != 'NORMAL':
            continue

        is_target = current_node.print_node_running_nemesis(node.ip_address)
        if node not in peers_details.keys():
            LOGGER.debug(debug_message)
            yield ClusterHealthValidatorEvent.NodeSchemaVersion(
                severity=Severity.ERROR,
                node=current_node.name,
                error=f"Current node {current_node}. "
                      f"Node {node}{is_target} exists in the gossip but missed in SYSTEM.PEERS.",
            )
            continue

        if node_info['schema'] != str(peers_details[node]['schema_version']):
            LOGGER.debug(debug_message)
            yield ClusterHealthValidatorEvent.NodeSchemaVersion(
                severity=Severity.ERROR,
                node=current_node.name,
                error=f"Current node {current_node}. Wrong Schema version. "
                      f"Node {node}{is_target} schema version in SYSTEM.PEERS is "
                      f"{peers_details[node]['schema_version']}, "
                      f"but schema version in gossip {node_info['schema']}",
            )

    # Validate that all nodes in SYSTEM.PEERS exist in gossip
    not_in_gossip = list(set(peers_details.keys()) - set(gossip_info.keys()))
    if not_in_gossip:
        LOGGER.debug(debug_message)
        yield ClusterHealthValidatorEvent.NodeSchemaVersion(
            severity=Severity.ERROR,
            node=current_node.name,
            error=f"Current node {current_node}. Nodes {','.join(node.ip_address for node in not_in_gossip)}"
                  f" exists in the SYSTEM.PEERS but missed in gossip.",
        )

    # Validate that same schema on all nodes in the gossip
    schema_version_on_all_nodes = [values['schema'] for values in gossip_info.values()
                                   if values['status'] not in current_node.GOSSIP_STATUSES_FILTER_OUT]

    if len(set(schema_version_on_all_nodes)) > 1:
        LOGGER.debug(debug_message)
        gossip_info_str = '\n'.join(
            f"{node}: {schema_version['schema']}" for node, schema_version in gossip_info.items())
        yield ClusterHealthValidatorEvent.NodeSchemaVersion(
            severity=Severity.WARNING,
            node=current_node.name,
            message=f"Current node {current_node}. "
                    f"Schema version is not same on all nodes in gossip info: {gossip_info_str}",
        )

    # Validate that same schema on all nodes in the SYSTEM.PEERS
    schema_version_on_all_nodes = [str(values['schema_version']) for node, values in peers_details.items()
                                   if node in gossip_info and gossip_info[node]['status'] not in
                                   current_node.GOSSIP_STATUSES_FILTER_OUT]

    if len(set(schema_version_on_all_nodes)) > 1:
        LOGGER.debug(debug_message)
        peers_info_str = '\n'.join(
            f"{node}: {schema_version['schema_version']}" for node, schema_version in peers_details.items())
        yield ClusterHealthValidatorEvent.NodeSchemaVersion(
            severity=Severity.WARNING,
            node=current_node.name,
            message=f"Current node {current_node}. "
                    f"Schema version is not same on all nodes in SYSTEM.PEERS info: {peers_info_str}",
        )


def check_schema_agreement_in_gossip_and_peers(node, retries: int = CHECK_NODE_HEALTH_RETRIES) -> str:
    err = ''
    for retry_n in range(retries):
        if retry_n:
            err = ''
            time.sleep(CHECK_NODE_HEALTH_RETRY_DELAY)
        message_pref = f'Check for schema agreement on the node {node.name} ({node}) [attempt #{retry_n}].'
        gossip_info = node.get_gossip_info()
        LOGGER.debug("%s Gossip info: %s", message_pref, gossip_info)
        if not gossip_info:
            err = f"{message_pref} Unable to get the gossip info"
            LOGGER.warning(err)
            continue

        peers_info = node.get_peers_info()
        LOGGER.debug("%s Peers info: %s", message_pref, peers_info)
        if not peers_info:
            err = f"{message_pref} Unable to get the peers info"
            LOGGER.warning(err)
            continue

        gossip_schema = {data["schema"] for data in gossip_info.values()
                         if data['status'] == "NORMAL"}
        if len(gossip_schema) > 1:
            err = f"{message_pref} Schema version is not same on nodes in the gossip"
            LOGGER.warning(err)
            continue

        for current_node, data in gossip_info.items():
            if not (data['status'] == "NORMAL" and current_node in peers_info):
                continue
            if data["schema"] != str(peers_info[current_node]['schema_version']):
                current_err = (f"{message_pref} Schema version is not same in "
                               f"the gossip and peers for {current_node}")
                LOGGER.warning(current_err)
                err += f"{current_err}\n"
                continue
        break  # Everything is OK, break the cycle.
    if not err:
        LOGGER.debug('Schema agreement has been completed on all nodes')
    return err


def check_group0_tokenring_consistency(group0_members: list[dict[str, str]],
                                       tokenring_members: list[dict[str, str]],
                                       current_node) -> HealthEventsGenerator:
    return current_node.raft.check_group0_tokenring_consistency(group0_members, tokenring_members)
