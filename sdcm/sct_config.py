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

"""
Handling Scylla-cluster-test configuration loading
"""
# pylint: disable=too-many-lines
import os
import re
import ast
import json
import logging
import getpass
import pathlib
import tempfile
import yaml
import copy
from copy import deepcopy
from typing import List, Union, Set

from distutils.util import strtobool
import anyconfig
from argus.client.sct.types import Package
from pydantic import BaseModel

from sdcm import sct_abs_path
import sdcm.provision.azure.utils as azure_utils
from sdcm.provision.aws.capacity_reservation import SCTCapacityReservation
from sdcm.utils import alternator
from sdcm.utils.aws_utils import get_arch_from_instance_type, aws_check_instance_type_supported
from sdcm.utils.common import (
    ami_built_by_scylla,
    get_ami_tags,
    get_branched_ami,
    get_branched_gce_images,
    get_scylla_ami_versions,
    get_scylla_gce_images_versions,
    convert_name_to_ami_if_needed,
    get_sct_root_path,
)
from sdcm.utils.operations_thread import ConfigParams
from sdcm.utils.version_utils import (
    ARGUS_VERSION_RE,
    get_branch_version,
    get_branch_version_for_multiple_repositories,
    get_scylla_docker_repo_from_version,
    resolve_latest_repo_symlink,
    get_specific_tag_of_docker_image,
    find_scylla_repo,
    is_enterprise,
)
from sdcm.sct_events.base import add_severity_limit_rules, print_critical_events
from sdcm.utils.gce_utils import (
    SUPPORTED_REGIONS as GCE_SUPPORTED_REGIONS,
    get_gce_image_tags,
    get_gce_compute_machine_types_client,
    gce_check_if_machine_type_supported,
)
from sdcm.utils.azure_utils import (
    azure_check_instance_type_available,
)
from sdcm.remote import LOCALRUNNER, shell_script_cmd
from sdcm.test_config import TestConfig
from sdcm.kafka.kafka_config import SctKafkaConfiguration
from sdcm.mgmt.common import AgentBackupParameters


def _str(value: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"{value} isn't a string, it is '{type(value)}'")


def _file(value: str) -> str:
    file_path = pathlib.Path(value)
    if file_path.is_file() and file_path.exists():
        return value
    raise ValueError(f"{value} isn't an existing file")


def str_or_list(value: Union[str, List[str], List[List[str]]]) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        for element in value:
            if isinstance(element, str):
                continue
            raise ValueError(
                f"Found non-str ({type(element)}) element in the list: {value}")
        return value
    raise ValueError(f"{value} isn't a string or a list of strings.")


def str_or_list_or_eval(value: Union[str, List[str]]) -> List[str]:
    """Convert an environment variable into a Python's list."""

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass
        return [str(value), ] if str(value) else []

    if isinstance(value, list):
        ret_values = []
        for val in value:
            try:
                ret_values += [ast.literal_eval(val)]
            except Exception:  # pylint: disable=broad-except  # noqa: BLE001
                ret_values += [str(val)]
        return ret_values

    raise ValueError(f"{value} isn't a string or a list")


def int_or_space_separated_ints(value):
    try:
        value = int(value)
        return value
    except Exception:  # pylint: disable=broad-except  # noqa: BLE001
        pass

    if isinstance(value, str):
        try:
            values = value.split()
            [int(v) for v in values]  # pylint: disable=expression-not-assigned
            return value
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass

    raise ValueError("{} isn't int or list".format(value))


def dict_or_str(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass

        # ast.literal_eval() can fail on some strings (e.g. which contain lowercased booleans), try parsing such strings
        # using yaml.safe_load()
        try:
            return yaml.safe_load(value)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass

    if isinstance(value, dict):
        return value

    raise ValueError('"{}" isn\'t a dict'.format(value))


def dict_or_str_or_pydantic(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass

    if isinstance(value, (dict, BaseModel)):
        return value

    raise ValueError('"{}" isn\'t a dict, str or Pydantic model'.format(value))


def boolean(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return bool(strtobool(value))
    else:
        raise ValueError("{} isn't a boolean".format(type(value)))


def is_config_option_appendable(option_name: str) -> bool:
    for option in SCTConfiguration.config_options:
        if option['name'] == option_name:
            break
    else:
        raise ValueError(f"Option {option_name} not found in SCTConfiguration.config_options")

    return option.get('appendable', option.get('type') in (str, str_or_list_or_eval, str_or_list))


def merge_dicts_append_strings(d1, d2):
    """
    merge two dictionaries, while having option
    to append string if the value starts with '++'
    and append list if first item is '++'
    """

    for key, value in copy.deepcopy(d2).items():
        if isinstance(value, str) and value.startswith('++'):
            assert is_config_option_appendable(key), f"Option {key} is not appendable"
            if key not in d1:
                d1[key] = ''
            d1[key] += value[2:]
            del d2[key]
        if isinstance(value, list) and value and isinstance(value[0], str) and value[0].startswith('++'):
            assert is_config_option_appendable(key), f"Option {key} is not appendable"
            if key not in d1:
                d1[key] = []
            d1[key].extend(value[1:])
            del d2[key]

    anyconfig.merge(d1, d2, ac_merge=anyconfig.MS_DICTS)


class SCTConfiguration(dict):
    """
    Class the hold the SCT configuration
    """
    available_backends = [
        'azure',
        'baremetal',
        'docker',
        # TODO: remove 'aws-siren' and 'gce-siren' backends completely when
        #       'siren-tests' project gets switched to the 'aws' and 'gce' ones.
        #       Such a switch must be fast change.
        'aws', 'aws-siren', 'k8s-local-kind-aws', 'k8s-eks',
        'gce', 'gce-siren', 'k8s-local-kind-gce', 'k8s-gke',
        'k8s-local-kind',
    ]

    config_options = [
        dict(name="config_files", env="SCT_CONFIG_FILES", type=str_or_list_or_eval,
             help="a list of config files that would be used", appendable=False),

        dict(name="cluster_backend", env="SCT_CLUSTER_BACKEND", type=str,
             help="backend that will be used, aws/gce/docker", appendable=False),

        dict(name="test_method", env="SCT_TEST_METHOD", type=str,
             help="class.method used to run the test. Filled automatically with run-test sct command.",
             appendable=False),

        dict(name="test_duration", env="SCT_TEST_DURATION", type=int,
             help="""
                  Test duration (min). Parameter used to keep instances produced by tests
                  and for jenkins pipeline timeout and TimoutThread.
             """),
        dict(name="prepare_stress_duration", env="SCT_PREPARE_STRESS_DURATION", type=int,
             help="""
                  Time in minutes, which is required to run prepare stress commands
                  defined in prepare_*_cmd for dataset generation, and is used in
                  test duration calculation
             """),
        dict(name="stress_duration", env="SCT_STRESS_DURATION", type=int,
             help="""
                  Time in minutes, Time of execution for stress commands from stress_cmd parameters
                  and is used in test duration calculation
             """),
        dict(name="n_db_nodes", env="SCT_N_DB_NODES", type=int_or_space_separated_ints,
             help="""Number list of database data nodes in multiple data centers. To use with
             multi data centers and zero nodes, dc with zero-nodes only should be set as 0,
             ex. "3 3 0"."""),

        dict(name="n_test_oracle_db_nodes", env="SCT_N_TEST_ORACLE_DB_NODES", type=int_or_space_separated_ints,
             help="Number list of oracle test nodes in multiple data centers."),

        dict(name="n_loaders", env="SCT_N_LOADERS", type=int_or_space_separated_ints,
             help="Number list of loader nodes in multiple data centers"),

        dict(name="n_monitor_nodes", env="SCT_N_MONITORS_NODES", type=int_or_space_separated_ints,
             help="Number list of monitor nodes in multiple data centers"),

        dict(name="intra_node_comm_public", env="SCT_INTRA_NODE_COMM_PUBLIC", type=boolean,
             help="If True, all communication between nodes are via public addresses"),

        dict(name="endpoint_snitch", env="SCT_ENDPOINT_SNITCH", type=str,
             help="""
                The snitch class scylla would use

                'GossipingPropertyFileSnitch' - default
                'Ec2MultiRegionSnitch' - default on aws backend
                'GoogleCloudSnitch'
             """),

        dict(name="user_credentials_path", env="SCT_USER_CREDENTIALS_PATH", type=str,
             help="""Path to your user credentials. qa key are downloaded automatically from S3 bucket"""),

        dict(name="cloud_credentials_path", env="SCT_CLOUD_CREDENTIALS_PATH", type=str,
             help="""Path to your user credentials. qa key are downloaded automatically from S3 bucket"""),

        dict(name="cloud_cluster_id", env="SCT_CLOUD_CLUSTER_ID", type=int,
             help="""scylla cloud cluster id"""),

        dict(name="cloud_prom_bearer_token", env="SCT_CLOUD_PROM_BEARER_TOKEN", type=str,
             help="""scylla cloud promproxy bearer_token to federate monitoring data into our monitoring instance"""),

        dict(name="cloud_prom_path", env="SCT_CLOUD_PROM_PATH", type=str,
             help="""scylla cloud promproxy path to federate monitoring data into our monitoring instance"""),

        dict(name="cloud_prom_host", env="SCT_CLOUD_PROM_HOST", type=str,
             help="""scylla cloud promproxy hostname to federate monitoring data into our monitoring instance"""),

        dict(name="ip_ssh_connections", env="SCT_IP_SSH_CONNECTIONS", type=str,
             help="""
                Type of IP used to connect to machine instances.
                This depends on whether you are running your tests from a machine inside
                your cloud provider, where it makes sense to use 'private', or outside (use 'public')

                Default: Use public IPs to connect to instances (public)
                Use private IPs to connect to instances (private)
                Use IPv6 IPs to connect to instances (ipv6)
             """,
             choices=("public", "private", "ipv6"),
             ),

        dict(name="scylla_repo", env="SCT_SCYLLA_REPO", type=str,
             help="Url to the repo of scylla version to install scylla. Can provide specific version after a colon "
                  "e.g: `https://s3.amazonaws.com/downloads.scylladb.com/deb/ubuntu/scylla-2021.1.list:2021.1.18`"),

        dict(name="scylla_apt_keys", env="SCT_SCYLLA_APT_KEYS", type=str_or_list,
             help="APT keys for ScyllaDB repos"),

        dict(name="unified_package", env="SCT_UNIFIED_PACKAGE", type=str,
             help="Url to the unified package of scylla version to install scylla"),

        dict(name="nonroot_offline_install", env="SCT_NONROOT_OFFLINE_INSTALL", type=boolean,
             help="Install Scylla without required root priviledge"),

        dict(name="install_mode", env="SCT_INSTALL_MODE", type=str,
             help="Scylla install mode, repo/offline/web",
             appendable=False),

        dict(name="scylla_version", env="SCT_SCYLLA_VERSION",
             type=str,
             help="""Version of scylla to install, ex. '2.3.1'
                     Automatically lookup AMIs and repo links for formal versions.
                     WARNING: can't be used together with 'scylla_repo' or 'ami_id_db_scylla'""",
             appendable=False),

        dict(name="user_data_format_version", env="SCT_USER_DATA_FORMAT_VERSION",
             type=str,
             help="""Format version of the user-data to use for scylla images,
                     default to what tagged on the image used""",
             appendable=False),

        dict(name="oracle_user_data_format_version", env="SCT_ORACLE_USER_DATA_FORMAT_VERSION",
             type=str,
             help="""Format version of the user-data to use for scylla images,
                 default to what tagged on the image used""",
             appendable=False),

        dict(name="oracle_scylla_version", env="SCT_ORACLE_SCYLLA_VERSION",
             type=str,
             help="""Version of scylla to use as oracle cluster with gemini tests, ex. '3.0.11'
                     Automatically lookup AMIs for formal versions.
                     WARNING: can't be used together with 'ami_id_db_oracle'""",
             appendable=False),

        dict(name="scylla_linux_distro", env="SCT_SCYLLA_LINUX_DISTRO", type=str,
             help="""The distro name and family name to use. Example: 'ubuntu-jammy' or 'debian-bookworm'.""",
             appendable=False),

        dict(name="scylla_linux_distro_loader", env="SCT_SCYLLA_LINUX_DISTRO_LOADER", type=str,
             help="""The distro name and family name to use. Example: 'ubuntu-jammy' or 'debian-bookworm'.""",
             appendable=False),

        dict(name="assert_linux_distro_features", env="SCT_ASSERT_LINUX_DISTRO_FEATURES", type=str_or_list_or_eval,
             help="""List of distro features relevant to SCT test. Example: 'fips'.""",
             appendable=True),

        dict(name="scylla_repo_m", env="SCT_SCYLLA_REPO_M", type=str,
             help="Url to the repo of scylla version to install scylla from for managment tests"),

        dict(name="scylla_repo_loader", env="SCT_SCYLLA_REPO_LOADER", type=str,
             help="Url to the repo of scylla version to install c-s for loader"),

        dict(name="scylla_mgmt_address", env="SCT_SCYLLA_MGMT_ADDRESS",
             type=str,
             help="Url to the repo of scylla manager version to install for management tests"),

        dict(name="scylla_mgmt_agent_address", env="SCT_SCYLLA_MGMT_AGENT_ADDRESS",
             type=str,
             help="Url to the repo of scylla manager agent version to install for management tests"),

        dict(name="manager_version", env="SCT_MANAGER_VERSION",
             type=str,
             help="Branch of scylla manager server and agent to install. Options in defaults/manager_versions.yaml",
             appendable=False),

        dict(name="target_manager_version", env="SCT_TARGET_MANAGER_VERSION",
             type=str,
             help="Branch of scylla manager server and agent to upgrade to. Options in defaults/manager_versions.yaml",
             appendable=False),

        dict(name="manager_scylla_backend_version", env="SCT_MANAGER_SCYLLA_BACKEND_VERSION",
             type=str,
             help="Branch of scylla db enterprise to install. Options in defaults/manager_versions.yaml",
             appendable=False),

        dict(name="scylla_mgmt_agent_version", env="SCT_SCYLLA_MGMT_AGENT_VERSION", type=str,
             help="",
             appendable=False),

        dict(name="scylla_mgmt_pkg", env="SCT_SCYLLA_MGMT_PKG",
             type=str,
             help="Url to the scylla manager packages to install for management tests"),

        dict(name="stress_cmd_lwt_i", env="SCT_STRESS_CMD_LWT_I",
             type=str,
             help="Stress command for LWT performance test for INSERT baseline"),

        dict(name="stress_cmd_lwt_d", env="SCT_STRESS_CMD_LWT_D",
             type=str,
             help="Stress command for LWT performance test for DELETE baseline"),

        dict(name="stress_cmd_lwt_u", env="SCT_STRESS_CMD_LWT_U",
             type=str,
             help="Stress command for LWT performance test for UPDATE baseline"),

        dict(name="stress_cmd_lwt_ine", env="SCT_STRESS_CMD_LWT_INE",
             type=str,
             help="Stress command for LWT performance test for INSERT with IF NOT EXISTS"),

        dict(name="stress_cmd_lwt_uc", env="SCT_STRESS_CMD_LWT_UC",
             type=str,
             help="Stress command for LWT performance test for UPDATE with IF <condition>"),

        dict(name="stress_cmd_lwt_ue", env="SCT_STRESS_CMD_LWT_UE",
             type=str,
             help="Stress command for LWT performance test for UPDATE with IF EXISTS"),

        dict(name="stress_cmd_lwt_de", env="SCT_STRESS_CMD_LWT_DE",
             type=str,
             help="Stress command for LWT performance test for DELETE with IF EXISTS"),

        dict(name="stress_cmd_lwt_dc", env="SCT_STRESS_CMD_LWT_DC",
             type=str,
             help="Stress command for LWT performance test for DELETE with IF condition>"),

        dict(name="stress_cmd_lwt_mixed", env="SCT_STRESS_CMD_LWT_MIXED",
             type=str,
             help="Stress command for LWT performance test for mixed lwt load"),

        dict(name="stress_cmd_lwt_mixed_baseline", env="SCT_STRESS_CMD_LWT_MIXED_BASELINE",
             type=str,
             help="Stress command for LWT performance test for mixed lwt load baseline"),

        dict(name="use_cloud_manager", env="SCT_USE_CLOUD_MANAGER", type=boolean,
             help="When define true, will install scylla cloud manager"),

        dict(name="use_ldap", env="SCT_USE_LDAP", type=boolean,
             help="When defined true, LDAP is going to be used."),

        dict(name="use_ldap_authorization", env="SCT_USE_LDAP_AUTHORIZATION", type=boolean,
             help="When defined true, will create a docker container with LDAP and configure scylla.yaml to use it"),

        dict(name="use_ldap_authentication", env="SCT_USE_LDAP_AUTHENTICATION", type=boolean,
             help="When defined true, will create a docker container with LDAP and configure scylla.yaml to use it"),

        dict(name="prepare_saslauthd", env="SCT_PREPARE_SASLAUTHD", type=boolean,
             help="When defined true, will install and start saslauthd service"),

        dict(name="ldap_server_type", env="SCT_LDAP_SERVER_TYPE", type=str,
             help="This option indicates which server is going to be used for LDAP operations. [openldap, ms_ad]"),

        dict(name="use_mgmt", env="SCT_USE_MGMT", type=boolean,
             help="When define true, will install scylla management"),

        dict(name="parallel_node_operations", env="SCT_PARALLEL_NODE_OPERATIONS", type=boolean,
             help="When defined true, will run node operations in parallel. Supported operations: startup"),

        dict(name="manager_prometheus_port", env="SCT_MANAGER_PROMETHEUS_PORT", type=int,
             help="Port to be used by the manager to contact Prometheus"),

        dict(name="target_scylla_mgmt_server_address", env="SCT_TARGET_SCYLLA_MGMT_SERVER_ADDRESS", type=str,
             help="Url to the repo of scylla manager version used to upgrade the manager server"),

        dict(name="target_scylla_mgmt_agent_address", env="SCT_TARGET_SCYLLA_MGMT_AGENT_ADDRESS", type=str,
             help="Url to the repo of scylla manager version used to upgrade the manager agents"),

        dict(name="update_db_packages", env="SCT_UPDATE_DB_PACKAGES", type=str,
             help="""A local directory of rpms to install a custom version on top of
                     the scylla installed (or from repo or from ami)"""),

        dict(name="monitor_branch", env="SCT_MONITOR_BRANCH", type=str,
             help="The port of scylla management"),

        dict(name="db_type", env="SCT_DB_TYPE", type=str,
             help="Db type to install into db nodes, scylla/cassandra"),

        dict(name="user_prefix", env="SCT_USER_PREFIX", type=str,

             help="the prefix of the name of the cloud instances, defaults to username"),

        dict(name="ami_id_db_scylla_desc", env="SCT_AMI_ID_DB_SCYLLA_DESC", type=str,
             help="version name to report stats to Elasticsearch and tagged on cloud instances"),

        dict(name="sct_public_ip", env="SCT_SCT_PUBLIC_IP", type=str,
             help="""
                Override the default hostname address of the sct test runner,
                for the monitoring of the Nemesis.
                can only work out of the box in AWS
             """),
        dict(name="sct_ngrok_name", env="SCT_NGROK_NAME", type=str,
             help="""
            Override the default hostname address of the sct test runner,
            using ngrok server, see readme for more instructions
         """),

        dict(name="backtrace_decoding", env="SCT_BACKTRACE_DECODING", type=boolean,
             help="""If True, all backtraces found in db nodes would be decoded automatically"""),

        dict(name="print_kernel_callstack", env="SCT_PRINT_KERNEL_CALLSTACK", type=boolean,
             help="""Scylla will print kernel callstack to logs if True, otherwise, it will try and may print a message
             that it failed to."""),

        dict(name="instance_provision", env="SCT_INSTANCE_PROVISION", type=str,
             help="instance_provision: spot|on_demand|spot_fleet"),

        dict(name="instance_provision_fallback_on_demand", env="SCT_INSTANCE_PROVISION_FALLBACK_ON_DEMAND",
             type=boolean,
             help="instance_provision_fallback_on_demand: create instance on_demand provision type if instance with selected "
                  "'instance_provision' type creation failed. "
                  "Expected values: true|false (default - false"),

        dict(name="reuse_cluster", env="SCT_REUSE_CLUSTER", type=str,
             help="""
            If reuse_cluster is set it should hold test_id of the cluster that will be reused.
            `reuse_cluster: 7dc6db84-eb01-4b61-a946-b5c72e0f6d71`
         """),

        dict(name="test_id", env="SCT_TEST_ID", type=str,
             help="""Set the test_id of the run manually. Use only from the env before running Hydra"""),

        dict(name="db_nodes_shards_selection", env="SCT_NODES_SHARDS_SELECTION", type=str,
             choices=['default', 'random'],
             help="""How to select number of shards of Scylla. Expected values: default/random.
             Default value: 'default'.
             In case of random option - Scylla will start with different (random) shards on every node of the cluster
             """),

        dict(name="seeds_selector", env="SCT_SEEDS_SELECTOR", type=str,
             choices=['random', 'first', 'all'],
             help="""How to select the seeds. Expected values: random/first/all"""),

        dict(name="seeds_num", env="SCT_SEEDS_NUM", type=int,
             help="""Number of seeds to select"""),

        dict(name="email_recipients", env="SCT_EMAIL_RECIPIENTS", type=str_or_list,
             help="""list of email of send the performance regression test to"""),

        dict(name="email_subject_postfix", env="SCT_EMAIL_SUBJECT_POSTFIX", type=str,
             help="""Email subject postfix"""),

        dict(name="enable_test_profiling", env="SCT_ENABLE_TEST_PROFILING", type=boolean,
             help="""Turn on sct profiling"""),
        dict(name="ssh_transport", env="SSH_TRANSPORT", type=str,
             help="""Set type of ssh library to use. Could be 'fabric' (default) or 'libssh2'"""),

        # Scylla command line arguments options
        dict(name="experimental_features", env="SCT_EXPERIMENTAL_FEATURES", type=list,
             help="unlock specified experimental features"),

        dict(name="server_encrypt", env="SCT_SERVER_ENCRYPT", type=boolean,
             help="when enable scylla will use encryption on the server side"),

        dict(name="client_encrypt", env="SCT_CLIENT_ENCRYPT", type=boolean,
             help="when enable scylla will use encryption on the client side"),

        dict(name="client_encrypt_mtls", env="SCT_CLIENT_ENCRYPT_MTLS", type=boolean,
             help="when enabled scylla will enforce mutual authentication when client-to-node encryption is enabled"),

        dict(name="server_encrypt_mtls", env="SCT_SERVER_ENCRYPT_MTLS", type=boolean,
             help="when enabled scylla will enforce mutual authentication when node-to-node encryption is enabled"),

        dict(name="hinted_handoff", env="SCT_HINTED_HANDOFF", type=str,
             help="when enable or disable scylla hinted handoff (enabled/disabled)"),

        dict(name="authenticator", env="SCT_AUTHENTICATOR", type=str,
             help="which authenticator scylla will use AllowAllAuthenticator/PasswordAuthenticator",
             choices=("PasswordAuthenticator", "AllowAllAuthenticator", "com.scylladb.auth.SaslauthdAuthenticator"),
             ),

        dict(name="authenticator_user", env="SCT_AUTHENTICATOR_USER", type=str,
             help="the username if PasswordAuthenticator is used"),

        dict(name="authenticator_password", env="SCT_AUTHENTICATOR_PASSWORD", type=str,
             help="the password if PasswordAuthenticator is used"),

        dict(name="authorizer", env="SCT_AUTHORIZER", type=str,
             help="which authorizer scylla will use AllowAllAuthorizer/CassandraAuthorizer"),

        # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
        dict(name="sla", env="SCT_SLA", type=boolean,
             help="run SLA nemeses if the test is SLA only"),

        dict(name="service_level_shares", env="SCT_SERVICE_LEVEL_SHARES", type=list,
             help="List if service level shares - how many server levels to create and test. Uses in SLA test."
                  "list of int, like: [100, 200]"),

        dict(name="alternator_port", env="SCT_ALTERNATOR_PORT", type=int,
             help="Port to configure for alternator in scylla.yaml"),
        dict(name="dynamodb_primarykey_type", env="SCT_DYNAMODB_PRIMARYKEY_TYPE", type=str,
             help=f"Type of dynamodb table to create with range key or not, can be:\n"
                  f"{','.join([schema.value for schema in alternator.enums.YCSBSchemaTypes])}",
             choices=[schema.value for schema in alternator.enums.YCSBSchemaTypes]),
        dict(name="alternator_write_isolation", env="SCT_ALTERNATOR_WRITE_ISOLATION", type=str,
             help="Set the write isolation for the alternator table, see https://github.com/scylladb/scylla/blob"
                  "/master/docs/alternator/alternator.md#write-isolation-policies for more details"),
        dict(name="alternator_use_dns_routing", env="SCT_ALTERNATOR_USE_DNS_ROUTING", type=boolean,
             help="If true, spawn a docker with a dns server for the ycsb loader to point to"),
        dict(name="alternator_enforce_authorization", env="SCT_ALTERNATOR_ENFORCE_AUTHORIZATION", type=boolean,
             help="If true, enable the authorization check in dynamodb api (alternator)"),
        dict(name="alternator_access_key_id", env="SCT_ALTERNATOR_ACCESS_KEY_ID", type=str,
             help="the aws_access_key_id that would be used for alternator"),
        dict(name="alternator_secret_access_key", env="SCT_ALTERNATOR_SECRET_ACCESS_KEY", type=str,
             help="the aws_secret_access_key that would be used for alternator"),

        dict(name="region_aware_loader", env="SCT_REGION_AWARE_LOADER", type=boolean,
             help="When in multi region mode, run stress on loader that is located in the same region as db node"),

        dict(name="append_scylla_args", env="SCT_APPEND_SCYLLA_ARGS", type=str,
             help="More arguments to append to scylla command line"),

        dict(name="append_scylla_args_oracle", env="SCT_APPEND_SCYLLA_ARGS_ORACLE", type=str,
             help="More arguments to append to oracle command line"),

        dict(name="append_scylla_yaml", env="SCT_APPEND_SCYLLA_YAML", type=dict_or_str,
             help="More configuration to append to /etc/scylla/scylla.yaml"),

        dict(name="append_scylla_node_exporter_args", env="SCT_SCYLLA_NODE_EXPORTER_ARGS", type=str,
             help="More arguments to append to scylla-node-exporter command line"),

        # Nemesis config options

        dict(name="nemesis_class_name", env="SCT_NEMESIS_CLASS_NAME",
             type=_str, k8s_multitenancy_supported=True,
             help="""
                    Nemesis class to use (possible types in sdcm.nemesis).
                    Next syntax supporting:
                    - nemesis_class_name: "NemesisName"  Run one nemesis in single thread
                    - nemesis_class_name: "<NemesisName>:<num>" Run <NemesisName> in <num>
                      parallel threads on different nodes. Ex.: "ChaosMonkey:2"
                    - nemesis_class_name: "<NemesisName1>:<num1> <NemesisName2>:<num2>" Run
                      <NemesisName1> in <num1> parallel threads and <NemesisName2> in <num2>
                      parallel threads. Ex.: "DisruptiveMonkey:1 NonDisruptiveMonkey:2"
            """),

        dict(name="nemesis_interval", env="SCT_NEMESIS_INTERVAL",
             type=int, k8s_multitenancy_supported=True,
             help="""Nemesis sleep interval to use if None provided specifically in the test"""),
        dict(name="nemesis_sequence_sleep_between_ops", env="SCT_NEMESIS_SEQUENCE_SLEEP_BETWEEN_OPS",
             type=int, k8s_multitenancy_supported=True,
             help="""Sleep interval between nemesis operations for use in unique_sequence nemesis kind of tests"""),

        dict(name="nemesis_during_prepare", env="SCT_NEMESIS_DURING_PREPARE",
             type=boolean, k8s_multitenancy_supported=True,
             help="""Run nemesis during prepare stage of the test"""),

        dict(name="nemesis_seed", env="SCT_NEMESIS_SEED",
             type=int, k8s_multitenancy_supported=True,
             help="""A seed number in order to repeat nemesis sequence as part of SisyphusMonkey"""),

        dict(name="nemesis_add_node_cnt",
             env="SCT_NEMESIS_ADD_NODE_CNT",
             type=int, k8s_multitenancy_supported=True,
             help="""Add/remove nodes during GrowShrinkCluster nemesis"""),

        dict(name="nemesis_grow_shrink_instance_type",
             env="SCT_NEMESIS_GROW_SHRINK_INSTANCE_TYPE",
             type=_str, k8s_multitenancy_supported=True,
             help="""Instance type to use for adding/removing nodes during GrowShrinkCluster nemesis"""),

        dict(name="cluster_target_size", env="SCT_CLUSTER_TARGET_SIZE", type=int,
             help="""Used for scale test: max size of the cluster"""),

        dict(name="space_node_threshold", env="SCT_SPACE_NODE_THRESHOLD",
             type=int, k8s_multitenancy_supported=True,
             help="""
                 Space node threshold before starting nemesis (bytes)
                 The default value is 6GB (6x1024^3 bytes)
                 This value is supposed to reproduce
                 https://github.com/scylladb/scylla/issues/1140
             """),

        dict(name="nemesis_filter_seeds", env="SCT_NEMESIS_FILTER_SEEDS",
             type=boolean, k8s_multitenancy_supported=True,
             help="""If true runs the nemesis only on non seed nodes"""),

        # Stress Commands

        dict(name="stress_cmd", env="SCT_STRESS_CMD",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="gemini_schema_url", env="SCT_GEMINI_SCHEMA_URL", type=str,
             help="""Url of the schema/configuration the gemini tool would use """),

        dict(name="gemini_cmd", env="SCT_GEMINI_CMD", type=str,
             help="""gemini command to run (for now used only in GeminiTest)"""),

        dict(name="gemini_seed", env="SCT_GEMINI_SEED", type=int,
             help="Seed number for gemini command"),
        dict(name="gemini_table_options", env="SCT_GEMINI_TABLE_OPTIONS", type=list,
             help="""table options for created table. example:
                     ["cdc={'enabled': true}"]
                     ["cdc={'enabled': true}", "compaction={'class': 'IncrementalCompactionStrategy'}"] """),
        # AWS config options

        dict(name="instance_type_loader", env="SCT_INSTANCE_TYPE_LOADER", type=str,
             help="AWS image type of the loader node"),

        dict(name="instance_type_monitor", env="SCT_INSTANCE_TYPE_MONITOR", type=str,
             help="AWS image type of the monitor node"),

        dict(name="instance_type_db", env="SCT_INSTANCE_TYPE_DB", type=str,
             help="AWS image type of the db node"),

        dict(name="instance_type_db_oracle", env="SCT_INSTANCE_TYPE_DB_ORACLE", type=str,
             help="AWS image type of the oracle node"),

        dict(name="instance_type_runner", env="SCT_INSTANCE_TYPE_RUNNER", type=str,
             help="instance type of the sct-runner node"),

        dict(name="region_name", env="SCT_REGION_NAME", type=str_or_list_or_eval,
             help="AWS regions to use", appendable=False),

        dict(name="security_group_ids", env="SCT_SECURITY_GROUP_IDS", type=str_or_list,
             help="AWS security groups ids to use"),

        dict(name="use_placement_group", env="SCT_USE_PLACEMENT_GROUP", type=str,
             help="if true, create 'cluster' placement group for test case "
                  "for low-latency network performance achievement"),

        dict(name="subnet_id", env="SCT_SUBNET_ID", type=str_or_list,
             help="AWS subnet ids to use"),

        dict(name="ami_id_db_scylla", env="SCT_AMI_ID_DB_SCYLLA", type=str,
             help="AMS AMI id to use for scylla db node"),

        dict(name="ami_id_loader", env="SCT_AMI_ID_LOADER", type=str,
             help="AMS AMI id to use for loader node"),

        dict(name="ami_id_monitor", env="SCT_AMI_ID_MONITOR", type=str,
             help="AMS AMI id to use for monitor node"),

        dict(name="ami_id_db_cassandra", env="SCT_AMI_ID_DB_CASSANDRA", type=str,
             help="AMS AMI id to use for cassandra node"),

        dict(name="ami_id_db_oracle", env="SCT_AMI_ID_DB_ORACLE", type=str,
             help="AMS AMI id to use for oracle node"),

        dict(name="root_disk_size_db", env="SCT_ROOT_DISK_SIZE_DB", type=int,
             help=""),

        dict(name="root_disk_size_monitor", env="SCT_ROOT_DISK_SIZE_MONITOR", type=int,
             help=""),

        dict(name="root_disk_size_loader", env="SCT_ROOT_DISK_SIZE_LOADER", type=int,
             help=""),

        dict(name="root_disk_size_runner", env="SCT_ROOT_DISK_SIZE_RUNNER", type=int,
             help="root disk size in Gb for sct-runner"),

        dict(name="ami_db_scylla_user", env="SCT_AMI_DB_SCYLLA_USER", type=str,
             help=""),

        dict(name="ami_monitor_user", env="SCT_AMI_MONITOR_USER", type=str,
             help=""),

        dict(name="ami_loader_user", env="SCT_AMI_LOADER_USER", type=str,
             help=""),

        dict(name="ami_db_cassandra_user", env="SCT_AMI_DB_CASSANDRA_USER", type=str,
             help=""),

        dict(name="spot_max_price", env="SCT_SPOT_MAX_PRICE", type=float,
             help="The max percentage of the on demand price we set for spot/fleet instances"),

        dict(name="extra_network_interface", env="SCT_EXTRA_NETWORK_INTERFACE", type=boolean,
             help="if true, create extra network interface on each node"),

        dict(name="aws_instance_profile_name_db", env="SCT_AWS_INSTANCE_PROFILE_NAME_DB", type=str,
             help="This is the name of the instance profile to set on all db instances"),

        dict(name="aws_instance_profile_name_loader", env="SCT_AWS_INSTANCE_PROFILE_NAME_LOADER", type=str,
             help="This is the name of the instance profile to set on all loader instances"),

        dict(name="backup_bucket_backend", env="SCT_BACKUP_BUCKET_BACKEND", type=str,
             help="the backend to be used for backup (e.g., 's3', 'gcs' or 'azure')"),

        dict(name="backup_bucket_location", env="SCT_BACKUP_BUCKET_LOCATION", type=str_or_list,
             help="the bucket name to be used for backup (e.g., 'manager-backup-tests')"),

        dict(name="use_prepared_loaders", env="SCT_USE_PREPARED_LOADERS", type=boolean,
             help="If True, we use prepared VMs for loader (instead of using docker images)"),

        dict(name="scylla_d_overrides_files", env="SCT_SCYLLA_D_OVERRIDES_FILES", type=str_or_list_or_eval,
             help="list of files that should upload to /etc/scylla.d/ directory to override scylla config files"),

        # GCE config options
        dict(name="gce_project", env="SCT_GCE_PROJECT", type=str,
             help="gcp project name to use"),

        dict(name="gce_datacenter", env="SCT_GCE_DATACENTER", type=str_or_list_or_eval,
             help="Supported: us-east1 - means that the zone will be selected automatically or "
                  "you can mention the zone explicitly, for example: us-east1-b",
             appendable=False),

        dict(name="gce_network", env="SCT_GCE_NETWORK", type=str,
             help=""),

        dict(name="gce_image_db", env="SCT_GCE_IMAGE_DB", type=str,
             help=""),

        dict(name="gce_image_monitor", env="SCT_GCE_IMAGE_MONITOR", type=str,
             help=""),

        dict(name="gce_image_loader", env="SCT_GCE_IMAGE_LOADER", type=str,
             help=""),

        dict(name="gce_image_username", env="SCT_GCE_IMAGE_USERNAME", type=str,
             help=""),

        dict(name="gce_instance_type_loader", env="SCT_GCE_INSTANCE_TYPE_LOADER", type=str,
             help=""),

        dict(name="gce_root_disk_type_loader", env="SCT_GCE_ROOT_DISK_TYPE_LOADER", type=str,
             help=""),

        dict(name="gce_n_local_ssd_disk_loader", env="SCT_GCE_N_LOCAL_SSD_DISK_LOADER", type=int,
             help=""),

        dict(name="gce_instance_type_monitor", env="SCT_GCE_INSTANCE_TYPE_MONITOR", type=str,
             help=""),

        dict(name="gce_root_disk_type_monitor", env="SCT_GCE_ROOT_DISK_TYPE_MONITOR", type=str,
             help=""),

        dict(name="gce_n_local_ssd_disk_monitor", env="SCT_GCE_N_LOCAL_SSD_DISK_MONITOR", type=int,
             help=""),

        dict(name="gce_instance_type_db", env="SCT_GCE_INSTANCE_TYPE_DB", type=str,
             help=""),

        dict(name="gce_root_disk_type_db", env="SCT_GCE_ROOT_DISK_TYPE_DB", type=str,
             help=""),

        dict(name="gce_n_local_ssd_disk_db", env="SCT_GCE_N_LOCAL_SSD_DISK_DB", type=int,
             help=""),

        dict(name="gce_pd_standard_disk_size_db", env="SCT_GCE_PD_STANDARD_DISK_SIZE_DB", type=int,
             help=""),

        dict(name="gce_pd_ssd_disk_size_db", env="SCT_GCE_PD_SSD_DISK_SIZE_DB", type=int,
             help=""),

        dict(name="gce_setup_hybrid_raid", env="SCT_GCE_SETUP_HYBRID_RAID", type=boolean,
             help="If True, SCT configures a hybrid RAID of NVMEs and an SSD for scylla's data"),

        dict(name="gce_pd_ssd_disk_size_loader", env="SCT_GCE_PD_SSD_DISK_SIZE_LOADER", type=int,
             help=""),

        dict(name="gce_pd_ssd_disk_size_monitor", env="SCT_GCE_SSD_DISK_SIZE_MONITOR", type=int,
             help=""),

        # azure options
        dict(name="azure_region_name", env="SCT_AZURE_REGION_NAME", type=str_or_list_or_eval,
             help="Supported: eastus ",
             appendable=False),

        dict(name="azure_instance_type_loader", env="SCT_AZURE_INSTANCE_TYPE_LOADER", type=str,
             help=""),

        dict(name="azure_instance_type_monitor", env="SCT_AZURE_INSTANCE_TYPE_MONITOR", type=str,
             help=""),

        dict(name="azure_instance_type_db", env="SCT_AZURE_INSTANCE_TYPE_DB", type=str,
             help=""),

        dict(name="azure_instance_type_db_oracle", env="SCT_AZURE_INSTANCE_TYPE_DB_ORACLE", type=str,
             help=""),

        dict(name="azure_image_db", env="SCT_AZURE_IMAGE_DB", type=str,
             help=""),

        dict(name="azure_image_monitor", env="SCT_AZURE_IMAGE_MONITOR", type=str,
             help=""),

        dict(name="azure_image_loader", env="SCT_AZURE_IMAGE_LOADER", type=str,
             help=""),

        dict(name="azure_image_username", env="SCT_AZURE_IMAGE_USERNAME", type=str,
             help=""),

        # k8s-eks options
        dict(name="eks_service_ipv4_cidr", env="SCT_EKS_SERVICE_IPV4_CIDR", type=str,
             help=""),

        dict(name="eks_vpc_cni_version", env="SCT_EKS_VPC_CNI_VERSION", type=str,
             help=""),

        dict(name="eks_role_arn", env="SCT_EKS_ROLE_ARN", type=str,
             help=""),

        dict(name="eks_cluster_version", env="SCT_EKS_CLUSTER_VERSION", type=str,
             help=""),

        dict(name="eks_nodegroup_role_arn", env="SCT_EKS_NODEGROUP_ROLE_ARN", type=str,
             help=""),

        # k8s-gke options
        dict(name="gke_cluster_version", env="SCT_GKE_CLUSTER_VERSION", type=str,
             help=""),
        dict(name="gke_k8s_release_channel", env="SCT_GKE_K8S_RELEASE_CHANNEL", type=str,
             help="K8S release channel name to be used. Expected values are: "
                  "'rapid', 'regular', 'stable' and '' (static / No channel)."),

        # k8s options
        dict(name="k8s_scylla_utils_docker_image",
             env="SCT_K8S_SCYLLA_UTILS_DOCKER_IMAGE", type=str,
             help=(
                 "Docker image to be used by Scylla operator to tune K8S nodes for performance. "
                 "Used when k8s_enable_performance_tuning' is defined to 'True'. "
                 "If not set then the default from operator will be used.")),

        dict(name="k8s_enable_performance_tuning", env="SCT_K8S_ENABLE_PERFORMANCE_TUNING",
             type=boolean, help="Define whether performance tuning must run or not."),

        dict(name="k8s_deploy_monitoring", env="SCT_K8S_DEPLOY_MONITORING", type=boolean,
             help=""),

        dict(name="k8s_local_volume_provisioner_type", env="SCT_K8S_LOCAL_VOLUME_PROVISIONER_TYPE",
             type=str, choices=("static", "dynamic"),
             help="Defines the type of the K8S local volume provisioner to be deployed. "
                  "It may be either 'static' or 'dynamic'. Details about 'dynamic': "
                  "'dynamic': https://github.com/scylladb/k8s-local-volume-provisioner; "
                  "'static': sdcm/k8s_configs/static-local-volume-provisioner.yaml"),

        dict(name="k8s_scylla_operator_docker_image",
             env="SCT_K8S_SCYLLA_OPERATOR_DOCKER_IMAGE", type=str,
             help="Docker image to be used for installation of scylla operator."),
        dict(name="k8s_scylla_operator_upgrade_docker_image",
             env="SCT_K8S_SCYLLA_OPERATOR_UPGRADE_DOCKER_IMAGE", type=str,
             help="Docker image to be used for upgrade of scylla operator."),

        dict(name="k8s_scylla_operator_helm_repo", env="SCT_K8S_SCYLLA_OPERATOR_HELM_REPO",
             type=str,
             help="Link to the Helm repository where to get 'scylla-operator' charts from."),
        dict(name="k8s_scylla_operator_upgrade_helm_repo", env="SCT_K8S_SCYLLA_OPERATOR_UPGRADE_HELM_REPO",
             type=str,
             help="Link to the Helm repository where to get 'scylla-operator' charts for upgrade."),

        dict(name="k8s_scylla_operator_chart_version",
             env="SCT_K8S_SCYLLA_OPERATOR_CHART_VERSION",
             type=str,
             help=("Version of 'scylla-operator' Helm chart to use. "
                   "If not set then latest one will be used.")),
        dict(name="k8s_scylla_operator_upgrade_chart_version",
             env="SCT_K8S_SCYLLA_OPERATOR_UPGRADE_CHART_VERSION",
             type=str,
             help="Version of 'scylla-operator' Helm chart to use for upgrade."),
        dict(name="k8s_functional_test_dataset",
             env="SCT_K8S_FUNCTIONAL_TEST_DATASET", type=str,
             help="Defines whether dataset uses for pre-fill cluster in functional test. "
                  "Defined in sdcm.utils.sstable.load_inventory. "
                  "Expected values: BIG_SSTABLE_MULTI_COLUMNS_DATA, MULTI_COLUMNS_DATA"),

        dict(name="k8s_scylla_cpu_limit", env="SCT_K8S_SCYLLA_CPU_LIMIT",
             type=str, k8s_multitenancy_supported=True,
             help="The CPU limit that will be set for each Scylla cluster deployed in K8S. "
                  "If not set, then will be autocalculated. Example: '500m' or '2'"),
        dict(name="k8s_scylla_memory_limit", env="SCT_K8S_SCYLLA_MEMORY_LIMIT",
             type=str, k8s_multitenancy_supported=True,
             help="The memory limit that will be set for each Scylla cluster deployed in K8S. "
                  "If not set, then will be autocalculated. Example: '16384Mi'"),

        dict(name="k8s_scylla_cluster_name", env="SCT_K8S_SCYLLA_CLUSTER_NAME", type=str,
             help=""),
        dict(name="k8s_n_scylla_pods_per_cluster", env="K8S_N_SCYLLA_PODS_PER_CLUSTER",
             type=int_or_space_separated_ints,
             help="Number of loader pods per loader cluster."),

        dict(name="k8s_scylla_disk_gi", env="SCT_K8S_SCYLLA_DISK_GI", type=int,
             help=""),

        dict(name="k8s_scylla_disk_class", env="SCT_K8S_SCYLLA_DISK_CLASS", type=str,
             help=""),

        dict(name="k8s_loader_cluster_name", env="SCT_K8S_LOADER_CLUSTER_NAME", type=str,
             help=""),
        dict(name="k8s_n_loader_pods_per_cluster", env="SCT_K8S_N_LOADER_PODS_PER_CLUSTER",
             type=int_or_space_separated_ints,
             help="Number of loader pods per loader cluster."),
        dict(name="k8s_loader_run_type", env="SCT_K8S_LOADER_RUN_TYPE",
             type=str, choices=("static", "dynamic"),
             help="Defines how the loader pods must run. "
                  "It may be either 'static' (default, run stress command on the constantly "
                  "existing idle pod having reserved resources, perf-oriented) or "
                  "'dynamic' (run stress commad in a separate pod as main thread and get logs "
                  "in a searate retryable API call not having resource reservations)."),
        dict(name="k8s_instance_type_auxiliary", env="SCT_K8S_INSTANCE_TYPE_AUXILIARY", type=str,
             help="Instance type for the nodes of the K8S auxiliary/default node pool."),
        dict(name="k8s_instance_type_monitor", env="SCT_K8S_INSTANCE_TYPE_MONITOR", type=str,
             help="Instance type for the nodes of the K8S monitoring node pool."),

        dict(name="mini_k8s_version", env="SCT_MINI_K8S_VERSION", type=str,
             help=""),

        dict(name="k8s_cert_manager_version", env="SCT_K8S_CERT_MANAGER_VERSION", type=str,
             help=""),
        dict(name="k8s_minio_storage_size", env="SCT_K8S_MINIO_STORAGE_SIZE", type=str,
             help=""),
        dict(name="k8s_log_api_calls", env="SCT_K8S_LOG_API_CALLS", type=boolean,
             help="Defines whether the K8S API server logging must be enabled and "
                  "it's logs gathered. Be aware that it may be really huge set of data."),
        dict(name="k8s_tenants_num", env="SCT_TENANTS_NUM", type=int,
             help="Number of Scylla clusters to create in the K8S cluster."),

        dict(name="k8s_enable_tls", env="SCT_K8S_ENABLE_TLS", type=boolean,
             help="Defines whether we enable the scylla operator TLS feature or not."),
        dict(name="k8s_enable_sni", env="SCT_K8S_ENABLE_SNI", type=boolean,
             help="Defines whether we install SNI and use it or not (serverless feature)."),
        dict(name="k8s_enable_alternator", env="SCT_K8S_ENABLE_ALTERNATOR", type=boolean,
             help="Defines whether we enable the alternator feature using scylla-operator or not."),

        dict(name="k8s_connection_bundle_file", env="SCT_K8S_CONNECTION_BUNDLE_FILE", type=_file,
             help="Serverless configuration bundle file", k8s_multitenancy_supported=True),

        # NOTE: following 'k8s_db_node_service_type', 'k8s_db_node_to_node_broadcast_ip_type' and
        #       'k8s_db_node_to_client_broadcast_ip_type' options are supported only starting with
        #       the 'v1.11.0-rc.0' scylla-operator version.
        dict(name="k8s_db_node_service_type", env="SCT_K8S_DB_NODE_SERVICE_TYPE",
             type=str, choices=("", "ClusterIP", "Headless", "LoadBalancer"),
             help="Defines the type of the K8S 'Service' objects type used for ScyllaDB pods. "
                  "Empty value means 'do not set and allow scylla-operator to choose'."),
        dict(name="k8s_db_node_to_node_broadcast_ip_type", env="SCT_K8S_DB_NODE_TO_NODE_BROADCAST_IP_TYPE",
             type=str, choices=("", "ServiceClusterIP", "PodIP", "ServiceLoadBalancerIngress"),
             help="Defines the source of the IP address to be used for the 'broadcast_address' config "
                  "option in the 'scylla.yaml' files. "
                  "Empty value means 'do not set and allow scylla-operator to choose'."),
        dict(name="k8s_db_node_to_client_broadcast_ip_type", env="SCT_K8S_DB_NODE_TO_CLIENT_BROADCAST_IP_TYPE",
             type=str, choices=("", "ServiceClusterIP", "PodIP", "ServiceLoadBalancerIngress"),
             help="Defines the source of the IP address to be used for the 'broadcast_rpc_address' config "
                  "option in the 'scylla.yaml' files. "
                  "Empty value means 'do not set and allow scylla-operator to choose'."),

        dict(name="k8s_use_chaos_mesh", env="SCT_K8S_USE_CHAOS_MESH", type=boolean,
             help="""enables chaos-mesh for k8s testing"""),

        dict(name="k8s_n_auxiliary_nodes", env="SCT_K8S_N_AUXILIARY_NODES", type=int,
             help="Number of nodes in auxiliary pool"),
        dict(name="k8s_n_monitor_nodes", env="SCT_K8S_N_MONITOR_NODES", type=int,
             help="Number of nodes in monitoring pool that will be used for scylla-operator's "
                  "deployed monitoring pods."),

        # docker config options
        dict(name="mgmt_docker_image", env="SCT_MGMT_DOCKER_IMAGE", type=str,
             help="Scylla manager docker image, i.e. 'scylladb/scylla-manager:2.2.1' "),

        dict(name="docker_image", env="SCT_DOCKER_IMAGE", type=str,
             help="Scylla docker image repo, i.e. 'scylladb/scylla', if omitted is calculated from scylla_version"),

        dict(name="docker_network", env="SCT_DOCKER_NETWORK", type=str,
             help="local docker network to use, if there's need to have db cluster connect to other services running in docker"),

        # baremetal config options

        dict(name="s3_baremetal_config", env="SCT_S3_BAREMETAL_CONFIG", type=str,
             help=""),

        dict(name="db_nodes_private_ip", env="SCT_DB_NODES_PRIVATE_IP", type=str_or_list_or_eval,
             help=""),

        dict(name="db_nodes_public_ip", env="SCT_DB_NODES_PUBLIC_IP", type=str_or_list_or_eval,
             help=""),

        dict(name="loaders_private_ip", env="SCT_LOADERS_PRIVATE_IP", type=str_or_list_or_eval,
             help=""),

        dict(name="loaders_public_ip", env="SCT_LOADERS_PUBLIC_IP", type=str_or_list_or_eval,
             help=""),

        dict(name="monitor_nodes_private_ip", env="SCT_MONITOR_NODES_PRIVATE_IP", type=str_or_list_or_eval,
             help=""),

        dict(name="monitor_nodes_public_ip", env="SCT_MONITOR_NODES_PUBLIC_IP", type=str_or_list_or_eval,
             help=""),

        # test specific config parameters

        # GrowClusterTest
        dict(name="cassandra_stress_population_size", env="SCT_CASSANDRA_STRESS_POPULATION_SIZE", type=int,
             help=""),
        dict(name="cassandra_stress_threads", env="SCT_CASSANDRA_STRESS_THREADS", type=int,
             help=""),
        dict(name="add_node_cnt", env="SCT_ADD_NODE_CNT", type=int,
             help=""),

        # LongevityTest
        dict(name="stress_multiplier", env="SCT_STRESS_MULTIPLIER", type=int,
             help="Number of cassandra-stress processes"),
        dict(name="stress_multiplier_w", env="SCT_STRESS_MULTIPLIER_W", type=int,
             help="Number of cassandra-stress processes for write workload"),
        dict(name="stress_multiplier_r", env="SCT_STRESS_MULTIPLIER_R", type=int,
             help="Number of cassandra-stress processes for read workload"),
        dict(name="stress_multiplier_m", env="SCT_STRESS_MULTIPLIER_M", type=int,
             help="Number of cassandra-stress processes for mixed workload"),
        dict(name="run_fullscan", env="SCT_RUN_FULLSCAN", type=list,
             help=""),
        dict(name="run_full_partition_scan", env="SCT_run_full_partition_scan", type=str,
             help="Runs a background thread that issues reversed-queries on a table random partition by an interval"),
        dict(name="run_tombstone_gc_verification", env="SCT_RUN_TOMBSTONE_GC_VERIFICATION", type=str,
             help="Runs a background thread that verifies Tombstones GC on a table by an interval"),
        dict(name="keyspace_num", env="SCT_KEYSPACE_NUM", type=int,
             help=""),
        dict(name="round_robin", env="SCT_ROUND_ROBIN",
             type=boolean, k8s_multitenancy_supported=True,
             help=""),
        dict(name="batch_size", env="SCT_BATCH_SIZE", type=int,
             help=""),
        dict(name="pre_create_schema", env="SCT_PRE_CREATE_SCHEMA", type=boolean,
             help=""),
        dict(name="pre_create_keyspace", env="SCT_PRE_CREATE_KEYSPACE", type=str_or_list,
             help="Command to create keysapce to be pre-create before running workload"),

        dict(name="post_prepare_cql_cmds", env="SCT_POST_PREPARE_CQL_CMDS", type=str_or_list,
             help="CQL Commands to run after prepare stage finished (relevant only to longevity_test.py)"),

        dict(name="prepare_wait_no_compactions_timeout", env="SCT_PREPARE_WAIT_NO_COMPACTIONS_TIMEOUT", type=int,
             help="At the end of prepare stage, run major compaction and wait for this time (in minutes) for compaction to finish. "
                  "(relevant only to longevity_test.py)"
                  ", Should be use only for when facing issue like compaction is affect the test or load"),

        dict(name="compaction_strategy", env="SCT_COMPACTION_STRATEGY", type=str,
             help="Choose a specific compaction strategy to pre-create schema with."),

        dict(name="sstable_size", env="SSTABLE_SIZE", type=int,
             help="Configure sstable size for the usage of pre-create-schema mode"),

        dict(name="cluster_health_check", env="SCT_CLUSTER_HEALTH_CHECK", type=boolean,
             help="When true, start cluster health checker for all nodes"),

        dict(name="data_validation", env="SCT_DATA_VALIDATION", type=str,
             help="""A group of sub-parameters: validate_partitions, table_name, primary_key_column,
                   partition_range_with_data_validation, max_partitions_in_test_table.
                   1. validate_partitions - when true, validating the same number of rows-per-partition before/after a Nemesis.
                   2. table_name - table name to check for the validate_partitions check.
                   3. primary_key_column - primary key of the table to check for the validate_partitions check
                   4. partition_range_with_data_validation - Relevant for scylla-bench. A range (min - max) of PK values
                       for partitions to be validated by reads and not to be deleted during test. Example: 0-250.
                   5. max_partitions_in_test_table - Relevant for scylla-bench. Max partition keys (partition-count)
                       in the scylla_bench.test table.
                  """),

        dict(name="stress_read_cmd", env="SCT_STRESS_READ_CMD",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="prepare_verify_cmd", env="SCT_PREPARE_VERIFY_CMD",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""cassandra-stress commands.
            You can specify everything but the -node parameter, which is going to
            be provided by the test suite infrastructure.
            multiple commands can passed as a list"""),

        dict(name="user_profile_table_count", env="SCT_USER_PROFILE_TABLE_COUNT", type=int,
             help="""number of tables to create for template user c-s"""),

        # MgmtCliTest
        dict(name="scylla_mgmt_upgrade_to_repo", env="SCT_SCYLLA_MGMT_UPGRADE_TO_REPO", type=str,
             help="Url to the repo of scylla manager version to upgrade to for management tests"),

        dict(name="mgmt_restore_extra_params", env="SCT_MGMT_RESTORE_EXTRA_PARAMS", type=str,
             help="Manager restore operation extra parameters: batch-size, parallel, etc."
                  "For example, `--batch-size 2 --parallel 1`. Provided string appends the restore cmd"),

        dict(name="mgmt_agent_backup_config", env="SCT_MGMT_AGENT_BACKUP_CONFIG", type=dict_or_str_or_pydantic,
             help="Manager agent backup general configuration: checkers, transfers, low_level_retries. "
                  "For example, {'checkers': 100, 'transfers': 2, 'low_level_retries': 20}"),

        dict(name="mgmt_reuse_backup_snapshot_name", env="SCT_MGMT_REUSE_BACKUP_SNAPSHOT_NAME", type=str,
             help="Name of backup snapshot to use in Manager restore benchmark test, for example, 500gb_2t_ics. "
                  "The name provides the info about dataset size (500gb), tables number (2) and compaction (ICS)"),

        dict(name="mgmt_skip_post_restore_stress_read", env="SCT_MGMT_SKIP_POST_RESTORE_STRESS_READ", type=boolean,
             help="Skip post-restore c-s verification read in the Manager restore benchmark tests"),

        dict(name="mgmt_nodetool_refresh_flags",
             env="SCT_MGMT_NODETOOL_REFRESH_FLAGS", type=str,
             help="Nodetool refresh extra options like --load-and-stream or --primary-replica-only"),

        dict(name="mgmt_prepare_snapshot_size",
             env="SCT_MGMT_PREPARE_SNAPSHOT_SIZE", type=int,
             help="Size of backup snapshot in Gb to be prepared to be prepared for backup"),

        # PerformanceRegressionTest

        dict(name="stress_cmd_w", env="SCT_STRESS_CMD_W",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="stress_cmd_r", env="SCT_STRESS_CMD_R",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="stress_cmd_m", env="SCT_STRESS_CMD_M",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="stress_cmd_cache_warmup", env="SCT_STRESS_CMD_CACHE_WARM_UP",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""cassandra-stress commands for warm-up before read workload.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="prepare_write_cmd", env="SCT_PREPARE_WRITE_CMD",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="stress_cmd_no_mv", env="SCT_STRESS_CMD_NO_MV", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="stress_cmd_no_mv_profile", env="SCT_STRESS_CMD_NO_MV_PROFILE", type=str,
             help=""),

        dict(name="perf_extra_jobs_to_compare", env="SCT_PERF_EXTRA_JOBS_TO_COMPARE", type=str_or_list_or_eval,
             help="jobs to compare performance results with, for example if running in staging, "
                  "we still can compare with official jobs"),

        # PerformanceRegressionUserProfilesTest
        dict(name="cs_user_profiles", env="SCT_CS_USER_PROFILES", type=str_or_list,
             help="cassandra-stress user-profiles list. Executed in test step"),
        dict(name="prepare_cs_user_profiles", env="SCT_PREPARE_CS_USER_PROFILES", type=str_or_list,
             help="cassandra-stress user-profiles list. Executed in prepare step"),
        dict(name="cs_duration", env="SCT_CS_DURATION", type=str,
             help=""),
        dict(name="cs_debug", env="SCT_CS_DEBUG", type=boolean,
             help="enable debug for cassandra-stress"),

        dict(name="stress_cmd_mv", env="SCT_STRESS_CMD_MV", type=str_or_list,
             help="""cassandra-stress commands.
            You can specify everything but the -node parameter, which is going to
            be provided by the test suite infrastructure.
            multiple commands can passed as a list"""),

        dict(name="prepare_stress_cmd", env="SCT_PREPARE_STRESS_CMD", type=str_or_list,
             help="""cassandra-stress commands.
            You can specify everything but the -node parameter, which is going to
            be provided by the test suite infrastructure.
            multiple commands can passed as a list"""),

        dict(name="perf_gradual_threads", env="SCT_PERF_GRADUAL_THREADS", type=dict,
             help="Threads amount of c-s load for gradual performance test per sub-test. "
                  "Example: {'read': 100, 'write': 200, 'mixed': 300}"),
        dict(name="perf_gradual_throttle_steps", env="SCT_PERF_GRADUAL_THROTTLE_STEPS", type=dict,
             help="Used for gradual performance test. Define throttle for load step in ops. Example: {'read': ['100000', '150000'], 'mixed': ['300']}"),

        # RefreshTest
        dict(name="skip_download", env="SCT_SKIP_DOWNLOAD", type=boolean,
             help=""),
        dict(name="sstable_file", env="SCT_SSTABLE_FILE", type=str,
             help=""),
        dict(name="sstable_url", env="SCT_SSTABLE_URL", type=str,
             help=""),
        dict(name="sstable_md5", env="SCT_SSTABLE_MD5", type=str,
             help=""),
        dict(name="flush_times", env="SCT_FLUSH_TIMES", type=int,
             help=""),
        dict(name="flush_period", env="SCT_FLUSH_PERIOD", type=int,
             help=""),

        # UpgradeTest
        dict(name="new_scylla_repo", env="SCT_NEW_SCYLLA_REPO", type=str,
             help=""),

        dict(name="new_version", env="SCT_NEW_VERSION", type=str,
             help="Assign new upgrade version, use it to upgrade to specific minor release. eg: 3.0.1"),

        dict(name="target_upgrade_version", env="SCT_TARGET_UPGRADE_VERSION", type=str,
             help="Assign target upgrade version, use for decide if the truncate entries test should be run. "
                  "This test should be performed in case the target upgrade version >= 3.1"),

        dict(name="disable_raft", env="SCT_DISABLE_RAFT", type=boolean,
             help="As for now, raft will be enable by default in all [upgrade] tests, so this flag will allow us"
                  "to still run [upgrade] test without raft enabled (or disabling raft), so we will have better"
                  "coverage"),

        dict(name="enable_tablets_on_upgrade", env="SCT_ENABLE_TABLETS_ON_UPGRADE", type=boolean,
             help="By default, the tablets feature is disabled. With this parameter, created for the upgrade test,"
                  "the tablets feature will only be enabled after the upgrade"),

        dict(name="upgrade_node_packages", env="SCT_UPGRADE_NODE_PACKAGES", type=str,
             help=""),

        dict(name="upgrade_node_system", env="SCT_UPGRADE_NODE_SYSTEM", type=boolean,
             help="Upgrade system packages on nodes before upgrading Scylla. Enabled by default"),

        dict(name="stress_cmd_1", env="SCT_STRESS_CMD_1", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="stress_cmd_complex_prepare", env="SCT_STRESS_CMD_COMPLEX_PREPARE", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="prepare_write_stress", env="SCT_PREPARE_WRITE_STRESS", type=str_or_list,

             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="stress_cmd_read_10m", env="SCT_STRESS_CMD_READ_10M", type=str_or_list,

             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="stress_cmd_read_cl_one", env="SCT_STRESS_CMD_READ_CL_ONE", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure."""),

        dict(name="stress_cmd_read_60m", env="SCT_STRESS_CMD_READ_60M", type=str_or_list,

             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="stress_cmd_complex_verify_read", env="SCT_STRESS_CMD_COMPLEX_VERIFY_READ", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="stress_cmd_complex_verify_more", env="SCT_STRESS_CMD_COMPLEX_VERIFY_MORE", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="write_stress_during_entire_test", env="SCT_WRITE_STRESS_DURING_ENTIRE_TEST", type=str_or_list,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="verify_data_after_entire_test", env="SCT_VERIFY_DATA_AFTER_ENTIRE_TEST", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure."""),

        dict(name="stress_cmd_read_cl_quorum", env="SCT_STRESS_CMD_READ_CL_QUORUM", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="verify_stress_after_cluster_upgrade", env="SCT_VERIFY_STRESS_AFTER_CLUSTER_UPGRADE",
             type=str_or_list,
             help="""cassandra-stress commands.
            You can specify everything but the -node parameter, which is going to
            be provided by the test suite infrastructure.
            multiple commands can passed as a list"""),

        dict(name="stress_cmd_complex_verify_delete", env="SCT_STRESS_CMD_COMPLEX_VERIFY_DELETE",
             type=str_or_list,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="scylla_encryption_options", env="SCT_SCYLLA_ENCRYPTION_OPTIONS", type=str_or_list,
             help="options will be used for enable encryption at-rest for tables"),

        dict(name="kms_key_rotation_interval", env="SCT_KMS_KEY_ROTATION_INTERVAL", type=int,
             help="The time interval in minutes which gets waited before the KMS key rotation happens."
                  " Applied when the AWS KMS service is configured to be used."),

        dict(name="enterprise_disable_kms", env="SCT_ENTERPRISE_DISABLE_KMS", type=boolean,
             help="An escape hatch to disable KMS for enterprise run, when needed, "
                  "we enable kms by default since if we use scylla 2023.1.3 and up"),

        dict(name="logs_transport", env="SCT_LOGS_TRANSPORT", type=str,
             help="How to transport logs: syslog-ng, ssh or docker", choices=("ssh", "docker", "syslog-ng")),

        dict(name="collect_logs", env="SCT_COLLECT_LOGS", type=boolean,
             help="Collect logs from instances and sct runner"),

        dict(name="execute_post_behavior", env="SCT_EXECUTE_POST_BEHAVIOR", type=boolean,
             help="Run post behavior actions in sct teardown step"),

        dict(name="post_behavior_db_nodes", env="SCT_POST_BEHAVIOR_DB_NODES", type=str,
             help="""
                Failure/post test behavior, i.e. what to do with the db cloud instances at the end of the test.

                'destroy' - Destroy instances and credentials (default)
                'keep' - Keep instances running and leave credentials alone
                'keep-on-failure' - Keep instances if testrun failed
             """,
             choices=("keep", "keep-on-failure", "destroy")),

        dict(name="post_behavior_loader_nodes", env="SCT_POST_BEHAVIOR_LOADER_NODES", type=str,
             help="""
                Failure/post test behavior, i.e. what to do with the loader cloud instances at the end of the test.

                'destroy' - Destroy instances and credentials (default)
                'keep' - Keep instances running and leave credentials alone
                'keep-on-failure' - Keep instances if testrun failed
             """,
             choices=("keep", "keep-on-failure", "destroy")),

        dict(name="post_behavior_monitor_nodes", env="SCT_POST_BEHAVIOR_MONITOR_NODES", type=str,
             help="""
                Failure/post test behavior, i.e. what to do with the monitor cloud instances at the end of the test.

                'destroy' - Destroy instances and credentials (default)
                'keep' - Keep instances running and leave credentials alone
                'keep-on-failure' - Keep instances if testrun failed
             """,
             choices=("keep", "keep-on-failure", "destroy")),

        dict(name="post_behavior_k8s_cluster", env="SCT_POST_BEHAVIOR_K8S_CLUSTER", type=str,
             help="""
            Failure/post test behavior, i.e. what to do with the k8s cluster at the end of the test.

            'destroy' - Destroy k8s cluster and credentials (default)
            'keep' - Keep k8s cluster running and leave credentials alone
            'keep-on-failure' - Keep k8s cluster if testrun failed
         """,
             choices=("keep", "keep-on-failure", "destroy")),

        dict(name="internode_compression", env="SCT_INTERNODE_COMPRESSION", type=str,
             help="scylla option: internode_compression"),
        dict(name="internode_encryption", env="SCT_INTERNODE_ENCRYPTION", type=str,
             help="scylla sub option of server_encryption_options: internode_encryption"),

        dict(name="jmx_heap_memory", env="SCT_JMX_HEAP_MEMORY", type=int,
             help="The total size of the memory allocated to JMX. Values in MB, so for 1GB enter 1024(MB)"),

        dict(name="store_perf_results", env="SCT_STORE_PERF_RESULTS", type=boolean,
             help="""A flag that indicates whether or not to gather the prometheus stats at the end of the run.
                Intended to be used in performance testing"""),

        dict(name="append_scylla_setup_args", env="SCT_APPEND_SCYLLA_SETUP_ARGS", type=str,
             help="More arguments to append to scylla_setup command line"),

        dict(name="use_preinstalled_scylla", env="SCT_USE_PREINSTALLED_SCYLLA", type=boolean,
             help="Don't install/update ScyllaDB on DB nodes"),

        dict(name="force_run_iotune", env="SCT_FORCE_RUN_IOTUNE", type=boolean,
             help="Force running iotune on the DB nodes, regdless if image has predefined values"),

        dict(name="stress_cdclog_reader_cmd", env="SCT_STRESS_CDCLOG_READER_CMD",
             type=str,
             help="""cdc-stressor command to read cdc_log table.
                    You can specify everything but the -node , -keyspace, -table, parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="store_cdclog_reader_stats_in_es", env="SCT_STORE_CDCLOG_READER_STATS_IN_ES",
             type=boolean,
             help="""Add cdclog reader stats to ES for future performance result calculating"""),
        dict(name="stop_test_on_stress_failure", env="SCT_STOP_TEST_ON_STRESS_FAILURE",
             type=boolean,
             help="""If set to True the test will be stopped immediately when stress command failed.
                     When set to False the test will continue to run even when there are errors in the
                     stress process"""),
        dict(name="stress_cdc_log_reader_batching_enable", env="SCT_STRESS_CDC_LOG_READER_BATCHING_ENABLE",
             type=boolean,
             help="""retrieving data from multiple streams in one poll"""),

        dict(name="use_legacy_cluster_init", env="SCT_USE_LEGACY_CLUSTER_INIT", type=boolean,
             help="""Use legacy cluster initialization with autobootsrap disabled and parallel node setup"""),
        dict(name="availability_zone", env="SCT_AVAILABILITY_ZONE",
             type=str,
             help="""Availability zone to use. Specify multiple (comma separated) to deploy resources to multi az (works on AWS).
                  "Same for multi-region scenario."""),
        dict(name="aws_fallback_to_next_availability_zone", env="SCT_AWS_FALLBACK_TO_NEXT_AVAILABILITY_ZONE",
             type=boolean,
             help="""Try all availability zones one by one in order to maximize the chances of getting
                   the requested instance capacity."""),

        dict(name="num_nodes_to_rollback", env="SCT_NUM_NODES_TO_ROLLBACK",
             type=str,
             help="Number of nodes to upgrade and rollback in test_generic_cluster_upgrade"),

        dict(name="upgrade_sstables", env="SCT_UPGRADE_SSTABLES",
             type=boolean,
             help="Whether to upgrade sstables as part of upgrade_node or not"),

        dict(name="stress_before_upgrade", env="SCT_STRESS_BEFORE_UPGRADE",
             type=str,
             help="Stress command to be run before upgrade (preapre stage)"),

        dict(name="stress_during_entire_upgrade", env="SCT_STRESS_DURING_ENTIRE_UPGRADE",
             type=str,
             help="Stress command to be run during the upgrade - user should take care for suitable duration"),

        dict(name="stress_after_cluster_upgrade", env="SCT_STRESS_AFTER_CLUSTER_UPGRADE",
             type=str,
             help="Stress command to be run after full upgrade - usually used to read the dataset for verification"),

        # Jepsen test.
        dict(name="jepsen_scylla_repo", env="SCT_JEPSEN_SCYLLA_REPO", type=str,
             help="Link to the git repository with Jepsen Scylla tests"),
        dict(name="jepsen_test_cmd", env="SCT_JEPSEN_TEST_CMD", type=str_or_list,
             help="Jepsen test command (e.g., 'test-all')"),
        dict(name="jepsen_test_count", env="SCT_JEPSEN_TEST_COUNT", type=int,
             help="possible number of reruns of single Jepsen test command"),
        dict(name="jepsen_test_run_policy", env="SCT_JEPSEN_TEST_RUN_POLICY", type=str,
             help="""
                Jepsen test run policy (i.e., what we want to consider as passed for a single test)

                'most' - most test runs are passed
                'any'  - one pass is enough
                'all'  - all test runs should pass
             """,
             choices=("most", "any", "all")),

        dict(name="max_events_severities", env="SCT_MAX_EVENTS_SEVERITIES", type=str_or_list,
             help="Limit severity level for event types"),

        dict(name="scylla_rsyslog_setup", env="SCT_SCYLLA_RSYSLOG_SETUP", type=boolean,
             help="Configure rsyslog on Scylla nodes to send logs to monitoring nodes"),

        dict(name="events_limit_in_email", env="SCT_EVENTS_LIMIT_IN_EMAIL", type=int,
             help="Limit number events in email reports"),

        dict(name="data_volume_disk_num", env="SCT_DATA_VOLUME_DISK_NUM",
             type=int,
             help="""Number of additional data volumes attached to instances
             if data_volume_disk_num > 0, then data volumes (ebs on aws) will be
             used for scylla data directory"""),
        dict(name="data_volume_disk_type", env="SCT_DATA_VOLUME_DISK_TYPE",
             type=str,
             help="Type of addtitional volumes: gp2|gp3|io2|io3"),

        dict(name="data_volume_disk_size", env="SCT_DATA_VOLUME_DISK_SIZE",
             type=int,
             help="Size of additional volume in GB"),

        dict(name="data_volume_disk_iops", env="SCT_DATA_VOLUME_DISK_IOPS",
             type=int,
             help="Number of iops for ebs type io2|io3|gp3"),
        dict(name="data_volume_disk_throughput", env="SCT_DATA_VOLUME_DISK_THROUGHPUT",
             type=int,
             help="Throughput in MiB/sec for ebs type gp3. Min is 125. Max is 1000."),
        dict(name="run_db_node_benchmarks", env="SCT_RUN_DB_NODE_BENCHMARKS",
             type=boolean,
             help="Flag for running db node benchmarks before the tests"),
        dict(name="nemesis_selector", env="SCT_NEMESIS_SELECTOR",
             type=str_or_list, k8s_multitenancy_supported=True,
             help="""nemesis_selector gets a list of "nemesis properties" and filters IN all the nemesis that has
             ALL the properties in that list which are set to true (the intersection of all properties).
             (In other words filters out all nemesis that doesn't ONE of these properties set to true)
             IMPORTANT: If a property doesn't exist, ALL the nemesis will be included."""),
        dict(name="nemesis_exclude_disabled", env="SCT_NEMESIS_EXCLUDE_DISABLED",
             type=boolean, k8s_multitenancy_supported=True,
             help="""nemesis_exclude_disabled determines whether 'disabled' nemeses are filtered out from list
             or are allowed to be used. This allows to easily disable too 'risky' or 'extreme' nemeses by default,
             for all longevities. For example: it is unwanted to run the ToggleGcModeMonkey in standard longevities
             that runs a stress with data validation."""),

        dict(name="nemesis_multiply_factor", env="SCT_NEMESIS_MULTIPLY_FACTOR",
             type=int, k8s_multitenancy_supported=True,
             help="Multiply the list of nemesis to execute by the specified factor"),

        dict(name="nemesis_double_load_during_grow_shrink_duration", env="SCT_NEMESIS_DOUBLE_LOAD_DURING_GROW_SHRINK_DURATION", type=int,
             help="After growing (and before shrink) in GrowShrinkCluster nemesis it will double the load for provided duration."),

        dict(name="raid_level", env="SCT_RAID_LEVEL",
             type=int,
             help="Number of of raid level: 0 - RAID0, 5 - RAID5"),

        dict(name="bare_loaders", env="SCT_BARE_LOADERS", type=boolean,
             help="Don't install anything but node_exporter to the loaders during cluster setup"),
        dict(name="stress_image", env="SCT_STRESS_IMAGE", type=dict_or_str,
             help="Dict of the images to use for the stress tools"),

        dict(name="scylla_network_config", env="SCT_SCYLLA_NETWORK_CONFIG", type=list,
             help="""Configure Scylla networking with single or multiple NIC/IP combinations.
                  It must be defined for listen_address and rpc_address. For each address mandatory parameters are:
                  - address: listen_address/rpc_address/broadcast_rpc_address/broadcast_address/test_communication
                  - ip_type: ipv4 or ipv6
                  - public: false or true
                  - nic: number of NIC. 0, 1
                  Supported for AWS only meanwhile"""),

        dict(name="enable_argus", env="SCT_ENABLE_ARGUS", type=boolean,
             help="Control reporting to argus"),

        dict(name="cs_populating_distribution", env="SCT_CS_POPULATING_DISTRIBUTION", type=str,
             help="""set c-s parameter '-pop' with gauss/uniform distribution for
             performance gradual throughtput grow tests"""),

        dict(name="latte_schema_parameters", env="SCT_LATTE_SCHEMA_PARAMETERS", type=dict,
             help="""Optional. Allows to pass through custom rune script parameters to the 'latte schema' command."""),

        dict(name="num_loaders_step", env="SCT_NUM_LOADERS_STEP", type=int,
             help="Number of loaders which should be added per step"),
        dict(name="stress_threads_start_num", env="SCT_STRESS_THREADS_START_NUM", type=int,
             help="Number of threads for c-s command"),
        dict(name="num_threads_step", env="SCT_NUM_THREADS_STEP", type=int,
             help="Number of threads which should be added on per step"),
        dict(name="stress_step_duration", env="SCT_STRESS_STEP_DURATION", type=str,
             help="Duration of time for stress round"),
        dict(name="max_deviation", env="SCT_MAX_DEVIATION", type=float,
             help="""Max relative difference between best and current throughput,
             if current throughput larger then best on max_rel_diff, it become new best one"""),
        dict(name="n_stress_process", env="SCT_N_STRESS_PROCESS", type=int,
             help="""Number of stress processes per loader"""),
        dict(name="stress_process_step", env="SCT_STRESS_PROCESS_STEP", type=int,
             help="""add/remove num of process on each round"""),
        dict(name="use_hdr_cs_histogram", env="SCT_USE_HDR_CS_HISTOGRAM", type=boolean,
             help="""Enable hdr histogram logging for cs"""),

        dict(name="stop_on_hw_perf_failure", env="SCT_STOP_ON_HW_PERF_FAILURE", type=boolean,
             help="""Stop sct performance test if hardware performance test failed

                    Hardware performance tests runs on each node with sysbench and cassandra-fio tools.
                    Results stored in ES. HW perf tests run during cluster setups and not affect
                    SCT Performance tests. Results calculated as average among all results for certain
                    instance type or among all nodes during single run.
                    if results for a single node is not in margin 0.01 of
                    average result for all nodes, hw test considered as Failed.
                    If stop_on_hw_perf_failure is True, then sct performance test will be terminated
                       after hw perf tests detect node with hw results not in margin with average
                    If stop_on_hw_perf_failure is False, then sct performance test will be run
                       even after hw perf tests detect node with hw results not in margin with average"""),
        dict(name="custom_es_index", env="SCT_CUSTOM_ES_INDEX", type=str,
             help="""Use custom ES index for storing test results"""),

        dict(name="simulated_regions", env="SCT_SIMULATED_REGIONS", type=int, choices=[0, 2, 3, 4, 5],
             help="""Defines how many regions must be simulated on the Scylla config side. If set then
             nodes will be provisioned only using the very first real region defined in the configuration."""),
        dict(name="simulated_racks", env="SCT_SIMULATED_RACKS", type=int,
             help="""Forces GossipingPropertyFileSnitch (regardless `endpoint_snitch`) to simulate racks.
             Provide number of racks to simulate."""),

        dict(name="use_dns_names", env="SCT_USE_DNS_NAMES", type=boolean,
             help="""Use dns names instead of ip addresses for nodes in cluster"""),

        dict(name="validate_large_collections", env="SCT_VALIDATE_LARGE_COLLECTIONS", type=boolean,
             help="Enable validation for large cells in system table and logs"),

        dict(name="run_commit_log_check_thread", env="SCT_RUN_COMMIT_LOG_CHECK_THREAD", type=boolean,
             help="""Run commit log check thread if commitlog_use_hard_size_limit is True"""),

        dict(name="teardown_validators", env="SCT_TEARDOWN_VALIDATORS", type=dict_or_str,
             help="""Configuration for additional validations executed after the test"""),

        dict(name="use_capacity_reservation", env="SCT_USE_CAPACITY_RESERVATION", type=boolean,
             help="""reserves instances capacity for whole duration of the test run (AWS only).
             Fallbacks to next availabilit zone if capacity is not available"""),

        dict(name="bisect_start_date", env="SCT_BISECT_START_DATE", type=str,
             help="""Scylla build date from which bisecting should start.
              Setting this date enables bisection. Format: YYYY-MM-DD"""),

        dict(name="bisect_end_date", env="SCT_BISECT_END_DATE", type=str,
             help="""Scylla build date until which bisecting should run. Format: YYYY-MM-DD"""),

        dict(name="kafka_backend", env="SCT_KAFKA_BACKEND", type=str,
             help="Enable validation for large cells in system table and logs",
             choices=(None, "localstack", "vm", "msk")),

        dict(name="kafka_connectors", env="SCT_KAFKA_CONNECTORS", type=str_or_list_or_eval,
             help="configuration for setup up kafka connectors"),

        dict(name="run_scylla_doctor", env="SCT_RUN_SCYLLA_DOCTOR", type=boolean,
             help="Run scylla-doctor in artifact tests"),

        dict(name="skip_test_stages", env="SCT_SKIP_TEST_STAGES", type=dict_or_str,
             help="""Skip selected stages of a test scenario"""),

        dict(name="use_zero_nodes", env="SCT_USE_ZERO_NODES", type=boolean,
             help="If True, enable support in sct of zero nodes(configuration, nemesis)"),

        dict(name="n_db_zero_token_nodes", env="SCT_N_DB_ZERO_TOKEN_NODES", type=int_or_space_separated_ints,
             help="""Number of zero token nodes in cluster. Value should be set as "0 1 1"
               for multidc configuration in same manner as 'n_db_nodes' and should be equal
               number of regions"""),

        dict(name="zero_token_instance_type_db", env="SCT_ZERO_TOKEN_INSTANCE_TYPE_DB", type=str,
             help="""Instance type for zero token node"""),

        dict(name="sct_aws_account_id", env="SCT_AWS_ACCOUNT_ID", type=str,
             help="AWS account id on behalf of which the test is run"),

        dict(name="latency_decorator_error_thresholds", env="SCT_LATENCY_DECORATOR_ERROR_THRESHOLDS", type=dict_or_str,
             help="Error thresholds for latency decorator."
                  " Defined by dict: {<write, read, mixed>: {<default|nemesis_name>:{<metric_name>: {<rule>: <value>}}}"),

        dict(name="workload_name", env="SCT_WORKLOAD_NAME", type=str,
             help="Workload name, can be: write|read|mixed|unset."
                  "Used for e.g. latency_calculator_decorator (use with 'use_hdr_cs_histogram' set to true)."
                  "If unset, workload is taken from test name."),
    ]

    required_params = ['cluster_backend', 'test_duration', 'n_db_nodes', 'n_loaders', 'use_preinstalled_scylla',
                       'user_credentials_path', 'root_disk_size_db', "root_disk_size_monitor", 'root_disk_size_loader']

    # those can be added to a json scheme to validate / or write the validation code for it to be a bit clearer output
    backend_required_params = {
        'aws': ['user_prefix', "instance_type_loader", "instance_type_monitor", "instance_type_db",
                "region_name", "ami_id_db_scylla", "ami_id_loader",
                "ami_id_monitor", "aws_root_disk_name_monitor", "ami_db_scylla_user",
                "ami_monitor_user", "scylla_network_config"],

        'gce': ['user_prefix', 'gce_network', 'gce_image_db', 'gce_image_username', 'gce_instance_type_db',
                'gce_root_disk_type_db',  'gce_n_local_ssd_disk_db',
                'gce_instance_type_loader', 'gce_root_disk_type_loader', 'gce_n_local_ssd_disk_loader',
                'gce_instance_type_monitor', 'gce_root_disk_type_monitor',
                'gce_n_local_ssd_disk_monitor', 'gce_datacenter'],

        'azure': ['user_prefix', 'azure_image_db', 'azure_image_username', 'azure_instance_type_db',
                  'azure_root_disk_type_db', 'azure_n_local_ssd_disk_db',
                  'azure_instance_type_loader', 'azure_root_disk_type_loader', 'azure_n_local_ssd_disk_loader',
                  'azure_instance_type_monitor', 'azure_n_local_ssd_disk_monitor', 'azure_region_name'],

        'docker': ['user_credentials_path', 'scylla_version'],

        'baremetal': ['s3_baremetal_config', 'db_nodes_private_ip', 'db_nodes_public_ip', 'user_credentials_path'],

        'aws-siren': ["user_prefix", "instance_type_loader", "region_name", "cloud_credentials_path",
                      "nemesis_filter_seeds"],

        'gce-siren': ['user_prefix', 'gce_network', 'gce_image_username', 'gce_instance_type_db',
                      'gce_root_disk_type_db', 'gce_n_local_ssd_disk_db',
                      'gce_instance_type_loader', 'gce_root_disk_type_loader', 'gce_n_local_ssd_disk_loader',
                      'gce_instance_type_monitor', 'gce_root_disk_type_monitor',
                      'gce_n_local_ssd_disk_monitor', 'gce_datacenter'],

        'k8s-local-kind': ['user_credentials_path', 'scylla_version', 'scylla_mgmt_agent_version',
                           'k8s_scylla_operator_helm_repo',
                           'k8s_scylla_cluster_name', 'k8s_scylla_disk_gi', 'mini_k8s_version',
                           'mgmt_docker_image'],

        'k8s-local-kind-aws': ['user_credentials_path', 'scylla_version', 'scylla_mgmt_agent_version',
                               'k8s_scylla_operator_helm_repo',
                               'k8s_scylla_cluster_name', 'k8s_scylla_disk_gi', 'mini_k8s_version',
                               'mgmt_docker_image'],

        'k8s-local-kind-gce': ['user_credentials_path', 'scylla_version', 'scylla_mgmt_agent_version',
                               'k8s_scylla_operator_helm_repo',
                               'k8s_scylla_cluster_name', 'k8s_scylla_disk_gi', 'mini_k8s_version',
                               'mgmt_docker_image'],

        'k8s-gke': ['gke_cluster_version', 'gce_instance_type_db', 'gce_root_disk_type_db',
                    'gce_n_local_ssd_disk_db', 'user_credentials_path', 'scylla_version',
                    'scylla_mgmt_agent_version', 'k8s_scylla_operator_helm_repo',
                    'k8s_scylla_cluster_name', 'k8s_loader_cluster_name', 'gce_instance_type_loader',
                    'gce_image_monitor', 'gce_instance_type_monitor', 'gce_root_disk_type_monitor',
                    'gce_n_local_ssd_disk_monitor', 'mgmt_docker_image'],

        'k8s-eks': ['instance_type_loader', 'instance_type_monitor', 'instance_type_db', 'region_name',
                    'ami_id_db_scylla', 'ami_id_monitor',
                    'aws_root_disk_name_monitor', 'ami_db_scylla_user', 'ami_monitor_user', 'user_credentials_path',
                    'scylla_version', 'scylla_mgmt_agent_version', 'k8s_scylla_operator_docker_image',
                    'k8s_scylla_cluster_name', 'k8s_loader_cluster_name',
                    'mgmt_docker_image', 'eks_service_ipv4_cidr', 'eks_vpc_cni_version', 'eks_role_arn',
                    'eks_cluster_version', 'eks_nodegroup_role_arn'],
    }

    defaults_config_files = {
        "aws": [sct_abs_path('defaults/aws_config.yaml')],
        "gce": [sct_abs_path('defaults/gce_config.yaml')],
        "azure": [sct_abs_path('defaults/azure_config.yaml')],
        "docker": [sct_abs_path('defaults/docker_config.yaml')],
        "baremetal": [sct_abs_path('defaults/baremetal_config.yaml')],
        "aws-siren": [sct_abs_path('defaults/aws_config.yaml')],
        "gce-siren": [sct_abs_path('defaults/gce_config.yaml')],
        "k8s-local-kind": [sct_abs_path('defaults/k8s_local_kind_config.yaml')],
        "k8s-local-kind-aws": [
            sct_abs_path('defaults/aws_config.yaml'),
            sct_abs_path('defaults/k8s_local_kind_aws_config.yaml'),
            sct_abs_path('defaults/k8s_local_kind_config.yaml')],
        "k8s-local-kind-gce": [
            sct_abs_path('defaults/k8s_local_kind_gce_config.yaml'),
            sct_abs_path('defaults/k8s_local_kind_config.yaml')],
        "k8s-gke": [sct_abs_path('defaults/gce_config.yaml'), sct_abs_path('defaults/k8s_gke_config.yaml')],
        "k8s-eks": [sct_abs_path('defaults/aws_config.yaml'), sct_abs_path('defaults/k8s_eks_config.yaml')],
    }

    per_provider_multi_region_params = {
        "aws": ['region_name', 'n_db_nodes', 'ami_id_db_scylla', 'ami_id_loader'],
        "gce": ['gce_datacenter', 'n_db_nodes']
    }

    stress_cmd_params = [
        # this list is used for variouse checks against stress commands, such as:
        # 1. Check if all c-s profile files existing that are referred in the commands
        # 2. Check what stress tools test is needed when loader is prepared
        'gemini_cmd', 'stress_cmd', 'stress_read_cmd', 'stress_cmd_w', 'stress_cmd_r', 'stress_cmd_m',
        'prepare_write_cmd', 'stress_cmd_no_mv', 'stress_cmd_no_mv_profile',
        'prepare_stress_cmd', 'stress_cmd_1', 'stress_cmd_complex_prepare', 'prepare_write_stress',
        'stress_cmd_read_10m', 'stress_cmd_read_cl_one', 'stress_cmd_read_80m',
        'stress_cmd_complex_verify_read', 'stress_cmd_complex_verify_more',
        'write_stress_during_entire_test', 'verify_data_after_entire_test',
        'stress_cmd_read_cl_quorum', 'verify_stress_after_cluster_upgrade',
        'stress_cmd_complex_verify_delete', 'stress_cmd_lwt_mixed', 'stress_cmd_lwt_de',
        'stress_cmd_lwt_dc', 'stress_cmd_lwt_ue', 'stress_cmd_lwt_uc', 'stress_cmd_lwt_ine',
        'stress_cmd_lwt_d', 'stress_cmd_lwt_u', 'stress_cmd_lwt_i'
    ]
    ami_id_params = ['ami_id_db_scylla', 'ami_id_loader', 'ami_id_monitor', 'ami_id_db_cassandra', 'ami_id_db_oracle']
    aws_supported_regions = ['eu-west-1', 'eu-west-2', 'us-west-2', 'us-east-1', 'eu-north-1', 'eu-central-1']

    def __init__(self):  # noqa: PLR0912, PLR0914, PLR0915
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        super().__init__()
        self.scylla_version = None
        self.is_enterprise = False

        self.log = logging.getLogger(__name__)
        env = self._load_environment_variables()
        config_files = env.get('config_files', [])
        config_files = [sct_abs_path(f) for f in config_files]

        # prepend to the config list the defaults the config files
        backend = env.get('cluster_backend')
        backend_config_files = [sct_abs_path('defaults/test_default.yaml')]
        if backend:
            backend_config_files += self.defaults_config_files[str(backend)]
        self.multi_region_params = self.per_provider_multi_region_params.get(str(backend), [])

        # load docker images defaults
        self.load_docker_images_defaults()

        # 1) load the default backend config files
        files = anyconfig.load(list(backend_config_files))
        merge_dicts_append_strings(self, files)

        # 2) load the config files
        try:
            for conf_file in list(config_files):
                if not os.path.exists(conf_file):
                    raise FileNotFoundError(f"Couldn't find config file: {conf_file}")
            files = anyconfig.load(list(config_files))
            merge_dicts_append_strings(self, files)
        except ValueError:
            self.log.warning("Failed to load configuration files: %s", config_files)

        regions_data = self.get('regions_data') or {}
        if regions_data:
            del self['regions_data']

        # 2.2) load the region data

        cluster_backend = self.get('cluster_backend')
        cluster_backend = env.get('cluster_backend', cluster_backend)

        region_names = self.region_names

        if cluster_backend in ['aws', 'aws-siren', 'k8s-eks']:
            if regions_data:
                for region in region_names:
                    if region not in regions_data:
                        raise ValueError(f"{region} isn't supported, use: {list(regions_data.keys())}")

                    for key, value in regions_data.get(region, {}).items():
                        if key not in self.keys():
                            self[key] = value
                        elif len(self[key].split()) < len(region_names):
                            self[key] += " {}".format(value)
            else:
                for region in region_names:
                    if region not in self.aws_supported_regions:
                        raise ValueError(f"{region} isn't supported, use: {self.aws_supported_regions}")

        # 3) overwrite with environment variables
        merge_dicts_append_strings(self, env)

        # 4) update events max severities
        add_severity_limit_rules(self.get("max_events_severities"))
        print_critical_events()

        # 5) overwrite AMIs
        for key in self.ami_id_params:
            if param := self.get(key):
                self[key] = convert_name_to_ami_if_needed(param, tuple(self.region_names))

        # 6) handle scylla_version if exists
        scylla_linux_distro = self.get('scylla_linux_distro')
        dist_type = scylla_linux_distro.split('-')[0]
        dist_version = scylla_linux_distro.split('-')[-1]

        if scylla_version := self.get('scylla_version'):  # pylint: disable=too-many-nested-blocks
            if not self.get('docker_image'):
                self['docker_image'] = get_scylla_docker_repo_from_version(scylla_version)
            if self.get("cluster_backend") in (
                    "docker", "k8s-eks", "k8s-gke",
                    "k8s-local-kind", "k8s-local-kind-aws", "k8s-local-kind-gce"):
                self.log.info("Assume that Scylla Docker image has repo file pre-installed.")
                self._replace_docker_image_latest_tag()
            elif not self.get('ami_id_db_scylla') and self.get('cluster_backend') == 'aws':
                ami_list = []
                for region in region_names:
                    aws_arch = get_arch_from_instance_type(self.get('instance_type_db'), region_name=region)
                    try:
                        if ':' in scylla_version:
                            ami = get_branched_ami(scylla_version=scylla_version, region_name=region, arch=aws_arch)[0]
                        else:
                            ami = get_scylla_ami_versions(version=scylla_version, region_name=region, arch=aws_arch)[0]
                    except Exception as ex:  # noqa: BLE001
                        raise ValueError(f"AMIs for scylla_version='{scylla_version}' not found in {region} "
                                         f"arch={aws_arch}") from ex
                    self.log.debug("Found AMI %s(%s) for scylla_version='%s' in %s",
                                   ami.name, ami.image_id, scylla_version, region)
                    ami_list.append(ami)
                self['ami_id_db_scylla'] = " ".join(ami.image_id for ami in ami_list)
            elif not self.get("gce_image_db") and self.get("cluster_backend") == "gce":
                try:
                    if ":" in scylla_version:
                        gce_image = get_branched_gce_images(scylla_version=scylla_version)[0]
                    else:
                        # gce_image.name format examples: scylla-4-3-6 or scylla-enterprise-2021-1-2
                        gce_image = get_scylla_gce_images_versions(version=scylla_version)[0]
                except Exception as ex:  # noqa: BLE001
                    raise ValueError(f"GCE image for scylla_version='{scylla_version}' was not found") from ex

                self.log.debug("Found GCE image %s for scylla_version='%s'", gce_image.name, scylla_version)
                self["gce_image_db"] = gce_image.self_link
            elif not self.get("azure_image_db") and self.get("cluster_backend") == "azure":
                scylla_azure_images = []
                if isinstance(self.get('azure_region_name'), list):
                    azure_region_names = self.get('azure_region_name')
                else:
                    azure_region_names = [self.get('azure_region_name')]

                for region in azure_region_names:
                    try:
                        if ":" in scylla_version:
                            azure_image = azure_utils.get_scylla_images(
                                scylla_version=scylla_version, region_name=region)[0]
                        else:
                            azure_image = azure_utils.get_released_scylla_images(
                                scylla_version=scylla_version, region_name=region)[0]
                    except Exception as ex:  # noqa: BLE001
                        raise ValueError(
                            f"Azure Image for scylla_version='{scylla_version}' not found in {region}") from ex
                    self.log.debug("Found Azure Image %s for scylla_version='%s' in %s",
                                   azure_image.name, scylla_version, region)
                    scylla_azure_images.append(azure_image)
                self["azure_image_db"] = " ".join(getattr(image, 'id', None) or getattr(
                    image, 'unique_id', None) for image in scylla_azure_images)
            elif not self.get('scylla_repo'):
                self['scylla_repo'] = find_scylla_repo(scylla_version, dist_type, dist_version)
            else:
                raise ValueError("'scylla_version' can't used together with 'ami_id_db_scylla', 'gce_image_db' "
                                 "or with 'scylla_repo'")

            if (
                self.get("n_loaders") and
                not self.get("bare_loaders") and
                not self.get("scylla_repo_loader") and
                self.get("cluster_backend") != "aws"
            ):
                scylla_linux_distro_loader = self.get('scylla_linux_distro_loader')
                dist_type_loader = scylla_linux_distro_loader.split('-')[0]
                dist_version_loader = scylla_linux_distro_loader.split('-')[-1]

                scylla_version_for_loader = "nightly" if scylla_version == "latest" else scylla_version

                self['scylla_repo_loader'] = find_scylla_repo(scylla_version_for_loader,
                                                              dist_type_loader,
                                                              dist_version_loader)

        # 6.1) handle oracle_scylla_version if exists
        if (oracle_scylla_version := self.get('oracle_scylla_version')) \
           and self.get("db_type") == "mixed_scylla":  # pylint: disable=too-many-nested-blocks
            if not self.get('ami_id_db_oracle') and self.get('cluster_backend') == 'aws':
                ami_list = []
                for region in region_names:
                    aws_arch = get_arch_from_instance_type(self.get('instance_type_db_oracle'), region_name=region)
                    try:
                        if ':' in oracle_scylla_version:
                            ami = get_branched_ami(
                                scylla_version=oracle_scylla_version, region_name=region, arch=aws_arch)[0]
                        else:
                            ami = get_scylla_ami_versions(version=oracle_scylla_version,
                                                          region_name=region, arch=aws_arch)[0]
                    except Exception as ex:  # noqa: BLE001
                        raise ValueError(f"AMIs for oracle_scylla_version='{scylla_version}' not found in {region} "
                                         f"arch={aws_arch}") from ex

                    self.log.debug("Found AMI %s for oracle_scylla_version='%s' in %s",
                                   ami.image_id, oracle_scylla_version, region)
                    ami_list.append(ami)
                self["ami_id_db_oracle"] = " ".join(ami.image_id for ami in ami_list)
            else:
                raise ValueError("'oracle_scylla_version' and 'ami_id_db_oracle' can't used together")

        # 7) support lookup of repos for upgrade test
        new_scylla_version = self.get('new_version')
        if new_scylla_version and not 'k8s' in cluster_backend:
            if not self.get('ami_id_db_scylla') and cluster_backend == 'aws':  # pylint: disable=no-else-raise
                raise ValueError("'new_version' isn't supported for AWS AMIs")

            elif not self.get('new_scylla_repo'):
                self['new_scylla_repo'] = find_scylla_repo(new_scylla_version, dist_type, dist_version)

        # 8) resolve repo symlinks
        for repo_key in ("scylla_repo", "scylla_repo_loader", "new_scylla_repo",):
            if self.get(repo_key):
                self[repo_key] = resolve_latest_repo_symlink(self[repo_key])

        # 9) append username or ami_id_db_scylla_desc to the user_prefix
        version_tag = self.get('ami_id_db_scylla_desc') or getpass.getuser()
        user_prefix = self.get('user_prefix') or getpass.getuser()
        prefix_max_len = 35
        if version_tag != user_prefix:
            user_prefix = "{}-{}".format(user_prefix, version_tag)
        if self.get('cluster_backend') == 'azure':
            # for Azure need to shorten it more due longer region names
            prefix_max_len -= 2
        if (self.get("simulated_regions") or 0) > 1:
            # another shortening for simulated regions due added simulated dc suffix
            prefix_max_len -= 3
        self['user_prefix'] = user_prefix[:prefix_max_len]

        # remove any special characters from user_prefix, since later it will be used as a part of the instance names
        # and some platfrom don't support special characters in the instance names (docker, AWS and such)
        self['user_prefix'] = re.sub(r"[^a-zA-Z0-9-]", "-", self['user_prefix'])

        # 11) validate that supported instance_provision selected
        if self.get('instance_provision') not in ['spot', 'on_demand', 'spot_fleet', 'spot_low_price']:
            raise ValueError(f"Selected instance_provision type '{self.get('instance_provision')}' is not supported!")

        # 12) validate authenticator parameters
        if self.get('authenticator') and self.get('authenticator') == "PasswordAuthenticator":
            authenticator_user = self.get("authenticator_user")
            authenticator_password = self.get("authenticator_password")
            if not (authenticator_password and authenticator_user):
                raise ValueError("For PasswordAuthenticator authenticator authenticator_user and authenticator_password"
                                 " have to be provided")

        if self.get('alternator_enforce_authorization'):
            if not self.get('authenticator') or not self.get('authorizer'):
                raise ValueError(
                    "When enabling `alternator_enforce_authorization` both `authenticator` and `authorizer` should be defined")

        # 13) validate stress and prepare duration:
        if stress_duration := self.get('stress_duration'):
            try:
                self['stress_duration'] = abs(int(stress_duration))
            except ValueError:
                raise ValueError(f'Configured stress duration for generic test duratinon have to be \
                                 positive integer number in minutes. Current value: {stress_duration}') from ValueError
        if prepare_stress_duration := self.get('prepare_stress_duration'):
            try:
                self['prepare_stress_duration'] = abs(int(prepare_stress_duration))
            except ValueError:
                raise ValueError(f'Configured stress duration for generic test duratinon have to be \
                                 positive integer number in minutes. Current value: {prepare_stress_duration}') from ValueError

        # 14 Validate run_fullscan parmeters
        if run_fullscan_params := self.get("run_fullscan"):
            if not isinstance(run_fullscan_params, list) or not len(run_fullscan_params) > 0:
                raise ValueError(f"run_fullscan parameter must be non empty list, but got: {run_fullscan_params}")
            for param in run_fullscan_params:
                try:
                    ConfigParams(**json.loads(param))
                except json.decoder.JSONDecodeError as exp:
                    raise ValueError(
                        f"each item of run_fullscan list: {run_fullscan_params}, "
                        f"item {param}, must be JSON but got error: {repr(exp)}") from exp
                except TypeError as exp:
                    raise ValueError(
                        f" Got error: {repr(exp)}, on item '{param}'") from exp

        # 15 Force endpoint_snitch to GossipingPropertyFileSnitch if using simulated_regions or simulated_racks
        if (self.get("simulated_regions") or 0) > 1 or (self.get("simulated_racks") or 0) > 1:
            if snitch := self.get("endpoint_snitch"):
                assert snitch.endswith("GossipingPropertyFileSnitch"), \
                    f"Simulating racks requires endpoint_snitch to be GossipingPropertyFileSnitch while it set to {self['endpoint_snitch']}"
            self["endpoint_snitch"] = "org.apache.cassandra.locator.GossipingPropertyFileSnitch"

        # 16 Validate use_dns_names
        if self.get("use_dns_names"):
            if cluster_backend not in ("aws",):
                raise ValueError(f"use_dns_names is not supported for {cluster_backend} backend")

        # 17 Validate scylla network configuration mandatory values
        if scylla_network_config := self.get("scylla_network_config"):
            check_list = {"listen_address": None, "rpc_address": None,
                          "broadcast_rpc_address": None, "broadcast_address": None, "test_communication": None}
            number2word = {1: "first", 2: "second", 3: "third"}
            nics = set()
            for i, address_config in enumerate(scylla_network_config):
                for param in ["address", "ip_type", "public", "nic"]:
                    if address_config.get(param) is None:
                        raise ValueError(
                            f"'{param}' parameter value for {number2word[i + 1]} address is not defined. It is must parameter")

                if address_config["ip_type"] == "ipv4" and address_config["nic"] == 1 and address_config["public"] is True:
                    raise ValueError(
                        "If ipv4 and public is True it has to be primary network interface, it means device index (nic) is 0")

                nics.add(address_config["nic"])
                if address_config["address"] not in check_list:
                    continue

                check_list[address_config["address"]] = True

            if not_defined_address := ",".join([key for key, value in check_list.items() if value is None]):
                raise ValueError(f"Interface address(es) were not defined: {not_defined_address}")

            if len(nics) > 1 and len(self.region_names) >= 2:
                raise ValueError("Multiple network interfaces aren't supported for multi region use cases")

        # 18 Validate K8S TLS+SNI values
        if self.get("k8s_enable_sni") and not self.get("k8s_enable_tls"):
            raise ValueError("'k8s_enable_sni=true' requires 'k8s_enable_tls' also to be 'true'.")

        SCTCapacityReservation.get_cr_from_aws(self)

        # 19: validate kafka configuration
        if kafka_connectors := self.get('kafka_connectors'):
            self['kafka_connectors'] = [SctKafkaConfiguration(**connector)
                                        for connector in kafka_connectors]

        # 20 Validate Manager agent backup general parameters
        if backup_params := self.get("mgmt_agent_backup_config"):
            self["mgmt_agent_backup_config"] = AgentBackupParameters(**backup_params)
        # Validate zero token nodes
        if self.get("use_zero_nodes"):
            self._validate_zero_token_backend_support(backend=cluster_backend)
            zero_nodes_num = self.get("n_db_zero_token_nodes")
            data_nodes_num = self.get("n_db_nodes")
            # if number of zero nodes is set for cluster setup, check correctness of settings
            if zero_nodes_num:
                zero_nodes_num = [zero_nodes_num] if isinstance(zero_nodes_num, int) else [
                    int(i) for i in str(zero_nodes_num).split()]
                data_nodes_num = [data_nodes_num] if isinstance(data_nodes_num, int) else [
                    int(i) for i in str(data_nodes_num).split()]
                assert len(zero_nodes_num) == len(
                    data_nodes_num), "Config of zero token nodes is not equal config of data nodes for multi dc"

    def load_docker_images_defaults(self):
        docker_images_dir = pathlib.Path(sct_abs_path('defaults/docker_images'))
        if docker_images_dir.is_dir():
            yaml_files = []
            for root, _, files in os.walk(docker_images_dir):
                yaml_files.extend([os.path.join(root, f) for f in files if f.endswith('.yaml')])
            if yaml_files:
                docker_images_defaults = anyconfig.load(yaml_files)
                stress_image = {key: value.get('image') for key, value in docker_images_defaults.items()}
                anyconfig.merge(self, dict(stress_image=stress_image))

    def log_config(self):
        self.log.info(self.dump_config())

    @property
    def total_db_nodes(self) -> List[int]:
        """Used to get total number of db nodes data nodes and zero nodes"""
        use_zero_nodes = self.get("use_zero_nodes")
        zero_nodes_num = self.get("n_db_zero_token_nodes")
        data_nodes_num = self.get("n_db_nodes")
        zero_nodes_num = [zero_nodes_num] if isinstance(zero_nodes_num, int) else [
            int(i) for i in str(zero_nodes_num).split()]
        data_nodes_num = [data_nodes_num] if isinstance(data_nodes_num, int) else [
            int(i) for i in str(data_nodes_num).split()]
        total_nodes = data_nodes_num[:]
        if use_zero_nodes and zero_nodes_num:
            total_nodes = [n1 + n2 for n1,
                           n2 in zip(data_nodes_num, zero_nodes_num)]

        self.log.debug("Total nodes: %s", total_nodes)
        return total_nodes

    @property
    def region_names(self) -> List[str]:
        region_names = self.environment.get('region_name')
        if region_names is None:
            region_names = self.get('region_name')
        if region_names is None:
            region_names = ''
        if isinstance(region_names, str):
            region_names = region_names.split()
        output = []
        for region_name in region_names:
            output.extend(region_name.split())
        return output

    @property
    def gce_datacenters(self) -> List[str]:
        gce_datacenters = self.environment.get('gce_datacenter')
        if gce_datacenters is None:
            gce_datacenters = self.get('gce_datacenter')
        if gce_datacenters is None:
            gce_datacenters = ''
        if isinstance(gce_datacenters, str):
            gce_datacenters = gce_datacenters.split()
        output = []
        for gce_datacenter in gce_datacenters:
            output.extend(gce_datacenter.split())
        return output

    @property
    def environment(self) -> dict:
        return self._load_environment_variables()

    @classmethod
    def get_config_option(cls, name):
        return [o for o in cls.config_options if o['name'] == name][0]

    def get_default_value(self, key, include_backend=False):

        default_config_files = [sct_abs_path('defaults/test_default.yaml')]
        backend = self['cluster_backend']
        if backend and include_backend:
            default_config_files += self.defaults_config_files[str(backend)]

        return anyconfig.load(list(default_config_files)).get(key, None)

    def _load_environment_variables(self):
        environment_vars = {}
        for opt in self.config_options:
            if opt['env'] in os.environ:
                try:
                    environment_vars[opt['name']] = opt['type'](os.environ[opt['env']])
                except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
                    raise ValueError(
                        "failed to parse {} from environment variable".format(opt['env'])) from ex
            nested_keys = [key for key in os.environ if key.startswith(opt['env'] + '.')]
            if nested_keys:
                list_value = []
                dict_value = {}
                for key in nested_keys:
                    nest_key, *_ = key.split('.')[1:]
                    if nest_key.isdigit():
                        list_value.insert(int(nest_key), os.environ.get(key))
                    else:
                        dict_value[nest_key] = os.environ.get(key)
                current_value = environment_vars.get(opt['name'])
                if current_value and isinstance(current_value, dict):
                    current_value.update(dict_value)
                else:
                    environment_vars[opt['name']] = opt['type'](list_value or dict_value)

        return environment_vars

    def get(self, key: str | None):
        """
        get the value of test configuration parameter by the name
        """

        if key and '.' in key:
            if ret_val := self._dotted_get(key):
                return ret_val
        ret_val = super().get(key)

        if key in self.multi_region_params and isinstance(ret_val, list):
            ret_val = ' '.join(ret_val)

        return ret_val

    def _dotted_get(self, key: str):
        """
        if key for retrieval is dot notation, ex. 'stress_image.ycsb'
        we assume `stress_image` would be a dict
        """
        keys = key.split('.')
        current = self.get(keys[0])
        for k in keys[1:]:
            if not isinstance(current, dict):
                break
            current = current.get(k)
        return current

    def _validate_value(self, opt):
        opt['is_k8s_multitenant_value'] = False
        try:
            opt['type'](self.get(opt['name']))
        except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
            if not (self.get("cluster_backend").startswith("k8s")
                    and self.get("k8s_tenants_num") > 1
                    and opt.get("k8s_multitenancy_supported")
                    and isinstance(self.get(opt['name']), list)):
                if not (opt['name'] == "nemesis_selector"
                        and isinstance(self.get('nemesis_class_name'), str)
                        and len(self.get("nemesis_class_name").split(" ")) > 1):
                    raise ValueError("failed to validate {}".format(opt['name'])) from ex
            for list_element in self.get(opt['name']):
                try:
                    opt['type'](list_element)
                except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
                    raise ValueError("failed to validate {}".format(opt['name'])) from ex
            opt['is_k8s_multitenant_value'] = True

        choices = opt.get('choices')
        if choices:
            cur_val = self.get(opt['name'])
            if not opt.get('is_k8s_multitenant_value'):
                cur_val = [cur_val]
            for cur_val_element in cur_val:
                assert cur_val_element in choices, "failed to validate '{}': {} not in {}".format(
                    opt['name'], cur_val_element, choices)

    @property
    def list_of_stress_tools(self) -> Set[str]:
        stress_tools = set()
        for param_name in self.stress_cmd_params:
            stress_cmds = self.get(param_name)
            if not (isinstance(stress_cmds, (list, str)) and stress_cmds):
                continue
            if isinstance(stress_cmds, str):
                stress_cmds = [stress_cmds]

            for stress_cmd in stress_cmds:
                if not stress_cmd:
                    continue
                if not isinstance(stress_cmd, list):
                    stress_cmd = [stress_cmd]  # noqa: PLW2901
                for cmd in stress_cmd:
                    if stress_tool := cmd.split(maxsplit=2)[0]:
                        stress_tools.add(stress_tool)

        return stress_tools

    def check_required_files(self):
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-branches
        for param_name in self.stress_cmd_params:
            stress_cmds = self.get(param_name)
            if stress_cmds is None:
                continue
            if isinstance(stress_cmds, str):
                stress_cmds = [stress_cmds]
            for stress_cmd in stress_cmds:
                if not stress_cmd:
                    continue
                if not isinstance(stress_cmd, list):
                    stress_cmd = [stress_cmd]  # noqa: PLW2901
                for cmd in stress_cmd:
                    cmd = cmd.strip(' ')  # noqa: PLW2901
                    if cmd.startswith('latte'):
                        script_name_regx = re.compile(r'([/\w-]*\.rn)')
                        script_name = script_name_regx.search(cmd).group(1)
                        full_path = pathlib.Path(get_sct_root_path()) / script_name
                        assert full_path.exists(), f"{full_path} doesn't exists, please check your configuration"

                    if not cmd.startswith('cassandra-stress'):
                        continue
                    for option in cmd.split():
                        if option.startswith('profile='):
                            option = option.split('=', 1)  # noqa: PLW2901
                            if len(option) < 2:
                                continue
                            profile_path = option[1]
                            if 'scylla-qa-internal' in profile_path:
                                continue
                            if not profile_path.startswith('/tmp'):
                                raise ValueError(f"Stress command parameter '{param_name}' contains wrong path "
                                                 f"'{profile_path}' to profile, it should be formed in following "
                                                 "manner '/tmp/{file_name_from_data_dir}'")
                            profile_name = profile_path[5:]
                            if pathlib.Path(sct_abs_path(os.path.join('data_dir', profile_name))).exists():
                                break  # We are ok here and skipping whole command if file is there
                            raise ValueError(f"Stress command parameter '{param_name}' contains profile "
                                             f"'{profile_path}' that does not exists under data_dir/")
        self._validate_scylla_d_overrides_files_exists()

    def verify_configuration(self):
        """
        Check that all required values are set, and validated each value to be of correct type or value
        also check required options per backend

        :return: None
        :raises ValueError: on failures in validations
        :raise Exception: on unsupported backends
        """
        self._check_unexpected_sct_variables()
        self._validate_sct_variable_values()
        backend = self.get('cluster_backend')
        db_type = self.get('db_type')
        self._check_version_supplied(backend)
        self._check_per_backend_required_values(backend)
        if backend in ('aws', 'gce') and db_type != 'cloud_scylla' and (
                self.get('simulated_regions') or 0) < 2:
            self._check_multi_region_params(backend)
        if backend == 'docker':
            self._validate_docker_backend_parameters()

        self._verify_data_volume_configuration(backend)

        if self.get('n_db_nodes'):
            self._validate_seeds_number()
            self._validate_nemesis_can_run_on_non_seed()
            self._validate_number_of_db_nodes_divides_by_az_number()

        if self.get('use_zero_nodes'):
            self._validate_zero_token_backend_support(backend)

        self._check_partition_range_with_data_validation_correctness()
        self._verify_scylla_bench_mode_and_workload_parameters()

        self._validate_placement_group_required_values()
        self._instance_type_validation()

    def _replace_docker_image_latest_tag(self):
        docker_repo = self.get('docker_image')
        scylla_version = self.get('scylla_version')

        if scylla_version == 'latest':
            result = get_specific_tag_of_docker_image(docker_repo=docker_repo)
            if result == 'latest':
                raise ValueError(
                    "scylla-operator expects semver-like tags for Scylla docker images. "
                    "'latest' should not be used.")
            self['scylla_version'] = result

    def _get_target_upgrade_version(self):
        # 10) update target_upgrade_version automatically
        if new_scylla_repo := self.get('new_scylla_repo'):
            if not self.get('target_upgrade_version'):
                self['target_upgrade_version'] = get_branch_version(new_scylla_repo)
            scylla_version = get_branch_version(new_scylla_repo, full_version=True)
            self.update_argus_with_version(scylla_version, "scylla-server-upgrade-target")

    def _check_unexpected_sct_variables(self):
        # check if there are SCT_* environment variable which aren't documented
        config_keys = {opt['env'] for opt in self.config_options}
        env_keys = {o.split('.')[0] for o in os.environ if o.startswith('SCT_')}
        unknown_env_keys = env_keys.difference(config_keys)
        if unknown_env_keys:
            output = ["{}={}".format(key, os.environ.get(key)) for key in unknown_env_keys]
            raise ValueError("Unsupported environment variables were used:\n\t - {}".format("\n\t - ".join(output)))

        # check for unsupported configuration
        config_names = {o['name'] for o in self.config_options}
        unsupported_option = set(self.keys()).difference(config_names)

        if unsupported_option:
            res = "Unsupported config option/s found:\n"
            for option in unsupported_option:
                res += "\t * '{}: {}'\n".format(option, self[option])
            raise ValueError(res)

    def _validate_sct_variable_values(self):
        for opt in self.config_options:
            if opt['name'] in self:
                self._validate_value(opt)

    def _check_multi_region_params(self, backend):
        region_param_names = {"aws": "region_name", "gce": "gce_datacenter"}
        current_region_param_name = region_param_names[backend]
        region_count = {}
        for opt in self.multi_region_params:
            val = self.get(opt)
            if isinstance(val, str):
                region_count[opt] = len(self.get(opt).split())
            elif isinstance(val, list):
                region_count[opt] = len(val)
            else:
                region_count[opt] = 1
        if not all(region_count[current_region_param_name] == x for x in region_count.values()):
            raise ValueError("not all multi region values are equal: \n\t{}".format(region_count))

    def _validate_seeds_number(self):
        seeds_num = self.get('seeds_num')
        assert seeds_num > 0, "Seed number should be at least one"

        num_of_db_nodes = sum([int(i) for i in str(self.get('n_db_nodes')).split(' ')])
        assert not num_of_db_nodes or seeds_num <= num_of_db_nodes, \
            f"Seeds number ({seeds_num}) should be not more then nodes number ({num_of_db_nodes})"

    def _validate_nemesis_can_run_on_non_seed(self) -> None:
        if self.get('nemesis_filter_seeds') is False or self.get('nemesis_class_name') == "NoOpMonkey":
            return
        seeds_num = self.get('seeds_num')
        num_of_db_nodes = sum([int(i) for i in str(self.get('n_db_nodes')).split(' ')]) + int(self.get('add_node_cnt'))
        assert num_of_db_nodes > seeds_num, \
            "Nemesis cannot run when 'nemesis_filter_seeds' is true and seeds number is equal to nodes number"

    def _validate_number_of_db_nodes_divides_by_az_number(self):
        if self.get("cluster_backend").startswith("k8s"):
            return
        az_count = len(self.get('availability_zone').split(',')) if self.get('availability_zone') else 1
        for nodes_num in [int(i) for i in str(self.get('n_db_nodes')).split(' ')]:
            assert nodes_num % az_count == 0, \
                f"Number of db nodes ({nodes_num}) should be divisible by number of availability zones ({az_count})"

    def _validate_placement_group_required_values(self):
        if self.get("use_placement_group"):
            az_count = len(self.get('availability_zone').split(',')) if self.get('availability_zone') else 1
            regions_count = len(self.region_names)
            assert az_count == 1 and regions_count == 1, \
                (f"Number of Regions({regions_count}) and AZ({az_count}) should be 1 "
                 f"when param use_placement_group is used")

    def _validate_scylla_d_overrides_files_exists(self):
        if scylla_d_overrides_files := self.get("scylla_d_overrides_files"):
            for config_file_path in scylla_d_overrides_files:
                config_file = pathlib.Path(get_sct_root_path()) / config_file_path
                assert config_file.exists(), f"{config_file} doesn't exists, please check your configuration"

    def _check_per_backend_required_values(self, backend: str):
        if backend in self.available_backends:
            if backend in ('aws', 'gce') and self.get("db_type") == "cloud_scylla":
                backend += "-siren"
            self._check_backend_defaults(backend, self.backend_required_params[backend])
        else:
            raise ValueError("Unsupported backend [{}]".format(backend))

    def _check_backend_defaults(self, backend, required_params):
        opts = [o for o in self.config_options if o['name'] in required_params]
        for _opt in opts:
            assert _opt['name'] in self, "{} missing from config for {}".format(_opt['name'], backend)

    def _instance_type_validation(self):
        if instance_type := self.get('nemesis_grow_shrink_instance_type'):
            backend = self.get('cluster_backend')
            match backend:
                case 'aws':
                    for region in self.region_names:
                        assert aws_check_instance_type_supported(
                            instance_type, region), f"Instance type[{instance_type}] not supported in region [{region}]"
                case 'gce':
                    machine_types_client, info = get_gce_compute_machine_types_client()
                    for datacenter in self.gce_datacenters:
                        for zone in GCE_SUPPORTED_REGIONS.get(datacenter):
                            _zone = f"{datacenter}-{zone}"
                            assert gce_check_if_machine_type_supported(
                                machine_types_client, instance_type, project=info['project_id'],
                                zone=_zone), f"Instance type[{instance_type}] not supported in zone [{_zone}]"
                case 'azure':
                    if azure_region_names := self.get('azure_region_name'):
                        if not isinstance(azure_region_names, list):
                            azure_region_names = [self.get('azure_region_name')]
                        for region in self.get('azure_region_name'):
                            assert azure_check_instance_type_available(
                                instance_type, region), f"Instance type [{instance_type}] not supported in region [{region}]"
                case _:
                    raise ValueError(f"Unsupported backend [{backend}] for using nemesis_grow_shrink_instance_type")

    def _check_version_supplied(self, backend: str):
        options_must_exist = []

        if (not self.get('use_preinstalled_scylla') and
                not backend == 'baremetal' and
                not self.get('unified_package')):
            options_must_exist += ['scylla_repo']

        if self.get('db_type') == 'cloud_scylla':
            options_must_exist += ['cloud_cluster_id']
        elif backend == 'aws':
            options_must_exist += ['ami_id_db_scylla']
        elif backend == 'gce':
            options_must_exist += ['gce_image_db']
        elif backend == 'azure':
            options_must_exist += ['azure_image_db']
        elif backend == 'docker':
            options_must_exist += ['docker_image']
        elif backend == 'baremetal':
            options_must_exist += ['db_nodes_public_ip']
        elif 'k8s' in backend:
            options_must_exist += ['scylla_version']

        if not options_must_exist:
            return
        assert all(self.get(o) for o in options_must_exist), \
            "scylla version/repos wasn't configured correctly\n" \
            f"configure those options: {options_must_exist}\n" \
            f"and those environment variables: {['SCT_' + o.upper() for o in options_must_exist]}"

    def _check_partition_range_with_data_validation_correctness(self):
        partition_range_with_data_validation = self.get('partition_range_with_data_validation')
        if partition_range_with_data_validation:
            error_message_template = "Expected format of 'partition_range_with_data_validation' parameter is: " \
                                     "<min PK value>-<max PK value>. {}Example: 0-250. " \
                                     "Got value: %s" % partition_range_with_data_validation

            if '-' not in partition_range_with_data_validation:
                raise ValueError(error_message_template.format(''))

            partition_range_splitted = partition_range_with_data_validation.split('-')

            if not (partition_range_splitted[0].isdigit() and partition_range_splitted[1].isdigit()):
                raise ValueError(error_message_template.format('PK values should be integer. '))

            if int(partition_range_splitted[1]) < int(partition_range_splitted[0]):
                raise ValueError(error_message_template.format('<max PK value> should be bigger then <min PK value>. '))

    @staticmethod
    def _validate_zero_token_backend_support(backend: str):
        assert backend == "aws", "Only AWS supports zero nodes configuration"

    def verify_configuration_urls_validity(self):  # pylint: disable=too-many-branches
        """
        Check if ami_id and repo urls are valid
        """
        backend = self.get('cluster_backend')
        if backend in ("k8s-eks", "k8s-gke"):
            return

        self._get_target_upgrade_version()

        # verify that the AMIs used all have 'user_data_format_version' tag
        if backend == 'aws':
            ami_id_db_scylla = self.get('ami_id_db_scylla').split()
            region_names = self.region_names
            ami_id_db_oracle = self.get('ami_id_db_oracle').split()
            for key_to_update, ami_list in [('user_data_format_version', ami_id_db_scylla),
                                            ('oracle_user_data_format_version', ami_id_db_oracle)]:
                if ami_list:
                    user_data_format_versions = set()
                    self[key_to_update] = '3'
                    for ami_id, region_name in zip(ami_list, region_names):
                        if not ami_built_by_scylla(ami_id, region_name):
                            continue
                        tags = get_ami_tags(ami_id, region_name)
                        assert 'user_data_format_version' in tags.keys(), \
                            f"\n\t'user_data_format_version' tag missing from [{ami_id}] on {region_name}\n\texisting " \
                            f"tags: {tags}"
                        user_data_format_versions.add(tags['user_data_format_version'])
                    assert len(
                        user_data_format_versions) <= 1, f"shouldn't have mixed versions {user_data_format_versions}"
                    if user_data_format_versions:
                        self[key_to_update] = list(user_data_format_versions)[0]

        if backend == 'gce':
            gce_image_db = self.get('gce_image_db').split()
            for image in gce_image_db:
                tags = get_gce_image_tags(image)
                if 'user_data_format_version' not in tags.keys():
                    # since older release aren't tagged, we default to 2 which was the version on the first gce images
                    logging.warning("'user_data_format_version' tag missing from [%s]: existing tags: %s", image, tags)
                self['user_data_format_version'] = tags.get('user_data_format_version', '2')

        if backend == 'azure':
            azure_image_db = self.get('azure_image_db').split()
            for image in azure_image_db:
                tags = azure_utils.get_image_tags(image)
                if 'user_data_format_version' not in tags.keys():
                    # since older release aren't tagged, we default to 2 which was the version on the first gce images
                    logging.warning("'user_data_format_version' tag missing from [%s]: existing tags: %s", image, tags)
                self['user_data_format_version'] = tags.get('user_data_format_version', '2')

        # For each Scylla repo file we will check that there is at least one valid URL through which to download a
        # version of SCYLLA, otherwise we will get an error.
        repos_to_validate = ['scylla_repo_loader']
        if backend in ("aws", "gce", "baremetal"):
            repos_to_validate.extend([
                'new_scylla_repo',
                'scylla_repo_m',
                'scylla_mgmt_address',
                'scylla_mgmt_agent_address',
            ])
        get_branch_version_for_multiple_repositories(
            urls=(self.get(url) for url in repos_to_validate if self.get(url)))

    def get_version_based_on_conf(self):  # pylint: disable=too-many-locals
        """
        figure out which version and if it's enterprise version
        base on configuration only, before nodes are up and running
        so test configuration can set up things which need to happen
        before nodes are up

        this is information is cached on the SCTConfiguration object
        :return: tuple - (scylla_version, is_enterprise)
        """
        backend = self.get('cluster_backend')
        scylla_version = None
        _is_enterprise = False

        if unified_package := self.get('unified_package'):
            with tempfile.TemporaryDirectory() as tmpdirname:
                LOCALRUNNER.run(shell_script_cmd(f"""
                    cd {tmpdirname}
                    curl {unified_package} -o ./unified_package.tar.gz
                    tar xvfz ./unified_package.tar.gz
                    """), verbose=False)

                scylla_version = next(pathlib.Path(tmpdirname).glob('**/SCYLLA-VERSION-FILE')).read_text()
                scylla_product = next(pathlib.Path(tmpdirname).glob('**/SCYLLA-PRODUCT-FILE')).read_text()
                _is_enterprise = scylla_product == 'scylla-enterprise'
        elif not self.get('use_preinstalled_scylla'):
            scylla_repo = self.get('scylla_repo')
            scylla_version = get_branch_version(scylla_repo, full_version=True)
            _is_enterprise = is_enterprise(scylla_version)
        elif self.get('db_type') == 'cloud_scylla':
            _is_enterpise = True
        elif backend == 'aws':
            amis = self.get('ami_id_db_scylla').split()
            region_name = self.region_names[0]
            tags = get_ami_tags(ami_id=amis[0], region_name=region_name)
            scylla_version = tags.get('scylla_version') or tags.get('ScyllaVersion')
            _is_enterprise = is_enterprise(scylla_version)
        elif backend == 'gce':
            images = self.get('gce_image_db').split()
            tags = get_gce_image_tags(images[0])
            scylla_version = tags.get('scylla_version').replace('-', '.')
            _is_enterprise = is_enterprise(scylla_version)
        elif backend == 'azure':
            images = self.get('azure_image_db').split()
            tags = azure_utils.get_image_tags(images[0])
            scylla_version = tags.get('scylla_version')
            _is_enterprise = is_enterprise(scylla_version)
        elif backend == 'docker' or 'k8s' in backend:
            docker_repo = self.get('docker_image')
            scylla_version = self.get('scylla_version')
            _is_enterprise = 'enterprise' in docker_repo
        self.scylla_version = scylla_version
        self.is_enterprise = _is_enterprise
        self.update_argus_with_version(scylla_version, "scylla-server-target")

        return scylla_version, _is_enterprise

    def update_argus_with_version(self, scylla_version: str, package_name: str):
        try:
            version_regex = ARGUS_VERSION_RE
            if match := version_regex.match(scylla_version):
                version_info = match.groupdict()
                package = Package(name=package_name, date=version_info.get("date", "#NO_DATE"),
                                  version=version_info["short"],
                                  revision_id=version_info.get("commit", "#NO_COMMIT"),
                                  build_id="#NO_BUILDID")
                self.log.info("Saving upgraded Scylla version...")
                test_config = TestConfig()
                test_config.init_argus_client(params=self, test_id=self.get("reuse_cluster") or self.get("test_id"))
                test_config.argus_client().submit_packages([package])
                test_config.argus_client().update_scylla_version(version_info["short"])
        except Exception as exc:  # pylint: disable=broad-except
            self.log.exception("Failed to save target Scylla version in Argus", exc_info=exc)

    def dict(self):
        out = deepcopy(self)

        # handle pydantic object, and convert them back to dicts
        # TODO: automate the process if we gonna keep using them more, or replace the whole configuration with pydantic/dataclasses
        if kafka_connectors := self.get('kafka_connectors'):
            out['kafka_connectors'] = [connector.dict(by_alias=True, exclude_none=True)
                                       for connector in kafka_connectors]
        if mgmt_agent_backup_config := self.get("mgmt_agent_backup_config"):
            out["mgmt_agent_backup_config"] = mgmt_agent_backup_config.dict(by_alias=True, exclude_none=True)
        return out

    def dump_config(self):
        """
        Dump current configuration to string

        :return: str
        """
        return anyconfig.dumps(self.dict(), ac_parser="yaml")

    def dump_help_config_markdown(self):
        """
        Dump all configuration options with their defaults and help to string in markdown format

        :return: str
        """
        header = """
            # scylla-cluster-tests configuration options

            #### Appending with environment variables or with config files
            * **strings:** can be appended with adding `++` at the beginning of the string:
                   `export SCT_APPEND_SCYLLA_ARGS="++ --overprovisioned 1"`
            * **list:** can be appended by adding `++` as the first item of the list
                   `export SCT_SCYLLA_D_OVERRIDES_FILES='["++", "extra_file/scylla.d/io.conf"]'`

        """

        def strip_help_text(text):
            """
            strip all lines, and also remove empty lines from start or end
            """
            output = [l.strip() for l in text.splitlines()]
            return '\n'.join(output[1 if not output[0] else 0:-1 if not output[-1] else None])

        ret = strip_help_text(header) + '\n'

        for opt in self.config_options:
            if opt['help']:
                help_text = '<br>'.join(strip_help_text(opt['help']).splitlines())
            else:
                help_text = ''
            appendable = ' (appendable)' if is_config_option_appendable(opt.get('name')) else ''
            default = self.get_default_value(opt['name'])
            default_text = default if default else 'N/A'
            ret += f"""## **{opt['name']}** / {opt['env']}\n\n{help_text}\n\n**default:** {default_text}\n\n**type:** {opt.get('type').__name__}{appendable}\n\n\n"""

        return ret

    def dump_help_config_yaml(self):
        """
        Dump all configuration options with their defaults and help to string in yaml format

        :return: str
        """
        ret = ""
        for opt in self.config_options:
            if opt['help']:
                help_text = '\n'.join(["# {}".format(l.strip()) for l in opt['help'].splitlines() if l.strip()]) + '\n'
            else:
                help_text = ''
            default = self.get_default_value(opt['name'])
            default = default if default else 'N/A'
            ret += "{help_text}{name}: {default}\n\n".format(help_text=help_text, default=default, **opt)

        return ret

    def _verify_data_volume_configuration(self, backend):
        dev_num = self.get("data_volume_disk_num")
        if dev_num == 0:
            return

        if backend not in ['aws', 'k8s-eks']:
            raise ValueError('Data volume configuration is supported only for aws, k8s-eks')

        if not self.get('data_volume_disk_size') or not self.get('data_volume_disk_type'):
            raise ValueError('Data volume configuration requires: data_volume_disk_type, data_volume_disk_size')

    def _verify_scylla_bench_mode_and_workload_parameters(self):
        # pylint: disable=too-many-nested-blocks
        for param_name in self.stress_cmd_params:
            stress_cmds = self.get(param_name)
            if stress_cmds is None:
                continue
            if isinstance(stress_cmds, str):
                stress_cmds = [stress_cmds]
            for stress_cmd in stress_cmds:
                if not stress_cmd:
                    continue
                if not isinstance(stress_cmd, list):
                    stress_cmd = [stress_cmd]  # noqa: PLW2901
                for cmd in stress_cmd:
                    cmd = cmd.strip(' ')  # noqa: PLW2901
                    if not cmd.startswith('scylla-bench'):
                        continue
                    if "-mode=" not in cmd:
                        raise ValueError(f"Scylla-bench command {cmd} doesn't have parameter -mode")
                    if "-workload=" not in cmd:
                        raise ValueError(f"Scylla-bench command {cmd} doesn't have parameter -workload")

    def _validate_docker_backend_parameters(self):
        if self.get("use_mgmt"):
            raise ValueError("Scylla Manager is not supported for docker backend")


def init_and_verify_sct_config() -> SCTConfiguration:
    sct_config = SCTConfiguration()
    sct_config.log_config()
    sct_config.verify_configuration()
    sct_config.verify_configuration_urls_validity()
    sct_config.get_version_based_on_conf()
    sct_config.check_required_files()
    return sct_config
