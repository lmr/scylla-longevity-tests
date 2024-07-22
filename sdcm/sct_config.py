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
from typing import List, Union, Set, Literal

from distutils.util import strtobool
import anyconfig
from argus.client.sct.types import Package
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated
from pydantic.functional_validators import BeforeValidator

from sdcm import sct_abs_path
import sdcm.provision.azure.utils as azure_utils
from sdcm.provision.aws.capacity_reservation import SCTCapacityReservation
from sdcm.utils import alternator
from sdcm.utils.aws_utils import get_arch_from_instance_type
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
from sdcm.utils.gce_utils import get_gce_image_tags
from sdcm.remote import LOCALRUNNER, shell_script_cmd
from sdcm.test_config import TestConfig
from sdcm.kafka.kafka_config import SctKafkaConfiguration
from sdcm.mgmt.common import RestoreParameters, AgentBackupParameters

logger = logging.getLogger(__name__)


class IgnoredType:
    pass


def _str(value: str | None) -> str | None:
    if value is None:
        return value
    if isinstance(value, str):
        return value
    raise ValueError(f"{value} isn't a string, it is '{type(value)}'")


String = Annotated[str | None, BeforeValidator(_str)]


def _file(value: str) -> str:
    file_path = pathlib.Path(value).expanduser()
    if file_path.is_file() and file_path.exists():
        return value
    raise ValueError(f"{value} isn't an existing file")


ExistingFile = Annotated[str, BeforeValidator(_file)]


def str_or_list_or_eval(value: Union[str, List[str]]) -> List[str]:
    """Convert an environment variable into a Python's list."""

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass
        return [str(value), ]

    if isinstance(value, list):
        ret_values = []
        for val in value:
            try:
                ret_values += [ast.literal_eval(val)]
            except Exception:  # pylint: disable=broad-except  # noqa: BLE001
                ret_values += [str(val)]
        return ret_values

    raise ValueError(f"{value} isn't a string or a list")


StringOrList = Annotated[str | list[str], BeforeValidator(str_or_list_or_eval)]


def int_or_list(value) -> int | list[int]:
    try:
        value = int(value)
        return value
    except Exception:  # pylint: disable=broad-except  # noqa: BLE001
        pass

    if isinstance(value, list):
        assert all(isinstance(v, int) for v in value)
        return value

    if isinstance(value, str):
        try:
            values = value.split()
            return [int(v) for v in values]  # pylint: disable=expression-not-assigned
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass
        try:
            return ast.literal_eval(value)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass

    raise ValueError("{} isn't int or list".format(value))


IntOrList = Annotated[int | list[int], BeforeValidator(int_or_list)]


def dict_or_str(value: dict | str) -> dict:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return value

    raise ValueError('"{}" isn\'t a dict'.format(value))


DictOrStr = Annotated[dict | str, BeforeValidator(dict_or_str)]


def dict_or_str_or_pydantic(value: dict | str | BaseModel) -> dict | BaseModel:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass

    if isinstance(value, (dict, BaseModel)):
        return value

    raise ValueError('"{}" isn\'t a dict, str or Pydantic model'.format(value))


DictOrStrOrPydantic = Annotated[dict | str | BaseModel, BeforeValidator(dict_or_str_or_pydantic)]


def _boolean(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return bool(strtobool(value))
    else:
        raise ValueError("{} isn't a boolean".format(type(value)))


Boolean = Annotated[bool, BeforeValidator(_boolean)]


def sct_field(*args, **kwargs):
    kwargs.setdefault('default', None)
    assert 'env' in kwargs
    return Field(*args, **kwargs)


available_backends: list[str] = [
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


class SCTConfiguration(BaseModel):
    """
    Class the hold the SCT configuration
    """

    multi_region_params: Annotated[list[str], IgnoredType] = Field(default=[], exclude=True)
    regions_data: Annotated[dict[str, dict[str, str]], IgnoredType] = Field(default={}, exclude=True)

    artifact_scylla_version: str | None = None
    is_enterprise: bool = False

    config_files: StringOrList = sct_field(
        description="a list of config files that would be used",
        env="SCT_CONFIG_FILES",
    )
    cluster_backend: String = sct_field(
        description="backend that will be used, aws/gce/docker",
        env="SCT_CLUSTER_BACKEND",
    )
    test_duration: int = sct_field(
        description="""
              Test duration (min). Parameter used to keep instances produced by tests
              and for jenkins pipeline timeout and TimoutThread.
        """,
        env="SCT_TEST_DURATION",
    )
    db_type: String = sct_field(
        description="Db type to install into db nodes, scylla/cassandra",
        env="SCT_DB_TYPE",
    )
    prepare_stress_duration: int = sct_field(
        description="""
              Time in minutes, which is required to run prepare stress commands
              defined in prepare_*_cmd for dataset generation, and is used in
              test duration calculation
         """,
        env="SCT_PREPARE_STRESS_DURATION",
    )
    stress_duration: int = sct_field(
        description="""
              Time in minutes, Time of execution for stress commands from stress_cmd parameters
              and is used in test duration calculation
        """,
        env="SCT_STRESS_DURATION",
    )
    n_db_nodes: IntOrList = sct_field(
        description="Number list of database nodes in multiple data centers.",
        env="SCT_N_DB_NODES",
    )
    n_test_oracle_db_nodes: IntOrList = sct_field(
        description="Number list of oracle test nodes in multiple data centers.",
        env="SCT_N_TEST_ORACLE_DB_NODES",
    )
    n_loaders: IntOrList = sct_field(
        description="Number list of loader nodes in multiple data centers",
        env="SCT_N_LOADERS",
    )
    n_monitor_nodes: IntOrList = sct_field(
        description="Number list of monitor nodes in multiple data centers",
        env="SCT_N_MONITORS_NODES",
    )
    intra_node_comm_public: Boolean = sct_field(
        description="If True, all communication between nodes are via public addresses",
        env="SCT_INTRA_NODE_COMM_PUBLIC",
    )
    endpoint_snitch: String = sct_field(
        description="""
            The snitch class scylla would use

            'GossipingPropertyFileSnitch' - default
            'Ec2MultiRegionSnitch' - default on aws backend
            'GoogleCloudSnitch'
         """,
        env="SCT_ENDPOINT_SNITCH",
    )
    user_credentials_path: ExistingFile = sct_field(
        description="""Path to your user credentials. qa key are downloaded automatically from S3 bucket""",
        env="SCT_USER_CREDENTIALS_PATH",
    )
    cloud_credentials_path: str = sct_field(
        description="""Path to your user credentials. qa key are downloaded automatically from S3 bucket""",
        env="SCT_CLOUD_CREDENTIALS_PATH",
    )
    cloud_cluster_id: int = sct_field(
        description="""scylla cloud cluster id""",
        env="SCT_CLOUD_CLUSTER_ID",
    )
    cloud_prom_bearer_token: String = sct_field(
        description="""scylla cloud promproxy bearer_token to federate monitoring data into our monitoring instance""",
        env="SCT_CLOUD_PROM_BEARER_TOKEN",
    )
    cloud_prom_path: String = sct_field(
        description="""scylla cloud promproxy path to federate monitoring data into our monitoring instance""",
        env="SCT_CLOUD_PROM_PATH",
    )
    cloud_prom_host: str = sct_field(
        description="""scylla cloud promproxy hostname to federate monitoring data into our monitoring instance""",
        env="SCT_CLOUD_PROM_HOST",
    )
    ip_ssh_connections: str = sct_field(
        description="""
            Type of IP used to connect to machine instances.
            This depends on whether you are running your tests from a machine inside
            your cloud provider, where it makes sense to use 'private', or outside (use 'public')

            Default: Use public IPs to connect to instances (public)
            Use private IPs to connect to instances (private)
            Use IPv6 IPs to connect to instances (ipv6)
         """,
        choices=("public", "private", "ipv6"),
        env="SCT_IP_SSH_CONNECTIONS",
    )

    scylla_repo: str = sct_field(
        description="Url to the repo of scylla version to install scylla. Can provide specific version after a colon "
        "e.g: `https://s3.amazonaws.com/downloads.scylladb.com/deb/ubuntu/scylla-2021.1.list:2021.1.18`",
        env="SCT_SCYLLA_REPO",
    )
    scylla_apt_keys: StringOrList = sct_field(
        description="APT keys for ScyllaDB repos",
        env="SCT_SCYLLA_APT_KEYS",
    )
    unified_package: str = sct_field(
        description="Url to the unified package of scylla version to install scylla",
        env="SCT_UNIFIED_PACKAGE",
    )
    nonroot_offline_install: Boolean = sct_field(
        description="Install Scylla without required root privilege",
        env="SCT_NONROOT_OFFLINE_INSTALL",
    )

    install_mode: str = sct_field(
        description="Scylla install mode, repo/offline/web",
        env="SCT_INSTALL_MODE",
    )

    scylla_version: str = sct_field(
        description="""Version of scylla to install, ex. '2.3.1'
                       Automatically lookup AMIs and repo links for formal versions.
                       WARNING: can't be used together with 'scylla_repo' or 'ami_id_db_scylla'""",
        env="SCT_SCYLLA_VERSION",
    )
    user_data_format_version: str = sct_field(
        description="""Format version of the user-data to use for scylla images,
                       default to what tagged on the image used""",
        env="SCT_USER_DATA_FORMAT_VERSION",
    )
    oracle_user_data_format_version: str = sct_field(
        description="""Format version of the user-data to use for scylla images,
                       default to what tagged on the image used""",
        env="SCT_ORACLE_USER_DATA_FORMAT_VERSION",
    )
    oracle_scylla_version: str = sct_field(
        description="""Version of scylla to use as oracle cluster with gemini tests, ex. '3.0.11'
                 Automatically lookup AMIs for formal versions.
                 WARNING: can't be used together with 'ami_id_db_oracle'""",
        env="SCT_ORACLE_SCYLLA_VERSION",
    )
    scylla_linux_distro: str = sct_field(
        description="""The distro name and family name to use. Example: 'ubuntu-jammy' or 'debian-bookworm'.""",
        env="SCT_SCYLLA_LINUX_DISTRO",
    )
    scylla_linux_distro_loader: str = sct_field(
        description="""The distro name and family name to use. Example: 'ubuntu-jammy' or 'debian-bookworm'.""",
        env="SCT_SCYLLA_LINUX_DISTRO_LOADER",
    )
    scylla_repo_m: str = sct_field(
        description="Url to the repo of scylla version to install scylla from for management tests",
        env="SCT_SCYLLA_REPO_M",
    )
    scylla_repo_loader: str = sct_field(
        description="Url to the repo of scylla version to install c-s for loader",
        env="SCT_SCYLLA_REPO_LOADER",
    )
    scylla_mgmt_address: str = sct_field(
        description="Url to the repo of scylla manager version to install for management tests",
        env="SCT_SCYLLA_MGMT_ADDRESS",
    )
    scylla_mgmt_agent_address: str = sct_field(
        description="Url to the repo of scylla manager agent version to install for management tests",
        env="SCT_SCYLLA_MGMT_AGENT_ADDRESS",
    )
    manager_version: str = sct_field(
        description="Branch of scylla manager server and agent to install. Options in defaults/manager_versions.yaml",
        env="SCT_MANAGER_VERSION",
    )
    target_manager_version: str = sct_field(
        description="Branch of scylla manager server and agent to upgrade to. Options in defaults/manager_versions.yaml",
        env="SCT_TARGET_MANAGER_VERSION",
    )
    manager_scylla_backend_version: str = sct_field(
        description="Branch of scylla db enterprise to install. Options in defaults/manager_versions.yaml",
        env="SCT_MANAGER_SCYLLA_BACKEND_VERSION",
    )
    scylla_mgmt_agent_version: str = sct_field(
        description="",
        env="SCT_SCYLLA_MGMT_AGENT_VERSION",
    )
    scylla_mgmt_pkg: str = sct_field(
        description="Url to the scylla manager packages to install for management tests",
        env="SCT_SCYLLA_MGMT_PKG",
    )
    use_cloud_manager: Boolean = sct_field(
        description="When define true, will install scylla cloud manager",
        env="SCT_USE_CLOUD_MANAGER",

    )
    use_mgmt: Boolean = sct_field(
        description="When define true, will install scylla management",
        env="SCT_USE_MGMT",

    )
    manager_prometheus_port: int = sct_field(
        description="Port to be used by the manager to contact Prometheus",
        env="SCT_MANAGER_PROMETHEUS_PORT",
    )
    target_scylla_mgmt_server_address: str = sct_field(
        description="Url to the repo of scylla manager version used to upgrade the manager server",
        env="SCT_TARGET_SCYLLA_MGMT_SERVER_ADDRESS",
    )
    target_scylla_mgmt_agent_address: str = sct_field(
        description="Url to the repo of scylla manager version used to upgrade the manager agents",
        env="SCT_TARGET_SCYLLA_MGMT_AGENT_ADDRESS",
    )
    use_ldap: Boolean = sct_field(
        description="When defined true, LDAP is going to be used.",
        env="SCT_USE_LDAP",

    )
    use_ldap_authorization: Boolean = sct_field(
        description="When defined true, will create a docker container with LDAP and configure scylla.yaml to use it",
        env="SCT_USE_LDAP_AUTHORIZATION",

    )
    use_ldap_authentication: Boolean = sct_field(
        description="When defined true, will create a docker container with LDAP and configure scylla.yaml to use it",
        env="SCT_USE_LDAP_AUTHENTICATION",

    )
    prepare_saslauthd: Boolean = sct_field(
        description="When defined true, will install and start saslauthd service",
        env="SCT_PREPARE_SASLAUTHD",

    )
    ldap_server_type: str = sct_field(
        description="This option indicates which server is going to be used for LDAP operations. [openldap, ms_ad]",
        env="SCT_LDAP_SERVER_TYPE",
    )
    parallel_node_operations: Boolean = sct_field(
        description="When defined true, will run node operations in parallel. Supported operations: startup",
        env="SCT_PARALLEL_NODE_OPERATIONS",

    )
    update_db_packages: str = sct_field(
        description="""A local directory of rpms to install a custom version on top of
                 the scylla installed (or from repo or from ami)""",
        env="SCT_UPDATE_DB_PACKAGES",
    )
    monitor_branch: String = sct_field(
        description="The port of scylla management",
        env="SCT_MONITOR_BRANCH",
    )
    user_prefix: str = sct_field(
        description="the prefix of the name of the cloud instances, defaults to username",
        env="SCT_USER_PREFIX",
    )
    ami_id_db_scylla_desc: str = sct_field(
        description="version name to report stats to Elasticsearch and tagged on cloud instances",
        env="SCT_AMI_ID_DB_SCYLLA_DESC",
    )
    sct_public_ip: str = sct_field(
        description="""
            Override the default hostname address of the sct test runner,
            for the monitoring of the Nemesis.
            can only work out of the box in AWS
        """,
        env="SCT_SCT_PUBLIC_IP",
    )
    client_encrypt_mtls: Boolean = sct_field(
        description="when enabled scylla will enforce mutual authentication when client-to-node encryption is enabled",
        env="SCT_CLIENT_ENCRYPT_MTLS",

    )
    server_encrypt_mtls: Boolean = sct_field(
        description="when enabled scylla will enforce mutual authentication when node-to-node encryption is enabled",
        env="SCT_SERVER_ENCRYPT_MTLS",

    )
    sct_ngrok_name: str = sct_field(
        description="Override the default hostname address of the sct test runner, using ngrok server, see readme for more instructions",
        env="SCT_NGROK_NAME",
    )
    backtrace_decoding: Boolean = sct_field(
        description="""If True, all backtraces found in db nodes would be decoded automatically""",
        env="SCT_BACKTRACE_DECODING",

    )
    print_kernel_callstack: Boolean = sct_field(
        description="""Scylla will print kernel callstack to logs if True, otherwise, it will try and may print a message
         that it failed to.""",
        env="SCT_PRINT_KERNEL_CALLSTACK",

    )
    instance_provision: Literal["spot", "on_demand", "spot_fleet", "spot_low_price"] = sct_field(
        description="instance_provision: spot|on_demand|spot_fleet",
        env="SCT_INSTANCE_PROVISION",
    )
    instance_provision_fallback_on_demand: Boolean = sct_field(
        description="instance_provision_fallback_on_demand: create instance on_demand provision type if instance with selected "
        "'instance_provision' type creation failed. "
        "Expected values: true|false (default - false",
        env="SCT_INSTANCE_PROVISION_FALLBACK_ON_DEMAND",

    )
    reuse_cluster: String = sct_field(
        description="""
        If reuse_cluster is set it should hold test_id of the cluster that will be reused.
        `reuse_cluster: 7dc6db84-eb01-4b61-a946-b5c72e0f6d71`
        """,
        env="SCT_REUSE_CLUSTER",
    )
    test_id: String = sct_field(
        description="""Set the test_id of the run manually. Use only from the env before running Hydra""",
        env="SCT_TEST_ID",
    )
    db_nodes_shards_selection: Literal["default", "random"] = sct_field(
        description="""How to select number of shards of Scylla. Expected values: default/random.
         Default value: 'default'.
         In case of random option - Scylla will start with different (random) shards on every node of the cluster
         """,
        env="SCT_NODES_SHARDS_SELECTION",
    )
    seeds_selector: Literal['random', 'first', 'all'] = sct_field(
        description="""How to select the seeds. Expected values: random/first/all""",
        env="SCT_SEEDS_SELECTOR",
    )
    seeds_num: int = sct_field(
        description="""Number of seeds to select""",
        env="SCT_SEEDS_NUM",
    )
    email_recipients: StringOrList = sct_field(
        description="""list of email of send the performance regression test to""",
        env="SCT_EMAIL_RECIPIENTS",
    )
    email_subject_postfix: str = sct_field(
        description="""Email subject postfix""",
        env="SCT_EMAIL_SUBJECT_POSTFIX",
    )
    enable_test_profiling: Boolean = sct_field(
        description="""Turn on sct profiling""",
        env="SCT_ENABLE_TEST_PROFILING",

    )
    ssh_transport: Literal["libssh2", "fabric"] = sct_field(
        description="""Set type of ssh library to use. Could be 'fabric' (default) or 'libssh2'""",
        env="SSH_TRANSPORT",
        default='libssh2',
    )

    # Scylla command line arguments options
    experimental_features: StringOrList = sct_field(
        description="unlock specified experimental features",
        env="SCT_EXPERIMENTAL_FEATURES",
    )
    server_encrypt: Boolean = sct_field(
        description="when enable scylla will use encryption on the server side",
        env="SCT_SERVER_ENCRYPT",

    )
    client_encrypt: Boolean = sct_field(
        description="when enable scylla will use encryption on the client side",
        env="SCT_CLIENT_ENCRYPT",

    )
    hinted_handoff: str = sct_field(
        description="when enable or disable scylla hinted handoff (enabled/disabled)",
        env="SCT_HINTED_HANDOFF",
    )
    nemesis_double_load_during_grow_shrink_duration: int = sct_field(
        description="After growing (and before shrink) in GrowShrinkCluster nemesis it will double the load for provided duration.",
        env="SCT_NEMESIS_DOUBLE_LOAD_DURING_GROW_SHRINK_DURATION",
    )
    authenticator: Literal[
        "PasswordAuthenticator",
        "AllowAllAuthenticator",
        "com.scylladb.auth.SaslauthdAuthenticator"] = sct_field(
        description="which authenticator scylla will use AllowAllAuthenticator/PasswordAuthenticator",
        env="SCT_AUTHENTICATOR",
    )
    authenticator_user: str = sct_field(
        description="the username if PasswordAuthenticator is used",
        env="SCT_AUTHENTICATOR_USER",
    )
    authenticator_password: str = sct_field(
        description="the password if PasswordAuthenticator is used",
        env="SCT_AUTHENTICATOR_PASSWORD",
    )
    authorizer: Literal["AllowAllAuthorizer", "CassandraAuthorizer"] = sct_field(
        description="which authorizer scylla will use AllowAllAuthorizer/CassandraAuthorizer",
        env="SCT_AUTHORIZER",
    )
    # Temporary solution. We do not want to run SLA nemeses during not-SLA test until the feature is stable
    sla: Boolean = sct_field(
        description="run SLA nemeses if the test is SLA only",
        env="SCT_SLA",
    )
    service_level_shares: list = sct_field(
        description="List if service level shares - how many server levels to create and test. Uses in SLA test. list of int, like: [100, 200]",
        env="SCT_SERVICE_LEVEL_SHARES",
    )
    alternator_port: int = sct_field(
        description="Port to configure for alternator in scylla.yaml",
        env="SCT_ALTERNATOR_PORT",
    )
    dynamodb_primarykey_type: str = sct_field(
        description="Type of dynamodb table to create with range key or not, can be: " + ','.join([schema.value for schema in alternator.enums.YCSBSchemaTypes]), choices=[schema.value for schema in alternator.enums.YCSBSchemaTypes],
        env="SCT_DYNAMODB_PRIMARYKEY_TYPE",
    )
    alternator_write_isolation: str = sct_field(
        description="Set the write isolation for the alternator table, see https://github.com/scylladb/scylla/blob/master/docs/alternator/alternator.md#write-isolation-policies for more details",
        env="SCT_ALTERNATOR_WRITE_ISOLATION",
    )
    alternator_use_dns_routing: Boolean = sct_field(
        description="If true, spawn a docker with a dns server for the ycsb loader to point to",
        env="SCT_ALTERNATOR_USE_DNS_ROUTING",

    )
    alternator_enforce_authorization: Boolean = sct_field(
        description="If true, enable the authorization check in dynamodb api (alternator)",
        env="SCT_ALTERNATOR_ENFORCE_AUTHORIZATION",

    )
    alternator_access_key_id: str = sct_field(
        description="the aws_access_key_id that would be used for alternator",
        env="SCT_ALTERNATOR_ACCESS_KEY_ID"
    )
    alternator_secret_access_key: str = sct_field(
        description="the aws_secret_access_key that would be used for alternator",
        env="SCT_ALTERNATOR_SECRET_ACCESS_KEY"
    )
    region_aware_loader: Boolean = sct_field(
        description="When in multi region mode, run stress on loader that is located in the same region as db node",
        env="SCT_REGION_AWARE_LOADER"
    )
    append_scylla_args: str = sct_field(
        description="More arguments to append to scylla command line",
        env="SCT_APPEND_SCYLLA_ARGS",
    )
    append_scylla_args_oracle: str = sct_field(
        description="More arguments to append to oracle command line",
        env="SCT_APPEND_SCYLLA_ARGS_ORACLE",
    )
    append_scylla_yaml: str = sct_field(
        description="More configuration to append to /etc/scylla/scylla.yaml",
        env="SCT_APPEND_SCYLLA_YAML",
    )

    # Nemesis config options
    nemesis_class_name: str = sct_field(
        description="""
                Nemesis class to use (possible types in sdcm.nemesis).
                Next syntax supporting:
                - nemesis_class_name: "NemesisName"  Run one nemesis in single thread
                - nemesis_class_name: "<NemesisName>:<num>" Run <NemesisName> in <num>
                  parallel threads on different nodes. Ex.: "ChaosMonkey:2"
                - nemesis_class_name: "<NemesisName1>:<num1> <NemesisName2>:<num2>" Run
                  <NemesisName1> in <num1> parallel threads and <NemesisName2> in <num2>
                  parallel threads. Ex.: "DisruptiveMonkey:1 NonDisruptiveMonkey:2"
        """,
        env="SCT_NEMESIS_CLASS_NAME",
        k8s_multitenancy_supported=True,
    )
    nemesis_interval: int = sct_field(
        description="""Nemesis sleep interval to use if None provided specifically in the test""",
        env="SCT_NEMESIS_INTERVAL",
        k8s_multitenancy_supported=True,
    )
    nemesis_sequence_sleep_between_ops: int = sct_field(
        description="""Sleep interval between nemesis operations for use in unique_sequence nemesis kind of tests""",
        env="SCT_NEMESIS_SEQUENCE_SLEEP_BETWEEN_OPS",
        k8s_multitenancy_supported=True,
    )
    nemesis_during_prepare: Boolean = sct_field(
        description="""Run nemesis during prepare stage of the test""",
        env="SCT_NEMESIS_DURING_PREPARE",

        k8s_multitenancy_supported=True,
    )
    nemesis_seed: int = sct_field(
        description="""A seed number in order to repeat nemesis sequence as part of SisyphusMonkey""",
        env="SCT_NEMESIS_SEED",
        k8s_multitenancy_supported=True,
    )
    nemesis_add_node_cnt: int = sct_field(
        description="""Add/remove nodes during GrowShrinkCluster nemesis""",
        env="SCT_NEMESIS_ADD_NODE_CNT",
        k8s_multitenancy_supported=True,
    )
    cluster_target_size: int = sct_field(
        description="""Used for scale test: max size of the cluster""",
        env="SCT_CLUSTER_TARGET_SIZE",
    )
    space_node_threshold: int = sct_field(
        description="""
             Space node threshold before starting nemesis (bytes)
             The default value is 6GB (6x1024^3 bytes)
             This value is supposed to reproduce
             https://github.com/scylladb/scylla/issues/1140
         """,
        env="SCT_SPACE_NODE_THRESHOLD",
        k8s_multitenancy_supported=True,
    )
    nemesis_filter_seeds: Boolean = sct_field(
        description="""If true runs the nemesis only on non seed nodes""",
        env="SCT_NEMESIS_FILTER_SEEDS",

        k8s_multitenancy_supported=True,
    )

    # Stress Commands
    stress_cmd: StringOrList = sct_field(
        description="cassandra-stress commands. You can specify everything but the -node parameter, which is going to be provided by the test suite infrastructure. multiple commands can passed as a list",
        env="SCT_STRESS_CMD",
        k8s_multitenancy_supported=True,
    )
    gemini_schema_url: str = sct_field(
        description="Url of the schema/configuration the gemini tool would use",
        env="SCT_GEMINI_SCHEMA_URL",
    )
    gemini_cmd: str = sct_field(
        description="gemini command to run (for now used only in GeminiTest)",
        env="SCT_GEMINI_CMD",
    )
    gemini_seed: int = sct_field(
        description="Seed number for gemini command",
        env="SCT_GEMINI_SEED",
    )
    gemini_table_options: list = sct_field(
        description="table options for created table. example: ['cdc={\'enabled\': true}'], ['cdc={\'enabled\': true}', 'compaction={\'class\': 'IncrementalCompactionStrategy\'}']",
        env="SCT_GEMINI_TABLE_OPTIONS",
    )
    # AWS config options
    instance_type_loader: str = sct_field(
        description="AWS image type of the loader node",
        env="SCT_INSTANCE_TYPE_LOADER",
    )
    instance_type_monitor: str = sct_field(
        description="AWS image type of the monitor node",
        env="SCT_INSTANCE_TYPE_MONITOR",
    )
    instance_type_db: str = sct_field(
        description="AWS image type of the db node",
        env="SCT_INSTANCE_TYPE_DB",
    )
    instance_type_db_oracle: str = sct_field(
        description="AWS image type of the oracle node",
        env="SCT_INSTANCE_TYPE_DB_ORACLE",
    )
    instance_type_runner: str = sct_field(
        description="instance type of the sct-runner node",
        env="SCT_INSTANCE_TYPE_RUNNER",
    )
    region_name: StringOrList = sct_field(
        description="AWS regions to use",
        env="SCT_REGION_NAME",
    )
    use_placement_group: Boolean = sct_field(
        description="if true, create 'cluster' placement group for test case "
        "for low-latency network performance achievement",
        env="SCT_USE_PLACEMENT_GROUP",

    )
    ami_id_db_scylla: str = sct_field(
        description="AMS AMI id to use for scylla db node",
        env="SCT_AMI_ID_DB_SCYLLA",
    )
    ami_id_loader: str = sct_field(
        description="AMS AMI id to use for loader node",
        env="SCT_AMI_ID_LOADER",
    )
    ami_id_monitor: str = sct_field(
        description="AMS AMI id to use for monitor node",
        env="SCT_AMI_ID_MONITOR",
    )
    ami_id_db_cassandra: str = sct_field(
        description="AMS AMI id to use for cassandra node",
        env="SCT_AMI_ID_DB_CASSANDRA",
    )
    ami_id_db_oracle: str = sct_field(
        description="AMS AMI id to use for oracle node",
        env="SCT_AMI_ID_DB_ORACLE",
    )
    root_disk_size_db: int = sct_field(
        description="",
        env="SCT_ROOT_DISK_SIZE_DB",
    )
    root_disk_size_monitor: int = sct_field(
        description="",
        env="SCT_ROOT_DISK_SIZE_MONITOR",
    )
    root_disk_size_loader: int = sct_field(
        description="",
        env="SCT_ROOT_DISK_SIZE_LOADER",
    )
    root_disk_size_runner: int = sct_field(
        description="root disk size in Gb for sct-runner",
        env="SCT_ROOT_DISK_SIZE_RUNNER",
    )
    ami_db_scylla_user: str = sct_field(
        description="",
        env="SCT_AMI_DB_SCYLLA_USER",
    )
    ami_monitor_user: str = sct_field(
        description="",
        env="SCT_AMI_MONITOR_USER",
    )
    ami_loader_user: str = sct_field(
        description="",
        env="SCT_AMI_LOADER_USER",
    )
    ami_db_cassandra_user: str = sct_field(
        description="",
        env="SCT_AMI_DB_CASSANDRA_USER",
    )
    spot_max_price: float = sct_field(
        description="The max percentage of the on demand price we set for spot/fleet instances",
        env="SCT_SPOT_MAX_PRICE",
    )
    extra_network_interface: Boolean = sct_field(
        description="if true, create extra network interface on each node",
        env="SCT_EXTRA_NETWORK_INTERFACE",

    )
    aws_instance_profile_name_db: str = sct_field(
        description="This is the name of the instance profile to set on all db instances",
        env="SCT_AWS_INSTANCE_PROFILE_NAME_DB",
    )
    aws_instance_profile_name_loader: str = sct_field(
        description="This is the name of the instance profile to set on all loader instances",
        env="SCT_AWS_INSTANCE_PROFILE_NAME_LOADER",
    )
    backup_bucket_backend: str = sct_field(
        description="the backend to be used for backup (e.g., 's3', 'gcs' or 'azure')",
        env="SCT_BACKUP_BUCKET_BACKEND",
    )
    backup_bucket_location: StringOrList = sct_field(
        description="the bucket name to be used for backup (e.g., 'manager-backup-tests')",
        env="SCT_BACKUP_BUCKET_LOCATION",
    )
    backup_bucket_region: str = sct_field(
        description="the AWS region of a bucket to be used for backup (e.g., 'eu-west-1')",
        env="SCT_BACKUP_BUCKET_REGION",
    )
    use_prepared_loaders: Boolean = sct_field(
        description="If True, we use prepared VMs for loader (instead of using docker images)",
        env="SCT_USE_PREPARED_LOADERS",

    )
    scylla_d_overrides_files: StringOrList = sct_field(
        description="list of files that should upload to /etc/scylla.d/ directory to override scylla config files",
        env="SCT_SCYLLA_D_OVERRIDES_FILES",
    )
    gce_project: String = sct_field(
        description="gcp project name to use",
        env="SCT_GCE_PROJECT",
    )
    gce_datacenter: String = sct_field(
        description="Supported: us-east1 - means that the zone will be selected automatically or you can mention the zone explicitly, for example: us-east1-b",
        env="SCT_GCE_DATACENTER",
    )
    gce_network: str = sct_field(
        description="gce network to use",
        env="SCT_GCE_NETWORK",
    )
    gce_image_db: str = sct_field(
        description="gce image to use for db nodes",
        env="SCT_GCE_IMAGE_DB",
    )
    gce_image_monitor: str = sct_field(
        description="gce image to use for monitor nodes",
        env="SCT_GCE_IMAGE_MONITOR",
    )
    scylla_network_config: list = sct_field(
        description="""Configure Scylla networking with single or multiple NIC/IP combinations.
              It must be defined for listen_address and rpc_address. For each address mandatory parameters are:
              - address: listen_address/rpc_address/broadcast_rpc_address/broadcast_address/test_communication
              - ip_type: ipv4 or ipv6
              - public: false or true
              - nic: number of NIC. 0, 1
              Supported for AWS only meanwhile""",
        env="SCT_SCYLLA_NETWORK_CONFIG",
    )
    gce_image_loader: str = sct_field(
        description="Google Compute Engine image to use for loader nodes",
        env="SCT_GCE_IMAGE_LOADER",
    )
    gce_image_username: str = sct_field(
        description="Username for the Google Compute Engine image",
        env="SCT_GCE_IMAGE_USERNAME",
    )
    gce_instance_type_loader: str = sct_field(
        description="Instance type for loader nodes in Google Compute Engine",
        env="SCT_GCE_INSTANCE_TYPE_LOADER",
    )
    gce_root_disk_type_loader: str = sct_field(
        description="Root disk type for loader nodes in Google Compute Engine",
        env="SCT_GCE_ROOT_DISK_TYPE_LOADER",
    )
    gce_n_local_ssd_disk_loader: int = sct_field(
        description="Number of local SSD disks for loader nodes in Google Compute Engine",
        env="SCT_GCE_N_LOCAL_SSD_DISK_LOADER",
    )
    gce_instance_type_monitor: str = sct_field(
        description="Instance type for monitor nodes in Google Compute Engine",
        env="SCT_GCE_INSTANCE_TYPE_MONITOR",
    )
    gce_root_disk_type_monitor: str = sct_field(
        description="Root disk type for monitor nodes in Google Compute Engine",
        env="SCT_GCE_ROOT_DISK_TYPE_MONITOR",
    )
    validate_large_collections: Boolean = sct_field(
        description="Flag to validate large collections in the database",
        env="SCT_VALIDATE_LARGE_COLLECTIONS",

    )
    run_commit_log_check_thread: Boolean = sct_field(
        description="Flag to run a thread that checks commit logs",
        env="SCT_RUN_COMMIT_LOG_CHECK_THREAD",

    )
    teardown_validators: DictOrStr = sct_field(
        description="Validators to use during teardown phase",
        env="SCT_TEARDOWN_VALIDATORS",
    )
    use_capacity_reservation: Boolean = sct_field(
        description="Flag to use capacity reservation for instances",
        env="SCT_USE_CAPACITY_RESERVATION",

    )
    bisect_start_date: str = sct_field(
        description="Start date for bisecting test runs to find regressions",
        env="SCT_BISECT_START_DATE",
    )
    bisect_end_date: str = sct_field(
        description="End date for bisecting test runs to find regressions",
        env="SCT_BISECT_END_DATE",
    )
    kafka_backend: Literal["localstack", "vm", "msk"] | None = sct_field(
        description="Type of Kafka backend to use",
        env="SCT_KAFKA_BACKEND",
    )
    kafka_connectors: list[SctKafkaConfiguration] = sct_field(
        description="Kafka connectors to use",
        env="SCT_KAFKA_CONNECTORS",
    )
    run_scylla_doctor: Boolean = sct_field(
        description="Flag to run Scylla Doctor tool",
        env="SCT_RUN_SCYLLA_DOCTOR",

    )
    gce_n_local_ssd_disk_monitor: int = sct_field(
        description="Number of local SSD disks for monitor nodes in Google Compute Engine",
        env="SCT_GCE_N_LOCAL_SSD_DISK_MONITOR",
    )
    gce_instance_type_db: str = sct_field(
        description="Instance type for database nodes in Google Compute Engine",
        env="SCT_GCE_INSTANCE_TYPE_DB",
    )
    gce_root_disk_type_db: str = sct_field(
        description="Root disk type for database nodes in Google Compute Engine",
        env="SCT_GCE_ROOT_DISK_TYPE_DB",
    )
    gce_n_local_ssd_disk_db: int = sct_field(
        description="Number of local SSD disks for database nodes in Google Compute Engine",
        env="SCT_GCE_N_LOCAL_SSD_DISK_DB",
    )
    gce_pd_standard_disk_size_db: int = sct_field(
        description="The size of the standard persistent disk in GB used for GCE database nodes",
        env="SCT_GCE_PD_STANDARD_DISK_SIZE_DB",
    )
    gce_pd_ssd_disk_size_db: int = sct_field(
        description="",
        env="SCT_GCE_PD_SSD_DISK_SIZE_DB",
    )
    gce_setup_hybrid_raid: Boolean = sct_field(
        description="If True, SCT configures a hybrid RAID of NVMEs and an SSD for scylla's data",
        env="SCT_GCE_SETUP_HYBRID_RAID",

    )
    gce_pd_ssd_disk_size_loader: int = sct_field(
        description="",
        env="SCT_GCE_PD_SSD_DISK_SIZE_LOADER",
    )
    gce_pd_ssd_disk_size_monitor: int = sct_field(
        description="",
        env="SCT_GCE_SSD_DISK_SIZE_MONITOR",
    )

    # azure options
    azure_region_name: StringOrList = sct_field(
        description="Azure region(s) where the resources will be deployed. Supports single or multiple regions.",
        env="SCT_AZURE_REGION_NAME",
    )
    azure_instance_type_loader: str = sct_field(
        description="The Azure virtual machine size to be used for loader nodes.",
        env="SCT_AZURE_INSTANCE_TYPE_LOADER",
    )

    azure_instance_type_monitor: str = sct_field(
        description="The Azure virtual machine size to be used for monitor nodes.",
        env="SCT_AZURE_INSTANCE_TYPE_MONITOR",
    )
    azure_instance_type_db: str = sct_field(
        description="The Azure virtual machine size to be used for database nodes.",
        env="SCT_AZURE_INSTANCE_TYPE_DB",
    )

    azure_instance_type_db_oracle: str = sct_field(
        description="The Azure virtual machine size to be used for Oracle database nodes.",
        env="SCT_AZURE_INSTANCE_TYPE_DB_ORACLE",
    )
    azure_image_db: str = sct_field(
        description="The Azure image to be used for database nodes.",
        env="SCT_AZURE_IMAGE_DB",
    )
    azure_image_monitor: str = sct_field(
        description="The Azure image to be used for monitor nodes.",
        env="SCT_AZURE_IMAGE_MONITOR",
    )
    azure_image_loader: str = sct_field(
        description="The Azure image to be used for loader nodes.",
        env="SCT_AZURE_IMAGE_LOADER",
    )
    azure_image_username: str = sct_field(
        description="The username for the Azure image.",
        env="SCT_AZURE_IMAGE_USERNAME",
    )

    # k8s-eks options
    eks_service_ipv4_cidr: str = sct_field(
        description="EKS service IPv4 CIDR block",
        env="SCT_EKS_SERVICE_IPV4_CIDR",
    )
    eks_vpc_cni_version: str = sct_field(
        description="EKS VPC CNI plugin version",
        env="SCT_EKS_VPC_CNI_VERSION",
    )
    eks_role_arn: str = sct_field(
        description="ARN of the IAM role for EKS",
        env="SCT_EKS_ROLE_ARN",
    )
    eks_cluster_version: str = sct_field(
        description="EKS cluster Kubernetes version",
        env="SCT_EKS_CLUSTER_VERSION",
    )
    eks_nodegroup_role_arn: str = sct_field(
        description="ARN of the IAM role for EKS node groups",
        env="SCT_EKS_NODEGROUP_ROLE_ARN",
    )

    # k8s-gke options
    gke_cluster_version: str = sct_field(
        description="Specifies the version of the GKE cluster to be used.",
        env="SCT_GKE_CLUSTER_VERSION",
    )
    gke_k8s_release_channel: str = sct_field(
        description="K8S release channel name to be used. Expected values are: 'rapid', 'regular', 'stable' and '' (static / No channel).",
        env="SCT_GKE_K8S_RELEASE_CHANNEL",
    )
    k8s_scylla_utils_docker_image: str = sct_field(
        description="Docker image to be used by Scylla operator to tune K8S nodes for performance. Used when 'k8s_enable_performance_tuning' is defined to 'True'. If not set then the default from operator will be used.",
        env="SCT_K8S_SCYLLA_UTILS_DOCKER_IMAGE",
    )
    k8s_enable_performance_tuning: Boolean = sct_field(
        description="Define whether performance tuning must run or not.",
        env="SCT_K8S_ENABLE_PERFORMANCE_TUNING",

    )
    k8s_deploy_monitoring: Boolean = sct_field(
        description="Determines if monitoring should be deployed alongside the Scylla cluster.",
        env="SCT_K8S_DEPLOY_MONITORING",

    )
    k8s_local_volume_provisioner_type: str = sct_field(
        description="Defines the type of the K8S local volume provisioner to be deployed. It may be either 'static' or 'dynamic'. Details about 'dynamic': 'dynamic': https://github.com/scylladb/k8s-local-volume-provisioner; 'static': sdcm/k8s_configs/static-local-volume-provisioner.yaml",
        env="SCT_K8S_LOCAL_VOLUME_PROVISIONER_TYPE",
    )
    k8s_scylla_operator_docker_image: str = sct_field(
        description="Docker image to be used for installation of Scylla operator.",
        env="SCT_K8S_SCYLLA_OPERATOR_DOCKER_IMAGE",
    )
    k8s_scylla_operator_upgrade_docker_image: str = sct_field(
        description="Docker image to be used for upgrade of Scylla operator.",
        env="SCT_K8S_SCYLLA_OPERATOR_UPGRADE_DOCKER_IMAGE",
    )
    k8s_scylla_operator_helm_repo: str = sct_field(
        description="Link to the Helm repository where to get 'scylla-operator' charts from.",
        env="SCT_K8S_SCYLLA_OPERATOR_HELM_REPO",
    )
    k8s_scylla_operator_upgrade_helm_repo: str = sct_field(
        description="Link to the Helm repository where to get 'scylla-operator' charts for upgrade.",
        env="SCT_K8S_SCYLLA_OPERATOR_UPGRADE_HELM_REPO",
    )
    k8s_scylla_operator_chart_version: str = sct_field(
        description="Version of 'scylla-operator' Helm chart to use. If not set then latest one will be used.",
        env="SCT_K8S_SCYLLA_OPERATOR_CHART_VERSION",
    )
    k8s_scylla_operator_upgrade_chart_version: str = sct_field(
        description="Version of 'scylla-operator' Helm chart to use for upgrade.",
        env="SCT_K8S_SCYLLA_OPERATOR_UPGRADE_CHART_VERSION",
    )
    k8s_functional_test_dataset: str = sct_field(
        description="Defines whether dataset uses for pre-fill cluster in functional test. Defined in sdcm.utils.sstable.load_inventory. Expected values: BIG_SSTABLE_MULTI_COLUMNS_DATA, MULTI_COLUMNS_DATA",
        env="SCT_K8S_FUNCTIONAL_TEST_DATASET",
    )
    k8s_scylla_cpu_limit: str = sct_field(
        description="The CPU limit that will be set for each Scylla cluster deployed in K8S. If not set, then will be autocalculated. Example: '500m' or '2'",
        env="SCT_K8S_SCYLLA_CPU_LIMIT",
    )
    k8s_scylla_memory_limit: str = sct_field(
        description="The memory limit that will be set for each Scylla cluster deployed in K8S. If not set, then will be autocalculated. Example: '16384Mi'",
        env="SCT_K8S_SCYLLA_MEMORY_LIMIT",
    )
    k8s_scylla_cluster_name: str = sct_field(
        description="Specifies the name of the Scylla cluster to be deployed in K8S.",
        env="SCT_K8S_SCYLLA_CLUSTER_NAME",
    )
    k8s_n_scylla_pods_per_cluster: int = sct_field(
        description="Number of Scylla pods per cluster.",
        env="K8S_N_SCYLLA_PODS_PER_CLUSTER",
    )
    k8s_scylla_disk_gi: int = sct_field(
        description="Specifies the disk size in GiB for Scylla pods.",
        env="SCT_K8S_SCYLLA_DISK_GI",
    )
    k8s_scylla_disk_class: str = sct_field(
        description="Specifies the disk class for Scylla pods.",
        env="SCT_K8S_SCYLLA_DISK_CLASS",
    )
    k8s_loader_cluster_name: str = sct_field(
        description="Specifies the name of the loader cluster.",
        env="SCT_K8S_LOADER_CLUSTER_NAME",
    )
    k8s_n_loader_pods_per_cluster: int = sct_field(
        description="Number of loader pods per loader cluster.",
        env="SCT_K8S_N_LOADER_PODS_PER_CLUSTER",
    )
    k8s_loader_run_type: str = sct_field(
        description="Defines how the loader pods must run. It may be either 'static' (default, run stress command on the constantly existing idle pod having reserved resources, perf-oriented) or 'dynamic' (run stress command in a separate pod as main thread and get logs in a separate retryable API call not having resource reservations).",
        env="SCT_K8S_LOADER_RUN_TYPE",
    )
    k8s_instance_type_auxiliary: str = sct_field(
        description="Instance type for the nodes of the K8S auxiliary/default node pool.",
        env="SCT_K8S_INSTANCE_TYPE_AUXILIARY",
    )
    k8s_instance_type_monitor: str = sct_field(
        description="Instance type for the nodes of the K8S monitoring node pool.",
        env="SCT_K8S_INSTANCE_TYPE_MONITOR",
    )
    mini_k8s_version: str = sct_field(
        description="Specifies the version of the mini K8S cluster to be used.",
        env="SCT_MINI_K8S_VERSION",
    )
    k8s_cert_manager_version: str = sct_field(
        description="Specifies the version of the cert-manager to be used in K8S.",
        env="SCT_K8S_CERT_MANAGER_VERSION",
    )
    k8s_minio_storage_size: str = sct_field(
        description="Specifies the storage size for MinIO deployment in K8S.",
        env="SCT_K8S_MINIO_STORAGE_SIZE",
    )
    k8s_log_api_calls: Boolean = sct_field(
        description="Defines whether the K8S API server logging must be enabled and its logs gathered. Be aware that it may be a really huge set of data.",
        env="SCT_K8S_LOG_API_CALLS",

    )
    k8s_tenants_num: int = sct_field(
        description="Number of Scylla clusters to create in the K8S cluster.",
        env="SCT_TENANTS_NUM",
    )
    k8s_enable_tls: Boolean = sct_field(
        description="Defines whether to enable the operator serverless options.",
        env="SCT_K8S_ENABLE_TLS",

    )
    k8s_enable_sni: Boolean = sct_field(
        description="Defines whether we install SNI and use it or not (serverless feature).",
        env="SCT_K8S_ENABLE_SNI",

    )
    k8s_enable_alternator: Boolean = sct_field(
        description="Defines whether we enable the alternator feature using scylla-operator or not.",
        env="SCT_K8S_ENABLE_ALTERNATOR",

    )
    k8s_connection_bundle_file: str = sct_field(
        description="Serverless configuration bundle file.",
        env="SCT_K8S_CONNECTION_BUNDLE_FILE",
    )
    k8s_db_node_service_type: str = sct_field(
        description="Defines the type of the K8S 'Service' objects type used for ScyllaDB pods. Empty value means 'do not set and allow scylla-operator to choose'.",
        env="SCT_K8S_DB_NODE_SERVICE_TYPE",
    )
    k8s_db_node_to_node_broadcast_ip_type: str = sct_field(
        description="Defines the source of the IP address to be used for the 'broadcast_address' config option in the 'scylla.yaml' files. Empty value means 'do not set and allow scylla-operator to choose'.",
        env="SCT_K8S_DB_NODE_TO_NODE_BROADCAST_IP_TYPE",
    )
    k8s_db_node_to_client_broadcast_ip_type: str = sct_field(
        description="Defines the source of the IP address to be used for the 'broadcast_rpc_address' config option in the 'scylla.yaml' files. Empty value means 'do not set and allow scylla-operator to choose'.",
        env="SCT_K8S_DB_NODE_TO_CLIENT_BROADCAST_IP_TYPE",
    )
    k8s_use_chaos_mesh: Boolean = sct_field(
        description="Enables chaos-mesh for K8S testing.",
        env="SCT_K8S_USE_CHAOS_MESH",

    )
    k8s_n_auxiliary_nodes: int = sct_field(
        description="Number of nodes in the auxiliary pool.",
        env="SCT_K8S_N_AUXILIARY_NODES",
    )
    k8s_n_monitor_nodes: int = sct_field(
        description="Number of nodes in the monitoring pool that will be used for scylla-operator's deployed monitoring pods.",
        env="SCT_K8S_N_MONITOR_NODES",
    )

    # docker config options
    mgmt_docker_image: str = sct_field(
        description="Scylla manager docker image, i.e. 'scylladb/scylla-manager:2.2.1'",
        env="SCT_MGMT_DOCKER_IMAGE",
    )
    docker_image: str = sct_field(
        description="Scylla docker image repo, i.e. 'scylladb/scylla', if omitted is calculated from scylla_version",
        env="SCT_DOCKER_IMAGE",
    )
    docker_network: str = sct_field(
        description="Local docker network to use, if there's need to have db cluster connect to other services running in docker",
        env="SCT_DOCKER_NETWORK",
    )

    # baremetal config options

    s3_baremetal_config: str = sct_field(
        description="Configuration for S3 in baremetal setups. This includes details such as endpoint URL, access key, secret key, and bucket name.",
        env="SCT_S3_BAREMETAL_CONFIG",
    )
    db_nodes_private_ip: StringOrList = sct_field(
        description="Private IP addresses of DB nodes. Can be a single IP, a list of IPs, or an expression that evaluates to a list.",
        env="SCT_DB_NODES_PRIVATE_IP",
    )
    db_nodes_public_ip: StringOrList = sct_field(
        description="Public IP addresses of DB nodes. Can be a single IP, a list of IPs, or an expression that evaluates to a list.",
        env="SCT_DB_NODES_PUBLIC_IP",
    )
    loaders_private_ip: StringOrList = sct_field(
        description="Private IP addresses of loader nodes. Loaders are used for running stress tests or other workloads against the DB. Can be a single IP, a list of IPs, or an expression that evaluates to a list.",
        env="SCT_LOADERS_PRIVATE_IP",
    )
    loaders_public_ip: StringOrList = sct_field(
        description="Public IP addresses of loader nodes. These IPs are used for accessing the loaders from outside the private network. Can be a single IP, a list of IPs, or an expression that evaluates to a list.",
        env="SCT_LOADERS_PUBLIC_IP",
    )
    monitor_nodes_private_ip: StringOrList = sct_field(
        description="Private IP addresses of monitor nodes. Monitoring nodes host monitoring tools like Prometheus and Grafana for DB performance monitoring. Can be a single IP, a list of IPs, or an expression that evaluates to a list.",
        env="SCT_MONITOR_NODES_PRIVATE_IP",
    )
    monitor_nodes_public_ip: StringOrList = sct_field(
        description="Public IP addresses of monitor nodes. These IPs are used for accessing the monitoring tools from outside the private network. Can be a single IP, a list of IPs, or an expression that evaluates to a list.",
        env="SCT_MONITOR_NODES_PUBLIC_IP",
    )
    # test specific config parameters

    # GrowClusterTest
    cassandra_stress_population_size: int = sct_field(
        description="The total population size over which the Cassandra stress tests are run.",
        env="SCT_CASSANDRA_STRESS_POPULATION_SIZE",

    )
    cassandra_stress_threads: int = sct_field(
        description="The number of threads used by Cassandra stress tests.",
        env="SCT_CASSANDRA_STRESS_THREADS",

    )
    add_node_cnt: int = sct_field(
        description="The number of nodes to add during the test.",
        env="SCT_ADD_NODE_CNT",

    )

    # LongevityTest
    stress_multiplier: int = sct_field(
        description="Multiplier for stress command intensity",
        env="SCT_STRESS_MULTIPLIER",
    )
    stress_multiplier_w: int = sct_field(
        description="Write stress command intensity multiplier",
        env="SCT_STRESS_MULTIPLIER_W",
    )
    stress_multiplier_r: int = sct_field(
        description="Read stress command intensity multiplier",
        env="SCT_STRESS_MULTIPLIER_R",
    )
    stress_multiplier_m: int = sct_field(
        description="Mixed operations stress command intensity multiplier",
        env="SCT_STRESS_MULTIPLIER_M",
    )
    run_fullscan: list = sct_field(
        description="Enable or disable running full scans during tests",
        env="SCT_RUN_FULLSCAN",
    )
    run_full_partition_scan: str = sct_field(
        description="Enable or disable running full partition scans during tests",
        env="SCT_run_full_partition_scan",
    )
    run_tombstone_gc_verification: str = sct_field(
        description="Enable or disable tombstone garbage collection verification during tests",
        env="SCT_RUN_TOMBSTONE_GC_VERIFICATION",
    )
    keyspace_num: int = sct_field(
        description="Number of keyspaces to use in the test",
        env="SCT_KEYSPACE_NUM",
    )
    round_robin: Boolean = sct_field(
        description="Enable or disable round robin selection of nodes for operations",
        env="SCT_ROUND_ROBIN",
        k8s_multitenancy_supported=True,

    )
    batch_size: int = sct_field(
        description="Batch size for operations",
        env="SCT_BATCH_SIZE",
    )
    pre_create_schema: Boolean = sct_field(
        description="Enable or disable pre-creation of schema before running workload",
        env="SCT_PRE_CREATE_SCHEMA",

    )
    pre_create_keyspace: StringOrList = sct_field(
        description="Command to create keyspace to be pre-created before running workload",
        env="SCT_PRE_CREATE_KEYSPACE",
    )
    post_prepare_cql_cmds: StringOrList = sct_field(
        description="CQL Commands to run after prepare stage finished (relevant only to longevity_test.py)",
        env="SCT_POST_PREPARE_CQL_CMDS",
    )
    prepare_wait_no_compactions_timeout: int = sct_field(
        description="Time to wait for compaction to finish at the end of prepare stage. Use only when compaction affects the test or load",
        env="SCT_PREPARE_WAIT_NO_COMPACTIONS_TIMEOUT",
    )
    compaction_strategy: str = sct_field(
        description="Compaction strategy to use for pre-created schema",
        env="SCT_COMPACTION_STRATEGY",
    )
    sstable_size: int = sct_field(
        description="Configure sstable size for pre-create-schema mode",
        env="SSTABLE_SIZE",
    )
    cluster_health_check: Boolean = sct_field(
        description="Enable or disable starting cluster health checker for all nodes",
        env="SCT_CLUSTER_HEALTH_CHECK",

    )
    data_validation: str = sct_field(
        description="Specify the type of data validation to perform",
        env="SCT_DATA_VALIDATION",
    )
    stress_read_cmd: StringOrList = sct_field(
        description="""cassandra-stress commands.
            You can specify everything but the -node parameter, which is going to
            be provided by the test suite infrastructure.
            multiple commands can passed as a list""",
        env="SCT_STRESS_READ_CMD",
    )
    prepare_verify_cmd: StringOrList = sct_field(
        description="""cassandra-stress commands.
            You can specify everything but the -node parameter, which is going to
            be provided by the test suite infrastructure.
            multiple commands can passed as a list""",
        env="SCT_PREPARE_VERIFY_CMD",
    )
    user_profile_table_count: int = sct_field(
        description="Number of user profile tables to create for the test",
        env="SCT_USER_PROFILE_TABLE_COUNT",
    )

    # MgmtCliTest
    scylla_mgmt_upgrade_to_repo: str = sct_field(
        description="Url to the repo of scylla manager version to upgrade to for management tests",
        env="SCT_SCYLLA_MGMT_UPGRADE_TO_REPO",
    )
    mgmt_restore_params: DictOrStrOrPydantic = sct_field(
        description="Manager restore operation specific parameters: batch_size, parallel. For example, {'batch_size': 100, 'parallel': 10}",
        env="SCT_MGMT_RESTORE_PARAMS",
    )
    mgmt_agent_backup_config: DictOrStrOrPydantic = sct_field(
        description="Manager agent backup general configuration: checkers, transfers, low_level_retries. For example, {'checkers': 100, 'transfers': 2, 'low_level_retries': 20}",
        env="SCT_MGMT_AGENT_BACKUP_CONFIG",
    )

    # PerformanceRegressionTest
    stress_cmd_w: StringOrList = sct_field(
        description="cassandra-stress commands. You can specify everything but the -node parameter, which is going to be provided by the test suite infrastructure. Multiple commands can be passed as a list",
        env="SCT_STRESS_CMD_W",
        k8s_multitenancy_supported=True,
    )
    stress_cmd_r: StringOrList = sct_field(
        description="cassandra-stress commands. You can specify everything but the -node parameter, which is going to be provided by the test suite infrastructure. Multiple commands can be passed as a list",
        env="SCT_STRESS_CMD_R",
        k8s_multitenancy_supported=True,
    )
    stress_cmd_m: StringOrList = sct_field(
        description="cassandra-stress commands. You can specify everything but the -node parameter, which is going to be provided by the test suite infrastructure. Multiple commands can be passed as a list",
        env="SCT_STRESS_CMD_M",
        k8s_multitenancy_supported=True,
    )
    prepare_write_cmd: StringOrList = sct_field(
        description="cassandra-stress commands. You can specify everything but the -node parameter, which is going to be provided by the test suite infrastructure. Multiple commands can be passed as a list",
        env="SCT_PREPARE_WRITE_CMD",
        k8s_multitenancy_supported=True,
    )
    stress_cmd_no_mv: StringOrList = sct_field(
        description="cassandra-stress commands. You can specify everything but the -node parameter, which is going to be provided by the test suite infrastructure. Multiple commands can be passed as a list",
        env="SCT_STRESS_CMD_NO_MV",
    )
    stress_cmd_no_mv_profile: str = sct_field(
        description="",
        env="SCT_STRESS_CMD_NO_MV_PROFILE",
    )
    cs_user_profiles: StringOrList = sct_field(
        description="cassandra-stress user-profiles list. Executed in test step",
        env="SCT_CS_USER_PROFILES",
    )
    prepare_cs_user_profiles: StringOrList = sct_field(
        description="cassandra-stress user-profiles list. Executed in prepare step",
        env="SCT_PREPARE_CS_USER_PROFILES",
    )
    cs_duration: str = sct_field(
        description="",
        env="SCT_CS_DURATION",
    )
    cs_debug: Boolean = sct_field(
        description="enable debug for cassandra-stress",
        env="SCT_CS_DEBUG",

    )
    stress_cmd_mv: StringOrList = sct_field(
        description="cassandra-stress commands. You can specify everything but the -node parameter, which is going to be provided by the test suite infrastructure. Multiple commands can be passed as a list",
        env="SCT_STRESS_CMD_MV",
    )
    prepare_stress_cmd: StringOrList = sct_field(
        description="cassandra-stress commands. You can specify everything but the -node parameter, which is going to be provided by the test suite infrastructure. Multiple commands can be passed as a list",
        env="SCT_PREPARE_STRESS_CMD",
    )

    # PerformanceRegressionLWTTest
    stress_cmd_lwt_i: str = sct_field(
        description="Stress command for LWT performance test for INSERT baseline",
        env="SCT_STRESS_CMD_LWT_I",
    )
    stress_cmd_lwt_d: str = sct_field(
        description="Stress command for LWT performance test for DELETE baseline",
        env="SCT_STRESS_CMD_LWT_D",
    )
    stress_cmd_lwt_u: str = sct_field(
        description="Stress command for LWT performance test for UPDATE baseline",
        env="SCT_STRESS_CMD_LWT_U",
    )
    stress_cmd_lwt_ine: str = sct_field(
        description="Stress command for LWT performance test for INSERT with IF NOT EXISTS",
        env="SCT_STRESS_CMD_LWT_INE",
    )
    stress_cmd_lwt_uc: str = sct_field(
        description="Stress command for LWT performance test for UPDATE with IF <condition>",
        env="SCT_STRESS_CMD_LWT_UC",
    )
    stress_cmd_lwt_ue: str = sct_field(
        description="Stress command for LWT performance test for UPDATE with IF EXISTS",
        env="SCT_STRESS_CMD_LWT_UE",
    )

    stress_cmd_lwt_de: str = sct_field(
        description="Stress command for LWT performance test for DELETE with IF EXISTS",
        env="SCT_STRESS_CMD_LWT_DE",
    )
    stress_cmd_lwt_dc: str = sct_field(
        description="Stress command for LWT performance test for DELETE with IF <condition>",
        env="SCT_STRESS_CMD_LWT_DC",
    )
    stress_cmd_lwt_mixed: str = sct_field(
        description="Stress command for LWT performance test for mixed lwt load",
        env="SCT_STRESS_CMD_LWT_MIXED",
    )
    stress_cmd_lwt_mixed_baseline: str = sct_field(
        description="Stress command for LWT performance test for mixed lwt load baseline",
        env="SCT_STRESS_CMD_LWT_MIXED_BASELINE",
    )

    # RefreshTest
    skip_download: Boolean = sct_field(description="", env="SCT_SKIP_DOWNLOAD")
    sstable_file: str = sct_field(description="", env="SCT_SSTABLE_FILE")
    sstable_url: str = sct_field(description="", env="SCT_SSTABLE_URL")
    sstable_md5: str = sct_field(description="", env="SCT_SSTABLE_MD5")
    flush_times: int = sct_field(description="", env="SCT_FLUSH_TIMES")
    flush_period: int = sct_field(description="", env="SCT_FLUSH_PERIOD")

    # UpgradeTest
    new_scylla_repo: str = sct_field(
        description="URL to the Scylla repository for new versions.",
        env="SCT_NEW_SCYLLA_REPO",
    )
    new_version: str = sct_field(
        description="Assign new upgrade version, use it to upgrade to specific minor release. eg: 3.0.1",
        env="SCT_NEW_VERSION",
    )
    target_upgrade_version: str = sct_field(
        description="The target version to upgrade Scylla to.",
        env="SCT_TARGET_UPGRADE_VERSION"
    )
    disable_raft: Boolean = sct_field(
        description="Flag to disable Raft consensus for LWT operations.",
        env="SCT_DISABLE_RAFT",
    )
    enable_tablets_on_upgrade: Boolean = sct_field(
        description="By default, the tablets feature is disabled. With this parameter, created for the upgrade test, the tablets feature will only be enabled after the upgrade",
        env="SCT_ENABLE_TABLETS_ON_UPGRADE",

    )
    upgrade_node_packages: str = sct_field(
        description="Specifies the packages to be upgraded on the node.",
        env="SCT_UPGRADE_NODE_PACKAGES"
    )
    test_sst3: Boolean = sct_field(
        description="Flag to enable testing with SSTable format version 3.",
        env="SCT_TEST_SST3",
    )
    test_upgrade_from_installed_3_1_0: Boolean = sct_field(
        description="Flag to test upgrade from an installed version of 3.1.0.",
        env="SCT_TEST_UPGRADE_FROM_INSTALLED_3_1_0",
    )
    recover_system_tables: Boolean = sct_field(
        description="Flag to enable recovery of system tables after upgrade.",
        env="SCT_RECOVER_SYSTEM_TABLES",
    )
    stress_cmd_1: StringOrList = sct_field(
        description="Primary stress command to be executed.",
        env="SCT_STRESS_CMD_1",
    )
    stress_cmd_complex_prepare: StringOrList = sct_field(
        description="Stress command for complex preparation steps.",
        env="SCT_STRESS_CMD_COMPLEX_PREPARE",
    )
    prepare_write_stress: StringOrList = sct_field(
        description="Stress command to prepare write operations.",
        env="SCT_PREPARE_WRITE_STRESS",
    )
    stress_cmd_read_10m: StringOrList = sct_field(
        description="Stress command to perform read operations for 10 minutes.",
        env="SCT_STRESS_CMD_READ_10M",
    )
    stress_cmd_read_cl_one: StringOrList = sct_field(
        description="Stress command to perform read operations with consistency level ONE.",
        env="SCT_STRESS_CMD_READ_CL_ONE",
    )
    stress_cmd_read_60m: StringOrList = sct_field(
        description="Stress command to perform read operations for 60 minutes.",
        env="SCT_STRESS_CMD_READ_60M",
    )
    stress_cmd_complex_verify_read: StringOrList = sct_field(
        description="Stress command to verify complex read operations.",
        env="SCT_STRESS_CMD_COMPLEX_VERIFY_READ",
    )
    stress_cmd_complex_verify_more: StringOrList = sct_field(
        description="Additional stress command to verify complex operations.",
        env="SCT_STRESS_CMD_COMPLEX_VERIFY_MORE",
    )
    write_stress_during_entire_test: StringOrList = sct_field(
        description="Stress command to perform write operations throughout the entire test.",
        env="SCT_WRITE_STRESS_DURING_ENTIRE_TEST",
    )
    verify_data_after_entire_test: StringOrList = sct_field(
        description="Stress command to verify data integrity after the entire test.",
        env="SCT_VERIFY_DATA_AFTER_ENTIRE_TEST",
    )
    stress_cmd_read_cl_quorum: StringOrList = sct_field(
        description="Stress command to perform read operations with consistency level QUORUM.",
        env="SCT_STRESS_CMD_READ_CL_QUORUM",
    )
    verify_stress_after_cluster_upgrade: StringOrList = sct_field(
        description="""cassandra-stress commands.
        You can specify everything but the -node parameter, which is going to
        be provided by the test suite infrastructure.
        multiple commands can passed as a list""",
        env="SCT_VERIFY_STRESS_AFTER_CLUSTER_UPGRADE",
    )
    stress_cmd_complex_verify_delete: StringOrList = sct_field(
        description="""cassandra-stress commands.
        You can specify everything but the -node parameter, which is going to
        be provided by the test suite infrastructure.
        multiple commands can passed as a list""",
        env="SCT_STRESS_CMD_COMPLEX_VERIFY_DELETE",
    )
    scylla_encryption_options: String = sct_field(
        description="options will be used for enable encryption at-rest for tables",
        env="SCT_SCYLLA_ENCRYPTION_OPTIONS",
    )
    kms_key_rotation_interval: int = sct_field(
        description="The time interval in minutes which gets waited before the KMS key rotation happens."
        " Applied when the AWS KMS service is configured to be used.",
        env="SCT_KMS_KEY_ROTATION_INTERVAL",
    )
    enterprise_disable_kms: Boolean = sct_field(
        description="An escape hatch to disable KMS for enterprise run, when needed. We enable KMS by default since if we use Scylla 2023.1.3 and up",
        env="SCT_ENTERPRISE_DISABLE_KMS",

    )
    logs_transport: Literal["ssh", "docker", "syslog-ng"] = sct_field(
        description="How to transport logs: syslog-ng, ssh or docker",
        env="SCT_LOGS_TRANSPORT",
    )
    collect_logs: Boolean = sct_field(
        description="Collect logs from instances and sct runner",
        env="SCT_COLLECT_LOGS",

    )
    execute_post_behavior: Boolean = sct_field(
        description="Run post behavior actions in sct teardown step",
        env="SCT_EXECUTE_POST_BEHAVIOR",

    )
    post_behavior_db_nodes: Literal["destroy", "keep", "keep-on-failure"] = sct_field(
        description="""
            Failure/post test behavior, i.e. what to do with the db cloud instances at the end of the test.

            'destroy' - Destroy instances and credentials (default)
            'keep' - Keep instances running and leave credentials alone
            'keep-on-failure' - Keep instances if testrun failed
         """,
        env="SCT_POST_BEHAVIOR_DB_NODES",
    )
    post_behavior_loader_nodes: Literal["destroy", "keep", "keep-on-failure"] = sct_field(
        description="""
            Failure/post test behavior, i.e. what to do with the loader cloud instances at the end of the test.

            'destroy' - Destroy instances and credentials (default)
            'keep' - Keep instances running and leave credentials alone
            'keep-on-failure' - Keep instances if testrun failed
         """,
        env="SCT_POST_BEHAVIOR_LOADER_NODES",
    )
    post_behavior_monitor_nodes: Literal["destroy", "keep", "keep-on-failure"] = sct_field(
        description="""
            Failure/post test behavior, i.e. what to do with the monitor cloud instances at the end of the test.

            'destroy' - Destroy instances and credentials (default)
            'keep' - Keep instances running and leave credentials alone
            'keep-on-failure' - Keep instances if testrun failed
         """,
        env="SCT_POST_BEHAVIOR_MONITOR_NODES",
    )
    post_behavior_k8s_cluster: Literal["destroy", "keep", "keep-on-failure"] = sct_field(
        description="""
        Failure/post test behavior, i.e. what to do with the k8s cluster at the end of the test.

        'destroy' - Destroy k8s cluster and credentials (default)
        'keep' - Keep k8s cluster running and leave credentials alone
        'keep-on-failure' - Keep k8s cluster if testrun failed
        """,
        env="SCT_POST_BEHAVIOR_K8S_CLUSTER",
    )

    internode_compression: str = sct_field(
        description="Scylla option: internode_compression.", env="SCT_INTERNODE_COMPRESSION")
    internode_encryption: str = sct_field(
        description="Scylla sub option of server_encryption_options: internode_encryption.", env="SCT_INTERNODE_ENCRYPTION")
    jmx_heap_memory: int = sct_field(
        description="The total size of the memory allocated to JMX. Values in MB, so for 1GB enter 1024(MB).", env="SCT_JMX_HEAP_MEMORY")

    loader_swap_size: int = sct_field(
        description="The size of the swap file for the loaders. Its size in bytes calculated by x * 1MB",
        env="SCT_LOADER_SWAP_SIZE",
    )
    monitor_swap_size: int = sct_field(
        description="The size of the swap file for the monitors. Its size in bytes calculated by x * 1MB",
        env="SCT_MONITOR_SWAP_SIZE",
    )
    store_perf_results: Boolean = sct_field(
        description="""A flag that indicates whether or not to gather the prometheus stats at the end of the run. Intended to be used in performance testing""",
        env="SCT_STORE_PERF_RESULTS",

    )
    append_scylla_setup_args: str = sct_field(
        description="More arguments to append to scylla_setup command line",
        env="SCT_APPEND_SCYLLA_SETUP_ARGS",
    )
    use_preinstalled_scylla: Boolean = sct_field(
        description="Don't install/update ScyllaDB on DB nodes",
        env="SCT_USE_PREINSTALLED_SCYLLA",

    )
    stress_cdclog_reader_cmd: str = sct_field(
        description="""cdc-stressor command to read cdc_log table.
                       You can specify everything but the -node, -keyspace, -table parameter, which is going to
                       be provided by the test suite infrastructure.
                       Multiple commands can be passed as a list.""",
        env="SCT_STRESS_CDCLOG_READER_CMD",
    )
    store_cdclog_reader_stats_in_es: Boolean = sct_field(
        description="Add cdclog reader stats to ES for future performance result calculating",
        env="SCT_STORE_CDCLOG_READER_STATS_IN_ES",

    )
    stop_test_on_stress_failure: Boolean = sct_field(
        description="""If set to True the test will be stopped immediately when stress command failed.
                       When set to False the test will continue to run even when there are errors in the
                       stress process""",
        env="SCT_STOP_TEST_ON_STRESS_FAILURE",

    )
    stress_cdc_log_reader_batching_enable: Boolean = sct_field(
        description="""retrieving data from multiple streams in one poll""",
        env="SCT_STRESS_CDC_LOG_READER_BATCHING_ENABLE",

    )
    use_legacy_cluster_init: Boolean = sct_field(
        description="""Use legacy cluster initialization with autobootsrap disabled and parallel node setup""",
        env="SCT_USE_LEGACY_CLUSTER_INIT",

    )
    availability_zone: str = sct_field(
        description="""Availability zone to use. Specify multiple (comma separated) to deploy resources to multi az (works on AWS).
              "Same for multi-region scenario.""",
        env="SCT_AVAILABILITY_ZONE",
    )
    aws_fallback_to_next_availability_zone: Boolean = sct_field(
        description="Try all availability zones one by one in order to maximize the chances of getting the requested instance capacity.",
        env="SCT_AWS_FALLBACK_TO_NEXT_AVAILABILITY_ZONE",

    )
    num_nodes_to_rollback: int = sct_field(
        description="Number of nodes to upgrade and rollback in test_generic_cluster_upgrade",
        env="SCT_NUM_NODES_TO_ROLLBACK",
    )
    upgrade_sstables: Boolean = sct_field(
        description="Whether to upgrade sstables as part of upgrade_node or not",
        env="SCT_UPGRADE_SSTABLES",

    )
    stress_before_upgrade: str = sct_field(
        description="Stress command to be run before upgrade (prepare stage)",
        env="SCT_STRESS_BEFORE_UPGRADE",
    )
    stress_during_entire_upgrade: str = sct_field(
        description="Stress command to be run during the upgrade - user should take care for suitable duration",
        env="SCT_STRESS_DURING_ENTIRE_UPGRADE",
    )
    stress_after_cluster_upgrade: str = sct_field(
        description="Stress command to be run after full upgrade - usually used to read the dataset for verification",
        env="SCT_STRESS_AFTER_CLUSTER_UPGRADE",
    )

    # Jepsen test.
    jepsen_scylla_repo: str = sct_field(
        description="Link to the git repository with Jepsen Scylla tests",
        env="SCT_JEPSEN_SCYLLA_REPO",
    )
    jepsen_test_cmd: StringOrList = sct_field(
        description="Jepsen test command (e.g., 'test-all')",
        env="SCT_JEPSEN_TEST_CMD",
    )
    jepsen_test_count: int = sct_field(
        description="Possible number of reruns of single Jepsen test command",
        env="SCT_JEPSEN_TEST_COUNT"
    )
    jepsen_test_run_policy: Literal["most", "any", "all"] = sct_field(
        description="""
        Jepsen test run policy (i.e., what we want to consider as passed for a single test)

        'most' - most test runs are passed
        'any'  - one pass is enough
        'all'  - all test runs should pass
        """,
        env="SCT_JEPSEN_TEST_RUN_POLICY",
    )
    max_events_severities: StringOrList = sct_field(
        default=[],
        description="Limit severity level for event types",
        env="SCT_MAX_EVENTS_SEVERITIES",
    )
    scylla_rsyslog_setup: Boolean = sct_field(
        description="Configure rsyslog on Scylla nodes to send logs to monitoring nodes",
        env="SCT_SCYLLA_RSYSLOG_SETUP",

    )
    events_limit_in_email: int = sct_field(
        description="Limit number events in email reports",
        env="SCT_EVENTS_LIMIT_IN_EMAIL",
    )
    data_volume_disk_num: int = sct_field(
        description="""Number of additional data volumes attached to instances
         if data_volume_disk_num > 0, then data volumes (ebs on aws) will be
         used for scylla data directory""",
        env="SCT_DATA_VOLUME_DISK_NUM",
    )
    data_volume_disk_type: Literal["gp2", "gp3", "io2", "io3", ""] = sct_field(
        description="Type of additional volumes: gp2|gp3|io2|io3",
        env="SCT_DATA_VOLUME_DISK_TYPE",
    )
    data_volume_disk_size: int = sct_field(
        description="Size of additional volume in GB",
        env="SCT_DATA_VOLUME_DISK_SIZE",
    )
    data_volume_disk_iops: int = sct_field(
        description="Number of iops for ebs type io2|io3|gp3",
        env="SCT_DATA_VOLUME_DISK_IOPS",
    )
    run_db_node_benchmarks: Boolean = sct_field(
        description="Flag for running db node benchmarks before the tests",
        env="SCT_RUN_DB_NODE_BENCHMARKS",

    )
    nemesis_selector: StringOrList = sct_field(
        description="""nemesis_selector gets a list of "nemesis properties" and filters IN all the nemesis that has
        ALL the properties in that list which are set to true (the intersection of all properties).
        (In other words filters out all nemesis that doesn't ONE of these properties set to true)
        IMPORTANT: If a property doesn't exist, ALL the nemesis will be included.""",
        env="SCT_NEMESIS_SELECTOR",
        k8s_multitenancy_supported=True,
    )
    nemesis_exclude_disabled: Boolean = sct_field(
        description="""nemesis_exclude_disabled determines whether 'disabled' nemeses are filtered out from list
        or are allowed to be used. This allows to easily disable too 'risky' or 'extreme' nemeses by default,
        for all longevities. For example: it is unwanted to run the ToggleGcModeMonkey in standard longevities
        that runs a stress with data validation.""",
        env="SCT_NEMESIS_EXCLUDE_DISABLED",

        k8s_multitenancy_supported=True,
    )
    nemesis_multiply_factor: int = sct_field(
        description="Multiply the list of nemesis to execute by the specified factor",
        env="SCT_NEMESIS_MULTIPLY_FACTOR",
        k8s_multitenancy_supported=True,
    )
    raid_level: int = sct_field(
        description="Number of of raid level: 0 - RAID0, 5 - RAID5",
        env="SCT_RAID_LEVEL",
    )
    bare_loaders: Boolean = sct_field(
        description="Don't install anything but node_exporter to the loaders during cluster setup",
        env="SCT_BARE_LOADERS",

    )
    stress_image: DictOrStr = sct_field(
        description="Dict of the images to use for the stress tools",
        env="SCT_STRESS_IMAGE",
    )
    enable_argus: Boolean = sct_field(
        description="Control reporting to argus",
        env="SCT_ENABLE_ARGUS")
    cs_populating_distribution: str = sct_field(
        description="set c-s parameter '-pop' with gauss/uniform distribution for performance gradual throughput grow tests",
        env="SCT_CS_POPULATING_DISTRIBUTION",
    )
    num_loaders_step: int = sct_field(
        description="Number of loaders which should be added per step",
        env="SCT_NUM_LOADERS_STEP",
    )
    stress_threads_start_num: int = sct_field(
        description="Number of threads for c-s command",
        env="SCT_STRESS_THREADS_START_NUM",
    )
    num_threads_step: int = sct_field(
        description="Number of threads which should be added on per step",
        env="SCT_NUM_THREADS_STEP",
    )
    stress_step_duration: str = sct_field(
        description="Duration of time for stress round",
        env="SCT_STRESS_STEP_DURATION",
    )
    max_deviation: float = sct_field(
        description="Max relative difference between best and current throughput, if current throughput larger then best on max_rel_diff, it become new best one",
        env="SCT_MAX_DEVIATION",
    )
    n_stress_process: int = sct_field(
        description="Number of stress processes per loader",
        env="SCT_N_STRESS_PROCESS",
    )
    stress_process_step: int = sct_field(
        description="add/remove num of process on each round",
        env="SCT_STRESS_PROCESS_STEP",
    )
    use_hdr_cs_histogram: Boolean = sct_field(
        description="Enable hdr histogram logging for cs",
        env="SCT_USE_HDR_CS_HISTOGRAM",

    )
    stop_on_hw_perf_failure: Boolean = sct_field(
        description="""Stop sct performance test if hardware performance test failed

    Hardware performance tests runs on each node with sysbench and cassandra-fio tools.
    Results stored in ES. HW perf tests run during cluster setups and not affect
    SCT Performance tests. Results calculated as average among all results for certain
    instance type or among all nodes during single run.
    if results for a single node is not in margin 0.01 of
    average result for all nodes, hw test considered as Failed.
    If stop_on_hw_perf_failure is True, then sct performance test will be terminated
       after hw perf tests detect node with hw results not in margin with average
    If stop_on_hw_perf_failure is False, then sct performance test will be run
       even after hw perf tests detect node with hw results not in margin with average""",
        env="SCT_STOP_ON_HW_PERF_FAILURE",

    )
    custom_es_index: str = sct_field(
        description="Use custom ES index for storing test results",
        env="SCT_CUSTOM_ES_INDEX",
    )

    simulated_regions: int = sct_field(
        description="Number of simulated regions for the test",
        env="SCT_SIMULATED_REGIONS",
        choices=[0, 2, 3, 4, 5],
    )
    simulated_racks: int = sct_field(
        description="""Forces GossipingPropertyFileSnitch (regardless `endpoint_snitch`) to simulate racks.
         Provide number of racks to simulate.""",
        env="SCT_SIMULATED_RACKS",
    )
    use_dns_names: Boolean = sct_field(
        description="""Use dns names instead of ip addresses for nodes in cluster""",
        env="SCT_USE_DNS_NAMES",

    )

    required_params: Annotated[list, IgnoredType] = ['cluster_backend', 'test_duration', 'n_db_nodes', 'n_loaders', 'use_preinstalled_scylla',
                                                     'user_credentials_path', 'root_disk_size_db', "root_disk_size_monitor", 'root_disk_size_loader']

    # those can be added to a json scheme to validate / or write the validation code for it to be a bit clearer output
    backend_required_params: Annotated[dict, IgnoredType] = {
        'aws': ['user_prefix', "instance_type_loader", "instance_type_monitor", "instance_type_db",
                "region_name", "ami_id_db_scylla", "ami_id_loader",
                "ami_id_monitor", "aws_root_disk_name_monitor", "ami_db_scylla_user",
                "ami_monitor_user", "scylla_network_config"],

        'gce': ['user_prefix', 'gce_network', 'gce_image_db', 'gce_image_username', 'gce_instance_type_db',
                'gce_root_disk_type_db',  'gce_n_local_ssd_disk_db',
                'gce_instance_type_loader', 'gce_root_disk_type_loader',
                'gce_instance_type_monitor', 'gce_root_disk_type_monitor',
                'gce_datacenter'],

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
                      'gce_instance_type_loader', 'gce_root_disk_type_loader',
                      'gce_instance_type_monitor', 'gce_root_disk_type_monitor',
                      'gce_datacenter'],

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

    defaults_config_files: Annotated[dict, IgnoredType] = {
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

    per_provider_multi_region_params: Annotated[dict, IgnoredType] = {
        "aws": ['region_name', 'n_db_nodes', 'ami_id_db_scylla', 'ami_id_loader'],
        "gce": ['gce_datacenter', 'n_db_nodes']
    }

    stress_cmd_params: Annotated[list, IgnoredType] = [
        # this list is used for variouse checks against stress commands, such as:
        # 1. Check if all c-s profile files existing that are referred in the commands
        # 2. Check what stress tools test is needed when loader is prepared
        'gemini_cmd', 'stress_cmd', 'stress_read_cmd', 'stress_cmd_w', 'stress_cmd_r', 'stress_cmd_m',
        'prepare_write_cmd', 'stress_cmd_no_mv', 'stress_cmd_no_mv_profile',
        'prepare_stress_cmd', 'stress_cmd_1', 'stress_cmd_complex_prepare', 'prepare_write_stress',
        'stress_cmd_read_10m', 'stress_cmd_read_cl_one',
        'stress_cmd_complex_verify_read', 'stress_cmd_complex_verify_more',
        'write_stress_during_entire_test', 'verify_data_after_entire_test',
        'stress_cmd_read_cl_quorum', 'verify_stress_after_cluster_upgrade',
        'stress_cmd_complex_verify_delete', 'stress_cmd_lwt_mixed', 'stress_cmd_lwt_de',
        'stress_cmd_lwt_dc', 'stress_cmd_lwt_ue', 'stress_cmd_lwt_uc', 'stress_cmd_lwt_ine',
        'stress_cmd_lwt_d', 'stress_cmd_lwt_u', 'stress_cmd_lwt_i'
    ]
    ami_id_params: Annotated[list, IgnoredType] = ['ami_id_db_scylla',
                                                   'ami_id_loader', 'ami_id_monitor', 'ami_id_db_cassandra', 'ami_id_db_oracle']
    aws_supported_regions: Annotated[list, IgnoredType] = ['eu-west-1',
                                                           'eu-west-2', 'us-west-2', 'us-east-1', 'eu-north-1', 'eu-central-1']

    model_config = ConfigDict(ignored_types=(IgnoredType,),
                              validate_assignment=True,
                              )

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise ValueError(f"Unknown configuration {key=}")
        setattr(self, key, value)

    def __init__(self):  # noqa: PLR0912, PLR0914, PLR0915
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        super().__init__()

        env = self._load_environment_variables()
        config_files = env.get('config_files', [])
        config_files = [sct_abs_path(f) for f in config_files]

        # prepend to the config list the defaults the config files
        backend = env.get('cluster_backend')
        backend_config_files = [sct_abs_path('defaults/test_default.yaml')]
        if backend:
            backend_config_files += self.defaults_config_files[str(backend)]
        self.multi_region_params = self.per_provider_multi_region_params.get(str(backend), [])

        # 1) load the default backend config files
        files = anyconfig.load(list(backend_config_files))
        anyconfig.merge(self, files)

        # 2) load the config files
        files = anyconfig.load(list(config_files))
        anyconfig.merge(self, files)

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
        anyconfig.merge(self, env)

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
                logger.info("Assume that Scylla Docker image has repo file pre-installed.")
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
                    logger.debug("Found AMI %s(%s) for scylla_version='%s' in %s",
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

                logger.debug("Found GCE image %s for scylla_version='%s'", gce_image.name, scylla_version)
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
                    logger.debug("Found Azure Image %s for scylla_version='%s' in %s",
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
                    aws_arch = get_arch_from_instance_type(self.get('instance_type_db'), region_name=region)
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

                    logger.debug("Found AMI %s for oracle_scylla_version='%s' in %s",
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
            if repo_link := self.get(repo_key):
                setattr(self, repo_key, resolve_latest_repo_symlink(repo_link))

        # 9) append username or ami_id_db_scylla_desc to the user_prefix
        version_tag = self.get('ami_id_db_scylla_desc') or getpass.getuser()
        user_prefix = self.get('user_prefix') or getpass.getuser()
        if version_tag != user_prefix:
            self['user_prefix'] = "{}-{}".format(user_prefix, version_tag)[:35]
        else:
            self['user_prefix'] = user_prefix[:35]

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

        # 14 Validate run_fullscan parameters
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

        # 20 Validate Manager restore parameters
        if restore_params := self.get("mgmt_restore_params"):
            self["mgmt_restore_params"] = RestoreParameters(**restore_params)

        # 21 Validate Manager agent backup general parameters
        if backup_params := self.get("mgmt_agent_backup_config"):
            self["mgmt_agent_backup_config"] = AgentBackupParameters(**backup_params)

    def log_config(self):
        logger.info(self.dump_config())

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
        if self.cluster_backend and include_backend:
            default_config_files += self.defaults_config_files[str(self.cluster_backend)]

        return anyconfig.load(list(default_config_files)).get(key, None)

    def _load_environment_variables(self):
        environment_vars = {}
        for field_name, field in self.model_fields.items():
            if field.exclude or not field.json_schema_extra:
                continue

            field_env = field.json_schema_extra.get('env')

            def no_op(x):
                return x

            for annotation in field.metadata:
                if isinstance(annotation, BeforeValidator):
                    from_env_func = annotation.func
                    break
            else:
                from_env_func = no_op

            if field_env and field_env in os.environ:
                try:
                    environment_vars[field_name] = from_env_func(os.environ[field_env])
                except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
                    raise ValueError(
                        "failed to parse {} from environment variable".format(field_env)) from ex
                nested_keys = [key for key in os.environ if key.startswith(field_env + '.')]
                if nested_keys:
                    list_value = []
                    dict_value = {}
                    for key in nested_keys:
                        nest_key, *_ = key.split('.')[1:]
                        if nest_key.isdigit():
                            list_value.insert(int(nest_key), os.environ.get(key))
                        else:
                            dict_value[nest_key] = os.environ.get(key)
                    current_value = environment_vars.get(field.title)
                    if current_value and isinstance(current_value, dict):
                        current_value.update(dict_value)
                    else:
                        environment_vars[field_name] = list_value or dict_value

        return environment_vars

    def get(self, key: str | None):
        """
        get the value of test configuration parameter by the name
        """
        if key is None:
            return None

        if key and '.' in key:
            if ret_val := self._dotted_get(key):
                return ret_val
        ret_val = getattr(self, key)

        if key in self.multi_region_params and isinstance(ret_val, list):
            ret_val = ' '.join(str(v) for v in ret_val)

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

    def _validate_value(self, field: Field):
        field.json_schema_extra['is_k8s_multitenant_value'] = False

        def no_op(x):
            return x

        for annotation in field.metadata:
            if isinstance(annotation, BeforeValidator):
                from_env_func = annotation.func
                break
        else:
            from_env_func = no_op

        try:
            from_env_func(self.get(field.title))
        except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
            if not (self.get("cluster_backend").startswith("k8s")
                    and self.get("k8s_tenants_num") > 1
                    and field.json_schema_extra.get("k8s_multitenancy_supported")
                    and isinstance(self.get(field.title), list)):
                if not (field.title == "nemesis_selector"
                        and isinstance(self.get('nemesis_class_name'), str)
                        and len(self.get("nemesis_class_name").split(" ")) > 1):
                    raise ValueError("failed to validate {}".format(field.title)) from ex
            for list_element in self.get(field.title):
                try:
                    from_env_func(list_element)
                except Exception as ex:  # pylint: disable=broad-except  # noqa: BLE001
                    raise ValueError("failed to validate {}".format(field.title)) from ex
            field.json_schema_extra['is_k8s_multitenant_value'] = True

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

        self._verify_data_volume_configuration(backend)

        if self.get('n_db_nodes'):
            self._validate_seeds_number()
            self._validate_nemesis_can_run_on_non_seed()
            self._validate_number_of_db_nodes_divides_by_az_number()

        if 'extra_network_interface' in self and len(self.region_names) >= 2:
            raise ValueError("extra_network_interface isn't supported for multi region use cases")

        self._check_partition_range_with_data_validation_correctness()
        self._verify_scylla_bench_mode_and_workload_parameters()

        self._validate_placement_group_required_values()

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
        new_scylla_repo = self.get('new_scylla_repo')
        if new_scylla_repo and not self.get('target_upgrade_version'):
            self['target_upgrade_version'] = get_branch_version(new_scylla_repo)
            self.update_argus_with_version(self.get('target_upgrade_version'), "scylla-server-upgrade-target")

    def _check_unexpected_sct_variables(self):
        # check if there are SCT_* environment variable which aren't documented
        config_keys = {field.json_schema_extra.get('env', f'SCT_{field_name}'.upper())
                       for field_name, field in self.model_fields.items() if field.json_schema_extra}
        env_keys = {o.split('.')[0] for o in os.environ if o.startswith('SCT_')}
        unknown_env_keys = env_keys.difference(config_keys)
        if unknown_env_keys:
            output = ["{}={}".format(key, os.environ.get(key)) for key in unknown_env_keys]
            raise ValueError("Unsupported environment variables were used:\n\t - {}".format("\n\t - ".join(output)))

    def _validate_sct_variable_values(self):
        for field_name, field in self.model_fields.items():
            if field_name in self:
                self._validate_value(field)

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
        if backend in available_backends:
            if backend in ('aws', 'gce') and self.get("db_type") == "cloud_scylla":
                backend += "-siren"
            self._check_backend_defaults(backend, self.backend_required_params[backend])
        else:
            raise ValueError("Unsupported backend [{}]".format(backend))

    def _check_backend_defaults(self, backend, required_params):
        fields = [field_name for field_name in self.model_fields if field_name in required_params]
        for field in fields:
            assert self.get(field), "{} missing from config for {}".format(field, backend)

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
        # TODO: check if this validation is still relevant and needed
        data_validation = self.get('data_validation')
        if not data_validation:
            return
        partition_range_with_data_validation = data_validation.get('partition_range_with_data_validation')
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
            scylla_version = get_branch_version(scylla_repo)
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
        self.artifact_scylla_version = scylla_version
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

    # def dict(self):
    #    out = deepcopy(self)
    #
    #    # handle pydantic object, and convert them back to dicts
    #    # TODO: automate the process if we gonna keep using them more, or replace the whole configuration with pydantic/dataclasses
    #    if kafka_connectors := self.get('kafka_connectors'):
    #        out['kafka_connectors'] = [connector.dict(by_alias=True, exclude_none=True)
    #                                   for connector in kafka_connectors]
    #    if mgmt_restore_params := self.get("mgmt_restore_params"):
    #        out["mgmt_restore_params"] = mgmt_restore_params.dict(by_alias=True, exclude_none=True)
    #    if mgmt_agent_backup_config := self.get("mgmt_agent_backup_config"):
    #        out["mgmt_agent_backup_config"] = mgmt_agent_backup_config.dict(by_alias=True, exclude_none=True)
    #    return out

    def dump_config(self):
        """
        Dump current configuration to string

        :return: str
        """
        return anyconfig.dumps(self.dict(exclude_none=True), ac_parser="yaml")

    def dump_help_config_markdown(self):
        """
        Dump all configuration options with their defaults and help to string in markdown format

        :return: str
        """
        header = """
            # scylla-cluster-tests configuration options
            | Parameter | Description  | Default | Override environment<br>variable
            | :-------  | :----------  | :------ | :-------------------------------
        """

        def strip_help_text(text):
            """
            strip all lines, and also remove empty lines from start or end
            """
            output = [l.strip() for l in text.splitlines()]
            return '\n'.join(output[1 if not output[0] else 0:-1 if not output[-1] else None])

        ret = strip_help_text(header) + '\n'

        for field_name, field in self.model_fields.items():
            if field.exclude or not field.json_schema_extra:
                continue

            if description := field.description:
                help_text = '<br>'.join(strip_help_text(description).splitlines())
            else:
                help_text = ''

            default = self.get_default_value(field_name)
            default_text = default if default else 'N/A'
            ret += """| **<a href="#user-content-{name}" name="{name}">{name}</a>**  | {help_text} | {default_text} | {env}\n""".format(
                help_text=help_text, default_text=default_text, name=field_name, env=field.json_schema_extra.get('env', f'SCT_{field_name}'.upper()))

        return ret

    def dump_help_config_yaml(self):
        """
        Dump all configuration options with their defaults and help to string in yaml format

        :return: str
        """
        ret = ""
        for field_name, field in self.model_fields.items():
            if field.exclude or not field.json_schema_extra:
                continue

            if description := field.description:
                help_text = '\n'.join(["# {}".format(l.strip()) for l in description.splitlines() if l.strip()]) + '\n'
            else:
                help_text = ''
            default = self.get_default_value(field_name)
            default = default if default else 'N/A'
            ret += "{help_text}{name}: {default}\n\n".format(help_text=help_text, default=default, name=field_name)

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


def init_and_verify_sct_config() -> SCTConfiguration:
    sct_config = SCTConfiguration()
    sct_config.log_config()
    sct_config.verify_configuration()
    sct_config.verify_configuration_urls_validity()
    sct_config.get_version_based_on_conf()
    sct_config.check_required_files()
    return sct_config
