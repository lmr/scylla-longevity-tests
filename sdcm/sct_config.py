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
import ast
import logging
import getpass
import pathlib
from typing import List, Union

from distutils.util import strtobool  # pylint: disable=import-error,no-name-in-module

import anyconfig

from sdcm import sct_abs_path
from sdcm.utils import alternator
from sdcm.utils.common import find_scylla_repo, get_scylla_ami_versions, get_branched_ami, get_ami_tags, \
    ami_built_by_scylla, MAX_SPOT_DURATION_TIME
from sdcm.utils.version_utils import get_branch_version, get_branch_version_for_multiple_repositories, \
    get_scylla_docker_repo_from_version, resolve_latest_repo_symlink
from sdcm.sct_events.base import add_severity_limit_rules, print_critical_events


def str_or_list(value: Union[str, List[str]]) -> List[str]:
    """Convert an environment variable into a Python's list."""

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
        return [str(value), ]

    if isinstance(value, list):
        return value

    raise ValueError(f"{value} isn't a string or a list")


def int_or_list(value):
    try:
        value = int(value)
        return value
    except Exception:  # pylint: disable=broad-except
        pass

    if isinstance(value, str):
        try:
            values = value.split()
            [int(v) for v in values]  # pylint: disable=expression-not-assigned
            return value
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            return ast.literal_eval(value)
        except Exception:  # pylint: disable=broad-except
            pass

    raise ValueError("{} isn't int or list".format(value))


def boolean(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return bool(strtobool(value))
    else:
        raise ValueError("{} isn't a boolean".format(type(value)))


class SCTConfiguration(dict):
    """
    Class the hold the SCT configuration
    """
    available_backends = ['aws', 'gce', 'docker', 'baremetal', 'aws-siren', 'k8s-gce-minikube', 'k8s-gke', 'gce-siren']

    config_options = [
        dict(name="config_files", env="SCT_CONFIG_FILES", type=str_or_list,
             help="a list of config files that would be used"),

        dict(name="cluster_backend", env="SCT_CLUSTER_BACKEND", type=str,
             help="backend that will be used, aws/gce/docker"),

        dict(name="test_duration", env="SCT_TEST_DURATION", type=int,
             help="""
                  Test duration (min). Parameter used to keep instances produced by tests
                  and for jenkins pipeline timeout and TimoutThread.
             """),

        dict(name="n_db_nodes", env="SCT_N_DB_NODES", type=int_or_list,
             help="Number list of database nodes in multiple data centers."),

        dict(name="n_test_oracle_db_nodes", env="SCT_N_TEST_ORACLE_DB_NODES", type=int_or_list,
             help="Number list of oracle test nodes in multiple data centers."),

        dict(name="n_loaders", env="SCT_N_LOADERS", type=int_or_list,
             help="Number list of loader nodes in multiple data centers"),

        dict(name="n_monitor_nodes", env="SCT_N_MONITORS_NODES", type=int_or_list,
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
             help="Url to the repo of scylla version to install scylla"),

        dict(name="unified_package", env="SCT_UNIFIED_PACKAGE", type=str,
             help="Url to the unified package of scylla version to install scylla"),

        dict(name="nonroot_offline_install", env="SCT_NONROOT_OFFLINE_INSTALL", type=boolean,
             help="Install Scylla without required root priviledge"),

        dict(name="scylla_version", env="SCT_SCYLLA_VERSION",
             type=str,
             help="""Version of scylla to install, ex. '2.3.1'
                     Automatically lookup AMIs and repo links for formal versions.
                     WARNING: can't be used together with 'scylla_repo' or 'ami_id_db_scylla'"""),

        dict(name="oracle_scylla_version", env="SCT_ORACLE_SCYLLA_VERSION",
             type=str,
             help="""Version of scylla to use as oracle cluster with gemini tests, ex. '3.0.11'
                     Automatically lookup AMIs for formal versions.
                     WARNING: can't be used together with 'ami_id_db_oracle'"""),

        dict(name="scylla_linux_distro", env="SCT_SCYLLA_LINUX_DISTRO", type=str,
             help="""The distro name and family name to use [centos/ubuntu-xenial/debian-jessie]"""),

        dict(name="scylla_linux_distro_loader", env="SCT_SCYLLA_LINUX_DISTRO_LOADER", type=str,
             help="""The distro name and family name to use [centos/ubuntu-xenial/debian-jessie]"""),

        dict(name="scylla_repo_m", env="SCT_SCYLLA_REPO_M", type=str,
             help="Url to the repo of scylla version to install scylla from for managment tests"),

        dict(name="scylla_repo_loader", env="SCT_SCYLLA_REPO_LOADER", type=str,
             help="Url to the repo of scylla version to install c-s for loader"),

        dict(name="scylla_mgmt_repo", env="SCT_SCYLLA_MGMT_REPO",
             type=str,
             help="Url to the repo of scylla manager version to install for management tests"),

        dict(name="scylla_mgmt_agent_repo", env="SCT_SCYLLA_MGMT_AGENT_REPO",
             type=str,
             help="Url to the repo of scylla manager agent version to install for management tests"),

        dict(name="scylla_mgmt_agent_version", env="SCT_SCYLLA_MGMT_AGENT_VERSION", type=str,
             help=""),

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

        dict(name="use_ldap_authorization", env="SCT_USE_LDAP_AUTHORIZATION", type=boolean,
             help="When defined true, will create a docker container with LDAP and configure scylla.yaml to use it"),

        dict(name="use_mgmt", env="SCT_USE_MGMT", type=boolean,
             help="When define true, will install scylla management"),

        dict(name="mgmt_port", env="SCT_MGMT_PORT", type=int,
             help="The port of scylla management"),

        dict(name="manager_prometheus_port", env="SCT_MANAGER_PROMETHEUS_PORT", type=int,
             help="Port to be used by the manager to contact Prometheus"),

        dict(name="target_scylla_mgmt_server_repo", env="SCT_TARGET_SCYLLA_MGMT_SERVER_REPO", type=str,
             help="Url to the repo of scylla manager version used to upgrade the manager server"),

        dict(name="target_scylla_mgmt_agent_repo", env="SCT_TARGET_SCYLLA_MGMT_AGENT_REPO", type=str,
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

        dict(name="instance_provision", env="SCT_INSTANCE_PROVISION", type=str,
             help="instance_provision: spot|on_demand|spot_fleet"),

        dict(name="instance_provision_fallback_on_demand", env="SCT_INSTANCE_PROVISION_FALLBACK_ON_DEMAND", type=boolean,
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

        dict(name="seeds_selector", env="SCT_SEEDS_SELECTOR", type=str,
             choices=['reflector', 'random', 'first'],
             help="""How to select the seeds. Expected values: reflector/random/first"""),

        dict(name="seeds_num", env="SCT_SEEDS_NUM", type=int,
             help="""Number of seeds to select"""),

        dict(name="send_email", env="SCT_SEND_EMAIL", type=boolean,
             help="""If true would send email out of the performance regression test"""),

        dict(name="email_recipients", env="SCT_EMAIL_RECIPIENTS", type=str_or_list,
             help="""list of email of send the performance regression test to"""),

        dict(name="email_subject_postfix", env="SCT_EMAIL_SUBJECT_POSTFIX", type=str,
             help="""Email subject postfix"""),

        dict(name="enable_test_profiling", env="SCT_ENABLE_TEST_PROFILING", type=bool,
             help="""Turn on sct profiling"""),
        dict(name="ssh_transport", env="SSH_TRANSPORT", type=str,
             help="""Set type of ssh library to use. Could be 'fabric' (default) or 'libssh2'"""),
        # should be removed once stress commands would be refactored
        dict(name="bench_run", env="SCT_BENCH_RUN", type=boolean,
             help="""If true would kill the scylla-bench thread in the test teardown"""),

        # should be removed once stress commands would be refactored
        dict(name="fullscan", env="SCT_FULLSCAN", type=boolean,
             help="""If true would kill the fullscan thread in the test teardown"""),

        # Scylla command line arguments options
        dict(name="experimental", env="SCT_EXPERIMENTAL", type=boolean,
             help="when enabled scylla will use it's experimental features"),

        dict(name="server_encrypt", env="SCT_SERVER_ENCRYPT", type=boolean,
             help="when enable scylla will use encryption on the server side"),

        dict(name="client_encrypt", env="SCT_CLIENT_ENCRYPT", type=boolean,
             help="when enable scylla will use encryption on the client side"),

        dict(name="hinted_handoff", env="SCT_HINTED_HANDOFF", type=str,
             help="when enable or disable scylla hinted handoff (enabled/disabled)"),

        dict(name="authenticator", env="SCT_AUTHENTICATOR", type=str,
             help="which authenticator scylla will use AllowAllAuthenticator/PasswordAuthenticator",
             choices=("PasswordAuthenticator", "AllowAllAuthenticator"),
             ),

        dict(name="authenticator_user", env="SCT_AUTHENTICATOR_USER", type=str,
             help="the username if PasswordAuthenticator is used"),

        dict(name="authenticator_password", env="SCT_AUTHENTICATOR_PASSWORD", type=str,
             help="the password if PasswordAuthenticator is used"),

        dict(name="authorizer", env="SCT_AUTHORIZER", type=str,
             help="which authorizer scylla will use AllowAllAuthorizer/CassandraAuthorizer"),

        dict(name="system_auth_rf", env="SCT_SYSTEM_AUTH_RF", type=int,
             help="Replication factor will be set to system_auth"),

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

        dict(name="region_aware_loader", env="SCT_REGION_AWARE_LOADER", type=bool,
             help="When in multi region mode, run stress on loader that is located in the same region as db node"),

        dict(name="append_scylla_args", env="SCT_APPEND_SCYLLA_ARGS", type=str,
             help="More arguments to append to scylla command line"),

        dict(name="append_scylla_args_oracle", env="SCT_APPEND_SCYLLA_ARGS_ORACLE", type=str,
             help="More arguments to append to oracle command line"),

        dict(name="append_scylla_yaml", env="SCT_APPEND_SCYLLA_YAML", type=str,
             help="More configuration to append to /etc/scylla/scylla.yaml"),

        # Nemesis config options

        dict(name="nemesis_class_name", env="SCT_NEMESIS_CLASS_NAME", type=str,
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

        dict(name="nemesis_interval", env="SCT_NEMESIS_INTERVAL", type=int,
             help="""Nemesis sleep interval to use if None provided specifically in the test"""),

        dict(name="nemesis_sequence_sleep_between_ops", env="SCT_NEMESIS_SEQUENCE_SLEEP_BETWEEN_OPS", type=int,
             help="""Sleep interval between nemesis operations for use in unique_sequence nemesis kind of tests"""),

        dict(name="nemesis_during_prepare", env="SCT_NEMESIS_DURING_PREPARE", type=boolean,
             help="""Run nemesis during prepare stage of the test"""),

        dict(name="nemesis_seed", env="SCT_NEMESIS_SEED", type=int,
             help="""A seed number in order to repeat nemesis sequence as part of SisyphusMonkey"""),

        dict(name="nemesis_add_node_cnt",
             env="SCT_NEMESIS_ADD_NODE_CNT",
             type=int,
             help="""Add/remove nodes during GrowShrinkCluster nemesis"""),

        dict(name="cluster_target_size", env="SCT_CLUSTER_TARGET_SIZE", type=int,
             help="""Used for scale test: max size of the cluster"""),

        dict(name="space_node_threshold", env="SCT_SPACE_NODE_THRESHOLD", type=int,
             help="""
                 Space node threshold before starting nemesis (bytes)
                 The default value is 6GB (6x1024^3 bytes)
                 This value is supposed to reproduce
                 https://github.com/scylladb/scylla/issues/1140
             """),

        dict(name="nemesis_filter_seeds", env="SCT_NEMESIS_FILTER_SEEDS", type=boolean,
             help="""If true runs the nemesis only on non seed nodes"""),

        # Stress Commands

        dict(name="stress_cmd", env="SCT_STRESS_CMD", type=str_or_list,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="gemini_version", env="SCT_GEMINI_VERSION", type=str,
             help="""Version of download of the binaries of gemini tool"""),

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

        dict(name="region_name", env="SCT_REGION_NAME", type=str_or_list,
             help="AWS regions to use"),

        dict(name="security_group_ids", env="SCT_SECURITY_GROUP_IDS", type=str_or_list,
             help="AWS security groups ids to use"),

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

        dict(name="aws_root_disk_size_db", env="SCT_AWS_ROOT_DISK_SIZE_DB", type=int,
             help=""),

        dict(name="aws_root_disk_size_monitor", env="SCT_AWS_ROOT_DISK_SIZE_MONITOR", type=int,
             help=""),

        dict(name="aws_root_disk_size_loader", env="SCT_AWS_ROOT_DISK_SIZE_LOADER", type=int,
             help=""),

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

        dict(name="aws_instance_profile_name", env="SCT_AWS_INSTANCE_PROFILE_NAME", type=str,
             help="This is the name of the instance profile to set on all instances"),

        dict(name="backup_bucket_location", env="SCT_BACKUP_BUCKET_LOCATION", type=str_or_list,
             help="""This is the bucket name to be used for backup with its region
             (e.g. backup_bucket_location: 'manager-backup-tests')"""),

        dict(name="tag_ami_with_result", env="SCT_TAG_AMI_WITH_RESULT", type=boolean,
             help="If True, would tag the ami with the test final result"),

        # GCE config options

        dict(name="gce_datacenter", env="SCT_GCE_DATACENTER", type=str,
             help="Supported: us-east1 - means that the zone will be selected automatically or "
                  "you can mention the zone explicitly, for example: us-east1-b"),

        dict(name="gce_network", env="SCT_GCE_NETWORK", type=str,
             help=""),

        dict(name="gce_image", env="SCT_GCE_IMAGE", type=str,
             help="GCE image to use for all node types: db, loader and monitor"),

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

        dict(name="gce_root_disk_size_monitor", env="SCT_GCE_ROOT_DISK_SIZE_MONITOR", type=int,
             help=""),

        dict(name="gce_n_local_ssd_disk_monitor", env="SCT_GCE_N_LOCAL_SSD_DISK_MONITOR", type=int,
             help=""),

        dict(name="gce_instance_type_db", env="SCT_GCE_INSTANCE_TYPE_DB", type=str,
             help=""),

        dict(name="gce_root_disk_type_db", env="SCT_GCE_ROOT_DISK_TYPE_DB", type=str,
             help=""),

        dict(name="gce_root_disk_size_db", env="SCT_GCE_ROOT_DISK_SIZE_DB", type=int,
             help=""),

        dict(name="gce_n_local_ssd_disk_db", env="SCT_GCE_N_LOCAL_SSD_DISK_DB", type=int,
             help=""),

        dict(name="gce_pd_standard_disk_size_db", env="SCT_GCE_PD_STANDARD_DISK_SIZE_DB", type=int,
             help=""),

        dict(name="gce_pd_ssd_disk_size_db", env="SCT_GCE_PD_SSD_DISK_SIZE_DB", type=int,
             help=""),

        dict(name="gce_pd_ssd_disk_size_loader", env="SCT_GCE_PD_SSD_DISK_SIZE_LOADER", type=int,
             help=""),

        dict(name="gce_pd_ssd_disk_size_monitor", env="SCT_GCE_SSD_DISK_SIZE_MONITOR", type=int,
             help=""),

        # k8s-gce-minikube options
        dict(name="gce_image_minikube", env="SCT_GCE_IMAGE_MINIKUBE", type=str,
             help=""),

        dict(name="gce_instance_type_minikube", env="SCT_GCE_INSTANCE_TYPE_MINIKUBE", type=str,
             help=""),

        dict(name="gce_root_disk_type_minikube", env="SCT_GCE_ROOT_DISK_TYPE_MINIKUBE", type=str,
             help=""),

        dict(name="gce_root_disk_size_minikube", env="SCT_GCE_ROOT_DISK_SIZE_MINIKUBE", type=int,
             help=""),

        # k8s-gke options
        dict(name="gke_cluster_version", env="SCT_GKE_CLUSTER_VERSION", type=str,
             help=""),

        dict(name="gke_cluster_n_nodes", env="SCT_GKE_CLUSTER_N_NODES", type=int,
             help=""),

        # k8s options
        dict(name="k8s_deploy_monitoring", env="SCT_K8S_DEPLOY_MONITORING", type=str,
             help=""),

        dict(name="k8s_scylla_operator_docker_image", env="SCT_K8S_SCYLLA_OPERATOR_DOCKER_IMAGE", type=str,
             help=""),

        dict(name="k8s_scylla_datacenter", env="SCT_K8S_SCYLLA_DATACENTER", type=str,
             help=""),

        dict(name="k8s_scylla_rack", env="SCT_K8S_SCYLLA_RACK", type=str,
             help=""),

        dict(name="k8s_scylla_cluster_name", env="SCT_K8S_SCYLLA_CLUSTER_NAME", type=str,
             help=""),

        dict(name="k8s_scylla_cpu_n", env="SCT_K8S_SCYLLA_CPU_N", type=int,
             help=""),

        dict(name="k8s_scylla_mem_gi", env="SCT_K8S_SCYLLA_MEM_GI", type=int,
             help=""),

        dict(name="k8s_scylla_disk_gi", env="SCT_K8S_SCYLLA_DISK_GI", type=int,
             help=""),

        dict(name="k8s_loader_cluster_name", env="SCT_K8S_LOADER_CLUSTER_NAME", type=str,
             help=""),

        dict(name="k8s_loader_cpu_n", env="SCT_K8S_LOADER_CPU_N", type=int,
             help=""),

        dict(name="k8s_loader_mem_gi", env="SCT_K8S_LOADER_MEM_GI", type=int,
             help=""),

        dict(name="minikube_version", env="SCT_MINIKUBE_VERSION", type=str,
             help=""),

        dict(name="k8s_cert_manager_version", env="SCT_K8S_CERT_MANAGER_VERSION", type=str,
             help=""),

        # docker config options
        dict(name="mgmt_docker_image", env="SCT_MGMT_DOCKER_IMAGE", type=str,
             help="Scylla manager docker image, i.e. 'scylladb/scylla-manager:2.2.1' "),

        dict(name="docker_image", env="SCT_DOCKER_IMAGE", type=str,
             help="Scylla docker image repo, i.e. 'scylladb/scylla', if omitted is calculated from scylla_version"),

        # baremetal config options

        dict(name="db_nodes_private_ip", env="SCT_DB_NODES_PRIVATE_IP", type=str_or_list,
             help=""),

        dict(name="db_nodes_public_ip", env="SCT_DB_NODES_PUBLIC_IP", type=str_or_list,
             help=""),

        dict(name="loaders_private_ip", env="SCT_LOADERS_PRIVATE_IP", type=str_or_list,
             help=""),

        dict(name="loaders_public_ip", env="SCT_LOADERS_PUBLIC_IP", type=str_or_list,
             help=""),

        dict(name="monitor_nodes_private_ip", env="SCT_MONITOR_NODES_PRIVATE_IP", type=str_or_list,
             help=""),

        dict(name="monitor_nodes_public_ip", env="SCT_MONITOR_NODES_PUBLIC_IP", type=str_or_list,
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
             help=""),
        dict(name="run_fullscan", env="SCT_RUN_FULLSCAN", type=str,
             help=""),
        dict(name="keyspace_num", env="SCT_KEYSPACE_NUM", type=int,
             help=""),
        dict(name="round_robin", env="SCT_ROUND_ROBIN", type=boolean,
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

        dict(name="validate_partitions", env="SCT_VALIDATE_PARTITIONS", type=boolean,
             help="when true, log of the partitions before and after the nemesis run is compacted"),
        dict(name="table_name", env="SCT_TABLE_NAME", type=str,
             help="table name to check for the validate_partitions check"),
        dict(name="primary_key_column", env="SCT_PRIMARY_KEY_COLUMN", type=str,
             help="primary key of the table to check for the validate_partitions check"),

        dict(name="stress_read_cmd", env="SCT_STRESS_READ_CMD", type=str_or_list,
             help="""cassandra-stress commands.
                You can specify everything but the -node parameter, which is going to
                be provided by the test suite infrastructure.
                multiple commands can passed as a list"""),

        dict(name="prepare_verify_cmd", env="SCT_PREPARE_VERIFY_CMD", type=str_or_list,
             help="""cassandra-stress commands.
            You can specify everything but the -node parameter, which is going to
            be provided by the test suite infrastructure.
            multiple commands can passed as a list"""),

        dict(name="user_profile_table_count", env="SCT_USER_PROFILE_TABLE_COUNT", type=int,
             help="""number of tables to create for template user c-s"""),

        # MgmtCliTest
        dict(name="scylla_mgmt_upgrade_to_repo", env="SCT_SCYLLA_MGMT_UPGRADE_TO_REPO", type=str,
             help="Url to the repo of scylla manager version to upgrade to for management tests"),

        # PerformanceRegressionTest
        dict(name="partition_range_with_data_validation", env="SCT_PARTITION_RANGE_WITH_DATA_VALIDATION", type=str,
             help="""Relevant for scylla-bench. Hold range (min - max) of PKs values for partitions that data was
                     written with validate data and will be validate during the read.
                     Example: 0-250.
                     Optional parameter for DeleteByPartitionsMonkey and DeleteByRowsRangeMonkey"""),

        dict(name="max_partitions_in_test_table", env="SCT_MAX_PARTITIONS_IN_TEST_TABLE", type=int,
             help="""Relevant for scylla-bench. MAX partition keys (partition-count) in the scylla_bench.test table.
                    Mandatory parameter for DeleteByPartitionsMonkey and DeleteByRowsRangeMonkey"""),

        dict(name="stress_cmd_w", env="SCT_STRESS_CMD_W", type=str_or_list,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="stress_cmd_r", env="SCT_STRESS_CMD_R", type=str_or_list,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="stress_cmd_m", env="SCT_STRESS_CMD_M", type=str_or_list,
             help="""cassandra-stress commands.
                    You can specify everything but the -node parameter, which is going to
                    be provided by the test suite infrastructure.
                    multiple commands can passed as a list"""),

        dict(name="prepare_write_cmd", env="SCT_PREPARE_WRITE_CMD", type=str_or_list,
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

        # PerformanceRegressionUserProfilesTest
        dict(name="cs_user_profiles", env="SCT_CS_USER_PROFILES", type=str_or_list,
             help=""),
        dict(name="cs_duration", env="SCT_CS_DURATION", type=str,
             help=""),

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
        dict(name="target_upgrade_version", env="SCT_TAGRET_UPGRADE_VERSION", type=str,
             help="Assign target upgrade version, use for decide if the truncate entries test should be run. "
                  "This test should be performed in case the target upgrade version >= 3.1"),
        dict(name="upgrade_node_packages", env="SCT_UPGRADE_NODE_PACKAGES", type=str,
             help=""),

        dict(name="test_sst3", env="SCT_TEST_SST3", type=boolean,
             help=""),

        dict(name="test_upgrade_from_installed_3_1_0", env="SCT_TEST_UPGRADE_FROM_INSTALLED_3_1_0", type=boolean,
             help="Enable an option for installed 3.1.0 for work around a scylla issue if it's true"),

        dict(name="authorization_in_upgrade", env="SCT_AUTHORIZATION_IN_UPGRADE", type=str,
             help="Which Authorization to enable after upgrade"),

        dict(name="remove_authorization_in_rollback", env="SCT_REMOVE_AUTHORIZATION_IN_ROLLBACK", type=boolean,
             help="Disable Authorization after rollback to old Scylla"),

        dict(name="new_introduced_pkgs", env="SCT_NEW_INTRODUCED_PKGS", type=str,
             help=""),

        dict(name="recover_system_tables", env="SCT_RECOVER_SYSTEM_TABLES", type=boolean,
             help=""),

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

        dict(name="logs_transport", env="SCT_LOGS_TRANSPORT", type=str,
             help="How to transport logs: rsyslog, ssh or docker", choices=("rsyslog", "ssh", "docker")),

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
        dict(name="workaround_kernel_bug_for_iotune", env="SCT_WORKAROUND_KERNEL_BUG_FOR_IOTUNE", type=bool,
             help="Workaround a known kernel bug which causes iotune to fail in scylla_io_setup, "
                  "only effect GCE backend"),
        dict(name="internode_compression", env="SCT_INTERNODE_COMPRESSION", type=str,
             help="scylla option: internode_compression"),
        dict(name="internode_encryption", env="SCT_INTERNODE_ENCRYPTION", type=str,
             help="scylla sub option of server_encryption_options: internode_encryption"),

        dict(name="jmx_heap_memory", env="SCT_JMX_HEAP_MEMORY", type=int,
             help="The total size of the memory allocated to JMX. Values in MB, so for 1GB enter 1024(MB)"),

        dict(name="loader_swap_size", env="SCT_LOADER_SWAP_SIZE", type=int,
             help="The size of the swap file for the loaders. Its size in bytes calculated by x * 1MB"),

        dict(name="monitor_swap_size", env="SCT_MONITOR_SWAP_SIZE", type=int,
             help="The size of the swap file for the monitors. Its size in bytes calculated by x * 1MB"),

        dict(name="store_perf_results", env="SCT_STORE_PERF_RESULTS", type=boolean,
             help="""A flag that indicates whether or not to gather the prometheus stats at the end of the run.
                Intended to be used in performance testing"""),

        dict(name="append_scylla_setup_args", env="SCT_APPEND_SCYLLA_SETUP_ARGS", type=str,
             help="More arguments to append to scylla_setup command line"),

        dict(name="use_preinstalled_scylla", env="SCT_USE_PREINSTALLED_SCYLLA", type=bool,
             help="Don't install/update ScyllaDB on DB nodes"),
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

        dict(name="use_legacy_cluster_init", env="SCT_USE_LEGACY_CLUSTER_INIT", type=bool,
             help="""Use legacy cluster initialization with autobootsrap disabled and parallel node setup"""),
        dict(name="availability_zone", env="SCT_AVAILABILITY_ZONE",
             type=str,
             help="Availability zone to use. Same for multi-region scenario."),

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
        dict(name="jepsen_test_cmd", env="SCT_JEPSEN_TEST_CMD", type=str,
             help="Jepsen test command (e.g., 'test-all')"),

        dict(name="max_events_severities", env="SCT_MAX_EVENTS_SEVERITIES", type=str_or_list,
             help="Limit severity level for event types"),

        dict(name="scylla_rsyslog_setup", env="SCT_SCYLLA_RSYSLOG_SETUP", type=boolean,
             help="Configure rsyslog on Scylla nodes to send logs to monitoring nodes"),

        dict(name="data_device", env="SCT_DATA_DEVICE", type=str,
             help="""Define which device to use for scylla data. Default for gce is attached. Default for aws is instance_store.
                     attached - means use additionaly attached devices
                     instance_store - means use emphereal local NVMes devices(AWS)""",
             choices=("attached", "instance_store")),

        dict(name="aws_ebs_volume_num", env="SCT_AWS_EBS_VOLUME_NUM",
             type=int,
             help="Number of additional ebs volumes attached to instances"),

        dict(name="aws_ebs_volume_type", env="SCT_AWS_EBS_VOLUME_IOPS",
             type=str,
             help="Type of addtitional volumes: gp2|gp3|io2|io3"),

        dict(name="aws_ebs_volume_size", env="SCT_EBS_VOLUME_SIZE",
             type=int,
             help="Size of additional volume in GB"),

        dict(name="aws_ebs_volume_iops", env="SCT_EBS_VOLUME_IOPS",
             type=int,
             help="Number of iops for ebs type io2|io3|gp3"),
    ]

    required_params = ['cluster_backend', 'test_duration', 'n_db_nodes', 'n_loaders', 'use_preinstalled_scylla',
                       'user_credentials_path']

    # those can be added to a json scheme to validate / or write the validation code for it to be a bit clearer output
    backend_required_params = {
        'aws': ['user_prefix', "instance_type_loader", "instance_type_monitor", "instance_type_db",
                "region_name", "ami_id_db_scylla", "ami_id_loader",
                "ami_id_monitor", "aws_root_disk_size_monitor", "aws_root_disk_name_monitor", "ami_db_scylla_user",
                "ami_monitor_user"],

        'gce': ['user_prefix', 'gce_network', 'gce_image_db', 'gce_image_username', 'gce_instance_type_db',
                'gce_root_disk_type_db', 'gce_root_disk_size_db', 'gce_n_local_ssd_disk_db',
                'gce_instance_type_loader', 'gce_root_disk_type_loader', 'gce_n_local_ssd_disk_loader',
                'gce_instance_type_monitor', 'gce_root_disk_type_monitor', 'gce_root_disk_size_monitor',
                'gce_n_local_ssd_disk_monitor', 'gce_datacenter'],

        'docker': ['user_credentials_path', 'scylla_version'],

        'baremetal': ['db_nodes_private_ip', 'db_nodes_public_ip', 'user_credentials_path'],

        'aws-siren': ["user_prefix", "instance_type_loader", "region_name", "cloud_credentials_path",
                      "cloud_cluster_id", "nemesis_filter_seeds"],

        'gce-siren': ['user_prefix', 'gce_network', 'gce_image_username', 'gce_instance_type_db',
                      'gce_root_disk_type_db', 'gce_root_disk_size_db', 'gce_n_local_ssd_disk_db',
                      'gce_instance_type_loader', 'gce_root_disk_type_loader', 'gce_n_local_ssd_disk_loader',
                      'gce_instance_type_monitor', 'gce_root_disk_type_monitor', 'gce_root_disk_size_monitor',
                      'gce_n_local_ssd_disk_monitor', 'gce_datacenter'],

        'k8s-gce-minikube': ['gce_image_minikube', 'gce_instance_type_minikube', 'gce_root_disk_type_minikube',
                             'gce_root_disk_size_minikube', 'user_credentials_path', 'scylla_version',
                             'scylla_mgmt_agent_version', 'k8s_scylla_operator_docker_image', 'k8s_scylla_datacenter',
                             'k8s_scylla_rack', 'k8s_scylla_cluster_name', 'k8s_scylla_cpu_n', 'k8s_scylla_mem_gi',
                             'k8s_scylla_disk_gi', 'gce_image', 'gce_instance_type_loader', 'gce_root_disk_type_loader',
                             'gce_n_local_ssd_disk_loader', 'gce_instance_type_monitor', 'gce_root_disk_type_monitor',
                             'gce_root_disk_size_monitor', 'gce_n_local_ssd_disk_monitor', 'minikube_version',
                             'mgmt_docker_image'],

        'k8s-gke': ['gke_cluster_version', 'gke_cluster_n_nodes', 'gce_instance_type_db', 'gce_root_disk_type_db',
                    'gce_root_disk_size_db', 'gce_n_local_ssd_disk_db', 'user_credentials_path', 'scylla_version',
                    'scylla_mgmt_agent_version', 'k8s_scylla_operator_docker_image', 'k8s_scylla_datacenter',
                    'k8s_scylla_rack', 'k8s_scylla_cluster_name', 'k8s_scylla_cpu_n', 'k8s_scylla_mem_gi',
                    'k8s_loader_cluster_name', 'k8s_loader_cpu_n', 'k8s_loader_mem_gi', 'gce_instance_type_loader',
                    'gce_image_monitor', 'gce_instance_type_monitor', 'gce_root_disk_type_monitor',
                    'gce_root_disk_size_monitor', 'gce_n_local_ssd_disk_monitor', 'mgmt_docker_image'],
    }

    defaults_config_files = {
        "aws": [sct_abs_path('defaults/aws_config.yaml')],
        "gce": [sct_abs_path('defaults/gce_config.yaml')],
        "docker": [sct_abs_path('defaults/docker_config.yaml')],
        "baremetal": [sct_abs_path('defaults/baremetal_config.yaml')],
        "aws-siren": [sct_abs_path('defaults/aws_config.yaml')],
        "gce-siren": [sct_abs_path('defaults/gce_config.yaml')],
        "k8s-gce-minikube": [sct_abs_path('defaults/k8s_gce_minikube_config.yaml')],
        "k8s-gke": [sct_abs_path('defaults/k8s_gke_config.yaml')],
    }

    multi_region_params = [
        'region_name', 'n_db_nodes', 'ami_id_db_scylla', 'ami_id_loader'
    ]

    def __init__(self):
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        super().__init__()
        self.log = logging.getLogger(__name__)
        env = self._load_environment_variables()
        config_files = env.get('config_files', [])
        config_files = [sct_abs_path(f) for f in config_files]

        # prepend to the config list the defaults the config files
        backend = env.get('cluster_backend')
        backend_config_files = [sct_abs_path('defaults/test_default.yaml')]
        if backend:
            backend_config_files += self.defaults_config_files[str(backend)]

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
        region_names = (self.get('region_name') or '').split()
        region_names = env.get('region_name', region_names)

        cluster_backend = self.get('cluster_backend')
        cluster_backend = env.get('cluster_backend', cluster_backend)

        if 'aws' in cluster_backend:
            for region in region_names:
                for key, value in regions_data[region].items():
                    if key not in self.keys():
                        self[key] = value
                    elif len(self[key].split()) < len(region_names):
                        self[key] += " {}".format(value)

        # 3) overwrite with environment variables
        anyconfig.merge(self, env)

        # 4) update events max severities
        add_severity_limit_rules(self.get("max_events_severities"))
        print_critical_events()

        # 5) assume multi dc by n_db_nodes set size
        if 'aws' in cluster_backend:
            num_of_regions = len((self.get('region_name')).split())
            num_of_db_nodes_sets = len(str(self.get('n_db_nodes')).split(' '))
            if num_of_db_nodes_sets > num_of_regions:
                for region in list(regions_data.keys())[:num_of_db_nodes_sets]:
                    for key, value in regions_data[region].items():
                        if key not in self.keys():
                            self[key] = value
                        else:
                            self[key] += " {}".format(value)

        # 6) handle scylla_version if exists
        scylla_version = self.get('scylla_version')
        scylla_linux_distro = self.get('scylla_linux_distro')
        dist_type = scylla_linux_distro.split('-')[0]
        dist_version = scylla_linux_distro.split('-')[-1]

        if scylla_version:
            if not self.get('docker_image'):
                self['docker_image'] = get_scylla_docker_repo_from_version(scylla_version)
            if self.get("cluster_backend") in ["docker", "k8s-gce-minikube", "k8s-gke"]:
                self.log.info("Assume that Scylla Docker image has repo file pre-installed.")
            elif not self.get('ami_id_db_scylla') and self.get('cluster_backend') == 'aws':
                ami_list = []
                for region in self.get('region_name').split():
                    if ':' in scylla_version:
                        amis = get_branched_ami(scylla_version, region_name=region)
                        ami_list.append(amis[0].id)
                        continue

                    amis = get_scylla_ami_versions(region)
                    for ami in amis:
                        if scylla_version in ami['Name']:
                            ami_list.append(ami['ImageId'])
                            break
                    else:
                        raise ValueError("AMI for scylla version {} wasn't found".format(scylla_version))
                self['ami_id_db_scylla'] = " ".join(ami_list)
            elif not self.get('scylla_repo'):
                self['scylla_repo'] = find_scylla_repo(scylla_version, dist_type, dist_version)
            else:
                raise ValueError("'scylla_version' can't used together with  'ami_id_db_scylla' or with 'scylla_repo'")

            if self.get("n_loaders") and not self.get("scylla_repo_loader") and self.get("cluster_backend") != "aws":
                scylla_linux_distro_loader = self.get('scylla_linux_distro_loader')
                dist_type_loader = scylla_linux_distro_loader.split('-')[0]
                dist_version_loader = scylla_linux_distro_loader.split('-')[-1]

                scylla_version_for_loader = "nightly" if scylla_version == "latest" else scylla_version

                self['scylla_repo_loader'] = find_scylla_repo(scylla_version_for_loader,
                                                              dist_type_loader, dist_version_loader)

        # 6.1) handle oracle scylla_version if exists
        oracle_scylla_version = self.get('oracle_scylla_version')
        if oracle_scylla_version:
            if not self.get('ami_id_db_oracle') and self.get('cluster_backend') == 'aws':
                ami_list = []
                for region in self.get('region_name').split():
                    if ':' in oracle_scylla_version:
                        amis = get_branched_ami(oracle_scylla_version, region_name=region)
                        ami_list.append(amis[0].id)
                        continue

                    amis = get_scylla_ami_versions(region)
                    for ami in amis:
                        if oracle_scylla_version in ami['Name']:
                            ami_list.append(ami['ImageId'])
                            break
                    else:
                        raise ValueError("AMI for scylla version {} wasn't found".format(scylla_version))
                self['ami_id_db_oracle'] = " ".join(ami_list)
            else:
                raise ValueError("oracle_scylla_version and ami_id_db_oracle can't used together")

        # 7) support lookup of repos for upgrade test
        new_scylla_version = self.get('new_version')
        if new_scylla_version:
            if not self.get('ami_id_db_scylla') and cluster_backend == 'aws':  # pylint: disable=no-else-raise
                raise ValueError("'new_version' isn't supported for AWS AMIs")

            elif not self.get('new_scylla_repo'):
                self['new_scylla_repo'] = find_scylla_repo(new_scylla_version, dist_type, dist_version)

        # 8) resolve repo symlinks
        for repo_key in ("scylla_repo", "scylla_repo_loader", "new_scylla_repo", ):
            if self.get(repo_key):
                self[repo_key] = resolve_latest_repo_symlink(self[repo_key])

        # 9) append username or ami_id_db_scylla_desc to the user_prefix
        version_tag = self.get('ami_id_db_scylla_desc')
        user_prefix = self.get('user_prefix')
        if user_prefix:
            if not version_tag:
                version_tag = getpass.getuser()

            self['user_prefix'] = "{}-{}".format(user_prefix, version_tag)[:35]

        # 10) update target_upgrade_version automatically
        new_scylla_repo = self.get('new_scylla_repo')
        if new_scylla_repo and not self.get('target_upgrade_version'):
            self['target_upgrade_version'] = get_branch_version(new_scylla_repo)

        # 11) validate that supported instance_provision selected
        if self.get('instance_provision') not in ['spot', 'on_demand', 'spot_fleet', 'spot_low_price', 'spot_duration']:
            raise ValueError(f"Selected instance_provision type '{self.get('instance_provision')}' is not supported!")

        # 12) spot_duration instance can be created for test duration
        if self.get('instance_provision').lower() == "spot_duration":
            test_duration = self.get('test_duration')
            if test_duration:
                assert test_duration <= MAX_SPOT_DURATION_TIME, \
                    f'Test duration too long for spot_duration instance type. ' \
                    f'Max possible test duration time for this instance type is {MAX_SPOT_DURATION_TIME} minutes'

        # 13) validate authenticator parameters
        if self.get('authenticator') and self.get('authenticator') == "PasswordAuthenticator":
            authenticator_user = self.get("authenticator_user")
            authenticator_password = self.get("authenticator_password")
            if not (authenticator_password and authenticator_user):
                raise ValueError("For PasswordAuthenticator authenticator authenticator_user and authenticator_password"
                                 " have to be provided")

        self._update_environment_variables()

    def log_config(self):
        self.log.info(self.dump_config())

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
                except Exception as ex:
                    raise ValueError("failed to parse {} from environment variable:\n\t\t{}".format(opt['env'], ex))
        return environment_vars

    def _update_environment_variables(self, replace=False):
        for opt in self.config_options:
            if opt["name"] in self and (opt["env"] not in os.environ or replace):
                os.environ[opt["env"]] = str(self[opt["name"]])

    def get(self, key: str):
        """
        get the value of test configuration parameter by the name
        """

        ret_val = super(SCTConfiguration, self).get(key)

        if key in self.multi_region_params and isinstance(ret_val, list):
            ret_val = ' '.join(ret_val)

        return ret_val

    def _validate_value(self, opt):
        try:
            opt['type'](self.get(opt['name']))
        except Exception as ex:
            raise ValueError("failed to validate {}:\n\t\t{}".format(opt['name'], ex))
        choices = opt.get('choices')
        if choices:
            cur_val = self.get(opt['name'])
            assert cur_val in choices, "failed to validate '{}': {} not in {}".format(opt['name'], cur_val, choices)

    def check_required_files(self):
        # pylint: disable=too-many-nested-blocks
        for param_name in ['stress_cmd', 'stress_read_cmd', 'stress_cmd_w', 'stress_cmd_r', 'stress_cmd_m',
                           'prepare_write_cmd', 'stress_cmd_no_mv', 'stress_cmd_no_mv_profile',
                           'prepare_stress_cmd', 'stress_cmd_1', 'stress_cmd_complex_prepare', 'prepare_write_stress',
                           'stress_cmd_read_10m', 'stress_cmd_read_cl_one', 'stress_cmd_read_80m',
                           'stress_cmd_complex_verify_read', 'stress_cmd_complex_verify_more',
                           'write_stress_during_entire_test', 'verify_data_after_entire_test',
                           'stress_cmd_read_cl_quorum', 'verify_stress_after_cluster_upgrade',
                           'stress_cmd_complex_verify_delete', 'stress_cmd_lwt_mixed', 'stress_cmd_lwt_de',
                           'stress_cmd_lwt_dc', 'stress_cmd_lwt_ue', 'stress_cmd_lwt_uc', 'stress_cmd_lwt_ine',
                           'stress_cmd_lwt_d', 'stress_cmd_lwt_u', 'stress_cmd_lwt_i']:
            stress_cmds = self.get(param_name)
            if stress_cmds is None:
                continue
            if isinstance(stress_cmds, str):
                stress_cmds = [stress_cmds]
            for stress_cmd in stress_cmds:
                if not stress_cmd:
                    continue
                stress_cmd = stress_cmd.strip(' ')
                if stress_cmd.startswith('cassandra-stress'):
                    for option in stress_cmd.split():
                        if option.startswith('profile='):
                            option = option.split('=', 1)
                            if len(option) < 2:
                                continue
                            profile_path = option[1]
                            if not profile_path.startswith('/tmp'):
                                raise ValueError(f"Stress command parameter '{param_name}' contains wrong path "
                                                 f"'{profile_path}' to profile, it should be formed in following "
                                                 "manner '/tmp/{file_name_from_data_dir}'")
                            profile_name = profile_path[5:]
                            if pathlib.Path(sct_abs_path(os.path.join('data_dir', profile_name))).exists():
                                break  # We are ok here and skipping whole command if file is there
                            raise ValueError(f"Stress command parameter '{param_name}' contains profile "
                                             f"'{profile_path}' that does not exists under data_dir/")

    def verify_configuration(self):  # pylint: disable=too-many-statements
        """
        Check that all required values are set, and validated each value to be of correct type or value
        also check required options per backend

        :return: None
        :raises ValueError: on failures in validations
        :raise Exception: on unsupported backends
        """

        # pylint: disable=too-many-locals,too-many-branches

        # check if there are SCT_* environment variable which aren't documented
        config_keys = {opt['env'] for opt in self.config_options}
        env_keys = {o for o in os.environ.keys() if o.startswith('SCT_')}
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

        # validate passed configuration
        for opt in self.config_options:
            if opt['name'] in self:
                self._validate_value(opt)

        # validated per backend
        def _check_backend_defaults(backend, required_params):
            opts = [o for o in self.config_options if o['name'] in required_params]
            for _opt in opts:
                assert _opt['name'] in self, "{} missing from config for {}".format(_opt['name'], backend)

        backend = self.get('cluster_backend')
        if backend in self.available_backends:
            _check_backend_defaults(backend, self.backend_required_params[backend])
        else:
            raise ValueError("Unsupported backend [{}]".format(backend))

        # verify multi-region aws params
        if backend in ['aws']:
            region_count = {}
            for opt in self.multi_region_params:
                val = self.get(opt)
                if isinstance(val, str):
                    region_count[opt] = len(self.get(opt).split())
                elif isinstance(val, list):
                    region_count[opt] = len(val)
                else:
                    region_count[opt] = 1
            if not all(region_count['region_name'] == x for x in region_count.values()):
                raise ValueError("not all multi region values are equal: \n\t{}".format(region_count))

        if 'extra_network_interface' in self and len(self.get('region_name').split()) >= 2:
            raise ValueError("extra_network_interface isn't supported for multi region use cases")

        # validate seeds number
        seeds_num = self.get('seeds_num')
        assert seeds_num > 0, "Seed number should be at least one"

        num_of_db_nodes = sum([int(i) for i in str(self.get('n_db_nodes')).split(' ')])
        assert not num_of_db_nodes or seeds_num <= num_of_db_nodes, \
            f"Seeds number ({seeds_num}) should be not more then nodes number ({num_of_db_nodes})"

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

        # verify that the AMIs used all have 'user_data_format_version' tag
        if 'aws' in backend:
            ami_id_db_scylla = self.get('ami_id_db_scylla').split()
            region_names = self.get('region_name').split()
            ami_id_db_oracle = self.get('ami_id_db_oracle').split()

            for ami_list in [ami_id_db_scylla, ami_id_db_oracle]:
                if ami_list:
                    for ami_id, region_name in zip(ami_list, region_names):
                        if not ami_built_by_scylla(ami_id, region_name):
                            continue
                        tags = get_ami_tags(ami_id, region_name)
                        assert 'user_data_format_version' in tags.keys(), \
                            f"\n\t'user_data_format_version' tag missing from [{ami_id}] on {region_name}\n\texisting " \
                            f"tags: {tags}"
        # verify that correct configuration set for using ebs-storage
        self._verfiy_ebs_volumes_configuration(backend)
        # For each Scylla repo file we will check that there is at least one valid URL through which to download a
        # version of SCYLLA, otherwise we will get an error.
        get_branch_version_for_multiple_repositories(urls=[self.get(url) for url in [
            'new_scylla_repo', 'scylla_repo_m', 'scylla_repo_loader', 'scylla_mgmt_repo', 'scylla_mgmt_agent_repo',
            'scylla_mgmt_agent_repo'] if self.get(url)])

    def dump_config(self):
        """
        Dump current configuration to string

        :return: str
        """
        return anyconfig.dumps(self, ac_parser="yaml")

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

        for opt in self.config_options:
            if opt['help']:
                help_text = '<br>'.join(strip_help_text(opt['help']).splitlines())
            else:
                help_text = ''

            default = self.get_default_value(opt['name'])
            default_text = default if default else 'N/A'
            ret += """| **<a href="#user-content-{name}" name="{name}">{name}</a>**  | {help_text} | {default_text} | {env}\n""".format(
                help_text=help_text, default_text=default_text, **opt)

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

    def _verfiy_ebs_volumes_configuration(self, backend):
        if backend in ['aws']:
            device_type = self.get("data_device")
            dev_num = self.get("aws_ebs_volume_num")
            if dev_num < 1 and device_type == "attached":
                raise ValueError(f"Wrong configuration for ebs volume usage")
