#!/usr/bin/env python3

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
# Copyright (c) 2021 ScyllaDB

# pylint: disable=too-many-lines
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import os
import re
import sys
import unittest
import logging
import time
import subprocess
import traceback
import uuid
import pprint
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from functools import partial, reduce
from typing import List
from uuid import UUID

import pytest
import click
import yaml
from prettytable import PrettyTable
from argus.client.sct.types import LogLink
from argus.client.base import ArgusClientError
from argus.common.enums import TestStatus

import sct_ssh
import sct_scan_issues
from sdcm.keystore import KeyStore
from sdcm.localhost import LocalHost
from sdcm.provision import AzureProvisioner
from sdcm.provision.provisioner import VmInstance
from sdcm.remote import LOCALRUNNER
from sdcm.nemesis import SisyphusMonkey
from sdcm.results_analyze import PerformanceResultsAnalyzer, BaseResultsAnalyzer
from sdcm.sct_config import SCTConfiguration
from sdcm.sct_provision.common.layout import SCTProvisionLayout, create_sct_configuration
from sdcm.sct_provision.instances_provider import provision_sct_resources
from sdcm.sct_runner import AwsSctRunner, GceSctRunner, AzureSctRunner, get_sct_runner, clean_sct_runners, \
    update_sct_runner_tags, list_sct_runners
from sdcm.utils.ci_tools import get_job_name, get_job_url
from sdcm.utils.git import get_git_commit_id, get_git_status_info
from sdcm.utils.argus import argus_offline_collect_events, get_argus_client
from sdcm.utils.aws_kms import AwsKms
from sdcm.utils.azure_region import AzureRegion
from sdcm.utils.cloud_monitor import cloud_report, cloud_qa_report
from sdcm.utils.cloud_monitor.cloud_monitor import cloud_non_qa_report
from sdcm.utils.common import (
    S3Storage,
    aws_tags_to_dict,
    create_pretty_table,
    format_timestamp,
    get_ami_images,
    get_ami_images_versioned,
    get_gce_images,
    get_gce_images_versioned,
    gce_meta_to_dict,
    get_builder_by_test_id,
    get_testrun_dir,
    list_clusters_eks,
    list_clusters_gke,
    list_elastic_ips_aws,
    list_test_security_groups,
    list_load_balancers_aws,
    list_cloudformation_stacks_aws,
    list_instances_aws,
    list_placement_groups_aws,
    list_instances_gce,
    list_logs_by_test_id,
    list_resources_docker,
    list_parallel_timelines_report_urls,
    search_test_id_in_latest,
    get_latest_scylla_release,
)
from sdcm.utils.resources_cleanup import (
    clean_cloud_resources,
    clean_resources_according_post_behavior,
)
from sdcm.utils.nemesis import NemesisJobGenerator
from sdcm.utils.net import get_sct_runner_ip
from sdcm.utils.jepsen import JepsenResults
from sdcm.utils.docker_utils import docker_hub_login
from sdcm.monitorstack import (restore_monitoring_stack, get_monitoring_stack_services,
                               kill_running_monitoring_stack_services)
from sdcm.utils.log import setup_stdout_logger, disable_loggers_during_startup
from sdcm.utils.aws_region import AwsRegion
from sdcm.utils.aws_builder import AwsCiBuilder, AwsBuilder
from sdcm.utils.gce_region import GceRegion
from sdcm.utils.gce_builder import GceBuilder
from sdcm.utils.aws_peering import AwsVpcPeering
from sdcm.utils.get_username import get_username
from sdcm.utils.sct_cmd_helpers import add_file_logger, CloudRegion, get_test_config, get_all_regions
from sdcm.send_email import get_running_instances_for_email_report, read_email_data_from_file, build_reporter, \
    send_perf_email
from sdcm.parallel_timeline_report.generate_pt_report import ParallelTimelinesReportGenerator
from sdcm.utils.aws_utils import AwsArchType, is_using_aws_mock
from sdcm.utils.aws_okta import try_auth_with_okta
from sdcm.utils.gce_utils import SUPPORTED_PROJECTS, gce_public_addresses
from sdcm.utils.context_managers import environment
from sdcm.cluster_k8s import mini_k8s
from sdcm.utils.es_index import create_index, get_mapping
from sdcm.utils.version_utils import get_s3_scylla_repos_mapping
import sdcm.provision.azure.utils as azure_utils
from utils.build_system.create_test_release_jobs import JenkinsPipelines  # pylint: disable=no-name-in-module,import-error
from utils.get_supported_scylla_base_versions import UpgradeBaseVersion  # pylint: disable=no-name-in-module,import-error
from utils.mocks.aws_mock import AwsMock  # pylint: disable=no-name-in-module,import-error


SUPPORTED_CLOUDS = ("aws", "gce", "azure",)
DEFAULT_CLOUD = SUPPORTED_CLOUDS[0]

SCT_RUNNER_HOST = get_sct_runner_ip()

LOGGER = setup_stdout_logger()


def sct_option(name, sct_name, **kwargs):
    sct_opt = SCTConfiguration.get_config_option(sct_name)
    multimple_use = kwargs.pop('multiple', False)
    sct_opt.update(kwargs)
    return click.option(name, type=sct_opt['type'], help=sct_opt['help'], multiple=multimple_use)


def install_callback(ctx, _, value):
    if not value or ctx.resilient_parsing:
        return value
    LOGGER.info("install-bash-completion current path: %s", os.getcwd())
    shell, path = "bash", Path.home() / '.bash_completion'
    path.write_text((Path(__file__).parent / 'utils' / '.bash_completion').read_text())
    click.echo('%s completion installed in %s' % (shell, path))
    return sys.exit(0)


def install_package_from_dir(ctx, _, directories):
    if directories or not ctx.resilient_parsing:
        for directory in directories:
            subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", directory])
    return directories


def cloud_provider_option(function=None, default: str | None = DEFAULT_CLOUD,
                          required: bool = True, help: str = "Cloud provider"):  # pylint:disable=redefined-builtin
    def actual_decorator(func):
        return click.option(
            "-c", "--cloud-provider",
            required=required,
            type=click.Choice(choices=SUPPORTED_CLOUDS, case_sensitive=False),
            default=default,
            is_eager=True,
            help=help
        )(func)
    if function:
        return actual_decorator(function)
    return actual_decorator


class SctLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_cases = super().getTestCaseNames(testCaseClass)
        num_of_cases = len(test_cases)
        assert num_of_cases < 2, f"SCT expect only one test case to be selected, found {num_of_cases}:" \
                                 f"\n{pprint.pformat(test_cases)}"
        return test_cases


@click.group()
@click.option("--install-bash-completion",
              is_flag=True,
              callback=install_callback,
              expose_value=False,
              help="Install completion for the current shell. Make sure to have psutil installed.")
@click.option("--install-package-from-directory",
              callback=install_package_from_dir,
              multiple=True,
              envvar="PACKAGES_PATHS",
              type=click.Path(),
              expose_value=False,
              help="Install paths for extra python packages to install, scylla-cluster-plugins for example")
def cli():
    disable_loggers_during_startup()
    try_auth_with_okta()

    if not is_using_aws_mock():
        key_store = KeyStore()
        # TODO: still leaving old keys, until we'll rebuild runner images - and reconfigure jenkins
        key_store.sync(keys=['scylla-qa-ec2', 'scylla-test', 'scylla_test_id_ed25519'],
                       local_path=Path('~/.ssh/').expanduser(), permissions=0o0600)

    docker_hub_login(remoter=LOCALRUNNER)


@cli.command('provision-resources', help="Provision resources for the test")
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends), help="Backend to use")
@click.option('-t', '--test-name', type=str, help="Test name")
@click.option('-c', '--config', multiple=True, type=click.Path(exists=True), help="Test config .yaml to use, can have multiple of those")
def provision_resources(backend, test_name: str, config: str):
    if config:
        os.environ['SCT_CONFIG_FILES'] = str(list(config))
    if backend:
        os.environ['SCT_CLUSTER_BACKEND'] = backend

    add_file_logger()

    params = create_sct_configuration(test_name=test_name)
    test_config = get_test_config()
    test_id = test_config.test_id()
    if not test_id or test_id == "None":
        raise ValueError("No test_id was provided. Aborting provisioning.")
    localhost = LocalHost(user_prefix=params.get("user_prefix"), test_id=test_config.test_id())

    if params.get("logs_transport") == 'syslog-ng':
        click.echo("Provision syslog-ng logging service")
        test_config.configure_syslogng(localhost)
    else:
        click.echo("No need provision logging service")

    click.echo(f"Provision {backend} cloud resources")
    if backend == "aws":
        layout = SCTProvisionLayout(params=params)
        layout.provision()
    elif backend == "azure":
        provision_sct_resources(params=params, test_config=test_config)
    else:
        raise ValueError(f"backend {backend} is not supported")


@cli.command("clean-aws-kms-aliases", help="clean AWS KMS old aliases")
@click.option("-r", "--regions", type=CloudRegion(cloud_provider="aws"), multiple=True,
              help="List of regions to cover")
@click.option("--time-delta-h", type=int, required=False,
              help="Time delta in hours. Used to detect 'old' aliases.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Only show result of search not deleting aliases")
@click.pass_context
def clean_aws_kms_aliases(ctx, regions, time_delta_h, dry_run):
    """Clean AWS KMS old aliases."""
    add_file_logger()
    regions = regions or SCTConfiguration.aws_supported_regions
    aws_kms, kwargs = AwsKms(region_names=regions), {"dry_run": dry_run}
    if time_delta_h:
        kwargs["time_delta_h"] = time_delta_h
    aws_kms.cleanup_old_aliases(**kwargs)


@cli.command('clean-resources', help='clean tagged instances in both clouds (AWS/GCE)')
@click.option('--post-behavior', is_flag=True, default=False, help="clean all resources according to post behavior")
@click.option('--user', type=str, help='user name to filter instances by')
@sct_option('--test-id', 'test_id', help='test id to filter by. Could be used multiple times', multiple=True)
@click.option('--logdir', type=str, help='directory with test run')
@click.option('--dry-run', is_flag=True, default=False, help='dry run')
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends), help="Backend to use")
@click.pass_context
def clean_resources(ctx, post_behavior, user, test_id, logdir, dry_run, backend):  # pylint: disable=too-many-arguments,too-many-branches
    """Clean cloud resources.

    There are different options how to run clean up:
      - To clean resources for the latest run according to post behavior
        $ hydra clean-resources --post-behavior
      - The same as above but with altered logdir
        $ hydra clean-resources --post-behavior --logdir /path/to/logdir
      - To clean resources for some Test ID according to post behavior (test run status extracted from logdir)
        $ hydra clean-resources --post-behavior --test-id TESTID
      - The same as above but with altered logdir
        $ hydra clean-resources --post-behavior --test-id TESTID --logdir /path/to/logdir
      - To clean resources for the latest run ignoring post behavior
        $ hydra clean-resources
      - The same as above but with altered logdir
        $ hydra clean-resources --logdir /path/to/logdir
      - To clean all resources belong to some Test ID:
        $ hydra clean-resources --test-id TESTID
      - To clean all resources belong to some user:
        $ hydra clean-resources --user vasya.pupkin

    Also you can add --dry-run option to see what should be cleaned.
    """
    add_file_logger()

    user_param = {"RunByUser": user} if user else {}

    if not post_behavior and user and not test_id and not logdir:
        click.echo(f"Clean all resources belong to user `{user}'")
        user_param["CreatedBy"] = "SCT"
        params = (user_param, )
    else:
        if not logdir and (post_behavior or not test_id):
            logdir = get_test_config().base_logdir()

        if not test_id and (latest_test_id := search_test_id_in_latest(logdir)):
            click.echo(f"Latest TestId in {logdir} is {latest_test_id}")
            test_id = (latest_test_id, )

        if not test_id:
            click.echo(clean_resources.get_help(ctx))
            return

        if post_behavior:
            click.echo(f"Clean resources according to post behavior for following Test IDs: {test_id}")
        else:
            click.echo(f"Clean all resources for following Test IDs: {test_id}")

        params = ({"TestId": tid, **user_param} for tid in test_id)

    if backend is None:
        if os.environ.get('SCT_CLUSTER_BACKEND', None) is None:
            os.environ['SCT_CLUSTER_BACKEND'] = 'aws'
    else:
        os.environ['SCT_CLUSTER_BACKEND'] = backend

    if test_id:
        os.environ['SCT_TEST_ID'] = test_id[0]
    config = SCTConfiguration()
    if post_behavior:
        click.echo(f"Use {logdir} as a logdir")
        clean_func = partial(clean_resources_according_post_behavior, config=config, logdir=logdir)
    else:
        clean_func = partial(clean_cloud_resources, config=config)

    if dry_run:
        click.echo("Make a dry-run")

    for param in params:
        clean_func(param, dry_run=dry_run)
        click.echo(f"Cleanup for the {param} resources has been finished")


@cli.command('list-resources', help='list tagged instances in cloud (AWS/GCE/Azure)')
@click.option('--user', type=str, help='user name to filter instances by')
@click.option('--get-all', is_flag=True, default=False, help='All resources')
@click.option('--get-all-running', is_flag=True, default=False, help='All running resources')
@sct_option('--test-id', 'test_id', help='test id to filter by')
@click.option('--verbose', is_flag=True, default=False, help='if enable, will log progress')
@click.option('-b', '--backend', 'backend_type', type=click.Choice(SCTConfiguration.available_backends + ['all']), default='all', help="use specific backend")
@click.pass_context
def list_resources(ctx, user, test_id, get_all, get_all_running, verbose, backend_type):  # noqa: PLR0912, PLR0914, PLR0915
    # pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements

    add_file_logger()

    params = {}

    if user:
        params['RunByUser'] = user
    if test_id:
        params['TestId'] = test_id
    if all([not get_all, not get_all_running, not user, not test_id]):
        click.echo(list_resources.get_help(ctx))
        sys.exit(1)

    if get_all_running:
        table_header = ["Name", "Region-AZ", "PublicIP", "TestId", "RunByUser", "LaunchTime"]
    else:
        table_header = ["Name", "Region-AZ", "State", "TestId", "RunByUser", "LaunchTime"]

    def list_resources_on_aws():
        click.secho("Checking AWS EC2...", fg='green')
        aws_instances = list_instances_aws(tags_dict=params, running=get_all_running, verbose=verbose)

        if aws_instances:
            aws_table = PrettyTable(table_header)
            aws_table.align = "l"
            aws_table.sortby = 'LaunchTime'
            for instance in aws_instances:
                tags = aws_tags_to_dict(instance.get('Tags'))
                name = tags.get("Name", "N/A")
                test_id = tags.get("TestId", "N/A")
                run_by_user = tags.get("RunByUser", "N/A")
                aws_table.add_row([
                    name,
                    instance['Placement']['AvailabilityZone'],
                    instance.get('PublicIpAddress', 'N/A') if get_all_running else instance['State']['Name'],
                    test_id,
                    run_by_user,
                    instance['LaunchTime'].ctime()])
            click.echo(aws_table.get_string(title="Instances used on AWS"))
        else:
            click.secho("Nothing found for selected filters in AWS!", fg="yellow")

        click.secho("Checking AWS Elastic IPs...", fg='green')
        elastic_ips_aws = list_elastic_ips_aws(tags_dict=params, verbose=verbose)
        if elastic_ips_aws:
            aws_table = PrettyTable(["AllocationId", "PublicIP", "TestId", "RunByUser", "InstanceId (attached to)"])
            aws_table.align = "l"
            aws_table.sortby = 'AllocationId'
            for eip in elastic_ips_aws:
                tags = aws_tags_to_dict(eip.get('Tags'))
                test_id = tags.get("TestId", "N/A")
                run_by_user = tags.get("RunByUser", "N/A")
                aws_table.add_row([
                    eip['AllocationId'],
                    eip['PublicIp'],
                    test_id,
                    run_by_user,
                    eip.get('InstanceId', 'N/A')])
            click.echo(aws_table.get_string(title="EIPs used on AWS"))
        else:
            click.secho("No elastic ips found for selected filters in AWS!", fg="yellow")

        click.secho("Checking AWS Security Groups...", fg='green')
        security_groups = list_test_security_groups(tags_dict=params, verbose=verbose)
        if security_groups:
            aws_table = PrettyTable(["Name", "Id", "TestId", "RunByUser"])
            aws_table.align = "l"
            aws_table.sortby = 'Id'
            for group in security_groups:
                tags = aws_tags_to_dict(group.get('Tags'))
                test_id = tags.get("TestId", "N/A")
                run_by_user = tags.get("RunByUser", "N/A")
                name = tags.get("Name", "N/A")
                aws_table.add_row([
                    name,
                    group['GroupId'],
                    test_id,
                    run_by_user])
            click.echo(aws_table.get_string(title="SGs used on AWS"))
        else:
            click.secho("No security groups found for selected filters in AWS!", fg="yellow")

        click.secho("Checking AWS Placement Groups...", fg='green')
        placement_groups = list_placement_groups_aws(tags_dict=params, available=get_all_running, verbose=verbose)
        if placement_groups:
            aws_table = PrettyTable(["Name", "Id", "TestId", "RunByUser"])
            aws_table.align = "l"
            aws_table.sortby = 'Id'
            for group in placement_groups:
                tags = aws_tags_to_dict(group.get('Tags'))
                test_id = tags.get("TestId", "N/A")
                run_by_user = tags.get("RunByUser", "N/A")
                name = tags.get("Name", "N/A")
                aws_table.add_row([
                    name,
                    group['GroupId'],
                    test_id,
                    run_by_user])
            click.echo(aws_table.get_string(title="SGs used on AWS"))
        else:
            click.secho("No placement groups found for selected filters in AWS!", fg="yellow")

    def list_resources_on_gce():
        for project in SUPPORTED_PROJECTS:
            with environment(SCT_GCE_PROJECT=project):
                click.secho(f"Checking GCE ({project})...", fg='green')
                gce_instances = list_instances_gce(tags_dict=params, running=get_all_running, verbose=verbose)
                if gce_instances:
                    gce_table = PrettyTable(table_header)
                    gce_table.align = "l"
                    gce_table.sortby = 'LaunchTime'
                    for instance in gce_instances:
                        tags = gce_meta_to_dict(instance.metadata)
                        public_ips = gce_public_addresses(instance)
                        public_ips = ", ".join(public_ips) if None not in public_ips else "N/A"
                        gce_table.add_row([instance.name,
                                           instance.zone.split('/')[-1],
                                           public_ips if get_all_running else instance.status,
                                           tags.get('TestId', 'N/A') if tags else "N/A",
                                           tags.get('RunByUser', 'N/A') if tags else "N/A",
                                           instance.creation_timestamp,
                                           ])
                    click.echo(gce_table.get_string(title="Resources used on GCE"))
                else:
                    click.secho("Nothing found for selected filters in GCE!", fg="yellow")

    def list_resources_on_eks():
        click.secho("Checking EKS...", fg='green')
        eks_clusters = list_clusters_eks(tags_dict=params, verbose=verbose)
        if eks_clusters:
            eks_table = PrettyTable(["Name", "TestId", "Region", "RunByUser", "CreateTime"])
            eks_table.align = "l"
            eks_table.sortby = 'CreateTime'
            for cluster in eks_clusters:
                tags = gce_meta_to_dict(cluster.extra['metadata'])
                eks_table.add_row([cluster.name,
                                   tags.get('TestId', 'N/A') if tags else "N/A",
                                   cluster.region_name,
                                   tags.get('RunByUser', 'N/A') if tags else "N/A",
                                   cluster.create_time,
                                   ])
            click.echo(eks_table.get_string(title="EKS clusters"))
        else:
            click.secho("Nothing found for selected filters in EKS!", fg="yellow")

        click.secho("Checking AWS Load Balancers...", fg='green')
        load_balancers = list_load_balancers_aws(tags_dict=params, verbose=verbose)
        if load_balancers:
            aws_table = PrettyTable(["Name", "Region", "TestId", "RunByUser"])
            aws_table.align = "l"
            aws_table.sortby = 'Name'
            for elb in load_balancers:
                tags = aws_tags_to_dict(elb.get('Tags'))
                test_id = tags.get("TestId", "N/A")
                run_by_user = tags.get("RunByUser", "N/A")
                _, _, _, region, _, name = elb['ResourceARN'].split(':')
                aws_table.add_row([
                    name,
                    region,
                    test_id,
                    run_by_user,
                ])
            click.echo(aws_table.get_string(title="ELBs used on AWS"))
        else:
            click.secho("No load balancers found for selected filters in AWS!", fg="yellow")

        click.secho("Checking AWS Cloudformation Stacks ...", fg='green')
        cfn_stacks = list_cloudformation_stacks_aws(tags_dict=params, verbose=verbose)
        if cfn_stacks:
            aws_table = PrettyTable(["Name", "Region", "TestId", "RunByUser"])
            aws_table.align = "l"
            aws_table.sortby = 'Name'
            for stack in cfn_stacks:
                tags = aws_tags_to_dict(stack.get('Tags'))
                test_id = tags.get("TestId", "N/A")
                run_by_user = tags.get("RunByUser", "N/A")
                _, _, _, region, _, name = stack['ResourceARN'].split(':')
                aws_table.add_row([
                    name,
                    region,
                    test_id,
                    run_by_user,
                ])
            click.echo(aws_table.get_string(title="Cloudformation Stacks used on AWS"))
        else:
            click.secho("No Cloudformation stacks found for selected filters in AWS!", fg="yellow")

    def list_resources_on_gke():
        click.secho("Checking GKE...", fg='green')
        gke_clusters = list_clusters_gke(tags_dict=params, verbose=verbose)
        if gke_clusters:
            gke_table = PrettyTable(["Name", "Region-AZ", "TestId", "RunByUser", "CreateTime"])
            gke_table.align = "l"
            gke_table.sortby = 'CreateTime'
            for cluster in gke_clusters:
                tags = gce_meta_to_dict(cluster.metadata)
                gke_table.add_row([cluster.name,
                                   cluster.zone,
                                   tags.get('TestId', 'N/A') if tags else "N/A",
                                   tags.get('RunByUser', 'N/A') if tags else "N/A",
                                   cluster.cluster_info['createTime'],
                                   ])
            click.echo(gke_table.get_string(title="GKE clusters"))
        else:
            click.secho("Nothing found for selected filters in GKE!", fg="yellow")

    def list_resources_on_docker():
        click.secho("Checking Docker...", fg="green")
        docker_resources = \
            list_resources_docker(tags_dict=params, running=get_all_running, group_as_builder=True, verbose=verbose)

        if any(docker_resources.values()):
            if docker_resources.get("containers"):
                docker_table = PrettyTable(["Name", "Builder", "Public IP" if get_all_running else "Status",
                                            "TestId", "RunByUser", "Created"])
                docker_table.align = "l"
                docker_table.sortby = "Created"
                for builder_name, docker_containers in docker_resources["containers"].items():
                    for container in docker_containers:
                        container.reload()
                        docker_table.add_row([
                            container.name,
                            builder_name,
                            container.attrs["NetworkSettings"]["IPAddress"] if get_all_running else container.status,
                            container.labels.get("TestId", "N/A"),
                            container.labels.get("RunByUser", "N/A"),
                            container.attrs.get("Created", "N/A"),
                        ])
                click.echo(docker_table.get_string(title="Containers used on Docker"))
            if docker_resources.get("images"):
                docker_table = PrettyTable(["Name", "Builder", "TestId", "RunByUser", "Created"])
                docker_table.align = "l"
                docker_table.sortby = "Created"
                for builder_name, docker_images in docker_resources["images"].items():
                    for image in docker_images:
                        image.reload()
                        for tag in image.tags:
                            docker_table.add_row([
                                tag,
                                builder_name,
                                image.labels.get("TestId", "N/A"),
                                image.labels.get("RunByUser", "N/A"),
                                image.attrs.get("Created", "N/A"),
                            ])
                click.echo(docker_table.get_string(title="Images used on Docker"))
        else:
            click.secho("Nothing found for selected filters in Docker!", fg="yellow")

    def list_resources_on_azure():
        click.secho("Checking Azure instances...", fg='green')
        instances: List[VmInstance] = []
        for provisioner in AzureProvisioner.discover_regions(params.get("TestId", "")):
            instances += provisioner.list_instances()
        if user:
            instances = [inst for inst in instances if inst.tags.get("RunByUser") == user]
        if instances:
            azure_table = PrettyTable(["Name", "Region-AZ", "PublicIP", "TestId", "RunByUser", "LaunchTime"])
            azure_table.align = "l"
            azure_table.sortby = 'RunByUser'

            for instance in instances:
                creation_time = instance.creation_time.isoformat(
                    sep=" ", timespec="seconds") if instance.creation_time else "N/A"
                tags = instance.tags
                test_id = tags.get("TestId", "N/A")
                run_by_user = tags.get("RunByUser", "N/A")
                azure_table.add_row([
                    instance.name,
                    instance.region,
                    instance.public_ip_address,
                    test_id,
                    run_by_user,
                    creation_time])
            click.echo(azure_table.get_string(title="Instances used on Azure"))
        else:
            click.secho("Nothing found for selected filters in Azure!", fg="yellow")

    backend_listing_map = {
        "aws": list_resources_on_aws,
        "gce": list_resources_on_gce,
        "k8s-gke": list_resources_on_gke,
        "k8s-eks": list_resources_on_eks,
        "docker": list_resources_on_docker,
        "azure": list_resources_on_azure
    }
    if list_resources_per_backend_type := backend_listing_map.get(backend_type):
        list_resources_per_backend_type()
    else:
        for list_resources_per_backend_type in backend_listing_map.values():
            list_resources_per_backend_type()


@cli.command('list-images', help="List machine images")
@cloud_provider_option(default="aws", required=False, help="Cloud provided to query. Defaults to aws.")
@click.option('-br', '--branch',
              type=str,
              help="Branch to query images for. Defaults to 'master:latest' Mutually exclusive with --version.")
@click.option('-v', '--version',
              type=str,
              help="List images by version. Use '-v all' for all versions. "
                   "OSS format: <4.3> Enterprise format: <enterprise-2021.1>. Mutually exclusive with --branch.")
@click.option('-r', '--region',
              type=CloudRegion(),
              help="Cloud region to query images in. Defaults to eu-west-1",
              default="eu-west-1")
@click.option('-a', '--arch',
              type=click.Choice(AwsArchType.__args__),
              default='x86_64',
              help="architecture of the AMI (default: x86_64)")  # pylint: disable=too-many-locals
def list_images(cloud_provider: str, branch: str, version: str, region: str, arch: AwsArchType):
    add_file_logger()
    version_fields = ["Backend", "Name", "ImageId", "CreationDate"]
    version_fields_with_tag_name = version_fields + ["NameTag"]
    #  TODO: align branch and version fields once scylla-pkg#2995 is resolved
    branch_specific_fields = ["BuildId", "Arch", "ScyllaVersion"]
    account_field = ["OwnerId"]
    branch_fields = version_fields + branch_specific_fields
    branch_fields_with_tag_name = version_fields_with_tag_name + branch_specific_fields + account_field
    if version and branch:
        click.echo("Use --version or --branch, not both.")
        return

    branch = branch or "master:latest"

    if version is not None:
        match cloud_provider:
            case "aws":
                rows = get_ami_images_versioned(region_name=region, arch=arch, version=version)
                click.echo(
                    create_pretty_table(rows=rows, field_names=version_fields_with_tag_name).get_string(
                        title=f"AWS Machine Images by Version in region {region}")
                )
            case "gce":
                if arch:
                    #  TODO: align branch and version fields once scylla-pkg#2995 is resolved
                    click.echo("WARNING:--arch option not implemented currently for GCE machine images.")
                rows = get_gce_images_versioned(version=version)

                click.echo(
                    create_pretty_table(rows=rows, field_names=version_fields).get_string(
                        title="GCE Machine Images by version")
                )
            case "azure":
                if arch:
                    click.echo("WARNING:--arch option not implemented currently for Azure machine images.")
                azure_images = azure_utils.get_released_scylla_images(scylla_version=version, region_name=region)
                rows = []
                for image in azure_images:
                    rows.append(['Azure', image.name, image.unique_id, 'N/A'])
                click.echo(
                    create_pretty_table(rows=rows, field_names=version_fields).get_string(
                        title="Azure Machine Images by version")
                )

            case _:
                click.echo(f"Cloud provider {cloud_provider} is not supported")

    elif branch:
        if ":" not in branch:
            branch += ":all"

        match cloud_provider:
            case "aws":
                region = region or "eu-west-1"
                ami_images = get_ami_images(branch=branch, region=region, arch=arch)
                click.echo(
                    create_pretty_table(rows=ami_images, field_names=branch_fields_with_tag_name).get_string(
                        title=f"AMI Machine Images for {branch} in region {region}"
                    )
                )
            case "gce":
                gce_images = get_gce_images(branch=branch, arch=arch)
                click.echo(
                    create_pretty_table(rows=gce_images, field_names=branch_fields).get_string(
                        title=f"GCE Machine Images for {branch}"
                    )
                )
            case "azure":
                if arch:
                    click.echo("WARNING:--arch option not implemented currently for Azure machine images.")
                azure_images = azure_utils.get_scylla_images(scylla_version=branch, region_name=region)
                rows = []
                for image in azure_images:
                    rows.append(['Azure', image.name, image.id, 'N/A'])
                click.echo(
                    create_pretty_table(rows=rows, field_names=version_fields).get_string(
                        title="Azure Machine Images by version")
                )

            case _:
                click.echo(f"Cloud provider {cloud_provider} is not supported")


@cli.command('list-repos', help='List repos url of Scylla formal versions')
@click.option('-d', '--dist-type', type=click.Choice(['centos', 'ubuntu', 'debian']),
              default='centos', help='Distribution type')
@click.option('-v', '--dist-version', type=click.Choice(['xenial', 'trusty', 'bionic', 'focal',  # Ubuntu
                                                         'jessie', 'stretch', 'buster', 'bullseye']),  # Debian
              default=None, help='deb style versions')
def list_repos(dist_type, dist_version):
    add_file_logger()

    if not dist_type == 'centos' and dist_version is None:
        click.secho("when passing --dist-type=debian/ubuntu need to pass --dist-version as well", fg='red')
        sys.exit(1)

    repo_maps = get_s3_scylla_repos_mapping(dist_type, dist_version)

    tbl = PrettyTable(["Version Family", "Repo Url"])
    tbl.align = "l"

    for version_prefix, repo_url in repo_maps.items():
        tbl.add_row([version_prefix, repo_url])

    click.echo(tbl.get_string(title="Scylla Repos"))


@cli.command('get-scylla-base-versions', help='Get Scylla base versions of upgrade')
@click.option('-s', '--scylla-version', type=str,
              help='Scylla version, eg: 4.5, 2021.1')
@click.option('-r', '--scylla-repo', type=str,
              help='Scylla repo')
@click.option('-d', '--linux-distro', type=str, help='Linux Distribution type')
@click.option('-o', '--only-print-versions', type=bool, default=False, required=False, help='')
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends), help="Backend to use")
def get_scylla_base_versions(scylla_version, scylla_repo, linux_distro, only_print_versions, backend):  # pylint: disable=too-many-locals
    """
    Upgrade test try to upgrade from multiple supported base versions, this command is used to
    get the base versions according to the scylla repo and distro type, then we don't need to hardcode
    the base version for each branch.
    """
    add_file_logger()

    with Path("defaults/test_default.yaml").open(mode="r", encoding="utf-8") as test_defaults_yaml:
        test_defaults = yaml.safe_load(test_defaults_yaml)

    if not linux_distro or linux_distro == "null":
        linux_distro = test_defaults.get("scylla_linux_distro")

    version_detector = UpgradeBaseVersion(scylla_repo, linux_distro, scylla_version)

    if not version_detector.dist_type == 'centos' and version_detector.dist_version is None:
        click.secho("when passing --dist-type=debian/ubuntu need to pass --dist-version as well", fg='red')
        sys.exit(1)

    # We can't detect the support versions for this distro, which shares the repo with others, eg: centos8
    # so we need to assign the start support versions for it.
    version_detector.set_start_support_version(backend)

    supported_versions, version_list = version_detector.get_version_list()
    click.echo(f'Supported Versions: {supported_versions}')

    if only_print_versions:
        click.echo(f"Base Versions: {' '.join(version_list)}")
        return

    tbl = PrettyTable(["Version Family", "Repo Url"])
    tbl.align = "l"
    for version in version_list:
        tbl.add_row([version, version_detector.repo_maps[version]])
    click.echo(tbl.get_string(title="Base Versions"))
    return


@cli.command('output-conf', help="Output test configuration readed from the file")
@click.argument('config_files', type=str, default='')
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends))
def output_conf(config_files, backend):
    add_file_logger()

    if backend:
        os.environ['SCT_CLUSTER_BACKEND'] = backend
    if config_files:
        os.environ['SCT_CONFIG_FILES'] = config_files
    config = SCTConfiguration()
    click.secho(config.dump_config(), fg='green')
    sys.exit(0)


def _run_yaml_test(backend, full_path, env):
    output = []
    error = False
    output.append(f'---- linting: {full_path} -----')
    while os.environ:
        os.environ.popitem()
    for key, value in env.items():
        os.environ[key] = value
    os.environ['SCT_CLUSTER_BACKEND'] = backend
    os.environ['SCT_CONFIG_FILES'] = full_path
    logging.getLogger().handlers = []
    logging.getLogger().disabled = True
    try:
        config = SCTConfiguration()
        config.verify_configuration()
        config.check_required_files()
    except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
        output.append(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        error = True
    return error, output


@cli.command(help="Test yaml in test-cases directory")
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends), default='aws')
@click.option('-i', '--include', type=str, default='')
@click.option('-e', '--exclude', type=str, default='')
def lint_yamls(backend, exclude: str, include: str):  # pylint: disable=too-many-locals,too-many-branches
    if not include:
        raise ValueError('You did not provide include filters')

    exclude_filters = []
    for flt in exclude.split(','):
        if not flt:
            continue
        try:
            exclude_filters.append(re.compile(flt))
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            raise ValueError(f'Exclude filter "{flt}" compiling failed with: {exc}') from exc

    include_filters = []
    for flt in include.split(','):
        if not flt:
            continue
        try:
            include_filters.append(re.compile(flt))
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            raise ValueError(f'Include filter "{flt}" compiling failed with: {exc}') from exc

    original_env = {**os.environ}
    process_pool = ProcessPoolExecutor(max_workers=5)  # pylint: disable=consider-using-with

    features = []
    for root, _, files in os.walk('./test-cases'):
        for file in files:
            full_path = os.path.join(root, file)
            if not any((flt.search(file) or flt.search(full_path) for flt in include_filters)):
                continue
            if any((flt.search(file) or flt.search(full_path) for flt in exclude_filters)):
                continue
            features.append(process_pool.submit(_run_yaml_test, backend, full_path, original_env))

    failed = False
    for pp_feature in features:
        error, pp_output = pp_feature.result()
        if error:
            failed = True
            click.secho('\n'.join(pp_output), fg='red')
        else:
            click.secho('\n'.join(pp_output), fg='green')
    print()
    sys.exit(1 if failed else 0)


@cli.command(help="Check test configuration file")
@click.argument('config_file', type=str, default='')
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends), default='aws')
def conf(config_file, backend):
    add_file_logger()

    if backend:
        os.environ['SCT_CLUSTER_BACKEND'] = backend
    if config_file:
        os.environ['SCT_CONFIG_FILES'] = config_file
    config = SCTConfiguration()
    try:
        config.verify_configuration()
        config.check_required_files()
    except Exception as ex:  # pylint: disable=broad-except
        logging.exception(str(ex))
        click.secho(str(ex), fg='red')
        sys.exit(1)
    else:
        click.secho(config.dump_config(), fg='green')
        sys.exit(0)


@cli.command('conf-docs', help="Show all available configuration in yaml/markdown format")
@click.option('-o', '--output-format', type=click.Choice(["yaml", "markdown"]), default="yaml", help="type of the output")
def conf_docs(output_format):
    add_file_logger()

    os.environ['SCT_CLUSTER_BACKEND'] = "aws"  # just to pass SCTConfiguration() verification.

    config_logger = logging.getLogger('sdcm.sct_config')
    config_logger.setLevel(logging.ERROR)
    if output_format == 'markdown':
        click.secho(SCTConfiguration().dump_help_config_markdown())
    elif output_format == 'yaml':
        click.secho(SCTConfiguration().dump_help_config_yaml())


@cli.command('update-conf-docs', help="Update the docs configuration markdown")
def update_conf_docs():
    add_file_logger()

    os.environ['SCT_CLUSTER_BACKEND'] = "aws"  # just to pass SCTConfiguration() verification.

    config_logger = logging.getLogger('sdcm.sct_config')
    config_logger.setLevel(logging.ERROR)
    markdown_file = Path(__name__).parent / 'docs' / 'configuration_options.md'
    markdown_file.write_text(SCTConfiguration().dump_help_config_markdown())
    click.secho(f"docs written into {markdown_file}")


@cli.command("perf-regression-report", help="Generate and send performance regression report")
@click.option("-i", "--es-id", required=True, type=str, help="Id of the run in Elastic Search")
@click.option("-e", "--emails", required=True, type=str, help="Comma separated list of emails. Example a@b.com,c@d.com")
@click.option("--es-index", default="performancestatsv2", help="Elastic Search index")
@click.option("--extra-jobs-to-compare", default=None, type=str, multiple=True, help="Extra jobs to compare")
def perf_regression_report(es_id, emails, es_index, extra_jobs_to_compare):
    add_file_logger()
    emails = emails.split(',')
    if not emails:
        LOGGER.warning("No email recipients. Email will not be sent")
        sys.exit(1)
    results_analyzer = PerformanceResultsAnalyzer(es_index=es_index, es_doc_type="test_stats",
                                                  email_recipients=emails, logger=LOGGER)
    results_analyzer.check_regression(es_id, extra_jobs_to_compare=extra_jobs_to_compare)

    logdir = Path(get_test_config().logdir())
    email_results_file = logdir / "email_data.json"
    test_results = read_email_data_from_file(email_results_file)
    if not test_results:
        LOGGER.error("Test Results file not found")
        sys.exit(1)
    LOGGER.info('Email will be sent to next recipients: %s', emails)
    start_time = format_timestamp(time.time())
    logs = list_logs_by_test_id(test_results.get('test_id', es_id.split('_')[0]))
    send_perf_email(results_analyzer, test_results, logs, emails, logdir, start_time)


@click.group(help="Group of commands for investigating testrun")
def investigate():
    pass


@investigate.command('show-logs', help="Show logs collected for testrun filtered by test-id")
@click.argument('test_id')
@click.option('-o', '--output-format', type=click.Choice(["table", "markdown"]), default="table", help="type of the output")
@click.option("--update-argus/--no-update-argus", type=bool, required=False, default=False, help='Update argus with links')
def show_log(test_id, output_format, update_argus: bool):
    add_file_logger()

    files = list_logs_by_test_id(test_id)

    if output_format == 'table':
        table = PrettyTable(["Date", "Log type", "Link"])
        table.align = "l"
        for log in files:
            table.add_row([log["date"].strftime("%Y%m%d_%H%M%S"), log["type"], log["link"]])
        click.echo(table.get_string(title="Log links for testrun with test id {}".format(test_id)))
    elif output_format == 'markdown':
        click.echo("\n## Logs\n")
        for log in files:
            click.echo(f'* **{log["type"]}** - {log["link"]}')

    if update_argus:
        try:
            store_logs_in_argus(test_id=test_id, logs=reduce(lambda acc, log: acc[log["type"]].append(
                log["link"]) or acc, files, defaultdict(list)), update=True)
        except Exception:  # pylint: disable=broad-except # noqa: BLE001
            LOGGER.error("Error updating logs in argus.", exc_info=True)


@investigate.command('show-monitor', help="Run monitoring stack with saved data locally")
@click.argument('test_id')
@click.option("--cluster-name", type=str, required=False, help='Cluster name (relevant for multi-tenant test)')
@click.option("--date-time", type=str, required=False, help='Datetime of monitor-set archive is collected')
@click.option("--kill", type=bool, required=False, help='Kill and remove containers')
def show_monitor(test_id, date_time, kill, cluster_name):
    add_file_logger()

    click.echo('Search monitoring stack archive files for test id {} and restoring...'.format(test_id))
    containers = {}
    try:
        containers = restore_monitoring_stack(test_id, date_time)
    except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
        LOGGER.error(details)

    if not containers:
        click.echo('Errors were found when restoring Scylla monitoring stack')
        kill_running_monitoring_stack_services()
        sys.exit(1)

    for cluster, containers_ports in containers.items():
        if cluster_name and cluster != cluster_name:
            continue

        click.echo(f'Monitoring stack for cluster {cluster} restored')
        table = PrettyTable(['Service', 'Container', 'Link'], align="l")
        for docker in get_monitoring_stack_services(ports=containers_ports):
            table.add_row([docker["service"], docker["name"], f"http://{SCT_RUNNER_HOST}:{docker['port']}"])
        click.echo(table.get_string(title=f'Monitoring stack services for cluster {cluster}'))
        click.echo("")
        if kill:
            kill_running_monitoring_stack_services(ports=containers_ports)


@investigate.command('show-jepsen-results', help="Run a server with Jepsen results")
@click.argument('test_id')
def show_jepsen_results(test_id):
    add_file_logger()

    click.secho(message=f"\nSearch Jepsen results archive files for test id {test_id} and restoring...\n", fg="green")
    jepsen = JepsenResults()
    if jepsen.restore_jepsen_data(test_id):
        click.secho(message=f"\nJepsen data restored, starting web server on "
                            f"http://{SCT_RUNNER_HOST}:{jepsen.jepsen_results_port}/",
                    fg="green")
        detach = SCT_RUNNER_HOST != "127.0.0.1"
        if not detach:
            click.secho(message="Press Ctrl-C to stop the server.", fg="green")
        click.echo("")
        jepsen.run_jepsen_web_server(detach=detach)


@investigate.command('search-builder', help='Search builder where test run with test-id located')
@click.argument('test-id')
def search_builder(test_id):
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)
    add_file_logger()

    results = get_builder_by_test_id(test_id)
    tbl = PrettyTable(['Builder Name', "Public IP", "path"])
    tbl.align = 'l'
    for result in results:
        tbl.add_row([result['builder']['name'], result['builder']['public_ip'], result['path']])

    click.echo(tbl.get_string(title='Found builders for Test-id: {}'.format(test_id)))


@investigate.command('show-events', help='Return content of file events_log/events for running job by test-id')
@click.argument('test-id')
@click.option("--follow", type=bool, required=False, is_flag=True, default=False,
              help="Follow job events log file (similar tail -f <file>)")
@click.option("--last-n", type=int, required=False, help="return last n lines from events.log file")
@click.option("--save-to", type=str, required=False, help="Download events.log file and save to provided dir")
def show_events(test_id: str, follow: bool = False, last_n: int = None, save_to: str = None):
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)
    add_file_logger()
    builders = get_builder_by_test_id(test_id)

    if not builders:
        LOGGER.info("Builder was not found for provided test-id %s", test_id)

    for builder in builders:
        LOGGER.info(
            "Applying action for events.log on builder %s:%s...", builder['builder']['name'], builder['builder']['public_ip'])
        remoter = builder["builder"]["remoter"]

        if follow or last_n:
            options = "-f " if follow else ""
            options += f"-n {last_n} " if last_n else ""
            try:
                remoter.run("tail %s %s/events_log/events.log", options, builder['path'])
            except KeyboardInterrupt:
                LOGGER.info('Monitoring events.log for test-id %s stopped!', test_id)
        elif save_to:
            remoter.receive_files(f"{builder['path']}/events_log/events.log", save_to)
            LOGGER.info("Events saved to %s", save_to)
        else:
            remoter.run(f"cat {builder['path']}/events_log/events.log")
    click.echo("Show events done.")


cli.add_command(investigate)


@cli.command('unit-tests', help="Run all the SCT internal unit-tests")
@click.option("-t", "--test", required=False, default="",
              help="Run specific test file from unit-tests directory")
def unit_tests(test):
    sys.exit(pytest.main(['-v', '-p', 'no:warnings', '-m', 'not integration', 'unit_tests/{}'.format(test)]))


@cli.command('integration-tests', help="Run all the SCT internal integration-tests")
@click.option("-t", "--test", required=False, default="",
              help="Run specific test file from unit-tests directory")
def integration_tests(test):
    get_test_config().logdir()
    add_file_logger()

    # setup prerequisites for the integration test is identical
    # to the kind local functional tests
    # TODO: to refactor setup_prerequisites out of LocalKindCluster
    sct_config = SCTConfiguration()
    local_cluster = mini_k8s.LocalKindCluster(
        software_version="",
        user_prefix="",
        params=sct_config,
    )
    local_cluster.setup_prerequisites()

    sys.exit(pytest.main(['-v', '-p', 'no:warnings', '-m', 'integration', 'unit_tests/{}'.format(test)]))


@cli.command('pre-commit', help="Run pre-commit checkers")
def pre_commit():
    result = 0
    target = "origin/$CHANGE_TARGET" if 'CHANGE_TARGET' in os.environ else 'upstream/master'
    result += os.system(
        "bash -ec 'rm *.commit_msg || true ;"
        f"for c in $(git rev-list {target}..HEAD --no-merges); do git show -s --format='%B' $c > $c.commit_msg ; done; "
        "for f in *.commit_msg ; do echo linting $f ; pre-commit run --hook-stage commit-msg --commit-msg-filename $f; done'"
    )
    result += os.system('pre-commit run -a --show-diff-on-failure')
    result = 1 if result else 0
    sys.exit(result)


class OutputLogger():
    def __init__(self, filename, terminal):
        self.terminal = terminal
        self.log = open(filename, "a", encoding="utf-8")  # pylint: disable=consider-using-with

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):  # pylint: disable=no-self-use
        return False


@cli.command('run-test', help="Run SCT test using unittest")
@click.argument('argv')
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends), help="Backend to use")
@click.option('-c', '--config', multiple=True, type=click.Path(exists=True), help="Test config .yaml to use, can have multiple of those")
@click.option('-l', '--logdir', help="Directory to use for logs")
def run_test(argv, backend, config, logdir):
    if config:
        os.environ['SCT_CONFIG_FILES'] = str(list(config))
    if backend:
        os.environ['SCT_CLUSTER_BACKEND'] = backend

    if logdir:
        os.environ['_SCT_LOGDIR'] = logdir

    os.environ['SCT_TEST_METHOD'] = argv
    logfile = os.path.join(get_test_config().logdir(), 'output.log')
    sys.stdout = OutputLogger(logfile, sys.stdout)
    sys.stderr = OutputLogger(logfile, sys.stderr)

    unittest.main(module=None, argv=['python -m unittest', argv],
                  failfast=False, buffer=False, catchbreak=True, testLoader=SctLoader())


@cli.command('run-pytest', help="Run tests using pytest")
@click.argument('target')
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends), help="Backend to use")
@click.option('-c', '--config', multiple=True, type=click.Path(exists=True), help="Test config .yaml to use, can have multiple of those")
@click.option('-l', '--logdir', help="Directory to use for logs")
def run_pytest(target, backend, config, logdir):
    if config:
        os.environ['SCT_CONFIG_FILES'] = str(list(config))
    if backend:
        os.environ['SCT_CLUSTER_BACKEND'] = backend

    if logdir:
        os.environ['_SCT_LOGDIR'] = logdir
    _logdir = Path(get_test_config().logdir())
    logfile = _logdir / 'output.log'
    junit_file = _logdir / 'junit.xml'
    sys.stdout = OutputLogger(logfile, sys.stdout)
    sys.stderr = OutputLogger(logfile, sys.stderr)
    if not target:
        print("argv is referring to the directory or file that contain tests, it can't be empty")
        sys.exit(1)
    return_code = pytest.main(['-s', '-v', f'--junit-xml={junit_file}', '-p', 'no:warnings', target])
    test_config = get_test_config()
    test_config.argus_client().sct_submit_junit_report(file_name=junit_file.name, raw_content=junit_file.read_text())
    sys.exit(return_code)


@cli.command("cloud-usage-report", help="Generate and send Cloud usage report")
@click.option("-e", "--emails", required=True, type=str, help="Comma separated list of emails. Example a@b.com,c@d.com")
@click.option("-t", "--report-type", required=True,
              type=click.Choice(choices=["general", "last-7-days-qa", "last-7-days-non-qa"], case_sensitive=False),
              help="Type of the report")
@click.option("-u", "--user", required=False, type=str, default="",
              help="User or instance owner. Applicable for last-7-days-* reports")
def cloud_usage_report(emails, report_type, user):
    add_file_logger()

    email_list = emails.split(",")
    click.secho(message=f"Will send {user} Cloud Usage '{report_type}' report to {email_list}", fg="green")
    match report_type:
        case "general": cloud_report(mail_to=email_list)
        case "last-7-days-qa": cloud_qa_report(mail_to=email_list, user=user)
        case "last-7-days-non-qa": cloud_non_qa_report(mail_to=email_list, user=user)
    click.secho(message="Done.", fg="yellow")


@cli.command('collect-logs', help='Collect logs from cluster by test-id')
@click.option('--test-id', help='Find cluster by test-id')
@click.option('--logdir', help='Path to directory with sct results')
@click.option('--backend', help='Cloud where search nodes', default=None)
@click.option('--config-file', type=str, help='config test file path')
def collect_logs(test_id=None, logdir=None, backend=None, config_file=None):
    # pylint: disable=too-many-nested-blocks,too-many-branches
    add_file_logger()

    from sdcm.logcollector import Collector  # pylint: disable=import-outside-toplevel
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)
    if backend is None:
        if os.environ.get('SCT_CLUSTER_BACKEND', None) is None:
            os.environ['SCT_CLUSTER_BACKEND'] = backend = 'aws'
    else:
        os.environ['SCT_CLUSTER_BACKEND'] = backend

    if config_file and not os.environ.get('SCT_CONFIG_FILES', None):
        os.environ['SCT_CONFIG_FILES'] = config_file

    config = SCTConfiguration()

    collector = Collector(test_id=test_id, params=config, test_dir=logdir)

    collected_logs = collector.run()

    table = PrettyTable(['Cluster set', 'Link'])
    table.align = 'l'
    for cluster_type, s3_links in collected_logs.items():
        for link in s3_links:
            current_cluster_type = cluster_type
            # Cover case when archive is created per log file not all logs in one archive.
            # Here log name is extracted from archive name. For example:
            # for link https://cloudius-jenkins-test.s3.amazonaws.com/c63a6913-6253-45a0-b5cf-d553f713fe81/20211222_
            # 101636/warning-c63a6913.log.tar.gz
            #  current_cluster_type will be "warning"
            if cluster_type == 'sct-runner' and cluster_type not in link:
                current_cluster_type = link.split("/")[-1].split("-")[0]
            table.add_row([current_cluster_type, link])

    click.echo(table.get_string(title="Collected logs by test-id: {}".format(collector.test_id)))
    update_sct_runner_tags(backend=backend, test_id=collector.test_id, tags={"logs_collected": True})

    if collector.test_id:
        store_logs_in_argus(test_id=UUID(collector.test_id), logs=collected_logs)


def store_logs_in_argus(test_id: UUID, logs: dict[str, list[list[str] | str]], update: bool = False):
    try:
        argus_client = get_argus_client(run_id=test_id)
        log_links = []
        existing_links = [name for [name, _] in argus_client.get_run().get('logs', [])] if update else []
        for _, s3_links in logs.items():
            for link in s3_links:
                file_name = link.split("/")[-1]
                if update and file_name not in existing_links:
                    LOGGER.info("Adding missing log link %s to Argus...", file_name)
                    log_links.append(LogLink(log_name=file_name, log_link=link))
                elif not update:
                    log_links.append(LogLink(log_name=file_name, log_link=link))
                else:
                    LOGGER.info("Skipping %s because it has been already sent to argus.", file_name)
        argus_client.submit_sct_logs(log_links)

        if not argus_client.get_run().get("events"):
            argus_offline_collect_events(client=argus_client)
    except Exception:  # pylint: disable=broad-except
        LOGGER.error("Error saving logs to argus", exc_info=True)


def get_test_results_for_failed_test(test_status, start_time):
    return {
        "job_url": os.environ.get("BUILD_URL"),
        "subject": f"{test_status}: {os.environ.get('JOB_NAME')}: {start_time}",
        "start_time": start_time,
        "end_time": format_timestamp(time.time()),
        "grafana_screenshots": "",
        "nodes": "",
        "test_id": "",
        "username": ""
    }


# pylint: disable=too-many-arguments,too-many-branches,too-many-statements
@cli.command('send-email', help='Send email with results for testrun')
@click.option('--test-id', help='Test-id of run')
@click.option('--test-status', help='Override test status FAILED|ABORTED')
@click.option('--start-time', help='Override test start time')
@click.option('--started-by', help='Default user that started the test')
@click.option('--runner-ip', type=str, required=False, help="Sct runner ip for the running test")
@click.option('--email-recipients', help="Send email to next recipients")
@click.option('--logdir', help='Directory where to find testrun folder')
def send_email(test_id=None, test_status=None, start_time=None, started_by=None, runner_ip=None,  # noqa: PLR0912
               email_recipients=None, logdir=None):
    if started_by is None:
        started_by = get_username()
    add_file_logger()

    if not email_recipients:
        LOGGER.warning("No email recipients. Email will not be sent")
        sys.exit(1)
    LOGGER.info('Email will be sent to next recipients: %s', email_recipients)
    email_recipients = email_recipients.split(',')

    if not logdir:
        logdir = os.path.expanduser('~/sct-results')
    test_results = None
    if start_time is None:
        start_time = format_timestamp(time.time())
    else:
        start_time = format_timestamp(int(start_time))
    testrun_dir = get_testrun_dir(test_id=test_id, base_dir=logdir)
    if testrun_dir:
        with open(os.path.join(testrun_dir, 'test_id'), encoding='utf-8') as file:
            test_id = file.read().strip()
        email_results_file = os.path.join(testrun_dir, "email_data.json")
        if not os.path.exists(email_results_file):
            email_results_file = "email_data.json" if os.path.exists("email_data.json") else None
        if not email_results_file:
            LOGGER.error("Results file not found")
        else:
            test_results = read_email_data_from_file(email_results_file)
    else:
        LOGGER.warning("Failed to find test directory for %s", test_id)
    if not test_results:
        if not test_status:
            test_status = 'ABORTED'
        test_results = get_test_results_for_failed_test(test_status, start_time)
        if started_by:
            test_results["username"] = started_by
        if test_id:
            test_results.update({
                "test_id": test_id,
                "nodes": get_running_instances_for_email_report(test_id, runner_ip),
                "log_links": list_logs_by_test_id(test_id)
            })
        reporter = build_reporter('TestAborted', email_recipients, testrun_dir)
        if reporter:
            reporter.send_report(test_results)
            sys.exit(1)
        else:
            LOGGER.error('failed to get a reporter')
            sys.exit(1)
        return
    job_name = os.environ.get('JOB_NAME', '')
    if reporter := test_results.get("reporter", ""):
        test_results['nodes'] = get_running_instances_for_email_report(test_id, runner_ip)
        test_results['logs_links'] = list_logs_by_test_id(test_results.get('test_id', test_id))
        if 'longevity' in job_name:
            pt_report_urls = list_parallel_timelines_report_urls(test_id=test_results.get('test_id', test_id))
            test_results['parallel_timelines_report'] = pt_report_urls[0] if pt_report_urls else None

        reporter = build_reporter(reporter, email_recipients, testrun_dir)
        if not reporter:
            LOGGER.warning("No reporter found")
            sys.exit(1)
        try:
            reporter.send_report(test_results)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.error("Failed to create email due to the following error:\n%s", traceback.format_exc())
            build_reporter("TestAborted", email_recipients, testrun_dir).send_report({
                "job_url": os.environ.get("BUILD_URL"),
                "subject": f"FAILED: {os.environ.get('JOB_NAME')}: {start_time}",
            })
    elif any(['email_body' in value for value in test_results.values()]):
        # figure out it's a perf tests with multiple emails in single file
        # based on the structure of file
        logs = list_logs_by_test_id(test_results.get('test_id', test_id))
        reporter = BaseResultsAnalyzer(es_index=test_id, es_doc_type='test_stats',
                                       email_recipients=email_recipients)
        send_perf_email(reporter, test_results, logs, email_recipients, testrun_dir, start_time)
    else:
        LOGGER.warning("failed to figure out what what to send out")
        sys.exit(1)


@cli.command('create-operator-test-release-jobs',
             help="Create pipeline jobs for a new scylla-operator branch/release")
@click.argument('branch', type=str)
@click.argument('username', envvar='JENKINS_USERNAME', type=str, required=False)
@click.argument('password', envvar='JENKINS_PASSWORD', type=str, required=False)
@click.option('--sct_branch', default='master', type=str)
@click.option('--sct_repo', default='git@github.com:scylladb/scylla-cluster-tests.git', type=str)
@click.option('--triggers/--no-triggers', default=False)
def create_operator_test_release_jobs(branch, username, password, sct_branch, sct_repo, triggers):
    add_file_logger()

    base_job_dir = f"scylla-operator/{branch}"
    server = JenkinsPipelines(
        username=username, password=password, base_job_dir=base_job_dir,
        sct_branch_name=sct_branch, sct_repo=sct_repo)
    server.create_job_tree(f'{server.base_sct_dir}/jenkins-pipelines/operator',
                           create_freestyle_jobs=triggers,
                           template_context={'release_version': get_latest_scylla_release(product='scylla-enterprise')})


@cli.command('create-qa-tools-jobs',
             help="Create pipeline jobs for a new scylla-operator branch/release")
@click.argument('username', envvar='JENKINS_USERNAME', type=str, required=False)
@click.argument('password', envvar='JENKINS_PASSWORD', type=str, required=False)
@click.option('--sct_branch', default='master', type=str)
@click.option('--sct_repo', default='git@github.com:scylladb/scylla-cluster-tests.git', type=str)
@click.option('--triggers/--no-triggers', default=False)
def create_qa_tools_jobs(username, password, sct_branch, sct_repo, triggers):
    add_file_logger()

    base_job_dir = "QA-tools"
    server = JenkinsPipelines(
        username=username, password=password, base_job_dir=base_job_dir,
        sct_branch_name=sct_branch, sct_repo=sct_repo)
    server.create_job_tree(f'{server.base_sct_dir}/jenkins-pipelines/qa',
                           create_freestyle_jobs=triggers,
                           job_name_suffix='')


@cli.command('create-performance-jobs',
             help="Create pipeline jobs for performance")
@click.argument('username', envvar='JENKINS_USERNAME', type=str, required=False)
@click.argument('password', envvar='JENKINS_PASSWORD', type=str, required=False)
@click.option('--sct_branch', default='branch-perf-v15', type=str)
@click.option('--sct_repo', default='git@github.com:scylladb/scylla-cluster-tests.git', type=str)
@click.option('--triggers/--no-triggers', default=False)
def create_performance_jobs(username, password, sct_branch, sct_repo, triggers):
    add_file_logger()

    # we start from the root of jenkins, because we have jobs to scylla-master and scylla-enterprise
    base_job_dir = ""
    server = JenkinsPipelines(
        username=username, password=password, base_job_dir=base_job_dir,
        sct_branch_name=sct_branch, sct_repo=sct_repo)
    server.create_job_tree(f'{server.base_sct_dir}/jenkins-pipelines/performance/{sct_branch}',
                           create_freestyle_jobs=triggers, job_name_suffix='')


@cli.command("create-nemesis-pipelines")
@click.option("--base-job", default=None, type=str)
@click.option("--backend", default=NemesisJobGenerator.BACKEND_TO_REGION.keys(), multiple=True)
def create_nemesis_pipelines(base_job: str, backend: list[str]):
    for backend_name in backend:
        if backend_name not in NemesisJobGenerator.BACKEND_TO_REGION.keys():
            LOGGER.warning("## Unsupported backend: %s", backend_name)
            continue
        LOGGER.info("## Generating jobs for backend %s", backend_name)
        gen = NemesisJobGenerator(base_job=base_job, backend=backend_name)
        gen.render_base_job_config()
        gen.create_test_cases_from_template()
        gen.create_job_files_from_template()


@cli.command('create-test-release-jobs', help="Create pipeline jobs for a new branch")
@click.argument('branch', type=str)
@click.argument('username', envvar='JENKINS_USERNAME', type=str, required=False)
@click.argument('password', envvar='JENKINS_PASSWORD', type=str, required=False)
@click.option('--sct_branch', default='master', type=str)
@click.option('--sct_repo', default='git@github.com:scylladb/scylla-cluster-tests.git', type=str)
def create_test_release_jobs(branch, username, password, sct_branch, sct_repo):
    add_file_logger()

    base_job_dir = f'{branch}'
    server = JenkinsPipelines(username=username, password=password, base_job_dir=base_job_dir,
                              sct_branch_name=sct_branch, sct_repo=sct_repo)
    base_path = f'{server.base_sct_dir}/jenkins-pipelines/oss'
    server.create_job_tree(base_path)

    if branch == "scylla-master":
        base_path = f'{server.base_sct_dir}/jenkins-pipelines/master-triggers'
        server.create_job_tree(base_path)


@cli.command('create-test-release-jobs-enterprise', help="Create pipeline jobs for a new branch")
@click.argument('branch', type=str)
@click.argument('username', envvar='JENKINS_USERNAME', type=str, required=False)
@click.argument('password', envvar='JENKINS_PASSWORD', type=str, required=False)
@click.option('--sct_branch', default='master', type=str)
@click.option('--sct_repo', default='git@github.com:scylladb/scylla-cluster-tests.git', type=str)
def create_test_release_jobs_enterprise(branch, username, password, sct_branch, sct_repo):
    add_file_logger()

    base_job_dir = f'{branch}'
    server = JenkinsPipelines(username=username, password=password, base_job_dir=base_job_dir,
                              sct_branch_name=sct_branch, sct_repo=sct_repo)
    base_path = f'{server.base_sct_dir}/jenkins-pipelines/enterprise'
    server.create_job_tree(base_path)


@cli.command("prepare-regions", help="Configure all required resources for SCT runs in selected cloud region")
@cloud_provider_option
@click.option("-r", "--regions", type=CloudRegion(), help="Cloud region", multiple=True)
def prepare_regions(cloud_provider, regions):
    add_file_logger()
    regions = regions or get_all_regions(cloud_provider)

    for region in regions:
        if cloud_provider == "aws":
            region = AwsRegion(region_name=region)  # noqa: PLW2901
        elif cloud_provider == "azure":
            region = AzureRegion(region_name=region)  # noqa: PLW2901
        elif cloud_provider == "gce":
            region = GceRegion(region_name=region)  # noqa: PLW2901
        else:
            raise Exception(f'Unsupported Cloud provider: `{cloud_provider}')
        region.configure()


@cli.command("configure-aws-peering", help="Configure all required resources for SCT to run in multi-dc")
@click.option("-r", "--regions", type=CloudRegion(cloud_provider='aws'),
              default=[], help="Cloud regions", multiple=True)
def configure_aws_peering(regions):
    add_file_logger()
    peering = AwsVpcPeering(regions)
    peering.configure()


@cli.command("create-runner-image",
             help=f"Create an SCT runner image in the selected cloud region."
                  f" If the requested region is not a source region"
                  f" (aws: {AwsSctRunner.SOURCE_IMAGE_REGION}, gce: {GceSctRunner.SOURCE_IMAGE_REGION},"
                  f" azure: {AzureSctRunner.SOURCE_IMAGE_REGION}) the image will be first created in the"
                  f" source region and then copied to the chosen one.")
@cloud_provider_option
@click.option("-r", "--region", required=True, type=CloudRegion(), help="Cloud region")
@click.option("-z", "--availability-zone", default="", type=str, help="Name of availability zone, ex. 'a'")
def create_runner_image(cloud_provider, region, availability_zone):
    if cloud_provider == "aws":
        assert len(availability_zone) == 1, f"Invalid AZ: {availability_zone}, availability-zone is one-letter a-z."
    add_file_logger()
    os.environ.setdefault('SCT_CLUSTER_BACKEND', cloud_provider)
    sct_config = SCTConfiguration()
    sct_runner = get_sct_runner(cloud_provider=cloud_provider, region_name=region,
                                availability_zone=availability_zone, params=sct_config)
    sct_runner.create_image()


@cli.command("create-runner-instance", help="Create an SCT runner instance in the selected cloud region")
@cloud_provider_option
@click.option("-r", "--region", required=True, type=CloudRegion(), help="Cloud region")
@click.option("-z", "--availability-zone", default="", type=str, help="Name of availability zone, ex. 'a'")
@click.option("-i", "--instance-type", required=False, type=str, default="", help="Instance type")
@click.option("-i", "--root-disk-size-gb", required=False, type=int, default=0, help="Root disk size in Gb")
@click.option("-t", "--test-id", required=True, type=str, help="Test ID")
@click.option("-tn", "--test-name", required=False, type=str, default="", help="Test Name")
@click.option("-d", "--duration", required=True, type=int, help="Test duration in MINUTES")
@click.option("-rm", "--restore-monitor", required=False, type=bool,
              help="Is the runner for restore monitor purpose or not")
@click.option("-rt", "--restored-test-id", required=False, type=str,
              help="Test ID of the test that the runner is created for restore monitor")
@click.option("-p", "--address-pool", required=False, type=str, help="ElasticIP pool to use")
def create_runner_instance(cloud_provider, region, availability_zone, instance_type, root_disk_size_gb,
                           test_id, test_name, duration, restore_monitor=False, restored_test_id="", address_pool=None):
    # pylint: disable=too-many-locals

    if cloud_provider == "aws":
        assert len(availability_zone) == 1, f"Invalid AZ: {availability_zone}, availability-zone is one-letter a-z."
    add_file_logger()
    sct_runner_ip_path = Path("sct_runner_ip")
    sct_runner_ip_path.unlink(missing_ok=True)

    os.environ.setdefault('SCT_CLUSTER_BACKEND', cloud_provider)
    sct_config = SCTConfiguration()
    sct_runner = get_sct_runner(cloud_provider=cloud_provider, region_name=region,
                                availability_zone=availability_zone, params=sct_config)

    instance_type = instance_type or sct_config.get('instance_type_runner')
    root_disk_size_gb = root_disk_size_gb or sct_config.get('root_disk_size_runner')
    test_name = test_name or test_id

    instance = sct_runner.create_instance(
        instance_type=instance_type,
        root_disk_size_gb=root_disk_size_gb,
        test_id=test_id,
        test_name=test_name,
        test_duration=duration,
        restore_monitor=restore_monitor,
        restored_test_id=restored_test_id,
        address_pool=address_pool,
    )
    if not instance:
        sys.exit(1)

    LOGGER.info("Verifying SSH connectivity...")
    runner_public_ip = sct_runner.get_instance_public_ip(instance=instance)
    remoter = sct_runner.get_remoter(host=runner_public_ip, connect_timeout=240)
    if remoter.run("true", timeout=200, verbose=False, ignore_status=True).ok:
        LOGGER.info("Successfully connected the SCT Runner. Public IP: %s", runner_public_ip)
        with sct_runner_ip_path.open(mode="w", encoding="utf-8") as sct_runner_ip_file:
            sct_runner_ip_file.write(runner_public_ip)
    else:
        LOGGER.error("Unable to SSH to %s! Exiting...", runner_public_ip)
        sys.exit(1)


@cli.command("set-runner-tags")
@click.argument("runner-ip", type=str)
@click.option("-t", "--tags", type=(str, str),
              help="Space separated key value pair to add as a new tag to the runner",
              multiple=True)
def set_runner_tags(runner_ip, tags):
    add_file_logger()
    update_sct_runner_tags(test_runner_ip=runner_ip, tags=dict(tags))


@cli.command("clean-runner-instances", help="Clean all unused SCT runner instances")
@click.option("-ip", "--runner-ip", required=False, type=str, default="")
@click.option("-ts", "--test-status", type=str, help="The result of the test run")
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends),
              help="Specific backend to use")
@click.option('--dry-run', is_flag=True, default=False, help='dry run')
@click.option("--force", is_flag=True, default=False, help="Skip cleaning logic and terminate the instance")
def clean_runner_instances(runner_ip, test_status, backend, dry_run, force):
    add_file_logger()
    clean_sct_runners(
        test_runner_ip=runner_ip, test_status=test_status, backend=backend, dry_run=dry_run, force=force)


@cli.command("run-aws-mock", help="Start AWS Mock server Docker container")
@click.option(
    "-r", "--mock-region",
    required=True,
    multiple=True,
    type=CloudRegion(cloud_provider="aws"),
    help="Mock this AWS region",
)
@click.option("-f", "--force", is_flag=True, default=False, help="don't check aws_mock_ip")
@click.option("-t", "--test-id", required=False, help="SCT Test ID")
def run_aws_mock(mock_region: list[str], force: bool = False, test_id: str | None = None) -> None:
    add_file_logger()
    if test_id is None:
        test_id = str(uuid.uuid4())
    aws_mock_ip = AwsMock(test_id=test_id, regions=mock_region).run(force=force)
    LOGGER.info("New mock for %r AWS regions started and listen on %s:443 (TestId=%s)",
                mock_region, aws_mock_ip, test_id)


@cli.command("clean-aws-mocks", help="Clean running AWS mock Docker containers")
@click.option("-t", "--test-id", required=False, help="Clean AWS Mock container for test id")
@click.option(
    "-a", "--all", "all_mocks",
    is_flag=True,
    default=False,
    help="Clean all AWS Mock containers running on this host",
)
@click.option('--verbose', is_flag=True, default=False, help="if enable, will log progress")
@click.option("--dry-run", is_flag=True, default=False, help="dry run")
def clean_aws_mocks(test_id: str | None, all_mocks: bool, verbose: bool, dry_run: bool) -> None:
    add_file_logger()
    AwsMock.clean(test_id=test_id, all_mocks=all_mocks, verbose=verbose, dry_run=dry_run)


@cli.command("generate-pt-report", help="Generate parallel timelines representation for the SCT test events")
@click.option("-t", "--test-id", envvar='SCT_TEST_ID', help="Test ID to search in sct-results")
@click.option("-d", "--logdir", envvar='HOME', type=click.Path(exists=True),
              help="Directory with sct-results folder")
def generate_parallel_timelines_report(logdir: str | None, test_id: str | None) -> None:
    add_file_logger()

    event_log_file = "raw_events.log"

    LOGGER.debug("Searching for the required test run directory in %s...", logdir)
    testrun_dir = get_testrun_dir(os.path.join(logdir, "sct-results"), test_id)
    if not testrun_dir:
        click.secho(message=f"Couldn't find directory for the required test run in '{logdir}'! Aborting...", fg="red")
        sys.exit(1)
    LOGGER.info("Found the test run directory '%s'", testrun_dir)

    LOGGER.debug("Searching for the %s in %s...", event_log_file, testrun_dir)
    raw_events_log_path = next(Path(testrun_dir).glob(f"**/{event_log_file}"), None)

    if raw_events_log_path is None:
        click.secho(message=f"Couldn't find '{event_log_file}' in '{testrun_dir}'! Aborting...", fg="red")
        sys.exit(1)
    LOGGER.info("Found the file '%s'", raw_events_log_path)
    pt_report_generator = ParallelTimelinesReportGenerator(events_file=raw_events_log_path)
    pt_report_generator.generate_full_report()


@cli.command("create-es-index", help="Create ElasticSearch index with mapping ")
@click.option("-n", "--name", envvar='SCT_ES_INDEX_NAME', required=True, help="ES index name")
@click.option("-dt", "--doc-type", envvar='SCT_ES_DOC_TYPE', default="")
@click.option("-f", "--mapping-file", envvar='SCT_MAPPING_FILEPATH', type=click.Path(exists=True),
              required=True, help="Full path to es index mapping file")
def create_es_index(name: str, doc_type: str, mapping_file: str) -> None:
    add_file_logger()

    mapping_data = get_mapping(mapping_file)
    create_index(index_name=name, doc_type=doc_type, mappings=mapping_data)


@cli.command("configure-jenkins-builders", help="Configure all required jenkins builders for SCT")
@cloud_provider_option
@click.option("-r", "--regions", type=CloudRegion(), default=[], help="Cloud regions", multiple=True)
def configure_jenkins_builders(cloud_provider, regions):
    add_file_logger()
    logging.basicConfig(level=logging.INFO)

    match cloud_provider:
        case 'aws':
            AwsCiBuilder(AwsRegion('eu-west-1')).configure_auto_scaling_group()
            AwsBuilder.configure_in_all_region(regions=regions)
        case 'gce':
            GceBuilder.configure_in_all_region(regions=regions)
        case 'azure':
            raise NotImplementedError("configure_jenkins_builders doesn't support Azure yet")


@cli.command("nemesis-list", help="get the list of select disrupt function for SisyphusMonkey")
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends), help="Backend to use")
@click.option('-c', '--config', multiple=True, type=click.Path(exists=True), help="Test config .yaml to use, can have multiple of those")
def get_nemesis_list(backend, config):
    """
    # usage via command line:
    hydra nemesis-list -c test-cases/longevity/longevity-cdc-100gb-4h.yaml -c configurations/tablets.yaml

    # usage with environment variables
    export SCT_CONFIG_FILES='["test-cases/longevity/longevity-cdc-100gb-4h.yaml", "configurations/tablets.yaml"]'
    hydra nemesis-list

    """

    # NOTE: this import messes up logging for the test, since it's importing tester.py
    # directly down the line
    from unit_tests.test_nemesis import FakeTester  # pylint: disable=import-outside-toplevel

    add_file_logger()
    logging.basicConfig(level=logging.WARNING)

    if config:
        os.environ['SCT_CONFIG_FILES'] = str(list(config))
    if backend:
        os.environ['SCT_CLUSTER_BACKEND'] = backend

    tester = FakeTester()

    tester.params = SCTConfiguration()
    sisyphus_nemesis = SisyphusMonkey(tester, None)

    collected_disrupt_methods_names = [disrupt.__name__ for disrupt in sisyphus_nemesis.disruptions_list]

    click.secho(f'config files used: {pprint.pformat(tester.params.get("config_files"))}\n\n',  fg='green')
    click.secho(pprint.pformat(collected_disrupt_methods_names), fg='green')


@cli.command("create-argus-test-run", help="Initialize an argus test run.")
def create_argus_test_run():
    try:
        params = SCTConfiguration()
        git_status = get_git_status_info()
        test_config = get_test_config()
        if not params.get('test_id'):
            LOGGER.error("test_id is not set")
            return
        test_config.set_test_id_only(params.get('test_id'))
        test_config.init_argus_client(params)
        test_config.argus_client().submit_sct_run(
            job_name=get_job_name(),
            job_url=get_job_url(),
            started_by=get_username(),
            commit_id=git_status.get('branch.oid', get_git_commit_id()),
            origin_url=git_status.get('upstream.url'),
            branch_name=git_status.get('branch.upstream'),
            sct_config=None,
        )
        LOGGER.info("Initialized Argus TestRun with test id %s", get_test_config().argus_client().run_id)
    except ArgusClientError:
        LOGGER.error("Failed to submit data to Argus", exc_info=True)


@cli.command("finish-argus-test-run", help="Finish argus test run if it is not finished by SCT.")
@click.option("-s", "--jenkins-status", type=str, help="jenkins build status", required=True)
def finish_argus_test_run(jenkins_status):
    try:
        params = SCTConfiguration()
        test_config = get_test_config()
        if not params.get('test_id'):
            LOGGER.error("test_id is not set")
            return
        test_config.set_test_id_only(params.get('test_id'))
        test_config.init_argus_client(params)
        status = test_config.argus_client().get_status()
        if status in [TestStatus.PASSED, TestStatus.FAILED, TestStatus.TEST_ERROR]:
            LOGGER.info("Argus TestRun already finished with status %s", status.value)
            return
        new_status = TestStatus.FAILED
        if jenkins_status == "ABORTED":
            new_status = TestStatus.ABORTED
        test_config.argus_client().set_sct_run_status(new_status)
    except ArgusClientError:
        LOGGER.error("Failed to submit data to Argus", exc_info=True)


@cli.command("fetch-junit-from-runner", help="copy the junit.xml back")
@click.argument("runner-ip", type=str)
@click.option('-b', '--backend', type=click.Choice(SCTConfiguration.available_backends),
              help="Allows to skip making backend detection API calls.")
def fetch_junit(runner_ip, backend):
    add_file_logger()

    runner = list_sct_runners(backend=backend, test_runner_ip=runner_ip)
    proxy_command, target_ip, target_username, target_key = sct_ssh.get_proxy_command(runner[0].instance, True)
    cmd = (f'ssh -tt {proxy_command}'
           f' -i {target_key} -o "UserKnownHostsFile=/dev/null" '
           f'-o "StrictHostKeyChecking=no" -o ServerAliveInterval=10 {target_username}@{target_ip} '
           f'\'find ~/sct-results -iname junit.xml | xargs cat\'')

    output = subprocess.run(['bash', '-c', cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    junit_xml_file = Path('results') / 'junit.xml'
    junit_xml_file.parent.mkdir(parents=True, exist_ok=True)
    junit_xml_file.write_text(output.stdout.strip())


@cli.command("upload", help="Upload arbitrary log/screenshot to s3 corresponding to the test_id")
@click.option("--test-id", type=str, required=True)
@click.option("--use-argus/--no-use-argus", default=True)
@click.argument("file-path", type=str, required=True)
def upload_artifact_file(test_id: str, file_path: str, use_argus: bool):
    add_file_logger()
    if use_argus:
        params = SCTConfiguration()
        params["test_id"] = test_id
        test_config = get_test_config()
        test_config.set_test_id_only(params.get('test_id'))
        test_config.init_argus_client(params)
        client = test_config.argus_client()
        try:
            client.get_status()
        except ArgusClientError:
            LOGGER.error("Failed getting status for %s in Argus, aborting...", test_id)
            return
    else:
        client = get_test_config().argus_client()  # MagicMock

    image_exts = [".jpg", ".png"]
    file = Path(file_path)
    if file.exists():
        timestamp = datetime.now(timezone(timedelta(hours=0, minutes=0)))
        subfolder = timestamp.strftime("upload_%Y%m%d_%H%M%S")
        s3_path = f"{test_id}/{subfolder}"
        LOGGER.info("Going to upload %s to S3...", file.absolute())
        s3 = S3Storage()
        file_url = s3.upload_file(file.absolute(), s3_path)
        LOGGER.info("Uploaded %s to %s", file.absolute(), file_url)
        client.submit_sct_logs([LogLink(log_name=file.name, log_link=file_url)])
        if file.suffix in image_exts:
            client.submit_screenshots([file_url])
    else:
        LOGGER.error("File %s does not exist", file.absolute())
        return


cli.add_command(sct_ssh.ssh)
cli.add_command(sct_ssh.tunnel)
cli.add_command(sct_ssh.copy_cmd)
cli.add_command(sct_ssh.attach_test_sg_cmd)
cli.add_command(sct_ssh.ssh_cmd)
cli.add_command(sct_ssh.gcp_allow_public)
cli.add_command(sct_ssh.update_scylla_packages)
cli.add_command(sct_scan_issues.scan_issue_skips)

if __name__ == '__main__':
    cli.main(prog_name="hydra")
