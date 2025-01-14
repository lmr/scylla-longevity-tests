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
# Copyright (c) 2017 ScyllaDB

from __future__ import absolute_import, annotations
import logging
import os
from typing import Optional
from unittest.mock import MagicMock

from botocore.exceptions import ClientError
import boto3
from google.cloud.compute_v1.types import Instance as GceInstance
from mypy_boto3_ec2 import EC2Client

from sdcm.provision.aws.capacity_reservation import SCTCapacityReservation
from sdcm.provision.aws.dedicated_host import SCTDedicatedHosts
from sdcm.provision.azure.provisioner import AzureProvisioner
from sdcm.utils.argus import ArgusError, get_argus_client, terminate_resource_in_argus
from sdcm.utils.aws_kms import AwsKms
from sdcm.utils.common import (
    ParallelObject,
    all_aws_regions,
    aws_tags_to_dict,
    get_post_behavior_actions,
    get_testrun_status,
    list_cloudformation_stacks_aws,
    list_clusters_gke,
    list_clusters_eks,
    list_elastic_ips_aws,
    list_instances_aws,
    list_instances_gce,
    list_launch_templates_aws,
    list_load_balancers_aws,
    list_placement_groups_aws,
    list_resources_docker,
    list_test_security_groups,
)
from sdcm.utils.context_managers import environment
from sdcm.utils.decorators import retrying
from sdcm.utils.gce_utils import (
    GkeCleaner,
    get_gce_compute_instances_client,
)


LOGGER = logging.getLogger('utils')


def clean_cloud_resources(tags_dict, config=None, dry_run=False):
    """
    Clean up cloud resources created in various cloud platforms.

    :param config: instance of the 'SCTConfiguration' class
    :param tags_dict: key-value pairs used for filtering
    :param dry_run: boolean value which defines whether we should really cleanup resources or not
    :return: None
    """
    if "TestId" not in tags_dict and "RunByUser" not in tags_dict:
        LOGGER.error("Can't clean cloud resources, TestId or RunByUser is missing")
        return False

    cluster_backend = config.get("cluster_backend") or ''
    aws_regions = config.region_names
    gce_projects = [config.get("gce_project") or 'gcp-sct-project-1']

    if cluster_backend.startswith("k8s-local"):
        LOGGER.info("No remote resources are expected in the local K8S setups. Skipping.")
        return
    if cluster_backend in ('k8s-eks', ''):
        clean_clusters_eks(tags_dict, regions=aws_regions, dry_run=dry_run)
        clean_launch_templates_aws(tags_dict, regions=aws_regions, dry_run=dry_run)
        clean_load_balancers_aws(tags_dict, regions=aws_regions, dry_run=dry_run)
        clean_cloudformation_stacks_aws(tags_dict, regions=aws_regions, dry_run=dry_run)
    if cluster_backend in ('k8s-gke', ''):
        for project in gce_projects:
            with environment(SCT_GCE_PROJECT=project):
                clean_clusters_gke(tags_dict, dry_run=dry_run)
                clean_orphaned_gke_disks(tags_dict, dry_run=dry_run)

    if cluster_backend in ('aws', 'k8s-eks', ''):
        clean_instances_aws(tags_dict, regions=aws_regions, dry_run=dry_run)
        SCTCapacityReservation.cancel(config)
        SCTDedicatedHosts.release_by_tags(tags_dict, regions=aws_regions, dry_run=dry_run)
        clean_elastic_ips_aws(tags_dict, regions=aws_regions, dry_run=dry_run)
        clean_test_security_groups(tags_dict, regions=aws_regions, dry_run=dry_run)
        clean_placement_groups_aws(tags_dict, regions=aws_regions, dry_run=dry_run)
        if cluster_backend == 'aws' and not dry_run:
            clean_aws_kms_alias(tags_dict, aws_regions or all_aws_regions())
    if cluster_backend in ('gce', 'k8s-gke', ''):
        for project in gce_projects:
            with environment(SCT_GCE_PROJECT=project):
                clean_instances_gce(tags_dict, dry_run=dry_run)
    if cluster_backend in ('azure', ''):
        azure_regions = config.get("azure_region_name") or []
        if isinstance(azure_regions, str):
            azure_regions = [region for azure_region in azure_regions for region in azure_region.split(" ")]
        clean_instances_azure(tags_dict, regions=azure_regions, dry_run=dry_run)
    if cluster_backend in ('docker', ''):
        clean_resources_docker(tags_dict, dry_run=dry_run)
    return True


def clean_resources_docker(tags_dict: dict, builder_name: Optional[str] = None, dry_run: bool = False) -> None:
    assert tags_dict, "tags_dict not provided (can't clean all instances)"

    def delete_container(container):
        container.reload()
        LOGGER.info("Going to delete Docker container %s on `%s'", container, container.client.info()["Name"])
        if not dry_run:
            container.remove(v=True, force=True)
            LOGGER.debug("Done.")

    def delete_image(image):
        LOGGER.info("Going to delete Docker image tag(s) %s on `%s'", image.tags, image.client.info()["Name"])
        if not dry_run:
            image.client.images.remove(image=image.id, force=True)
            LOGGER.debug("Done.")

    resources_to_clean = list_resources_docker(tags_dict=tags_dict, builder_name=builder_name, group_as_builder=False)
    containers = resources_to_clean.get("containers", [])
    images = resources_to_clean.get("images", [])

    if not containers and not images:
        LOGGER.info("There are no resources to clean in Docker")
        return

    for container in containers:
        try:
            delete_container(container)
        except Exception:  # noqa: BLE001
            LOGGER.error("Failed to delete container %s on host `%s'", container, container.client.info()["Name"])

    for image in images:
        try:
            delete_image(image)
        except Exception:  # noqa: BLE001
            LOGGER.error("Failed to delete image tag(s) %s on host `%s'", image.tags, image.client.info()["Name"])


def clean_instances_aws(tags_dict: dict, regions=None, dry_run=False):
    """Remove all instances with specific tags in AWS."""

    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    if regions:
        aws_instances = {}
        for region in regions:
            aws_instances |= list_instances_aws(
                tags_dict=tags_dict, region_name=region, group_as_region=True)
    else:
        aws_instances = list_instances_aws(tags_dict=tags_dict, group_as_region=True)
    try:
        argus_client = get_argus_client(run_id=tags_dict.get("TestId"))
    except ArgusError as exc:
        LOGGER.warning("Unable to initialize Argus: %s", exc.message)
        argus_client = MagicMock()

    for region, instance_list in aws_instances.items():
        if not instance_list:
            LOGGER.info("There are no instances to remove in AWS region %s", region)
            continue
        client: EC2Client = boto3.client('ec2', region_name=region)
        for instance in instance_list:
            tags = aws_tags_to_dict(instance.get('Tags'))
            name = tags.get("Name", "N/A")
            node_type = tags.get("NodeType")
            instance_id = instance['InstanceId']
            if node_type and node_type == "sct-runner":
                LOGGER.info("Skipping Sct Runner instance '%s'", instance_id)
                continue
            LOGGER.info("Going to delete '{instance_id}' [name={name}] ".format(instance_id=instance_id, name=name))
            if not dry_run:
                response = client.terminate_instances(InstanceIds=[instance_id])
                terminate_resource_in_argus(client=argus_client, resource_name=name)
                LOGGER.debug("Done. Result: %s\n", response['TerminatingInstances'])


def clean_elastic_ips_aws(tags_dict, regions=None, dry_run=False):
    """
    Remove all elastic ips with specific tags AWS

    :param tags_dict: key-value pairs used for filtering
    :param regions: list of the AWS regions to consider
    :return: None
    """
    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    if regions:
        aws_instances = {}
        for region in regions:
            aws_instances |= list_elastic_ips_aws(
                tags_dict=tags_dict, region_name=region, group_as_region=True)
    else:
        aws_instances = list_elastic_ips_aws(tags_dict=tags_dict, group_as_region=True)

    for region, eip_list in aws_instances.items():
        if not eip_list:
            LOGGER.info("There are no EIPs to remove in AWS region %s", region)
            continue
        client: EC2Client = boto3.client('ec2', region_name=region)
        for eip in eip_list:
            association_id = eip.get('AssociationId')
            if association_id and not dry_run:
                response = client.disassociate_address(AssociationId=association_id)
                LOGGER.debug("disassociate_address. Result: %s\n", response)
            allocation_id = eip['AllocationId']
            LOGGER.info("Going to release '%s' [public_ip={%s}]", allocation_id, eip['PublicIp'])
            if not dry_run:
                response = client.release_address(AllocationId=allocation_id)
                LOGGER.debug("Done. Result: %s\n", response)


def clean_test_security_groups(tags_dict, regions=None, dry_run=False):
    """
    Remove all security groups with specific tags AWS

    :param tags_dict: key-value pairs used for filtering
    :param regions: list of the AWS regions to consider
    :return: None
    """
    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    if regions:
        aws_instances = {}
        for region in regions:
            aws_instances |= list_test_security_groups(
                tags_dict=tags_dict, region_name=region, group_as_region=True)
    else:
        aws_instances = list_test_security_groups(tags_dict=tags_dict, group_as_region=True)

    for region, sg_list in aws_instances.items():
        if not sg_list:
            LOGGER.info("There are no SGs to remove in AWS region %s", region)
            continue
        client: EC2Client = boto3.client('ec2', region_name=region)
        for security_group in sg_list:
            group_id = security_group.get('GroupId')
            LOGGER.info("Going to delete '%s'", group_id)
            if not dry_run:
                try:
                    response = client.delete_security_group(GroupId=group_id)
                    LOGGER.debug("Done. Result: %s\n", response)
                except Exception as ex:  # noqa: BLE001
                    LOGGER.debug("Failed with: %s", str(ex))


def clean_aws_kms_alias(tags_dict, region_names):
    # NOTE: try to delete KMS key alias which could be created by the AWS-KMS nemesis
    test_id = tags_dict.get("TestId", "TestIdNotFound")
    if any(('db' in node_type for node_type in tags_dict.get("NodeType", []))):
        AwsKms(region_names=region_names).delete_alias(
            f"alias/testid-{test_id}", tolerate_errors=True)
    else:
        LOGGER.info("Skip AWS KMS alias deletion because DB nodes deletion was not scheduled")


def clean_load_balancers_aws(tags_dict, regions=None, dry_run=False):
    """
    Remove all load balancers with specific tags AWS

    :param tags_dict: key-value pairs used for filtering
    :param regions: list of the AWS regions to consider
    :param dry_run: if True, wouldn't delete any resource
    :return: None
    """

    # don't if `post_behavior_k8s_cluster` was set to not destroy
    if "NodeType" in tags_dict and "k8s" not in tags_dict.get("NodeType"):
        return

    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    elbs_per_region = list_load_balancers_aws(tags_dict=tags_dict, regions=regions, group_as_region=True)

    for region, elb_list in elbs_per_region.items():
        if not elb_list:
            LOGGER.info("There are no ELBs to remove in AWS region %s", region)
            continue
        client = boto3.client('elb', region_name=region)
        for elb in elb_list:
            arn = elb.get('ResourceARN')
            LOGGER.info("Going to delete '%s'", arn)
            if not dry_run:
                try:
                    response = client.delete_load_balancer(LoadBalancerName=arn.split('/')[1])
                    LOGGER.debug("Done. Result: %s\n", response)
                except Exception as ex:  # noqa: BLE001
                    LOGGER.debug("Failed with: %s", str(ex))


def clean_cloudformation_stacks_aws(tags_dict, regions=None, dry_run=False):
    """
    Remove all cloudformation stacks with specific tags AWS

    :param tags_dict: key-value pairs used for filtering
    :param regions: list of the AWS regions to consider
    :param dry_run: if True, wouldn't delete any resource
    :return: None
    """

    # don't if `post_behavior_k8s_cluster` was set to not destroy
    if "NodeType" in tags_dict and "k8s" not in tags_dict.get("NodeType"):
        return

    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    stacks_per_region = list_cloudformation_stacks_aws(tags_dict=tags_dict, regions=regions, group_as_region=True)

    for region, stacks_list in stacks_per_region.items():
        if not stacks_list:
            LOGGER.info("There are no cloudformation stacks to remove in AWS region %s", region)
            continue
        client = boto3.client('cloudformation', region_name=region)
        for stack in stacks_list:
            arn = stack.get('ResourceARN')
            LOGGER.info("Going to delete '%s'", arn)
            if not dry_run:
                try:
                    response = client.delete_stack(StackName=arn.split('/')[1])
                    LOGGER.debug("Done. Result: %s\n", response)
                except Exception as ex:  # noqa: BLE001
                    LOGGER.debug("Failed with: %s", str(ex))


def clean_launch_templates_aws(tags_dict, regions=None, dry_run=False):
    """
    Remove all VM launch templates with specific tags.

    :param tags_dict: key-value pairs used for filtering
    :param regions: list of the AWS regions to consider
    :param dry_run: if True, wouldn't delete any resource
    :return: None
    """

    assert tags_dict, "Can't cleanup launch templates because 'tags_dict' was not provided."

    lts_per_region = list_launch_templates_aws(tags_dict=tags_dict, regions=regions)
    for region_name, lt_list in lts_per_region.items():
        if not lt_list:
            LOGGER.info("There are no LaunchTemplates to remove in AWS region %s", region_name)
            continue
        ec2_client = boto3.client("ec2", region_name=region_name)
        for current_lt in lt_list:
            current_lt_name = current_lt.get("LaunchTemplateName")
            current_lt_id = current_lt.get("LaunchTemplateId")
            LOGGER.info(
                "Going to delete LaunchTemplate, ID: '%s', Name: '%s'",
                current_lt_id, current_lt_name)
            if dry_run:
                continue
            try:
                if current_lt_name:
                    deletion_args = {"LaunchTemplateName": current_lt_name}
                else:
                    deletion_args = {"LaunchTemplateId": current_lt_id}
                response = ec2_client.delete_launch_template(**deletion_args)
                LOGGER.info(
                    "Successfully deleted '%s' LaunchTemplate. Response: %s\n",
                    (current_lt_name or current_lt_id), response)
            except Exception as ex:  # noqa: BLE001
                LOGGER.info("Failed to delete the '%s' LaunchTemplate: %s", deletion_args, str(ex))


def clean_instances_gce(tags_dict: dict, dry_run=False):
    """
    Remove all instances with specific tags GCE

    :param tags_dict: key-value pairs used for filtering
    :return: None
    """
    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    gce_instances_to_clean = list_instances_gce(tags_dict=tags_dict)

    if not gce_instances_to_clean:
        LOGGER.info("There are no GCE instances to remove in the %s project", os.environ.get("SCT_GCE_PROJECT"))
        return

    def delete_instance(instance_with_tags: tuple[GceInstance, dict]):
        instance, tags_dict = instance_with_tags
        LOGGER.info("Going to delete: %s (%s project)", instance.name, os.environ.get("SCT_GCE_PROJECT"))
        try:
            argus_client = get_argus_client(run_id=tags_dict.get("TestId"))
        except ArgusError as exc:
            LOGGER.warning("Unable to initialize Argus: %s", exc.message)
            argus_client = MagicMock()

        if not dry_run:
            instances_client, info = get_gce_compute_instances_client()
            res = instances_client.delete(instance=instance.name,
                                          project=info['project_id'],
                                          zone=instance.zone.split('/')[-1])
            res.done()
            terminate_resource_in_argus(client=argus_client, resource_name=instance.name)
            LOGGER.info("%s deleted=%s", instance.name, res)

    ParallelObject(map(lambda i: (i, tags_dict), gce_instances_to_clean),
                   timeout=60).run(delete_instance, ignore_exceptions=False)


def clean_instances_azure(tags_dict: dict, regions=None, dry_run=False):
    """
    Cleans instances by tags.

    :param tags_dict: key-value pairs used for filtering
    :return: None
    """
    assert tags_dict, "Running clean instances without tags would remove all SCT related resources in all regions"
    try:
        argus_client = get_argus_client(run_id=tags_dict.get("TestId"))
    except ArgusError as exc:
        LOGGER.warning("Unable to initialize Argus: %s", exc.message)
        argus_client = MagicMock()
    provisioners = AzureProvisioner.discover_regions(tags_dict.get("TestId", ""), regions=regions)
    for provisioner in provisioners:
        all_instances = provisioner.list_instances()
        instances_to_clean = []
        for instance in all_instances:
            tags = instance.tags
            for tag_k, tag_v in tags_dict.items():
                if tag_k not in tags or (tags[tag_k] not in tag_v if isinstance(tag_v, list) else tags[tag_k] != tag_v):
                    break
            else:
                instances_to_clean.append(instance)
        if len(all_instances) == len(instances_to_clean):
            LOGGER.info("Cleaning everything for test id: %s in region: %s",
                        provisioner.test_id, provisioner.region)
            if not dry_run:
                provisioner.cleanup(wait=False)
                for instance in instances_to_clean:
                    terminate_resource_in_argus(client=argus_client, resource_name=instance.name)
        else:
            LOGGER.info("test id %s from %s - instances to clean: %s",
                        provisioner.test_id, provisioner.region, [inst.name for inst in instances_to_clean])
            if not dry_run:
                for instance in instances_to_clean:
                    instance.terminate(wait=False)


def clean_clusters_gke(tags_dict: dict, dry_run: bool = False) -> None:
    if "NodeType" in tags_dict and "k8s" not in tags_dict.get("NodeType"):
        return
    assert tags_dict, "tags_dict not provided (can't clean all clusters)"
    gke_clusters_to_clean = list_clusters_gke(tags_dict=tags_dict)

    if not gke_clusters_to_clean:
        LOGGER.info("There are no GKE clusters to remove in the %s project", os.environ.get("SCT_GCE_PROJECT"))
        return

    def delete_cluster(cluster):
        if not dry_run:
            LOGGER.info("Going to delete %s GKE cluster from the %s project",
                        cluster.name, os.environ.get("SCT_GCE_PROJECT"))
            try:
                res = cluster.destroy()
                LOGGER.info("%s deleted=%s", cluster.name, res)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(exc)

    ParallelObject(gke_clusters_to_clean, timeout=180).run(delete_cluster, ignore_exceptions=True)


def clean_orphaned_gke_disks(tags_dict: dict, dry_run: bool = False) -> None:
    if "NodeType" in tags_dict and "k8s" not in tags_dict.get("NodeType"):
        return
    try:
        gke_cleaner = GkeCleaner()
        orphaned_disks = gke_cleaner.list_orphaned_gke_disks()
        LOGGER.info("Found following orphaned GKE disks in the %s project: %s",
                    os.environ.get("SCT_GCE_PROJECT"), orphaned_disks)
        if not dry_run:
            for zone, disk_names in orphaned_disks.items():
                gke_cleaner.clean_disks(disk_names=disk_names, zone=zone)
                LOGGER.info("Deleted following orphaned GKE disks in the '%s' zone (%s project): %s",
                            zone, os.environ.get("SCT_GCE_PROJECT"), disk_names)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(exc)


def clean_clusters_eks(tags_dict: dict, regions: list = None, dry_run: bool = False) -> None:
    if "NodeType" in tags_dict and "k8s" not in tags_dict.get("NodeType"):
        return
    assert tags_dict, "tags_dict not provided (can't clean all clusters)"
    eks_clusters_to_clean = list_clusters_eks(tags_dict=tags_dict, regions=regions)

    if not eks_clusters_to_clean:
        LOGGER.info("There are no EKS clusters to remove in %s region(s)" % ','.join(regions) or 'all')
        return

    def delete_cluster(cluster):
        if not dry_run:
            LOGGER.info("Going to delete '%s' EKS cluster in the '%s' region.",
                        cluster.name, cluster.region_name)
            try:
                res = cluster.destroy()
                LOGGER.info("'%s' EKS cluster in the '%s' region has been deleted. Response=%s",
                            cluster.name, cluster.region_name, res)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(exc)

    ParallelObject(eks_clusters_to_clean, timeout=180).run(delete_cluster, ignore_exceptions=True)


def clean_resources_according_post_behavior(params, config, logdir, dry_run=False):
    critical_events = get_testrun_status(params.get('TestId'), logdir, only_critical=True)
    actions_per_type = get_post_behavior_actions(config)
    LOGGER.debug(actions_per_type)

    node_types_to_cleanup = []
    for cluster_nodes_type, action_type in actions_per_type.items():
        if action_type["action"] == "keep":
            LOGGER.info("Post behavior %s for %s. Keep resources running", action_type["action"], cluster_nodes_type)
        elif action_type["action"] == "destroy":
            LOGGER.info("Post behavior %s for %s. Schedule cleanup", action_type["action"], cluster_nodes_type)
            for node_type in action_type["node_types"]:
                node_types_to_cleanup.append(node_type)
            continue
        elif action_type["action"] == "keep-on-failure" and not critical_events:
            LOGGER.info("Post behavior %s for %s. No critical events found. Schedule cleanup.",
                        action_type["action"], cluster_nodes_type)
            for node_type in action_type["node_types"]:
                node_types_to_cleanup.append(node_type)
            continue
        else:
            LOGGER.info("Post behavior %s for %s. Test run Failed. Keep resources running",
                        action_type["action"], cluster_nodes_type)
            continue
    clean_cloud_resources(params | {"NodeType": node_types_to_cleanup}, config=config, dry_run=dry_run)


@retrying(n=30, sleep_time=10, allowed_exceptions=(ClientError,))
def clean_placement_groups_aws(tags_dict: dict, regions=None, dry_run=False):
    """Remove all placement groups with specific tags in AWS."""
    tags_dict = {key: value for key, value in tags_dict.items() if key != "NodeType"}
    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    if regions:
        aws_placement_groups = {}
        for region in regions:
            aws_placement_groups |= list_placement_groups_aws(
                tags_dict=tags_dict, region_name=region, group_as_region=True)
    else:
        aws_placement_groups = list_placement_groups_aws(tags_dict=tags_dict, group_as_region=True)

    for region, instance_list in aws_placement_groups.items():
        if not instance_list:
            LOGGER.debug("There are no placement groups to remove in AWS region %s", region)
            continue
        client: EC2Client = boto3.client('ec2', region_name=region)
        for instance in instance_list:
            name = instance.get("GroupName")
            LOGGER.info("Going to delete placement group '{name} ".format(name=name))
            if not dry_run:
                try:
                    response = client.delete_placement_group(GroupName=name)
                    LOGGER.info("Placement group deleted: %s\n", response)
                except Exception as ex:  # pylint: disable=broad-except
                    LOGGER.debug("Failed to delete placement group: %s", str(ex))
                    raise
