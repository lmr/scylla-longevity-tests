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

# pylint: disable=too-many-lines, too-many-public-methods
import ipaddress
import json
import logging
import os
import random
import re
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime
from functools import cached_property
from typing import Dict, Optional, ParamSpec, TypeVar

import boto3
import botocore.exceptions
import tenacity
import yaml
from mypy_boto3_ec2 import EC2Client
from mypy_boto3_ec2.service_resource import EC2ServiceResource

from sdcm import ec2_client, cluster, wait
from sdcm.ec2_client import CreateSpotInstancesError
from sdcm.provision.aws.utils import configure_set_preserve_hostname_script
from sdcm.provision.common.utils import configure_hosts_set_hostname_script
from sdcm.provision.network_configuration import NetworkInterface, ScyllaNetworkConfiguration, is_ip_ssh_connections_ipv6, \
    network_interfaces_count, ssh_connection_ip_type
from sdcm.provision.scylla_yaml import SeedProvider
from sdcm.provision.helpers.cloud_init import wait_cloud_init_completes
from sdcm.sct_provision.aws.cluster import PlacementGroup

from sdcm.remote import LocalCmdRunner, shell_script_cmd, NETWORK_EXCEPTIONS
from sdcm.sct_events.database import DatabaseLogEvent
from sdcm.sct_events.filters import DbEventsFilter
from sdcm.sct_events.system import SpotTerminationEvent
from sdcm.utils.aws_utils import tags_as_ec2_tags, ec2_instance_wait_public_ip
from sdcm.utils.common import list_instances_aws
from sdcm.utils.decorators import retrying
from sdcm.wait import exponential_retry

LOGGER = logging.getLogger(__name__)

INSTANCE_PROVISION_ON_DEMAND = 'on_demand'
INSTANCE_PROVISION_SPOT_FLEET = 'spot_fleet'
INSTANCE_PROVISION_SPOT_LOW_PRICE = 'spot_low_price'
SPOT_CNT_LIMIT = 20
SPOT_FLEET_LIMIT = 50
SPOT_TERMINATION_CHECK_OVERHEAD = 15
LOCAL_CMD_RUNNER = LocalCmdRunner()
EBS_VOLUME = "attached"
INSTANCE_STORE = "instance_store"

SPOT_TERMINATION_CHECK_DELAY = 0

P = ParamSpec("P")  # pylint: disable=invalid-name
R = TypeVar("R")  # pylint: disable=invalid-name

# pylint: disable=too-many-lines


class AWSCluster(cluster.BaseCluster):  # pylint: disable=too-many-instance-attributes,abstract-method,

    """
    Cluster of Node objects, started on Amazon EC2.
    """

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 services, credentials, cluster_uuid=None,
                 ec2_instance_type='c6i.xlarge', ec2_ami_username='root',
                 ec2_user_data='', ec2_block_device_mappings=None,
                 cluster_prefix='cluster',
                 node_prefix='node', n_nodes=10, params=None, node_type=None,
                 extra_network_interface=False, add_nodes=True, nodes_smp=[]):
        # pylint: disable=too-many-locals
        region_names = params.region_names
        if len(credentials) > 1 or len(region_names) > 1:
            assert len(credentials) == len(region_names)
        self._ec2_ami_id = ec2_ami_id
        self._ec2_subnet_id = ec2_subnet_id
        self._ec2_security_group_ids = ec2_security_group_ids
        self._ec2_services = services
        self._credentials = credentials
        self._reuse_credentials = None
        self._ec2_instance_type = ec2_instance_type
        self._ec2_ami_username = ec2_ami_username
        if ec2_block_device_mappings is None:
            ec2_block_device_mappings = []
        self._ec2_block_device_mappings = ec2_block_device_mappings
        self._ec2_user_data = ec2_user_data
        self.region_names = region_names
        self.params = params
        self.placement_group_name = PlacementGroup(
            params=params, common_tags=self.test_config.common_tags(),
            test_id=self.test_config.test_id()).placement_group_name

        super().__init__(cluster_uuid=cluster_uuid,
                         cluster_prefix=cluster_prefix,
                         node_prefix=node_prefix,
                         n_nodes=n_nodes,
                         params=params,
                         region_names=self.region_names,
                         node_type=node_type,
                         extra_network_interface=extra_network_interface,
                         add_nodes=add_nodes,
                         nodes_smp=nodes_smp
                         )

    def __str__(self):
        return 'Cluster %s (AMI: %s Type: %s)' % (self.name,
                                                  self._ec2_ami_id,
                                                  self._ec2_instance_type)

    @property
    def instance_profile_name(self) -> str | None:
        if self.node_type == "scylla-db":
            return self.params.get('aws_instance_profile_name_db')
        if self.node_type == "loader":
            return self.params.get('aws_instance_profile_name_loader')
        return None

    def _create_on_demand_instances(self, count, interfaces, ec2_user_data, dc_idx=0, instance_type=None, is_zero_node=False):  # pylint: disable=too-many-arguments
        ami_id = self._ec2_ami_id[dc_idx]
        self.log.debug(f"Creating {count} on-demand instances using AMI id '{ami_id}'... ")
        params = dict(ImageId=ami_id,
                      UserData=ec2_user_data,
                      MinCount=count,
                      MaxCount=count,
                      KeyName=self._credentials[dc_idx].key_pair_name,
                      BlockDeviceMappings=self._ec2_block_device_mappings,
                      NetworkInterfaces=interfaces,
                      InstanceType=instance_type or self._ec2_instance_type)
        instance_profile = self.instance_profile_name
        if instance_profile:
            params['IamInstanceProfile'] = {'Name': instance_profile}
        ec2 = ec2_client.EC2ClientWrapper(region_name=self.region_names[dc_idx])
        subnet_info = ec2.get_subnet_info(interfaces[0]["SubnetId"])
        region_name_with_az = subnet_info['AvailabilityZone']
        ec2.add_placement_group_name_param(params, self.placement_group_name)
        if self.params.get('use_capacity_reservation'):
            ec2.add_capacity_reservation_param(params, region_name_with_az[-1])
        LOGGER.debug('Sending an On-Demand request with params: %s', params)
        LOGGER.debug('Using EC2 service with DC-index: %s, (associated with region: %s)',
                     dc_idx, self.region_names[dc_idx])
        LOGGER.debug("Using Availability Zone of: %s", region_name_with_az)

        instances = self._ec2_services[dc_idx].create_instances(**params)

        ec2.add_tags(instances, {'Name': 'on_demand'})
        if is_zero_node:
            ec2.add_tags(instances, {"ZeroTokenNode": "True"})

        self.log.debug("Created instances: %s." % instances)
        return instances

    def _create_spot_instances(self, count, interfaces, ec2_user_data='', dc_idx=0, instance_type=None, is_zero_node=False):  # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        ec2 = ec2_client.EC2ClientWrapper(region_name=self.region_names[dc_idx],
                                          spot_max_price_percentage=self.params.get('spot_max_price'))
        subnet_info = ec2.get_subnet_info(interfaces[0]["SubnetId"])
        spot_params = dict(instance_type=instance_type or self._ec2_instance_type,
                           image_id=self._ec2_ami_id[dc_idx],
                           region_name=subnet_info['AvailabilityZone'],
                           network_if=interfaces,
                           key_pair=self._credentials[dc_idx].key_pair_name,
                           user_data=ec2_user_data,
                           count=count,
                           block_device_mappings=self._ec2_block_device_mappings,
                           aws_instance_profile=self.instance_profile_name,
                           placement_group_name=self.placement_group_name,)

        limit = SPOT_FLEET_LIMIT if self.instance_provision == INSTANCE_PROVISION_SPOT_FLEET else SPOT_CNT_LIMIT
        request_cnt = 1
        tail_cnt = 0
        if count > limit:
            request_cnt = count // limit
            spot_params['count'] = limit
            tail_cnt = count % limit
            if tail_cnt:
                request_cnt += 1
        instances = []
        for i in range(request_cnt):
            if tail_cnt and i == request_cnt - 1:
                spot_params['count'] = tail_cnt

            if self.instance_provision == INSTANCE_PROVISION_SPOT_FLEET and count > 1:
                instances_i = ec2.create_spot_fleet(**spot_params)
            else:
                instances_i = ec2.create_spot_instances(**spot_params)
            instances.extend(instances_i)

        if is_zero_node:
            ec2.add_tags(instances, {"ZeroTokenNode": "True"})

        return instances

    def _create_instances(self, count, ec2_user_data='', dc_idx=0, az_idx=0, instance_type=None, is_zero_node=False):  # pylint: disable=too-many-arguments
        if not count:  # EC2 API fails if we request zero instances.
            return []

        if not ec2_user_data:
            ec2_user_data = self._ec2_user_data
        self.log.debug("Passing user_data '%s' to create_instances", ec2_user_data)
        interfaces_amount = network_interfaces_count(self.params)
        interfaces = []
        for i in range(interfaces_amount):
            LOGGER.debug("SubnetId: %s, dc_idx: %s, az index: %s, interface index: %s",
                         self._ec2_subnet_id, dc_idx, az_idx, i)
            interfaces.append({'DeviceIndex': i,
                               'SubnetId': self._ec2_subnet_id[dc_idx][az_idx][i],
                               'Groups': self._ec2_security_group_ids[dc_idx]})
        self.log.debug("interfaces: '%s' - to create_instances", interfaces)

        # Can only be associated with a single network interface only with the device index of 0:
        # https://docs.aws.amazon.com/sdkfornet1/latest/apidocs/html/P_Amazon_EC2_Model_InstanceNetworkInterfaceSpecification_
        # AssociatePublicIpAddress.htm
        if interfaces_amount == 1:
            interfaces[0].update({'AssociatePublicIpAddress': True})

        self.log.info(f"Create {self.instance_provision} instance(s)")
        if self.instance_provision == 'mixed':
            instances = self._create_mixed_instances(count, interfaces, ec2_user_data, dc_idx,
                                                     instance_type=instance_type, is_zero_node=is_zero_node)
        elif self.instance_provision == INSTANCE_PROVISION_ON_DEMAND:
            instances = self._create_on_demand_instances(count, interfaces, ec2_user_data, dc_idx,
                                                         instance_type=instance_type, is_zero_node=is_zero_node)
        elif self.instance_provision == INSTANCE_PROVISION_SPOT_FLEET and count > 1:
            instances = self._create_spot_instances(count, interfaces, ec2_user_data, dc_idx,
                                                    instance_type=instance_type, is_zero_node=is_zero_node)
        else:
            instances = self.fallback_provision_type(count, interfaces, ec2_user_data, dc_idx,
                                                     instance_type=instance_type, is_zero_node=is_zero_node)

        return instances

    def fallback_provision_type(self, count, interfaces, ec2_user_data, dc_idx, instance_type=None, is_zero_node=False):  # pylint: disable=too-many-arguments
        instances = None

        if self.instance_provision.lower() == 'spot' or (self.instance_provision == INSTANCE_PROVISION_SPOT_FLEET and count == 1):
            instances_provision_fallbacks = [INSTANCE_PROVISION_SPOT_LOW_PRICE]
        else:
            # If self.instance_provision == "spot_low_price"
            instances_provision_fallbacks = [self.instance_provision]

        if self.params.get('instance_provision_fallback_on_demand'):
            instances_provision_fallbacks.append(INSTANCE_PROVISION_ON_DEMAND)

        self.log.debug(f"Instances provision fallbacks : {instances_provision_fallbacks}")

        for instances_provision_type in instances_provision_fallbacks:
            try:
                self.log.info(f"Create {instances_provision_type} instance(s)")
                if instances_provision_type == INSTANCE_PROVISION_ON_DEMAND:
                    instances = self._create_on_demand_instances(
                        count, interfaces, ec2_user_data, dc_idx, instance_type=instance_type, is_zero_node=is_zero_node)
                else:
                    instances = self._create_spot_instances(
                        count, interfaces, ec2_user_data, dc_idx, instance_type=instance_type, is_zero_node=is_zero_node)
                break
            except (CreateSpotInstancesError, botocore.exceptions.ClientError) as cl_ex:
                if instances_provision_type == instances_provision_fallbacks[-1]:
                    raise
                if not self.check_spot_error(str(cl_ex), instances_provision_type):
                    raise

        return instances

    def check_spot_error(self, cl_ex, instance_provision):
        if ec2_client.MAX_SPOT_EXCEEDED_ERROR in cl_ex or \
                ec2_client.FLEET_LIMIT_EXCEEDED_ERROR in cl_ex or \
                ec2_client.SPOT_CAPACITY_NOT_AVAILABLE_ERROR in cl_ex or \
                ec2_client.SPOT_PRICE_TOO_LOW in cl_ex or \
                ec2_client.SPOT_STATUS_UNEXPECTED_ERROR in cl_ex:
            self.log.error(f"Cannot create {instance_provision} instance(s): {cl_ex}")
            return True
        return False

    def _create_mixed_instances(self, count, interfaces, ec2_user_data, dc_idx, instance_type=None, is_zero_node=False):  # pylint: disable=too-many-arguments
        instances = []
        max_num_on_demand = 2
        if isinstance(self, (ScyllaAWSCluster, CassandraAWSCluster)):
            if count > 2:
                count_on_demand = max_num_on_demand
            elif count == 2:
                count_on_demand = 1
            else:
                count_on_demand = 0

            if self.nodes:
                num_of_on_demand = len([node for node in self.nodes if not node.is_spot])
                if num_of_on_demand < max_num_on_demand:
                    count_on_demand = max_num_on_demand - num_of_on_demand
                else:
                    count_on_demand = 0

            count_spot = count - count_on_demand

            if count_spot > 0:
                self.instance_provision = INSTANCE_PROVISION_SPOT_LOW_PRICE
                instances.extend(self._create_spot_instances(
                    count_spot, interfaces, ec2_user_data, dc_idx, instance_type=instance_type, is_zero_node=is_zero_node))
            if count_on_demand > 0:
                self.instance_provision = INSTANCE_PROVISION_ON_DEMAND
                instances.extend(self._create_on_demand_instances(
                    count_on_demand, interfaces, ec2_user_data, dc_idx, instance_type=instance_type, is_zero_node=is_zero_node))
            self.instance_provision = 'mixed'
        elif isinstance(self, LoaderSetAWS):
            self.instance_provision = INSTANCE_PROVISION_SPOT_LOW_PRICE
            instances = self._create_spot_instances(
                count, interfaces, ec2_user_data, dc_idx, instance_type=instance_type)
        elif isinstance(self, MonitorSetAWS):
            self.instance_provision = INSTANCE_PROVISION_ON_DEMAND
            instances.extend(self._create_on_demand_instances(
                count, interfaces, ec2_user_data, dc_idx, instance_type=instance_type))
        else:
            raise Exception('Unsuported type of cluster type %s' % self)
        return instances

    def _get_instances(self, dc_idx, az_idx=None):
        test_id = self.test_config.test_id()
        if not test_id:
            raise ValueError("test_id should be configured for using reuse_cluster")
        availability_zone = self.params.get('availability_zone').split(",")[az_idx] if az_idx is not None else None
        ec2 = ec2_client.EC2ClientWrapper(region_name=self.region_names[dc_idx],
                                          spot_max_price_percentage=self.params.get('spot_max_price'))
        results = list_instances_aws(tags_dict={'TestId': test_id, 'NodeType': self.node_type},
                                     region_name=self.region_names[dc_idx], group_as_region=True, availability_zone=availability_zone)
        instances = results[self.region_names[dc_idx]]

        def sort_by_index(item):
            for tag in item['Tags']:
                if tag['Key'] == 'NodeIndex':
                    return tag['Value']
            return '0'
        instances = sorted(instances, key=sort_by_index)
        return [ec2.get_instance(instance['InstanceId']) for instance in instances]

    @staticmethod
    def update_bootstrap(ec2_user_data, enable_auto_bootstrap):
        """
        Update --bootstrap argument inside ec2_user_data string.
        """
        if isinstance(ec2_user_data, dict):
            ec2_user_data['scylla_yaml']['auto_bootstrap'] = enable_auto_bootstrap
            return ec2_user_data

        if enable_auto_bootstrap:
            if '--bootstrap ' in ec2_user_data:
                ec2_user_data.replace('--bootstrap false', '--bootstrap true')
            else:
                ec2_user_data += ' --bootstrap true '
        else:  # noqa: PLR5501
            if '--bootstrap ' in ec2_user_data:
                ec2_user_data.replace('--bootstrap true', '--bootstrap false')
            else:
                ec2_user_data += ' --bootstrap false '
        return ec2_user_data

    def _create_or_find_instances(self, count, ec2_user_data, dc_idx, az_idx=0, instance_type=None, is_zero_node=False):  # pylint: disable=too-many-arguments
        nodes = [node for node in self.nodes if node.dc_idx == dc_idx and node.rack == az_idx]
        if nodes:
            return self._create_instances(count, ec2_user_data, dc_idx, az_idx, instance_type=instance_type, is_zero_node=is_zero_node)
        if self.test_config.REUSE_CLUSTER:
            instances = self._get_instances(dc_idx, az_idx)
            assert len(instances) == count, f"Found {len(instances)} instances, while expect {count}"
            self.log.info('Found instances to be reused from test [%s] = %s', self.test_config.REUSE_CLUSTER, instances)
            return instances
        if instances := self._get_instances(dc_idx, az_idx):
            self.log.info('Found provisioned instances = %s', instances)
            return instances
        self.log.info('Found no provisioned instances. Provision them.')
        return self._create_instances(count, ec2_user_data, dc_idx, az_idx, instance_type=instance_type, is_zero_node=is_zero_node)

    # pylint: disable=too-many-arguments
    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False, instance_type=None, is_zero_node=False):
        if not count:
            return []
        ec2_user_data = self.prepare_user_data(enable_auto_bootstrap=enable_auto_bootstrap)
        # NOTE: if simulated regions/dcs then create all instances in the same region
        instance_dc = 0 if self.params.get("simulated_regions") else dc_idx
        # if simulated_racks, create all instances in the same az
        instance_az = 0 if self.params.get("simulated_racks") else rack
        instances = []
        if instance_az is None:
            # define how many nodes should be created on each rack, e.g. for 3 nodes and 2 racks it will be [2, 1]
            base = count // self.racks_count
            extra = count % self.racks_count
            rack_distribution = [base + 1 if i < extra else base for i in range(self.racks_count)]
        else:
            # otherwise create all nodes on the specified rack
            rack_distribution = [count if i == instance_az else 0 for i in range(self.racks_count)]
        self.log.info('rack distribution: %s', rack_distribution)
        for rack_idx, rack_count in enumerate(rack_distribution):
            if rack_count == 0:
                continue  # when provisioning fewer nodes than racks
            instances.extend(self._create_or_find_instances(
                count=rack_count, ec2_user_data=ec2_user_data, dc_idx=instance_dc, az_idx=rack_idx, instance_type=instance_type, is_zero_node=is_zero_node))
        for node_index, instance in enumerate(instances):
            self._node_index += 1
            # in case rack is not specified, spread nodes to different racks
            node_rack = node_index % self.racks_count if rack is None else rack
            node = self._create_node(instance, self._ec2_ami_username, self.node_prefix,
                                     self._node_index, self.logdir, dc_idx=dc_idx, rack=node_rack)
            node.enable_auto_bootstrap = enable_auto_bootstrap
            if ssh_connection_ip_type(self.params) == 'ipv6' and not node.distro.is_ubuntu:
                node.config_ipv6_as_persistent()

            self.nodes.append(node)
        self.write_node_public_ip_file()
        self.write_node_private_ip_file()
        return self.nodes[-count:]

    def _create_node(self, instance, ami_username, node_prefix, node_index,  # pylint: disable=too-many-arguments
                     base_logdir, dc_idx, rack):
        ec2_service = self._ec2_services[0 if self.params.get("simulated_regions") else dc_idx]
        credentials = self._credentials[0 if self.params.get("simulated_regions") else dc_idx]
        node = AWSNode(ec2_instance=instance, ec2_service=ec2_service,
                       credentials=credentials, parent_cluster=self, ami_username=ami_username,
                       node_prefix=node_prefix, node_index=node_index,
                       base_logdir=base_logdir, dc_idx=dc_idx, rack=rack, shard_num=self.get_node_shard_num(self, node_index))
        node.init()
        return node


class AWSNode(cluster.BaseNode):
    """
    Wraps EC2.Instance, so that we can also control the instance through SSH.
    """

    log = LOGGER

    def __init__(self, ec2_instance, ec2_service, credentials, parent_cluster,  # pylint: disable=too-many-arguments
                 node_prefix='node', node_index=1, ami_username='root',
                 base_logdir=None, dc_idx=0, rack=0, shard_num=None):
        self.node_index = node_index
        self._instance = ec2_instance
        self._ec2_service: EC2ServiceResource = ec2_service
        self.eip_allocation_id = None
        self._aws_metadata_token = {"token": "", "expires": 0}
        ssh_login_info = {'hostname': None,
                          'user': ami_username,
                          'key_file': credentials.key_file}
        super().__init__(name=f"{node_prefix}-{self.node_index}",
                         parent_cluster=parent_cluster,
                         ssh_login_info=ssh_login_info,
                         base_logdir=base_logdir,
                         node_prefix=node_prefix,
                         dc_idx=dc_idx, rack=rack, shard_num=shard_num)

    def __str__(self):
        # If multiple network interface is defined on the node, private address in the `nodetool status` is IP that defined in
        # broadcast_address. Keep this output in correlation with `nodetool status`
        if self.scylla_network_configuration.broadcast_address_ip_type == "private":
            node_private_ip = self.scylla_network_configuration.broadcast_address
        else:
            node_private_ip = self.private_ip_address

        return 'Node %s [%s | %s%s]%s' % (
            self.name,
            self.public_ip_address,
            node_private_ip,
            " | %s" % self.ipv6_ip_address if self.test_config.IP_SSH_CONNECTIONS == "ipv6" else "",
            self._dc_info_str())

    @property
    def network_configuration(self):
        # Output example:
        #   0a:7b:18:de:f9:71: eth0
        #   0a:7b:18:de:f9:72: eth1
        ip_link_cmd = """ip -o link | awk '$2 != "lo:" {gsub(/:/,"",$2);print $17": " $2}'"""
        network_config = self.remoter.run(ip_link_cmd).stdout.strip()
        network_devices = yaml.safe_load(network_config)
        self.log.debug("Node %s ethernets: %s", self.name, network_devices)
        return network_devices

    @property
    def network_interfaces(self):
        interfaces_tmp = []
        device_indexes = []
        devices = {}
        if self.remoter:
            devices = self.network_configuration
            self.log.debug("Node %s devices: %s", self.name, devices)

        for interface in self._instance.network_interfaces:
            private_ip_addresses = [private_address["PrivateIpAddress"]
                                    for private_address in interface.private_ip_addresses]
            # make sure we use ipv6 long format (some tools remove leading zeros)
            ipv6_addresses = [ipaddress.ip_address(
                ipv6_address['Ipv6Address']).exploded for ipv6_address in interface.ipv6_addresses]
            device_indexes.append(interface.attachment['DeviceIndex'])
            ipv4_public_address = interface.association_attribute['PublicIp'] if interface.association_attribute else None
            dns_public_name = interface.association_attribute['PublicDnsName'] if interface.association_attribute else None
            interfaces_tmp.append(NetworkInterface(ipv4_public_address=ipv4_public_address,
                                                   ipv6_public_addresses=ipv6_addresses,
                                                   ipv4_private_addresses=private_ip_addresses,
                                                   ipv6_private_address='',
                                                   dns_private_name=interface.private_dns_name,
                                                   dns_public_name=dns_public_name,
                                                   device_index=interface.attachment['DeviceIndex'],
                                                   device_name=devices[interface.mac_address] if devices else '',
                                                   mac_address=interface.mac_address,
                                                   )
                                  )
        # Order interfaces by device_index (set primary interface first)
        interfaces = sorted(interfaces_tmp, key=lambda d: d.device_index)
        self.log.debug("Sorted interfaces: %s", interfaces)
        return interfaces

    def init(self):
        LOGGER.debug("Waiting until instance {0._instance} starts running...".format(self))
        self._instance_wait_safe(self._instance.wait_until_running)

        if not self.test_config.REUSE_CLUSTER:
            resources_to_tag = [self._instance.id, ]
            if len(self._instance.network_interfaces) == 2:
                # first we need to configure the both networks so we'll have public ip
                self.allocate_and_attach_elastic_ip(self.parent_cluster, self.dc_idx)
                resources_to_tag.append(self.eip_allocation_id)
            self._ec2_service.create_tags(Resources=resources_to_tag, Tags=tags_as_ec2_tags(self.tags))

        self._wait_public_ip()
        self.scylla_network_configuration = ScyllaNetworkConfiguration(
            network_interfaces=self.network_interfaces,
            scylla_network_config=self.parent_cluster.params["scylla_network_config"])
        # TODO: keep next two for debug purpose
        self.log.debug("Node %s scylla_network_config: %s", self.name,
                       self.parent_cluster.params["scylla_network_config"])
        self.log.debug("Node %s network_interfaces: %s", self.name,
                       self.scylla_network_configuration.network_interfaces)
        super().init()
        # Refresh network interfaces info after node remoter init
        self.refresh_network_interfaces_info()
        for tag in self._instance.tags:
            if tag['Key'] == 'ZeroTokenNode' and tag['Value'] == "True":
                self._is_zero_token_node = True

    def wait_for_cloud_init(self):
        if self.remoter.sudo("bash -c 'command -v cloud-init'", ignore_status=True).ok:
            wait_cloud_init_completes(self.remoter, self)

    @property
    def short_hostname(self):
        """
        Don't need to get hostname from the system since we already know what it should be
        """
        return self.name

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**super().tags,
                "NodeIndex": str(self.node_index), }

    def _set_keep_alive(self):
        self._ec2_service.create_tags(Resources=[self._instance.id], Tags=[{"Key": "keep", "Value": "alive"}])
        return super()._set_keep_alive()

    @property
    def vm_region(self):
        return self._ec2_service.meta.client.meta.region_name

    def _set_hostname(self) -> bool:
        return self.remoter.sudo(f"hostnamectl set-hostname --static {self.name}").ok

    @retrying(n=3, sleep_time=5, allowed_exceptions=NETWORK_EXCEPTIONS, message="Retrying set_hostname")
    def set_hostname(self):
        self.log.debug('Changing hostname to %s', self.name)
        # Using https://aws.amazon.com/premiumsupport/knowledge-center/linux-static-hostname-rhel7-centos7/
        # FIXME: workaround to avoid host rename generating errors on other commands
        if self.distro.is_debian:
            return
        if wait.wait_for(func=self._set_hostname, step=10, text='Retrying set hostname on the node', timeout=300):
            self.log.debug('Hostname has been changed successfully. Apply')
            script = configure_hosts_set_hostname_script(self.name) + configure_set_preserve_hostname_script()
            self.remoter.sudo(f"bash -cxe '{script}'")
        else:
            self.log.warning('Hostname has not been changed. Continue with old name')
        self.log.debug('Continue node %s set up', self.name)

    @property
    def is_spot(self):
        return bool(self._instance.instance_lifecycle and 'spot' in self._instance.instance_lifecycle.lower())

    def query_aws_metadata(self, path):
        """Queries aws metadata endpoint with token (IMDSv2 which is set as mandatory for some instances and optional in others).
        Token is cached for 6h. If token is expired, it is refreshed."""
        if self._aws_metadata_token["expires"] < time.time():
            self._aws_metadata_token["token"] = self.remoter.run(
                'curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"'
            ).stdout.strip()
            self._aws_metadata_token["expires"] = round(time.time()) + 21500
        token = self._aws_metadata_token["token"]
        return self.remoter.run(f'curl -s -H "X-aws-ec2-metadata-token: {token}" http://169.254.169.254/{path}', verbose=False).stdout.strip()

    def check_spot_termination(self):
        try:
            self.wait_ssh_up(verbose=False)
            status = self.query_aws_metadata("latest/meta-data/spot/instance-action")

            if '404 - Not Found' in status:
                return SPOT_TERMINATION_CHECK_DELAY

            if status.strip():
                self.log.warning('Got spot termination notification from AWS %s', status)
                terminate_action = json.loads(status)
                terminate_action_timestamp = time.mktime(datetime.strptime(
                    terminate_action['time'], "%Y-%m-%dT%H:%M:%SZ").timetuple())
                next_check_delay = terminate_action['time-left'] = terminate_action_timestamp - time.time()
                SpotTerminationEvent(node=self, message=terminate_action).publish()

                return max(next_check_delay - SPOT_TERMINATION_CHECK_OVERHEAD, 0)

        except Exception as details:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.warning('Error during getting spot termination notification %s', details)

        return SPOT_TERMINATION_CHECK_DELAY

    @property
    def is_data_device_lost_after_reboot(self) -> bool:
        if self.parent_cluster.params.get("data_volume_disk_num") > 0:
            return False
        result = self._ec2_service.meta.client.describe_instance_types(InstanceTypes=[self._instance.instance_type])
        if instance_types := result.get('InstanceTypes'):
            return instance_types[0].get('InstanceStorageSupported', False)
        return False

    @property
    def external_address(self):
        """
        the communication address for usage between the test and the nodes
        :return:
        """
        self.log.debug("external_address is: %s", self.scylla_network_configuration.test_communication)
        return self.scylla_network_configuration.test_communication

    @cached_property
    def public_dns_name(self) -> str:
        return self.query_aws_metadata("latest/meta-data/public-hostname")

    @cached_property
    def private_dns_name(self) -> str:
        return self.scylla_network_configuration.dns_private_name

    def _get_ipv6_ip_address(self) -> Optional[str]:
        return self.scylla_network_configuration.interface_ipv6_address

    def refresh_network_interfaces_info(self):
        self.scylla_network_configuration.network_interfaces = self.network_interfaces

    def _refresh_instance_state(self):
        self._wait_public_ip()
        self.refresh_network_interfaces_info()
        public_ipv4_addresses = [interface.ipv4_public_address for interface in self.scylla_network_configuration.network_interfaces
                                 if interface.ipv4_public_address]
        # AWS allows to have a few private IPv4 addresses per interface. For
        # now the first one address is picking, until we have anything else defined/implemented
        private_ipv4_addresses = [
            interface.ipv4_private_addresses[0] for interface in self.scylla_network_configuration.network_interfaces]
        return public_ipv4_addresses, private_ipv4_addresses

    def allocate_and_attach_elastic_ip(self, parent_cluster, dc_idx):
        for interface in self._instance.network_interfaces:
            if interface.attachment['DeviceIndex'] == 0 and interface.association_attribute is None:
                # create and attach EIP
                client: EC2Client = boto3.client('ec2', region_name=parent_cluster.region_names[dc_idx])
                response = client.allocate_address(Domain='vpc')

                self.eip_allocation_id = response['AllocationId']
                client.associate_address(
                    AllocationId=self.eip_allocation_id,
                    NetworkInterfaceId=interface.id,
                )
                break

    def _instance_wait_safe(self, instance_method: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return exponential_retry(func=lambda: instance_method(*args, **kwargs), logger=None)
        except tenacity.RetryError:
            try:
                self._instance.reload()
            except Exception as ex:  # pylint: disable=broad-except
                LOGGER.exception("Error while reloading instance metadata: %s", ex)
            finally:
                LOGGER.debug(self._instance.meta.data)
                raise cluster.NodeError(
                    f"Timeout while running '{instance_method.__name__}' method on AWS instance '{self._instance.id}'"
                ) from None

    def _wait_public_ip(self):
        ec2_instance_wait_public_ip(self._instance)

    @cached_property
    def cql_address(self):
        address = self.scylla_network_configuration.test_communication \
            if self.test_config.IP_SSH_CONNECTIONS == 'public' else self.scylla_network_configuration.broadcast_rpc_address
        self.log.debug("cql_address is: %s", address)
        return address

    @property
    def ip_address(self):
        self.log.debug("ip_address is: %s", self.scylla_network_configuration.broadcast_address)
        return self.scylla_network_configuration.broadcast_address

    def config_ipv6_as_persistent(self):
        if self.distro.is_ubuntu:
            self.remoter.sudo("sh -c \"echo 'iface eth0 inet6 auto' >> /etc/network/interfaces\"")
            LOGGER.debug('adding ipv6 autogenerated config to /etc/network/interfaces')
        else:
            base_meta_url = "latest/meta-data/network/interfaces/macs/"
            mac = self.query_aws_metadata(base_meta_url)
            ipv6_cidr = self.query_aws_metadata(f"{base_meta_url}{mac}/subnet-ipv6-cidr-blocks")
            self.remoter.run(
                f"sudo sh -c  \"echo 'sudo ip route add {ipv6_cidr} dev eth0' >> /etc/sysconfig/network-scripts/init.ipv6-global\"")

            res = self.remoter.run(f"grep '{ipv6_cidr}' /etc/sysconfig/network-scripts/init.ipv6-global")
            LOGGER.debug('init.ipv6-global was {}updated'.format('' if res.stdout.strip else 'NOT '))

    def restart(self):
        # We differentiate between "Restart" and "Reboot".
        # Restart in AWS will be a Stop and Start of an instance.
        # When using storage optimized instances like i2 or i3, the data on disk is deleted upon STOP.  Therefore, we
        # need to setup the instance and treat it as a new instance.
        if self._instance.spot_instance_request_id:
            LOGGER.debug("target node is spot instance, impossible to stop this instance, skipping the restart")
            return

        if self.is_seed:
            # Due to https://github.com/scylladb/scylla/issues/7588, when we restart a node that is defined as "seed",
            # we must state a different, living node as the seed provider in the scylla yaml of the restarted node
            other_nodes = list(set(self.parent_cluster.nodes) - {self})
            free_nodes = [node for node in other_nodes if not node.running_nemesis]
            random_node = random.choice(free_nodes)
            seed_provider = SeedProvider(
                class_name="org.apache.cassandra.locator.SimpleSeedProvider",
                parameters=[{"seeds": f"{random_node.ip_address}"}]
            )

            with self.remote_scylla_yaml() as scylla_yml:
                scylla_yml.seed_provider = [seed_provider]

        with ExitStack() as stack:
            if self.is_data_device_lost_after_reboot:
                # There is no disk yet, lots of the errors here are acceptable, and we'll ignore them.
                for db_filter in (DbEventsFilter(db_event=DatabaseLogEvent.DATABASE_ERROR, node=self),
                                  DbEventsFilter(db_event=DatabaseLogEvent.SCHEMA_FAILURE, node=self),
                                  DbEventsFilter(db_event=DatabaseLogEvent.NO_SPACE_ERROR, node=self),
                                  DbEventsFilter(db_event=DatabaseLogEvent.FILESYSTEM_ERROR, node=self),
                                  DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR, node=self), ):
                    stack.enter_context(db_filter)

                if self.is_replacement_by_host_id_supported:
                    replace_option_name = "replace_node_first_boot"
                    replace_option_value = self.host_id
                else:
                    replace_option_name = "replace_address_first_boot"
                    replace_option_value = self.ip_address
                self.remoter.sudo(shell_script_cmd(f"""\
                    sed -e '/.*scylla/s/^/#/g' -i /etc/fstab
                    sed -e '/auto_bootstrap:.*/s/false/true/g' -i /etc/scylla/scylla.yaml
                    if ! grep ^{replace_option_name}: /etc/scylla/scylla.yaml; then
                        echo '{replace_option_name}: {replace_option_value}' | \
                            tee --append /etc/scylla/scylla.yaml
                    fi
                """))

            self._instance.stop()
            self._instance_wait_safe(self._instance.wait_until_stopped)
            self._instance.start()
            self._instance_wait_safe(self._instance.wait_until_running)
            self._wait_public_ip()

            self.log.debug("Got a new public IP: %s", self._instance.public_ip_address)

            self.refresh_ip_address()
            self.wait_ssh_up()

            if self.is_data_device_lost_after_reboot:
                self.stop_scylla_server(verify_down=False)

                # Moving var-lib-scylla.mount away, since scylla_create_devices fails if it already exists
                mount_path = "/etc/systemd/system/var-lib-scylla.mount"
                if self.remoter.sudo(f"test -e {mount_path}", ignore_status=True).ok:
                    self.remoter.sudo(f"mv {mount_path} /tmp/")

                # The scylla_create_devices has been moved to the '/opt/scylladb' folder in the master branch.
                for create_devices_file in ("/usr/lib/scylla/scylla-ami/scylla_create_devices",
                                            "/opt/scylladb/scylla-ami/scylla_create_devices",
                                            "/opt/scylladb/scylla-machine-image/scylla_create_devices", ):
                    if self.remoter.sudo(f"test -x {create_devices_file}", ignore_status=True).ok:
                        self.remoter.sudo(create_devices_file)
                        break
                else:
                    raise IOError("scylla_create_devices file isn't found")

                self.start_scylla_server(verify_up=False)

                self.remoter.sudo(shell_script_cmd("""\
                    sed -e '/auto_bootstrap:.*/s/true/false/g' -i /etc/scylla/scylla.yaml
                    sed -e 's/^replace_address_first_boot:/# replace_address_first_boot:/g' -i /etc/scylla/scylla.yaml
                    sed -e 's/^replace_node_first_boot:/# replace_node_first_boot:/g' -i /etc/scylla/scylla.yaml
                """))

    def hard_reboot(self):
        self._instance_wait_safe(self._instance.reboot)

    def release_address(self):
        self._instance.wait_until_terminated()

        client: EC2Client = boto3.client('ec2', region_name=self.parent_cluster.region_names[self.dc_idx])
        client.release_address(AllocationId=self.eip_allocation_id)

    def destroy(self):
        self.stop_task_threads()
        self.wait_till_tasks_threads_are_stopped()
        self._instance.terminate()
        if self.eip_allocation_id:
            self.release_address()
        super().destroy()

    def get_console_output(self):
        """Get instance console Output

        Get console output of instance which is printed during initiating and loading
        Get only last 64KB of output data.
        """
        result = self._ec2_service.meta.client.get_console_output(
            InstanceId=self._instance.id,
        )
        console_output = result.get('Output', '')

        if not console_output:
            self.log.warning('Some error during getting console output')
        return console_output

    def get_console_screenshot(self):
        result = self._ec2_service.meta.client.get_console_screenshot(
            InstanceId=self._instance.id
        )
        imagedata = result.get('ImageData', '')

        if not imagedata:
            self.log.warning('Some error during getting console screenshot')
        return imagedata.encode('ascii')

    def traffic_control(self, tcconfig_params=None):
        """
        run tcconfig locally to create tc commands, and run them on the node
        :param tcconfig_params: commandline arguments for tcset, if None will call tcdel
        :return: None
        """

        self.remoter.run("sudo modprobe sch_netem")

        if tcconfig_params is None:
            tc_command = LOCAL_CMD_RUNNER.run("tcdel eth1 --tc-command", ignore_status=True).stdout
            self.remoter.run('sudo bash -cxe "%s"' % tc_command, ignore_status=True)
        else:
            tc_command = LOCAL_CMD_RUNNER.run("tcset eth1 {} --tc-command".format(tcconfig_params)).stdout
            self.remoter.run('sudo bash -cxe "%s"' % tc_command)

    def install_traffic_control(self):
        return self.remoter.run("/sbin/tc -h", ignore_status=True).ok

    @property
    def image(self):
        return self._instance.image_id

    @property
    def ena_support(self) -> bool:
        return self._instance.ena_support


class ScyllaAWSCluster(cluster.BaseScyllaCluster, AWSCluster):

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 services, credentials, ec2_instance_type='c6i.xlarge',
                 ec2_ami_username='centos',
                 ec2_block_device_mappings=None,
                 user_prefix=None,
                 n_nodes=3,
                 params=None,
                 node_type: str = 'scylla-db',
                 nodes_smp=[]
                 ):
        # pylint: disable=too-many-locals
        # We have to pass the cluster name in advance in user_data
        cluster_uuid = self.test_config.test_id()
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'db-cluster')
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'db-node')

        super().__init__(
            ec2_ami_id=ec2_ami_id,
            ec2_subnet_id=ec2_subnet_id,
            ec2_security_group_ids=ec2_security_group_ids,
            ec2_instance_type=ec2_instance_type,
            ec2_ami_username=ec2_ami_username,
            ec2_block_device_mappings=ec2_block_device_mappings,
            cluster_uuid=cluster_uuid,
            services=services,
            credentials=credentials,
            cluster_prefix=cluster_prefix,
            node_prefix=node_prefix,
            n_nodes=n_nodes,
            params=params,
            node_type=node_type,
            extra_network_interface=network_interfaces_count(params) > 1,
            nodes_smp=nodes_smp)
        self.version = '2.1'

    # pylint: disable=too-many-arguments
    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False, instance_type=None, is_zero_node=False):
        if not ec2_user_data:
            if self._ec2_user_data and isinstance(self._ec2_user_data, str):
                ec2_user_data = re.sub(r'(--totalnodes\s)(\d*)(\s)',
                                       r'\g<1>{}\g<3>'.format(len(self.nodes) + count), self._ec2_user_data)
            elif self._ec2_user_data and isinstance(self._ec2_user_data, dict):
                ec2_user_data = self._ec2_user_data
            else:
                ec2_user_data = ('--clustername %s --totalnodes %s ' % (self.name, count))
        if self.nodes and isinstance(ec2_user_data, str):
            node_ips = [node.ip_address for node in self.nodes if node.is_seed]
            seeds = ",".join(node_ips)

            if not seeds:
                seeds = self.nodes[0].ip_address

            ec2_user_data += ' --seeds %s ' % seeds

        ec2_user_data = self.update_bootstrap(ec2_user_data, enable_auto_bootstrap)
        added_nodes = super().add_nodes(
            count=count,
            ec2_user_data=ec2_user_data,
            dc_idx=dc_idx,
            rack=rack,
            enable_auto_bootstrap=enable_auto_bootstrap,
            instance_type=instance_type,
            is_zero_node=is_zero_node
        )
        return added_nodes

    @staticmethod
    def _wait_for_preinstalled_scylla(node):
        node.wait_for_machine_image_configured()

    def _scylla_post_install(self, node: AWSNode, new_scylla_installed: bool, devname: str) -> None:
        super()._scylla_post_install(node, new_scylla_installed, devname)
        if is_ip_ssh_connections_ipv6(self.params):
            node.set_web_listen_address()

    def _reuse_cluster_setup(self, node):
        node.run_startup_script()  # Reconfigure syslog-ng.

    def destroy(self):
        self.stop_nemesis()
        super().destroy()


class CassandraAWSCluster(ScyllaAWSCluster):

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments
                 services, credentials, ec2_instance_type='c6i.xlarge',
                 ec2_ami_username='ubuntu',
                 ec2_block_device_mappings=None,
                 user_prefix=None,
                 n_nodes=3,
                 params=None):
        # pylint: disable=too-many-locals
        if ec2_block_device_mappings is None:
            ec2_block_device_mappings = []
        # We have to pass the cluster name in advance in user_data
        cluster_uuid = uuid.uuid4()
        cluster_prefix = cluster.prepend_user_prefix(user_prefix,
                                                     'cs-db-cluster')
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'cs-db-node')
        node_type = 'cs-db'
        shortid = str(cluster_uuid)[:8]
        name = '%s-%s' % (cluster_prefix, shortid)
        user_data = ('--clustername %s '
                     '--totalnodes %s --version community '
                     '--release 2.1.15' % (name, sum(n_nodes)))
        # pylint: disable=unexpected-keyword-arg
        super().__init__(
            ec2_ami_id=ec2_ami_id,
            ec2_subnet_id=ec2_subnet_id,
            ec2_security_group_ids=ec2_security_group_ids,
            ec2_instance_type=ec2_instance_type,
            ec2_ami_username=ec2_ami_username,
            ec2_user_data=user_data,
            ec2_block_device_mappings=ec2_block_device_mappings,
            cluster_uuid=cluster_uuid,
            services=services,
            credentials=credentials,
            cluster_prefix=cluster_prefix,
            node_prefix=node_prefix,
            n_nodes=n_nodes,
            params=params,
            node_type=node_type)

    def get_seed_nodes(self):
        node = self.nodes[0]
        yaml_dst_path = os.path.join(tempfile.mkdtemp(prefix='sct-cassandra'), 'cassandra.yaml')
        node.remoter.receive_files(src='/etc/cassandra/cassandra.yaml',
                                   dst=yaml_dst_path)
        with open(yaml_dst_path, encoding="utf-8") as yaml_stream:
            conf_dict = yaml.safe_load(yaml_stream)
            try:
                return conf_dict['seed_provider'][0]['parameters'][0]['seeds'].split(',')
            except Exception as exc:  # noqa: BLE001
                raise ValueError('Unexpected cassandra.yaml. Contents:\n%s' % yaml_stream.read()) from exc

    # pylint: disable=too-many-arguments
    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False, instance_type=None):
        if not ec2_user_data:
            if self.nodes:
                seeds = ",".join(self.get_seed_nodes())
                ec2_user_data = ('--clustername %s '
                                 '--totalnodes %s --seeds %s '
                                 '--version community '
                                 '--release 2.1.15' % (self.name,
                                                       count,
                                                       seeds))
        ec2_user_data = self.update_bootstrap(ec2_user_data, enable_auto_bootstrap)
        added_nodes = super().add_nodes(
            count=count,
            ec2_user_data=ec2_user_data,
            dc_idx=dc_idx,
            rack=rack,
            instance_type=instance_type,
        )
        return added_nodes

    def node_setup(self, node, verbose=False, timeout=3600):
        node.wait_ssh_up(verbose=verbose)
        node.wait_db_up(verbose=verbose)

        if self.test_config.REUSE_CLUSTER:
            # for reconfigure syslog-ng
            node.run_startup_script()
            return

        node.remoter.run('sudo apt-get update')
        node.remoter.run('sudo apt-get install -o DPkg::Lock::Timeout=300 -y openjdk-6-jdk')

    @cluster.wait_for_init_wrap
    def wait_for_init(self, node_list=None, verbose=False, timeout=None, check_node_health=True):  # pylint: disable=too-many-arguments
        self.get_seed_nodes()


class LoaderSetAWS(cluster.BaseLoaderSet, AWSCluster):

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments
                 services, credentials, ec2_instance_type='c6i.xlarge',
                 ec2_block_device_mappings=None,
                 ec2_ami_username='centos',
                 user_prefix=None, n_nodes=10, params=None):
        # pylint: disable=too-many-locals
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'loader-node')
        node_type = 'loader'
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'loader-set')
        user_data = ('--clustername %s --totalnodes %s --bootstrap false --stop-services' %
                     (cluster_prefix, n_nodes))
        cluster.BaseLoaderSet.__init__(self,
                                       params=params)

        AWSCluster.__init__(self,
                            ec2_ami_id=ec2_ami_id,
                            ec2_subnet_id=ec2_subnet_id,
                            ec2_security_group_ids=ec2_security_group_ids,
                            ec2_instance_type=ec2_instance_type,
                            ec2_ami_username=ec2_ami_username,
                            ec2_user_data=user_data,
                            services=services,
                            ec2_block_device_mappings=ec2_block_device_mappings,
                            credentials=credentials,
                            cluster_prefix=cluster_prefix,
                            node_prefix=node_prefix,
                            n_nodes=n_nodes,
                            params=params,
                            node_type=node_type)


class MonitorSetAWS(cluster.BaseMonitorSet, AWSCluster):

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments  # noqa: PLR0913
                 services, credentials, ec2_instance_type='c6i.xlarge',
                 ec2_block_device_mappings=None,
                 ec2_ami_username='centos',
                 user_prefix=None, n_nodes=10, targets=None, params=None,
                 add_nodes=True, monitor_id=None):
        # pylint: disable=too-many-locals
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-node')
        node_type = 'monitor'
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-set')
        cluster.BaseMonitorSet.__init__(
            self, targets=targets, params=params, monitor_id=monitor_id,
        )

        AWSCluster.__init__(self,
                            ec2_ami_id=ec2_ami_id,
                            ec2_subnet_id=ec2_subnet_id,
                            ec2_security_group_ids=ec2_security_group_ids,
                            ec2_instance_type=ec2_instance_type,
                            ec2_ami_username=ec2_ami_username,
                            services=services,
                            ec2_block_device_mappings=ec2_block_device_mappings,
                            credentials=credentials,
                            cluster_prefix=cluster_prefix,
                            node_prefix=node_prefix,
                            n_nodes=n_nodes,
                            params=params,
                            node_type=node_type,
                            add_nodes=add_nodes)
