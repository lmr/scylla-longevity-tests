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

import os
import re
import json
import time
import uuid
import base64
import random
import logging
import tempfile
from math import floor
from typing import List, Dict, Optional
from datetime import datetime
from textwrap import dedent
from functools import cached_property
from contextlib import ExitStack

import yaml
import boto3
from mypy_boto3_ec2 import EC2Client
from pkg_resources import parse_version
from botocore.exceptions import WaiterError

from sdcm import ec2_client, cluster
from sdcm.remote import LocalCmdRunner, shell_script_cmd, NETWORK_EXCEPTIONS
from sdcm.cluster import INSTANCE_PROVISION_ON_DEMAND
from sdcm.ec2_client import CreateSpotInstancesError
from sdcm.utils.common import list_instances_aws, get_ami_tags, ec2_instance_wait_public_ip, MAX_SPOT_DURATION_TIME
from sdcm.utils.decorators import retrying
from sdcm.sct_events.system import SpotTerminationEvent
from sdcm.sct_events.filters import DbEventsFilter
from sdcm.sct_events.database import DatabaseLogEvent


LOGGER = logging.getLogger(__name__)

INSTANCE_PROVISION_SPOT_FLEET = 'spot_fleet'
INSTANCE_PROVISION_SPOT_LOW_PRICE = 'spot_low_price'
INSTANCE_PROVISION_SPOT_DURATION = 'spot_duration'
SPOT_CNT_LIMIT = 20
SPOT_FLEET_LIMIT = 50
SPOT_TERMINATION_CHECK_OVERHEAD = 15
LOCAL_CMD_RUNNER = LocalCmdRunner()

# pylint: disable=too-many-lines


def _tags_as_ec2_tags(tags: Dict[str, str]) -> List[Dict[str, str]]:
    return [{"Key": key, "Value": value} for key, value in tags.items()]


class AWSCluster(cluster.BaseCluster):  # pylint: disable=too-many-instance-attributes,abstract-method,

    """
    Cluster of Node objects, started on Amazon EC2.
    """

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments
                 services, credentials, cluster_uuid=None,
                 ec2_instance_type='c5.xlarge', ec2_ami_username='root',
                 ec2_user_data='', ec2_block_device_mappings=None,
                 cluster_prefix='cluster',
                 node_prefix='node', n_nodes=10, params=None, node_type=None, extra_network_interface=False):
        # pylint: disable=too-many-locals
        region_names = params.get('region_name').split()
        if len(credentials) > 1 or len(region_names) > 1:
            assert len(credentials) == len(region_names)
        for idx, _ in enumerate(region_names):
            credential = credentials[idx]
            cluster.CREDENTIALS.append(credential)

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

        super(AWSCluster, self).__init__(cluster_uuid=cluster_uuid,
                                         cluster_prefix=cluster_prefix,
                                         node_prefix=node_prefix,
                                         n_nodes=n_nodes,
                                         params=params,
                                         region_names=self.region_names,
                                         node_type=node_type,
                                         extra_network_interface=extra_network_interface
                                         )

    def __str__(self):
        return 'Cluster %s (AMI: %s Type: %s)' % (self.name,
                                                  self._ec2_ami_id,
                                                  self._ec2_instance_type)

    @staticmethod
    def calculate_spot_duration_for_test():
        return floor(cluster.TEST_DURATION / 60) * 60 + 60

    def _create_on_demand_instances(self, count, interfaces, ec2_user_data, dc_idx=0):  # pylint: disable=too-many-arguments
        ami_id = self._ec2_ami_id[dc_idx]
        self.log.debug(f"Creating {count} on-demand instances using AMI id '{ami_id}'... ")
        params = dict(ImageId=ami_id,
                      UserData=ec2_user_data,
                      MinCount=count,
                      MaxCount=count,
                      KeyName=self._credentials[dc_idx].key_pair_name,
                      BlockDeviceMappings=self._ec2_block_device_mappings,
                      NetworkInterfaces=interfaces,
                      InstanceType=self._ec2_instance_type)
        instance_profile = self.params.get('aws_instance_profile_name')
        if instance_profile:
            params['IamInstanceProfile'] = {'Name': instance_profile}
        instances = self._ec2_services[dc_idx].create_instances(**params)
        self.log.debug("Created instances: %s." % instances)
        return instances

    def _create_spot_instances(self, count, interfaces, ec2_user_data='', dc_idx=0):  # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        ec2 = ec2_client.EC2ClientWarpper(region_name=self.region_names[dc_idx],
                                          spot_max_price_percentage=self.params.get('spot_max_price'))
        subnet_info = ec2.get_subnet_info(self._ec2_subnet_id[dc_idx])
        spot_params = dict(instance_type=self._ec2_instance_type,
                           image_id=self._ec2_ami_id[dc_idx],
                           region_name=subnet_info['AvailabilityZone'],
                           network_if=interfaces,
                           key_pair=self._credentials[dc_idx].key_pair_name,
                           user_data=ec2_user_data,
                           count=count,
                           block_device_mappings=self._ec2_block_device_mappings,
                           aws_instance_profile=self.params.get('aws_instance_profile_name'))
        if self.instance_provision == INSTANCE_PROVISION_SPOT_DURATION:
            # duration value must be a multiple of 60
            spot_params.update({'duration': self.calculate_spot_duration_for_test()})

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

        return instances

    def _create_instances(self, count, ec2_user_data='', dc_idx=0):
        if not count:  # EC2 API fails if we request zero instances.
            return []

        if not ec2_user_data:
            ec2_user_data = self._ec2_user_data
        self.log.debug("Passing user_data '%s' to create_instances", ec2_user_data)
        interfaces = [{'DeviceIndex': 0,
                       'SubnetId': self._ec2_subnet_id[dc_idx],
                       'AssociatePublicIpAddress': True,
                       'Groups': self._ec2_security_group_ids[dc_idx]}]
        if self.extra_network_interface:
            interfaces = [{'DeviceIndex': 0,
                           'SubnetId': self._ec2_subnet_id[dc_idx],
                           'Groups': self._ec2_security_group_ids[dc_idx]},
                          {'DeviceIndex': 1,
                           'SubnetId': self._ec2_subnet_id[dc_idx],
                           'Groups': self._ec2_security_group_ids[dc_idx]}]

        self.log.info(f"Create {self.instance_provision} instance(s)")
        if self.instance_provision == 'mixed':
            instances = self._create_mixed_instances(count, interfaces, ec2_user_data, dc_idx)
        elif self.instance_provision == INSTANCE_PROVISION_ON_DEMAND:
            instances = self._create_on_demand_instances(count, interfaces, ec2_user_data, dc_idx)
        elif self.instance_provision == INSTANCE_PROVISION_SPOT_FLEET and count > 1:
            instances = self._create_spot_instances(count, interfaces, ec2_user_data, dc_idx)
        else:
            instances = self.fallback_provision_type(count, interfaces, ec2_user_data, dc_idx)

        return instances

    def fallback_provision_type(self, count, interfaces, ec2_user_data, dc_idx):
        # If user requested to use spot instances, first try to get spot_duration instances (according to test_duration)
        # - if test_duration > 360 or spot_duration failed, try to get spot_low_price
        # - if instance_provision_fallback_on_demand==true and previous attempts failed then create on_demand instance
        # else fail the test
        instances = None

        if self.instance_provision.lower() == 'spot' or self.instance_provision == INSTANCE_PROVISION_SPOT_DURATION \
                or (self.instance_provision == INSTANCE_PROVISION_SPOT_FLEET and count == 1):
            instances_provision_fallbacks = [INSTANCE_PROVISION_SPOT_DURATION, INSTANCE_PROVISION_SPOT_LOW_PRICE]
        else:
            # If self.instance_provision == "spot_low_price"
            instances_provision_fallbacks = [self.instance_provision]

        if 'instance_provision_fallback_on_demand' in self.params \
                and self.params['instance_provision_fallback_on_demand']:
            instances_provision_fallbacks.append(INSTANCE_PROVISION_ON_DEMAND)

        self.log.debug(f"Instances provision fallbacks : {instances_provision_fallbacks}")

        for instances_provision_type in instances_provision_fallbacks:
            # If original instance provision type was not spot_duration, we need to check that test duration
            # not exceeds maximum allowed spot duration (360 min right for now). If it exceeds - it's impossible to
            # request for spot duration provision type.
            if instances_provision_type == INSTANCE_PROVISION_SPOT_DURATION and \
                    self.calculate_spot_duration_for_test() > MAX_SPOT_DURATION_TIME:
                self.log.info(f"Create {instances_provision_type} instance(s) is not alowed: "
                              f"Test duration too long for spot_duration instance type. "
                              f"Max possible test duration time for this instance type is {MAX_SPOT_DURATION_TIME} "
                              f"minutes"
                              )
                continue

            try:
                self.log.info(f"Create {instances_provision_type} instance(s)")
                if instances_provision_type == INSTANCE_PROVISION_ON_DEMAND:
                    instances = self._create_on_demand_instances(count, interfaces, ec2_user_data, dc_idx)
                else:
                    instances = self._create_spot_instances(count, interfaces, ec2_user_data, dc_idx)
                break
            except CreateSpotInstancesError as cl_ex:
                if instances_provision_type == instances_provision_fallbacks[-1]:
                    raise
                elif not self.check_spot_error(str(cl_ex), instances_provision_type):
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

    def _create_mixed_instances(self, count, interfaces, ec2_user_data, dc_idx):  # pylint: disable=too-many-arguments
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
                instances.extend(self._create_spot_instances(count_spot, interfaces, ec2_user_data, dc_idx))
            if count_on_demand > 0:
                self.instance_provision = INSTANCE_PROVISION_ON_DEMAND
                instances.extend(self._create_on_demand_instances(count_on_demand, interfaces, ec2_user_data, dc_idx))
            self.instance_provision = 'mixed'
        elif isinstance(self, LoaderSetAWS):
            self.instance_provision = INSTANCE_PROVISION_SPOT_LOW_PRICE
            instances = self._create_spot_instances(count, interfaces, ec2_user_data, dc_idx)
        elif isinstance(self, MonitorSetAWS):
            self.instance_provision = INSTANCE_PROVISION_ON_DEMAND
            instances.extend(self._create_on_demand_instances(count, interfaces, ec2_user_data, dc_idx))
        else:
            raise Exception('Unsuported type of cluster type %s' % self)
        return instances

    def _get_instances(self, dc_idx):

        test_id = cluster.Setup.test_id()
        if not test_id:
            raise ValueError("test_id should be configured for using reuse_cluster")

        ec2 = ec2_client.EC2ClientWarpper(region_name=self.region_names[dc_idx],
                                          spot_max_price_percentage=self.params.get('spot_max_price'))
        results = list_instances_aws(tags_dict={'TestId': test_id, 'NodeType': self.node_type},
                                     region_name=self.region_names[dc_idx], group_as_region=True)
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
        else:
            if '--bootstrap ' in ec2_user_data:
                ec2_user_data.replace('--bootstrap true', '--bootstrap false')
            else:
                ec2_user_data += ' --bootstrap false '
        return ec2_user_data

    # This is workaround for issue #5179: IPv6 - routing in scylla AMI isn't configured properly (when not Amazon Linux)
    @staticmethod
    def network_config_ipv6_workaround_script():  # pylint: disable=invalid-name
        return dedent("""
            BASE_EC2_NETWORK_URL=http://169.254.169.254/latest/meta-data/network/interfaces/macs/
            MAC=`curl -s ${BASE_EC2_NETWORK_URL}`
            IPv6_CIDR=`curl -s ${BASE_EC2_NETWORK_URL}${MAC}/subnet-ipv6-cidr-blocks`

            grep -qi "amazon linux" /etc/os-release || sudo ip route add $IPv6_CIDR dev eth0

            grep -q IPV6_AUTOCONF /etc/sysconfig/network-scripts/ifcfg-eth0
            if [[ $? -eq 0 ]]; then
                sudo sed -i 's/^IPV6_AUTOCONF=[^ ]*/IPV6_AUTOCONF=yes/' /etc/sysconfig/network-scripts/ifcfg-eth0
            else
                sudo echo "IPV6_AUTOCONF=yes" >> /etc/sysconfig/network-scripts/ifcfg-eth0
            fi

            grep -q IPV6_DEFROUTE /etc/sysconfig/network-scripts/ifcfg-eth0
            if [[ $? -eq 0 ]]; then
                sudo sed -i 's/^IPV6_DEFROUTE=[^ ]*/IPV6_DEFROUTE=yes/' /etc/sysconfig/network-scripts/ifcfg-eth0
            else
                sudo echo "IPV6_DEFROUTE=yes" >> /etc/sysconfig/network-scripts/ifcfg-eth0
            fi
            sudo systemctl restart network
        """)

    @staticmethod
    def configure_eth1_script():
        return dedent("""
            BASE_EC2_NETWORK_URL=http://169.254.169.254/latest/meta-data/network/interfaces/macs/
            NUMBER_OF_ENI=`curl -s ${BASE_EC2_NETWORK_URL} | wc -w`
            for mac in `curl -s ${BASE_EC2_NETWORK_URL}`
            do
                DEVICE_NUMBER=`curl -s ${BASE_EC2_NETWORK_URL}${mac}/device-number`
                if [[ "$DEVICE_NUMBER" == "1" ]]; then
                   ETH1_MAC=${mac}
                fi
            done

            if [[ ! "${DEVICE_NUMBER}x" == "x" ]]; then
               ETH1_IP_ADDRESS=`curl -s ${BASE_EC2_NETWORK_URL}${ETH1_MAC}/local-ipv4s`
               ETH1_CIDR_BLOCK=`curl -s ${BASE_EC2_NETWORK_URL}${ETH1_MAC}/subnet-ipv4-cidr-block`
            fi

            sudo bash -c "echo 'GATEWAYDEV=eth0' >> /etc/sysconfig/network"

            echo "
            DEVICE="eth1"
            BOOTPROTO="dhcp"
            ONBOOT="yes"
            TYPE="Ethernet"
            USERCTL="yes"
            PEERDNS="yes"
            IPV6INIT="no"
            PERSISTENT_DHCLIENT="1"
            " > /etc/sysconfig/network-scripts/ifcfg-eth1

            echo "
            default via 10.0.0.1 dev eth1 table 2
            ${ETH1_CIDR_BLOCK} dev eth1 src ${ETH1_IP_ADDRESS} table 2
            " > /etc/sysconfig/network-scripts/route-eth1

            echo "
            from ${ETH1_IP_ADDRESS}/32 table 2
            " > /etc/sysconfig/network-scripts/rule-eth1

            sudo systemctl restart network
        """)

    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False):
        post_boot_script = cluster.Setup.get_startup_script()
        if self.extra_network_interface:
            post_boot_script += self.configure_eth1_script()

        if self.params.get('ip_ssh_connections') == 'ipv6':
            post_boot_script += self.network_config_ipv6_workaround_script()

        if isinstance(ec2_user_data, dict):
            ec2_user_data['post_configuration_script'] = base64.b64encode(
                post_boot_script.encode('utf-8')).decode('ascii')
            ec2_user_data = json.dumps(ec2_user_data)
        else:
            if 'clustername' in ec2_user_data:
                ec2_user_data += " --base64postscript={0}".format(
                    base64.b64encode(post_boot_script.encode('utf-8')).decode('ascii'))
            else:
                ec2_user_data = post_boot_script

        if cluster.Setup.REUSE_CLUSTER:
            instances = self._get_instances(dc_idx)
        else:
            instances = self._create_instances(count, ec2_user_data, dc_idx)

        added_nodes = [self._create_node(instance, self._ec2_ami_username,
                                         self.node_prefix, node_index,
                                         self.logdir, dc_idx=dc_idx)
                       for node_index, instance in
                       enumerate(instances, start=self._node_index + 1)]
        for node in added_nodes:
            node.enable_auto_bootstrap = enable_auto_bootstrap
            if self.params.get('ip_ssh_connections') == 'ipv6' and not node.distro.is_amazon2:
                node.config_ipv6_as_persistent()
        self._node_index += len(added_nodes)
        self.nodes += added_nodes
        self.write_node_public_ip_file()
        self.write_node_private_ip_file()
        return added_nodes

    def _create_node(self, instance, ami_username, node_prefix, node_index,  # pylint: disable=too-many-arguments
                     base_logdir, dc_idx):
        node = AWSNode(ec2_instance=instance, ec2_service=self._ec2_services[dc_idx],
                       credentials=self._credentials[dc_idx], parent_cluster=self, ami_username=ami_username,
                       node_prefix=node_prefix, node_index=node_index,
                       base_logdir=base_logdir, dc_idx=dc_idx)
        node.init()
        return node


class AWSNode(cluster.BaseNode):
    """
    Wraps EC2.Instance, so that we can also control the instance through SSH.
    """

    log = LOGGER

    def __init__(self, ec2_instance, ec2_service, credentials, parent_cluster,  # pylint: disable=too-many-arguments
                 node_prefix='node', node_index=1, ami_username='root',
                 base_logdir=None, dc_idx=0):
        self.node_index = node_index
        self._instance = ec2_instance
        self._ec2_service = ec2_service
        self._eth1_private_ip_address = None
        self.eip_allocation_id = None
        ssh_login_info = {'hostname': None,
                          'user': ami_username,
                          'key_file': credentials.key_file}
        super(AWSNode, self).__init__(name=f"{node_prefix}-{self.node_index}",
                                      parent_cluster=parent_cluster,
                                      ssh_login_info=ssh_login_info,
                                      base_logdir=base_logdir,
                                      node_prefix=node_prefix,
                                      dc_idx=dc_idx)

    def init(self):
        LOGGER.debug("Waiting until instance {0._instance} starts running...".format(self))
        self._instance_wait_safe(self._instance.wait_until_running)

        if not cluster.Setup.REUSE_CLUSTER:
            resources_to_tag = [self._instance.id, ]
            if len(self._instance.network_interfaces) == 2:
                # first we need to configure the both networks so we'll have public ip
                self.allocate_and_attach_elastic_ip(self.parent_cluster, self.dc_idx)
                resources_to_tag.append(self.eip_allocation_id)
            self._ec2_service.create_tags(Resources=resources_to_tag, Tags=_tags_as_ec2_tags(self.tags))

        self._wait_public_ip()

        super().init()

    @cached_property
    def tags(self) -> Dict[str, str]:
        return {**super().tags,
                "NodeIndex": str(self.node_index), }

    def _set_keep_alive(self):
        self._ec2_service.create_tags(Resources=[self._instance.id], Tags=[{"Key": "keep", "Value": "alive"}])
        return super()._set_keep_alive()

    @property
    def region(self):
        return self._ec2_service.meta.client.meta.region_name

    @retrying(n=10, sleep_time=5, allowed_exceptions=NETWORK_EXCEPTIONS, message="Retrying set_hostname")
    def set_hostname(self):
        self.log.info('Changing hostname to %s', self.name)
        # Using https://aws.amazon.com/premiumsupport/knowledge-center/linux-static-hostname-rhel7-centos7/
        # FIXME: workaround to avoid host rename generating errors on other commands
        if self.is_debian():
            return
        result = self.remoter.run(f"sudo hostnamectl set-hostname --static {self.name}", ignore_status=True)
        if result.ok:
            self.log.debug('Hostname has been changed succesfully. Apply')
            apply_hostname_change_script = dedent(f"""
                if ! grep "\\$LocalHostname {self.name}" /etc/rsyslog.conf; then
                    echo "" >> /etc/rsyslog.conf
                    echo "\\$LocalHostname {self.name}" >> /etc/rsyslog.conf
                fi
                grep -P "127.0.0.1[^\\\\n]+{self.name}" /etc/hosts || sed -ri "s/(127.0.0.1[ \\t]+localhost[^\\n]*)$/\\1\\t{self.name}/" /etc/hosts
                grep "preserve_hostname: true" /etc/cloud/cloud.cfg 1>/dev/null 2>&1 || echo "preserve_hostname: true" >> /etc/cloud/cloud.cfg
                systemctl restart rsyslog
            """)
            self.remoter.run(f"sudo bash -cxe '{apply_hostname_change_script}'")
            self.log.debug('Continue node %s set up', self.name)
        else:
            self.log.warning('Hostname has not been changed. Error: %s.\n Continue with old name', result.stderr)

    @property
    def is_spot(self):
        return bool(self._instance.instance_lifecycle and 'spot' in self._instance.instance_lifecycle.lower())

    def check_spot_termination(self):
        try:
            result = self.remoter.run(
                'curl http://169.254.169.254/latest/meta-data/spot/instance-action', verbose=False)
            status = result.stdout.strip()
        except Exception as details:  # pylint: disable=broad-except
            self.log.warning('Error during getting spot termination notification %s', details)
            return 0

        if '404 - Not Found' in status:
            return 0

        self.log.warning('Got spot termination notification from AWS %s', status)
        terminate_action = json.loads(status)
        terminate_action_timestamp = time.mktime(datetime.strptime(
            terminate_action['time'], "%Y-%m-%dT%H:%M:%SZ").timetuple())
        next_check_delay = terminate_action['time-left'] = terminate_action_timestamp - time.time()
        SpotTerminationEvent(node=self, message=terminate_action).publish()

        return max(next_check_delay - SPOT_TERMINATION_CHECK_OVERHEAD, 0)

    @property
    def external_address(self):
        """
        the communication address for usage between the test and the nodes
        :return:
        """
        if self.parent_cluster.params.get("ip_ssh_connections") == "ipv6":
            return self.ipv6_ip_address
        elif cluster.IP_SSH_CONNECTIONS == 'public' or cluster.Setup.INTRA_NODE_COMM_PUBLIC:
            return self.public_ip_address
        else:
            return self._instance.private_ip_address

    def _get_public_ip_address(self) -> Optional[str]:
        return self._instance.public_ip_address

    def _get_private_ip_address(self) -> Optional[str]:
        if self._eth1_private_ip_address:
            return self._eth1_private_ip_address
        return self._instance.private_ip_address

    def _get_ipv6_ip_address(self) -> Optional[str]:
        return self._instance.network_interfaces[0].ipv6_addresses[0]["Ipv6Address"]

    def _refresh_instance_state(self):
        raise NotImplementedError()

    def allocate_and_attach_elastic_ip(self, parent_cluster, dc_idx):
        primary_interface = [
            interface for interface in self._instance.network_interfaces if interface.attachment['DeviceIndex'] == 0][0]
        if primary_interface.association_attribute is None:
            # create and attach EIP
            client: EC2Client = boto3.client('ec2', region_name=parent_cluster.region_names[dc_idx])
            response = client.allocate_address(Domain='vpc')

            self.eip_allocation_id = response['AllocationId']
            client.associate_address(
                AllocationId=self.eip_allocation_id,
                NetworkInterfaceId=primary_interface.id,
            )
        self._eth1_private_ip_address = [interface for interface in self._instance.network_interfaces if
                                         interface.attachment['DeviceIndex'] == 1][0].private_ip_address

    def _instance_wait_safe(self, instance_method, *args, **kwargs):
        """
        Wrapper around AWS instance waiters that is safer to use.

        Since AWS adopts an eventual consistency model, sometimes the method
        wait_until_running will raise a botocore.exceptions.WaiterError saying
        the instance does not exist. AWS API guide [1] recommends that the
        procedure is retried using an exponencial backoff algorithm [2].

        :see: [1] http://docs.aws.amazon.com/AWSEC2/latest/APIReference/query-api-troubleshooting.html#eventual-consistency
        :see: [2] http://docs.aws.amazon.com/general/latest/gr/api-retries.html
        """
        threshold = 300
        ok = False
        retries = 0
        max_retries = 9
        while not ok and retries <= max_retries:
            try:
                instance_method(*args, **kwargs)
                ok = True
            except WaiterError:
                time.sleep(min((2 ** retries) * 2, threshold))
                retries += 1

        if not ok:
            try:
                self._instance.reload()
            except Exception as ex:  # pylint: disable=broad-except
                LOGGER.exception("Error while reloading instance metadata: %s", ex)
            finally:
                method_name = instance_method.__name__
                instance_id = self._instance.id
                LOGGER.debug(self._instance.meta.data)
                msg = "Timeout while running '{method_name}' method on AWS instance '{instance_id}'".format(
                    method_name=method_name, instance_id=instance_id)
                raise cluster.NodeError(msg)

    def _wait_public_ip(self):
        ec2_instance_wait_public_ip(self._instance)

    def config_ipv6_as_persistent(self):
        cidr = dedent("""
                        BASE_EC2_NETWORK_URL=http://169.254.169.254/latest/meta-data/network/interfaces/macs/
                        MAC=`curl -s ${BASE_EC2_NETWORK_URL}`
                        curl -s ${BASE_EC2_NETWORK_URL}${MAC}/subnet-ipv6-cidr-blocks
                    """)
        output = self.remoter.run(f"sudo bash -cxe '{cidr}'")
        ipv6_cidr = output.stdout.strip()
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

            seed_provider = [{
                "class_name": "org.apache.cassandra.locator.SimpleSeedProvider",
                "parameters": [{
                    "seeds": f"{random_node.ip_address}"
                }, ],
            }, ]

            with self.remote_scylla_yaml() as scylla_yml:
                scylla_yml["seed_provider"] = seed_provider

        need_to_setup = any(ss in self._instance.instance_type for ss in ("i2", "i3", ))

        if self.parent_cluster.is_ebs_volumes_attached():
            need_to_setup = False

        with ExitStack() as stack:
            if need_to_setup:
                # There is no disk yet, lots of the errors here are acceptable, and we'll ignore them.
                for db_filter in (DbEventsFilter(db_event=DatabaseLogEvent.DATABASE_ERROR, node=self),
                                  DbEventsFilter(db_event=DatabaseLogEvent.SCHEMA_FAILURE, node=self),
                                  DbEventsFilter(db_event=DatabaseLogEvent.NO_SPACE_ERROR, node=self),
                                  DbEventsFilter(db_event=DatabaseLogEvent.FILESYSTEM_ERROR, node=self),
                                  DbEventsFilter(db_event=DatabaseLogEvent.RUNTIME_ERROR, node=self), ):
                    stack.enter_context(db_filter)

                self.remoter.sudo(shell_script_cmd(f"""\
                    sed -e '/.*scylla/s/^/#/g' -i /etc/fstab
                    sed -e '/auto_bootstrap:.*/s/false/true/g' -i /etc/scylla/scylla.yaml
                    if ! grep replace_address_first_boot: /etc/scylla/scylla.yaml; then
                        echo 'replace_address_first_boot: {self.ip_address}' | tee --append /etc/scylla/scylla.yaml
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

            if need_to_setup:
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
        if self.distro.is_amazon2:
            self.log.debug("Installing iproute-tc package for AMAZON2")
            self.remoter.run("sudo yum install -y iproute-tc", ignore_status=True)

        return self.remoter.run("/sbin/tc -h", ignore_status=True).ok

    @property
    def image(self):
        return self._instance.image_id

    @property
    def ena_support(self) -> bool:
        return self._instance.ena_support


class ScyllaAWSCluster(cluster.BaseScyllaCluster, AWSCluster):

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments
                 services, credentials, ec2_instance_type='c5.xlarge',
                 ec2_ami_username='centos',
                 ec2_block_device_mappings=None,
                 user_prefix=None,
                 n_nodes=3,
                 params=None):
        # pylint: disable=too-many-locals
        # We have to pass the cluster name in advance in user_data
        cluster_uuid = cluster.Setup.test_id()
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'db-cluster')
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'db-node')

        node_type = 'scylla-db'
        shortid = str(cluster_uuid)[:8]
        name = '%s-%s' % (cluster_prefix, shortid)

        ami_tags = get_ami_tags(ec2_ami_id[0], region_name=params.get('region_name').split()[0])
        # TODO: remove once all other related code is merged in scylla-pkg and scylla-machine-image
        user_data_format_version = ami_tags.get('sci_version', '2')
        user_data_format_version = ami_tags.get('user_data_format_version', user_data_format_version)

        data_device_type = params.get("data_device")

        if parse_version(user_data_format_version) >= parse_version('2'):
            user_data = dict(scylla_yaml=dict(cluster_name=name),
                             start_scylla_on_first_boot=False,
                             data_device=data_device_type)
        else:
            user_data = ('--clustername %s '
                         '--totalnodes %s' % (name, sum(n_nodes)))
            user_data += ' --stop-services'

        super(ScyllaAWSCluster, self).__init__(ec2_ami_id=ec2_ami_id,
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
                                               node_type=node_type,
                                               extra_network_interface=params.get('extra_network_interface'))
        self.version = '2.1'

    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False):
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
        added_nodes = super(ScyllaAWSCluster, self).add_nodes(count=count,
                                                              ec2_user_data=ec2_user_data,
                                                              dc_idx=dc_idx,
                                                              rack=rack,
                                                              enable_auto_bootstrap=enable_auto_bootstrap)
        return added_nodes

    def node_config_setup(self, node, seed_address=None, endpoint_snitch=None, murmur3_partitioner_ignore_msb_bits=None, client_encrypt=None):  # pylint: disable=too-many-arguments
        setup_params = dict(
            enable_exp=self.params.get('experimental'),
            endpoint_snitch=endpoint_snitch,
            authenticator=self.params.get('authenticator'),
            server_encrypt=self.params.get('server_encrypt'),
            client_encrypt=client_encrypt if client_encrypt is not None else self.params.get('client_encrypt'),
            append_scylla_args=self.get_scylla_args(),
            authorizer=self.params.get('authorizer'),
            hinted_handoff=self.params.get('hinted_handoff'),
            alternator_port=self.params.get('alternator_port'),
            seed_address=seed_address,
            append_scylla_yaml=self.params.get('append_scylla_yaml'),
            murmur3_partitioner_ignore_msb_bits=murmur3_partitioner_ignore_msb_bits,
            ip_ssh_connections=self.params.get('ip_ssh_connections'),
            alternator_enforce_authorization=self.params.get('alternator_enforce_authorization'),
            internode_compression=self.params.get('internode_compression'),
            internode_encryption=self.params.get('internode_encryption'),
            ldap=self.params.get('use_ldap_authorization'),
        )
        if cluster.Setup.INTRA_NODE_COMM_PUBLIC:
            setup_params.update(dict(
                broadcast=node.public_ip_address,
            ))

        if self.extra_network_interface:
            setup_params.update(dict(
                seed_address=seed_address,
                broadcast=node.private_ip_address,
                listen_on_all_interfaces=True,
            ))

        node.config_setup(**setup_params)

    @staticmethod
    def _wait_for_preinstalled_scylla(node):
        node.wait_for_machine_image_configured()

    def _scylla_post_install(self, node: AWSNode, new_scylla_installed: bool, devname: str) -> None:
        super()._scylla_post_install(node, new_scylla_installed, devname)
        if self.params.get('ip_ssh_connections') == 'ipv6':
            node.set_web_listen_address()

    def _reuse_cluster_setup(self, node):
        node.run_startup_script()  # Reconfigure rsyslog.

    def get_endpoint_snitch(self, default_multi_region="Ec2MultiRegionSnitch"):
        return super().get_endpoint_snitch(default_multi_region)

    def destroy(self):
        self.stop_nemesis()
        super(ScyllaAWSCluster, self).destroy()


class CassandraAWSCluster(ScyllaAWSCluster):

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments
                 services, credentials, ec2_instance_type='c5.xlarge',
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

        super(CassandraAWSCluster, self).__init__(ec2_ami_id=ec2_ami_id,  # pylint: disable=unexpected-keyword-arg
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
        with open(yaml_dst_path, 'r') as yaml_stream:
            conf_dict = yaml.safe_load(yaml_stream)
            try:
                return conf_dict['seed_provider'][0]['parameters'][0]['seeds'].split(',')
            except:
                raise ValueError('Unexpected cassandra.yaml '
                                 'contents:\n%s' % yaml_stream.read())

    def add_nodes(self, count, ec2_user_data='', dc_idx=0, rack=0, enable_auto_bootstrap=False):
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
        added_nodes = super(CassandraAWSCluster, self).add_nodes(count=count,
                                                                 ec2_user_data=ec2_user_data,
                                                                 dc_idx=dc_idx,
                                                                 rack=0)
        return added_nodes

    def node_setup(self, node, verbose=False, timeout=3600):
        node.wait_ssh_up(verbose=verbose)
        node.wait_db_up(verbose=verbose)

        if cluster.Setup.REUSE_CLUSTER:
            # for reconfigure rsyslog
            node.run_startup_script()
            return

        node.wait_apt_not_running()
        node.remoter.run('sudo apt-get update')
        node.remoter.run('sudo apt-get install -y collectd collectd-utils')
        node.remoter.run('sudo apt-get install -y openjdk-6-jdk')

    @cluster.wait_for_init_wrap
    def wait_for_init(self, node_list=None, verbose=False, timeout=None, check_node_health=True):  # pylint: disable=too-many-arguments
        self.get_seed_nodes()


class LoaderSetAWS(cluster.BaseLoaderSet, AWSCluster):

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments
                 services, credentials, ec2_instance_type='c5.xlarge',
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

    def __init__(self, ec2_ami_id, ec2_subnet_id, ec2_security_group_ids,  # pylint: disable=too-many-arguments
                 services, credentials, ec2_instance_type='c5.xlarge',
                 ec2_block_device_mappings=None,
                 ec2_ami_username='centos',
                 user_prefix=None, n_nodes=10, targets=None, params=None):
        # pylint: disable=too-many-locals
        node_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-node')
        node_type = 'monitor'
        cluster_prefix = cluster.prepend_user_prefix(user_prefix, 'monitor-set')
        cluster.BaseMonitorSet.__init__(self,
                                        targets=targets,
                                        params=params)

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
                            node_type=node_type)
