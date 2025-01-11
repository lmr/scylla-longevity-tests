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

import abc
from functools import cached_property
from typing import List, Dict, Tuple

from pydantic import BaseModel, ConfigDict
from sdcm import cluster
from sdcm.provision.aws.instance_parameters import AWSInstanceParams
from sdcm.provision.aws.provisioner import AWSInstanceProvisioner
from sdcm.provision.common.provision_plan import ProvisionPlan
from sdcm.provision.common.provision_plan_builder import ProvisionPlanBuilder
from sdcm.provision.common.provisioner import TagsType
from sdcm.provision.network_configuration import network_interfaces_count
from sdcm.sct_config import SCTConfiguration
from sdcm.sct_provision.aws.instance_parameters_builder import ScyllaInstanceParamsBuilder, \
    LoaderInstanceParamsBuilder, MonitorInstanceParamsBuilder, OracleScyllaInstanceParamsBuilder, ScyllaZeroTokenParamsBuilder
from sdcm.sct_provision.aws.user_data import ScyllaUserDataBuilder, AWSInstanceUserDataBuilder
from sdcm.sct_provision.common.utils import INSTANCE_PROVISION_SPOT, INSTANCE_PROVISION_SPOT_FLEET
from sdcm.test_config import TestConfig
from sdcm.provision.aws.utils import create_cluster_placement_groups_aws


class ClusterNode(BaseModel):
    parent_cluster: 'ClusterBase' = None
    region_id: int
    az_id: int
    node_num: int
    node_name_prefix: str

    @property
    def name(self):
        return self.node_name_prefix + '-' + str(self.node_num)

    @property
    def tags(self) -> Dict[str, str]:
        return self.parent_cluster.tags | {'NodeIndex': str(self.node_num)}


class ClusterBase(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: SCTConfiguration
    test_id: str
    common_tags: TagsType
    _NODE_TYPE = None
    _NODE_PREFIX = None
    _INSTANCE_TYPE_PARAM_NAME = None
    _NODE_NUM_PARAM_NAME = None
    _INSTANCE_PARAMS_BUILDER = None
    _USER_PARAM = None
    _USE_PLACEMENT_GROUP = True

    @property
    def _provisioner(self):
        return AWSInstanceProvisioner()

    @property
    def nodes(self):
        nodes = []
        node_num = 0
        for region_id in range(len(self._regions_with_nodes)):
            for idx in range(self._node_nums[region_id]):
                node_num += 1
                nodes.append(
                    ClusterNode(
                        parent_cluster=self,
                        node_num=node_num,
                        region_id=region_id,
                        az_id=idx % len(self._azs),
                        node_name_prefix=self._node_prefix,
                    )
                )
        return nodes

    @property
    def _test_config(self):
        return TestConfig()

    @property
    def _cluster_postfix(self):
        return self._NODE_PREFIX + '-cluster'

    @property
    def _node_postfix(self):
        return self._NODE_PREFIX + '-node'

    @property
    def _user_prefix(self):
        return self.params.get('user_prefix')

    @property
    def cluster_name(self):
        return '%s-%s' % (cluster.prepend_user_prefix(self._user_prefix, self._cluster_postfix), self._short_id)

    @property
    def placement_group_name(self):
        if self.params.get("use_placement_group") and self._USE_PLACEMENT_GROUP:
            return '%s-%s' % (
                cluster.prepend_user_prefix(self._user_prefix, "placement_group"), self._short_id)
        else:
            return None

    @property
    def _node_prefix(self):
        return '%s-%s' % (cluster.prepend_user_prefix(self._user_prefix, self._node_postfix), self._short_id)

    @property
    def _short_id(self):
        return str(self.test_id)[:8]

    @property
    def tags(self):
        return self.common_tags | {"NodeType": str(self._NODE_TYPE), "UserName": self.params.get(self._USER_PARAM)}

    def _az_nodes(self, region_id: int) -> List[int]:
        az_nodes = [0] * len(self._azs)
        for node_num in range(self._node_nums[region_id]):
            az_nodes[node_num % len(self._azs)] += 1
        return az_nodes

    def _node_tags(self, region_id: int, az_id: int) -> List[TagsType]:
        return [node.tags for node in self.nodes if node.region_id == region_id and node.az_id == az_id]

    def _node_names(self, region_id: int, az_id: int) -> List[str]:
        return [node.name for node in self.nodes if node.region_id == region_id and node.az_id == az_id]

    @property
    def _instance_provision(self):
        instance_provision = self.params.get('instance_provision')
        return INSTANCE_PROVISION_SPOT if instance_provision == INSTANCE_PROVISION_SPOT_FLEET else instance_provision

    @property
    @abc.abstractmethod
    def _user_data(self) -> str:
        pass

    @cached_property
    def _regions(self) -> List[str]:
        return self.params.region_names

    @cached_property
    def _regions_with_nodes(self) -> List[str]:
        output = []
        for region_id, region_name in enumerate(self.params.region_names):
            if len(self._node_nums) <= region_id:
                continue
            if self._node_nums[region_id] > 0:
                output.append(region_name)
        return output

    def _region(self, region_id: int) -> str:
        return self.params.region_names[region_id]

    @cached_property
    def _azs(self) -> str:
        return self.params.get('availability_zone').split(',')

    @cached_property
    def _node_nums(self) -> List[int]:
        node_nums = self.params.get(self._NODE_NUM_PARAM_NAME)
        if isinstance(node_nums, list):
            return [int(num) for num in node_nums]
        if isinstance(node_nums, int):
            return [node_nums]
        if isinstance(node_nums, str):
            return [int(num) for num in node_nums.split()]
        raise ValueError('Unexpected value of %s parameter' % (self._NODE_NUM_PARAM_NAME,))

    @property
    def _instance_type(self) -> str:
        return self.params.get(self._INSTANCE_TYPE_PARAM_NAME)

    @property
    def _test_duration(self) -> int:
        return self.params.get('test_duration')

    def _spot_low_price(self, region_id: int) -> float:
        from sdcm.utils.pricing import AWSPricing  # pylint: disable=import-outside-toplevel

        aws_pricing = AWSPricing()
        on_demand_price = float(aws_pricing.get_on_demand_instance_price(
            region_name=self._region(region_id),
            instance_type=self._instance_type,
        ))
        return on_demand_price * self.params.get('spot_max_price')

    def provision_plan(self, region_id: int, availability_zone: str) -> ProvisionPlan:
        return ProvisionPlanBuilder(
            initial_provision_type=self._instance_provision,
            duration=self._test_duration,
            fallback_provision_on_demand=self.params.get('instance_provision_fallback_on_demand'),
            region_name=self._region(region_id),
            availability_zone=availability_zone,
            spot_low_price=self._spot_low_price(region_id),
            provisioner=AWSInstanceProvisioner(),
        ).provision_plan

    def _instance_parameters(self, region_id: int, availability_zone: int = 0) -> AWSInstanceParams:
        params_builder = self._INSTANCE_PARAMS_BUILDER(  # pylint: disable=not-callable
            params=self.params,
            region_id=region_id,
            user_data_raw=self._user_data,
            availability_zone=availability_zone,
            placement_group=self.placement_group_name
        )
        return AWSInstanceParams(**params_builder.dict(exclude_none=True, exclude_unset=True, exclude_defaults=True))

    def provision(self):
        if self._node_nums == [0]:
            return []
        total_instances_provisioned = []
        for region_id in range(len(self._regions_with_nodes)):
            az_nodes = self._az_nodes(region_id=region_id)
            for az_id, _ in enumerate(self._azs):
                node_count = az_nodes[az_id]
                if not node_count:
                    continue
                instance_parameters = self._instance_parameters(region_id=region_id, availability_zone=az_id)
                node_tags = self._node_tags(region_id=region_id, az_id=az_id)
                node_names = self._node_names(region_id=region_id, az_id=az_id)
                instances = self.provision_plan(region_id, self._azs[az_id]).provision_instances(
                    instance_parameters=instance_parameters,
                    node_tags=node_tags,
                    node_names=node_names,
                    node_count=node_count
                )
                if not instances:
                    raise RuntimeError('End of provision plan reached, but no instances provisioned')
                total_instances_provisioned.extend(instances)
        return total_instances_provisioned


class DBCluster(ClusterBase):
    _NODE_TYPE = 'scylla-db'
    _NODE_PREFIX = 'db'
    _INSTANCE_TYPE_PARAM_NAME = 'instance_type_db'
    _NODE_NUM_PARAM_NAME = 'n_db_nodes'
    _ZEROTOKEN_NODE_NUM_PARAM_NAME = 'n_db_zero_token_nodes'
    _ZEROTOKEN_NODE_INSTANCE_TYPE_PARAM_NAME = "zero_token_instance_type_db"
    _USE_ZERO_NODES_PARAM_NAME = "use_zero_nodes"
    _INSTANCE_PARAMS_BUILDER = ScyllaInstanceParamsBuilder
    _ZERO_TOKEN_INSTANCE_PARAMS_BUILDER = ScyllaZeroTokenParamsBuilder
    _USER_PARAM = 'ami_db_scylla_user'

    @property
    def _user_data(self) -> str:
        return ScyllaUserDataBuilder(
            params=self.params,
            cluster_name=self.cluster_name,
            user_data_format_version=self.params.get('user_data_format_version'),
            syslog_host_port=self._test_config.get_logging_service_host_port(),
        ).to_string()

    def _zero_token_instance_parameters(self, region_id: int, availability_zone: int = 0) -> AWSInstanceParams:
        params_builder = self._ZERO_TOKEN_INSTANCE_PARAMS_BUILDER(  # pylint: disable=not-callable
            params=self.params,
            region_id=region_id,
            user_data_raw=self._user_data,
            availability_zone=availability_zone,
            placement_group=self.placement_group_name
        )
        return AWSInstanceParams(**params_builder.dict(exclude_none=True, exclude_unset=True, exclude_defaults=True))

    def _az_nodes(self, region_id: int) -> Tuple[List[int], List[int]]:
        az_token_nodes = [0] * len(self._azs)
        az_zerotoken_nodes = [0] * len(self._azs)
        for node_num in range(self._get_data_nodes()[region_id]):
            az_token_nodes[node_num % len(self._azs)] += 1
        if zero_token_nodes := self._get_zero_token_nodes():
            for node_num in range(zero_token_nodes[region_id]):
                az_zerotoken_nodes[node_num % len(self._azs)] += 1

        return az_token_nodes, az_zerotoken_nodes

    def _get_data_nodes(self) -> list[int]:
        node_nums = self.params.get(self._NODE_NUM_PARAM_NAME)
        if isinstance(node_nums, list):
            return [int(num) for num in node_nums]
        elif isinstance(node_nums, int):
            return [node_nums]
        elif isinstance(node_nums, str):
            return [int(num) for num in node_nums.split()]
        else:
            raise ValueError('Unexpected value of %s parameter' % (self._NODE_NUM_PARAM_NAME,))

    def _get_zero_token_nodes(self) -> list[int]:
        zero_token_nodes_num = self.params.get(self._ZEROTOKEN_NODE_NUM_PARAM_NAME)
        use_zero_nodes = self.params.get(self._USE_ZERO_NODES_PARAM_NAME)
        if not use_zero_nodes or not zero_token_nodes_num:
            return []
        if isinstance(zero_token_nodes_num, list):
            return [int(num) for num in zero_token_nodes_num]
        elif isinstance(zero_token_nodes_num, int):
            return [zero_token_nodes_num]
        elif isinstance(zero_token_nodes_num, str):
            return [int(num) for num in zero_token_nodes_num.split()]
        else:
            raise ValueError('Unexpected value of %s parameter' % (self._ZEROTOKEN_NODE_NUM_PARAM_NAME,))

    @cached_property
    def _node_nums(self) -> List[int]:
        total_nodes = self._get_data_nodes()
        zero_token_nodes = self._get_zero_token_nodes()
        if zero_token_nodes:
            total_nodes = [n1 + n2 for n1,
                           n2 in zip(total_nodes, zero_token_nodes)]
        return total_nodes

    def provision(self):
        if self._node_nums == [0]:
            return []
        total_instances_provisioned = []
        for region_id in range(len(self._regions_with_nodes)):
            az_nodes, az_zero_nodes = self._az_nodes(region_id=region_id)
            for az_id, _ in enumerate(self._azs):

                node_count = az_nodes[az_id]
                zero_node_count = az_zero_nodes[az_id]
                if node_count:

                    instance_parameters = self._instance_parameters(region_id=region_id, availability_zone=az_id)
                    node_tags = self._node_tags(region_id=region_id, az_id=az_id)[:node_count]
                    node_names = self._node_names(region_id=region_id, az_id=az_id)[:node_count]
                    instances = self.provision_plan(region_id, self._azs[az_id]).provision_instances(
                        instance_parameters=instance_parameters,
                        node_tags=node_tags,
                        node_names=node_names,
                        node_count=node_count
                    )
                    if not instances:
                        raise RuntimeError('End of provision plan reached, but no instances provisioned')
                    total_instances_provisioned.extend(instances)

                if zero_node_count:
                    instance_parameters = self._zero_token_instance_parameters(
                        region_id=region_id, availability_zone=az_id)
                    node_tags = self._node_tags(region_id=region_id, az_id=az_id)[node_count:]
                    for node_tag in node_tags:
                        node_tag.update({"ZeroTokenNode": "True"})
                    node_names = self._node_names(region_id=region_id, az_id=az_id)[node_count:]
                    instances = self.provision_plan(region_id, self._azs[az_id]).provision_instances(
                        instance_parameters=instance_parameters,
                        node_tags=node_tags,
                        node_names=node_names,
                        node_count=zero_node_count
                    )
                    if not instances:
                        raise RuntimeError('End of provision plan reached, but no instances provisioned')
                    total_instances_provisioned.extend(instances)
        return total_instances_provisioned


class OracleDBCluster(ClusterBase):
    _NODE_TYPE = 'oracle-db'
    _NODE_PREFIX = 'oracle'
    _INSTANCE_TYPE_PARAM_NAME = 'instance_type_db_oracle'
    _NODE_NUM_PARAM_NAME = 'n_test_oracle_db_nodes'
    _INSTANCE_PARAMS_BUILDER = OracleScyllaInstanceParamsBuilder
    _USER_PARAM = 'ami_db_scylla_user'

    @property
    def _user_data(self) -> str:
        return ScyllaUserDataBuilder(
            params=self.params,
            cluster_name=self.cluster_name,
            user_data_format_version=self.params.get('oracle_user_data_format_version'),
            syslog_host_port=self._test_config.get_logging_service_host_port(),
        ).to_string()


class LoaderCluster(ClusterBase):
    _NODE_TYPE = 'loader'
    _NODE_PREFIX = 'loader'
    _INSTANCE_TYPE_PARAM_NAME = 'instance_type_loader'
    _NODE_NUM_PARAM_NAME = 'n_loaders'
    _INSTANCE_PARAMS_BUILDER = LoaderInstanceParamsBuilder
    _USER_PARAM = 'ami_loader_user'

    @property
    def _user_data(self) -> str:
        return AWSInstanceUserDataBuilder(
            params=self.params,
            syslog_host_port=self._test_config.get_logging_service_host_port(),
            aws_additional_interface=network_interfaces_count(self.params) > 1,
        ).to_string()


class MonitoringCluster(ClusterBase):
    _NODE_TYPE = 'monitor'
    _NODE_PREFIX = 'monitor'
    _INSTANCE_TYPE_PARAM_NAME = 'instance_type_monitor'
    _NODE_NUM_PARAM_NAME = 'n_monitor_nodes'
    _INSTANCE_PARAMS_BUILDER = MonitorInstanceParamsBuilder
    _USER_PARAM = 'ami_monitor_user'
    # disable placement group for monitor nodes, because it doesn't need low-latency network performance
    _USE_PLACEMENT_GROUP = False

    @property
    def _user_data(self) -> str:
        return AWSInstanceUserDataBuilder(
            params=self.params,
            syslog_host_port=self._test_config.get_logging_service_host_port(),
        ).to_string()


class PlacementGroup(ClusterBase):

    @property
    def _user_data(self) -> str:
        return ''

    def provision(self):
        if self.placement_group_name:
            create_cluster_placement_groups_aws(
                name=self.placement_group_name, tags=self.common_tags, region=self._region(0))


ClusterNode.update_forward_refs()
