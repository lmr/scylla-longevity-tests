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

import contextlib
import datetime
import logging
import time
from typing import List, Optional, Union

from mypy_boto3_ec2 import EC2Client
from mypy_boto3_ec2.service_resource import Instance

from sdcm.provision.aws.capacity_reservation import SCTCapacityReservation
from sdcm.provision.aws.dedicated_host import SCTDedicatedHosts
from sdcm.provision.aws.instance_parameters import AWSInstanceParams
from sdcm.provision.aws.utils import ec2_services, ec2_clients, find_instance_by_id, set_tags_on_instances, \
    wait_for_provision_request_done, create_spot_fleet_instance_request, \
    create_spot_instance_request
from sdcm.provision.aws.constants import SPOT_CNT_LIMIT, SPOT_FLEET_LIMIT, SPOT_REQUEST_TIMEOUT, STATUS_FULFILLED, \
    SPOT_STATUS_UNEXPECTED_ERROR, FLEET_LIMIT_EXCEEDED_ERROR, SPOT_CAPACITY_NOT_AVAILABLE_ERROR
from sdcm.provision.common.provisioner import TagsType, ProvisionParameters, InstanceProvisionerBase

LOGGER = logging.getLogger(__name__)


class AWSInstanceProvisioner(InstanceProvisionerBase):  # pylint: disable=too-few-public-methods
    # TODO: Make them configurable
    _wait_interval = 5
    _iam_fleet_role = 'arn:aws:iam::797456418907:role/aws-ec2-spot-fleet-role'

    def provision(  # pylint: disable=too-many-arguments
            self,
            provision_parameters: ProvisionParameters,
            instance_parameters: AWSInstanceParams,
            count: int,
            tags: Union[List[TagsType], TagsType] = None,
            names: List[str] = None) -> List[Instance]:
        if tags is None:
            tags = {}
        if isinstance(tags, dict):
            tags = [tags] * count
        elif isinstance(tags, list):
            tags = tags.copy()

        if names:
            assert len(names) == count, "Names length should be equal to count"
        assert len(tags) == count, "Tags length should be equal to count"

        for node_id, name in enumerate(names):
            tag = tags[node_id]
            tag['Name'] = name

        if provision_parameters.spot:
            return self._provision_spot_instances(
                provision_parameters=provision_parameters,
                instance_parameters=instance_parameters,
                count=count,
                tags=tags,
            )
        # None spot instances are called On-Demand instances in AWS naming
        if provision_parameters.price:
            raise ValueError("AWS does not support price targeting for on-demand instances")
        if provision_parameters.duration:
            LOGGER.info("AWS does not support duration for on-demand instances")
        return self._provision_on_demand_instances(
            provision_parameters=provision_parameters,
            instance_parameters=instance_parameters,
            count=count,
            tags=tags,
        )

    @staticmethod
    def _is_provision_type_fleet(count: int) -> bool:
        return count > SPOT_CNT_LIMIT

    def _provision_instance_limit(self, count: int) -> int:
        return SPOT_FLEET_LIMIT if self._is_provision_type_fleet(count) else SPOT_CNT_LIMIT

    @property
    def _spot_valid_until(self) -> datetime.datetime:
        return datetime.datetime.now() + datetime.timedelta(minutes=SPOT_REQUEST_TIMEOUT / 60 + 5)

    @staticmethod
    def _ec2_client(provision_parameters: ProvisionParameters) -> EC2Client:
        return ec2_clients[provision_parameters.region_name]

    @staticmethod
    def _full_availability_zone_name(provision_parameters: ProvisionParameters) -> str:
        return provision_parameters.region_name + provision_parameters.availability_zone

    @staticmethod
    def _provision_on_demand_instances(
            provision_parameters: ProvisionParameters,
            instance_parameters: AWSInstanceParams,
            count: int,
            tags: List[TagsType]) -> List[Instance]:
        instance_parameters_dict = instance_parameters.dict(
            exclude_none=True, exclude_defaults=True, exclude_unset=True, encode_user_data=False)
        if cr_id := SCTCapacityReservation.reservations.get(provision_parameters.availability_zone, {}).get(
                instance_parameters.InstanceType):
            instance_parameters_dict['CapacityReservationSpecification'] = {
                'CapacityReservationTarget': {
                    'CapacityReservationId': cr_id
                }
            }
        if host_id := SCTDedicatedHosts.get_host(provision_parameters.region_name + provision_parameters.availability_zone,
                                                 instance_parameters.InstanceType):
            instance_parameters_dict['Placement'] = {
                # **instance_parameters_dict.get('Placement', {}),
                'HostId': host_id
            }
        LOGGER.info("[%s] Creating {count} on-demand instances using AMI id '%s' with following parameters:\n%s",
                    provision_parameters.region_name,
                    instance_parameters.ImageId,
                    instance_parameters_dict
                    )
        instances = ec2_services[provision_parameters.region_name].create_instances(
            **instance_parameters_dict, MinCount=count, MaxCount=count)
        LOGGER.info("Created instances: %s.", instances)
        if instances:
            for ind, instance in enumerate(instances):
                instance_tags = tags.pop()
                set_tags_on_instances(
                    region_name=provision_parameters.region_name,
                    instance_ids=[instance.instance_id],
                    tags={'Name': 'spot_fleet_{}_{}'.format(instance.instance_id, ind)} | instance_tags,
                )
        return instances

    def _provision_spot_instances(
            self,
            provision_parameters: ProvisionParameters,
            instance_parameters: AWSInstanceParams,
            count: int,
            tags: Union[List[TagsType], TagsType]) -> List[Instance]:
        rest_to_provision = count
        provisioned_instances = []
        while rest_to_provision:
            if rest_to_provision // self._provision_instance_limit(count):
                instances_to_provision = self._provision_instance_limit(count)
            else:
                instances_to_provision = rest_to_provision

            if self._is_provision_type_fleet(count) and instances_to_provision > 1:
                new_instances = self._execute_spot_fleet_instance_request(
                    provision_parameters=provision_parameters,
                    instance_parameters=instance_parameters,
                    count=instances_to_provision,
                    tags=tags)
            else:
                new_instances = self._execute_spot_instance_request(
                    provision_parameters=provision_parameters,
                    instance_parameters=instance_parameters,
                    count=instances_to_provision,
                    tags=tags)
            provisioned_instances.extend(new_instances)
            rest_to_provision -= instances_to_provision
        return provisioned_instances

    def _execute_spot_fleet_instance_request(
            self,
            provision_parameters: ProvisionParameters,
            instance_parameters: AWSInstanceParams,
            count: int,
            tags: List[TagsType]) -> List[Instance]:
        request_id = create_spot_fleet_instance_request(
            region_name=provision_parameters.region_name,
            count=count,
            price=provision_parameters.price,
            fleet_role=self._iam_fleet_role,
            instance_parameters=instance_parameters.dict(
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
                encode_user_data=True,
            ),
        )
        instance_ids = wait_for_provision_request_done(
            region_name=provision_parameters.region_name,
            request_ids=[request_id],
            is_fleet=True,
        )
        if not instance_ids:
            with contextlib.suppress(Exception):
                self._ec2_client(provision_parameters).cancel_spot_fleet_requests(
                    SpotFleetRequestIds=[request_id], TerminateInstances=True)
            return []
        LOGGER.info('Spot fleet instances: %s', instance_ids)
        for ind, instance_id in enumerate(instance_ids):
            instance_tags = tags.pop()
            set_tags_on_instances(
                region_name=provision_parameters.region_name,
                instance_ids=[instance_id],
                tags={'Name': 'spot_fleet_{}_{}'.format(instance_id, ind)} | instance_tags,
            )
        self._ec2_client(provision_parameters).cancel_spot_fleet_requests(
            SpotFleetRequestIds=[request_id], TerminateInstances=False)
        return [find_instance_by_id(
            region_name=provision_parameters.region_name, instance_id=instance_id) for instance_id in instance_ids]

    def _get_provisioned_fleet_instance_ids(
            self,
            provision_parameters: ProvisionParameters,
            request_ids: List[str]) -> Optional[List[str]]:
        try:
            resp = self._ec2_client(provision_parameters).describe_spot_fleet_requests(SpotFleetRequestIds=request_ids)
            LOGGER.info("%s: - %s", request_ids, resp)
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.info("%s: - failed to get status: %s", request_ids, exc)
            return []
        for req in resp['SpotFleetRequestConfigs']:
            if req['SpotFleetRequestState'] == 'active' and 'ActivityStatus' not in req and \
                    req['ActivityStatus'] == STATUS_FULFILLED:
                continue
            if 'ActivityStatus' in req and req['ActivityStatus'] == SPOT_STATUS_UNEXPECTED_ERROR:
                current_time = datetime.datetime.now().timetuple()
                search_start_time = datetime.datetime(
                    current_time.tm_year, current_time.tm_mon, current_time.tm_mday)
                resp = self._ec2_client(provision_parameters).describe_spot_fleet_request_history(
                    SpotFleetRequestId=request_ids[0],
                    StartTime=search_start_time,
                    MaxResults=10,
                )
                LOGGER.info('Fleet request error history: %s', resp)
                errors = [i['EventInformation']['EventSubType'] for i in resp['HistoryRecords']]
                for error in [FLEET_LIMIT_EXCEEDED_ERROR, SPOT_CAPACITY_NOT_AVAILABLE_ERROR]:
                    if error in errors:
                        return None
            return []
        return [inst['InstanceId'] for inst in resp['ActiveInstances']]

    def _wait_for_fleet_request_done(self, provision_parameters: ProvisionParameters, request_id: str):
        """
        Wait for spot fleet request fulfilled
        :param request_id: spot fleet request id
        :return: list of spot instance id-s
        """
        LOGGER.info('Waiting for spot fleet...')
        waiting_time = 0
        provisioned_instance_ids = []
        while not provisioned_instance_ids and waiting_time < SPOT_REQUEST_TIMEOUT:
            time.sleep(self._wait_interval)
            provisioned_instance_ids = self._get_provisioned_fleet_instance_ids(provision_parameters, [request_id])
            if provisioned_instance_ids is None:
                break
            waiting_time += self._wait_interval
        if not provisioned_instance_ids:
            with contextlib.suppress(Exception):
                self._ec2_client(provision_parameters).cancel_spot_fleet_requests(
                    SpotFleetRequestIds=[request_id], TerminateInstances=True)
        return provisioned_instance_ids

    def _execute_spot_instance_request(
            self,
            provision_parameters: ProvisionParameters,
            instance_parameters: AWSInstanceParams,
            count: int,
            tags: List[TagsType]) -> List[Instance]:
        request_ids = create_spot_instance_request(
            region_name=provision_parameters.region_name,
            count=count,
            price=getattr(provision_parameters, 'price', None),
            instance_parameters=instance_parameters.dict(
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
                encode_user_data=True,
            ),
            full_availability_zone=self._full_availability_zone_name(provision_parameters),
            valid_until=self._spot_valid_until,
        )
        instance_ids = wait_for_provision_request_done(
            region_name=provision_parameters.region_name,
            request_ids=request_ids,
            is_fleet=False,
        )
        if not instance_ids:
            with contextlib.suppress(Exception):
                self._ec2_client(provision_parameters).cancel_spot_instance_requests(SpotInstanceRequestIds=request_ids)
            return []
        for ind, instance_id in enumerate(instance_ids):
            instance_tags = tags.pop()
            set_tags_on_instances(
                region_name=provision_parameters.region_name,
                instance_ids=[instance_id],
                tags={'Name': 'spot_{}_{}'.format(instance_id, ind)} | instance_tags,
            )
        self._ec2_client(provision_parameters).cancel_spot_instance_requests(SpotInstanceRequestIds=request_ids)
        return [find_instance_by_id(
            region_name=provision_parameters.region_name, instance_id=instance_id) for instance_id in instance_ids]
