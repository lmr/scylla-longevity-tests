import logging
import datetime
import time
import base64

import boto3
from mypy_boto3_ec2 import EC2Client, EC2ServiceResource
from mypy_boto3_ec2.service_resource import Instance
from botocore.exceptions import ClientError, NoRegionError

from sdcm.provision.aws.capacity_reservation import SCTCapacityReservation
from sdcm.provision.aws.dedicated_host import SCTDedicatedHosts
from sdcm.utils.decorators import retrying
from sdcm.utils.aws_utils import tags_as_ec2_tags
from sdcm.test_config import TestConfig
from sdcm.utils.common import list_placement_groups_aws
LOGGER = logging.getLogger(__name__)

STATUS_FULFILLED = 'fulfilled'
SPOT_STATUS_UNEXPECTED_ERROR = 'error'
SPOT_PRICE_TOO_LOW = 'price-too-low'
FLEET_LIMIT_EXCEEDED_ERROR = 'spotInstanceCountLimitExceeded'
SPOT_CAPACITY_NOT_AVAILABLE_ERROR = 'capacity-not-available'
MAX_SPOT_EXCEEDED_ERROR = 'MaxSpotInstanceCountExceeded'
REQUEST_TIMEOUT = 300


class GetSpotPriceHistoryError(Exception):
    pass


class CreateSpotInstancesError(Exception):
    pass


class GetInstanceByPrivateIpError(Exception):
    pass


class GetPlacementGroupError(Exception):
    pass


class CreateEC2ClientNoRegionError(NoRegionError):
    pass


class CreateSpotFleetError(ClientError):
    pass


class EC2ClientWrapper():

    def __init__(self, timeout=REQUEST_TIMEOUT, region_name=None, spot_max_price_percentage=None):
        self._client = self._get_ec2_client(region_name)
        self._resource: EC2ServiceResource = boto3.resource('ec2', region_name=region_name)
        self.region_name = region_name
        self._timeout = timeout  # request timeout in seconds
        self._price_index = 1.5
        self._wait_interval = 5  # seconds
        self.spot_max_price_percentage = spot_max_price_percentage

    def _get_ec2_client(self, region_name=None) -> EC2Client:
        try:
            return boto3.client(service_name='ec2', region_name=region_name)
        except NoRegionError as err:
            if not region_name:
                raise CreateEC2ClientNoRegionError() from err
            # next class instance could be created without region_name parameter
            boto3.setup_default_session(region_name=region_name)
            return self._get_ec2_client()

    def _request_spot_instance(self, instance_type, image_id, region_name, network_if, spot_price, key_pair='',  # pylint: disable=too-many-arguments  # noqa: PLR0913
                               user_data='', count=1, duration=0, request_type='one-time', block_device_mappings=None,
                               aws_instance_profile=None, placement_group_name=None):
        """
        Create a spot instance request
        :return: list of request id-s
        """

        # pylint: disable=too-many-locals
        params = dict(DryRun=False,
                      InstanceCount=count,
                      Type=request_type,
                      SpotPrice=str(spot_price),
                      LaunchSpecification={'ImageId': image_id,
                                           'InstanceType': instance_type,
                                           'NetworkInterfaces': network_if,
                                           },
                      ValidUntil=datetime.datetime.now() + datetime.timedelta(minutes=self._timeout/60 + 5)
                      )
        self.add_placement_group_name_param(params['LaunchSpecification'], placement_group_name)
        if aws_instance_profile:
            params['LaunchSpecification']['IamInstanceProfile'] = {'Name': aws_instance_profile}
        LOGGER.debug("block_device_mappings: %s", block_device_mappings)
        if block_device_mappings:
            params['LaunchSpecification']['BlockDeviceMappings'] = block_device_mappings
        if not duration:
            params.update({'AvailabilityZoneGroup': region_name})
        if key_pair:
            params['LaunchSpecification'].update({'KeyName': key_pair})
        if user_data:
            params['LaunchSpecification'].update({'UserData': self._encode_user_data(user_data)})

        LOGGER.info('Sending spot request with params: %s', params)
        resp = self._client.request_spot_instances(**params)

        request_ids = [req['SpotInstanceRequestId'] for req in resp['SpotInstanceRequests']]
        LOGGER.debug('Spot requests: %s', request_ids)
        return request_ids

    def _request_spot_fleet(self, instance_type, image_id, region_name, network_if, key_pair='', user_data='', count=3,  # pylint: disable=too-many-arguments
                            block_device_mappings=None, aws_instance_profile=None, placement_group_name=None):

        spot_price = self._get_spot_price(instance_type)
        fleet_config = {'LaunchSpecifications':
                        [
                            {'ImageId': image_id,
                             'InstanceType': instance_type,
                             'NetworkInterfaces': network_if,
                             'Placement': {'AvailabilityZone': region_name},
                             },
                        ],
                        'IamFleetRole': 'arn:aws:iam::797456418907:role/aws-ec2-spot-fleet-role',
                        'SpotPrice': str(spot_price['desired']),
                        'TargetCapacity': count,
                        }
        self.add_placement_group_name_param(fleet_config['LaunchSpecifications'][0], placement_group_name)
        if aws_instance_profile:
            fleet_config['LaunchSpecifications'][0]['IamInstanceProfile'] = {'Name': aws_instance_profile}
        if key_pair:
            fleet_config['LaunchSpecifications'][0].update({'KeyName': key_pair})
        if user_data:
            fleet_config['LaunchSpecifications'][0].update({'UserData': self._encode_user_data(user_data)})
        if block_device_mappings:
            fleet_config['LaunchSpecifications'][0]['BlockDeviceMappings'] = block_device_mappings
        LOGGER.info('Sending spot fleet request with params: %s', fleet_config)
        resp = self._client.request_spot_fleet(DryRun=False,
                                               SpotFleetRequestConfig=fleet_config)

        request_id = resp['SpotFleetRequestId']
        LOGGER.debug('Spot fleet request: %s', request_id)
        return request_id

    def _get_spot_price(self, instance_type):
        """
        Calculate spot price for bidding
        :return: spot bid price
        """
        LOGGER.info('Calculating spot price based on OnDemand price')
        from sdcm.utils.pricing import AWSPricing  # pylint: disable=import-outside-toplevel
        aws_pricing = AWSPricing()
        on_demand_price = float(aws_pricing.get_on_demand_instance_price(self.region_name, instance_type))

        price = dict(max=on_demand_price, desired=on_demand_price * self.spot_max_price_percentage)
        LOGGER.info('Spot bid price: %s', price)
        return price

    def _is_request_fulfilled(self, request_ids):
        """
        Check request status
        """
        resp = self._client.describe_spot_instance_requests(SpotInstanceRequestIds=request_ids)
        for req in resp['SpotInstanceRequests']:
            if req['Status']['Code'] != STATUS_FULFILLED or req['State'] != 'active':
                if req['Status']['Code'] in [SPOT_PRICE_TOO_LOW, SPOT_CAPACITY_NOT_AVAILABLE_ERROR]:
                    return False, req['Status']['Code']
                return False, resp
        return True, resp

    def _wait_for_request_done(self, request_ids):
        """
        Wait for spot requests fulfilled
        :param request_ids: spot request id-s
        :return: list of spot instance id-s
        """
        LOGGER.info('Waiting for spot instances...')
        timeout = 0
        status = False
        while not status and timeout < self._timeout:
            time.sleep(self._wait_interval)
            status, resp = self._is_request_fulfilled(request_ids)
            LOGGER.debug("%s: [%s] - %s", request_ids, status, resp)
            if not status and resp in [SPOT_PRICE_TOO_LOW, SPOT_CAPACITY_NOT_AVAILABLE_ERROR]:
                break
            timeout += self._wait_interval
        if not status:
            self._client.cancel_spot_instance_requests(SpotInstanceRequestIds=request_ids)
            return [], resp
        return [req['InstanceId'] for req in resp['SpotInstanceRequests']], resp

    def _is_fleet_request_fulfilled(self, request_id):
        """
        Check fleet request status
        """
        resp = self._client.describe_spot_fleet_requests(SpotFleetRequestIds=[request_id])
        for req in resp['SpotFleetRequestConfigs']:
            if req['SpotFleetRequestState'] != 'active' or 'ActivityStatus' not in req or\
                    req['ActivityStatus'] != STATUS_FULFILLED:
                if 'ActivityStatus' in req and req['ActivityStatus'] == SPOT_STATUS_UNEXPECTED_ERROR:
                    current_time = datetime.datetime.now().timetuple()
                    search_start_time = datetime.datetime(
                        current_time.tm_year, current_time.tm_mon, current_time.tm_mday)
                    resp = self._client.describe_spot_fleet_request_history(SpotFleetRequestId=request_id,
                                                                            StartTime=search_start_time,
                                                                            MaxResults=10)
                    LOGGER.debug('Fleet request error history: %s', resp)
                    errors = [i['EventInformation']['EventSubType'] for i in resp['HistoryRecords']]
                    for error in [FLEET_LIMIT_EXCEEDED_ERROR, SPOT_CAPACITY_NOT_AVAILABLE_ERROR]:
                        if error in errors:
                            return False, error
                return False, resp
        return True, resp

    def _wait_for_fleet_request_done(self, request_id):
        """
        Wait for spot fleet request fulfilled
        :param request_id: spot fleet request id
        :return: list of spot instance id-s
        """
        LOGGER.info('Waiting for spot fleet...')
        timeout = 0
        status = False
        while not status and timeout < self._timeout:
            time.sleep(self._wait_interval)
            status, resp = self._is_fleet_request_fulfilled(request_id)
            if not status and resp in [FLEET_LIMIT_EXCEEDED_ERROR, SPOT_CAPACITY_NOT_AVAILABLE_ERROR]:
                break
            timeout += self._wait_interval
        if not status:
            self._client.cancel_spot_fleet_requests(SpotFleetRequestIds=[request_id], TerminateInstances=True)
            return [], resp
        resp = self._client.describe_spot_fleet_instances(SpotFleetRequestId=request_id)
        return [inst['InstanceId'] for inst in resp['ActiveInstances']], resp

    @staticmethod
    def _encode_user_data(user_data):
        return base64.b64encode(user_data.encode('utf-8')).decode("ascii")

    def get_instance(self, instance_id: str) -> Instance:
        """
        Get instance object by id
        :param instance_id: instance id
        :return: EC2.Instance object
        """
        instance = self._resource.Instance(id=instance_id)
        return instance

    @retrying(n=5, sleep_time=10, allowed_exceptions=(ClientError,),
              message="Waiting for instance is available")
    def add_tags(self, instance_ids: list[str] | list[Instance] | str | Instance, tags: dict = None) -> None:
        """
        Add tags to instances
        """
        if not isinstance(instance_ids, list):
            instance_ids = [instance_ids]
        instance_ids = [instance_or_id if isinstance(instance_or_id, str) else instance_or_id.instance_id for
                        instance_or_id in instance_ids]
        tags = tags_as_ec2_tags(tags) if tags else []
        tags += tags_as_ec2_tags(TestConfig().common_tags())
        self._client.create_tags(Resources=instance_ids, Tags=tags)

    def create_spot_instances(self, instance_type, image_id, region_name, network_if, key_pair='', user_data='',  # pylint: disable=too-many-arguments
                              count=1, duration=0, block_device_mappings=None, aws_instance_profile=None, placement_group_name=None):
        """
        Create spot instances

        :param instance_type: instance type
        :param image_id: image id
        :param region_name: availability zone name
        :param network_if: network interfaces
        :param key_pair: user credentials
        :param user_data: user data to be passed to instances
        :param count: number of instances to launch
        :param duration: (optional) instance life time in minutes(multiple of 60)
        :param aws_instance_profile: instance profile granting access to S3 objects
        :param placement_group_name: to create instances in the placement group

        :return: list of instance id-s
        """

        # pylint: disable=too-many-locals

        spot_price = self._get_spot_price(instance_type)

        request_ids = self._request_spot_instance(instance_type, image_id, region_name, network_if, spot_price['desired'],
                                                  key_pair, user_data, count, duration,
                                                  block_device_mappings=block_device_mappings,
                                                  aws_instance_profile=aws_instance_profile,
                                                  placement_group_name=placement_group_name)
        instance_ids, resp = self._wait_for_request_done(request_ids)

        if not instance_ids:
            raise CreateSpotInstancesError("Failed to get spot instances: %s" % resp)

        LOGGER.info('Spot instances: %s', instance_ids)
        for ind, instance_id in enumerate(instance_ids):
            self.add_tags(instance_id, {'Name': 'spot_{}_{}'.format(instance_id, ind)})

        self._client.cancel_spot_instance_requests(SpotInstanceRequestIds=request_ids)

        instances = [self.get_instance(instance_id) for instance_id in instance_ids]
        return instances

    def create_spot_fleet(self, instance_type, image_id, region_name, network_if, key_pair='', user_data='', count=3,  # pylint: disable=too-many-arguments
                          block_device_mappings=None, aws_instance_profile=None, placement_group_name=None):
        """
        Create spot fleet
        :param instance_type: instance type
        :param image_id: image id
        :param region_name: availability zone name
        :param network_if: network interfaces
        :param key_pair: user credentials
        :param user_data: user data to be passed to instances
        :param count: number of instances to launch
        :param block_device_mappings:
        :param aws_instance_profile: instance profile granting access to S3 objects
        :param placement_group_name: to create instances in the placement group

        :return: list of instance id-s
        """
        # pylint: disable=too-many-locals

        request_id = self._request_spot_fleet(instance_type, image_id, region_name, network_if, key_pair,
                                              user_data, count, block_device_mappings=block_device_mappings,
                                              aws_instance_profile=aws_instance_profile,
                                              placement_group_name=placement_group_name)
        instance_ids, resp = self._wait_for_fleet_request_done(request_id)
        if not instance_ids:
            err_code = resp if resp in [FLEET_LIMIT_EXCEEDED_ERROR,
                                        SPOT_CAPACITY_NOT_AVAILABLE_ERROR] else SPOT_STATUS_UNEXPECTED_ERROR
            raise CreateSpotFleetError(error_response={'Error': {'Code': err_code, 'Message': resp}},
                                       operation_name='create_spot_fleet')

        LOGGER.info('Spot instances: %s', instance_ids)
        for ind, instance_id in enumerate(instance_ids):
            self.add_tags(instance_id, {'Name': 'spot_fleet_{}_{}'.format(instance_id, ind)})

        self._client.cancel_spot_fleet_requests(SpotFleetRequestIds=[request_id], TerminateInstances=False)

        instances = [self.get_instance(instance_id) for instance_id in instance_ids]
        return instances

    def terminate_instances(self, instance_ids):
        """
        Terminate instances
        """
        self._client.terminate_instances(InstanceIds=instance_ids)

    def get_subnet_info(self, subnet_id):
        resp = self._client.describe_subnets(SubnetIds=[subnet_id])
        return [subnet for subnet in resp['Subnets'] if subnet['SubnetId'] == subnet_id][0]

    def get_instance_by_private_ip(self, private_ip):
        """
        Find instance by private IP address
        :param private_ip: private IP address
        :return: EC2.Instance object
        """
        instances = self._client.describe_instances(Filters=[
            {'Name': 'private-ip-address',
             'Values': [private_ip]}
        ])
        if not instances['Reservations']:
            raise GetInstanceByPrivateIpError("Cannot find instance by private ip: %s" % private_ip)

        return self.get_instance(instances['Reservations'][0]['Instances'][0]['InstanceId'])

    def add_placement_group_name_param(self, boto3_params, placement_group_name):
        if placement_group_name:
            placement_group_tags = {"Name": placement_group_name}
            list_placement_groups = list_placement_groups_aws(
                placement_group_tags, self.region_name)
            if list_placement_groups:
                if len(list_placement_groups) == 1:
                    LOGGER.info('use placement group name: %s', placement_group_name)
                    if "Placement" not in boto3_params:
                        boto3_params['Placement'] = {}
                    boto3_params['Placement']['GroupName'] = placement_group_name
                else:
                    error_message = f"more than one placement group with tags {placement_group_tags} found: \n {list_placement_groups}"
                    raise GetPlacementGroupError(error_message)
            else:
                error_message = f"use_placement_group param is true but no placement group with tags {placement_group_tags} found"
                raise GetPlacementGroupError(error_message)

    @staticmethod
    def add_capacity_reservation_param(boto3_params, availability_zone):
        if cr_id := SCTCapacityReservation.reservations.get(availability_zone).get(boto3_params["InstanceType"]):
            boto3_params['CapacityReservationSpecification'] = {
                'CapacityReservationTarget': {
                    'CapacityReservationId': cr_id
                }
            }

    @staticmethod
    def add_host_id_param(boto3_params, availability_zone):
        if host_id := SCTDedicatedHosts.get_host(availability_zone, boto3_params["InstanceType"]):
            boto3_params['Placement'] = {
                **boto3_params.get('Placement', {}),
                'HostId': host_id
            }
