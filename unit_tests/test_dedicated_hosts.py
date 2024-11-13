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
# Copyright (c) 2024 ScyllaDB

from unittest.mock import MagicMock, patch
import pytest

from sdcm.provision.aws.dedicated_host import SCTDedicatedHosts


@pytest.fixture(autouse=True)
def reset_hosts():
    SCTDedicatedHosts.hosts = {}


@pytest.fixture(name="params")
def params_fixture():
    class DotDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    return DotDict({"test_id": "example-test-id", "region_name": ["us-east-1"], "use_dedicated_host": True,
                    "instance_provision": "on_demand",
                    "availability_zone": "d",
                    "cluster_backend": "aws", "region_names": ["us-east-1"],
                    "n_db_nodes": 1, "instance_type_db": "i4i.4xlarge", "cluster_target_size": 1, "n_loaders": 1,
                    "instance_type_loader": "t3.small", "n_monitor_nodes": 1, "instance_type_monitor": "t3.small", "test_duration": 180})


@pytest.fixture(name="mock_ec2")
def mock_ec2_fixture():
    with patch('boto3.client') as mock_boto3_client:
        mock_ec2 = MagicMock()
        mock_boto3_client.return_value = mock_ec2
        mock_ec2.describe_hosts.side_effect = [{'Hosts': []}]
        yield mock_ec2


def test_allocate_success_existing(mock_ec2, params):
    mock_ec2.describe_hosts.side_effect = [
        {'Hosts': [{'AutoPlacement': 'off', 'AvailabilityZone': 'eu-west-1a', 'AvailableCapacity': {'AvailableInstanceCapacity': [
            {'AvailableCapacity': 64, 'InstanceType': 'i4i.large', 'TotalCapacity': 64}], 'AvailableVCpus': 128},
            'HostId': 'h-0af5175ea085157f0', 'HostProperties': {'Cores': 64, 'InstanceType': 'i4i.large', 'Sockets': 2, 'TotalVCpus': 128}, 'Instances': [], 'State': 'available', 'Tags': [{'Key': 'TestId', 'Value': 'None'}, {'Key': 'TestName', 'Value': 'None'}, {'Key': 'CreatedBy', 'Value': 'SCT'}, {'Key': 'version', 'Value': 'unknown'}, {'Key': 'RunByUser', 'Value': 'fruch'}, {'Key': 'test_id', 'Value': '1234-1234-1234'}], 'HostRecovery': 'off', 'AllowsMultipleInstanceTypes': 'off', 'OwnerId': '797456418907', 'AvailabilityZoneId': 'euw1-az3', 'MemberOfServiceLinkedResourceGroup': False, 'HostMaintenance': 'off'}]}]

    SCTDedicatedHosts.reserve(params)
    hosts = SCTDedicatedHosts.hosts
    assert len(hosts.keys()) == 1
    for key, value in hosts.items():
        assert key == "eu-west-1a"  # initial availability zone
        assert value == {"i4i.large": "h-0af5175ea085157f0"}


def test_allocate_success(mock_ec2, params):
    mock_ec2.allocate_hosts.side_effect = [{'HostIds': ['h-0867aada366f84ff1']}]

    SCTDedicatedHosts.reserve(params)
    hosts = SCTDedicatedHosts.hosts
    assert len(hosts.keys()) == 1
    for key, value in hosts.items():
        assert key == "us-east-1d"  # initial availability zone
        assert value == {"i4i.4xlarge": "h-0867aada366f84ff1"}


def test_release_success(mock_ec2, params):
    SCTDedicatedHosts.hosts['us-east-1d'] = {"i4i.4xlarge": "h-0867aada366f84ff1"}
    mock_ec2.release_hosts.side_effect = [{'Successful': ['h-0867aada366f84ff1']}]

    SCTDedicatedHosts.release(params)
    mock_ec2.release_hosts.assert_called_once_with(HostIds=['h-0867aada366f84ff1'])


def test_release_retries(mock_ec2, params):
    SCTDedicatedHosts.hosts['us-east-1d'] = {"i4i.4xlarge": "h-0867aada366f84ff1"}
    mock_ec2.release_hosts.side_effect = [
        {'Unsuccessful': [{'Error': {'Code': '000', 'Message': 'error'}}]}, {'Successful': ['h-0867aada366f84ff1']}]
    with patch("tenacity.nap.time.sleep"):
        SCTDedicatedHosts.release(params)
    mock_ec2.release_hosts.assert_called_with(HostIds=['h-0867aada366f84ff1'])
    assert mock_ec2.release_hosts.call_count == 2


def test_release_retries_fails(mock_ec2, params):
    SCTDedicatedHosts.hosts['us-east-1d'] = {"i4i.4xlarge": "h-0867aada366f84ff1"}
    mock_ec2.release_hosts.side_effect = [{'Unsuccessful': [{'Error': {'Code': '000', 'Message': 'error'}}]}] * 20

    with pytest.raises(TimeoutError), patch("tenacity.nap.time.sleep"):
        SCTDedicatedHosts.release(params)
    mock_ec2.release_hosts.assert_called_with(HostIds=['h-0867aada366f84ff1'])
    assert mock_ec2.release_hosts.call_count == 10


def test_release_by_tag_unit_unit(mock_ec2):
    mock_ec2.release_hosts.side_effect = [
        {'Unsuccessful': [{'Error': {'Code': '000', 'Message': 'error'}}]}, {'Successful': ['h-0867aada366f84ff1']}]
    mock_ec2.describe_hosts.side_effect = [
        {'Hosts': [{'HostId': 'h-0af5175ea085157f0'}]}]

    SCTDedicatedHosts.release_by_tags({"TestId": "no-existing"}, regions=['us-east-1'], dry_run=False)

    mock_ec2.release_hosts.assert_called_with(HostIds=['h-0af5175ea085157f0'])
    assert mock_ec2.release_hosts.call_count == 2


@pytest.mark.integration
def test_release_by_tag_integration():
    SCTDedicatedHosts.release_by_tags({"TestId": "no-existing"}, regions=['us-east-1', 'eu-west-1'], dry_run=True)
