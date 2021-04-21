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

# pylint: disable=too-many-lines

from __future__ import absolute_import

import atexit
import itertools
import os
import logging
import random
import re
import socket
import time
import datetime
import errno
import threading
import select
import shutil
import copy
import string
import warnings
import getpass
from typing import Iterable, List, Callable, Optional, Dict, Union
from urllib.parse import urlparse
from unittest.mock import Mock
from textwrap import dedent

from functools import wraps
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures.thread import _python_exit
import hashlib

import boto3
from mypy_boto3_s3 import S3Client, S3ServiceResource
from mypy_boto3_ec2 import EC2Client, EC2ServiceResource
import docker  # pylint: disable=wrong-import-order; false warning because of docker import (local file vs. package)
import libcloud.storage.providers
import libcloud.storage.types
import yaml
from libcloud.compute.providers import get_driver
from libcloud.compute.types import Provider
from packaging.version import Version

from sdcm.utils.ssh_agent import SSHAgent
from sdcm.utils.decorators import retrying
from sdcm import wait
from sdcm.utils.ldap import LDAP_PASSWORD, LDAP_USERS


LOGGER = logging.getLogger('utils')
DEFAULT_AWS_REGION = "eu-west-1"
DOCKER_CGROUP_RE = re.compile("/docker/([0-9a-f]+)")
SCYLLA_AMI_OWNER_ID = "797456418907"


def deprecation(message):
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def _remote_get_hash(remoter, file_path):
    try:
        result = remoter.run('md5sum {}'.format(file_path), verbose=True)
        return result.stdout.strip().split()[0]
    except Exception as details:  # pylint: disable=broad-except
        LOGGER.error(str(details))
        return None


def _remote_get_file(remoter, src, dst, user_agent=None):
    cmd = 'curl -L {} -o {}'.format(src, dst)
    if user_agent:
        cmd += ' --user-agent %s' % user_agent
    return remoter.run(cmd, ignore_status=True)


def remote_get_file(remoter, src, dst, hash_expected=None, retries=1, user_agent=None):  # pylint: disable=too-many-arguments
    _remote_get_file(remoter, src, dst, user_agent)
    if not hash_expected:
        return
    while retries > 0 and _remote_get_hash(remoter, dst) != hash_expected:
        _remote_get_file(remoter, src, dst, user_agent)
        retries -= 1
    assert _remote_get_hash(remoter, dst) == hash_expected


def get_profile_content(stress_cmd):
    """
    Looking profile yaml in data_dir or the path as is to get the user profile
    and loading it's yaml

    :return: (profile_filename, dict with yaml)
    """

    cs_profile = re.search(r'profile=(.*\.yaml)', stress_cmd).group(1)
    sct_cs_profile = os.path.join(os.path.dirname(__file__), '../../', 'data_dir', os.path.basename(cs_profile))
    if os.path.exists(sct_cs_profile):
        cs_profile = sct_cs_profile
    elif not os.path.exists(cs_profile):
        raise FileNotFoundError('User profile file {} not found'.format(cs_profile))

    with open(cs_profile, 'r') as yaml_stream:
        profile = yaml.safe_load(yaml_stream)
    return cs_profile, profile


def generate_random_string(length):
    return random.choice(string.ascii_uppercase) + ''.join(
        random.choice(string.ascii_uppercase + string.digits) for x in range(length - 1))


def get_data_dir_path(*args):
    sct_root_path = get_sct_root_path()
    data_dir = os.path.join(sct_root_path, "data_dir", *args)
    return os.path.abspath(data_dir)


def get_sct_root_path():
    import sdcm  # pylint: disable=import-outside-toplevel
    sdcm_path = os.path.realpath(sdcm.__path__[0])
    sct_root_dir = os.path.join(sdcm_path, "..")
    return os.path.abspath(sct_root_dir)


def get_job_name():
    return os.environ.get('JOB_NAME', 'local_run')


def verify_scylla_repo_file(content, is_rhel_like=True):
    LOGGER.info('Verifying Scylla repo file')
    if is_rhel_like:
        body_prefix = ['#', '[scylla', 'name=', 'baseurl=', 'enabled=', 'gpgcheck=', 'type=',
                       'skip_if_unavailable=', 'gpgkey=', 'repo_gpgcheck=', 'enabled_metadata=']
    else:
        body_prefix = ['#', 'deb']
    for line in content.split('\n'):
        valid_prefix = False
        for prefix in body_prefix:
            if line.startswith(prefix) or not line.strip():
                valid_prefix = True
                break
        LOGGER.debug(line)
        assert valid_prefix, 'Repository content has invalid line: {}'.format(line)


def remove_comments(data):
    """Remove comments line from data

    Remove any string which is start from # in data

    Arguments:
        data {str} -- data expected the command output, file contents
    """
    return '\n'.join([i.strip() for i in data.split('\n') if not i.startswith('#')])


class S3Storage():

    bucket_name = 'cloudius-jenkins-test'
    enable_multipart_threshold_size = 1024 * 1024 * 1024  # 1GB
    multipart_chunksize = 50 * 1024 * 1024  # 50 MB
    num_download_attempts = 5

    def __init__(self, bucket=None):
        if bucket:
            self.bucket_name = bucket
        self._bucket: S3ServiceResource.Bucket = boto3.resource("s3").Bucket(name=self.bucket_name)
        self.transfer_config = boto3.s3.transfer.TransferConfig(multipart_threshold=self.enable_multipart_threshold_size,
                                                                multipart_chunksize=self.multipart_chunksize,
                                                                num_download_attempts=self.num_download_attempts)

    def get_s3_fileojb(self, key):
        objects = []
        for obj in self._bucket.objects.filter(Prefix=key):
            objects.append(obj)
        return objects

    def search_by_path(self, path=''):
        files = []
        for obj in self._bucket.objects.filter(Prefix=path):
            files.append(obj.key)
        return files

    def generate_url(self, file_path, dest_dir=''):
        bucket_name = self.bucket_name
        file_name = os.path.basename(os.path.normpath(file_path))
        return "https://{bucket_name}.s3.amazonaws.com/{dest_dir}/{file_name}".format(dest_dir=dest_dir,
                                                                                      file_name=file_name,
                                                                                      bucket_name=bucket_name)

    def upload_file(self, file_path, dest_dir=''):
        s3_url = self.generate_url(file_path, dest_dir)
        s3_obj = "{}/{}".format(dest_dir, os.path.basename(file_path))
        try:
            LOGGER.info("Uploading '{file_path}' to {s3_url}".format(file_path=file_path, s3_url=s3_url))
            self._bucket.upload_file(Filename=file_path,
                                     Key=s3_obj,
                                     Config=self.transfer_config)
            LOGGER.info("Uploaded to {0}".format(s3_url))
            LOGGER.info("Set public read access")
            self.set_public_access(key=s3_obj)
            return s3_url
        except Exception as details:  # pylint: disable=broad-except
            LOGGER.debug("Unable to upload to S3: %s", details)
            return ""

    def set_public_access(self, key):
        acl_obj: S3ServiceResource = boto3.resource('s3').ObjectAcl(self.bucket_name, key)

        grants = copy.deepcopy(acl_obj.grants)
        grantees = {
            'Grantee': {
                "Type": "Group",
                "URI": "http://acs.amazonaws.com/groups/global/AllUsers"
            },
            'Permission': "READ"
        }
        grants.append(grantees)
        acl_obj.put(ACL='', AccessControlPolicy={'Grants': grants, 'Owner': acl_obj.owner})

    def download_file(self, link, dst_dir):
        key_name = link.replace("https://{0.bucket_name}.s3.amazonaws.com/".format(self), "")
        file_name = os.path.basename(key_name)
        try:
            LOGGER.info("Downloading {0} from {1}".format(key_name, self.bucket_name))
            self._bucket.download_file(Key=key_name,
                                       Filename=os.path.join(dst_dir, file_name),
                                       Config=self.transfer_config)
            LOGGER.info("Downloaded finished")
            return os.path.join(os.path.abspath(dst_dir), file_name)

        except Exception as details:  # pylint: disable=broad-except
            LOGGER.warning("File {} is not downloaded by reason: {}".format(key_name, details))
            return ""


def get_latest_gemini_version():
    bucket_name = 'downloads.scylladb.com'

    results = S3Storage(bucket_name).search_by_path(path='gemini')
    versions = {Version(result_file.split('/')[1]) for result_file in results}
    latest_version = max(versions)

    return latest_version.public


def list_logs_by_test_id(test_id):
    log_types = ['db-cluster', 'monitor-set', 'loader-set', 'sct-runner',
                 'prometheus', 'grafana',
                 'job', 'monitoring_data_stack', 'events']
    results = []

    if not test_id:
        return results

    def convert_to_date(date_str):
        time_formats = [
            "%Y%m%d_%H%M%S",
            "%Y_%m_%d_%H_%M_%S",
        ]
        for time_format in time_formats:
            try:
                return datetime.datetime.strptime(date_str, time_format)
            except ValueError:
                continue
        # for old data collected or uploaded without datetime,
        # return the old date to display them earlier
        return datetime.datetime(2019, 1, 1, 1, 1, 1)

    log_files = S3Storage().search_by_path(path=test_id)
    for log_file in log_files:
        for log_type in log_types:
            if log_type in log_file:
                results.append({"file_path": log_file,
                                "type": log_type,
                                "link": "https://{}.s3.amazonaws.com/{}".format(S3Storage.bucket_name, log_file),
                                "date": convert_to_date(log_file.split('/')[1])
                                })
                break
    results = sorted(results, key=lambda x: x["date"])

    return results


def all_aws_regions(cached=False):
    if cached:
        return [
            'eu-north-1',
            'ap-south-1',
            'eu-west-3',
            'eu-west-2',
            'eu-west-1',
            'ap-northeast-2',
            'ap-northeast-1',
            'sa-east-1',
            'ca-central-1',
            'ap-southeast-1',
            'ap-southeast-2',
            'eu-central-1',
            'us-east-1',
            'us-east-2',
            'us-west-1',
            'us-west-2'
        ]
    else:
        client: EC2Client = boto3.client('ec2', region_name=DEFAULT_AWS_REGION)
        return [region['RegionName'] for region in client.describe_regions()['Regions']]


class ParallelObject:
    """
        Run function in with supplied args in parallel using thread.
    """

    def __init__(self, objects: Iterable, timeout: int = 6,  # pylint: disable=redefined-outer-name
                 num_workers: int = None, disable_logging: bool = False):
        """Constructor for ParallelObject

        Build instances of Parallel object. Item of objects is used as parameter for
        func which will be run in parallel.

        :param objects: if item in object is list, it will be upacked to func argument, ex *arg
                if item in object is dict, it will be upacked to func keyword argument, ex **kwarg
                if item in object is any other type, will be passed to func as is.
                if function accept list as parameter, the item shuld be list of list item = [[]]

        :param timeout: global timeout for running all
        :param num_workers: num of parallel threads, defaults to None
        :param disable_logging: disable logging for running func, defaults to False
        """
        self.objects = objects
        self.timeout = timeout
        self.num_workers = num_workers
        self.disable_logging = disable_logging
        self._thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)

    def run(self, func: Callable, ignore_exceptions=False, unpack_objects: bool = False):
        """Run callable object "func" in parallel

        Allow to run callable object in parallel.
        if ignore_exceptions is true,  return
        list of FutureResult object instances which contains
        two attributes:
            - result - result of callable object execution
            - exc - exception object, if happened during run
        if ignore_exceptions is False, then running will
        terminated on future where happened exception or by timeout
        what has stepped first.

        :param func: Callable object to run in parallel
        :param ignore_exceptions: ignore exception and return result, defaults to False
        :param unpack_objects: set to True when unpacking of objects to the func as args or kwargs needed
        :returns: list of FutureResult object
        :rtype: {List[FutureResult]}
        """

        def func_wrap(fun):
            @wraps(fun)
            def inner(*args, **kwargs):
                thread_name = threading.current_thread().name
                fun_args = args
                fun_kwargs = kwargs
                fun_name = fun.__name__
                LOGGER.debug("[{thread_name}] {fun_name}({fun_args}, {fun_kwargs})".format(thread_name=thread_name,
                                                                                           fun_name=fun_name,
                                                                                           fun_args=fun_args,
                                                                                           fun_kwargs=fun_kwargs))
                return_val = fun(*args, **kwargs)
                LOGGER.debug("[{thread_name}] Done.".format(thread_name=thread_name))
                return return_val
            return inner

        results = []

        if not self.disable_logging:
            LOGGER.debug("Executing in parallel: '{}' on {}".format(func.__name__, self.objects))
            func = func_wrap(func)

        futures = []

        for obj in self.objects:
            if unpack_objects and isinstance(obj, (list, tuple)):
                futures.append(self._thread_pool.submit(func, *obj))
            elif unpack_objects and isinstance(obj, dict):
                futures.append(self._thread_pool.submit(func, **obj))
            else:
                futures.append(self._thread_pool.submit(func, obj))
        time_out = self.timeout
        for obj_idx, future in enumerate(futures):
            try:
                result = future.result(time_out)
            except FuturesTimeoutError as exception:
                results.append(ParallelObjectResult(obj=self.objects[obj_idx], exc=exception, result=None))
                time_out = 0.001  # if there was a timeout on one of the futures there is no need to wait for all
            except Exception as exception:  # pylint: disable=broad-except
                results.append(ParallelObjectResult(obj=self.objects[obj_idx], exc=exception, result=None))
            else:
                results.append(ParallelObjectResult(obj=self.objects[obj_idx], exc=None, result=result))

        self.clean_up(futures)

        if ignore_exceptions:
            return results

        timed_out = [result for result in results if isinstance(result.exc, FuturesTimeoutError)]
        if timed_out:
            raise FuturesTimeoutError("when running on: %s" % [r.obj for r in results])
        runs_that_finished_with_exception = [res for res in results if res.exc]
        if runs_that_finished_with_exception:
            raise ParallelObjectException(results=results)
        return results

    def clean_up(self, futures):
        # if there are futures that didn't run  we cancel them
        for future in futures:
            future.cancel()
        self._thread_pool.shutdown(wait=False)
        # we need to unregister internal function that waits for all threads to finish when interpreter exits
        atexit.unregister(_python_exit)


class ParallelObjectResult:  # pylint: disable=too-few-public-methods
    """Object for result of future in ParallelObject

    Return as a result of ParallelObject.run method
    and contain result of func was run in parallel
    and exception if it happened during run.
    """

    def __init__(self, obj, result=None, exc=None):
        self.obj = obj
        self.result = result
        self.exc = exc


class ParallelObjectException(Exception):
    def __init__(self, results: List[ParallelObjectResult]):
        super(ParallelObjectException, self).__init__()
        self.results = results

    def __str__(self):
        ex_str = ""
        for res in self.results:
            if res.exc:
                ex_str += f"{res.obj}: {res.exc}"
        return ex_str


def clean_cloud_resources(tags_dict):
    """
    Remove all instances with specific tags from both AWS/GCE

    :param tags_dict: a dict of the tag to select the instances,e.x. {"TestId": "9bc6879f-b1ef-47e1-99ab-020810aedbcc"}
    :return: None
    """
    if "TestId" not in tags_dict and "RunByUser" not in tags_dict:
        LOGGER.error("Can't clean cloud resources, TestId or RunByUser is missing")
        return False
    clean_instances_aws(tags_dict)
    clean_elastic_ips_aws(tags_dict)
    clean_instances_gce(tags_dict)
    clean_resources_docker(tags_dict)
    return True


def docker_current_container_id() -> Optional[str]:
    with open("/proc/1/cgroup") as cgroup:
        for line in cgroup:
            match = DOCKER_CGROUP_RE.search(line)
            if match:
                return match.group(1)
    return None


def list_clients_docker(builder_name: Optional[str] = None, verbose: bool = False) -> Dict[str, docker.DockerClient]:
    log = LOGGER if verbose else Mock()
    docker_clients = {}

    def get_builder_docker_client(builder: Dict[str, str]) -> None:
        if not can_connect_to(builder["public_ip"], 22, timeout=5):
            log.error("%(name)s: can't establish connection to %(public_ip)s:22, port is closed", builder)
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client = docker.DockerClient(base_url=f"ssh://{builder['user']}@{builder['public_ip']}")
            client.ping()
            log.info("%(name)s: connected via SSH (%(user)s@%(public_ip)s)", builder)
        except:
            log.error("%(name)s: failed to connect to Docker via SSH", builder)
            raise
        docker_clients[builder["name"]] = client

    if builder_name is None or builder_name == "local":
        docker_clients["local"] = docker.from_env()

    if builder_name != "local" and getpass.getuser() != "jenkins":
        builders = [item["builder"] for item in list_builders(running=True)]
        if builder_name:
            builders = {builder_name: builders[builder_name], } if builder_name in builders else {}
        if builders:
            SSHAgent.start(verbose=verbose)
            SSHAgent.add_keys(set(b["key_file"] for b in builders), verbose)
            ParallelObject(builders, timeout=20).run(get_builder_docker_client, ignore_exceptions=True)
            log.info("%d builders from %d available to scan for Docker resources", len(docker_clients), len(builders))
        else:
            log.warning("No builders found")

    return docker_clients


def list_resources_docker(tags_dict: Optional[dict] = None,
                          builder_name: Optional[str] = None,
                          running: bool = False,
                          group_as_builder: bool = False,
                          verbose: bool = False) -> Dict[str, Union[list, dict]]:
    log = LOGGER if verbose else Mock()
    filters = {}

    current_container_id = docker_current_container_id()

    if tags_dict:
        filters["label"] = [f"{key}={value}" for key, value in tags_dict.items()]

    containers = {}
    images = {}

    def get_containers(builder_name: str, docker_client: docker.DockerClient) -> None:
        log.info("%s: scan for Docker containers", builder_name)
        containers_list = docker_client.containers.list(filters=filters, sparse=True)
        if current_container_id:
            containers_list = [container for container in containers_list if container.id != current_container_id]
        if running:
            containers_list = [container for container in containers_list if container.status == "running"]
        else:
            containers_list = [container for container in containers_list if container.status != "removing"]
        if containers_list:
            log.info("%s: found %d containers", builder_name, len(containers_list))
            containers[builder_name] = containers_list

    def get_images(builder_name: str, docker_client: docker.DockerClient) -> None:
        log.info("%s: scan for Docker images", builder_name)
        images_list = docker_client.images.list(filters=filters)
        if images_list:
            log.info("%s: found %s images", builder_name, len(images_list))
            images[builder_name] = images_list

    docker_clients = tuple(list_clients_docker(builder_name=builder_name, verbose=verbose).items())

    ParallelObject(docker_clients, timeout=30).run(get_containers, ignore_exceptions=True, unpack_objects=True)
    ParallelObject(docker_clients, timeout=30).run(get_images, ignore_exceptions=True, unpack_objects=True)

    if not group_as_builder:
        containers = list(itertools.chain.from_iterable(containers.values()))
        images = list(itertools.chain.from_iterable(images.values()))

    return dict(containers=containers, images=images)


def clean_resources_docker(tags_dict: dict, builder_name: Optional[str] = None) -> None:
    assert tags_dict, "tags_dict not provided (can't clean all instances)"

    def delete_container(container):
        container.reload()
        container.remove(v=True, force=True)
        LOGGER.info("Docker container `%s' on host `%s' deleted", container.name, container.client.info()["Name"])

    def delete_image(image):
        image.client.images.remove(image=image.id, force=True)
        LOGGER.info("Docker image tag(s) %s on host `%s' deleted", image.tags, image.client.info()["Name"])

    resources_to_clean = list_resources_docker(tags_dict, builder_name=builder_name, group_as_builder=False)
    containers = resources_to_clean.get("containers", [])
    images = resources_to_clean.get("images", [])

    for container in containers:
        try:
            delete_container(container)
        except Exception:  # pylint: disable=broad-except
            LOGGER.error("Failed to delete container `%s' on host `%s'",
                         container.name, container.client.info()["Name"])

    for image in images:
        try:
            delete_image(image)
        except Exception:  # pylint: disable=broad-except
            LOGGER.error("Failed to delete image tag(s) %s on host `%s'", image.tags, image.client.info()["Name"])


def aws_tags_to_dict(tags_list):
    tags_dict = {}
    if tags_list:
        for item in tags_list:
            tags_dict[item["Key"]] = item["Value"]
    return tags_dict


def list_instances_aws(tags_dict=None, region_name=None, running=False, group_as_region=False, verbose=False):
    """
    list all instances with specific tags AWS

    :param tags_dict: a dict of the tag to select the instances, e.x. {"TestId": "9bc6879f-b1ef-47e1-99ab-020810aedbcc"}
    :param region_name: name of the region to list
    :param running: get all running instances
    :param group_as_region: if True the results would be grouped into regions
    :param verbose: if True will log progress information

    :return: instances dict where region is a key
    """
    instances = {}
    aws_regions = [region_name] if region_name else all_aws_regions()

    def get_instances(region):
        if verbose:
            LOGGER.info('Going to list aws region "%s"', region)
        time.sleep(random.random())
        client: EC2Client = boto3.client('ec2', region_name=region)
        custom_filter = []
        if tags_dict:
            custom_filter = [{'Name': 'tag:{}'.format(key), 'Values': [value]} for key, value in tags_dict.items()]
        response = client.describe_instances(Filters=custom_filter)
        instances[region] = [instance for reservation in response['Reservations'] for instance in reservation[
            'Instances']]

        if verbose:
            LOGGER.info("%s: done [%s/%s]", region, len(list(instances.keys())), len(aws_regions))

    ParallelObject(aws_regions, timeout=100).run(get_instances, ignore_exceptions=True)

    for curr_region_name in instances:
        if running:
            instances[curr_region_name] = [i for i in instances[curr_region_name] if i['State']['Name'] == 'running']
        else:
            instances[curr_region_name] = [i for i in instances[curr_region_name]
                                           if not i['State']['Name'] == 'terminated']
    if not group_as_region:
        instances = list(itertools.chain(*list(instances.values())))  # flatten the list of lists
        total_items = len(instances)
    else:
        total_items = sum([len(value) for _, value in instances.items()])

    if verbose:
        LOGGER.info("Found total of {} instances.".format(total_items))

    return instances


def clean_instances_aws(tags_dict):
    """
    Remove all instances with specific tags AWS

    :param tags_dict: a dict of the tag to select the instances, e.x. {"TestId": "9bc6879f-b1ef-47e1-99ab-020810aedbcc"}
    :return: None
    """
    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    aws_instances = list_instances_aws(tags_dict=tags_dict, group_as_region=True)

    for region, instance_list in aws_instances.items():
        client: EC2Client = boto3.client('ec2', region_name=region)
        for instance in instance_list:
            tags = aws_tags_to_dict(instance.get('Tags'))
            name = tags.get("Name", "N/A")
            instance_id = instance['InstanceId']
            LOGGER.info("Going to delete '{instance_id}' [name={name}] ".format(instance_id=instance_id, name=name))
            response = client.terminate_instances(InstanceIds=[instance_id])
            LOGGER.debug("Done. Result: %s\n", response['TerminatingInstances'])


def list_elastic_ips_aws(tags_dict=None, region_name=None, group_as_region=False, verbose=False):
    """
    list all elastic ips with specific tags AWS

    :param tags_dict: a dict of the tag to select the instances, e.x. {"TestId": "9bc6879f-b1ef-47e1-99ab-020810aedbcc"}
    :param region_name: name of the region to list
    :param group_as_region: if True the results would be grouped into regions
    :param verbose: if True will log progress information

    :return: instances dict where region is a key
    """
    elastic_ips = {}
    aws_regions = [region_name] if region_name else all_aws_regions()

    def get_elastic_ips(region):
        if verbose:
            LOGGER.info('Going to list aws region "%s"', region)
        time.sleep(random.random())
        client: EC2Client = boto3.client('ec2', region_name=region)
        custom_filter = []
        if tags_dict:
            custom_filter = [{'Name': 'tag:{}'.format(key), 'Values': [value]} for key, value in tags_dict.items()]
        response = client.describe_addresses(Filters=custom_filter)
        elastic_ips[region] = response['Addresses']
        if verbose:
            LOGGER.info("%s: done [%s/%s]", region, len(list(elastic_ips.keys())), len(aws_regions))

    ParallelObject(aws_regions, timeout=100).run(get_elastic_ips, ignore_exceptions=True)

    if not group_as_region:
        elastic_ips = list(itertools.chain(*list(elastic_ips.values())))  # flatten the list of lists
        total_items = elastic_ips
    else:
        total_items = sum([len(value) for _, value in elastic_ips.items()])
    if verbose:
        LOGGER.info("Found total of %s ips.", total_items)
    return elastic_ips


def clean_elastic_ips_aws(tags_dict):
    """
    Remove all elastic ips with specific tags AWS

    :param tags_dict: a dict of the tag to select the instances, e.x. {"TestId": "9bc6879f-b1ef-47e1-99ab-020810aedbcc"}
    :return: None
    """
    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    aws_instances = list_elastic_ips_aws(tags_dict=tags_dict, group_as_region=True)

    for region, eip_list in aws_instances.items():
        client: EC2Client = boto3.client('ec2', region_name=region)
        for eip in eip_list:
            association_id = eip.get('AssociationId', None)
            if association_id:
                response = client.disassociate_address(AssociationId=association_id)
                LOGGER.debug("disassociate_address. Result: %s\n", response)

            allocation_id = eip['AllocationId']
            LOGGER.info("Going to release '{allocation_id}' [public_ip={public_ip}] ".format(
                allocation_id=allocation_id, public_ip=eip['PublicIp']))
            response = client.release_address(AllocationId=allocation_id)
            LOGGER.debug("Done. Result: %s\n", response)


def get_gce_driver():
    # avoid cyclic dependency issues, since too many things import utils.py
    from sdcm.keystore import KeyStore

    gcp_credentials = KeyStore().get_gcp_credentials()
    gce_driver = get_driver(Provider.GCE)

    return gce_driver(gcp_credentials["project_id"] + "@appspot.gserviceaccount.com",
                      gcp_credentials["private_key"], project=gcp_credentials["project_id"])


def get_all_gce_regions():

    compute_engine = get_gce_driver()
    all_gce_regions = [region_obj.name for region_obj in compute_engine.region_list]
    return all_gce_regions


def gce_meta_to_dict(metadata):
    meta_dict = {}
    data = metadata.get("items")
    if data:
        for item in data:
            key = item["key"]
            if key:  # sometimes key is empty string
                meta_dict[key] = item["value"]
    return meta_dict


def filter_gce_by_tags(tags_dict, instances):
    filtered_instances = []
    for instance in instances:
        tags = gce_meta_to_dict(instance.extra['metadata'])
        found_keys = set(k for k in tags_dict if k in tags and tags_dict[k] == tags[k])
        if found_keys == set(tags_dict.keys()):
            filtered_instances.append(instance)
    return filtered_instances


def list_instances_gce(tags_dict=None, running=False, verbose=False):
    """
    list all instances with specific tags GCE

    :param tags_dict: a dict of the tag to select the instances, e.x. {"TestId": "9bc6879f-b1ef-47e1-99ab-020810aedbcc"}

    :return: None
    """

    compute_engine = get_gce_driver()

    if verbose:
        LOGGER.info("Going to get all instances from GCE")
    all_gce_instances = compute_engine.list_nodes()
    # filter instances by tags since libcloud list_nodes() doesn't offer any filtering
    if tags_dict:
        instances = filter_gce_by_tags(tags_dict=tags_dict, instances=all_gce_instances)
    else:
        instances = all_gce_instances

    if running:
        # https://libcloud.readthedocs.io/en/latest/compute/api.html#libcloud.compute.types.NodeState
        instances = [i for i in instances if i.state == 'running']
    else:
        instances = [i for i in instances if not i.state == 'terminated']
    if verbose:
        LOGGER.info("Done. Found total of %s instances.", len(instances))
    return instances


def list_static_ips_gce(region_name="all", group_by_region=False, verbose=False):
    compute_engine = get_gce_driver()
    if verbose:
        LOGGER.info("Getting all GCE static IPs...")
    all_static_ips = compute_engine.ex_list_addresses(region_name)
    if verbose:
        LOGGER.info("Found total %s GCE static IPs.", len(all_static_ips))

    if group_by_region:
        ips_grouped_by_region = defaultdict(list)
        for ip in all_static_ips:
            ips_grouped_by_region[ip.region.name].append(ip)
        return ips_grouped_by_region
    return all_static_ips


def clean_instances_gce(tags_dict):
    """
    Remove all instances with specific tags GCE

    :param tags_dict: a dict of the tag to select the instances, e.x. {"TestId": "9bc6879f-b1ef-47e1-99ab-020810aedbcc"}
    :return: None
    """
    assert tags_dict, "tags_dict not provided (can't clean all instances)"
    gce_instances_to_clean = list_instances_gce(tags_dict=tags_dict)

    def delete_instance(instance):
        LOGGER.info("Going to delete: {}".format(instance.name))
        # https://libcloud.readthedocs.io/en/latest/compute/api.html#libcloud.compute.base.Node.destroy
        res = instance.destroy()
        LOGGER.info("{} deleted={}".format(instance.name, res))
    ParallelObject(gce_instances_to_clean, timeout=60).run(delete_instance, ignore_exceptions=True)


_SCYLLA_AMI_CACHE = defaultdict(dict)


def get_scylla_ami_versions(region):
    """
    get the list of all the formal scylla ami from specific region

    :param region: the aws region to look in
    :return: list of ami data
    :rtype: list
    """

    if _SCYLLA_AMI_CACHE[region]:
        return _SCYLLA_AMI_CACHE[region]

    client: EC2Client = boto3.client('ec2', region_name=region)
    response = client.describe_images(
        Owners=['797456418907'],  # ScyllaDB
        Filters=[
            {'Name': 'name', 'Values': ['ScyllaDB *']},
        ],
    )

    _SCYLLA_AMI_CACHE[region] = sorted(response['Images'],
                                       key=lambda x: x['CreationDate'],
                                       reverse=True)

    return _SCYLLA_AMI_CACHE[region]


_S3_SCYLLA_REPOS_CACHE = defaultdict(dict)


def get_s3_scylla_repos_mapping(dist_type='centos', dist_version=None):
    """
    get the mapping from version prefixes to rpm .repo or deb .list files locations

    :param dist_type: which distro to look up centos/ubuntu/debian
    :param dist_version: famaily name of the distro version

    :return: a mapping of versions prefixes to repos
    :rtype: dict
    """
    if (dist_type, dist_version) in _S3_SCYLLA_REPOS_CACHE:
        return _S3_SCYLLA_REPOS_CACHE[(dist_type, dist_version)]

    s3_client: S3Client = boto3.client('s3', region_name=DEFAULT_AWS_REGION)
    bucket = 'downloads.scylladb.com'

    if dist_type == 'centos':
        response = s3_client.list_objects(Bucket=bucket, Prefix='rpm/centos/', Delimiter='/')

        for repo_file in response['Contents']:
            filename = os.path.basename(repo_file['Key'])
            # only if path look like 'rpm/centos/scylla-1.3.repo', we deem it formal one
            if filename.startswith('scylla-') and filename.endswith('.repo'):
                version_prefix = filename.replace('.repo', '').split('-')[-1]
                _S3_SCYLLA_REPOS_CACHE[(
                    dist_type, dist_version)][version_prefix] = "https://s3.amazonaws.com/{bucket}/{path}".format(bucket=bucket, path=repo_file['Key'])

    elif dist_type in ('ubuntu', 'debian'):
        response = s3_client.list_objects(Bucket=bucket, Prefix='deb/{}/'.format(dist_type), Delimiter='/')
        for repo_file in response['Contents']:
            filename = os.path.basename(repo_file['Key'])

            # only if path look like 'deb/debian/scylla-3.0-jessie.list', we deem it formal one
            if filename.startswith('scylla-') and filename.endswith('-{}.list'.format(dist_version)):

                version_prefix = filename.replace('-{}.list'.format(dist_version), '').split('-')[-1]
                _S3_SCYLLA_REPOS_CACHE[(
                    dist_type, dist_version)][version_prefix] = "https://s3.amazonaws.com/{bucket}/{path}".format(bucket=bucket, path=repo_file['Key'])

    else:
        raise NotImplementedError("[{}] is not yet supported".format(dist_type))
    return _S3_SCYLLA_REPOS_CACHE[(dist_type, dist_version)]


def pid_exists(pid):
    """
    Return True if a given PID exists.

    :param pid: Process ID number.
    """
    try:
        os.kill(pid, 0)
    except OSError as detail:
        if detail.errno == errno.ESRCH:
            return False
    return True


def safe_kill(pid, signal):
    """
    Attempt to send a signal to a given process that may or may not exist.

    :param signal: Signal number.
    """
    try:
        os.kill(pid, signal)
        return True
    except Exception:  # pylint: disable=broad-except
        return False


class FileFollowerIterator():  # pylint: disable=too-few-public-methods
    def __init__(self, filename, thread_obj):
        self.filename = filename
        self.thread_obj = thread_obj

    def __iter__(self):
        with open(self.filename, 'r') as input_file:
            line = ''
            while not self.thread_obj.stopped():
                poller = select.poll()  # pylint: disable=no-member
                poller.register(input_file, select.POLLIN)  # pylint: disable=no-member
                if poller.poll(100):
                    line += input_file.readline()
                if not line or not line.endswith('\n'):
                    time.sleep(0.1)
                    continue
                poller.unregister(input_file)
                yield line
                line = ''
            yield line


class FileFollowerThread():
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(1)
        self._stop_event = threading.Event()
        self.future = None

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        raise NotImplementedError()

    def start(self):
        self.future = self.executor.submit(self.run)
        return self.future

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def follow_file(self, filename):
        return FileFollowerIterator(filename, self)


class ScyllaCQLSession:
    def __init__(self, session, cluster, verbose=True):
        self.session = session
        self.cluster = cluster
        self.verbose = verbose

    def __enter__(self):
        execute_orig = self.session.execute

        def execute_verbose(*args, **kwargs):
            if args:
                query = args[0]
            else:
                query = kwargs.get("query")
            LOGGER.debug(f"Executing CQL '{query}'...")
            return execute_orig(*args, **kwargs)

        if self.verbose:
            self.session.execute = execute_verbose
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cluster.shutdown()


class MethodVersionNotFound(Exception):
    pass


class version():  # pylint: disable=invalid-name,too-few-public-methods
    VERSIONS = {}
    """
        Runs a method according to the version attribute of the class method
        Limitations: currently, can't work if the same method name in the same file used in different
                     classes
        Example:
                In [3]: class VersionedClass(object):
                   ...:     def __init__(self, current_version):
                   ...:         self.version = current_version
                   ...:
                   ...:     @version("1.2")
                   ...:     def setup(self):
                   ...:         return "1.2"
                   ...:
                   ...:     @version("2")
                   ...:     def setup(self):
                   ...:         return "2"

                In [4]: vc = VersionedClass("2")

                In [5]: vc.setup()
                Out[5]: '2'

                In [6]: vc = VersionedClass("1.2")

                In [7]: vc.setup()
                Out[7]: '1.2'
    """

    def __init__(self, ver):
        self.version = ver

    def __call__(self, func):
        self.VERSIONS[(self.version, func.__name__, func.__code__.co_filename)] = func

        @wraps(func)
        def inner(*args, **kwargs):
            cls_self = args[0]
            func_to_run = self.VERSIONS.get((cls_self.version, func.__name__, func.__code__.co_filename))
            if func_to_run:
                return func_to_run(*args, **kwargs)
            else:
                raise MethodVersionNotFound("Method '{}' with version '{}' not defined in '{}'!".format(
                    func.__name__,
                    cls_self.version,
                    cls_self.__class__.__name__))
        return inner


def get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    addr = sock.getsockname()
    port = addr[1]
    sock.close()
    return port


def get_my_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    ip = sock.getsockname()[0]
    sock.close()
    return ip


@retrying(n=60, sleep_time=5, allowed_exceptions=(OSError, ))
def wait_for_port(host, port):
    socket.create_connection((host, port)).close()


def can_connect_to(ip: str, port: int, timeout: int = 1) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    result = sock.connect_ex((ip, port))
    sock.close()
    return result == 0


def get_branched_ami(ami_version, region_name):
    """
    Get a list of AMIs, based on version match

    :param ami_version: branch version to look for, ex. 'branch-2019.1:latest', 'branch-3.1:all'
    :param region_name: the region to look AMIs in
    :return: list of ec2.images
    """
    branch, build_id = ami_version.split(':', 1)
    ec2_resource: EC2ServiceResource = boto3.resource('ec2', region_name=region_name)

    LOGGER.info("Looking for AMI match [%s]", ami_version)
    if build_id in ('latest', 'all'):
        filters = [{'Name': 'tag:branch', 'Values': [branch]}]
    else:
        filters = [{'Name': 'tag:branch', 'Values': [branch]}, {'Name': 'tag:build-id', 'Values': [build_id]}]

    amis = list(ec2_resource.images.filter(Filters=filters))

    amis = sorted(amis, key=lambda x: x.creation_date, reverse=True)

    assert amis, "AMI matching [{}] wasn't found on {}".format(ami_version, region_name)
    if build_id == 'all':
        return amis
    else:
        return amis[:1]


def ami_built_by_scylla(ami_id: str, region_name: str) -> bool:
    ec2_resource = boto3.resource("ec2", region_name=region_name)
    image = ec2_resource.Image(ami_id)
    return image.owner_id == SCYLLA_AMI_OWNER_ID


def get_ami_tags(ami_id, region_name):
    """
    Get a list of tags of a specific AMI

    :param ami_id:
    :param region_name: the region to look AMIs in
    :return: dict of tags
    """
    ec2_resource: EC2ServiceResource = boto3.resource('ec2', region_name=region_name)
    test_image = ec2_resource.Image(ami_id)
    if test_image.tags:
        return {i['Key']: i['Value'] for i in test_image.tags}
    else:
        return {}


def tag_ami(ami_id, tags_dict, region_name):
    tags = [{'Key': key, 'Value': value} for key, value in tags_dict.items()]

    ec2_resource: EC2ServiceResource = boto3.resource('ec2', region_name=region_name)
    test_image = ec2_resource.Image(ami_id)
    tags += test_image.tags
    test_image.create_tags(Tags=tags)

    LOGGER.info("tagged %s with %s", ami_id, tags)


def get_db_tables(session, ks, with_compact_storage=True):
    """
    Return tables from keystore based on their compact storage feature
    Arguments:
        session -- DB session
        ks -- Keypsace name
        with_compact_storage -- If True, return non compact tables, if False, return compact tables

    """
    output = []
    for table in session.cluster.metadata.keyspaces[ks].tables.keys():
        table_code = session.cluster.metadata.keyspaces[ks].tables[table].as_cql_query()
        if with_compact_storage is None:
            output.append(table)
        elif ("with compact storage" in table_code.lower()) == with_compact_storage:
            output.append(table)
    return output

# Add @retrying to prevent situation when nemesis failed on connection timeout when try to receive the
# keyspace and table for the test


@retrying(n=5, sleep_time=5)
def get_non_system_ks_cf_list(db_node, request_timeout=300, filter_out_table_with_counter=False,  # pylint: disable=too-many-arguments
                              filter_out_mv=False, filter_empty_tables=True):
    """Get all not system keyspace.tables pairs

    Arguments:
        loader_node {BaseNode} -- LoaderNoder to send request
        db_node_ip {str} -- ip of db_node
    """
    # pylint: disable=too-many-locals

    def get_tables_columns_list(entity_type):
        if entity_type == 'view':
            cmd = "paging off; SELECT keyspace_name, view_name FROM system_schema.views"
        else:
            cmd = "paging off; SELECT keyspace_name, table_name, type FROM system_schema.columns"
        result = db_node.run_cqlsh(cmd=cmd, timeout=request_timeout, verbose=False,
                                   split=True, connect_timeout=request_timeout)
        if not result:
            return []

        splitter_result = []
        for row in result[4:]:
            if '|' not in row:
                continue
            if row.startswith('system') or row.startswith('alternator_usertable'):
                continue
            splitter_result.append(row.split('|'))
        return splitter_result

    views_list = set()
    if filter_out_mv:
        tables = get_tables_columns_list('view')

        for table in tables:
            views_list.add('.'.join([name.strip() for name in table[:2]]))
        views_list = list(views_list)

    result = get_tables_columns_list('column')
    if not result:
        return []

    avaialable_ks_cf = defaultdict(list)
    for row in result:
        ks_cf_name = '.'.join([name.strip() for name in row[:2]])

        if filter_out_mv and ks_cf_name in views_list:
            continue

        column_type = row[2].strip()
        avaialable_ks_cf[ks_cf_name].append(column_type)

    if filter_out_table_with_counter:
        for ks_cf, column_types in list(avaialable_ks_cf.items()):
            if 'counter' in column_types:
                avaialable_ks_cf.pop(ks_cf)

    if filter_empty_tables:
        for ks_cf in list(avaialable_ks_cf.keys()):
            cmd = f'SELECT * FROM {ks_cf} LIMIT 1'
            # TODO: make this work with cql python driver, but would need to refactor the session of of tester class
            result = db_node.run_cqlsh(cmd=cmd, timeout=request_timeout, verbose=False,
                                       split=False, connect_timeout=request_timeout)
            if '(0 rows)' in result.stdout:
                avaialable_ks_cf.pop(ks_cf)

    return list(avaialable_ks_cf.keys())


def remove_files(path):
    LOGGER.debug("Remove path %s", path)
    try:
        if os.path.isdir(path):
            shutil.rmtree(path=path, ignore_errors=True)
        if os.path.isfile(path):
            os.remove(path)
    except Exception as details:  # pylint: disable=broad-except
        LOGGER.error("Error during remove archived logs %s", details)
        LOGGER.info("Remove temporary data manually: \"%s\"", path)


def format_timestamp(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def wait_ami_available(client, ami_id):
    """Wait while ami_id become available

    Wait while ami_id become available, after
    10 minutes return an error

    Arguments:
        client {boto3.EC2.Client} -- client of EC2 service
        ami_id {str} -- ami id to check availability
    """
    waiter = client.get_waiter('image_available')
    waiter.wait(ImageIds=[ami_id],
                WaiterConfig={
                    'Delay': 30,
                    'MaxAttempts': 20}
                )


def update_certificates():
    """
    Update the certificate of server encryption, which might be expired.
    """
    try:
        from sdcm.remote import LocalCmdRunner
        localrunner = LocalCmdRunner()
        localrunner.run('openssl x509 -req -in data_dir/ssl_conf/example/db.csr -CA data_dir/ssl_conf/cadb.pem -CAkey data_dir/ssl_conf/example/cadb.key -CAcreateserial -out data_dir/ssl_conf/db.crt -days 365')
        localrunner.run('openssl x509 -enddate -noout -in data_dir/ssl_conf/db.crt')
    except Exception as ex:
        raise Exception('Failed to update certificates by openssl: %s' % ex)


def s3_download_dir(bucket, path, target):
    """
    Downloads recursively the given S3 path to the target directory.
    :param bucket: the name of the bucket to download from
    :param path: The S3 directory to download.
    :param target: the local directory to download the files to.
    """

    client: S3Client = boto3.client('s3', region_name=DEFAULT_AWS_REGION)

    # Handle missing / at end of prefix
    if not path.endswith('/'):
        path += '/'
    if path.startswith('/'):
        path = path[1:]
    result = client.list_objects_v2(Bucket=bucket, Prefix=path)
    # Download each file individually
    for key in result['Contents']:
        # Calculate relative path
        rel_path = key['Key'][len(path):]
        # Skip paths ending in /
        if not key['Key'].endswith('/'):
            local_file_path = os.path.join(target, rel_path)
            # Make sure directories exist
            local_file_dir = os.path.dirname(local_file_path)
            os.makedirs(local_file_dir, exist_ok=True)
            LOGGER.info("Downloading %s from s3 to %s", key['Key'], local_file_path)
            client.download_file(bucket, key['Key'], local_file_path)


def gce_download_dir(bucket, path, target):
    """
    Downloads recursively the given google storage path to the target directory.
    :param bucket: the name of the bucket to download from
    :param path: The google storage directory to download.
    :param target: the local directory to download the files to.
    """

    from sdcm.keystore import KeyStore
    gcp_credentials = KeyStore().get_gcp_credentials()
    gce_driver = libcloud.storage.providers.get_driver(libcloud.storage.types.Provider.GOOGLE_STORAGE)

    driver = gce_driver(gcp_credentials["project_id"] + "@appspot.gserviceaccount.com",
                        gcp_credentials["private_key"],
                        project=gcp_credentials["project_id"])

    if not path.endswith('/'):
        path += '/'
    if path.startswith('/'):
        path = path[1:]

    container = driver.get_container(container_name=bucket)
    dir_listing = driver.list_container_objects(container, ex_prefix=path)
    for obj in dir_listing:
        rel_path = obj.name[len(path):]
        local_file_path = os.path.join(target, rel_path)

        local_file_dir = os.path.dirname(local_file_path)
        os.makedirs(local_file_dir, exist_ok=True)
        LOGGER.info("Downloading %s from gcp to %s", obj.name, local_file_path)
        obj.download(destination_path=local_file_path, overwrite_existing=True)


def download_dir_from_cloud(url):
    """
    download a directory from AWS S3 or from google storage

    :param url: a url that starts with `s3://` or `gs://`
    :return: the temp directory create with the downloaded content
    """
    if url is None:
        return url

    md5 = hashlib.md5()  # deepcode ignore insecureHash: can't change it
    md5.update(url.encode('utf-8'))
    tmp_dir = os.path.join('/tmp/download_from_cloud', md5.hexdigest())
    parsed = urlparse(url)
    LOGGER.info("Downloading [%s] to [%s]", url, tmp_dir)
    if os.path.isdir(tmp_dir) and os.listdir(tmp_dir):
        LOGGER.warning("[{}] already exists, skipping download".format(tmp_dir))
    else:
        if url.startswith('s3://'):
            s3_download_dir(parsed.hostname, parsed.path, tmp_dir)
        elif url.startswith('gs://'):
            gce_download_dir(parsed.hostname, parsed.path, tmp_dir)
        elif os.path.isdir(url):
            tmp_dir = url
        else:
            raise ValueError("Unsupported url schema or non-existing directory [{}]".format(url))
    if not tmp_dir.endswith('/'):
        tmp_dir += '/'
    LOGGER.info("Finished downloading [%s]", url)
    return tmp_dir


def filter_aws_instances_by_type(instances):
    filtered_instances = {
        "db_nodes": [],
        "loader_nodes": [],
        "monitor_nodes": []
    }

    for instance in instances:
        name = [tag['Value']
                for tag in instance['Tags'] if tag['Key'] == 'Name']
        if 'db-node' in name[0]:
            filtered_instances["db_nodes"].append(instance)
        if 'monitor-node' in name[0]:
            filtered_instances["monitor_nodes"].append(instance)
        if 'loader-node' in name[0]:
            filtered_instances["loader_nodes"].append(instance)
    return filtered_instances


def filter_gce_instances_by_type(instances):
    filtered_instances = {
        "db_nodes": [],
        "loader_nodes": [],
        "monitor_nodes": []
    }

    for instance in instances:
        if 'db-node' in instance.name:
            filtered_instances["db_nodes"].append(instance)
        if 'monitor-node' in instance.name:
            filtered_instances["monitor_nodes"].append(instance)
        if 'loader-node' in instance.name:
            filtered_instances["loader_nodes"].append(instance)
    return filtered_instances


def filter_docker_containers_by_type(containers):
    filtered_containers = {
        "db_nodes": [],
        "loader_nodes": [],
        "monitor_nodes": []
    }

    for container in containers:
        if "db-node" in container.name:
            filtered_containers["db_nodes"].append(container)
        if "monitor-node" in container.name:
            filtered_containers["monitor_nodes"].append(container)
        if "loader-node" in container.name:
            filtered_containers["loader_nodes"].append(container)
    return filtered_containers


SSH_KEY_DIR = "~/.ssh"
SSH_KEY_AWS_DEFAULT = "scylla-qa-ec2"
SSH_KEY_GCE_DEFAULT = "scylla-test"


def get_aws_builders(tags=None, running=False):
    builders = []
    ssh_key_path = os.path.join(SSH_KEY_DIR, SSH_KEY_AWS_DEFAULT)

    aws_builders = list_instances_aws(tags_dict=tags, running=running)

    for aws_builder in aws_builders:
        builder_name = [tag["Value"] for tag in aws_builder["Tags"] if tag["Key"] == "Name"][0]
        builders.append({"builder": {
            "public_ip": aws_builder["PublicIpAddress"],
            "name": builder_name,
            "user": "jenkins",
            "key_file": os.path.expanduser(ssh_key_path)
        }})

    return builders


def get_gce_builders(tags=None, running=False):
    builders = []
    ssh_key_path = os.path.join(SSH_KEY_DIR, SSH_KEY_GCE_DEFAULT)

    gce_builders = list_instances_gce(tags_dict=tags, running=running)

    for gce_builder in gce_builders:
        builders.append({"builder": {
            "public_ip": gce_builder.public_ips[0],
            "name": gce_builder.name,
            "user": "scylla-test",
            "key_file": os.path.expanduser(ssh_key_path)
        }})

    return builders


def list_builders(running=False):
    builder_tag = {"NodeType": "Builder"}
    aws_builders = get_aws_builders(builder_tag, running=running)
    gce_builders = get_gce_builders(builder_tag, running=running)

    return aws_builders + gce_builders


def get_builder_by_test_id(test_id):
    from sdcm.remote import RemoteCmdRunnerBase

    base_path_on_builder = "/home/jenkins/slave/workspace"
    found_builders = []

    builders = list_builders()

    def search_test_id_on_builder(builder):
        remoter = RemoteCmdRunnerBase.create_remoter(
            builder['public_ip'], user=builder["user"], key_file=builder["key_file"])

        LOGGER.info('Search on %s', builder['name'])
        result = remoter.run("find {where} -name test_id | xargs grep -rl {test_id}".format(where=base_path_on_builder,
                                                                                            test_id=test_id),
                             ignore_status=True, verbose=False)

        if not result.exited and result.stdout:
            builder["remoter"] = remoter
            path = result.stdout.strip()
            LOGGER.info("Builder name %s, ip %s, folder %s", builder['name'], builder['public_ip'], path)
            return {
                "builder": builder,
                "path": os.path.dirname(path)
            }
        else:
            LOGGER.info("Nothing found")
            return None

    search_obj = ParallelObject(builders, timeout=30, num_workers=len(builders))
    results = search_obj.run(search_test_id_on_builder, ignore_exceptions=True, unpack_objects=True)
    found_builders = [builder.result for builder in results if not builder.exc and builder.result]
    if not found_builders:
        LOGGER.info("Nothing found for %s", test_id)

    return found_builders


def get_post_behavior_actions(config):
    action_per_type = {
        "db_nodes": {"NodeType": "scylla-db", "action": None},
        "monitor_nodes": {"NodeType": "monitor", "action": None},
        "loader_nodes": {"NodeType": "loader", "action": None},
    }

    for key in action_per_type:
        config_key = 'post_behavior_{}'.format(key)
        action_per_type[key]["action"] = config.get(config_key)

    return action_per_type


def clean_resources_according_post_behavior(params, config, logdir):
    success = get_testrun_status(params.get('TestId'), logdir, only_critical=True)
    actions_per_type = get_post_behavior_actions(config)
    LOGGER.debug(actions_per_type)
    for cluster_nodes_type, action_type in actions_per_type.items():
        if action_type["action"] == "keep":
            LOGGER.info("Post behavior %s for %s. Keep resources running", action_type["action"], cluster_nodes_type)
        elif action_type["action"] == "destroy":
            params["NodeType"] = action_type["NodeType"]
            LOGGER.info("Post behavior %s for %s. Clean resources", action_type["action"], cluster_nodes_type)
            clean_cloud_resources(params)
            continue
        elif action_type["action"] == "keep-on-failure" and not success:
            params["NodeType"] = action_type["NodeType"]
            LOGGER.info("Post behavior %s for %s. Test run Successful. Clean resources",
                        action_type["action"], cluster_nodes_type)
            clean_cloud_resources(params)
            continue
        else:
            LOGGER.info("Post behavior %s for %s. Test run Failed. Keep resources running",
                        action_type["action"], cluster_nodes_type)
            continue


def search_test_id_in_latest(logdir):
    from sdcm.remote import LocalCmdRunner

    test_id = None
    result = LocalCmdRunner().run('cat {0}/latest/test_id'.format(logdir), ignore_status=True)
    if not result.exited and result.stdout:
        test_id = result.stdout.strip()
        LOGGER.info("Found latest test_id: {}".format(test_id))
        LOGGER.info("Collect logs for test-run with test-id: {}".format(test_id))
    else:
        LOGGER.error('test_id not found. Exit code: %s; Error details %s', result.exited, result.stderr)
    return test_id


def get_testrun_dir(base_dir, test_id=None):
    from sdcm.remote import LocalCmdRunner

    if not test_id:
        test_id = search_test_id_in_latest(base_dir)
    LOGGER.info('Search dir with logs locally for test id: %s', test_id)
    search_cmd = "find {base_dir} -name test_id | xargs grep -rl {test_id}".format(**locals())
    result = LocalCmdRunner().run(cmd=search_cmd, ignore_status=True)
    LOGGER.info("Search result %s", result)
    if result.exited == 0 and result.stdout:
        found_dirs = result.stdout.strip().split('\n')
        LOGGER.info(found_dirs)
        return os.path.dirname(found_dirs[0])
    LOGGER.info("No any dirs found locally for current test id")
    return None


def get_testrun_status(test_id=None, logdir=None, only_critical=False):
    testrun_dir = get_testrun_dir(logdir, test_id)
    if not testrun_dir:
        return None

    with open(os.path.join(testrun_dir, 'events_log/critical.log')) as f:  # pylint: disable=invalid-name
        status = f.readlines()

    if not only_critical:
        with open(os.path.join(testrun_dir, 'events_log/error.log')) as f:  # pylint: disable=invalid-name
            status += f.readlines()

    return status


def download_encrypt_keys():
    """
    Download certificate files of encryption at-rest from S3 KeyStore
    """
    from sdcm.keystore import KeyStore
    ks = KeyStore()
    for pem_file in ['CA.pem', 'SCYLLADB.pem', 'hytrust-kmip-cacert.pem', 'hytrust-kmip-scylla.pem']:
        if not os.path.exists('./data_dir/encrypt_conf/%s' % pem_file):
            ks.download_file(pem_file, './data_dir/encrypt_conf/%s' % pem_file)


def normalize_ipv6_url(ip_address):
    """adds square brackets on the IPv6 address in the URL"""
    if ":" in ip_address:  # IPv6
        return "[%s]" % ip_address
    return ip_address


def rows_to_list(rows):
    return [list(row) for row in rows]


# Copied from dtest
class Page:  # pylint: disable=too-few-public-methods
    data = None

    def __init__(self):
        self.data = []

    def add_row(self, row):
        self.data.append(row)


# Copied from dtest
class PageFetcher:
    """
    Requests pages, handles their receipt,
    and provides paged data for testing.

    The first page is automatically retrieved, so an initial
    call to request_one is actually getting the *second* page!
    """
    pages = None
    error = None
    future = None
    requested_pages = None
    retrieved_pages = None
    retrieved_empty_pages = None

    def __init__(self, future):
        self.pages = []

        # the first page is automagically returned (eventually)
        # so we'll count this as a request, but the retrieved count
        # won't be incremented until it actually arrives
        self.requested_pages = 1
        self.retrieved_pages = 0
        self.retrieved_empty_pages = 0

        self.future = future
        self.future.add_callbacks(
            callback=self.handle_page,
            errback=self.handle_error
        )

        # wait for the first page to arrive, otherwise we may call
        # future.has_more_pages too early, since it should only be
        # called after the first page is returned
        self.wait(seconds=30)

    def handle_page(self, rows):
        # occasionally get a final blank page that is useless
        if rows == []:
            self.retrieved_empty_pages += 1
            return

        page = Page()
        self.pages.append(page)

        for row in rows:
            page.add_row(row)

        self.retrieved_pages += 1

    def handle_error(self, exc):
        self.error = exc
        raise exc

    def request_one(self):
        """
        Requests the next page if there is one.

        If the future is exhausted, this is a no-op.
        """
        if self.future.has_more_pages:
            self.future.start_fetching_next_page()
            self.requested_pages += 1
            self.wait()

        return self

    def request_all(self):
        """
        Requests any remaining pages.

        If the future is exhausted, this is a no-op.
        """
        while self.future.has_more_pages:
            self.future.start_fetching_next_page()
            self.requested_pages += 1
            self.wait()

        return self

    def wait(self, seconds=10):
        """
        Blocks until all *requested* pages have been returned.

        Requests are made by calling request_one and/or request_all.

        Raises RuntimeError if seconds is exceeded.
        """
        def error_message(msg):
            return "{}. Requested: {}; retrieved: {}; empty retrieved {}".format(
                msg, self.requested_pages, self.retrieved_pages, self.retrieved_empty_pages)

        def missing_pages():
            pages = self.requested_pages - (self.retrieved_pages + self.retrieved_empty_pages)
            assert pages >= 0, error_message('Retrieved too many pages')
            return pages

        missing = missing_pages()
        if missing <= 0:
            return self
        expiry = time.time() + seconds * missing

        while time.time() < expiry:
            if missing_pages() <= 0:
                return self
            # small wait so we don't need excess cpu to keep checking
            time.sleep(0.1)

        raise RuntimeError(error_message('Requested pages were not delivered before timeout'))

    def pagecount(self):
        """
        Returns count of *retrieved* pages which were not empty.

        Pages are retrieved by requesting them with request_one and/or request_all.
        """
        return len(self.pages)

    def num_results(self, page_num):
        """
        Returns the number of results found at page_num
        """
        return len(self.pages[page_num - 1].data)

    def num_results_all(self):
        return [len(page.data) for page in self.pages]

    def page_data(self, page_num):
        """
        Returns retreived data found at pagenum.

        The page should have already been requested with request_one and/or request_all.
        """
        return self.pages[page_num - 1].data

    def all_data(self):
        """
        Returns all retrieved data flattened into a single list (instead of separated into Page objects).

        The page(s) should have already been requested with request_one and/or request_all.
        """
        all_pages_combined = []
        for page in self.pages:
            all_pages_combined.extend(page.data[:])

        return all_pages_combined

    @property  # make property to match python driver api
    def has_more_pages(self):
        """
        Returns bool indicating if there are any pages not retrieved.
        """
        return self.future.has_more_pages


def get_docker_stress_image_name(tool_name=None):
    if not tool_name:
        return None
    base_path = os.path.dirname(os.path.dirname((os.path.dirname(__file__))))
    with open(os.path.join(base_path, "docker", tool_name, "image"), "r") as image_file:
        result = image_file.read()

    return result.strip()


def search_database_enospc(node):
    """
    Search system log by executing cmd inside node, use shell tool to
    avoid return and process huge data.
    """
    cmd = "sudo journalctl --no-tail --no-pager -u scylla-server.service|grep 'No space left on device'|wc -l"
    result = node.remoter.run(cmd, verbose=True)
    return int(result.stdout)


def approach_enospc(node, orig_errors):
    # get the size of free space (default unit: KB)
    result = node.remoter.run("df -l|grep '/var/lib/scylla'")
    free_space_size = result.stdout.split()[3]

    occupy_space_size = int(int(free_space_size) * 90 / 100)
    occupy_space_cmd = 'sudo fallocate -l {}K /var/lib/scylla/occupy_90percent.{}'.format(
        occupy_space_size, datetime.datetime.now().strftime('%s'))
    LOGGER.debug('Cost 90% free space on /var/lib/scylla/ by {}'.format(occupy_space_cmd))
    try:
        node.remoter.run(occupy_space_cmd, verbose=True)
    except Exception as details:  # pylint: disable=broad-except
        LOGGER.error(str(details))
    return search_database_enospc(node) > orig_errors


def reach_enospc_on_node(target_node):
    # check original ENOSPC error
    orig_errors = search_database_enospc(target_node)
    wait.wait_for(func=approach_enospc,
                  timeout=300,
                  step=5,
                  text='Wait for new ENOSPC error occurs in database',
                  node=target_node,
                  orig_errors=orig_errors)


def clean_enospc_on_node(target_node, sleep_time):
    LOGGER.debug('Sleep {} seconds before releasing space to scylla'.format(sleep_time))
    time.sleep(sleep_time)

    LOGGER.debug('Delete occupy_90percent file to release space to scylla-server')
    target_node.remoter.run('sudo rm -rf /var/lib/scylla/occupy_90percent.*')

    LOGGER.debug('Sleep a while before restart scylla-server')
    time.sleep(sleep_time / 2)
    target_node.remoter.run('sudo systemctl restart scylla-server.service')
    target_node.wait_db_up()


def update_authenticator(nodes, authenticator='AllowAllAuthenticator', restart=True):
    """
    Update the authenticator of nodes, restart the nodes to make the change effective
    """
    for node in nodes:
        node.remoter.run(
            f"sudo sed -ie 's/^authenticator:.*/authenticator: {authenticator}/g' /etc/scylla/scylla.yaml")
        if restart:
            if authenticator == 'com.scylladb.auth.SaslauthdAuthenticator':
                node.run_cqlsh(f'ALTER ROLE \'{LDAP_USERS[0]}\' with password=\'{LDAP_PASSWORD}\'')
            node.use_saslauthd_authenticator = authenticator == 'com.scylladb.auth.SaslauthdAuthenticator'
            node.parent_cluster.params['are_ldap_users_on_scylla'] = node.use_saslauthd_authenticator
            node.restart_scylla_server()
            node.wait_db_up()


def prepare_and_start_saslauthd_service(node):
    """
    Install and setup saslauthd service.
    """
    if node.is_rhel_like():
        setup_script = dedent(f"""
            sudo yum install -y cyrus-sasl
            sudo systemctl enable saslauthd
            echo 'MECH=ldap' | sudo tee -a /etc/sysconfig/saslauthd
            sudo touch /etc/saslauthd.conf
        """)
    else:
        setup_script = dedent(f"""
            sudo apt-get install -y sasl2-bin
            sudo systemctl enable saslauthd
            echo -e 'MECHANISMS=ldap\nSTART=yes\n' | sudo tee -a /etc/default/saslauthd
            sudo touch /etc/saslauthd.conf
            sudo adduser scylla sasl  # to avoid the permission issue of unit socket
        """)
    node.remoter.run('bash -cxe "%s"' % setup_script)
    conf = node.get_saslauthd_config()
    for key in conf.keys():
        node.remoter.run(f'echo "{key}: {conf[key]}" | sudo tee -a /etc/saslauthd.conf')
    node.remoter.run(
        f"echo 'saslauthd_socket_path: /run/saslauthd/mux' |sudo tee --append /etc/scylla/scylla.yaml")
    node.remoter.run('sudo systemctl restart saslauthd')
