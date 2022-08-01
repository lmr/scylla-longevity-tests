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

import os
import json
import random
import logging
import time


from google.oauth2 import service_account
from googleapiclient.discovery import build
from libcloud.compute.base import NodeDriver
from libcloud.compute.providers import Provider, get_driver

from sdcm.keystore import KeyStore
from sdcm.utils.docker_utils import ContainerManager, DockerException, Container


# NOTE: we cannot use neither 'slim' nor 'alpine' versions because we need the 'beta' component be installed.
GOOGLE_CLOUD_SDK_IMAGE = "google/cloud-sdk:367.0.0"

LOGGER = logging.getLogger(__name__)

GceDriver = get_driver(Provider.GCE)  # pylint: disable=invalid-name

# The keys are the region name, the value is the available zones, which will be used for random.choice()
SUPPORTED_REGIONS = {
    # us-east1 zones: b, c, and d. Details: https://cloud.google.com/compute/docs/regions-zones#locations
    # Currently choose only zones c and d as zone b frequently fails allocating resources.
    'us-east1': 'cd',
    'us-west1': 'abc'}


def append_zone(region: str) -> str:
    dash_count = region.count("-")
    if dash_count == 2:
        # return as is if region already has availability zone in it
        return region
    if dash_count != 1:
        raise Exception(f'Wrong region name: {region}')
    availability_zones = SUPPORTED_REGIONS.get(region, None)
    if not availability_zones:
        raise Exception(f'Unsupported region: {region}')
    return f"{region}-{random.choice(availability_zones)}"


def _get_gce_service(credentials: dict, datacenter: str) -> GceDriver:
    return GceDriver(user_id=credentials["client_email"],
                     key=credentials["private_key"],
                     datacenter=datacenter,
                     project=credentials["project_id"])


def get_gce_services(regions: list) -> dict[str, NodeDriver]:
    credentials = KeyStore().get_gcp_credentials()
    return {region_az: _get_gce_service(credentials, region_az) for region_az in map(append_zone, regions)}


def get_gce_service(region: str) -> GceDriver:
    credentials = KeyStore().get_gcp_credentials()
    return _get_gce_service(credentials, append_zone(region))


class GcloudContextManager:
    def __init__(self, instance: 'GcloudContainerMixin', name: str):
        self._instance = instance
        self._name = name
        self._container = None
        self._span_counter = 0

    def _span_container(self):
        self._span_counter += 1
        if self._container:
            return
        try:
            self._container = self._instance._get_gcloud_container()  # pylint: disable=protected-access
        except Exception as exc:
            try:
                ContainerManager.destroy_container(self._instance, self._name)
            except Exception:  # pylint: disable=broad-except
                pass
            raise exc from None

    def _destroy_container(self):
        self._span_counter -= 1
        if self._span_counter != 0:
            return
        try:
            ContainerManager.destroy_container(self._instance, self._name)
        except Exception:  # pylint: disable=broad-except
            pass
        self._container = None

    def __enter__(self):
        self._span_container()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._destroy_container()

    def run(self, command) -> str:
        one_time = self._container is None
        if one_time:
            self._span_container()
        try:
            LOGGER.debug("Execute `gcloud %s'", command)
            res = self._container.exec_run(["sh", "-c", f"gcloud {command}"])
            if res.exit_code:
                raise DockerException(f"{self._container}: {res.output.decode('utf-8')}")
            return res.output.decode("utf-8")
        finally:
            if one_time:
                self._destroy_container()


class GcloudContainerMixin:
    """Run gcloud command using official Google Cloud SDK Docker image.

    See more details here: https://hub.docker.com/r/google/cloud-sdk
    """
    _gcloud_container_instance = None

    def gcloud_container_run_args(self) -> dict:
        kube_config_path = os.environ.get('KUBECONFIG', '~/.kube/config')
        kube_config_dir_path = os.path.expanduser(kube_config_path)
        volumes = {
            os.path.dirname(kube_config_dir_path): {"bind": os.path.dirname(kube_config_dir_path), "mode": "rw"},
        }
        return dict(image=GOOGLE_CLOUD_SDK_IMAGE,
                    command="cat",
                    tty=True,
                    name=f"{self.name}-gcloud",
                    volumes=volumes,
                    user=f"{os.getuid()}:{os.getgid()}",
                    tmpfs={'/.config': f'size=50M,uid={os.getuid()}'},
                    environment={'KUBECONFIG': kube_config_path},
                    )

    def _get_gcloud_container(self) -> Container:
        """Create Google Cloud SDK container.

        Cloud SDK requires to enable some authorization method first.  Because of that we start a container which
        runs forever using `cat' command (like Jenkins do), put a service account credentials and activate them.

        All consequent gcloud commands run using container.exec_run() method.
        """
        container = ContainerManager.run_container(self, "gcloud")
        credentials = KeyStore().get_gcp_credentials()
        credentials["client_email"] = f"{credentials['client_email']}"
        shell_command = f"umask 077 && echo '{json.dumps(credentials)}' > /tmp/gcloud_svc_account.json"
        shell_command += " && echo 'kubeletConfig:\n  cpuManagerPolicy: static' > /tmp/system_config.yaml"
        # NOTE: use 'bash' in case of non-alpine sdk image and 'sh' when it is 'alpine' one.
        res = container.exec_run(["bash", "-c", shell_command])
        if res.exit_code:
            raise DockerException(f"{container}: {res.output.decode('utf-8')}")
        res = container.exec_run(["gcloud", "auth", "activate-service-account", credentials["client_email"],
                                  "--key-file", "/tmp/gcloud_svc_account.json",
                                  "--project", credentials["project_id"]])
        if res.exit_code:
            raise DockerException(f"{container}[]: {res.output.decode('utf-8')}")
        return container

    @property
    def gcloud(self) -> GcloudContextManager:
        return GcloudContextManager(self, 'gcloud')


class GceLoggingClient:  # pylint: disable=too-few-public-methods

    def __init__(self, instance_name: str, zone: str):
        credentials = KeyStore().get_gcp_credentials()
        self.credentials = service_account.Credentials.from_service_account_info(credentials)
        self.project_id = credentials['project_id']
        self.instance_name = instance_name
        self.zone = zone

    def get_system_events(self, from_: float, until: float):
        """Gets system events logs entries from GCE between time (since -> to).

        Returns list of entries in a form of dictionaries. See example output in unit tests."""
        from_ = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(from_))
        until = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(until))
        query = {
            "resourceNames": [
                f"projects/{self.project_id}"
            ],
            "filter": f'protoPayload.resourceName="projects/{self.project_id}/zones/{self.zone}/instances/{self.instance_name}"'
                      f' logName : projects/{self.project_id}/logs/cloudaudit.googleapis.com%2Fsystem_event'
                      f' timestamp > "{from_}" timestamp < "{until}"'
        }
        with build('logging', 'v2', credentials=self.credentials, cache_discovery=False) as service:
            return self._get_log_entries(service, query)

    def _get_log_entries(self, service, query, page_token=None):
        if page_token:
            query.update({"page_token": page_token})
        ret = service.entries().list(body=query).execute()
        entries = ret.get('entries', [])
        if page_token := ret.get("nextPageToken"):
            entries.extend(self._get_log_entries(service, query, page_token))
        return entries
