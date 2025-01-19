from enum import Enum
import logging
import datetime
import yaml
from typing import Optional
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sdcm.utils.distro import Distro
from sdcm.utils.common import get_sct_root_path


DEFAULT_TASK_TIMEOUT = 7200  # 2 hours
LOGGER = logging.getLogger(__name__)


def get_persistent_snapshots():  # Snapshot sizes (dict keys) are in GB
    with open("defaults/manager_persistent_snapshots.yaml", encoding="utf-8") as mgmt_snapshot_yaml:
        persistent_manager_snapshots_dict = yaml.safe_load(mgmt_snapshot_yaml)
    return persistent_manager_snapshots_dict


def get_distro_name(distro_object: Distro) -> str:
    if distro_object.is_debian_like:
        return "debian"
    if distro_object.is_rhel_like:
        return "rhel"

    raise ValueError(f"Unsupported distribution for installing manager: {distro_object}")


def duration_to_timedelta(duration_string):
    total_seconds = 0
    if "d" in duration_string:
        total_seconds += int(duration_string[:duration_string.find('d')]) * 86400
        duration_string = duration_string[duration_string.find('d') + 1:]
    if "h" in duration_string:
        total_seconds += int(duration_string[:duration_string.find('h')]) * 3600
        duration_string = duration_string[duration_string.find('h') + 1:]
    if "m" in duration_string:
        total_seconds += int(duration_string[:duration_string.find('m')]) * 60
        duration_string = duration_string[duration_string.find('m') + 1:]
    if "s" in duration_string:
        total_seconds += int(duration_string[:duration_string.find('s')])
    return datetime.timedelta(seconds=total_seconds)


def create_cron_list_from_timedelta(minutes=0, hours=0):
    destined_time = datetime.datetime.now() + datetime.timedelta(hours=hours, minutes=minutes)
    cron_list = [str(destined_time.minute), str(destined_time.hour), "*", "*", "*"]
    return cron_list


def get_manager_repo_from_defaults(manager_version, distro):
    with (Path(get_sct_root_path()) / "defaults/manager_versions.yaml").open(encoding="utf-8") as mgmt_config:
        manager_repos_by_version_dict = yaml.safe_load(mgmt_config)["manager_repos_by_version"]

    # If the version is a patch version, we need to remove the patch part
    if len(version_parts := manager_version.split('.')) == 3:
        manager_version = f"{version_parts[0]}.{version_parts[1]}"

    version_specific_repos = manager_repos_by_version_dict.get(manager_version, None)
    assert version_specific_repos, f"Couldn't find manager version {manager_version} in manager defaults"

    distro_name = get_distro_name(distro)

    repo_address = version_specific_repos.get(distro_name, None)
    assert repo_address, f"Could not find manager repo for distro {distro_name} in version {manager_version}"

    return repo_address


def get_manager_scylla_backend(scylla_backend_version_name, distro):
    with (Path(get_sct_root_path()) / "defaults/manager_versions.yaml").open(encoding="utf-8") as mgmt_config:
        scylla_backend_repos_by_version_dict = yaml.safe_load(mgmt_config)["scylla_backend_repo_by_version"]

    version_specific_repos = scylla_backend_repos_by_version_dict.get(scylla_backend_version_name, None)
    assert version_specific_repos, f"Couldn't find scylla version {scylla_backend_version_name} in manager defaults"

    distro_name = get_distro_name(distro)

    backend_repo_address = version_specific_repos.get(distro_name, None)
    assert backend_repo_address, f"Could not find manager scylla backend repo for {distro}"

    return backend_repo_address


def reconfigure_scylla_manager(manager_node, logger, values_to_update=(), values_to_remove=()):
    with manager_node.remote_manager_yaml() as scylla_manager_yaml:
        for value in values_to_update:
            scylla_manager_yaml.update(value)
        for value in values_to_remove:
            del scylla_manager_yaml[value]
        logger.info("The new Scylla Manager is:\n%s", scylla_manager_yaml)
    manager_node.restart_manager_server()


class ScyllaManagerError(Exception):
    """
    A custom exception for Manager related errors
    """


class HostSsl(Enum):
    ON = "ON"
    OFF = "OFF"

    @classmethod
    def from_str(cls, output_str):
        if "SSL" in output_str:
            return HostSsl.ON
        return HostSsl.OFF


class HostStatus(Enum):
    UP = "UP"
    DOWN = "DOWN"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def from_str(cls, output_str):
        try:
            output_str = output_str.upper()
            if output_str == "-":
                return cls.DOWN
            return getattr(cls, output_str)
        except AttributeError as err:
            raise ScyllaManagerError("Could not recognize returned host status: {}".format(output_str)) from err


class HostRestStatus(Enum):
    UP = "UP"
    DOWN = "DOWN"
    TIMEOUT = "TIMEOUT"
    UNAUTHORIZED = "UNAUTHORIZED"
    HTTP = "HTTP"

    @classmethod
    def from_str(cls, output_str):
        try:
            output_str = output_str.upper()
            if output_str == "-":
                return cls.DOWN
            return getattr(cls, output_str)
        except AttributeError as err:
            raise ScyllaManagerError("Could not recognize returned host rest status: {}".format(output_str)) from err


class TaskStatus:  # pylint: disable=too-few-public-methods
    NEW = "NEW"
    RUNNING = "RUNNING"
    DONE = "DONE"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"
    ERROR_FINAL = "ERROR (4/4)"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    WAITING = "WAITING"
    STARTING = "STARTING"
    ABORTED = "ABORTED"
    SKIPPED = "SKIPPED"

    @classmethod
    def from_str(cls, output_str) -> str:
        try:
            output_str = output_str.upper()
            return getattr(cls, output_str)
        except AttributeError as err:
            raise ScyllaManagerError("Could not recognize returned task status: {}".format(output_str)) from err

    @classmethod
    def all_statuses(cls):
        return set(getattr(cls, name) for name in dir(cls) if name.isupper())


class AgentBackupParameters(BaseModel):
    checkers: Optional[int] = 100
    transfers: Optional[int] = 2
    low_level_retries: Optional[int] = 20

    model_config = ConfigDict(arbitrary_types_allowed=False)
