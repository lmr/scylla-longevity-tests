#!/usr/bin/env python3
import logging
import sys
import re
import os

from sdcm.utils.version_utils import is_enterprise, get_all_versions, get_branch_version
from sdcm.utils.version_utils import ComparableScyllaVersion, get_s3_scylla_repos_mapping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


LOGGER = logging.getLogger(__name__)


# We support to migrate from specific OSS version to enterprise
supported_src_oss = {'2021.1': ['4.3'], '2022.1': ['5.0'],
                     '2022.2': ['5.1'], '2023.1': ['5.2'], '2024.1': ['5.4'], '2024.2': ['6.0'],
                     '2025.1': ['6.2', '2024.2'],
                     }
# If new support distro shared repo with others, we need to assign the start support versions. eg: centos8
start_support_versions = {'centos-8': {'scylla': '4.1', 'enterprise': '2021.1'},
                          'centos-9': {'scylla': '5.4', 'enterprise': '2024.1'}}
start_support_backend = {'azure': {'scylla': '5.2', 'enterprise': '2023.1'}}

# list of version that are available, but aren't supported, and we should test upgrades from
unsupported_versions = ['5.3', ]


class UpgradeBaseVersion:  # pylint: disable=too-many-instance-attributes

    oss_start_support_version = None
    ent_start_support_version = None

    def __init__(self, scylla_repo: str, linux_distro: str, scylla_version: str = None):
        if len(linux_distro.split('-')) > 1:
            self.dist_type, *_, self.dist_version = linux_distro.split('-')
        else:
            self.dist_type = linux_distro.split('-')[0]
            self.dist_version = None
        self.actual_version = None
        self.scylla_repo = scylla_repo
        self.linux_distro = linux_distro
        self.scylla_version = self.get_version(scylla_version)
        self.repo_maps = get_s3_scylla_repos_mapping(self.dist_type, self.dist_version)

    def get_version(self, scylla_version: str = None) -> tuple[str, str]:
        """
        return scylla product name and major version. if scylla_version isn't assigned,
        we will try to get the major version from the scylla_repo url.
        """
        LOGGER.info("Getting scylla product and major version for upgrade versions listing...")

        if scylla_version is None:
            assert 'unstable/' in self.scylla_repo, "Did not find 'unstable/' in scylla_repo. " \
                                                    "Scylla repo: %s" % self.scylla_repo
            version = self.scylla_repo.split('unstable/')[1].split('/')[1]
            scylla_version = version.replace('branch-', '').replace('enterprise-', '')

        try:
            if actual_version := get_branch_version(self.scylla_repo):
                self.actual_version = actual_version
        except Exception as e:  # noqa: BLE001
            LOGGER.error("Failed to get the actual version: %s", e)

        LOGGER.info("Scylla major version used for upgrade versions listing: , %s", scylla_version)
        return scylla_version

    def set_start_support_version(self, backend: str = None) -> None:
        """
        We can't detect the support versions for this distro, which shares the repo with others, eg: centos8
        so we need to assign the start support versions for it.
        """
        LOGGER.info("Setting start support version...")
        if self.dist_type not in ['debian', 'ubuntu'] and self.linux_distro != 'centos':
            versions_dict = start_support_versions.get(self.linux_distro, None)
            if versions_dict is None:
                raise Exception(
                    f"We can't detect the support versions for this distro, which shares the repo with {self.dist_type}.")
            self.oss_start_support_version = versions_dict['scylla']
            self.ent_start_support_version = versions_dict['enterprise']
        backend_dict = start_support_backend.get(backend, None)
        if backend_dict:
            self.oss_start_support_version = backend_dict['scylla']
            self.ent_start_support_version = backend_dict['enterprise']
        LOGGER.info("Support start versions set: oss=%s enterprise=%s", self.oss_start_support_version,
                    self.ent_start_support_version)

    def get_supported_scylla_base_versions(self, supported_versions) -> list:  # pylint: disable=too-many-branches
        """
        We have special base versions list for each release, and we don't support to upgraded from enterprise
        to opensource. This function is used to get the base versions list which will be used in the upgrade test.

        @supported_version: all scylla release versions, the base versions will be filtered out from the supported_version
        """
        LOGGER.info("Getting supported scylla base versions...")
        all_base_version = []
        all_release_list = [v for v in supported_versions]
        ent_release_list = [v for v in supported_versions if is_enterprise(v)]
        oss_release_list = [v for v in supported_versions if not is_enterprise(v)]
        # The major version of unstable scylla, eg: 4.6.dev, 4.5, 2021.2.dev, 2021.1
        version = self.scylla_version

        if supported_extra_versions := supported_src_oss.get(version):
            all_base_version.extend(supported_extra_versions)
        if supported_extra_versions := supported_src_oss.get(self.actual_version):
            all_base_version.extend(supported_extra_versions)
        if version in supported_versions:
            # The dest version is a released enterprise version
            ent_idx = ent_release_list.index(version) if version in ent_release_list else -1
            all_base_version.append(version)

            if ent_idx >= 0:
                lts_version = re.compile(r'\d{4}\.1')  # lts = long term support
                sts_version = re.compile(r'\d{4}\.2')  # sts = short term support
                ent_idx = ent_release_list.index(version)
                if sts_version.search(version) or ComparableScyllaVersion(version) < '2023.1':
                    # we need to support last LTS version
                    all_base_version += ent_release_list[ent_idx -
                                                         1:][:2] if ent_idx > 0 else [ent_release_list[ent_idx]]
                elif lts_version.search(version):
                    # we need to support last version + last LTS version
                    ent_idx = 2 if ent_idx == 1 else ent_idx
                    all_base_version += ent_release_list[ent_idx -
                                                         2:][:2] if ent_idx > 0 else [ent_release_list[ent_idx]]
                else:
                    LOGGER.warning('enterprise version format not the default - %s', version)
            else:
                relevant_versions = [v for v in oss_release_list if ComparableScyllaVersion(v) < version]
                # If dest version is smaller than the first supported opensource release,
                # it might be an invalid dest version
                all_base_version.append(relevant_versions[-1])
        elif version == 'master':
            all_base_version.append(all_release_list[-1])
        elif re.match(r'\d+.\d+', version):
            current_list = ent_release_list if is_enterprise(version) else oss_release_list
            relevant_versions = [v for v in current_list if ComparableScyllaVersion(v) < version]
            all_base_version += relevant_versions[-2:]

        # Filter out unsupported version
        all_base_version = [v for v in all_base_version if v in supported_versions]

        # if there's only release candidates in those repos, skip this version
        all_base_version = self.filter_rc_only_version(all_base_version)
        LOGGER.info("Supported scylla base versions for: all=%s", all_base_version)
        return list(set(all_base_version))

    def filter_rc_only_version(self, base_version_list: list) -> list:
        LOGGER.info("Filtering rc versions from base version list...")
        base_version_list = sorted(list(set(base_version_list)))
        if base_version_list and self.scylla_version not in ('enterprise', 'master'):
            filter_rc = [v for v in get_all_versions(self.repo_maps[base_version_list[-1]]) if 'rc' not in v]
            if not filter_rc:
                base_version_list = base_version_list[:-1]
        return base_version_list

    def get_version_list(self) -> tuple[list, list]:
        """
        return all supported releases versions, and the supported base versions of upgrade test.
        """
        LOGGER.info("Getting the supported releases version list...")
        supported_versions = []

        # Filter out the unsupported versions
        for version_prefix, _ in self.repo_maps.items():
            # can't find the major version from the version_prefix string
            if not re.match(r'\d+.\d+', version_prefix):
                continue
            # OSS: the major version is smaller than the start support version
            if self.oss_start_support_version and not is_enterprise(version_prefix) and \
                    ComparableScyllaVersion(version_prefix) < self.oss_start_support_version:
                continue
            # Enterprise: the major version is smaller than the start support version
            if self.ent_start_support_version and is_enterprise(version_prefix) and \
                    ComparableScyllaVersion(version_prefix) < self.ent_start_support_version:
                continue
            if version_prefix in unsupported_versions:
                continue
            supported_versions.append(version_prefix)
        version_list = self.get_supported_scylla_base_versions(supported_versions)
        LOGGER.info("Got supported releases version list: %s", version_list)
        return supported_versions, version_list
