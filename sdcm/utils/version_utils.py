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

import re
import os
import logging
from enum import Enum, auto
from string import Template
from typing import List, Optional, Literal
from collections import namedtuple, defaultdict
from urllib.parse import urlparse
from functools import lru_cache, wraps

import yaml
import boto3
import requests
import dateutil.parser
from mypy_boto3_s3 import S3Client
from botocore import UNSIGNED
from botocore.client import Config
from repodataParser.RepoParser import Parser

from sdcm.remote import LOCALRUNNER
from sdcm.utils.common import ParallelObject, DEFAULT_AWS_REGION
from sdcm.sct_events.system import ScyllaRepoEvent
from sdcm.utils.decorators import retrying
from sdcm.utils.features import get_enabled_features

# Examples of ScyllaDB version strings:
#   - 666.development-0.20200205.2816404f575
#   - 3.3.rc1-0.20200209.0d0c1d43188
#   - 2019.1.4-0.20191217.b59e92dbd

# gemini version 1.0.1, commit ef7c6f422c78ef6b84a6f3bccf52ea9ec846bba0, date 2019-05-16T09:56:16Z
GEMINI_VERSION_RE = re.compile(r'\s(?P<gemini_version>([\d]+\.[\d]+\.[\d]+)?),')
REPO_VERSIONS_REGEX = re.compile(r'Version: (.*?)\n', re.DOTALL)

# NOTE: following regex is taken from the 'semver' package as is:
#       https://python-semver.readthedocs.io/en/2.10.0/readme.html
SEMVER_REGEX = re.compile(
    r"""
        ^
        (?P<major>0|[1-9]\d*)
        \.
        (?P<minor>0|[1-9]\d*)
        \.
        (?P<patch>0|[1-9]\d*)
        (?:-(?P<prerelease>
            (?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)
            (?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*
        ))?
        (?:\+(?P<build>
            [0-9a-zA-Z-]+
            (?:\.[0-9a-zA-Z-]+)*
        ))?
        $
    """,
    re.VERBOSE,
)

SCYLLA_VERSION_RE = re.compile(r"\d+(\.\d+)?\.[\d\w]+([.~][\d\w]+)?")
ARGUS_VERSION_RE = re.compile(r'((?P<short>[\w.~]+)-(0\.)?(?P<date>[0-9]{8,8})\.(?P<commit>\w+).*)')
SCYLLA_VERSION_GROUPED_RE = re.compile(r'(?P<version>[\w.~]+)-(?P<build>0|rc\d)?\.?(?P<date>[\d]+)\.(?P<commit_id>\w+)')
SSTABLE_FORMAT_VERSION_REGEX = re.compile(r'Feature (.*)_SSTABLE_FORMAT is enabled')
ENABLED_SSTABLE_FORMAT_VERSION_REGEXP = re.compile(r'(.*)_SSTABLE_FORMAT')
PRIMARY_XML_GZ_REGEX = re.compile(r'="(.*?primary.xml.gz)"')

# Example of output for `systemctl --version' command:
#   $ systemctl --version
#   systemd 237
#   +PAM ... default-hierarchy=hybrid
SYSTEMD_VERSION_RE = re.compile(r"^systemd (?P<version>\d+)")

REPOMD_XML_PATH = "repodata/repomd.xml"

SCYLLA_URL_RESPONSE_TIMEOUT = 30
SUPPORTED_XML_EXTENSIONS = ("xml", "xml.gz")
SUPPORTED_FILE_EXTENSIONS = ("list", "repo", "Packages", "gz") + SUPPORTED_XML_EXTENSIONS
VERSION_NOT_FOUND_ERROR = "The URL not supported, only Debian and Yum are supported"

SCYLLA_REPO_BUCKET = "downloads.scylladb.com"
LATEST_SYMLINK_NAME = "latest"
NO_TIMESTAMP = dateutil.parser.parse("1961-04-12T06:07:00Z", ignoretz=True)  # Poyekhali!

LOGGER = logging.getLogger(__name__)


class ScyllaFileType(Enum):
    DEBIAN = auto()
    YUM = auto()


FILE_REGEX_DICT = {
    ScyllaFileType.DEBIAN: [
        (re.compile(r"deb\s+\[arch=(?P<arch>[^\s]*).*?\].*\s(?P<url>http.*?)\s(?P<version_code_name>.*?)\s(?P<component>.*?)$"),
         "{url}/dists/{version_code_name}/{component}/binary-{arch}/Packages"),
    ],
    ScyllaFileType.YUM:
        [
            (re.compile(r"baseurl=(?P<url>http.*scylladb.com.*)"), "{url}" + REPOMD_XML_PATH),
            (re.compile(r"deb\s+(?P<url>http.*?)\s(?P<version_code_name>.*?)\s(?P<component>.*?)$"),
             "{url}/dists/{version_code_name}/{component}/binary-amd64/Packages.gz"),
    ],
}

# The variable "type" indices the type of URL (Debian or Yum)
# The variable "urls" contains all urls using the Scylla pattern
RepositoryDetails = namedtuple("RepositoryDetails", ["type", "urls"])


class ComparableScyllaVersion:
    """Accepts and compares known 'non-semver' and 'semver'-like Scylla versions."""

    def __init__(self, version_string: str):
        parsed_version = self.parse(version_string)
        self.v_major = int(parsed_version[0])
        self.v_minor = int(parsed_version[1])
        self.v_patch = int(parsed_version[2])
        self.v_pre_release = parsed_version[3] or ''
        self.v_build = parsed_version[4] or ''

    @staticmethod
    def parse(version_string: str):
        """Parse scylla-binary and scylla-docker-tag versions into a proper semver structure."""
        # NOTE: remove 'with build-id' part if exists and other possible non-semver parts
        _scylla_version = (version_string or '').split(" ")[0]

        # NOTE: replace '~' which gets returned by the scylla binary
        _scylla_version = _scylla_version.replace('~', '-')

        # NOTE: remove docker-specific parts if version is taken from a docker tag
        _scylla_version = _scylla_version.replace('-aarch64', '')
        _scylla_version = _scylla_version.replace('-x86_64', '')

        # NOTE: transform gce-image version like '2024.2.0.dev.0.20231219.c7cdb16538f2.1'
        if gce_image_v_match := re.search(r"(\d+\.\d+\.\d+\.)([a-z0-9]+\.)(.*)", _scylla_version):
            _scylla_version = f"{gce_image_v_match[1][:-1]}-{gce_image_v_match[2][:-1]}-{gce_image_v_match[3]}"

        # NOTE: make short scylla version like '5.2' be correct semver string
        _scylla_version_parts = _scylla_version.split('.')
        if len(_scylla_version_parts) == 2 and '-' in _scylla_version_parts[1]:
            # NOTE: support for older non-semver release like '2022.1~rc8'
            minor, extra = _scylla_version_parts[1].split('-')
            _scylla_version = f"{_scylla_version_parts[0]}.{minor}.0-{extra}"
        elif len(_scylla_version_parts) == 2:
            _scylla_version = f"{_scylla_version}.0"
        elif len(_scylla_version_parts) > 2 and re.search(
                r"\D+", _scylla_version_parts[2].split("-")[0]):
            _scylla_version = f"{_scylla_version_parts[0]}.{_scylla_version_parts[1]}.0-{_scylla_version_parts[2]}"
            for part in _scylla_version_parts[3:]:
                _scylla_version += f".{part}"

        # NOTE: replace '-0' with 'dev-0', '-1' with 'dev-1' and so on
        #       to match docker and scylla binary version structures correctly.
        if no_dev_match := re.search(r"(\d+\.\d+\.\d+)(\-\d+)(\.20[0-9]{6}.*)", _scylla_version):
            _scylla_version = f"{no_dev_match[1]}-dev{no_dev_match[2]}{no_dev_match[3]}"

        # NOTE: replace '.' with '+' symbol between build date and build commit
        #       to satisfy semver structure
        if dotted_build_id_match := re.search(r"(.*\.20[0-9]{6})(\.)([\.\d\w]+)", _scylla_version):
            _scylla_version = f"{dotted_build_id_match[1]}+{dotted_build_id_match[3]}"

        if match := SEMVER_REGEX.match(_scylla_version):
            return match.groups()
        raise ValueError(
            f"Cannot parse provided '{version_string}' scylla_version for the comparison. "
            f"Transformed scylla_version: {_scylla_version}")

    def __str__(self):
        result = f"{self.v_major}.{self.v_minor}.{self.v_patch}"
        if self.v_pre_release:
            result += f"-{self.v_pre_release}"
        if self.v_build:
            result += f"+{self.v_build}"
        return result

    def _transform_to_comparable(self, other):
        if isinstance(other, str):
            return self.__class__(other)
        elif isinstance(other, self.__class__):
            return other
        raise ValueError("Got unexpected type for the comparison: %s" % type(other))

    def as_comparable(self):
        # NOTE: absence of the 'pre-release' part means we have 'GA' version which is newer than
        #       any of the 'pre-release' ones.
        #       So, make empty 'pre-release' prevail over any defined one.
        return (self.v_major, self.v_minor, self.v_patch, self.v_pre_release or 'xyz')

    def __lt__(self, other):
        return self.as_comparable() < self._transform_to_comparable(other).as_comparable()

    def __le__(self, other):
        return self.as_comparable() <= self._transform_to_comparable(other).as_comparable()

    def __eq__(self, other):
        return self.as_comparable() == self._transform_to_comparable(other).as_comparable()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __gt__(self, other):
        return not self.__le__(other)


class ComparableScyllaOperatorVersion(ComparableScyllaVersion):
    """Accepts and compares known 'non-semver' and 'semver'-like Scylla Operator versions."""

    def as_comparable(self):
        v_pre_release = ''
        if not self.v_pre_release:
            v_pre_release = 'xyz'
        else:
            # 'alpha.0-100-gf796b97'
            dash_parts = self.v_pre_release.split("-")
            transformed_dash_parts = []
            # ['alpha.0', '100', 'gf796b97']
            for dash_part in dash_parts:
                dot_parts = dash_part.split('.')
                transformed_dot_parts = []
                # ['alpha', '0'] , ['100'] , ['gf796b97']
                for dot_part in dot_parts:
                    if re.search(r"\D+", dot_part):
                        transformed_dot_parts.append(dot_part)
                    else:
                        # ['alpha', '0000'] , ['0100'] , ['gf796b97']
                        transformed_dot_parts.append(dot_part.zfill(4))
                transformed_dash_parts.append('.'.join(transformed_dot_parts))
            v_pre_release = '-'.join(transformed_dash_parts)
        return (self.v_major, self.v_minor, self.v_patch, v_pre_release)

    @staticmethod
    def parse(version_string: str):
        """Parse scylla-operator versions into a proper semver structure."""
        _scylla_operator_version = version_string or ''

        # NOTE: remove redundant prefixes and suffixes if exist
        for prefix in ('scylla-operator-', 'scylla-manager-', 'scylla-', 'v'):
            if _scylla_operator_version.startswith(prefix):
                _scylla_operator_version = _scylla_operator_version[len(prefix):]
        for suffix in ('-nightly', ):
            if _scylla_operator_version.endswith(suffix):
                _scylla_operator_version = _scylla_operator_version[:-len(suffix)]

        if match := SEMVER_REGEX.match(_scylla_operator_version):
            return match.groups()
        raise ValueError(
            f"Cannot parse provided '{version_string}' scylla_operator_version for the comparison. "
            f"Transformed scylla_operator_version: {_scylla_operator_version}")


@lru_cache(maxsize=1024)
@retrying(n=10, sleep_time=0.1)
def get_url_content(url, return_url_data=True):
    response = requests.get(url=url)
    if response.status_code != 200:
        raise ValueError(f"The following repository URL '{url}' is incorrect")
    response_data = response.text
    if not response_data:
        raise ValueError(f"The repository URL '{url}' not contains any content")
    if return_url_data:
        return response_data.split('\n')
    # To overcome on Pylint's "inconsistent-return-statements", a value must be returned    return []
    return []


def get_scylla_urls_from_repository(repo_details):
    urls = set()
    for url in repo_details.urls:
        match = None
        url_format = None
        for (url_regex, url_format) in FILE_REGEX_DICT[repo_details.type]:
            match = url_regex.match(url)
            if match:
                break
        if not match or not url_format:
            continue
        url_details = {**match.groupdict()}

        archs = url_details.get('arch', None)
        if archs is None:
            archs = [None]
        else:
            archs = archs.split(',')

        for arch in archs:
            if arch == '':
                continue
            full_url = url_format.format(**{**url_details, 'arch': arch})
            # for scylla-manager we never used a noarch key
            basearch_list = ["x86_64"] if 'scylla-manager' in full_url else ["x86_64", "noarch"]
            for basearch in basearch_list:
                urls.add(Template(full_url).substitute(basearch=basearch, releasever='7'))
            # We found the correct regex and we can continue to next URL

    ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=lambda _url: get_url_content(
        url=_url, return_url_data=False))
    return urls


def get_branch_version_from_debian_repository(urls):
    def get_version(url):
        data = '\n'.join(get_url_content(url=url))
        # Get only the major version (i.e. "2019.1.1-0.20190709.9f724fedb-1~stretch", get only "2019.1.1")
        major_versions = [version.split('-', maxsplit=1)[0] for version in REPO_VERSIONS_REGEX.findall(data)]
        if not major_versions:
            return ""
        return max(set(major_versions), key=ComparableScyllaVersion)

    threads = ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=get_version)
    result = [thread.result for thread in threads]
    return max(result, key=ComparableScyllaVersion)


def get_branch_version_from_centos_repository(urls):
    def get_version(url):
        data = '\n'.join(get_url_content(url=url))
        primary_path = PRIMARY_XML_GZ_REGEX.search(data).groups()[0]
        xml_url = url.replace(REPOMD_XML_PATH, primary_path)

        parser = Parser(url=xml_url)
        major_versions = [package['version'][1]['ver'] for package in parser.getList()]
        return max(set(major_versions), key=ComparableScyllaVersion)

    threads = ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=get_version)
    result = [thread.result for thread in threads]
    return max(result, key=ComparableScyllaVersion)


def get_all_versions_from_debian_repository(urls: set[str]) -> set[str]:
    def get_version(url: str) -> set[str]:
        data = '\n'.join(get_url_content(url=url))
        # Get only the major version (i.e. "2019.1.1-0.20190709.9f724fedb-1~stretch", get only "2019.1.1")
        major_versions = [version.split('-', maxsplit=1)[0] for version in REPO_VERSIONS_REGEX.findall(data)]
        return set(major_versions)

    threads = ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=get_version)
    result = set.union(*[thread.result for thread in threads])
    return result


def get_all_versions_from_centos_repository(urls: set[str]) -> set[str]:
    def get_version(url: str) -> set[str]:
        data = '\n'.join(get_url_content(url=url))
        primary_path = PRIMARY_XML_GZ_REGEX.search(data).groups()[0]
        xml_url = url.replace(REPOMD_XML_PATH, primary_path)

        parser = Parser(url=xml_url)
        major_versions = [package['version'][1]['ver'] for package in parser.getList()]
        return set(major_versions)

    threads = ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=get_version)
    result = set.union(*[thread.result for thread in threads])
    return result


def get_repository_details(url):
    urls = list({*get_url_content(url=url)})

    for file_type, regex_list in FILE_REGEX_DICT.items():
        for _url in urls:
            for (url_regex, _) in regex_list:
                match = url_regex.match(_url)
                if match:
                    return RepositoryDetails(type=file_type, urls=urls)
    raise ValueError(VERSION_NOT_FOUND_ERROR)


def get_branch_version(url):
    repo_details = get_repository_details(url=url)
    urls = get_scylla_urls_from_repository(repo_details=repo_details)

    if repo_details.type == ScyllaFileType.DEBIAN:
        return get_branch_version_from_debian_repository(urls=urls)
    elif repo_details.type == ScyllaFileType.YUM:
        return get_branch_version_from_centos_repository(urls=urls)
    # To overcome on Pylint's "inconsistent-return-statements", a value must be returned
    return []


def get_all_versions(url: str) -> set[str]:
    repo_details = get_repository_details(url=url)
    urls = get_scylla_urls_from_repository(repo_details=repo_details)

    if repo_details.type == ScyllaFileType.DEBIAN:
        return get_all_versions_from_debian_repository(urls=urls)
    elif repo_details.type == ScyllaFileType.YUM:
        return get_all_versions_from_centos_repository(urls=urls)
    # To overcome on Pylint's "inconsistent-return-statements", a value must be returned
    return set()


def get_branch_version_for_multiple_repositories(urls):
    threads = ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=get_branch_version)
    return [thread.result for thread in threads]


def is_enterprise(scylla_version):
    """
    :param scylla_version: scylla version string
    :return: True if this version string passed is a scylla enterprise version
    """
    return bool(re.search(r"^20[0-9]{2}.*", scylla_version))


def assume_version(params: dict[str], scylla_version: Optional[str] = None) -> tuple[bool, str]:
    # Try to get the major version from the branch name, it will only be used when scylla_version isn't assigned.
    # It can be switched to RELEASE_BRANCH from upstream job
    git_branch = os.environ.get('GIT_BRANCH')  # origin/branch-4.5
    scylla_repo = params.get('scylla_repo')

    scylla_version_source = scylla_version or scylla_repo or git_branch
    LOGGER.debug("scylla_version_source: %s", scylla_version_source)
    if match := re.match(r'[\D\d]*-(\d+\.\d+)', scylla_version_source) or \
            re.match(r'\D*(\d+\.\d+)', scylla_version_source):
        version_type = f"nightly-{match.group(1)}"
        is_enterprise_version = is_enterprise(match.group(1))
    elif scylla_version_source and 'enterprise' in scylla_version_source:
        version_type = "nightly-enterprise"
        is_enterprise_version = True
    elif scylla_version_source and 'master' in scylla_version_source:
        version_type = "nightly-master"
        is_enterprise_version = False
    else:
        raise Exception("Scylla version for web install isn't identified")
    return is_enterprise_version, version_type


def get_gemini_version(output: str):
    # take only version number - 1.0.1
    result = GEMINI_VERSION_RE.search(output)

    if result:
        return result.groupdict().get("gemini_version", None)
    return None


def get_node_supported_sstable_versions(node_system_log) -> List[str]:
    output = []
    with open(node_system_log, encoding="utf-8") as file:
        for line in file.readlines():
            if match := SSTABLE_FORMAT_VERSION_REGEX.search(line):
                output.append(match.group(1).lower())
    return output


def get_node_enabled_sstable_version(session) -> list[str]:
    enabled_sstable_format_features = []
    all_features = get_enabled_features(session)
    for feature in all_features:
        if match := ENABLED_SSTABLE_FORMAT_VERSION_REGEXP.search(feature):
            enabled_sstable_format_features.append(match.group(1).lower())
    return enabled_sstable_format_features


def get_systemd_version(output: str) -> int:
    if match := SYSTEMD_VERSION_RE.match(output):
        try:
            return int(match.group("version"))
        except ValueError:
            pass
    return 0


def get_scylla_docker_repo_from_version(scylla_version: str):
    if scylla_version in ('latest', 'master:latest'):
        return 'scylladb/scylla-nightly'
    if scylla_version in ('enterprise', 'enterprise:latest'):
        return 'scylladb/scylla-enterprise-nightly'
    if is_enterprise(scylla_version):
        return 'scylladb/scylla-enterprise'
    return 'scylladb/scylla'


def _list_repo_file_etag(s3_client: S3Client, prefix: str) -> Optional[dict]:
    repo_file = s3_client.list_objects_v2(Bucket=SCYLLA_REPO_BUCKET, Prefix=prefix)
    if repo_file["KeyCount"] != 1:
        LOGGER.debug("No such file `%s' in %s bucket", prefix, SCYLLA_REPO_BUCKET)
        return None
    return repo_file["Contents"][0]["ETag"]


def resolve_latest_repo_symlink(url: str) -> str:
    """Resolve `url' to the actual repo link if it contains `latest' substring, otherwise, return `url' as is.

    If `url' doesn't point to the latest repo file then raise ScyllaRepoEvent (warning severity).

    Can raise ValueError if `url' is not a valid URL that points to a repo file stored in S3.
    """
    # pylint: disable=too-many-branches
    base, sep, rest = url.partition(LATEST_SYMLINK_NAME)
    if sep != LATEST_SYMLINK_NAME:
        LOGGER.debug("%s doesn't contain `%s' substring, use URL as is", url, LATEST_SYMLINK_NAME)
        return url

    parsed_base_url = urlparse(base)

    # URL can be in 3 forms:
    #  1. http://downloads.scylladb.com/...
    #  2. http://downloads.scylladb.com.s3.amazonaws.com/...
    #  3. http://s3.amazonaws.com/downloads.scylladb.com/...
    # Plus same forms for HTTPS.
    if parsed_base_url.netloc in (SCYLLA_REPO_BUCKET, f"{SCYLLA_REPO_BUCKET}.s3.amazonaws.com", ):
        prefix = parsed_base_url.path.lstrip("/")
    elif parsed_base_url.netloc == "s3.amazonaws.com" and parsed_base_url.path.startswith(f"/{SCYLLA_REPO_BUCKET}/"):
        prefix = parsed_base_url.path.split("/", 2)[-1]
    else:
        raise ValueError(f"Unsupported URL: {url}")

    s3_client: S3Client = boto3.client("s3", region_name=DEFAULT_AWS_REGION, config=Config(signature_version=UNSIGNED))

    latest_etag = _list_repo_file_etag(s3_client=s3_client, prefix=f"{prefix}{LATEST_SYMLINK_NAME}{rest}")
    if latest_etag is None:
        raise ValueError(f"{url} doesn't point to a file stored in S3")

    build_list = []
    s3_objects = s3_client.list_objects_v2(Bucket=SCYLLA_REPO_BUCKET, Delimiter="/", Prefix=prefix)
    continuation_token = "BEGIN"
    while continuation_token:
        for build in s3_objects.get("CommonPrefixes", []):
            build = build.get("Prefix", "").rstrip("/").rsplit("/", 1)[-1]  # noqa: PLW2901
            if build == LATEST_SYMLINK_NAME:
                continue
            timestamp = NO_TIMESTAMP
            if len(build) >= 12:  # `build' should be a string like `202001010000' or `2020-01-01T00:00:00Z'
                try:
                    timestamp = dateutil.parser.parse(build, ignoretz=True)
                except ValueError:
                    pass
            build_list.append((timestamp, build, ))
        if continuation_token := s3_objects.get("NextContinuationToken"):
            s3_objects = s3_client.list_objects_v2(
                Bucket=SCYLLA_REPO_BUCKET,
                Delimiter="/",
                Prefix=prefix,
                ContinuationToken=continuation_token,
            )
    build_list.sort(reverse=True)

    for timestamp, build in build_list:
        if _list_repo_file_etag(s3_client=s3_client, prefix=f"{prefix}{build}{rest}") == latest_etag:
            break
    else:
        ScyllaRepoEvent(
            url=url,
            error=f"There is no a sibling directory which contains same repo file (ETag={latest_etag})"
        ).publish_or_dump(default_logger=LOGGER)
        return url

    if (timestamp, build) != build_list[0]:
        ScyllaRepoEvent(
            url=url,
            error=f"{url} doesn't point to the latest repo ({base}{build_list[0][1]}{rest})"
        ).publish_or_dump(default_logger=LOGGER)

    resolved_url = f"{base}{build}{rest}"
    LOGGER.debug("%s resolved to %s", url, resolved_url)
    return resolved_url


def get_specific_tag_of_docker_image(docker_repo: str):
    if docker_repo == 'scylladb/scylla-nightly':
        product = 'scylla'
        branch = 'master'
    elif docker_repo == 'scylladb/scylla-enterprise-nightly':
        product = 'scylla-enterprise'
        branch = 'enterprise'
    else:
        raise ValueError(f"SCT doesn't support getting latest from {docker_repo}")

    build_url = f'https://s3.amazonaws.com/downloads.scylladb.com/unstable/{product}/{branch}/relocatable/latest/00-Build.txt'
    res = requests.get(build_url)
    res.raise_for_status()
    # example of 00-Build.txt content: (each line is formatted as 'key: value`)
    #
    #    url-id: 2022-08-29T08:05:34Z
    #    docker-image-name: scylla-nightly:5.2.0-dev-0.20220829.67c91e8bcd61
    build_info = yaml.safe_load(res.content)
    tag = build_info['docker-image-name'].split(':', maxsplit=1)[1]
    LOGGER.debug('found %s for %s repo', tag, docker_repo)
    return tag


def transform_non_semver_scylla_version_to_semver(scylla_version: str):
    # NOTE: as of May 2022 all the non-GA versions of Scylla are not semver-like and it is problem
    #       for the scylla-operator.
    if SEMVER_REGEX.match(scylla_version):
        return scylla_version
    version_parts = scylla_version.split(".")
    if len(version_parts) > 2:
        new_scylla_version = f"{version_parts[0]}.{version_parts[1]}.0-{'.'.join(version_parts[2:])}"
        if SEMVER_REGEX.match(new_scylla_version):
            return new_scylla_version
    raise ValueError("Cannot transform '%s' to semver-like string" % scylla_version)


def get_git_tag_from_helm_chart_version(chart_version: str) -> str:
    """Utility function used to parse out the git tag from a Helm chart version

    Designed to be used for 'scylla-operator' helm charts.
    See below the expected mapping of possible helm chart version structures
    and git tags:

    +-----------------------------------+----------------+
    | Helm chart version                | Git tag        |
    +-----------------------------------+----------------+
    | v1.1.0-rc.1                       | v1.1.0-rc.1    |
    | v1.1.0-rc.1-1-g6d35b37            | v1.1.0-rc.1    |
    | v1.1.0-alpha.0-3-g6594091-nightly | v1.1.0-alpha.0 |
    | v1.1.0-alpha.0-3-g6594091         | v1.1.0-alpha.0 |
    | v1.0.0                            | v1.0.0         |
    | v1.0.0-39-g5bc1839                | v1.0.0         |
    | v1.0.0-rc0-53-g489398a-nightly    | v1.0.0-rc0     |
    | v1.0.0-rc0-53-g489398a            | v1.0.0-rc0     |
    | v1.0.0-rc0-51-ga52c206-latest     | v1.0.0-rc0     |
    +-----------------------------------+----------------+
    """
    pattern = "^(v[a-z0-9.-]+)-[0-9]{1,3}-g[0-9a-z]{7}|(v[a-z0-9.-]+){1}$"
    search_result = re.search(pattern, chart_version)
    if search_result:
        git_tag = search_result.group(1) or search_result.group(2)
    else:
        raise ValueError(f"Got wrong chart version: {chart_version}")
    return git_tag


class MethodVersionNotFound(Exception):
    pass


class scylla_versions:  # pylint: disable=invalid-name,too-few-public-methods
    """Runs a versioned method that is suitable for the configured scylla version.

    Limitations:
        - Can't work if the same method name in the same file is used in different classes
        - Can't work with 'static' and 'class' methods. Only 'class instance' ones.
        - Depends on the 'params' or 'cluster.params' attributes presence
          which store SCT configuration.
    Example:
        # Having "scylla_version" be set to "4.4.4" in the sct config

        In [3]: class Foo():
           ...:
           ...:     @scylla_versions((None, "4.3"))
           ...:     def foo(self):
           ...:         return "any 4.3.x and lower"
           ...:
           ...:     @scylla_versions(("4.4.rc1", "4.4.rc1"), ("4.4.rc4", "4.5"))
           ...:     def foo(self):
           ...:         return "all 4.4 and 4.5 except 4.4.rc2 and 4.4.rc3"
           ...:
           ...:     @scylla_versions(("4.6.rc1", None))
           ...:     def foo(self):
           ...:         return "4.6.rc1 and higher"

        In [4]: Foo().foo()
        Out[4]: 'all 4.4 and 4.5 except 4.4.rc2 and 4.4.rc3'
    """
    VERSIONS = {}

    def __init__(self, *min_max_version_pairs: tuple):
        self.min_max_version_pairs = min_max_version_pairs

    def __call__(self, func):
        if (func.__name__, func.__code__.co_filename) not in self.VERSIONS:
            self.VERSIONS[(func.__name__, func.__code__.co_filename)] = {}
        for min_v, max_v in self.min_max_version_pairs:
            scylla_type = "enterprise" if any((is_enterprise(v) for v in (min_v, max_v) if v)) else "oss"
            min_v = min_v or ("3.0.0" if scylla_type == "oss" else "2019.1.rc0")  # noqa: PLW2901
            max_v = max_v or ("99.99.99" if scylla_type == "oss" else "2099.99.99")  # noqa: PLW2901
            if max_v.count(".") == 1:
                # NOTE: version parse function considers 4.4 as lower than 4.4.1,
                #       but we expect it to be any of the 4.4.x versions.
                #       So, update all such short versions with the patch part and make it to be huge.
                max_v = f"{max_v}.999"  # noqa: PLW2901
            self.VERSIONS[(func.__name__, func.__code__.co_filename)].update({(min_v, max_v): func})

        @wraps(func)
        def inner(*args, **kwargs):
            cls_self = args[0]
            try:
                cluster_object = getattr(cls_self, 'cluster', cls_self)
                scylla_version = cluster_object.params.get("scylla_version")
                if not scylla_version or scylla_version.endswith(":latest"):
                    # NOTE: in case we run Scylla cluster with "latest" version then we need
                    #       to pick up a more precise version from one of it's nodes, i.e. '4.7.dev'
                    scylla_version = cluster_object.nodes[0].artifact_scylla_version
            except (AttributeError, IndexError) as exc:
                LOGGER.warning("Failed to get scylla version: %s", exc)
                scylla_version = "n/a"
            func_version_mapping = self.VERSIONS.get((func.__name__, func.__code__.co_filename), {})
            for (min_v, max_v), mapped_func in func_version_mapping.items():
                try:
                    if ComparableScyllaVersion(min_v) <= ComparableScyllaVersion(scylla_version) <= max_v:
                        return mapped_func(*args, **kwargs)
                except ValueError as exc:
                    LOGGER.warning("Failed to parse '%s' scylla version: %s", scylla_version, exc)
            raise MethodVersionNotFound(
                "Method '{}' with version '{}' is not supported in '{}'!".format(
                    func.__name__, scylla_version, cls_self.__class__.__name__))
        return inner


def get_relocatable_pkg_url(scylla_version: str) -> str:
    relocatable_pkg = ""
    if scylla_version and "build-id" in scylla_version:
        try:
            scylla_build_id = scylla_version.split('build-id')[-1].split()[0]
            get_pkgs_cmd = f'curl -s -X POST http://backtrace.scylladb.com/index.html -d "build_id={scylla_build_id}&backtrace="'
            res = LOCALRUNNER.run(get_pkgs_cmd)
            relocatable_pkg = re.findall(fr"{scylla_build_id}.+(http:[/\w.:-]*\.tar\.gz)", res.stdout)[0]
        except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
            LOGGER.warning("Couldn't get relocatable_pkg link due to: %s", exc)
    return relocatable_pkg


_S3_SCYLLA_REPOS_CACHE = defaultdict(dict)


def get_s3_scylla_repos_mapping(dist_type='centos', dist_version=None):
    """
    get the mapping from version prefixes to rpm .repo or deb .list files locations

    :param dist_type: which distro to look up centos/ubuntu/debian
    :param dist_version: family name of the distro version

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
                    dist_type, dist_version)][version_prefix] = "https://s3.amazonaws.com/{bucket}/{path}".format(
                    bucket=bucket,
                    path=repo_file['Key'])

    elif dist_type in ('ubuntu', 'debian'):
        response = s3_client.list_objects(Bucket=bucket, Prefix='deb/{}/'.format(dist_type), Delimiter='/')
        for repo_file in response['Contents']:
            filename = os.path.basename(repo_file['Key'])

            # only if path look like 'deb/debian/scylla-3.0-jessie.list', we deem it formal one
            repo_regex = re.compile(r'\d+\.\d\.list')
            if filename.startswith('scylla-') and (
                    filename.endswith('-{}.list'.format(dist_version)) or
                    repo_regex.search(filename)):
                version_prefix = \
                    filename.replace('-{}.list'.format(dist_version), '').replace('.list', '').split('-')[-1]
                _S3_SCYLLA_REPOS_CACHE[(
                    dist_type, dist_version)][version_prefix] = "https://s3.amazonaws.com/{bucket}/{path}".format(
                    bucket=bucket,
                    path=repo_file['Key'])

    else:
        raise NotImplementedError("[{}] is not yet supported".format(dist_type))
    return _S3_SCYLLA_REPOS_CACHE[(dist_type, dist_version)]


def find_scylla_repo(scylla_version, dist_type='centos', dist_version=None):
    """
    Get a repo/list of scylla, based on scylla version match

    :param scylla_version: branch version to look for, ex. 'branch-2019.1:latest', 'branch-3.1:l'
    :param dist_type: one of ['centos', 'ubuntu', 'debian']
    :param dist_version: family name of the distro version
    :raises: ValueError if not found

    :return: str url repo/list
    """
    if ':' in scylla_version:
        branch_repo = get_branched_repo(scylla_version, dist_type)
        if branch_repo:
            return branch_repo

    repo_map = get_s3_scylla_repos_mapping(dist_type, dist_version)

    # pylint: disable=useless-else-on-loop
    for key in repo_map:
        if scylla_version.startswith(key):
            return repo_map[key]
    else:  # noqa: PLW0120
        raise ValueError(f"repo for scylla version {scylla_version} wasn't found")


def get_branched_repo(scylla_version: str,
                      dist_type: Literal["centos", "ubuntu", "debian"] = "centos",
                      bucket: str = "downloads.scylladb.com") -> Optional[str]:
    """
    Get a repo/list of scylla, based on scylla version match

    :param scylla_version: branch version to look for, ex. 'branch-2019.1:latest', 'branch-3.1:l'
    :param dist_type: one of ['centos', 'ubuntu', 'debian']
    :param bucket: which bucket to download from
    :return: str url repo/list, or None if not found
    """
    try:
        branch, branch_version = scylla_version.split(':', maxsplit=1)
    except ValueError:
        raise ValueError(f"{scylla_version=} should be in `branch-x.y:<date>' or `branch-x.y:latest' format") from None

    branch_id = branch.replace('branch-', '').replace('enterprise-', '')
    product = 'scylla-enterprise' if branch == 'enterprise' or is_enterprise(branch_id) else 'scylla'

    if dist_type == "centos":
        prefix = f"unstable/{product}/{branch}/rpm/centos/{branch_version}/"
        filename = "scylla.repo"
    elif dist_type in ("ubuntu", "debian",):
        prefix = f"unstable/{product}/{branch}/deb/unified/{branch_version}/scylladb-{branch_id}/"
        filename = "scylla.list"
    else:
        raise ValueError(f"Unsupported {dist_type=}")

    s3_client: S3Client = boto3.client("s3", region_name=DEFAULT_AWS_REGION)
    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')

    for repo_file in response.get("Contents", ()):
        if os.path.basename(repo_file['Key']) == filename:
            return f"https://s3.amazonaws.com/{bucket}/{repo_file['Key']}"

    if branch_version.isdigit():
        LOGGER.warning("Repo path doesn't include `build-id' anymore, try to use a date.")

    return None
