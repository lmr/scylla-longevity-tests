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
import logging
from enum import Enum, auto
from string import Template
from typing import List, Optional
from collections import namedtuple
from urllib.parse import urlparse

import boto3
import requests
import dateutil.parser
from mypy_boto3_s3 import S3Client
from botocore import UNSIGNED
from botocore.client import Config
from pkg_resources import parse_version
from repodataParser.RepoParser import Parser

from sdcm.utils.common import ParallelObject, DEFAULT_AWS_REGION
from sdcm.sct_events.system import ScyllaRepoEvent


# Examples of ScyllaDB version strings:
#   - 666.development-0.20200205.2816404f575
#   - 3.3.rc1-0.20200209.0d0c1d43188
#   - 2019.1.4-0.20191217.b59e92dbd

# gemini version 1.0.1, commit ef7c6f422c78ef6b84a6f3bccf52ea9ec846bba0, date 2019-05-16T09:56:16Z
GEMINI_VERSION_RE = re.compile(r'\s(?P<gemini_version>([\d]+\.[\d]+\.[\d]+)?),')
REPO_VERSIONS_REGEX = re.compile(r'Version: (.*?)\n', re.DOTALL)

SCYLLA_VERSION_RE = re.compile(r"\d+(\.\d+)?\.[\d\w]+([.~][\d\w]+)?")
SSTABLE_FORMAT_VERSION_REGEX = re.compile(r'Feature (.*)_SSTABLE_FORMAT is enabled')
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
        (re.compile(r"deb\s+\[arch=(?P<arch>.*?)\]\s(?P<url>http.*?)\s(?P<version_code_name>.*?)\s(?P<component>.*?)$"),
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
        for (url_regex, url_format) in FILE_REGEX_DICT[repo_details.type]:
            match = url_regex.match(url)
            if not match:
                # Continue until find the correct Regex
                continue

            full_url = url_format.format(**match.groupdict())
            # for scylla-manager we never used a noarch key
            basearch_list = ["x86_64"] if 'scylla-manager' in full_url else ["x86_64", "noarch"]
            for basearch in basearch_list:
                urls.add(Template(full_url).substitute(basearch=basearch, releasever='7'))
            # We found the correct regex and we can continue to next URL
            break

    urls = list(urls)
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
        return max(set(major_versions), key=major_versions.count)

    threads = ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=get_version)
    result = [thread.result for thread in threads]
    return max(result, key=result.count)


def get_branch_version_from_centos_repository(urls):
    def get_version(url):
        data = '\n'.join(get_url_content(url=url))
        primary_path = PRIMARY_XML_GZ_REGEX.search(data).groups()[0]
        xml_url = url.replace(REPOMD_XML_PATH, primary_path)

        parser = Parser(url=xml_url)
        major_versions = [package['version'][1]['ver'] for package in parser.getList()]
        return max(set(major_versions), key=major_versions.count)

    threads = ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=get_version)
    result = [thread.result for thread in threads]
    return max(result, key=result.count)


def get_repository_details(url):
    urls = list({line for line in get_url_content(url=url)})

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


def get_branch_version_for_multiple_repositories(urls):
    threads = ParallelObject(objects=urls, timeout=SCYLLA_URL_RESPONSE_TIMEOUT).run(func=get_branch_version)
    return [thread.result for thread in threads]


def is_enterprise(version):
    """
    :param version: version string
    :return: True if this version string passed is a scylla enterprise version
    """
    return parse_version(version) > parse_version('2000')


def get_gemini_version(output: str):
    # take only version number - 1.0.1
    result = GEMINI_VERSION_RE.search(output)

    if result:
        return result.groupdict().get("gemini_version", None)
    return None


def get_node_supported_sstable_versions(node_system_log) -> List[str]:
    output = []
    with open(node_system_log) as file:
        for line in file.readlines():
            if match := SSTABLE_FORMAT_VERSION_REGEX.search(line):
                output.append(match.group(1).lower())
    return output


def get_systemd_version(output: str) -> int:
    if match := SYSTEMD_VERSION_RE.match(output):
        try:
            return int(match.group("version"))
        except ValueError:
            pass
    return 0


def get_scylla_docker_repo_from_version(version: str):
    if version == 'latest':
        return 'scylladb/scylla-nightly'
    if is_enterprise(version):
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
    base, sep, rest = url.partition(LATEST_SYMLINK_NAME)
    if sep != LATEST_SYMLINK_NAME:
        LOGGER.info("%s doesn't contain `%s' substring, use URL as is", url, LATEST_SYMLINK_NAME)
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
            build = build.get("Prefix", "").rstrip("/").rsplit("/", 1)[-1]
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
        LOGGER.info("There is no a sibling directory which contains same repo file, use URL %s as is", url)
        return url

    if (timestamp, build) != build_list[0]:
        ScyllaRepoEvent(
            url=url,
            error=f"{url} doesn't point to the latest repo ({base}{build_list[0][1]}{rest})"
        ).publish_or_dump(default_logger=LOGGER)
        LOGGER.info("Actual latest build is %s, not %s", build_list[0][1], build)

    resolved_url = f"{base}{build}{rest}"
    LOGGER.info("%s resolved to %s", url, resolved_url)
    return resolved_url


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
    pattern = "^(v[a-z0-9.-]+)-[\d]{1,3}-g[0-9a-z]{7}|(v[a-z0-9.-]+){1}$"
    search_result = re.search(pattern, chart_version)
    if search_result:
        git_tag = search_result.group(1) or search_result.group(2)
    else:
        raise ValueError(f"Got wrong chart version: {chart_version}")
    return git_tag
