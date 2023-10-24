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
# Copyright (c) 2022 ScyllaDB

import pytest

from sdcm.gemini_thread import GeminiStressThread
from unit_tests.dummy_remote import LocalLoaderSetDummy

pytestmark = [
    pytest.mark.usefixtures("events"),
    pytest.mark.integration,
]


class DBCluster:  # pylint: disable=too-few-public-methods
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node_cql_ips(self):
        return [node.cql_address for node in self.nodes]


def test_01_gemini_thread(request, docker_scylla, params):
    loader_set = LocalLoaderSetDummy()
    test_cluster = DBCluster([docker_scylla])
    cmd = (
        "gemini -d --duration 1m --warmup 0s -c 5 -m write --non-interactive --cql-features basic --max-mutation-retries 100 "
        "--max-mutation-retries-backoff 100ms --replication-strategy \"{'class': 'NetworkTopologyStrategy', 'replication_factor': '1'}\" "
        "--table-options \"cdc = {'enabled': true, 'ttl': 0}\" --use-server-timestamps"
    )
    gemini_thread = GeminiStressThread(
        loaders=loader_set,
        stress_cmd=cmd,
        test_cluster=test_cluster,
        oracle_cluster=test_cluster,
        timeout=120,
        params=params,
    )

    def cleanup_thread():
        gemini_thread.kill()

    request.addfinalizer(cleanup_thread)

    gemini_thread.run()

    results = gemini_thread.get_gemini_results()
    gemini_thread.verify_gemini_results(results)


def test_02_gemini_thread(request, docker_scylla, params):
    loader_set = LocalLoaderSetDummy()
    test_cluster = DBCluster([docker_scylla])
    cmd = (
        "gemini -d --duration 1m --warmup 0s --schema unit_tests/test_data/gemini_schema.json -c 5 -m write "
        "--non-interactive --cql-features basic --max-mutation-retries 100 --max-mutation-retries-backoff 100ms"
    )
    gemini_thread = GeminiStressThread(
        loaders=loader_set,
        stress_cmd=cmd,
        test_cluster=test_cluster,
        oracle_cluster=test_cluster,
        timeout=420,
        params=params,
    )

    def cleanup_thread():
        gemini_thread.kill()

    request.addfinalizer(cleanup_thread)

    gemini_thread.run()
    results = gemini_thread.get_gemini_results()
    stats = gemini_thread.verify_gemini_results(results)
    assert stats["results"][0]["write_ops"] > 0, "there were no writes, so possibly something went wrong"
