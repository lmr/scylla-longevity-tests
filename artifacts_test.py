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
from functools import cached_property

import datetime
import re

from sdcm.cluster import SCYLLA_YAML_PATH
from sdcm.tester import ClusterTester
# TODO: uncomment when problem with SSL connection to MYSQL will be solved
# from sdcm.utils.housekeeping import HousekeepingDB

STRESS_CMD: str = "/usr/bin/cassandra-stress"


class ArtifactsTest(ClusterTester):
    # TODO: uncomment when problem with SSL connection to MYSQL will be solved
    # REPO_TABLE = "housekeeping.repo"
    # CHECK_VERSION_TABLE = "housekeeping.checkversion"

    # TODO: uncomment setUp and tearDown when problem with SSL connection to MYSQL will be solved
    # def setUp(self) -> None:
    #     super().setUp()
    #
    #     self.housekeeping = HousekeepingDB.from_keystore_creds()
    #     self.housekeeping.connect()
    #
    # def tearDown(self) -> None:
    #     self.housekeeping.close()
    #
    #     super().tearDown()

    # TODO: uncomment check_version when problem with SSL connection to MYSQL will be solved
    # def check_scylla_version_in_housekeepingdb(self, prev_id: int, expected_status_code: str, new_row_expected: bool) -> int:
    #     """
    #     Validate reported version
    #     prev_id: check if new version is created
    #     """
    #     assert self.node.uuid, "Node UUID wasn't created"
    #
    #     row = self.housekeeping.get_most_recent_record(query=f"SELECT id, version, ip, statuscode "
    #                                                          f"FROM {self.CHECK_VERSION_TABLE} "
    #                                                          f"WHERE uuid = %s", args=(self.node.uuid,))
    #     self.log.debug(f"Last row in {self.CHECK_VERSION_TABLE} for uuid '{self.node.uuid}': {row}")
    #
    #     # Validate public IP address
    #     assert self.node.public_ip_address == row[2], \
    #         f"Wrong IP address is saved in {self.CHECK_VERSION_TABLE}: expected {self.node.public_ip_address}, " \
    #         f"got: {row[2]}"
    #
    #     # Validate reported node version
    #     assert row[1] == self.version, \
    #         f"Wrong version is saved in {self.CHECK_VERSION_TABLE}: " \
    #         f"expected {self.node.public_ip_address}, got: {row[2]}"
    #
    #     # Validate expected status code
    #     assert row[3] == expected_status_code, \
    #         f"Wrong statuscode is saved in {self.CHECK_VERSION_TABLE}: " \
    #         f"expected {expected_status_code}, got: {row[3]}"
    #
    #     if prev_id:
    #         # Validate row id
    #         if new_row_expected:
    #             assert row[0] > prev_id, f"New row wasn't saved in {self.CHECK_VERSION_TABLE}"
    #         else:
    #             assert row[0] == prev_id, f"New row was saved in {self.CHECK_VERSION_TABLE} unexpectedly"
    #
    #     return row[0] if row else 0

    @property
    def node(self):
        if self.db_cluster is None or not self.db_cluster.nodes:
            raise ValueError('DB cluster has not been initiated')
        return self.db_cluster.nodes[0]

    @cached_property
    def version(self):
        version = self.node.get_scylla_version()
        return version.replace('scylladb-', '')

    def check_cluster_name(self):
        with self.node.remote_scylla_yaml(SCYLLA_YAML_PATH) as scylla_yaml:
            yaml_cluster_name = scylla_yaml.get('cluster_name', '')

        self.assertTrue(self.db_cluster.name == yaml_cluster_name,
                        f"Cluster name is not as expected. Cluster name in scylla.yaml: {yaml_cluster_name}. "
                        f"Cluster name: {self.db_cluster.name}")

    def run_cassandra_stress(self, args: str):
        result = self.node.remoter.run(
            f"{self.node.add_install_prefix(STRESS_CMD)} {args} -node {self.node.ip_address}")
        assert "java.io.IOException" not in result.stdout
        assert "java.io.IOException" not in result.stderr

    def check_scylla(self):
        self.node.run_nodetool("status")
        self.run_cassandra_stress("write n=10000 -mode cql3 native -pop seq=1..10000")
        self.run_cassandra_stress("mixed duration=1m -mode cql3 native -rate threads=10 -pop seq=1..10000")

    def verify_users(self):
        # We can't ship the image with Scylla internal users inside. So we
        # need to verify that mistakenly we didn't create all the users that we have project wide in the image
        self.log.info("Checking that all existent users except centos were created after boot")
        uptime = self.node.remoter.run(cmd="uptime -s").stdout.strip()
        datetime_format = "%Y-%m-%d %H:%M:%S"
        instance_start_time = datetime.datetime.strptime(uptime, datetime_format)
        self.log.info("Instance started at: %s", instance_start_time)
        out = self.node.remoter.run(cmd="ls -ltr --full-time /home", verbose=True).stdout.strip()
        for line in out.splitlines():
            splitted_line = line.split()
            if len(splitted_line) <= 2:
                continue
            user = splitted_line[-1]
            if user == "centos":
                self.log.info("Skipping user %s since it is a default image user.", user)
                continue
            self.log.info("Checking user: '%s'", user)
            if datetime_str := re.search(r"(\d+-\d+-\d+ \d+:\d+:\d+).", line):
                datetime_user_created = datetime.datetime.strptime(datetime_str.group(1), datetime_format)
                self.log.info("User '%s' created at '%s'", user, datetime_user_created)
                if datetime_user_created < instance_start_time and not user == "centos":
                    AssertionError("User %s was created in the image. Only user centos should exist in the image")
            else:
                raise AssertionError(f"Unable to parse/find timestamp of the user {user} creation in {line}")
        self.log.info("All users except image user 'centos' were created after the boot.")

    def test_scylla_service(self):
        if self.params["cluster_backend"] == "aws":
            with self.subTest("check ENA support"):
                assert self.node.ena_support, "ENA support is not enabled"

        if self.params["cluster_backend"] == "gce":
            self.verify_users()

        if self.params["use_preinstalled_scylla"] and "docker" not in self.params["cluster_backend"]:
            with self.subTest("check the cluster name"):
                self.check_cluster_name()

        with self.subTest("check Scylla server after installation"):
            self.check_scylla()

            # TODO: uncomment when problem with SSL connection to MYSQL will be solved
            # self.log.info(f"Validate version after install")
            # version_id_after_install = self.check_scylla_version_in_housekeepingdb(prev_id=0,
            #                                                                        expected_status_code='i',
            #                                                                        new_row_expected=False)

        with self.subTest("check Scylla server after stop/start"):
            self.node.stop_scylla(verify_down=True)
            self.node.start_scylla(verify_up=True)
            self.check_scylla()

            # TODO: uncomment when problem with SSL connection to MYSQL will be solved
            # self.log.info(f"Validate version after stop/start")
            # version_id_after_stop = self.check_scylla_version_in_housekeepingdb(prev_id=version_id_after_install,
            #                                                                     expected_status_code='r',
            #                                                                     new_row_expected=True)

        with self.subTest("check Scylla server after restart"):
            self.node.restart_scylla(verify_up_after=True)
            self.check_scylla()

            # TODO: uncomment when problem with SSL connection to MYSQL will be solved
            # self.log.info(f"Validate version after restart")
            # self.check_scylla_version_in_housekeepingdb(prev_id=version_id_after_stop,
            #                                             expected_status_code='r',
            #                                             new_row_expected=True)

    def get_email_data(self):
        self.log.info("Prepare data for email")
        email_data = self._get_common_email_data()
        try:
            node = self.node
        except:  # pylint: disable=bare-except
            node = None
        if node:
            scylla_packages = node.scylla_packages_installed
        else:
            scylla_packages = None
        if not scylla_packages:
            scylla_packages = ['No scylla packages are installed. Please check log files.']
        email_data.update({"scylla_node_image": node.image if node else 'Node has not been initialized',
                           "scylla_packages_installed": scylla_packages,
                           "unified_package": self.params.get("unified_package"),
                           "nonroot_offline_install": self.params.get("nonroot_offline_install"),
                           "scylla_repo": self.params.get("scylla_repo"), })

        return email_data
