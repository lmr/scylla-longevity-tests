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
import datetime
import pprint
import re
import typing
from functools import cached_property
import json

import yaml
import requests

from sdcm.sct_events import Severity
from sdcm.sct_events.database import ScyllaHousekeepingServiceEvent
from sdcm.tester import ClusterTester
from sdcm.utils.adaptive_timeouts import NodeLoadInfoServices
from sdcm.utils.housekeeping import HousekeepingDB
from sdcm.utils.common import get_latest_scylla_release, ScyllaProduct
from sdcm.utils.decorators import retrying
from sdcm.utils.perftune_validator import PerftuneOutputChecker

STRESS_CMD: str = "/usr/bin/cassandra-stress"


BACKENDS = {
    "aws": ["Ec2Snitch", "Ec2MultiRegionSnitch"],
    "gce": ["GoogleCloudSnitch"],
    "azure": ["AzureSnitch"],
    "docker": ["GossipingPropertyFileSnitch", "SimpleSnitch"]
}


class ArtifactsTest(ClusterTester):  # pylint: disable=too-many-public-methods
    REPO_TABLE = "housekeeping.repo"
    CHECK_VERSION_TABLE = "housekeeping.checkversion"

    def setUp(self) -> None:
        super().setUp()

        self.housekeeping = HousekeepingDB.from_keystore_creds()
        self.housekeeping.connect()

    def tearDown(self) -> None:
        if getattr(self, "housekeeping", None):
            self.housekeeping.close()

        super().tearDown()

    # since this logic id depended on code run by SCT to mark uuid as test, since commit 617026aa, this code it run in the background
    # and not being waited for, so we need to compensate for it here with retries
    @retrying(n=5, sleep_time=10, allowed_exceptions=(AssertionError,))
    def check_scylla_version_in_housekeepingdb(self, prev_id: int, expected_status_code: str,
                                               new_row_expected: bool, backend: str) -> int:
        """
        Validate reported version
        prev_id: check if new version is created
        """
        assert self.node.uuid, "Node UUID wasn't created"

        row = self.housekeeping.get_most_recent_record(query=f"SELECT id, version, ip, statuscode "
                                                             f"FROM {self.CHECK_VERSION_TABLE} "
                                                             f"WHERE uuid = %s", args=(self.node.uuid,))
        self.log.debug("Last row in %s for uuid '%s': %s", self.CHECK_VERSION_TABLE, self.node.uuid, row)

        public_ip_address = self.node.host_public_ip_address if backend == 'docker' else self.node.public_ip_address
        self.log.debug("public_ip_address = %s", public_ip_address)

        # Validate public IP address
        assert public_ip_address == row[2], (
            f"Wrong IP address is saved in '{self.CHECK_VERSION_TABLE}' table: "
            f"expected {self.node.public_ip_address}, got: {row[2]}")

        # Validate reported node version
        assert row[1] == self.node.scylla_version, (
            f"Wrong version is saved in '{self.CHECK_VERSION_TABLE}' table: "
            f"expected {self.node.public_ip_address}, got: {row[2]}")

        # Validate expected status code
        assert row[3] == expected_status_code, (
            f"Wrong statuscode is saved in '{self.CHECK_VERSION_TABLE}' table: "
            f"expected {expected_status_code}, got: {row[3]}")

        if prev_id:
            # Validate row id
            if new_row_expected:
                assert row[0] > prev_id, f"New row wasn't saved in {self.CHECK_VERSION_TABLE}"
            else:
                assert row[0] == prev_id, f"New row was saved in {self.CHECK_VERSION_TABLE} unexpectedly"

        return row[0] if row else 0

    @property
    def node(self):
        if self.db_cluster is None or not self.db_cluster.nodes:
            raise ValueError('DB cluster has not been initiated')
        return self.db_cluster.nodes[0]

    @cached_property
    def write_back_cache(self) -> int | None:
        res = self.node.remoter.run("cat /etc/scylla.d/perftune.yaml", ignore_status=True)
        if res.ok:
            return yaml.safe_load(res.stdout).get("write_back_cache")
        return None

    def check_cluster_name(self):
        with self.node.remote_scylla_yaml() as scylla_yaml:
            yaml_cluster_name = scylla_yaml.cluster_name

        self.assertTrue(self.db_cluster.name == yaml_cluster_name,
                        f"Cluster name is not as expected. Cluster name in scylla.yaml: {yaml_cluster_name}. "
                        f"Cluster name: {self.db_cluster.name}")

    def run_cassandra_stress(self, args: str):
        stress_cmd = f"{self.node.add_install_prefix(STRESS_CMD)} {args} -node {self.node.ip_address}"
        if self.params.get('client_encrypt'):
            stress_cmd += " -transport 'truststore=/etc/scylla/ssl_conf/client/cacerts.jks truststore-password=cassandra'"
        result = self.node.remoter.run(stress_cmd)
        assert "java.io.IOException" not in result.stdout
        assert "java.io.IOException" not in result.stderr

    def check_scylla(self):
        self.node.run_nodetool("status")
        self.run_cassandra_stress("write n=10000 -mode cql3 native -pop seq=1..10000")
        self.run_cassandra_stress("mixed duration=1m -mode cql3 native -rate threads=10 -pop seq=1..10000")

    def check_cqlsh(self):
        output = self.node.run_cqlsh("desc keyspaces")
        self.log.debug(output.stdout)

        output = self.node.run_cqlsh('select JSON host_id,broadcast_address from system.local ;', split=True)

        for line in output:
            try:
                host = json.loads(line)
            except json.decoder.JSONDecodeError:
                continue
            self.log.debug(host)
            assert 'broadcast_address' in host and 'host_id' in host, (
                f"system.local: {host}, doesn't have 'broadcast_address' or 'host_id'")

    def check_housekeeping_service_status(self, backend: str):
        housekeeping_service_name = "scylla-housekeeping" if backend == "docker" else "scylla-housekeeping-restart"
        status_out = self.node.get_service_status(housekeeping_service_name, ignore_status=True)
        # When the test runs with OEL81 operation system error "TMOUT: readonly variable" is printed from /etc/bashrc.
        # Ignore it
        if status_out.stderr and "TMOUT" not in status_out.stderr:
            ScyllaHousekeepingServiceEvent(message=f"scylla-housekeeping-restart service error: {status_out.stderr}",
                                           severity=Severity.ERROR).publish()
            return

        status = status_out.stdout.strip()
        # Expected output:
        #     "scylla-housekeeping-restart.service: Succeeded" - ubuntu, centos
        #     "RUNNING" - docker
        #     "Started Scylla Housekeeping restart mode" - other
        #     "Started scylla-housekeeping-restart.service - Scylla Housekeeping restart mode" - ubuntu24
        if "scylla-housekeeping-restart.service: Succeeded" in status or \
                "Started Scylla Housekeeping restart mode" in status or \
                "Started scylla-housekeeping-restart.service - Scylla Housekeeping restart mode" in status or \
                "RUNNING" in status:
            ScyllaHousekeepingServiceEvent(message="scylla-housekeeping-restart service running",
                                           severity=Severity.NORMAL).publish()
        else:
            ScyllaHousekeepingServiceEvent(message=f"scylla-housekeeping-restart service is not running: {status}",
                                           severity=Severity.ERROR).publish()

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

    def verify_node_health(self):
        self.node.check_node_health()

    def verify_snitch(self, backend_name: str):
        """
        Verify that the snitch used in the cluster is appropriate for
        the backend used.
        """
        if not self.params.get("use_preinstalled_scylla"):
            self.log.info("Skipping verifying the snitch due to the 'use_preinstalled_scylla' being set to False")
            return

        describecluster_snitch = self.get_describecluster_info().snitch
        with self.node.remote_scylla_yaml() as scylla_yaml:
            scylla_yaml_snitch = scylla_yaml.endpoint_snitch
        expected_snitches = BACKENDS[backend_name]

        snitch_patterns = [re.compile(f"({snitch})") for snitch in expected_snitches]
        snitch_matches_describecluster = [pattern.search(describecluster_snitch) for pattern in snitch_patterns]
        snitch_matches_scylla_yaml = [pattern.search(scylla_yaml_snitch) for pattern in snitch_patterns]

        with self.subTest('verify snitch against describecluster output'):
            self.assertTrue(any(snitch_matches_describecluster),
                            msg=f"Expected snitch matches for describecluster to not be empty, but was. Snitch "
                                f"matches: {snitch_matches_describecluster}"
                            )

        with self.subTest('verify snitch against scylla.yaml configuration'):
            self.assertTrue(any(snitch_matches_scylla_yaml),
                            msg=f"Expected snitch matches for scylla yaml to not be empty, but was. Snitch "
                                f"matches: {snitch_matches_scylla_yaml}"
                            )

    def verify_docker_latest_match_release(self) -> None:
        for product in typing.get_args(ScyllaProduct):
            latest_version = get_latest_scylla_release(product=product)

            url = 'https://hub.docker.com/v2/repositories/scylladb/{}/tags/{}'
            docker_latest = requests.get(url.format(product, 'latest')).json()
            docker_release = requests.get(url.format(product, latest_version)).json()
            self.log.debug('latest info: %s', pprint.pformat(docker_latest))
            self.log.debug('%s info: %s ', latest_version, pprint.pformat(docker_release))

            latest_digests = set(image['digest'] for image in docker_latest['images'])
            release_digests = set(image['digest'] for image in docker_release['images'])

            assert latest_digests == release_digests, \
                f"latest != {latest_version}, images digest differs [{latest_digests}] != [{release_digests}]"

    def verify_nvme_write_cache(self) -> None:
        if self.write_back_cache is None or self.node.parent_cluster.is_additional_data_volume_used():
            return
        expected_write_cache_value = "write back" if self.write_back_cache else "write through"

        run = self.node.remoter.sudo
        nvme_devices = re.findall(r"nvme[\d]+n[\d\w]+", run("ls -1 /dev/nvme*").stdout)
        self.assertGreater(len(nvme_devices), 0)

        for nvme_device in nvme_devices:
            self.log.info("Expected write cache for %s is %r", nvme_device, expected_write_cache_value)
            write_cache = run(f"cat /sys/block/{nvme_device}/queue/write_cache").stdout.strip()
            self.assertEqual(write_cache, expected_write_cache_value)

    def verify_xfs_online_discard_enabled(self) -> None:
        run = self.node.remoter.sudo

        self.log.info("Verify XFS mount options for /var/lib/scylla contain `discard'")
        mount_options = run("findmnt -no options -t xfs -T /var/lib/scylla").stdout.strip().split(",")
        self.assertIn("discard", mount_options)

        # we only test this in images created by scylla-machine-image
        if self.params.get("use_preinstalled_scylla"):
            self.log.info("Ensure that we don't run fstrim")
            self.assertEqual(run("systemctl is-enabled fstrim.timer", ignore_status=True).stdout.strip(), "disabled")

    def check_service_existence(self, service_name):
        res = self.node.remoter.run(f'systemctl list-units --full | grep -Fq "{service_name}"',
                                    ignore_status=True)
        if res.exit_status:
            return False
        return True  # exit_status = 1 means the service doesn't exist

    def run_pre_create_schema(self, replication_factor=1):
        pre_create_schema = self.params.get('pre_create_schema')
        if pre_create_schema:
            keyspace_num = self.params.get('keyspace_num')
            sstable_size = self.params.get('sstable_size')
            compaction_strategy = self.params.get('compaction_strategy')
            scylla_encryption_options = self.params.get('scylla_encryption_options')
            self.log.debug('Pre Creating Schema for c-s with %s keyspaces', keyspace_num)
            for i in range(1, keyspace_num+1):
                keyspace_name = 'keyspace{}'.format(i)
                self.create_keyspace(keyspace_name=keyspace_name, replication_factor=replication_factor)
                self.log.debug('%s Created', keyspace_name)
                col_num = 5
                columns = {}
                for col_idx in range(col_num):
                    cs_key = '"C' + str(col_idx) + '"'
                    columns[cs_key] = 'blob'
                self.create_table(name='standard1', keyspace_name=keyspace_name, key_type='blob', read_repair=0.0,
                                  columns=columns,
                                  scylla_encryption_options=scylla_encryption_options,
                                  compaction=compaction_strategy, sstable_size=sstable_size)

    # pylint: disable=too-many-statements,too-many-branches
    def test_scylla_service(self):

        self.run_pre_create_schema()

        backend = self.params.get("cluster_backend")

        if backend == "aws":
            with self.subTest("check ENA support"):
                assert self.node.ena_support, "ENA support is not enabled"

        with self.subTest("verify write cache for NVMe devices"):
            self.verify_nvme_write_cache()

        if (backend != "docker" and not self.params.get("nonroot_offline_install")
                and self.node.db_node_instance_type != "t3.micro"):
            with self.subTest("verify XFS online discard enabled"):
                self.verify_xfs_online_discard_enabled()

        if backend == "gce":
            with self.subTest("verify users"):
                self.verify_users()

        expected_housekeeping_status_code = 'cr' if backend == "docker" else 'r'

        if self.params.get("use_preinstalled_scylla") and backend != "docker":
            with self.subTest("check the cluster name"):
                self.check_cluster_name()

        with self.subTest('verify snitch'):
            self.verify_snitch(backend_name=backend)

        with self.subTest('verify node health'):
            self.verify_node_health()

        with self.subTest("check Scylla server after installation"):
            self.check_scylla()

        with self.subTest("check cqlsh installation"):
            self.check_cqlsh()

        with self.subTest("check node_exporter liveness"):
            node_info_service = NodeLoadInfoServices().get(self.node)
            assert node_info_service.cpu_load_5
            assert node_info_service.get_node_boot_time_seconds()

        # We don't install any time sync service in docker, so the test is unnecessary:
        # https://github.com/scylladb/scylla/tree/master/dist/docker/etc/supervisord.conf.d
        if backend != "docker":
            with self.subTest("check if scylla unnecessarily installed a time synchronization service"):
                # Checks https://github.com/scylladb/scylla/issues/8339
                # If the instance already has systemd-timesyncd
                is_timesyncd_service_installed = self.check_service_existence(service_name="systemd-timesyncd")
                # Do note: On Redhat based distributions the services are named ntpd and chronyd, while on debian based
                # distributions they're named ntp and chrony.
                if self.node.distro.is_rhel_like:
                    is_ntp_service_installed = self.check_service_existence(service_name="ntpd")
                    is_chrony_service_installed = self.check_service_existence(service_name="chronyd")
                else:
                    is_ntp_service_installed = self.check_service_existence(service_name="ntp")
                    is_chrony_service_installed = self.check_service_existence(service_name="chrony")
                try:
                    if is_timesyncd_service_installed:
                        assert not is_ntp_service_installed, \
                            "systemd-timesyncd is already installed, yet scylla installed ntp service on top of it"
                        assert not is_chrony_service_installed, \
                            "systemd-timesyncd is already installed, yet scylla installed chrony service on top of it"
                    elif not self.params.get("unified_package"):
                        assert is_ntp_service_installed or is_chrony_service_installed, \
                            "systemd-timesyncd is not installed on the instance, yet Scylla did not install ntp or " \
                            "chrony services as a replacement"
                    else:
                        # https://github.com/scylladb/scylla/issues/10608#issuecomment-1135770570
                        # Scylla doesn't install any time sync service when it's an offline installation,
                        # so if there's no time sync services active after the scylla installation, it's fine.
                        # However, in such scenario a warning message should be printed during the scylla installation
                        self.log.warning("No time sync service was installed. "
                                         "Passable since it's an offline installation.")
                except AssertionError:
                    full_list = self.node.remoter.run('systemctl list-units --full').stdout.strip()
                    self.log.warning("Seems that there was an issue with ntp check. Here's the full list of services "
                                     "active on the node: %s", full_list)
                    raise

            # TODO: implement after the new provision will be added
            # Task: https://trello.com/c/BIdIUwyT/4096-housekeeping-implemented-a-test-that-checks-i-value-when-scylla-
            # is-first-installed

            # Scylla service is stopping/starting after installation and re-configuration.
            # To validate version after installation, we need to perform validation before re-config.
            # For that the test should be changed to be able to call "add_nodes" function from BaseCluster.
            # if not self.node.is_nonroot_install:
            #     self.log.info("Validate version after install")
            #     self.check_housekeeping_service_status()
            #     self.check_scylla_version_in_housekeepingdb(prev_id=0,
            #                                                 expected_status_code='i',
            #                                                 new_row_expected=False,
            #                                                 backend=backend)

        version_id_after_stop = 0
        with self.subTest("check Scylla server after stop/start"):
            self.node.stop_scylla(verify_down=True)
            self.node.start_scylla(verify_up=True)

            # Scylla service has been stopped/started after installation and re-configuration.
            # So we don't need to stop and to start it again
            self.check_scylla()

            if not self.node.is_nonroot_install:
                self.log.info("Validate version after stop/start")
                self.check_housekeeping_service_status(backend=backend)
                version_id_after_stop = self.check_scylla_version_in_housekeepingdb(
                    prev_id=0,
                    expected_status_code=expected_housekeeping_status_code,
                    new_row_expected=False,
                    backend=backend)

        with self.subTest("check Scylla server after restart"):
            self.node.restart_scylla(verify_up_after=True)
            self.check_scylla()

            if not self.node.is_nonroot_install:
                self.log.info("Validate version after restart")
                self.check_housekeeping_service_status(backend=backend)
                self.check_scylla_version_in_housekeepingdb(prev_id=version_id_after_stop,
                                                            expected_status_code=expected_housekeeping_status_code,
                                                            new_row_expected=True,
                                                            backend=backend)

        if backend != 'docker':
            with self.subTest("Check the output of perftune.py"):
                perftune_checker = PerftuneOutputChecker(self.node)
                perftune_checker.compare_perftune_results()

        if backend == 'docker':
            with self.subTest("Check docker latest tags"):
                self.verify_docker_latest_match_release()

    def get_email_data(self):
        self.log.info("Prepare data for email")
        email_data = self._get_common_email_data()
        try:
            node = self.node
        except Exception:  # pylint: disable=broad-except
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
