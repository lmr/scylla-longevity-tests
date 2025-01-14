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
# Copyright (c) 2021 ScyllaDB


import unittest

from sdcm.utils.version_utils import ComparableScyllaVersion

from utils.get_supported_scylla_base_versions import UpgradeBaseVersion  # pylint: disable=no-name-in-module, import-error


def general_test(scylla_repo='', linux_distro='', cloud_provider=None):
    scylla_version = None

    version_detector = UpgradeBaseVersion(scylla_repo, linux_distro, scylla_version)
    version_detector.set_start_support_version(cloud_provider)
    _, version_list = version_detector.get_version_list()
    return version_list


class TestBaseVersion(unittest.TestCase):
    download_url_base = 'https://downloads.scylladb.com'
    url_base = f'{download_url_base}/unstable/scylla'

    def test_master(self):
        scylla_repo = self.url_base + '/master/rpm/centos/2021-08-29T00:58:58Z/scylla.repo'
        linux_distro = 'centos'
        version_list = general_test(scylla_repo, linux_distro)
        assert len(version_list) == 1
        assert ComparableScyllaVersion(version_list[0]) > '5.4'

    def test_ubuntu_22_azure_5_3(self):
        scylla_repo = self.url_base + '/branch-5.3/deb/unified/2023-05-23T22:36:08Z/scylladb-5.3/scylla.list'
        linux_distro = 'ubuntu-jammy'
        cloud_provider = 'azure'
        version_list = general_test(scylla_repo, linux_distro, cloud_provider)
        self.assertEqual(version_list, ['5.2'])

    def test_ubuntu_22_azure_2023_1(self):
        scylla_repo = self.url_base +\
            '-enterprise/enterprise-2023.1/deb/unified/2023-05-17T23:03:35Z/scylladb-2023.1/scylla.list'
        linux_distro = 'ubuntu-jammy'
        cloud_provider = 'azure'
        version_list = general_test(scylla_repo, linux_distro, cloud_provider)
        assert set(version_list) == {'5.2', '2023.1'}

    def test_4_5_with_centos8(self):
        scylla_repo = self.url_base + '/branch-4.5/rpm/centos/2021-08-29T00:58:58Z/scylla.repo'
        linux_distro = 'centos-8'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'4.4', '4.5'}

    def test_4_5(self):
        scylla_repo = self.url_base + '/4.5/rpm/centos/2021-08-29T00:58:58Z/scylla.repo'
        linux_distro = 'centos'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'4.4', '4.5'}

    def test_5_0(self):
        scylla_repo = self.url_base + '/branch-5.0/rpm/centos/2023-01-21T02:51:18Z/scylla.repo'
        linux_distro = 'centos-8'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'4.6', '5.0'}

    def test_5_1(self):
        scylla_repo = self.url_base + '/branch-5.1/deb/unified/2023-01-21T02:35:18Z/scylladb-5.1/scylla.list'
        linux_distro = 'ubuntu-focal'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'5.0', '5.1'}

    def test_2021_1(self):
        scylla_repo = self.url_base + '-enterprise/branch-2021.1/rpm/centos/2021-08-29T00:58:58Z/scylla.repo'
        linux_distro = 'centos'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'4.3', '2020.1', '2021.1'}

    def test_2021_1_with_centos8(self):
        scylla_repo = self.url_base + '-enterprise/branch-2021.1/rpm/centos/2021-08-29T00:58:58Z/scylla.repo'
        linux_distro = 'centos-8'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'4.3', '2021.1'}

    def test_2022_1_with_centos8(self):
        scylla_repo = self.url_base + \
            '-enterprise/enterprise-2022.1/deb/unified/2022-06-03T00:22:55Z/scylladb-2022.1/scylla.list'
        linux_distro = 'centos-8'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'5.0', '2021.1', '2022.1'}

    def test_2024_1_with_centos9(self):
        scylla_repo = self.url_base + '-enterprise/enterprise-2024.1/rpm/centos/latest/scylla.repo'
        linux_distro = 'centos-9'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'5.4', '2024.1'}

    def test_2022_2(self):
        scylla_repo = self.url_base + '-enterprise/enterprise-2022.2/rpm/centos/2022-10-09T10:39:11Z/scylla.repo'
        linux_distro = 'centos'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'5.1', '2022.1', '2022.2'}

    def test_2024_1(self):
        scylla_repo = self.url_base + '-enterprise/enterprise-2024.1/rpm/centos/latest/scylla.repo'
        linux_distro = 'centos'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'5.4', '2022.2', '2023.1', '2024.1'}

    def test_2024_2(self):
        scylla_repo = self.url_base + '-enterprise/enterprise-2024.2/rpm/centos/latest/scylla.repo'
        linux_distro = 'centos-9'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'2024.1', '2024.2', '6.0'}

    def test_2024_2_ubuntu(self):
        scylla_repo = self.url_base + '-enterprise/enterprise-2024.2/deb/unified/latest/scylladb-2024.2/scylla.list'
        linux_distro = 'ubuntu-focal'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'6.0', '2024.1', '2024.2'}

    def test_2025_1_dev(self):
        scylla_repo = self.url_base + '/master/rpm/centos/2025-01-15T09:25:01Z/scylla.repo'
        linux_distro = 'centos'
        version_list = general_test(scylla_repo, linux_distro)
        assert set(version_list) == {'6.2'}

    def test_2025_1_ubuntu(self):
        scylla_repo = self.url_base + '-enterprise/enterprise-2025.1/deb/unified/latest/scylladb-2025.1/scylla.list'
        linux_distro = 'ubuntu-focal'
        version_list = general_test(scylla_repo, linux_distro)
        assert {'6.2', '2024.1', '2024.2'}.issubset(set(version_list))


if __name__ == "__main__":
    unittest.main()
