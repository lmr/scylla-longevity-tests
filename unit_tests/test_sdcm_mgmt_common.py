from sdcm.mgmt.common import get_manager_scylla_backend, get_manager_repo_from_defaults
from sdcm.utils.distro import Distro


class TestManagerVersions:

    def test_get_manager_scylla_backend_returns_repo_address(self):  # pylint: disable=no-self-use
        url = get_manager_scylla_backend("2024", Distro.UBUNTU22)

        assert url == 'https://downloads.scylladb.com/deb/debian/scylla-2024.1.list'

    def test_get_manager_repo_from_defaults_returns_repo_address(self):  # pylint: disable=no-self-use
        url = get_manager_repo_from_defaults("3.2", Distro.UBUNTU22)

        assert url == 'https://downloads.scylladb.com/deb/debian/scylladb-manager-3.2.list'
