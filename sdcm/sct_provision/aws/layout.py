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

from functools import cached_property

from sdcm.provision.aws.capacity_reservation import SCTCapacityReservation
from sdcm.provision.aws.dedicated_host import SCTDedicatedHosts
from sdcm.sct_provision.aws.cluster import (
    OracleDBCluster, DBCluster, LoaderCluster, MonitoringCluster, PlacementGroup)
from sdcm.sct_provision.common.layout import SCTProvisionLayout
from sdcm.test_config import TestConfig


class SCTProvisionAWSLayout(SCTProvisionLayout, cluster_backend='aws'):

    @cached_property
    def _test_config(self):
        return TestConfig()

    def provision(self):
        if self.placement_group:
            self.placement_group.provision()
        SCTCapacityReservation.reserve(self._params)
        SCTDedicatedHosts.reserve(self._params)
        if self.db_cluster:
            self.db_cluster.provision()
        if self.monitoring_cluster:
            self.monitoring_cluster.provision()
        if self.loader_cluster:
            self.loader_cluster.provision()
        if self.cs_db_cluster:
            self.cs_db_cluster.provision()

    @cached_property
    def db_cluster(self):
        return DBCluster(
            params=self._params,
            common_tags=self._test_config.common_tags(),
            test_id=self._test_config.test_id(),
        )

    @cached_property
    def loader_cluster(self):
        return LoaderCluster(
            params=self._params,
            common_tags=self._test_config.common_tags(),
            test_id=self._test_config.test_id(),
        )

    @cached_property
    def monitoring_cluster(self):
        return MonitoringCluster(
            params=self._params,
            common_tags=self._test_config.common_tags(),
            test_id=self._test_config.test_id(),
        )

    @cached_property
    def cs_db_cluster(self):
        if not self._provision_another_scylla_cluster:
            return None
        return OracleDBCluster(
            params=self._params,
            common_tags=self._test_config.common_tags(),
            test_id=self._test_config.test_id(),
        )

    @cached_property
    def placement_group(self):
        return PlacementGroup(
            params=self._params,
            common_tags=self._test_config.common_tags(),
            test_id=self._test_config.test_id(),
        )
