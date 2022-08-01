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
from __future__ import annotations
import datetime
import logging
from typing import Dict, List

from libcloud.compute.base import Node, NodeLocation

from sdcm.keystore import KeyStore
from sdcm.provision.gce import GCE_REGIONS
from sdcm.provision.gce.disk_struct_provider import DiskStructProvider, DiskStructArgs
from sdcm.provision.gce.metadata_provider import MetadataProvider, MetadataArgs
from sdcm.provision.gce.virtual_machine_provider import VirtualMachineProvider
from sdcm.provision.provisioner import Provisioner, InstanceDefinition, VmInstance, PricingModel
from sdcm.utils.gce_utils import get_gce_service

LOGGER = logging.getLogger(__name__)


class GCEProvisioner(Provisioner):
    """Provides API for VM provisioning in GCE"""

    def __init__(self, test_id, region, params: dict):
        super().__init__(test_id, region)
        self._params = params
        self._gce_service = get_gce_service(region=self.region)
        self._zone = region
        self._service_accounts = KeyStore().get_gcp_service_accounts()
        self._ex_disk_struct_provider = DiskStructProvider
        self._ex_metadata_provider = MetadataProvider
        self._vm_provider = VirtualMachineProvider(region=region,
                                                   disk_struct_provider=DiskStructProvider,
                                                   metadata_provider=MetadataProvider)

    @classmethod
    def discover_regions(cls, test_id) -> List[Provisioner]:
        gce_service = get_gce_service(region=GCE_REGIONS.us_east1)
        gce_locations: list[NodeLocation] = gce_service.list_locations()
        return [cls(test_id=test_id, region=location.name, params={}) for location in gce_locations]

    @property
    def cluster_params(self):
        return self._params

    @cluster_params.setter
    def cluster_params(self, params: dict):
        self._params = params

    def get_or_create_instance(self,
                               definition: InstanceDefinition,
                               pricing_model: PricingModel = PricingModel.SPOT) -> VmInstance:
        """
        Create VmInstance in provided region, specified by InstanceDefinition
        """
        gce_project_name = self._gce_service.ex_get_project().name

        disk_struct_args = DiskStructArgs(
            instance_definition=definition,
            root_disk_type=self.cluster_params.get("gce_root_disk_type_db"),
            gce_services_project_name=gce_project_name,
            region_az=self.region,
            data_disks=definition.data_disks
        )

        metadata_args = MetadataArgs(
            name=definition.name,
            username=self.cluster_params.get("gce_image_username"),
            public_key=definition.ssh_key,
            tags=definition.tags,  # self.tags
            node_index=definition.instance_index,  # node_index
            startup_script=definition.startup_script,
            cluster_name=definition.name,  # self.name
            raid_level=self.cluster_params.get("raid_level")
        )

        new_node: Node = self._vm_provider.get_or_create_instance(disk_struct_args=disk_struct_args,
                                                                  metadata_args=metadata_args,
                                                                  instance_definition=definition,
                                                                  pricing_model=pricing_model,
                                                                  params=self.cluster_params)

        vm_instance = VmInstance(name=new_node.name,
                                 region=self.region,
                                 user_name=metadata_args.username,
                                 public_ip_address=new_node.public_ips[0],
                                 private_ip_address=new_node.private_ips[0],
                                 ssh_key_name=KeyStore().get_gce_ssh_key_pair().name,
                                 tags=definition.tags,
                                 pricing_model=pricing_model.value,
                                 image=definition.image_id,
                                 _provisioner=self,
                                 _node=new_node,
                                 creation_time=datetime.datetime.utcnow())
        return vm_instance

    def get_or_create_instances(self, definitions: List[InstanceDefinition],
                                pricing_model: PricingModel = PricingModel.SPOT) -> List[VmInstance]:
        vm_instances = []

        for definition in definitions:
            vm_instances.append(self.get_or_create_instance(definition=definition, pricing_model=pricing_model))
        return vm_instances

    def terminate_instance(self, name: str, wait: bool = False) -> bool:
        return self._vm_provider.destroy_instance(name)

    def reboot_instance(self, name: str, wait: bool) -> None:
        self._vm_provider.reboot_instance(name)

    def list_instances(self) -> List[VmInstance]:
        return self._vm_provider.list_running_instances()

    def cleanup(self, wait: bool = False) -> None:
        pass

    def add_instance_tags(self, name: str, tags: Dict[str, str]) -> None:
        self._vm_provider.add_tags(vm_instance_name=name, tags=tags)

    def _get_available_zones(self) -> list[str]:
        region = self._gce_service.region
        zones = [item.name for item in self._gce_service.list_locations() if item.name.startswith(region.name)]
        return zones
