import logging
import random
from typing import Type

from libcloud.compute.base import Node
from libcloud.compute.types import NodeState

from sdcm.keystore import KeyStore
from sdcm.provision.gce.disk_struct_provider import DiskStructProvider, DiskStructArgs
from sdcm.provision.gce.metadata_provider import MetadataProvider, MetadataArgs
from sdcm.provision.provisioner import InstanceDefinition, PricingModel
from sdcm.utils.gce_utils import get_gce_service


LOGGER = logging.getLogger(__name__)


class VMInstanceAlreadyExists(Exception):
    ...


class VMInstanceDestructionUnsuccessful(Exception):
    ...


class VMInstanceRebootUnsuccessful(Exception):
    ...


class VMProvisionError(Exception):
    ...


class VirtualMachineProvider:
    def __init__(self, region: str,
                 disk_struct_provider: Type[DiskStructProvider],
                 metadata_provider: Type[MetadataProvider]):
        self._region = region
        self._gce_service = get_gce_service(region=region)
        self._location = self._get_random_location_in_region()
        self._service_accounts = KeyStore().get_gcp_service_accounts()
        self._disk_struct_provider = disk_struct_provider
        self._metadata_provider = metadata_provider
        self._vm_node_cache: list[Node] = []

    @staticmethod
    def _sanitize_tags(tags: dict) -> dict:
        return {k.lower(): v.lower().replace(".", "-") for k, v in tags.items()}

    # pylint: disable=too-many-arguments
    def get_or_create_instance(self,
                               disk_struct_args: DiskStructArgs,
                               metadata_args: MetadataArgs,
                               instance_definition: InstanceDefinition,
                               pricing_model: PricingModel,
                               params: dict) -> Node:
        # check if the instance with a given name is already in the cache
        found_instance = self._get_node_from_cache(node_name=instance_definition.name)

        if found_instance:
            LOGGER.info("Requested the creation of instance with name %s but a "
                        "instance with that name already exists. Returning the "
                        "already existing instance.", instance_definition.name)
            return found_instance

        instance_params = {
            "name": instance_definition.name,
            "size": instance_definition.type,
            "image": instance_definition.image_id,
            "ex_network": params.get("gce_network"),
            "ex_disks_gce_struct": self._disk_struct_provider.get_disks_struct(disk_struct_args, params),
            "ex_metadata": self._metadata_provider.get_ex_metadata(metadata_args, params),
            "ex_service_accounts": self._service_accounts,
            "ex_preemptible": pricing_model == pricing_model.SPOT
        }

        LOGGER.info("Going to create the instance using the following instance params:\n%s", instance_params)

        try:
            new_node: Node = self._gce_service.create_node(**instance_params)
            self._vm_node_cache.append(new_node)
            return new_node
        except Exception as exc:
            raise VMProvisionError("Could not provision VM. Root error:\n%s") from exc

    def _get_node_from_cache(self, node_name: str) -> Node | None:
        found_node = [node for node in self._vm_node_cache if node.name == node_name]

        if found_node:
            return found_node[0]
        return None

    def _get_random_location_in_region(self) -> str:
        return random.choice(self._get_available_locations())

    def _get_available_locations(self) -> list[str]:
        locations = [item.name for item in self._gce_service.list_locations() if item.name.startswith(self._region)]
        return locations

    def add_tags(self, vm_instance_name: str, tags: dict):
        instance_to_tag = [node for node in self.list_running_instances() if node.name == vm_instance_name]
        if not instance_to_tag:
            raise VMProvisionError(f"Can't find instance to add tags too. Instance name requested: {vm_instance_name}")

        LOGGER.info("Sanitized tags: %s", self._sanitize_tags(tags))
        self._gce_service.ex_set_node_labels(instance_to_tag[0], self._sanitize_tags(tags))
        LOGGER.info("Tags as node labels added to node: %s", instance_to_tag[0].name)

    def list_instances(self) -> list[Node]:
        return self._gce_service.list_nodes()

    def list_running_instances(self) -> list[Node]:
        running_instances = [instance for instance in self.list_instances() if instance.state in NodeState.RUNNING]
        return running_instances

    def destroy_instance(self, instance_name: str) -> bool:
        instance_to_destroy = self._get_node_from_cache(instance_name)

        if instance_to_destroy:
            destroy_response = instance_to_destroy.destroy()

            if not destroy_response:
                raise VMInstanceDestructionUnsuccessful(f"Not able to destroy instance {instance_to_destroy.name}.")

            self._vm_node_cache.pop(self._vm_node_cache.index(instance_to_destroy))
            LOGGER.info("VM instance: %s successfully destroyed.", instance_to_destroy.name)
        else:
            LOGGER.warning("Instance %s does not exist. Shouldn't have called it.", instance_to_destroy)

    def reboot_instance(self, instance_name: str):
        instance_to_reboot = self._get_node_from_cache(instance_name)

        if instance_to_reboot:
            reboot_reponse = instance_to_reboot.reboot()

            if not reboot_reponse:
                raise VMInstanceRebootUnsuccessful(f"Not able to reboot VM instance {instance_to_reboot}")

            LOGGER.info("VM instance %s successfully rebooted.", instance_to_reboot)
        else:
            LOGGER.warning("Did not find the instance: %s. "
                           "Not proceeding with the reboot", instance_to_reboot.name)
