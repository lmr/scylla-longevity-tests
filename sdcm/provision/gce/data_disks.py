# pylint: disable=too-few-public-methods
from abc import ABC
from typing import NamedTuple

from sdcm.provision.provisioner import DataDisk


class GCEDiskTypes(NamedTuple):
    SCRATCH: str = "SCRATCH"
    PERSISTENT: str = "PERSISTENT"


GCE_DISK_TYPES = GCEDiskTypes()


class GCEDataDisk(DataDisk, ABC):
    type: str
    size: int
    gce_struct_type: str
    auto_delete: bool


class ScratchDisk(GCEDataDisk):
    type: str = GCE_DISK_TYPES.SCRATCH
    size: int = 375
    gce_struct_type: str = "local-ssd"
    auto_delete: bool = True
    interface: str = "NVME"


class PersistentStandardDisk(GCEDataDisk):
    type: str = GCE_DISK_TYPES.PERSISTENT
    size: int = 50
    gce_struct_type: str = "pd-ssd"
    auto_delete: bool = True
