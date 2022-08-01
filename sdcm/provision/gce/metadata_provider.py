import json
from dataclasses import dataclass


# pylint:disable=too-many-instance-attributes
@dataclass
class MetadataArgs:
    name: str
    username: str
    public_key: str
    tags: dict
    node_index: int
    startup_script: str
    cluster_name: str
    raid_level: int


# pylint:disable=too-few-public-methods
class MetadataProvider:
    @staticmethod
    def get_ex_metadata(metadata_args: MetadataArgs, params: dict):
        return {
            **metadata_args.tags,
            "Name": metadata_args.name,
            "NodeIndex": metadata_args.node_index,
            "startup-script": metadata_args.startup_script,
            "block-project-ssh-keys": params.get("gce-block-project-ssh-keys", "true"),
            "ssh-keys": f"{metadata_args.username}:ssh-rsa {metadata_args.public_key}",
            "user-data": json.dumps({
                "scylla_yaml": {"cluster_name": metadata_args.cluster_name},
                "start_scylla_on_first_boot": params.get("start_scylla_on_first_boot", False),
                "raid_level": metadata_args.raid_level
            })
        }
