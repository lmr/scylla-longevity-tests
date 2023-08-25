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
# Copyright (c) 2024 ScyllaDB
from typing import TypeVar
from sdcm.rest.remote_curl_client import RemoteCurlClient

BaseNode = TypeVar("BaseNode")


class RaftApi(RemoteCurlClient):
    """Raft api commands"""

    def __init__(self, node: BaseNode):
        super().__init__(host="localhost:10000", endpoint="raft", node=node)

    def read_barrier(self, group_id: str) -> str:
        path = f"read_barrier?group_id={group_id}"
        return self.run_remoter_curl(method="POST",
                                     path=path,
                                     params={}, timeout=30).stdout.strip()
