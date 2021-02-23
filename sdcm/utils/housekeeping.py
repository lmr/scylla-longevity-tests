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

import logging
from typing import Sequence, Optional, Any

import mysql.connector

from sdcm.keystore import KeyStore


DB_NAME = "housekeeping"

LOGGER = logging.getLogger(__name__)


Row = Sequence[Any]


class HousekeepingDB:
    def __init__(self, host: str, username: str, password: str):
        self.host = host
        self.username = username
        self.password = password
        self._connection = None
        self._cursor = None

    @classmethod
    def from_params(cls, params: dict) -> "HousekeepingDB":
        return cls(host=params["housekeeping_db_host"],
                   username=params["housekeeping_db_user"],
                   password=params["housekeeping_db_password"])

    @classmethod
    def from_keystore_creds(cls) -> "HousekeepingDB":
        return cls.from_params(KeyStore().get_housekeeping_db_credentials())

    def connect(self, db_name: str = DB_NAME) -> None:
        self._connection = mysql.connector.connect(host=self.host,
                                                   user=self.username,
                                                   password=self.password,
                                                   database=db_name)
        self._cursor = self._connection.cursor()

    def execute(self, query: str, args: Optional[Sequence[Any]] = None) -> Sequence[Row]:
        LOGGER.debug("Query: `%s', Args: %s", query, args)
        self._cursor.execute(query, args)
        result = self._cursor.fetchall()
        self._connection.commit()
        LOGGER.info("Result: %s", result)
        return result

    def close(self) -> None:
        self._cursor.close()
        self._connection.close()
        self._cursor = None
        self._connection = None

    def get_most_recent_record(self, query: str, args: Optional[Sequence[Any]] = None) -> Optional[Row]:
        result = self.execute(query + " ORDER BY -dt LIMIT 1", args)
        return result[0] if result else None

    def get_new_records(self, query: str, args: Optional[Sequence[Any]] = None, last_id: int = 0) -> Sequence[Row]:
        args = (tuple(args) if args else ()) + (last_id, )
        return self.execute(query + " id > ", args)
