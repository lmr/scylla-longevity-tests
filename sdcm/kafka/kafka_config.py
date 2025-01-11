from typing import Optional

from pydantic import Field, BaseModel, ConfigDict  # pylint: disable=no-name-in-module

# pylint: disable=too-few-public-methods


class ConnectorConfiguration(BaseModel):
    # general options
    connector_class: str = Field(alias="connector.class")
    topics: Optional[str] = ''

    # scylla-cdc-source-connector options
    # see https://github.com/scylladb/scylla-cdc-source-connector?tab=readme-ov-file#configuration
    # and https://github.com/scylladb/scylla-cdc-source-connector?tab=readme-ov-file#advanced-administration
    scylla_name: Optional[str] = Field(alias="scylla.name", default=None)
    scylla_table_names: Optional[str] = Field(alias="scylla.table.names", default=None)
    scylla_user: Optional[str] = Field(alias="scylla.user", default=None)
    scylla_password: Optional[str] = Field(alias="scylla.password", default=None)

    # kafka-connect-scylladb
    # see https://github.com/scylladb/kafka-connect-scylladb/blob/master/documentation/CONFIG.md
    scylladb_contact_points: Optional[str] = Field(alias="scylladb.contact.points", default=None)
    scylladb_keyspace: Optional[str] = Field(alias="scylladb.keyspace", default=None)
    scylladb_username: Optional[str] = Field(alias="scylladb.username", default=None)
    scylladb_password: Optional[str] = Field(alias="scylladb.password", default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SctKafkaConfiguration(BaseModel):
    source: str = Field(
        exclude=True
    )  # url to specific release or hub version ex. 'hub:scylladb/scylla-cdc-source-connector:1.1.2'
    name: str  # connector name, each one should be named differently
    config: ConnectorConfiguration
