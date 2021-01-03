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
from sdcm.utils.decorators import retrying

from time import sleep
from ldap3 import Server, Connection, ALL, ALL_ATTRIBUTES
from ldap3.core.exceptions import LDAPSocketOpenError


LOGGER = logging.getLogger(__name__)

LDAP_IMAGE = "osixia/openldap:1.4.0"
LDAP_PORT = 389
LDAP_SSL_PORT = 636
LDAP_SSH_TUNNEL_LOCAL_PORT = 5001
LDAP_SSH_TUNNEL_SSL_PORT = 5002
ORGANISATION = 'ScyllaDB'
LDAP_DOMAIN = 'scylladb.com'
LDAP_PASSWORD = 'scylla'
LDAP_ROLE = 'scylla_ldap'
LDAP_USERS = ['cassandra', 'scylla-qa', 'dummy-user']
LDAP_BASE_OBJECT = (lambda l: ','.join([f'dc={part}' for part in l.split('.')]))(LDAP_DOMAIN)


class LdapServerNotReady(Exception):
    pass


class LdapContainerMixin:  # pylint: disable=too-few-public-methods
    ldap_server = None
    ldap_conn = None
    ldap_server_port = None
    ldap_server_ssl_port = None

    def ldap_container_run_args(self) -> dict:
        return dict(image=LDAP_IMAGE,
                    name=f"{self.name}-ldap-server",
                    detach=True,
                    ports={f"{LDAP_PORT}/tcp": None, f"{LDAP_SSL_PORT}/tcp": None, },
                    environment=[f'LDAP_ORGANISATION={ORGANISATION}', f'LDAP_DOMAIN={LDAP_DOMAIN}',
                                 f'LDAP_ADMIN_PASSWORD={LDAP_PASSWORD}'],
                    )

    @staticmethod
    @retrying(n=10, sleep_time=6, allowed_exceptions=(LdapServerNotReady, LDAPSocketOpenError))
    def create_ldap_connection(ip, ldap_port, user, password):
        LdapContainerMixin.ldap_server = Server(host=f'ldap://{ip}:{ldap_port}', get_info=ALL)
        LdapContainerMixin.ldap_conn = Connection(server=LdapContainerMixin.ldap_server, user=user, password=password)
        sleep(5)
        LdapContainerMixin.ldap_conn.open()
        LdapContainerMixin.ldap_conn.bind()

    @staticmethod
    def is_ldap_connection_bound():
        return LdapContainerMixin.ldap_conn.bound if LdapContainerMixin.ldap_conn else False

    @staticmethod
    def add_ldap_entry(ip, ldap_port, user, password, ldap_entry):
        if not LdapContainerMixin.is_ldap_connection_bound():
            sleep(5)
            LOGGER.info('this is creating a ldap_connection with values {}'.format([ip, ldap_port, user, password]))
            LdapContainerMixin.create_ldap_connection(ip, ldap_port, user, password)
        LdapContainerMixin.ldap_conn.add(*ldap_entry)

    @staticmethod
    def search_ldap_entry(search_base, search_filter):
        LdapContainerMixin.ldap_conn.search(search_base=search_base, search_filter=search_filter,
                                            attributes=ALL_ATTRIBUTES)
        return LdapContainerMixin.ldap_conn.entries

    @staticmethod
    def modify_ldap_entry(*args, **kwargs):
        return LdapContainerMixin.ldap_conn.modify(*args, **kwargs)
