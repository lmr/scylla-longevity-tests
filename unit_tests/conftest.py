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

import os
import logging
import collections
from pathlib import Path

import pytest

from sdcm import wait, sct_config
from sdcm.cluster import BaseNode
from sdcm.prometheus import start_metrics_server
from sdcm.provision import provisioner_factory
from sdcm.remote import RemoteCmdRunnerBase
from sdcm.sct_events.continuous_event import ContinuousEventsRegistry
from sdcm.sct_provision import region_definition_builder
from sdcm.utils.docker_remote import RemoteDocker
from sdcm.utils.common import update_certificates

from unit_tests.dummy_remote import LocalNode, LocalScyllaClusterDummy

from unit_tests.lib.events_utils import EventsUtilsMixin
from unit_tests.lib.fake_provisioner import FakeProvisioner
from unit_tests.lib.fake_region_definition_builder import FakeDefinitionBuilder
from unit_tests.lib.fake_remoter import FakeRemoter


@pytest.fixture(scope='module')
def events():
    class LocalMixing(EventsUtilsMixin):
        pass
    mixing = LocalMixing()
    mixing.setup_events_processes(events_device=True, events_main_device=False, registry_patcher=True)
    yield mixing

    mixing.teardown_events_processes()


@pytest.fixture(scope='session')
def prom_address():
    yield start_metrics_server()


@pytest.fixture(name='docker_scylla', scope='function')
def fixture_docker_scylla(request: pytest.FixtureRequest):
    docker_scylla_args = {}
    if test_marker := request.node.get_closest_marker("docker_scylla_args"):
        docker_scylla_args = test_marker.kwargs
    ssl = docker_scylla_args.get('ssl')
    # make sure the path to the file is base on the host path, and not as the docker internal path i.e. /sct/
    # since we are going to mount it in a DinD (docker-inside-docker) setup
    base_dir = os.environ.get("_SCT_BASE_DIR", None)
    entryfile_path = Path(base_dir) if base_dir else Path(__file__).parent.parent
    entryfile_path = entryfile_path / 'docker' / 'scylla-sct' / ('entry_ssl.sh' if ssl else 'entry.sh')

    alternator_flags = "--alternator-port 8000 --alternator-write-isolation=always"
    docker_version = "scylladb/scylla-nightly:5.2.0-dev-0.20220820.516089beb0b8"
    cluster = LocalScyllaClusterDummy()

    if ssl:
        curr_dir = os.getcwd()
        try:
            os.chdir(Path(__file__).parent.parent)
            update_certificates()
        finally:
            os.chdir(curr_dir)

    ssl_dir = (Path(__file__).parent.parent / 'data_dir' / 'ssl_conf').absolute()
    scylla = RemoteDocker(LocalNode("scylla", cluster), image_name=docker_version,
                          command_line=f"--smp 1 --experimental 1 {alternator_flags}",
                          extra_docker_opts=(f'-p 8000 -p {BaseNode.CQL_PORT} --cpus="1" -v {entryfile_path}:/entry.sh'
                                             f' -v {ssl_dir}:/etc/scylla/ssl_conf'
                                             ' --entrypoint /entry.sh'))
    DummyRemoter = collections.namedtuple('DummyRemoter', ['run', 'sudo'])
    scylla.remoter = DummyRemoter(run=scylla.run, sudo=scylla.run)

    def db_up():
        try:
            return scylla.is_port_used(port=BaseNode.CQL_PORT, service_name="scylla-server")
        except Exception as details:  # pylint: disable=broad-except
            logging.error("Error checking for scylla up normal: %s", details)
            return False

    def db_alternator_up():
        try:
            return scylla.is_port_used(port=8000, service_name="scylla-server")
        except Exception as details:  # pylint: disable=broad-except
            logging.error("Error checking for scylla up normal: %s", details)
            return False

    wait.wait_for(func=db_up, step=1, text='Waiting for DB services to be up', timeout=120, throw_exc=True)
    wait.wait_for(func=db_alternator_up, step=1, text='Waiting for DB services to be up alternator)',
                  timeout=120, throw_exc=True)

    yield scylla
    scylla.kill()


@pytest.fixture
def fake_remoter():
    RemoteCmdRunnerBase.set_default_remoter_class(FakeRemoter)
    return FakeRemoter


@pytest.fixture(scope='session', autouse=True)
def fake_provisioner():  # pylint: disable=no-self-use
    provisioner_factory.register_provisioner(backend="fake", provisioner_class=FakeProvisioner)


@pytest.fixture(scope='session', autouse=True)
def fake_region_definition_builder():  # pylint: disable=no-self-use
    region_definition_builder.register_builder(backend="fake", builder_class=FakeDefinitionBuilder)


@pytest.fixture(scope="function", name="params")
def fixture_params(request: pytest.FixtureRequest):
    if sct_config_marker := request.node.get_closest_marker("sct_config"):
        config_files = sct_config_marker.kwargs.get('files')
        os.environ['SCT_CONFIG_FILES'] = config_files

    os.environ['SCT_CLUSTER_BACKEND'] = 'docker'
    params = sct_config.SCTConfiguration()  # pylint: disable=attribute-defined-outside-init

    yield params

    for k in os.environ:
        if k.startswith('SCT_'):
            del os.environ[k]


@pytest.fixture(scope='function', autouse=True)
def fixture_cleanup_continuous_events_registry():
    ContinuousEventsRegistry().cleanup_registry()


def pytest_addoption(parser):
    parser.addoption("--generate-nemesis-file", action="store_true", default="default name")


@pytest.fixture(scope='session')
def generate_nemesis_file(request: pytest.FixtureRequest) -> bool:
    return request.config.option.generate_nemesis_file
