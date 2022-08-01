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
from sdcm.provision.azure.provisioner import AzureProvisioner
from sdcm.provision.gce.provisioner import GCEProvisioner
from sdcm.provision.provisioner import provisioner_factory

provisioner_factory.register_provisioner(backend="azure", provisioner_class=AzureProvisioner)
provisioner_factory.register_provisioner(backend="gce", provisioner_class=GCEProvisioner)
