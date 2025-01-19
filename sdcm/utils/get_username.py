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
import getpass
import subprocess


def is_email_in_scylladb_domain(email_addr: str) -> bool:
    return bool(email_addr and "@scylladb.com" in email_addr)


def get_email_user(email_addr: str) -> str:
    return email_addr.strip().split("@")[0]


def get_username() -> str:  # pylint: disable=too-many-return-statements  # noqa: PLR0911
    # First we check if user is being impersonated by an api call
    actual_user_from_request = os.environ.get('BUILD_USER_REQUESTED_BY')
    if actual_user_from_request:
        return actual_user_from_request

    # Then check that we running on Jenkins try to get user email
    email = os.environ.get('BUILD_USER_EMAIL')
    if is_email_in_scylladb_domain(email):
        return get_email_user(email)

    user_id = os.environ.get('BUILD_USER_ID')
    if user_id:
        user_id = user_id.replace("[", "").replace("]", "")
        return user_id

    current_linux_user = getpass.getuser()
    if current_linux_user in ["jenkins", "runner"]:
        return current_linux_user
    if current_linux_user == "ubuntu":
        return "sct-runner"

    # We are not on Jenkins and running in Hydra, try to get email from Git
    # when running in Hydra there are env issues so we pass it using SCT_GIT_USER_EMAIL variable before docker run
    git_user_email = os.environ.get('GIT_USER_EMAIL')
    if is_email_in_scylladb_domain(git_user_email):
        return get_email_user(git_user_email)

    # We are outside of Hydra
    res = subprocess.run("git config --get user.email",
                         shell=True, check=False, stdout=subprocess.PIPE, encoding="utf-8")
    if is_email_in_scylladb_domain(res.stdout):
        return get_email_user(res.stdout)

    # We didn't find email, fallback to current user with unknown email user identifier
    return "linux_user={}".format(current_linux_user)


if __name__ == "__main__":
    print(get_username())
