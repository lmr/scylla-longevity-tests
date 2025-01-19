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

import os
import sys
import pathlib
import configparser
import datetime
import logging

import boto3
from botocore.exceptions import NoCredentialsError
from gimme_aws_creds.main import GimmeAWSCreds
from gimme_aws_creds.ui import CLIUserInterface

account_id = os.environ.get('SCT_AWS_ACCOUNT_ID', '797456418907')
role_to_assign = os.environ.get('SCT_AWS_ROLE_NAME', 'DeveloperAccessRole')

LOGGER = logging.getLogger(__name__)


def can_get_to_aws_account():
    """
    identify if we are connected to the account we are expecting
    """
    try:
        session = boto3.Session()
        sts = session.client("sts")
        response = sts.get_caller_identity()
        assert response['Account'] == account_id
        LOGGER.info("logged in as %s", response['Arn'])
    except (NoCredentialsError, AssertionError):
        LOGGER.exception("failed")
        return False
    return True


def check_current_token_expiration() -> str | None:
    config = configparser.ConfigParser()
    config.read(pathlib.Path("~/.aws/credentials").expanduser())
    for section in config.sections():
        if account_id in section and role_to_assign in section:
            # check token expiration
            expire_date = config.get(section, 'x_security_token_expires')
            expire_date = datetime.datetime.fromisoformat(expire_date)
            if expire_date > datetime.datetime.now(tz=datetime.timezone.utc):
                return section
    return None


def try_auth_with_okta():
    if not can_get_to_aws_account():
        if not pathlib.Path("~/.okta_aws_login_config").expanduser().exists():
            raise ValueError("OKTA isn't configured, use this guide to configure it:\n"
                             "https://www.notion.so/How-to-login-on-AWS-CLI-and-assume-a-role-bcc4e36042ea4ae9a76ea65e7aafe283?pvs=4")
        if profile := check_current_token_expiration():
            pass
        else:
            pattern = f'/:{account_id}:/'
            cli_ui = CLIUserInterface(argv=[sys.argv[0], '--roles', pattern])
            creds = GimmeAWSCreds(ui=cli_ui)

            for data in creds.iter_selected_aws_credentials():
                arn = data['role']['arn']
                if role_to_assign in arn:
                    creds.write_aws_creds_from_data(data)
                    profile = data['profile']['name']

        os.environ['AWS_PROFILE'] = profile
        boto3.DEFAULT_SESSION = None
        can_get_to_aws_account()
