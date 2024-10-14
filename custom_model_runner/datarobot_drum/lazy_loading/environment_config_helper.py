#
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc.
#
# This is proprietary source code of DataRobot, Inc. and its
# affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
import json
import logging
import os

from custom_model_runner.datarobot_drum.lazy_loading.constants import MLOPS_REPOSITORY_SECRET_PREFIX

logger = logging.getLogger(__name__)


def handle_credentials_param(credential_id):
    """
    Take a credential_id and create corresponding credentials object from env variable
    so the client that requires the credentials can use that object.

    :param credential_id:
    """
    credentials_env_variable = MLOPS_REPOSITORY_SECRET_PREFIX + credential_id.upper()
    param_json = os.environ.get(credentials_env_variable, None)
    if param_json is None:
        raise EnvironmentError(
            "expected environment variable '{}' to be set".format(credentials_env_variable)
        )
    # logger.debug(f"param_json: {param_json}") TODO: mask credentials for logging

    json_content = json.loads(param_json)
    if param_json is None:
        raise EnvironmentError(
            "expected environment variable '{}' to be json".format(credentials_env_variable)
        )

    logger.debug("Successfully loaded JSON content")
    return json_content["payload"]
