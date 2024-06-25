import json
import os
from collections import namedtuple
from mlpiper.common import model_connector_constants


def model_connector_mode():
    """
    Check if running in model connector mode
    :return: True/False
    """
    if model_connector_constants.JSON_INFO_ENV in os.environ:
        return True
    return False


def get_options_from_env_json_info(orig_options):
    """
    Generate a named tuple of options from the json info provided via the environment.
    Adding the cmd key from the orig_options to the result named tuple
    :param orig_options: Original options obtained from command line
    :return: The final options named tuple to use
    """
    if model_connector_constants.JSON_INFO_ENV in os.environ:
        options_from_json = json.loads(
            os.environ[model_connector_constants.JSON_INFO_ENV]
        )
        if model_connector_constants.MODEL_CONNECTOR_CMD_OPTION in orig_options:
            options_from_json[
                model_connector_constants.MODEL_CONNECTOR_CMD_OPTION
            ] = getattr(
                orig_options, model_connector_constants.MODEL_CONNECTOR_CMD_OPTION
            )
        new_options = {}
        for k, v in options_from_json.items():
            new_k = k.replace("-", "_")
            new_options[new_k] = v
        return namedtuple("options", new_options.keys())(*new_options.values())
    else:
        return orig_options
