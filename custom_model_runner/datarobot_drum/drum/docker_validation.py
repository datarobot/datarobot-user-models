import logging
import os
from os import path

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


def validate_docker_file(dockerfile_path):
    dockerfile = _load_dockerfile(dockerfile_path)
    errors_found = False
    msgs = _check_for_errors(dockerfile)
    for msg in msgs:
        if msg is not None:
            errors_found = True
            print(msg)
    if errors_found:
        print(
            "For example docker files please refer to the 'public_dropin_environments' directory."
        )
    else:
        print("No problems were found with DataRobot requirements for the docker file.  ")


def _check_for_errors(dockerfile):
    validators = [
        _base_image,
        _dr_requirements,
        _code_dir_copy,
        _code_dir_env,
        _address,
        _entrypoint,
    ]
    return [validator(dockerfile) for validator in validators]


def _load_dockerfile(dockerfile_path):
    if path.isdir(dockerfile_path):
        dockerfile_path = path.join(dockerfile_path, "Dockerfile")
    try:
        with open(dockerfile_path) as fd:
            file_contents = fd.read()
    except FileNotFoundError:
        logger.exception("No docker file found, exiting")
        os.exit()
    return file_contents


### Add individual valdiation functions below, and then add them to the list in _check_for_errors
def _base_image(dockerfile):
    if "FROM datarobot/" not in dockerfile:
        return "The dockerfile should be derived from a DataRobot base environment image."
    return None


def _dr_requirements(dockerfile):
    if "dr_requirments.txt" in dockerfile and "datarobot-drum" not in dockerfile:
        return "DataRobot requirements file was not found, please verify that datarobot-drum is installed."
    return None


def _code_dir_copy(dockerfile):
    if "COPY ./" not in dockerfile:
        return (
            "This DockerFile does not appear to copy code for a predict server and might not work"
            "as expected in DataRobot"
        )

    return None


def _code_dir_env(dockerfile):
    if "ENV CODE_DIR" not in dockerfile:
        return (
            "This DockerFile does not set CODE_DIR which is needed by Drum to execute within the DataRobot "
            "environment."
        )
    return None


def _address(dockerfile):
    if "ADDRESS=0.0.0.0:8080" not in dockerfile:
        return (
            "Drum server requires the ADDRESS environment variable to be set.  Please use `ENV ADDRESS=0.0.0.0:8080`"
            " to properly set the address so that the server can accept requests."
        )
    return None


def _entrypoint(dockerfile):
    if "ENTRYPOINT" not in dockerfile:
        return "The dockerfile must contain an ENTRYPOINT that starts the drum server for use within DataRobot."
    return None
