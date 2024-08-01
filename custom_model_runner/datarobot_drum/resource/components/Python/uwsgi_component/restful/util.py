"""
For internal use only. A generic utility for the RESTful module.
"""
import subprocess
from datarobot_drum.resource.components.Python.uwsgi_component.common.uwsgi_nginx_core_exception import (
    UwsgiNginxCoreException,
)


def verify_tool_installation(ver_cmd, dev_against_ver, logger):
    try:
        tool = ver_cmd.split()[0]
        logger.info("Verifying '{tool}' proper installation ...".format(tool=tool))

        ver_msg = subprocess.check_output(ver_cmd, shell=True)
        msg = "'{}', PM developed against: '{}'".format(tool, dev_against_ver)
        if ver_msg:
            msg += ", runtime: '{}'".format(ver_msg)
        logger.info(msg)
    except subprocess.CalledProcessError:
        raise UwsgiNginxCoreException(
            "'{tool}' is not installed! Please make sure to provide a docker container, with"
            "proper '{tool}' installation.".format(tool=tool)
        )
