"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import traceback

from datarobot_drum.drum.server import (
    empty_api_blueprint,
    get_flask_app,
    HTTP_513_DRUM_PIPELINE_ERROR,
)
from datarobot_drum.drum.common import verbose_stdout
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX, RunMode

from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.args_parser import ArgumentsOptions

from termcolor import colored

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
logger.setLevel(logging.ERROR)
logger_drum = logging.getLogger(LOGGER_NAME_PREFIX)


class DrumRuntime:
    def __init__(self):
        self.initialization_succeeded = False
        self.options = None
        self.cm_runner = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not exc_type:
            return True  # no exception, just return

        if not self.options:
            # exception occurred before args were parsed
            return False  # propagate exception further

        logger_drum.exception("")

        run_mode = RunMode(self.options.subparser_name)

        if (
            hasattr(self.options, "show_stacktrace")
            and not self.options.show_stacktrace
            and not (run_mode == RunMode.SERVER and self.options.with_error_server)
        ):
            if exc_type == DrumCommonException:
                exit(1)
            return False

        if run_mode != RunMode.SERVER:
            # drum is not run in server mode
            return False  # propagate exception further

        if getattr(self.options, "docker", None):
            # when run in docker mode,
            # drum is started from docker with the same options except `--docker`.
            # thus error server is started in docker as well.
            # return here to avoid starting error server 2nd time.
            return False  # propagate exception further

        if not self.options.with_error_server:
            # force start is not set
            return False  # propagate exception further

        if self.initialization_succeeded:
            # pipeline initialization was successful.
            # exceptions that occur during pipeline running
            # must be propagated further
            return False  # propagate exception further

        # start 'error server'
        host_port_list = self.options.address.split(":", 1)
        host = host_port_list[0]
        port = int(host_port_list[1]) if len(host_port_list) == 2 else None

        with verbose_stdout(self.options.verbose):
            run_error_server(host, port, exc_value)

        return False  # propagate exception further


def run_error_server(host, port, exc_value):
    model_api = empty_api_blueprint()

    @model_api.route("/", methods=["GET"])
    @model_api.route("/ping/", methods=["GET"], strict_slashes=False)
    @model_api.route("/health/", methods=["GET"])
    def health():
        return {"message": "ERROR: {}".format(exc_value)}, HTTP_513_DRUM_PIPELINE_ERROR

    @model_api.route("/predict/", methods=["POST"])
    @model_api.route("/predictions/", methods=["POST"])
    @model_api.route("/predictUnstructured/", methods=["POST"])
    @model_api.route("/predictionsUnstructured/", methods=["POST"])
    @model_api.route("/invocations", methods=["POST"])
    def predict():
        return {"message": "ERROR: {}".format(exc_value)}, HTTP_513_DRUM_PIPELINE_ERROR

    @model_api.route("/transform/", methods=["POST"])
    def transform():
        return {"message": "ERROR: {}".format(exc_value)}, HTTP_513_DRUM_PIPELINE_ERROR

    app = get_flask_app(model_api)
    app.run(host, port)
