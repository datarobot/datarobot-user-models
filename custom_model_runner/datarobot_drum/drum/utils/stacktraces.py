"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
from contextlib import contextmanager

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


@contextmanager
def capture_R_traceback_if_errors(r_handler, logger):
    from rpy2.rinterface_lib.embedded import RRuntimeError

    try:
        yield
    except RRuntimeError as e:
        try:
            out = "\n".join(r_handler("capture.output(traceback(max.lines = 50))"))
            logger.error("R Traceback:\n{}".format(str(out)))
        except Exception as traceback_exc:
            e.context = {
                "r_traceback": "(an error occurred while getting traceback from R)",
                "t_traceback_err": traceback_exc,
            }
        raise
