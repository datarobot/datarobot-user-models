"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os
import sys
from distutils.util import strtobool

from contextlib import contextmanager
from pathlib import Path

from datarobot_drum.drum.enum import (
    MODEL_CONFIG_FILENAME,
    PredictionServerMimetypes,
    PayloadFormat,
)
from datarobot_drum.drum.exceptions import DrumCommonException


@contextmanager
def reroute_stdout_to_stderr():
    keep = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = keep


@contextmanager
def verbose_stdout(verbose):
    new_target = sys.stdout
    old_target = sys.stdout
    if not verbose:
        new_target = open(os.devnull, "w")
        sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def config_logging():
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)s:  %(message)s")


def get_metadata(options):
    code_dir = Path(options.code_dir)
    if options.model_config is None:
        raise DrumCommonException(
            "You must have a file with the name {} in the directory {}. \n"
            "You don't. \nWhat you do have is these files: \n{} ".format(
                MODEL_CONFIG_FILENAME, code_dir, os.listdir(code_dir)
            )
        )
    return options.model_config


class SupportedPayloadFormats:
    def __init__(self):
        self._formats = {}
        self._mimetype_to_payload_format = {
            None: PayloadFormat.CSV,
            PredictionServerMimetypes.EMPTY: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_CSV: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_PLAIN: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_MTX: PayloadFormat.MTX,
            PredictionServerMimetypes.APPLICATION_X_APACHE_ARROW_STREAM: PayloadFormat.ARROW,
        }

    def add(self, payload_format, format_version=None):
        self._formats[payload_format] = format_version

    def is_mimetype_supported(self, mimetype):
        payload_format = self._mimetype_to_payload_format.get(mimetype)
        if payload_format is None:
            return False

        return payload_format in self._formats

    def __iter__(self):
        for payload_format, format_version in self._formats.items():
            yield payload_format, format_version


def make_predictor_capabilities(supported_payload_formats):
    return {
        "supported_payload_formats": {
            payload_format: format_version
            for payload_format, format_version in supported_payload_formats
        }
    }


try:
    import pyarrow
except ImportError:
    pyarrow = None


def get_pyarrow_module():
    return pyarrow


def verify_pyarrow_module():
    if pyarrow is None:
        raise ModuleNotFoundError("Please install pyarrow to support Arrow format")
    return pyarrow


def to_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return strtobool(value)


FIT_METADATA_FILENAME = "fit_runtime_data.json"
