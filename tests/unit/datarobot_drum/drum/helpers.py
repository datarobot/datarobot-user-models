"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# file name "helpers" avoids name conflict with utils/ directory
import os

from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParameters


def inject_runtime_parameter(name: str, value: str):
    """
    Inject a runtime parameter into the environment as a JSON string,
    as if the RuntimeParametersLoader had loaded it.
    Purpose: test more of the real predictor/adaptor stack with less mocking.
    """
    os.environ[
        f"{RuntimeParameters.PARAM_PREFIX}_{name}"
    ] = f'{{"payload": "{value}", "type": "string"}}'


def unset_runtime_parameter(name: str):
    """
    Unset a runtime parameter int the environment,
    as if it was never defined.
    """
    os.environ.pop(f"{RuntimeParameters.PARAM_PREFIX}_{name}", None)
