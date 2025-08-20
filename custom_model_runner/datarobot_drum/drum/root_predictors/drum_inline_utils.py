"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.

Example:

import json

payload = json.loads(open("input.json", "r").read())
code_dir = (
    '/datarobot-user-models/model_templates/python3_dummy_chat'
)

with drum_inline_predictor(target_type=TargetType.AGENTIC_WORKFLOW.value, custom_model_dir=code_dir,
                           target_name='response') as predictor:
    result = predictor.chat(payload)
    print(result)

"""

import contextlib
import os
import tempfile
from typing import Generator, List

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import setup_otel
from datarobot_drum.drum.utils.setup import setup_options
from datarobot_drum.drum.drum import CMRunner
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.drum.runtime import DrumRuntime
from datarobot_drum.drum.root_predictors.generic_predictor import GenericPredictorComponent
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParameters


@contextlib.contextmanager
def drum_inline_predictor(
    target_type: str, custom_model_dir: str, target_name: str, *cmd_args: List[str]
) -> Generator[BaseLanguagePredictor, None, None]:
    """
    Drum run for a custom model code definition. Yields a predictor, ready to work with.
    Caller can work with the predictor directly.

    :param target_type: Target type.
    :param custom_model_dir: Directory where the custom model code artifacts are located.
    :param target_name: Name of the target
    :param cmd_args: Extra command line arguments
    :return:
    """
    with DrumRuntime() as runtime, tempfile.NamedTemporaryFile(mode="wb") as tf:
        # setup

        os.environ["TARGET_NAME"] = target_name
        arg_parser = CMRunnerArgsRegistry.get_arg_parser()
        CMRunnerArgsRegistry.extend_sys_argv_with_env_vars()
        args = [
            "score",
            "--code-dir",
            custom_model_dir,
            # regular score is actually a CLI thing, so it expects input/output,
            # we can ignore these as we hand over the predictor directly to the caller to do I/O.
            "--input",
            tf.name,
            "--output",
            tf.name,
            "--target-type",
            target_type,
            *cmd_args,
        ]

        try:
            options = setup_options(args)
            runtime.options = options
        except Exception as exc:
            print(str(exc))
            exit(255)

        trace_provider, metric_provider = setup_otel(RuntimeParameters, options)
        runtime.cm_runner = CMRunner(runtime)
        params = runtime.cm_runner.get_predictor_params()
        predictor = GenericPredictorComponent(params)

        yield predictor.predictor
        if trace_provider is not None:
            trace_provider.shutdown()
        if metric_provider is not None:
            metric_provider.shutdown()
