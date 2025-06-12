
"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.

Example:

import json
payload = json.loads(open("input.json", "r").read())
code_dir = (
    'datarobot-user-models/model_templates/python3_dummy_chat'
)

with inline_predictor(code_dir, 'textgeneration') as predictor:
    result = predictor.chat(payload)
    print(result)


NOTE: expects TARGET_NAME env var for text gen, agentic, and VDB types.

"""
import contextlib
import tempfile

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import setup_required_environment_variables, setup_tracer
from datarobot_drum.drum.drum import CMRunner
from datarobot_drum.drum.runtime import DrumRuntime
from datarobot_drum.drum.root_predictors.generic_predictor import GenericPredictorComponent
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParameters


@contextlib.contextmanager
def inline_predictor(code_dir, target_type, *cmd_args):
    with DrumRuntime() as runtime, tempfile.NamedTemporaryFile(mode="wb") as tf:
        # setup
        arg_parser = CMRunnerArgsRegistry.get_arg_parser()
        CMRunnerArgsRegistry.extend_sys_argv_with_env_vars()
        args = [
            'score',
            '--code-dir',
            code_dir,

            # regular score is actually a CLI thing, so it expects input/output,
            # we can ignore these as we handover the predictor directly to the caller to do I/O.
            '--input',
            tf.name,
            '--output',
            tf.name,
            '--target-type',
            target_type,
            *cmd_args,
        ]
        options = arg_parser.parse_args(args)
        CMRunnerArgsRegistry.verify_options(options)
        setup_required_environment_variables(options)

        runtime.options = options
        setup_tracer(RuntimeParameters, options)
        runtime.cm_runner = CMRunner(runtime)
        params = runtime.cm_runner.get_predictor_params()
        predictor = GenericPredictorComponent(params)

        yield predictor.predictor