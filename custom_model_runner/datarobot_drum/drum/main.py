"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

#!/usr/bin/env python3

"""
Custom model runner is a tool to work with user models in scoring, custom tasks, custom estimator tasks, and other
modes.

Examples:

    # Prepare custom_model folder containing model artifact.
    # It also and may contain custom.py file implementing custom predict() method.

    # Run binary classification user model in a batch prediction mode. If output parameter is omitted,
    # results will be printed.
    drum score --code-dir ~/custom_model3/ --input input.csv --output output.csv --positive-class-label yes
          --negative-class-label no

    # Run regression user model in a predict mode.
    drum score --code-dir ~/custom_model3/ --input input.csv --output output.csv

    # Run binary classification user model in a prediction server mode.
    drum server --code-dir ~/custom_model3/ --positive-class-label yes --negative-class-label no
          --address host:port

    # Run regression user model in a prediction server mode.
    drum server --code-dir ~/custom_model3/ --address host:port

    # Run binary classification user model in fit mode.
    drum fit --code-dir <custom code dir> --input <input.csv> --output <output_dir> --target-type binary --target <target feature> --positive-class-label <class-label-1> --negative-class-label <class-label-0> --verbose

    # Run regression user model in fit mode.
    drum fit --code-dir <custom code dir> --input <input.csv> --output <output_dir> --target-type regression --target <target feature> --verbose
"""
import os
import signal
import sys

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import config_logging
from datarobot_drum.drum.enum import RunMode
from datarobot_drum.drum.enum import ExitCodes
from datarobot_drum.drum.exceptions import DrumSchemaValidationException
from datarobot_drum.drum.runtime import DrumRuntime
from datarobot_drum.runtime_parameters.exceptions import RuntimeParameterException
from datarobot_drum.runtime_parameters.runtime_parameters import (
    RuntimeParametersLoader,
    RuntimeParameters,
)


def main():
    with DrumRuntime() as runtime:
        config_logging()

        def signal_handler(sig, frame):
            # The signal is assigned so the stacktrace is not presented when Ctrl-C is pressed.
            # The cleanup itself is done only if we are NOT running in performance test mode which
            # has its own cleanup
            print("\nCtrl+C pressed, aborting drum")

            if runtime.options and RunMode(runtime.options.subparser_name) == RunMode.SERVER:
                if runtime.cm_runner:
                    runtime.cm_runner.terminate()

            os._exit(130)

        arg_parser = CMRunnerArgsRegistry.get_arg_parser()

        try:
            import argcomplete
        except ImportError:
            print(
                "WARNING: autocompletion of arguments is not supported "
                "as 'argcomplete' package is not found",
                file=sys.stderr,
            )
        else:
            # argcomplete call should be as close to the beginning as possible
            argcomplete.autocomplete(arg_parser)

        CMRunnerArgsRegistry.extend_sys_argv_with_env_vars()

        options = arg_parser.parse_args()
        CMRunnerArgsRegistry.verify_options(options)
        if "runtime_params_file" in options and options.runtime_params_file:
            try:
                loader = RuntimeParametersLoader(options.runtime_params_file, options.code_dir)
                loader.setup_environment_variables()
            except RuntimeParameterException as exc:
                print(str(exc))
                exit(255)
        if RuntimeParameters.has("CUSTOM_MODEL_WORKERS"):
            options.max_workers = RuntimeParameters.get("CUSTOM_MODEL_WORKERS")
        runtime.options = options

        # mlpiper restful_component relies on SIGINT to shutdown nginx and uwsgi,
        # so we don't intercept it.
        if hasattr(runtime.options, "production") and runtime.options.production:

            def raise_keyboard_interrupt(sig, frame):
                raise KeyboardInterrupt("Triggered from {}".format(sig))

            signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
        else:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        from datarobot_drum.drum.drum import CMRunner

        try:
            runtime.cm_runner = CMRunner(runtime)
            runtime.cm_runner.run()
        except DrumSchemaValidationException:
            sys.exit(ExitCodes.SCHEMA_VALIDATION_ERROR.value)


if __name__ == "__main__":
    main()
