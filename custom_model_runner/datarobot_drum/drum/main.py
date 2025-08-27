"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler

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

# Monkey patching for gevent compatibility if running with gunicorn-gevent
if (
    "gunicorn-gevent" in sys.argv or os.environ.get("SERVER_TYPE") == "gunicorn-gevent"
):
    try:
        from gevent import monkey

        monkey.patch_all()
    except ImportError:
        pass

from datarobot_drum.drum.common import config_logging, setup_otel
from datarobot_drum.drum.utils.setup import setup_options
from datarobot_drum.drum.enum import RunMode
from datarobot_drum.drum.enum import ExitCodes
from datarobot_drum.drum.exceptions import DrumSchemaValidationException
from datarobot_drum.drum.runtime import DrumRuntime
from datarobot_drum.runtime_parameters.runtime_parameters import (
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
            # Let traceer offload accumulated spans before shutdown.
            if runtime.trace_provider is not None:
                runtime.trace_provider.shutdown()
            if runtime.metric_provider is not None:
                runtime.metric_provider.shutdown()
            if runtime.log_provider is not None:
                runtime.log_provider.shutdown()

            os._exit(130)

        try:
            options = setup_options()
            runtime.options = options
        except Exception as exc:
            print(str(exc))
            exit(255)

        trace_provider, metric_provider, log_provider = setup_otel(RuntimeParameters, options)
        runtime.trace_provider = trace_provider
        runtime.metric_provider = metric_provider
        runtime.log_provider = log_provider

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
