#!/usr/bin/env python3

"""
Custom model runner is a tool to work with user models in scoring, training and other modes.

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
    drum fit --code-dir <custom code dir> --input <input.csv> --output <output_dir> --target <target feature> --positive-class-label <class-label-1> --negative-class-label <class-label-0> --verbose

    # Run regression user model in fit mode.
    drum fit --code-dir <custom code dir> --input <input.csv> --output <output_dir> --target <target feature> --verbose
"""
import os
import signal
import sys
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import config_logging
from datarobot_drum.drum.runtime import DrumRuntime


def main():
    with DrumRuntime() as runtime:
        config_logging()

        def signal_handler(sig, frame):
            # The signal is assigned so the stacktrace is not presented when Ctrl-C is pressed.
            # The cleanup itself is done only if we are NOT running in performance test mode which
            # has its own cleanup
            print("\nCtrl+C pressed, aborting drum")

            if (
                runtime.options
                and runtime.options.docker
                and not runtime.options.in_perf_mode_internal
            ):
                try:
                    import requests
                except ImportError:
                    print(
                        "WARNING: 'requests' package is not found - "
                        "cannot send shutdown to server",
                        file=sys.stderr,
                    )
                else:
                    url = "http://{}/shutdown/".format(runtime.options.address)
                    print("Sending shutdown to server: {}".format(url))
                    requests.post(url, timeout=2)

            os._exit(130)

        signal.signal(signal.SIGINT, signal_handler)

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

        options = arg_parser.parse_args()
        CMRunnerArgsRegistry.verify_options(options)

        runtime.options = options

        from datarobot_drum.drum.drum import CMRunner

        CMRunner(runtime).run()


if __name__ == "__main__":
    main()
