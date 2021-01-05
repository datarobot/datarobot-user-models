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
    drum fit --code-dir <custom code dir> --input <input.csv> --output <output_dir> --target-type binary --target <target feature> --positive-class-label <class-label-1> --negative-class-label <class-label-0> --verbose

    # Run regression user model in fit mode.
    drum fit --code-dir <custom code dir> --input <input.csv> --output <output_dir> --target-type regression --target <target feature> --verbose
"""
import os
import signal
import sys
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import (
    config_logging,
    RunMode,
    ArgumentsOptions,
    ArgumentOptionsEnvVars,
)
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
                and RunMode(runtime.options.subparser_name) == RunMode.SERVER
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

        def _extend_sys_argv_with_env_vars():
            """
            We want actions, types and checks defined in args parsers to take care of arguments,
            even if they are defined by env var.
            For this purpose, if arg is provided by env var, it's added to sys.argv according to the conditions:
            - env var is set;
            - arg is legit for the sub parser being used;
            - arg is not already provided as command line argument;
            :return:
            """

            def _is_arg_registered_in_subparser(sub_parser, arg):
                """
                Check that argument is legit for the subparser. E.g. for `fit` --target is legit, but not --address
                :param sub_parser: argparse.ArgumentParser
                :param arg: argument option, e.g. --address
                :return: True/False
                """

                # This accesses private parser's properties.
                for action in sub_parser._actions:
                    if arg in action.option_strings:
                        return True
                return False

            sub_parser_command = sys.argv[1]
            sub_parser = CMRunnerArgsRegistry._parsers.get(sub_parser_command)
            if sub_parser is None:
                return

            for env_var_key in ArgumentOptionsEnvVars.VALUE_VARS + ArgumentOptionsEnvVars.BOOL_VARS:
                env_var_value = os.environ.get(env_var_key)
                if env_var_value is not None and len(env_var_value) == 0:
                    env_var_value = None
                if (
                    # if env var is set
                    env_var_value is not None
                    # and if argument related to env var is supported by parser
                    and _is_arg_registered_in_subparser(
                        sub_parser, ArgumentsOptions.__dict__[env_var_key]
                    )
                    # and if argument related to env var is not already in sys.argv
                    and ArgumentsOptions.__dict__[env_var_key] not in sys.argv
                ):
                    # special handling for --class_labels_file as it and --class-labels can not be provided together
                    if (
                        env_var_key == ArgumentOptionsEnvVars.CLASS_LABELS_FILE
                        and ArgumentsOptions.CLASS_LABELS in sys.argv
                        or env_var_key == ArgumentOptionsEnvVars.CLASS_LABELS
                        and ArgumentsOptions.CLASS_LABELS_FILE in sys.argv
                    ):
                        continue

                    args_to_add = [ArgumentsOptions.__dict__[env_var_key]]
                    if env_var_key in ArgumentOptionsEnvVars.VALUE_VARS:
                        if env_var_key == ArgumentOptionsEnvVars.CLASS_LABELS:
                            args_to_add.extend(env_var_value.split())
                        else:
                            args_to_add.extend([env_var_value])

                    sys.argv.extend(args_to_add)

        _extend_sys_argv_with_env_vars()

        options = arg_parser.parse_args()
        CMRunnerArgsRegistry.verify_options(options)
        runtime.options = options

        # mlpiper restful_component relies on SIGINT to shutdown nginx and uwsgi,
        # so we don't intercept it.
        if hasattr(runtime.options, "production") and runtime.options.production:
            pass
        else:
            signal.signal(signal.SIGINT, signal_handler)

        from datarobot_drum.drum.drum import CMRunner

        CMRunner(runtime).run()


if __name__ == "__main__":
    main()
