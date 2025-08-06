"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import sys

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParametersLoader


def setup_options(args=None):
    """
    Setup options for the Drum runtime.
    This function is used to set up the command line arguments and options
    for the Drum runtime, including environment variables and maximum workers.

    Parameters
    ----------
    args : list, optional
        List of command line arguments to parse. If None, uses sys.argv[1:].
        Defaults to None, which means it will use sys.argv[1:].

    Returns
    -------
    options : argparse.Namespace
        Parsed command line options as an argparse.Namespace object.
    """
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

    options = arg_parser.parse_args(args)

    """Set max workers from runtime parameters if available."""
    if RuntimeParameters.has("CUSTOM_MODEL_WORKERS"):
        options.max_workers = RuntimeParameters.get("CUSTOM_MODEL_WORKERS")
    elif "max_workers" not in options or options.max_workers is None:
        options.max_workers = 1  # Default to 1 worker if not specified
    else:
        options.max_workers = int(options.max_workers)

    CMRunnerArgsRegistry.verify_options(options)

    if "runtime_params_file" in options and options.runtime_params_file:
        loader = RuntimeParametersLoader(options.runtime_params_file, options.code_dir)
        loader.setup_environment_variables()

    if "lazy_loading_file" in options and options.lazy_loading_file:
        LazyLoadingHandler.setup_environment_variables_from_values_file(options.lazy_loading_file)

    return options
