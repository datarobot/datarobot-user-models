import argparse
from argparse_formatter import FlexiFormatter

import os
from datarobot_drum.drum.push import PUSH_HELP_TEXT
import sys
import subprocess

from datarobot_drum.drum.description import version
from datarobot_drum.drum.common import LOG_LEVELS, ArgumentsOptions, RunLanguage, TargetType


class CMRunnerArgsRegistry(object):
    SUBPARSER_DEST_KEYWORD = "subparser_name"
    NEW_SUBPARSER_DEST_KEYWORD = "new_mode"
    _parsers = {}

    @staticmethod
    def _tokenize_parser_prog(parser):
        # example:
        # - for score_parser prog is "drum score"
        # - for new_model_parser prog is "drum new model"
        return parser.prog.split(" ")

    @staticmethod
    def _reg_arg_version(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.VERSION,
                action="version",
                version="%(prog)s {version}".format(version=version),
            )

    @staticmethod
    def _reg_arg_verbose(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.VERBOSE,
                action="store_true",
                default=False,
                help="Show verbose output",
            )

    @staticmethod
    def _is_valid_file(arg):
        abs_path = os.path.abspath(arg)
        if not os.path.exists(arg):
            raise argparse.ArgumentTypeError("The file {} does not exist!".format(arg))
        else:
            return os.path.realpath(abs_path)

    @staticmethod
    def _is_valid_dir(arg):
        abs_path = os.path.abspath(arg)
        if not os.path.isdir(arg):
            raise argparse.ArgumentTypeError("The path {} is not a directory!".format(arg))
        else:
            return os.path.realpath(abs_path)

    @staticmethod
    def _is_valid_output_dir(arg):
        abs_path = os.path.abspath(arg)
        if not os.path.isdir(arg):
            raise argparse.ArgumentTypeError(
                "The path {} is not a directory! For custom training models, "
                "the output directory will consist of the artifacts usable "
                "for making predictions. ".format(arg)
            )
        else:
            return os.path.realpath(abs_path)

    @staticmethod
    def _path_does_non_exist(arg):
        if os.path.exists(arg):
            raise argparse.ArgumentTypeError(
                "The path {} already exists! Please provide a non existing path!".format(arg)
            )
        return os.path.abspath(arg)

    @staticmethod
    def _reg_arg_input(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.INPUT,
                default=None,
                required=True,
                type=CMRunnerArgsRegistry._is_valid_file,
                help="Path to an input dataset",
            )

    @staticmethod
    def _reg_arg_output(*parsers):
        for parser in parsers:
            prog_name_lst = CMRunnerArgsRegistry._tokenize_parser_prog(parser)
            if prog_name_lst[1] == ArgumentsOptions.SCORE:
                help_message = "Path to a csv file to output predictions"
                type_callback = os.path.abspath
            elif prog_name_lst[1] == ArgumentsOptions.FIT:
                help_message = (
                    "DRUM will copy the contents of code_dir and create "
                    "the model artifact in the output folder"
                )
                type_callback = CMRunnerArgsRegistry._is_valid_output_dir
            else:
                raise ValueError(
                    "{} argument should be used only by score and fit parsers!".format(
                        ArgumentsOptions.OUTPUT
                    )
                )
            parser.add_argument(
                ArgumentsOptions.OUTPUT, default=None, type=type_callback, help=help_message
            )

    @staticmethod
    def _reg_arg_target_feature_and_filename(*parsers):
        for parser in parsers:
            group = parser.add_mutually_exclusive_group(required=True)
            group.add_argument(
                ArgumentsOptions.TARGET,
                type=str,
                required=False,
                help="Which column to use as the target. Argument is mutually exclusive with {} and {}.".format(
                    ArgumentsOptions.TARGET_FILENAME, ArgumentsOptions.UNSUPERVISED
                ),
            )

            group.add_argument(
                ArgumentsOptions.TARGET_FILENAME,
                type=CMRunnerArgsRegistry._is_valid_file,
                required=False,
                help="A file containing the target values. Argument is mutually exclusive with {} and {}.".format(
                    ArgumentsOptions.TARGET, ArgumentsOptions.UNSUPERVISED
                ),
            )

            group.add_argument(
                ArgumentsOptions.UNSUPERVISED,
                action="store_true",
                required=False,
                default=False,
                help="If present, indicates that this is an unsupervised model."
                " Argument is mutually exclusive with {} and {}.".format(
                    ArgumentsOptions.TARGET, ArgumentsOptions.TARGET_FILENAME
                ),
            )

    @staticmethod
    def _reg_arg_weights(*parsers):
        for parser in parsers:
            group = parser.add_mutually_exclusive_group(required=False)
            group.add_argument(
                ArgumentsOptions.WEIGHTS,
                type=str,
                required=False,
                default=None,
                help="A column name of row weights in your training dataframe. "
                "Argument is mutually exclusive with {}".format(ArgumentsOptions.WEIGHTS_CSV),
            )
            group.add_argument(
                ArgumentsOptions.WEIGHTS_CSV,
                type=CMRunnerArgsRegistry._is_valid_file,
                required=False,
                default=None,
                help="A one column csv file to be parsed as row weights. "
                "Argument is mutually exclusive with {}".format(ArgumentsOptions.WEIGHTS),
            )

    @staticmethod
    def _reg_arg_skip_predict(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.SKIP_PREDICT,
                required=False,
                default=False,
                action="store_true",
                help="By default we will attempt to predict using your model, but we give you the"
                "option to turn this off",
            )

    @staticmethod
    def _reg_arg_pos_neg_labels(*parsers):
        def are_both_labels_present(arg):
            error_message = (
                "\nError - for binary classification case, "
                "both positive and negative class labels have to be provided. \n"
                "See --help option for more information"
            )
            labels = [ArgumentsOptions.POSITIVE_CLASS_LABEL, ArgumentsOptions.NEGATIVE_CLASS_LABEL]
            if not all([x in sys.argv for x in labels]):
                raise argparse.ArgumentTypeError(error_message)
            return arg

        for parser in parsers:
            fit_intuit_message = ""
            prog_name_lst = CMRunnerArgsRegistry._tokenize_parser_prog(parser)
            if prog_name_lst[1] == ArgumentsOptions.FIT:
                fit_intuit_message = "If you do not provide these labels, but your dataset is classification, DRUM will choose the labels for you"

            parser.add_argument(
                ArgumentsOptions.POSITIVE_CLASS_LABEL,
                default=None,
                type=are_both_labels_present,
                help="Positive class label for a binary classification case. " + fit_intuit_message,
            )
            parser.add_argument(
                ArgumentsOptions.NEGATIVE_CLASS_LABEL,
                default=None,
                type=are_both_labels_present,
                help="Negative class label for a binary classification case. " + fit_intuit_message,
            )

    @staticmethod
    def _reg_arg_multiclass_labels(*parsers):
        class RequiredLength(argparse.Action):
            ERROR_MESSAGE = "Multiclass classification requires at least 3 labels."
            MIN_LABELS = 3

            def __call__(self, parser, namespace, values, option_string=None):
                if len(values) < self.MIN_LABELS:
                    raise argparse.ArgumentTypeError(self.ERROR_MESSAGE)
                setattr(namespace, self.dest, values)

        class ParseLabelsFile(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                with open(values) as f:
                    labels = [label for label in f.read().split(os.linesep) if label]
                    if len(labels) < RequiredLength.MIN_LABELS:
                        raise argparse.ArgumentTypeError(RequiredLength.ERROR_MESSAGE)
                    setattr(namespace, "class_labels", labels)

        def are_labels_double_specified(arg):
            label_options = [ArgumentsOptions.CLASS_LABELS_FILE, ArgumentsOptions.CLASS_LABELS]
            if all(opt in sys.argv for opt in label_options):
                error_message = (
                    "\nError - for multiclass classification, either the class labels or "
                    "a class labels file should be provided, but not both.\n"
                    "See --help option for more information"
                )
                raise argparse.ArgumentTypeError(error_message)
            return arg

        for parser in parsers:
            fit_intuit_message = ""
            class_label_order_message = (
                "Labels should be in the order as "
                "the predicted probabilities produced by the model. "
            )
            prog_name_lst = CMRunnerArgsRegistry._tokenize_parser_prog(parser)
            if prog_name_lst[1] == ArgumentsOptions.FIT:
                fit_intuit_message = (
                    "If you do not provide these labels, but your dataset is classification, "
                    "DRUM will choose the labels for you"
                )

            parser.add_argument(
                ArgumentsOptions.CLASS_LABELS,
                default=None,
                type=are_labels_double_specified,
                nargs="+",
                action=RequiredLength,
                help="The class labels for a multiclass classification case. "
                + class_label_order_message
                + fit_intuit_message,
            )

            parser.add_argument(
                ArgumentsOptions.CLASS_LABELS_FILE,
                default=None,
                type=are_labels_double_specified,
                action=ParseLabelsFile,
                help="A file containing newline separated class labels for a multiclass classification case. "
                + class_label_order_message
                + fit_intuit_message,
            )

    @staticmethod
    def _reg_arg_code_dir(*parsers):
        for parser in parsers:
            prog_name_lst = CMRunnerArgsRegistry._tokenize_parser_prog(parser)
            if prog_name_lst[1] == ArgumentsOptions.NEW:
                help_message = "Directory to use for creating the new template"
                type_callback = CMRunnerArgsRegistry._path_does_non_exist
            else:
                help_message = "Custom model code dir"
                type_callback = CMRunnerArgsRegistry._is_valid_dir

            parser.add_argument(
                "-cd",
                ArgumentsOptions.CODE_DIR,
                default=None,
                required=True,
                type=type_callback,
                help=help_message,
            )

    @staticmethod
    def _reg_arg_address(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.ADDRESS,
                default=None,
                required=True,
                help="Prediction server address host[:port]. Default Flask port is: 5000",
            )

    @staticmethod
    def _reg_arg_logging_level(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.LOGGING_LEVEL,
                required=False,
                choices=list(LOG_LEVELS.keys()),
                default="warning",
                help="Logging level to use",
            )

    @staticmethod
    def _reg_arg_docker(*parsers):
        for parser in parsers:
            prog_name_lst = CMRunnerArgsRegistry._tokenize_parser_prog(parser)
            parser.add_argument(
                ArgumentsOptions.DOCKER,
                default=None,
                required=False,
                help="Docker image to use to run {} in the {} mode, "
                "or a directory, containing a Dockerfile, "
                "which can be built into a docker image. ".format(
                    ArgumentsOptions.MAIN_COMMAND, prog_name_lst[1]
                ),
            )

    @staticmethod
    def _reg_arg_memory(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.MEMORY,
                default=None,
                required=False,
                help="Amount of memory to allow the docker container to consume. "
                "The value will be passed to the docker run command to both the  "
                "--memory and --memory-swap parameters. b,k,m,g suffixes are supported",
            ),

    @staticmethod
    def _reg_arg_production_server(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.PRODUCTION,
                action="store_true",
                default=False,
                help="Run prediction server in production mode uwsgi + nginx",
            )

    @staticmethod
    def _reg_arg_max_workers(*parsers):
        def type_callback(arg):
            ret_val = int(arg)
            if ArgumentsOptions.PRODUCTION not in sys.argv:
                raise argparse.ArgumentTypeError(
                    "can only be used in pair with {}".format(ArgumentsOptions.PRODUCTION)
                )
            if ret_val <= 0:
                raise argparse.ArgumentTypeError("must be > 0")
            return ret_val

        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.MAX_WORKERS,
                type=type_callback,
                # default 0 will be mapped into null in pipeline json
                default=0,
                help="Max number of uwsgi workers in server production mode",
            )

    @staticmethod
    def _reg_arg_show_perf(*parsers):
        for parser in parsers:
            parser.add_argument(
                "--show-perf", action="store_true", default=False, help="Show performance stats"
            )

    @staticmethod
    def _reg_arg_samples(*parsers):
        for parser in parsers:
            parser.add_argument("-s", "--samples", type=int, default=None, help="Number of samples")

    @staticmethod
    def _reg_arg_iterations(*parsers):
        for parser in parsers:
            parser.add_argument(
                "-i", "--iterations", type=int, default=None, help="Number of iterations"
            )

    @staticmethod
    def _reg_arg_timeout(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.TIMEOUT, type=int, default=180, help="Test case timeout"
            )

    @staticmethod
    def _reg_arg_in_server(*parsers):
        for parser in parsers:
            parser.add_argument(
                "--in-server",
                action="store_true",
                default=False,
                help="Show performance inside server",
            )

    @staticmethod
    def _reg_arg_url(*parsers):
        for parser in parsers:
            parser.add_argument(
                "--url", default=None, help="Run performance against the given prediction server"
            )

    @staticmethod
    def _reg_arg_language(*parsers):
        for parser in parsers:
            langs = [e.value for e in RunLanguage]
            prog_name_lst = CMRunnerArgsRegistry._tokenize_parser_prog(parser)
            if prog_name_lst[1] == ArgumentsOptions.NEW:
                langs.remove(RunLanguage.JAVA.value)
                required_val = True
            else:
                required_val = False

            parser.add_argument(
                ArgumentsOptions.LANGUAGE,
                choices=langs,
                default=None,
                required=required_val,
                help="Language to use for the new model/env template to create",
            )

    @staticmethod
    def _reg_arg_num_rows(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.NUM_ROWS,
                default="ALL",
                help="Number of rows to use for testing the fit functionality. "
                "Set to ALL to use all rows. Default is 100",
            )

    @staticmethod
    def _reg_arg_with_error_server(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.WITH_ERROR_SERVER,
                action="store_true",
                default=False,
                help="Start server even if pipeline initialization fails.",
            )

    @staticmethod
    def _reg_arg_show_stacktrace(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.SHOW_STACKTRACE,
                action="store_true",
                default=False,
                help="Show stacktrace when error happens.",
            )

    @staticmethod
    def _reg_args_monitoring(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.MONITOR,
                action="store_true",
                default="MONITOR" in os.environ,
                help="Monitor predictions using DataRobot MLOps. True or False. (env: MONITOR)."
                "Monitoring can not be used in unstructured mode.",
            )

            parser.add_argument(
                ArgumentsOptions.DEPLOYMENT_ID,
                default=os.environ.get("DEPLOYMENT_ID", None),
                help="Deployment id to use for monitoring model predictions (env: DEPLOYMENT_ID)",
            )

            parser.add_argument(
                ArgumentsOptions.MODEL_ID,
                default=os.environ.get("MODEL_ID", None),
                help="MLOps model id to use for monitoring predictions (env: MODEL_ID)",
            )

            parser.add_argument(
                ArgumentsOptions.MONITOR_SETTINGS,
                default=os.environ.get("MONITOR_SETTINGS", None),
                help="MLOps setting to use for connecting with the MLOps Agent (env: MONITOR_SETTINGS)",
            )

    # TODO: restrict params to be used with unstructured target type only
    @staticmethod
    def _reg_args_unstructured_mode(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.QUERY,
                default=None,
                help="Additional query params unstructured mode. (Simulates http request query params.)",
            )

            parser.add_argument(
                ArgumentsOptions.CONTENT_TYPE,
                default=None,
                help="Additional content type for unstructured mode. "
                "(Simulates http request Content-Type header, default: 'text/plain; charset=utf8')",
            )

    @staticmethod
    def _reg_arg_target_type(*parsers):
        target_types = [e for e in TargetType.ALL.value]
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.TARGET_TYPE,
                required=False,
                choices=target_types,
                default=None,
                help="Target type",
            )

    @staticmethod
    def _register_subcommand_perf_test(subparsers):
        desc = """
        Test the performance of an inference model. This is done by internally using the server
        sub command to serve the model. Then sending multiple requests to the server and 
        measuring the time it takes to complete each request. 
        
        The test is mixing several requests sizes. The idea is to get a coverage of several
        sizes, from the smallest request containing only 1 row of data, up to the largest 
        request containing up to 50MB of data.
        
        At the end of the test, a summary of the test will be displayed. For each request size,
        the following fields will be shown:
        
         size: size of the requests in bytes or Megabytes.
         samples: number of samples this request size contained.
         iters: number of times this request size was sent
         min: minimum time measured for this request size (in seconds)
         avg: average time of the this request size (in seconds)
         max: maximum time measured for this request size (in seconds)
         used: amount of memory used by drum at the end of this request size (MB)
         container limit: if tests run in docker container, memory limit for it (MB)
         total physical: total amount of physical memory avail on the current machine (MB)
        """
        parser = subparsers.add_parser(
            ArgumentsOptions.PERF_TEST,
            description=desc,
            help="Run performance tests",
            formatter_class=FlexiFormatter,
        )
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.PERF_TEST] = parser
        return parser

    @staticmethod
    def _register_subcommand_score(subparsers):
        desc = """
        Score an input file using the given model. 
        """

        parser = subparsers.add_parser(
            ArgumentsOptions.SCORE, help="Run predictions in batch mode", description=desc
        )
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.SCORE] = parser
        return parser

    @staticmethod
    def _register_subcommand_fit(subparsers):
        parser = subparsers.add_parser(ArgumentsOptions.FIT, help="Fit your model to your data")
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.FIT] = parser
        return parser

    @staticmethod
    def _register_subcommand_validation(subparsers):
        desc = """
        You can validate the model on a set of various checks. 
        It is highly recommended to run these checks, as they are performed in DataRobot 
        before the model can be deployed.

        List of checks:

        * null values imputation: each feature of the provided dataset is set to missing 
          and fed to the model. 
          

        Example:
        > drum validation --code-dir ~/user_code_dir/ --input 10k.csv 
              --positive-class-label yes --negative-class-label no
        """

        parser = subparsers.add_parser(
            ArgumentsOptions.VALIDATION,
            help="Run validation checks against the model",
            description=desc,
            formatter_class=FlexiFormatter,
        )
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.VALIDATION] = parser
        return parser

    @staticmethod
    def _register_subcommand_server(subparsers):
        desc = """
        Serve the given model using REST API. A web server will be started and will use 
        the {address} argument for the host and port to use.
        
        The drum prediction server provides the following routes. 
        You may provide the environment variable URL_PREFIX. 
        Note that URLs must end with /.

        A GET URL_PREFIX/ route, which checks if the server is alive.
        Example: GET http://localhost:6789/

        A POST URL_PREFIX/shutdown/ route, which shuts the server down.
        Example: POST http://localhost:6789/shutdown/

        A POST URL_PREFIX/predict/ route, which returns predictions on data.
        Example: POST http://localhost:6789/predict/
        For this /predict/ route, provide inference data 
        (for the model to make predictions) as form data with a key:value pair, 
        where: key = X and value = filename of the CSV that contains the inference data
        
        Example using curl:
        curl -X POST --form "X=@data_file.csv" localhost:6789/predict/
     
        """
        parser = subparsers.add_parser(
            ArgumentsOptions.SERVER,
            help="serve the model via REST APIs",
            description=desc,
            formatter_class=FlexiFormatter,
        )
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.SERVER] = parser
        return parser

    @staticmethod
    def _register_subcommand_new(subparsers):
        parser = subparsers.add_parser(
            ArgumentsOptions.NEW,
            description="Create new model/env template",
            help="Create new model/env template",
        )
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.NEW] = parser
        return parser

    @staticmethod
    def _register_subcommand_new_model(subparsers):
        parser = subparsers.add_parser(
            ArgumentsOptions.NEW_MODEL, help="Create a new modeling code directory template"
        )
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.NEW_MODEL] = parser
        return parser

    @staticmethod
    def _register_subcommand_push(subparsers):
        parser = subparsers.add_parser(
            ArgumentsOptions.PUSH,
            help="Add your modeling code into DataRobot",
            description=PUSH_HELP_TEXT,
            formatter_class=FlexiFormatter,
        )
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.PUSH] = parser
        return parser

    @staticmethod
    def get_arg_parser():
        parser = argparse.ArgumentParser(description="Run user model")
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.MAIN_COMMAND] = parser
        CMRunnerArgsRegistry._reg_arg_version(parser)
        subparsers = parser.add_subparsers(
            dest=CMRunnerArgsRegistry.SUBPARSER_DEST_KEYWORD, help="Commands"
        )

        score_parser = CMRunnerArgsRegistry._register_subcommand_score(subparsers)
        fit_parser = CMRunnerArgsRegistry._register_subcommand_fit(subparsers)
        perf_test_parser = CMRunnerArgsRegistry._register_subcommand_perf_test(subparsers)
        validation_parser = CMRunnerArgsRegistry._register_subcommand_validation(subparsers)
        server_parser = CMRunnerArgsRegistry._register_subcommand_server(subparsers)
        new_parser = CMRunnerArgsRegistry._register_subcommand_new(subparsers)

        new_subparsers = new_parser.add_subparsers(
            dest=CMRunnerArgsRegistry.NEW_SUBPARSER_DEST_KEYWORD, help="Commands"
        )
        new_model_parser = CMRunnerArgsRegistry._register_subcommand_new_model(new_subparsers)

        push_parser = CMRunnerArgsRegistry._register_subcommand_push(subparsers)

        # Note following args are not supported for perf-test, thus set as default
        perf_test_parser.set_defaults(logging_level="warning", verbose=False)
        validation_parser.set_defaults(logging_level="warning", verbose=False)

        CMRunnerArgsRegistry._reg_arg_code_dir(
            score_parser,
            perf_test_parser,
            server_parser,
            fit_parser,
            new_model_parser,
            validation_parser,
            push_parser,
        )
        CMRunnerArgsRegistry._reg_arg_verbose(
            score_parser,
            server_parser,
            fit_parser,
            new_parser,
            new_model_parser,
            push_parser,
            perf_test_parser,
        )
        CMRunnerArgsRegistry._reg_arg_input(
            score_parser, perf_test_parser, fit_parser, validation_parser
        )
        CMRunnerArgsRegistry._reg_arg_pos_neg_labels(
            score_parser, perf_test_parser, server_parser, fit_parser, validation_parser
        )
        CMRunnerArgsRegistry._reg_arg_multiclass_labels(
            score_parser, perf_test_parser, server_parser, fit_parser, validation_parser
        )
        CMRunnerArgsRegistry._reg_arg_logging_level(
            score_parser, server_parser, fit_parser, new_parser, new_model_parser, push_parser
        )
        CMRunnerArgsRegistry._reg_arg_docker(
            score_parser,
            perf_test_parser,
            server_parser,
            fit_parser,
            validation_parser,
            push_parser,
        )
        CMRunnerArgsRegistry._reg_arg_memory(
            score_parser,
            perf_test_parser,
            server_parser,
            fit_parser,
            validation_parser,
            push_parser,
        )

        CMRunnerArgsRegistry._reg_arg_output(score_parser, fit_parser)
        CMRunnerArgsRegistry._reg_arg_show_perf(score_parser, server_parser)

        CMRunnerArgsRegistry._reg_arg_target_feature_and_filename(fit_parser)
        CMRunnerArgsRegistry._reg_arg_weights(fit_parser)
        CMRunnerArgsRegistry._reg_arg_skip_predict(fit_parser)
        CMRunnerArgsRegistry._reg_arg_num_rows(fit_parser)

        CMRunnerArgsRegistry._reg_arg_samples(perf_test_parser)
        CMRunnerArgsRegistry._reg_arg_iterations(perf_test_parser)
        CMRunnerArgsRegistry._reg_arg_timeout(perf_test_parser)
        CMRunnerArgsRegistry._reg_arg_in_server(perf_test_parser)
        CMRunnerArgsRegistry._reg_arg_url(perf_test_parser)

        CMRunnerArgsRegistry._reg_arg_address(server_parser)
        CMRunnerArgsRegistry._reg_arg_production_server(server_parser, perf_test_parser)
        CMRunnerArgsRegistry._reg_arg_max_workers(server_parser, perf_test_parser)
        CMRunnerArgsRegistry._reg_arg_with_error_server(server_parser)

        CMRunnerArgsRegistry._reg_arg_language(
            new_model_parser, server_parser, score_parser, perf_test_parser, validation_parser
        )

        CMRunnerArgsRegistry._reg_arg_show_stacktrace(
            score_parser,
            perf_test_parser,
            server_parser,
            fit_parser,
            validation_parser,
            new_model_parser,
        )

        CMRunnerArgsRegistry._reg_args_monitoring(score_parser, server_parser)

        CMRunnerArgsRegistry._reg_arg_target_type(
            score_parser, perf_test_parser, server_parser, fit_parser, validation_parser
        )

        CMRunnerArgsRegistry._reg_args_unstructured_mode(
            score_parser, perf_test_parser, server_parser, validation_parser
        )

        return parser

    @staticmethod
    def verify_monitoring_options(options, parser_name):
        if options.subparser_name in [ArgumentsOptions.SERVER, ArgumentsOptions.SCORE]:
            if options.monitor:
                if options.target_type == TargetType.UNSTRUCTURED.value:
                    print("Error: MLOps monitoring can not be used in unstructured mode.")
                    exit(1)
                missing_args = []
                if options.model_id is None:
                    missing_args.append(ArgumentsOptions.MODEL_ID)
                if options.deployment_id is None:
                    missing_args.append(ArgumentsOptions.DEPLOYMENT_ID)
                if options.monitor_settings is None:
                    missing_args.append(ArgumentsOptions.MONITOR_SETTINGS)

                if len(missing_args) > 0:
                    print("\n")
                    print("Error: MLOps Monitoring requires all monitoring options to be present.")
                    print("Note: The following MLOps monitoring option(s) is/are missing:")
                    for arg in missing_args:
                        print("  {}".format(arg))
                    print("\n")
                    print("These options can also be obtained via environment variables")
                    print("\n")
                    CMRunnerArgsRegistry._parsers[parser_name].print_help()
                    exit(1)
        # Monitor options are used to fill in pipeline json,
        # so define them for the modes different from score and server
        else:
            options.monitor = False
            options.model_id = None
            options.deployment_id = None
            options.monitor_settings = None

    @staticmethod
    def verify_options(options):
        if not options.subparser_name:
            CMRunnerArgsRegistry._parsers[ArgumentsOptions.MAIN_COMMAND].print_help()
            exit(1)
        elif options.subparser_name == ArgumentsOptions.NEW:
            if not options.new_mode:
                CMRunnerArgsRegistry._parsers[ArgumentsOptions.NEW].print_help()
                exit(1)
        elif options.subparser_name in [ArgumentsOptions.SERVER, ArgumentsOptions.PERF_TEST]:
            if options.production:
                if options.verbose:
                    print("Checking if uwsgi is installed...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", "uwsgi"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if result.returncode != 0:
                    print(
                        "Looks like 'uwsgi` package is missing. Don't use '{}' option when running drum server or try to install 'uwsgi'.".format(
                            ArgumentsOptions.PRODUCTION
                        )
                    )
                    print(result.stdout.decode("utf8"))
                    print(result.stderr.decode("utf8"))
                    exit(1)
                else:
                    if options.verbose:
                        print("uwsgi detected")

        CMRunnerArgsRegistry.verify_monitoring_options(options, options.subparser_name)
