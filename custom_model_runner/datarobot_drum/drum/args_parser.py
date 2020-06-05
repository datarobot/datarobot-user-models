import argparse
import os
import sys

from datarobot_drum.drum.description import version
from datarobot_drum.drum.common import LOG_LEVELS, ArgumentsOptions, RunLanguage


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
                "--version", action="version", version="%(prog)s {version}".format(version=version)
            )

    @staticmethod
    def _reg_arg_verbose(*parsers):
        for parser in parsers:
            parser.add_argument(
                "--verbose", action="store_true", default=False, help="Show verbose output"
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
                help_message = "Path to an output directory"
                type_callback = CMRunnerArgsRegistry._is_valid_dir
            else:
                raise ValueError(
                    "{} argument should be used only by score and fit parsers!".format(
                        ArgumentsOptions.OUTPUT
                    )
                )
            parser.add_argument(
                ArgumentsOptions.OUTPUT, default=None, type=type_callback, help=help_message,
            )

    @staticmethod
    def _reg_arg_target_feature_and_filename(*parsers):
        for parser in parsers:
            group = parser.add_mutually_exclusive_group(required=True)
            group.add_argument(
                ArgumentsOptions.TARGET,
                type=str,
                required=False,
                help="Which column to use as the target. Argument is mutually exclusive with {}".format(
                    ArgumentsOptions.TARGET_FILENAME
                ),
            )

            group.add_argument(
                ArgumentsOptions.TARGET_FILENAME,
                type=CMRunnerArgsRegistry._is_valid_file,
                required=False,
                help="A file containing the target values. Argument is mutually exclusive with {}".format(
                    ArgumentsOptions.TARGET
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
            parser.add_argument(
                ArgumentsOptions.POSITIVE_CLASS_LABEL,
                default=None,
                type=are_both_labels_present,
                help="Positive class label for a binary classification case",
            )
            parser.add_argument(
                ArgumentsOptions.NEGATIVE_CLASS_LABEL,
                default=None,
                type=are_both_labels_present,
                help="Negative class label for a binary classification case",
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
                help="Docker image to use to run {} in the {} mode".format(
                    ArgumentsOptions.MAIN_COMMAND, prog_name_lst[1]
                ),
            )

    @staticmethod
    def _reg_arg_threaded(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.THREADED,
                action="store_true",
                default=False,
                help="Run prediction server in threaded mode",
            )

    @staticmethod
    def _reg_arg_show_perf(*parsers):
        for parser in parsers:
            parser.add_argument(
                "--show-perf", action="store_true", default=False, help="Show performance stats"
            )

    @staticmethod
    def _reg_arg_in_perf_mode_internal(*parsers):
        for parser in parsers:
            parser.add_argument(
                "--in-perf-mode-internal",
                action="store_true",
                default=False,
                help="Provide indication that {} is called as part of a performance test run ".format(
                    ArgumentsOptions.MAIN_COMMAND
                ),
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
        langs = [e.value for e in RunLanguage]
        langs.remove(RunLanguage.JAVA.value)
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.LANGUAGE,
                choices=langs,
                default=None,
                required=True,
                help="Language to use for the new model/env template to create",
            )

    @staticmethod
    def _reg_arg_num_rows(*parsers):
        for parser in parsers:
            parser.add_argument(
                ArgumentsOptions.NUM_ROWS,
                default=100,
                help="Number of rows to use for testing the fit functionality. "
                "Set to ALL to use all rows. Default is 100",
            )

    @staticmethod
    def get_arg_parser():
        parser = argparse.ArgumentParser(description="Run user model")
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.MAIN_COMMAND] = parser
        subparsers = parser.add_subparsers(
            dest=CMRunnerArgsRegistry.SUBPARSER_DEST_KEYWORD, help="Commands"
        )
        batch_parser = subparsers.add_parser(
            ArgumentsOptions.SCORE, help="Run predictions in batch mode"
        )

        fit_parser = subparsers.add_parser(ArgumentsOptions.FIT, help="Fit your model to your data")

        parser_perf_test = subparsers.add_parser(
            ArgumentsOptions.PERF_TEST, help="Run performance tests"
        )

        validation_parser = subparsers.add_parser(
            ArgumentsOptions.VALIDATION, help="Run validation checks"
        )

        # Note following args are not supported for perf-test, thus set as default
        parser_perf_test.set_defaults(logging_level="warning", verbose=False)
        validation_parser.set_defaults(logging_level="warning", verbose=False)

        server_parser = subparsers.add_parser(
            ArgumentsOptions.SERVER, help="Run predictions in server"
        )

        new_parser = subparsers.add_parser(
            ArgumentsOptions.NEW,
            description="Create new model/env template",
            help="Create new model/env template",
        )

        new_subparsers = new_parser.add_subparsers(
            dest=CMRunnerArgsRegistry.NEW_SUBPARSER_DEST_KEYWORD, help="Commands"
        )
        CMRunnerArgsRegistry._parsers[ArgumentsOptions.NEW] = new_parser

        new_model_parser = new_subparsers.add_parser(
            ArgumentsOptions.NEW_MODEL, help="Create a new modeling code directory template"
        )

        CMRunnerArgsRegistry._reg_arg_version(parser)
        CMRunnerArgsRegistry._reg_arg_verbose(
            batch_parser, server_parser, fit_parser, new_parser, new_model_parser
        )
        CMRunnerArgsRegistry._reg_arg_input(
            batch_parser, parser_perf_test, fit_parser, validation_parser
        )
        CMRunnerArgsRegistry._reg_arg_output(batch_parser, fit_parser)
        CMRunnerArgsRegistry._reg_arg_target_feature_and_filename(fit_parser)
        CMRunnerArgsRegistry._reg_arg_weights(fit_parser)
        CMRunnerArgsRegistry._reg_arg_skip_predict(fit_parser)
        CMRunnerArgsRegistry._reg_arg_pos_neg_labels(
            batch_parser, parser_perf_test, server_parser, fit_parser, validation_parser
        )
        CMRunnerArgsRegistry._reg_arg_code_dir(
            batch_parser,
            parser_perf_test,
            server_parser,
            fit_parser,
            new_model_parser,
            validation_parser,
        )
        CMRunnerArgsRegistry._reg_arg_logging_level(
            batch_parser, server_parser, fit_parser, new_parser, new_model_parser
        )
        CMRunnerArgsRegistry._reg_arg_num_rows(fit_parser)
        CMRunnerArgsRegistry._reg_arg_docker(
            batch_parser, parser_perf_test, server_parser, fit_parser, validation_parser
        )
        CMRunnerArgsRegistry._reg_arg_show_perf(batch_parser, server_parser)

        CMRunnerArgsRegistry._reg_arg_samples(parser_perf_test)
        CMRunnerArgsRegistry._reg_arg_iterations(parser_perf_test)
        CMRunnerArgsRegistry._reg_arg_timeout(parser_perf_test)
        CMRunnerArgsRegistry._reg_arg_in_server(parser_perf_test)
        CMRunnerArgsRegistry._reg_arg_url(parser_perf_test)

        CMRunnerArgsRegistry._reg_arg_address(server_parser)
        CMRunnerArgsRegistry._reg_arg_threaded(server_parser)
        CMRunnerArgsRegistry._reg_arg_in_perf_mode_internal(server_parser)

        CMRunnerArgsRegistry._reg_arg_language(new_model_parser)

        return parser

    @staticmethod
    def verify_options(options):
        if not options.subparser_name:
            CMRunnerArgsRegistry._parsers[ArgumentsOptions.MAIN_COMMAND].print_help()
            exit(1)
        elif options.subparser_name == ArgumentsOptions.NEW:
            if not options.new_mode:
                CMRunnerArgsRegistry._parsers[ArgumentsOptions.NEW].print_help()
                exit(1)
