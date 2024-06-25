import argparse
import logging
from mlpiper.pipeline.executor import Executor
from mlpiper.pipeline.components_desc import ComponentsDesc
from mlpiper.pipeline.component_language import ComponentLanguage


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "error": logging.ERROR,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executes PySpak pipelines of Python components"
    )

    parser.add_argument(
        "--run-locally",
        required=False,
        action="store_true",
        help="Run Spark locally with as many worker threads as logical cores on your machine.",
    )
    parser.add_argument("--logging-level", required=False, choices=list(LOG_LEVELS))

    subparsers = parser.add_subparsers()

    # PySpark pipeline execution
    parser_exec = subparsers.add_parser("exec", help="Execute a given PySpark pipeline")
    action = parser_exec.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "-p", "--pipeline", help="A json string, which represents a pipeline."
    )
    action.add_argument(
        "-f",
        "--pipeline-file",
        type=argparse.FileType("r"),
        help="A json file path, whose content is a pipeline.",
    )
    parser_exec.set_defaults(func=Executor.handle)

    # Extract components description that are inside this egg
    parser_desc = subparsers.add_parser(
        "desc", help="Save internal components description into a given path"
    )
    parser_desc.add_argument(
        "-c",
        "--comp-desc-out-path",
        required=False,
        help="An output full file path for components details",
    )
    parser_desc.set_defaults(func=ComponentsDesc.handle)

    # Get Python modules dependencies for the given pipeline
    deps = subparsers.add_parser(
        "deps",
        help="Return a list of module dependencies for a given pipeline, depending on"
        "the components programming language",
    )
    deps.add_argument(
        "lang",
        choices=[ComponentLanguage.PYTHON, ComponentLanguage.R],
        help="The programming language",
    )
    group = deps.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p", "--pipeline", help="A json string, which represents a pipeline."
    )
    group.add_argument(
        "-f",
        "--pipeline-file",
        type=argparse.FileType("r"),
        help="A json file path, whose content is a pipeline.",
    )
    deps.set_defaults(func=Executor.handle_deps)

    return parser.parse_args()


def set_logging_level(args):
    if args.logging_level:
        print(args.logging_level)
        logging.getLogger("mlpiper").setLevel(LOG_LEVELS[args.logging_level])


def main():
    FORMAT = "%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s"
    logging.basicConfig(format=FORMAT)

    args = parse_args()
    set_logging_level(args)
    args.func(args)


if __name__ == "__main__":
    main()
