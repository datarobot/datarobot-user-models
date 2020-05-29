import json
import logging
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import pandas as pd
from mlpiper.pipeline.executor import Executor
from mlpiper.pipeline.executor_config import ExecutorConfig

from datarobot_cmrunner.cmrunner.common import (
    ArgumentsOptions,
    JavaArtifacts,
    LOG_LEVELS,
    LOGGER_NAME_PREFIX,
    PythonArtifacts,
    RArtifacts,
    RunLanguage,
    RunMode,
    TemplateType,
)
from datarobot_cmrunner.cmrunner.perf_testing import CMRunTests
from datarobot_cmrunner.cmrunner.templates_generator import CMTemplateGenerator
from datarobot_cmrunner.cmrunner.utils import CMRunnerUtils
from datarobot_cmrunner.profiler.stats_collector import StatsCollector, StatsOperation
from datarobot_cmrunner.cmrunner.exceptions import CMRunnerCommonException
from mlpiper.pipeline.executor import Executor
from mlpiper.pipeline.executor_config import ExecutorConfig

EXTERNAL_SERVER_RUNNER = "external_prediction_server_runner.json"


@contextmanager
def verbose_stdout(verbose):
    new_target = sys.stdout
    old_target = sys.stdout
    if not verbose:
        new_target = open(os.devnull, "w")
        sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


class CMRunner(object):
    def __init__(self, options):
        self.options = options
        self.logger = CMRunner._config_logger(options)
        self.verbose = options.verbose
        self.run_mode = RunMode(options.subparser_name)

        self._functional_pipelines = {
            (RunMode.SCORE, RunLanguage.PYTHON): "python_predictor.json.j2",
            (RunMode.SERVER, RunLanguage.PYTHON): "python_predictor_for_server.json.j2",
            (RunMode.SCORE, RunLanguage.R): "r_predictor.json.j2",
            (RunMode.SERVER, RunLanguage.R): "r_predictor_for_server.json.j2",
            (RunMode.SCORE, RunLanguage.JAVA): "java_predictor.json.j2",
            (RunMode.SERVER, RunLanguage.JAVA): "java_predictor_for_server.json.j2",
            (RunMode.FIT, RunLanguage.PYTHON): "python_fit.json.j2",
        }

    @staticmethod
    def _config_logger(options):
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)s:  %(message)s")
        logger = logging.getLogger(LOGGER_NAME_PREFIX)
        logger.setLevel(LOG_LEVELS[options.logging_level])
        return logger

    def get_logger(self):
        return self.logger

    def _print_verbose(self, message):
        if self.verbose:
            print(message)

    def _print_welcome_header(self):
        mode_headers = {
            RunMode.SERVER: "Detected REST server mode - this is an advanced option",
            RunMode.SCORE: "Detected score mode",
            RunMode.PERF_TEST: "Detected perf-test mode",
            RunMode.VALIDATION: "Detected validation check mode",
            RunMode.FIT: "Detected fit mode",
            RunMode.NEW: "Detected template generation mode",
        }
        self._print_verbose(mode_headers[self.run_mode])

    def _check_artifacts_and_get_run_language(self):
        # Get custom model's abs path and add it to the python path
        custom_model_abspath = os.path.abspath(self.options.code_dir)

        python_artifacts = CMRunnerUtils.find_files_by_extensions(
            custom_model_abspath, PythonArtifacts.ALL
        )
        r_artifacts = CMRunnerUtils.find_files_by_extensions(custom_model_abspath, RArtifacts.ALL)

        java_artifacts = CMRunnerUtils.find_files_by_extensions(
            custom_model_abspath, JavaArtifacts.ALL
        )

        if bool(len(python_artifacts)) + bool(len(r_artifacts)) + bool(
            len(java_artifacts)
        ) > 1 or not any([len(python_artifacts), len(r_artifacts), len(java_artifacts)]):
            error_mes = (
                "Custom model folder must have one or more model artifacts belonging to the same language:"
                "Python/R/Java, with an extension:\n"
                "Python models: {}\n"
                "R models: {}\n"
                "Java models: {}".format(PythonArtifacts.ALL, RArtifacts.ALL, JavaArtifacts.ALL)
            )
            self.logger.error(error_mes)
            exit(1)

        if len(python_artifacts):
            run_language = RunLanguage.PYTHON
        elif len(r_artifacts):
            run_language = RunLanguage.R
        elif len(java_artifacts):
            run_language = RunLanguage.JAVA
        else:
            raise NotImplementedError("We don't support your language")
        return run_language

    def run(self):
        try:
            if self.options.docker:
                ret = self._run_inside_docker(self.options, self.run_mode)
                if ret:
                    exit(1)
                else:
                    return
        except CMRunnerCommonException as e:
            self.logger.error(e)
            exit(1)
        except AttributeError:
            # In some parser the options.docker does not exists
            pass

        self._print_welcome_header()

        if self.run_mode in [RunMode.SERVER, RunMode.SCORE, RunMode.FIT]:
            self._run_fit_and_predictions_pipelines_in_mlpiper()
            if self.run_mode == RunMode.FIT and not self.options.skip_predict:
                self.run_test_predict()
            return
        elif self.run_mode == RunMode.PERF_TEST:
            CMRunTests(self.options, self.run_mode).performance_test()
            return
        elif self.run_mode == RunMode.VALIDATION:
            CMRunTests(self.options, self.run_mode).validation_test()
            return
        elif self.run_mode == RunMode.NEW:
            self._generate_template()
        else:
            print("{} mode is not implemented".format(self.run_mode))
            exit(1)

    def run_test_predict(self):
        self.run_mode = RunMode.SCORE
        self.options.code_dir = self.options.output
        self.options.output = "/dev/null"
        if self.options.target:
            __tempfile = NamedTemporaryFile()
            df = pd.read_csv(self.options.input)
            df = df.drop(self.options.target, axis=1)
            df.to_csv(__tempfile.name, index=False)
            self.options.input = __tempfile.name
        self._run_fit_and_predictions_pipelines_in_mlpiper()

    def _generate_template(self):
        CMTemplateGenerator(
            template_type=TemplateType.MODEL,
            language=RunLanguage(self.options.language),
            dir=self.options.code_dir,
        ).generate()

    def _prepare_prediction_server_or_batch_pipeline(self, run_language):
        options = self.options
        # functional pipeline is predictor pipeline
        # they are a little different for batch and server predictions.
        functional_pipeline_name = self._functional_pipelines[(self.run_mode, run_language)]
        functional_pipeline_filepath = CMRunnerUtils.get_pipeline_filepath(functional_pipeline_name)
        # fields to replace in the functional pipeline (predictor)
        replace_data = {
            "positiveClassLabel": '"{}"'.format(options.positive_class_label)
            if options.positive_class_label
            else "null",
            "negativeClassLabel": '"{}"'.format(options.negative_class_label)
            if options.negative_class_label
            else "null",
            "customModelPath": os.path.abspath(options.code_dir),
        }

        if self.run_mode == RunMode.SCORE:
            replace_data.update(
                {
                    "input_filename": options.input,
                    "output_filename": '"{}"'.format(options.output) if options.output else "null",
                }
            )

        functional_pipeline_str = CMRunnerUtils.render_file(
            functional_pipeline_filepath, replace_data
        )
        ret_pipeline = functional_pipeline_str

        if self.run_mode == RunMode.SERVER:
            with open(CMRunnerUtils.get_pipeline_filepath(EXTERNAL_SERVER_RUNNER), "r") as f:
                runner_pipeline_json = json.load(f)
                # can not use template for pipeline as quotes won't be escaped
                args = runner_pipeline_json["pipe"][0]["arguments"]
                # in server mode, predictor pipeline is passed to server as param
                args["pipeline"] = functional_pipeline_str
                args["repo"] = CMRunnerUtils.get_components_repo()
                host_port_list = options.address.split(":", 1)
                args["host"] = host_port_list[0]
                args["port"] = int(host_port_list[1]) if len(host_port_list) == 2 else None
                args["threaded"] = options.threaded
                args["show_perf"] = options.show_perf
                ret_pipeline = json.dumps(runner_pipeline_json)
        return ret_pipeline

    def _prepare_fit_pipeline(self, run_language):
        options = self.options
        # functional pipeline is predictor pipeline
        # they are a little different for batch and server predictions.
        functional_pipeline_name = self._functional_pipelines[(self.run_mode, run_language)]
        functional_pipeline_filepath = CMRunnerUtils.get_pipeline_filepath(functional_pipeline_name)
        # fields to replace in the functional pipeline (predictor)
        replace_data = {
            "customModelPath": os.path.abspath(options.code_dir),
            "input_filename": options.input,
            "weights": '"{}"'.format(options.row_weights) if options.row_weights else "null",
            "weights_filename": '"{}"'.format(options.row_weights_csv)
            if options.row_weights_csv
            else "null",
            "target_column": '"{}"'.format(options.target) if options.target else "null",
            "target_filename": '"{}"'.format(options.target_csv) if options.target_csv else "null",
            "positiveClassLabel": '"{}"'.format(options.positive_class_label)
            if options.positive_class_label
            else "null",
            "negativeClassLabel": '"{}"'.format(options.negative_class_label)
            if options.negative_class_label
            else "null",
            "output_dir": options.output,
            "num_rows": options.num_rows,
        }

        functional_pipeline_str = CMRunnerUtils.render_file(
            functional_pipeline_filepath, replace_data
        )
        return functional_pipeline_str

    def _run_fit_and_predictions_pipelines_in_mlpiper(self):
        if self.run_mode == RunMode.SERVER:
            run_language = self._check_artifacts_and_get_run_language()
            # in prediction server mode infra pipeline == prediction server runner pipeline
            infra_pipeline_str = self._prepare_prediction_server_or_batch_pipeline(run_language)
        elif self.run_mode == RunMode.SCORE:
            tmp_output_filename = None
            # if output is not provided, output into tmp file and print
            if not self.options.output:
                # keep object reference so it will be destroyed only in the end of the process
                __tmp_output_file = tempfile.NamedTemporaryFile(mode="w")
                self.options.output = tmp_output_filename = __tmp_output_file.name
            run_language = self._check_artifacts_and_get_run_language()
            # in batch prediction mode infra pipeline == predictor pipeline
            infra_pipeline_str = self._prepare_prediction_server_or_batch_pipeline(run_language)
        elif self.run_mode == RunMode.FIT:
            run_language = RunLanguage.PYTHON
            infra_pipeline_str = self._prepare_fit_pipeline(run_language)
        else:
            print("{} mode is not supported here".format(self.run_mode))
            exit(1)

        config = ExecutorConfig(
            pipeline=infra_pipeline_str,
            pipeline_file=None,
            run_locally=True,
            comp_root_path=CMRunnerUtils.get_components_repo(),
            mlpiper_jar=None,
            spark_jars=None,
        )

        _pipeline_executor = Executor(config).standalone(True)

        self.logger.info(
            ">>> Start {} in the {} mode".format(ArgumentsOptions.MAIN_COMMAND, self.run_mode.value)
        )
        sc = StatsCollector(
            disable_instance=(
                not hasattr(self.options, "show_perf")
                or not self.options.show_perf
                or self.run_mode == RunMode.SERVER
            )
        )
        sc.register_report("Full time", "end", StatsOperation.SUB, "start")
        sc.register_report("Init time (incl model loading)", "init", StatsOperation.SUB, "start")
        sc.register_report("Run time (incl reading CSV)", "run", StatsOperation.SUB, "init")
        with verbose_stdout(self.options.verbose):
            sc.enable()
            try:
                sc.mark("start")
                _pipeline_executor.init_pipeline()
                sc.mark("init")
                _pipeline_executor.run_pipeline(cleanup=False)
                sc.mark("run")
            except CMRunnerCommonException as e:
                self.logger.error(e)
                exit(1)
            finally:
                _pipeline_executor.cleanup_pipeline()
                sc.mark("end")
                sc.disable()
        self.logger.info(
            "<<< Finish {} in the {} mode".format(
                ArgumentsOptions.MAIN_COMMAND, self.run_mode.value
            )
        )
        sc.print_reports()
        if self.run_mode == RunMode.SCORE:
            # print result if output is not provided
            if tmp_output_filename:
                print(pd.read_csv(tmp_output_filename))

    def _prepare_docker_command(self, options, run_mode):
        """
        Building a docker command line for running the model inside the docker - this command line can
        be used by the user independently of cmrun.
        Parameters
        Returns: docker command line to run as a string
        """

        in_docker_model = "/opt/model"
        in_docker_input_file = "/opt/input.csv"
        in_docker_output_file = "/opt/output.csv"
        in_docker_fit_output_dir = "/opt/fit_output_dir"
        in_docker_fit_target_filename = "/opt/fit_target.csv"
        in_docker_fit_row_weights_filename = "/opt/fit_row_weights.csv"

        docker_cmd = "docker run --rm --interactive  --user $(id -u):$(id -g) "
        docker_cmd_args = " -v {}:{}".format(options.code_dir, in_docker_model)

        in_docker_cmd_list = sys.argv
        in_docker_cmd_list[0] = ArgumentsOptions.MAIN_COMMAND

        CMRunnerUtils.delete_cmd_argument(in_docker_cmd_list, ArgumentsOptions.DOCKER)
        CMRunnerUtils.replace_cmd_argument_value(
            in_docker_cmd_list, ArgumentsOptions.CODE_DIR, in_docker_model
        )
        CMRunnerUtils.replace_cmd_argument_value(in_docker_cmd_list, "-cd", in_docker_model)
        CMRunnerUtils.replace_cmd_argument_value(
            in_docker_cmd_list, ArgumentsOptions.INPUT, in_docker_input_file
        )
        CMRunnerUtils.replace_cmd_argument_value(
            in_docker_cmd_list, ArgumentsOptions.OUTPUT, in_docker_output_file
        )

        if run_mode == RunMode.SERVER:
            host_port_list = options.address.split(":", 1)
            if len(host_port_list) == 1:
                raise CMRunnerCommonException(
                    "Error: when using the docker option provide argument --server host:port"
                )
            port = int(host_port_list[1])
            host_port_inside_docker = "{}:{}".format("0.0.0.0", port)
            CMRunnerUtils.replace_cmd_argument_value(
                in_docker_cmd_list, ArgumentsOptions.ADDRESS, host_port_inside_docker
            )
            docker_cmd_args += " -p {port}:{port}".format(port=port)

        if run_mode in [RunMode.SCORE, RunMode.PERF_TEST, RunMode.VALIDATION, RunMode.FIT]:
            docker_cmd_args += " -v {}:{}".format(options.input, in_docker_input_file)

            if run_mode == RunMode.SCORE and options.output:
                output_file = os.path.realpath(options.output)
                if not os.path.exists(output_file):
                    # Creating an empty file so the mount command will mount the file correctly -
                    # otherwise docker create an empty directory
                    open(output_file, "a").close()
                docker_cmd_args += " -v {}:{}".format(output_file, in_docker_output_file)
                CMRunnerUtils.replace_cmd_argument_value(
                    in_docker_cmd_list, ArgumentsOptions.OUTPUT, in_docker_output_file
                )
            elif run_mode == RunMode.FIT:
                fit_output_dir = os.path.realpath(options.output)
                docker_cmd_args += " -v {}:{}".format(fit_output_dir, in_docker_fit_output_dir)
                CMRunnerUtils.replace_cmd_argument_value(
                    in_docker_cmd_list, ArgumentsOptions.OUTPUT, in_docker_fit_output_dir
                )
                if options.target_filename:
                    fit_target_filename = os.path.realpath(options.target_filename)
                    docker_cmd_args += " -v {}:{}".format(
                        fit_target_filename, in_docker_fit_target_filename
                    )
                    CMRunnerUtils.replace_cmd_argument_value(
                        in_docker_cmd_list,
                        ArgumentsOptions.TARGET_FILENAME,
                        in_docker_fit_target_filename,
                    )
                if options.row_weghts:
                    fit_row_weights_filename = os.path.realpath(options.row_weghts)
                    docker_cmd_args += " -v {}:{}".format(
                        fit_row_weights_filename, in_docker_fit_row_weights_filename
                    )
                    CMRunnerUtils.replace_cmd_argument_value(
                        in_docker_cmd_list,
                        ArgumentsOptions.WEIGHTS,
                        in_docker_fit_row_weights_filename,
                    )

        docker_cmd += " {} {} {}".format(
            docker_cmd_args, options.docker, " ".join(in_docker_cmd_list)
        )

        self._print_verbose("docker command: [{}]".format(docker_cmd))
        return docker_cmd

    def _run_inside_docker(self, options, run_mode):
        docker_cmd = self._prepare_docker_command(options, run_mode)
        self._print_verbose("-" * 20)
        p = subprocess.Popen(docker_cmd, shell=True)
        retcode = p.wait()
        self._print_verbose("{bar} retcode: {retcode} {bar}".format(bar="-" * 10, retcode=retcode))
        return retcode
