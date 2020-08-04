import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import copy
from distutils.dir_util import copy_tree
from pathlib import Path
from tempfile import mkdtemp, NamedTemporaryFile

import docker.errors
import numpy as np
import pandas as pd
from mlpiper.pipeline.executor import Executor
from mlpiper.pipeline.executor_config import ExecutorConfig

from datarobot_drum.drum.common import (
    ArgumentsOptions,
    CUSTOM_FILE_NAME,
    JavaArtifacts,
    LOG_LEVELS,
    LOGGER_NAME_PREFIX,
    PythonArtifacts,
    RArtifacts,
    RunLanguage,
    RunMode,
    TemplateType,
    verbose_stdout,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.perf_testing import CMRunTests
from datarobot_drum.drum.push import drum_push, setup_validation_options
from datarobot_drum.drum.templates_generator import CMTemplateGenerator
from datarobot_drum.drum.utils import CMRunnerUtils
from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation

PREDICTION_SERVER_PIPELINE = "prediction_server_pipeline.json.j2"
PREDICTOR_PIPELINE = "prediction_pipeline.json.j2"


class CMRunner(object):
    def __init__(self, runtime):
        self.runtime = runtime
        self.options = runtime.options
        self.logger = CMRunner._config_logger(runtime.options)
        self.verbose = runtime.options.verbose
        self.run_mode = RunMode(runtime.options.subparser_name)
        self.raw_arguments = sys.argv

        self._functional_pipelines = {
            (RunMode.FIT, RunLanguage.PYTHON): "python_fit.json.j2",
            (RunMode.FIT, RunLanguage.R): "r_fit.json.j2",
        }

    @staticmethod
    def _config_logger(options):
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
            RunMode.PUSH: "Detected push mode",
        }
        self._print_verbose(mode_headers[self.run_mode])

    def _check_artifacts_and_get_run_language(self):
        # Get code dir's abs path and add it to the python path
        code_dir_abspath = os.path.abspath(self.options.code_dir)

        artifact_language = None
        custom_language = None
        # check which artifacts present in the code dir
        python_artifacts = CMRunnerUtils.find_files_by_extensions(
            code_dir_abspath, PythonArtifacts.ALL
        )
        r_artifacts = CMRunnerUtils.find_files_by_extensions(code_dir_abspath, RArtifacts.ALL)

        java_artifacts = CMRunnerUtils.find_files_by_extensions(code_dir_abspath, JavaArtifacts.ALL)

        # check which custom code files present in the code dir
        is_custom_py = CMRunnerUtils.filename_exists_and_is_file(code_dir_abspath, "custom.py")
        is_custom_r = CMRunnerUtils.filename_exists_and_is_file(
            code_dir_abspath, "custom.R"
        ) or CMRunnerUtils.filename_exists_and_is_file(code_dir_abspath, "custom.r")

        # if all the artifacts belong to the same language, set it
        if bool(len(python_artifacts)) + bool(len(r_artifacts)) + bool(len(java_artifacts)) == 1:
            if len(python_artifacts):
                artifact_language = RunLanguage.PYTHON
            elif len(r_artifacts):
                artifact_language = RunLanguage.R
            elif len(java_artifacts):
                artifact_language = RunLanguage.JAVA

        # if only one custom file found, set it:
        if is_custom_py + is_custom_r == 1:
            custom_language = RunLanguage.PYTHON if is_custom_py else RunLanguage.R

        # if both language values are None, or both are not None and not equal
        if (
            bool(custom_language) + bool(artifact_language) == 0
            or bool(custom_language) + bool(artifact_language) == 2
            and custom_language != artifact_language
        ):
            artifact_language = "None" if artifact_language is None else artifact_language.value
            custom_language = "None" if custom_language is None else custom_language.value
            error_mes = (
                "Can not detect language by artifacts and/or custom.py/R files.\n"
                "Detected: language by artifacts - {}; language by custom - {}.\n"
                "Code directory must have one or more model artifacts belonging to the same language:\n"
                "Python/R/Java, with an extension:\n"
                "Python models: {}\n"
                "R models: {}\n"
                "Java models: {}.\n"
                "Or one of custom.py/R files.".format(
                    artifact_language,
                    custom_language,
                    PythonArtifacts.ALL,
                    RArtifacts.ALL,
                    JavaArtifacts.ALL,
                )
            )
            all_files_message = "\n\nFiles(100 first) found in {}:\n{}\n".format(
                code_dir_abspath, "\n".join(sorted(os.listdir(code_dir_abspath))[0:100])
            )

            error_mes += all_files_message
            self.logger.error(error_mes)
            raise DrumCommonException(error_mes)

        run_language = custom_language if custom_language is not None else artifact_language
        return run_language

    def _get_fit_run_language(self):
        def raise_no_language(custom_language):
            custom_language = "None" if custom_language is None else custom_language.value
            error_mes = (
                "Can not detect language by custom.py/R files.\n"
                "Detected: language by custom - {}.\n"
                "Code directory must have either a custom.py/R file\n"
                "Or a python file using the drum_autofit() wrapper.".format(custom_language,)
            )
            all_files_message = "\n\nFiles(100 first) found in {}:\n{}\n".format(
                code_dir_abspath, "\n".join(sorted(os.listdir(code_dir_abspath))[0:100])
            )

            error_mes += all_files_message
            self.logger.error(error_mes)
            raise DrumCommonException(error_mes)

        code_dir_abspath = os.path.abspath(self.options.code_dir)

        custom_language = None
        run_language = None
        is_py = False

        # check which custom code files present in the code dir
        custom_py_paths = list(Path(code_dir_abspath).rglob("{}.py".format(CUSTOM_FILE_NAME)))
        custom_r_paths = list(Path(code_dir_abspath).rglob("{}.r".format(CUSTOM_FILE_NAME))) + list(
            Path(code_dir_abspath).rglob("{}.R".format(CUSTOM_FILE_NAME))
        )

        # if only one custom file found, set it:
        if len(custom_py_paths) + len(custom_r_paths) == 1:
            custom_language = RunLanguage.PYTHON if custom_py_paths else RunLanguage.R

        # if no custom files, look for any other python file to use
        elif len(custom_py_paths) + len(custom_r_paths) == 0:

            other_py = list(Path(code_dir_abspath).rglob("*.py"))

            other_r = list(Path(code_dir_abspath).rglob("*.r")) + list(
                Path(code_dir_abspath).rglob("*.R")
            )

            # if we find any py files and no R files set python, otherwise raise
            if len(other_py) and not len(other_r):
                is_py = True
            else:
                raise_no_language(custom_language)

        # otherwise, we're in trouble
        else:
            raise_no_language(custom_language)

        if custom_language is not None:
            run_language = custom_language
        elif is_py:
            run_language = RunLanguage.PYTHON
        return run_language

    def run(self):
        try:
            if self.options.docker and not self.run_mode == RunMode.PUSH:
                ret = self._run_inside_docker(self.options, self.run_mode, self.raw_arguments)
                if ret:
                    raise DrumCommonException("Error from docker process: {}".format(ret))
                else:
                    return
        except DrumCommonException as e:
            self.logger.error(e)
            raise
        except AttributeError as e:
            # In some parser the options.docker does not exists
            if "docker" not in str(e):
                raise e

        self._print_welcome_header()

        if self.run_mode in [RunMode.SERVER, RunMode.SCORE]:
            self._run_fit_and_predictions_pipelines_in_mlpiper()
        elif self.run_mode == RunMode.FIT:
            self.run_fit()
        elif self.run_mode == RunMode.PERF_TEST:
            CMRunTests(self.options, self.run_mode).performance_test()
        elif self.run_mode == RunMode.VALIDATION:
            CMRunTests(self.options, self.run_mode).validation_test()
        elif self.run_mode == RunMode.NEW:
            self._generate_template()
        elif self.run_mode == RunMode.PUSH:
            options, run_mode, raw_arguments = setup_validation_options(copy.deepcopy(self.options))
            validation_runner = CMRunner(self.runtime)
            validation_runner.options = options
            validation_runner.run_mode = run_mode
            validation_runner.raw_arguments = raw_arguments
            validation_runner.run()
            print(
                "Your model was successfully validated locally! Now we will add it into DataRobot"
            )
            drum_push(self.options)
        else:
            error_message = "{} mode is not implemented".format(self.run_mode)
            print(error_message)
            raise DrumCommonException(error_message)

    def run_fit(self):
        remove_temp_output = None
        if not self.options.output:
            self.options.output = mkdtemp()
            remove_temp_output = self.options.output
        self._run_fit_and_predictions_pipelines_in_mlpiper()
        if self.options.output or not self.options.skip_predict:
            create_custom_inference_model_folder(self.options.code_dir, self.options.output)
        if not self.options.skip_predict:
            self.run_test_predict()
        if remove_temp_output:
            print(
                "Validation Complete 🎉 Your model can be fit to your data, "
                "and predictions can be made on the fit model! \n"
                "You're ready to add it to DataRobot. "
            )
            shutil.rmtree(remove_temp_output)
        else:
            print("Success 🎉")

    def run_test_predict(self):
        self.run_mode = RunMode.SCORE
        self.options.code_dir = self.options.output
        self.options.output = os.devnull
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
        functional_pipeline_name = (
            PREDICTION_SERVER_PIPELINE if self.run_mode == RunMode.SERVER else PREDICTOR_PIPELINE
        )
        functional_pipeline_filepath = CMRunnerUtils.get_pipeline_filepath(functional_pipeline_name)

        # fields to replace in the pipeline
        replace_data = {
            "positiveClassLabel": '"{}"'.format(options.positive_class_label)
            if options.positive_class_label
            else "null",
            "negativeClassLabel": '"{}"'.format(options.negative_class_label)
            if options.negative_class_label
            else "null",
            "customModelPath": os.path.abspath(options.code_dir),
            "run_language": run_language.value,
        }
        if self.run_mode == RunMode.SCORE:
            replace_data.update(
                {
                    "input_filename": options.input,
                    "output_filename": '"{}"'.format(options.output) if options.output else "null",
                }
            )
        else:
            host_port_list = options.address.split(":", 1)
            host = host_port_list[0]
            port = int(host_port_list[1]) if len(host_port_list) == 2 else None
            replace_data.update(
                {
                    "host": host,
                    "port": port,
                    "threaded": str(options.threaded).lower(),
                    "show_perf": str(options.show_perf).lower(),
                }
            )

        functional_pipeline_str = CMRunnerUtils.render_file(
            functional_pipeline_filepath, replace_data
        )

        return functional_pipeline_str

    def _prepare_fit_pipeline(self, run_language):
        if not self.options.negative_class_label:
            (
                self.options.positive_class_label,
                self.options.negative_class_label,
            ) = possibly_intuit_order(
                self.options.input, self.options.target_csv, self.options.target
            )
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
            run_language = self._check_artifacts_and_get_run_language()
            tmp_output_filename = None
            # if output is not provided, output into tmp file and print
            if not self.options.output:
                # keep object reference so it will be destroyed only in the end of the process
                __tmp_output_file = tempfile.NamedTemporaryFile(mode="w")
                self.options.output = tmp_output_filename = __tmp_output_file.name
            # in batch prediction mode infra pipeline == predictor pipeline
            infra_pipeline_str = self._prepare_prediction_server_or_batch_pipeline(run_language)
        elif self.run_mode == RunMode.FIT:
            run_language = self._get_fit_run_language()
            infra_pipeline_str = self._prepare_fit_pipeline(run_language)
        else:
            error_message = "{} mode is not supported here".format(self.run_mode)
            print(error_message)
            raise DrumCommonException(error_message)

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
                self.runtime.initialization_succeeded = True
                sc.mark("init")

                _pipeline_executor.run_pipeline(cleanup=False)
                sc.mark("run")
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

    def _prepare_docker_command(self, options, run_mode, raw_arguments):
        """
        Building a docker command line for running the model inside the docker - this command line can
        be used by the user independently of drum.
        Parameters
        Returns: docker command line to run as a string
        """
        options.docker = self._maybe_build_image(options.docker)
        in_docker_model = "/opt/model"
        in_docker_input_file = "/opt/input.csv"
        in_docker_output_file = "/opt/output.csv"
        in_docker_fit_output_dir = "/opt/fit_output_dir"
        in_docker_fit_target_filename = "/opt/fit_target.csv"
        in_docker_fit_row_weights_filename = "/opt/fit_row_weights.csv"

        docker_cmd = "docker run --rm --interactive  --user $(id -u):$(id -g) "
        docker_cmd_args = " -v {}:{}".format(options.code_dir, in_docker_model)

        in_docker_cmd_list = raw_arguments
        in_docker_cmd_list[0] = ArgumentsOptions.MAIN_COMMAND
        in_docker_cmd_list[1] = run_mode.value

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
                raise DrumCommonException(
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
                if options.output:
                    fit_output_dir = os.path.realpath(options.output)
                    docker_cmd_args += " -v {}:{}".format(fit_output_dir, in_docker_fit_output_dir)
                CMRunnerUtils.replace_cmd_argument_value(
                    in_docker_cmd_list, ArgumentsOptions.OUTPUT, in_docker_fit_output_dir
                )
                if options.target_csv:
                    fit_target_filename = os.path.realpath(options.target_csv)
                    docker_cmd_args += " -v {}:{}".format(
                        fit_target_filename, in_docker_fit_target_filename
                    )
                    CMRunnerUtils.replace_cmd_argument_value(
                        in_docker_cmd_list,
                        ArgumentsOptions.TARGET_FILENAME,
                        in_docker_fit_target_filename,
                    )
                if options.row_weights_csv:
                    fit_row_weights_filename = os.path.realpath(options.row_weights_csv)
                    docker_cmd_args += " -v {}:{}".format(
                        fit_row_weights_filename, in_docker_fit_row_weights_filename
                    )
                    CMRunnerUtils.replace_cmd_argument_value(
                        in_docker_cmd_list,
                        ArgumentsOptions.WEIGHTS_CSV,
                        in_docker_fit_row_weights_filename,
                    )

        docker_cmd += " {} {} {}".format(
            docker_cmd_args, options.docker, " ".join(in_docker_cmd_list)
        )

        self._print_verbose("docker command: [{}]".format(docker_cmd))
        return docker_cmd

    def _run_inside_docker(self, options, run_mode, raw_arguments):
        docker_cmd = self._prepare_docker_command(options, run_mode, raw_arguments)
        self._print_verbose("-" * 20)
        p = subprocess.Popen(docker_cmd, shell=True)
        retcode = p.wait()
        self._print_verbose("{bar} retcode: {retcode} {bar}".format(bar="-" * 10, retcode=retcode))
        return retcode

    def _maybe_build_image(self, docker_image_or_directory):
        ret_docker_image = None
        client = docker.client.from_env()

        if os.path.isdir(docker_image_or_directory):
            docker_image_or_directory = os.path.abspath(docker_image_or_directory)
            self.logger.info(
                "Building a docker image from directory {}...".format(docker_image_or_directory)
            )
            self.logger.info("This may take some time")
            try:
                image, _ = client.images.build(path=docker_image_or_directory)
                ret_docker_image = image.id
            except docker.errors.BuildError as e:
                self.logger.error("Hey something went wrong with image build!")
                for line in e.build_log:
                    if "stream" in line:
                        self.logger.error(line["stream"].strip())
                raise
            self.logger.info("Done building image!")
        else:
            try:
                client.images.get(docker_image_or_directory)
                ret_docker_image = docker_image_or_directory
            except docker.errors.ImageNotFound:
                pass

        if not ret_docker_image:
            raise DrumCommonException(
                "The string '{}' does not represent a docker image "
                "in your registry or a directory".format(docker_image_or_directory)
            )

        return ret_docker_image


def possibly_intuit_order(input_data_file, target_data_file=None, target_col_name=None):
    if target_data_file:
        assert target_col_name is None

        y = pd.read_csv(target_data_file, index_col=False).sample(
            1000, random_state=1, replace=True
        )
        classes = np.unique(y.iloc[:, 0])
    else:
        assert target_data_file is None
        df = pd.read_csv(input_data_file)
        if not target_col_name in df.columns:
            e = "The column '{}' does not exist in your dataframe. \nThe columns in your dataframe are these: {}".format(
                target_col_name, list(df.columns)
            )
            print(e, file=sys.stderr)
            raise DrumCommonException(e)
        classes = np.unique(df[target_col_name].sample(1000, random_state=1, replace=True))
    if len(classes) == 2:
        return classes
    elif len(classes) == 1:
        raise DrumCommonException("Only one target label was provided, please revise training data")
    return None, None


def create_custom_inference_model_folder(code_dir, output_dir):
    readme = """
    This folder was generated by the DRUM tool. It provides functionality for making 
    predictions using the model trained by DRUM
    """
    files_in_output = set(glob.glob(output_dir + "/**"))
    copied_files = set(copy_tree(code_dir, output_dir))
    shutil.rmtree(os.path.join(output_dir, "__pycache__"), ignore_errors=True)
    with open(os.path.join(output_dir, "README.md"), "w") as fp:
        fp.write(readme)
    if files_in_output & copied_files:
        print("Files were overwritten: {}".format(files_in_output & copied_files))
