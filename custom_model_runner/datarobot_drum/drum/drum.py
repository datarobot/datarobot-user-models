"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import copy
import glob
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from distutils.dir_util import copy_tree
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
from typing import Dict
from typing import Union


import docker.errors
import pandas as pd
from datarobot_drum.drum.adapters.cli.drum_fit_adapter import DrumFitAdapter
from datarobot_drum.drum.adapters.model_adapters.abstract_model_adapter import AbstractModelAdapter
from datarobot_drum.drum.adapters.model_adapters.r_model_adapter import RModelAdapter
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import get_metadata, FIT_METADATA_FILENAME
from datarobot_drum.drum.model_metadata import (
    read_model_metadata_yaml,
    read_default_model_metadata_yaml,
)

from datarobot_drum.drum.description import version as drum_version
from datarobot_drum.drum.enum import CUSTOM_FILE_NAME
from datarobot_drum.drum.enum import LOG_LEVELS
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.enum import ArgumentOptionsEnvVars
from datarobot_drum.drum.enum import ArgumentsOptions
from datarobot_drum.drum.enum import JavaArtifacts
from datarobot_drum.drum.enum import JuliaArtifacts
from datarobot_drum.drum.enum import ModelMetadataHyperParamTypes
from datarobot_drum.drum.enum import ModelMetadataKeys
from datarobot_drum.drum.enum import PythonArtifacts
from datarobot_drum.drum.enum import RArtifacts
from datarobot_drum.drum.enum import RunLanguage
from datarobot_drum.drum.enum import RunMode
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.enum import TemplateType
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.exceptions import DrumPredException
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.perf_testing import CMRunTests
from datarobot_drum.drum.push import drum_push
from datarobot_drum.drum.push import setup_validation_options
from datarobot_drum.drum.templates_generator import CMTemplateGenerator
from datarobot_drum.drum.typeschema_validation import SchemaValidator
from datarobot_drum.drum.utils.dataframe import is_sparse_dataframe
from datarobot_drum.drum.utils.drum_utils import DrumUtils
from datarobot_drum.drum.utils.drum_utils import handle_missing_colnames
from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils
from datarobot_drum.profiler.stats_collector import StatsCollector
from datarobot_drum.profiler.stats_collector import StatsOperation
from memory_profiler import memory_usage
from mlpiper.pipeline import json_fields
from mlpiper.pipeline.executor import Executor
from mlpiper.pipeline.executor_config import ExecutorConfig
from progress.spinner import Spinner
from scipy.io import mmwrite

SERVER_PIPELINE = "prediction_server_pipeline.json.j2"
PREDICTOR_PIPELINE = "prediction_pipeline.json.j2"


class CMRunner:
    def __init__(self, runtime):
        self.runtime = runtime
        self.options = runtime.options
        self.options.model_config = read_model_metadata_yaml(self.options.code_dir)
        self.options.default_parameter_values = (
            get_default_parameter_values(self.options.model_config)
            if self.options.model_config
            else {}
        )
        self.logger = CMRunner._config_logger(runtime.options)
        self.verbose = runtime.options.verbose
        self.run_mode = RunMode(runtime.options.subparser_name)
        self.raw_arguments = sys.argv
        self.target_type = None

        self._resolve_target_type()
        self._resolve_class_labels()

        self._functional_pipelines = {
            (RunMode.FIT, RunLanguage.PYTHON): "python_fit.json.j2",
            (RunMode.FIT, RunLanguage.R): "r_fit.json.j2",
        }

        # require metadata for push mode
        if self.run_mode == RunMode.PUSH:
            get_metadata(self.options)

        if self.run_mode in [RunMode.FIT, RunMode.PUSH]:
            # always populate the validator, even if info isn't provided. Use the default type schema if no
            # schema is provided, strict validation is enabled, and the target type is transform
            type_schema = {}
            use_default_type_schema = False
            strict_validation = not self.options.disable_strict_validation
            if self.options.model_config:
                type_schema = self.options.model_config.get("typeSchema", {})

            if not type_schema and strict_validation and self.target_type == TargetType.TRANSFORM:
                print(
                    "WARNING: No type schema provided. For transforms, we enforce using the default type schema to "
                    "ensure there are no conflicts with other tasks downstream. Disable strict validation if you do "
                    "not want to use the default type schema."
                )
                type_schema = read_default_model_metadata_yaml()

            self.schema_validator = SchemaValidator(
                type_schema=type_schema,
                strict=strict_validation,
                verbose=self.verbose,
            )
        self._input_df = None
        self._pipeline_executor = None

    @property
    def input_df(self):
        if self._input_df is None:
            # Lazy load df
            self._input_df = StructuredInputReadUtils.read_structured_input_file_as_df(
                self.options.input,
                self.options.sparse_column_file,
            )
        return self._input_df

    def _resolve_target_type(self):
        if self.run_mode == RunMode.NEW:
            return

        target_type_options = getattr(self.options, "target_type", None)
        target_type_options = (
            None if target_type_options is None else TargetType(target_type_options)
        )
        target_type_model_config = None

        if self.options.model_config is not None:
            target_type_model_config = TargetType(self.options.model_config["targetType"])

        if target_type_options is None and target_type_model_config is None:
            raise DrumCommonException(
                "Target type is missing. It must be provided in --target-type argument, {} env var or model config file.".format(
                    ArgumentOptionsEnvVars.TARGET_TYPE
                )
            )
        elif (
            all([target_type_options, target_type_model_config])
            and target_type_options != target_type_model_config
        ):
            raise DrumCommonException(
                "Target type provided in --target-type argument doesn't match target type from model config file. "
                "Use either one of them or make them match."
            )
        else:
            self.target_type = (
                target_type_options if target_type_options is not None else target_type_model_config
            )

        if self.target_type != TargetType.UNSTRUCTURED:
            if getattr(self.options, "query", None):
                raise DrumCommonException(
                    "--query argument can be used only with --target-type unstructured"
                )
            if getattr(self.options, "content_type", None):
                raise DrumCommonException(
                    "--content-type argument can be used only with --target-type unstructured"
                )
        else:
            if self.options.content_type is None:
                self.options.content_type = "text/plain; charset=utf8"

    def _resolve_class_labels(self):
        if self.run_mode in [RunMode.NEW] or (
            self.run_mode == RunMode.PUSH
            and self.options.model_config[ModelMetadataKeys.TYPE] == "training"
        ):
            self.options.positive_class_label = None
            self.options.negative_class_label = None
            self.options.class_labels = None
            self.options.class_labels_file = None
            return

        if self.target_type == TargetType.BINARY:
            pos_options = getattr(self.options, "positive_class_label", None)
            neg_options = getattr(self.options, "negative_class_label", None)

            try:
                pos_model_config = self.options.model_config.get(
                    ModelMetadataKeys.INFERENCE_MODEL
                ).get("positiveClassLabel")
                neg_model_config = self.options.model_config.get(
                    ModelMetadataKeys.INFERENCE_MODEL
                ).get("negativeClassLabel")
            except AttributeError:
                pos_model_config = neg_model_config = None

            if (
                not all([pos_options, neg_options])
                and not all([pos_model_config, neg_model_config])
                and self.run_mode != RunMode.FIT
            ):
                raise DrumCommonException(
                    "Positive/negative class labels are missing. They must be provided with either one: {}/{} arguments, environment variables, model config file.".format(
                        ArgumentsOptions.POSITIVE_CLASS_LABEL, ArgumentsOptions.NEGATIVE_CLASS_LABEL
                    )
                )
            elif all([pos_options, neg_options, pos_model_config, neg_model_config]) and (
                pos_options != pos_model_config or neg_options != neg_model_config
            ):
                raise DrumCommonException(
                    "Positive/negative class labels provided with command arguments or environment variable don't match values from model config file. "
                    "Use either one of them or make them match."
                )
            else:
                self.options.positive_class_label = (
                    pos_options if pos_options is not None else pos_model_config
                )

                self.options.negative_class_label = (
                    neg_options if neg_options is not None else neg_model_config
                )

        elif self.target_type == TargetType.MULTICLASS:
            labels_options = getattr(self.options, "class_labels", None)
            try:
                labels_model_config = self.options.model_config.get(
                    ModelMetadataKeys.INFERENCE_MODEL
                ).get("classLabels")
            except AttributeError:
                labels_model_config = None

            if (
                labels_options is None
                and labels_model_config is None
                and self.run_mode != RunMode.FIT
            ):
                raise DrumCommonException(
                    "Class labels are missing. They must be provided with either one: {}/{} arguments, environment variables, model config file.".format(
                        ArgumentsOptions.CLASS_LABELS, ArgumentsOptions.CLASS_LABELS_FILE
                    )
                )
            # both not None but not set() equal
            elif all([labels_options, labels_model_config]) and set(labels_options) != set(
                labels_model_config
            ):
                raise DrumCommonException(
                    "Class labels provided with command arguments or environment variable don't match values from model config file. "
                    "Use either one of them or make them match."
                )
            else:
                self.options.class_labels = (
                    labels_options if labels_options is not None else labels_model_config
                )
        else:
            self.options.positive_class_label = None
            self.options.negative_class_label = None
            self.options.class_labels = None
            self.options.class_labels_file = None

    @staticmethod
    def _config_logger(options):
        logging.getLogger().setLevel(LOG_LEVELS[options.logging_level])
        logging.getLogger("werkzeug").setLevel(LOG_LEVELS[options.logging_level])
        return logging.getLogger(LOGGER_NAME_PREFIX)

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
        lang = getattr(self.options, "language", None)
        if lang:
            return RunLanguage(self.options.language)

        gpu_predictor = getattr(self.options, "gpu_predictor", None)
        if gpu_predictor:
            return RunLanguage.OTHER

        code_dir_abspath = os.path.abspath(self.options.code_dir)

        artifact_language = None
        custom_language = None
        # check which artifacts present in the code dir
        python_artifacts = DrumUtils.find_files_by_extensions(code_dir_abspath, PythonArtifacts.ALL)
        r_artifacts = DrumUtils.find_files_by_extensions(code_dir_abspath, RArtifacts.ALL)

        java_artifacts = DrumUtils.find_files_by_extensions(code_dir_abspath, JavaArtifacts.ALL)

        julia_artifacts = DrumUtils.find_files_by_extensions(code_dir_abspath, JuliaArtifacts.ALL)
        # check which custom code files present in the code dir
        is_custom_py = DrumUtils.filename_exists_and_is_file(code_dir_abspath, "custom.py")
        is_custom_r = DrumUtils.filename_exists_and_is_file(
            code_dir_abspath, "custom.R", "custom.r"
        )
        is_custom_jl = DrumUtils.filename_exists_and_is_file(code_dir_abspath, "custom.jl")

        # if all the artifacts belong to the same language, set it
        if (
            bool(len(python_artifacts))
            + bool(len(r_artifacts))
            + bool(len(java_artifacts))
            + bool(len(julia_artifacts))
            == 1
        ):
            if len(python_artifacts) > 0:
                artifact_language = RunLanguage.PYTHON
            elif len(r_artifacts) > 0:
                artifact_language = RunLanguage.R
            elif len(java_artifacts) > 0:
                artifact_language = RunLanguage.JAVA
            elif len(julia_artifacts) > 0:
                artifact_language = RunLanguage.JULIA

        # if only one custom file found, set it:
        if is_custom_py + is_custom_r + is_custom_jl == 1:
            if is_custom_py:
                custom_language = RunLanguage.PYTHON
            elif is_custom_r:
                custom_language = RunLanguage.R
            else:
                custom_language = RunLanguage.JULIA

        # if both language values are None, or both are not None and not equal
        if (
            bool(custom_language) + bool(artifact_language) == 0
            or bool(custom_language) + bool(artifact_language) == 2
            and custom_language != artifact_language
        ):
            error_mes = (
                "Can not detect language by artifacts and/or custom.py/R files.\n"
                "Detected: language by artifacts - {}; language by custom - {}.\n"
                "Code directory must have one or more model artifacts belonging to the same language:\n"
                "Python/R/Java/Julia, with an extension:\n"
                "Python models: {}\n"
                "R models: {}\n"
                "Java models: {}.\n"
                "Julia models: {}.\n"
                "Or one of custom.py/R files.".format(
                    "None" if artifact_language is None else artifact_language.value,
                    "None" if custom_language is None else custom_language.value,
                    PythonArtifacts.ALL,
                    RArtifacts.ALL,
                    JavaArtifacts.ALL,
                    JuliaArtifacts.ALL,
                )
            )
            all_files_message = "\n\nFiles(100 first) found in {}:\n{}\n".format(
                code_dir_abspath, "\n".join(sorted(os.listdir(code_dir_abspath))[0:100])
            )

            error_mes += all_files_message
            self.logger.error(error_mes)
            raise DrumCommonException(error_mes)

        run_language = custom_language if custom_language is not None else artifact_language
        self.options.language = run_language.value
        return run_language

    def _get_fit_run_language(self):
        def raise_no_language(custom_language):
            custom_language = "None" if custom_language is None else custom_language.value
            error_mes = (
                "Can not detect language by custom.py/R/jl files.\n"
                "Detected: language by custom - {}.\n"
                "Code directory must have either a custom.py/R file".format(
                    custom_language,
                )
            )
            all_files_message = "\n\nFiles(100 first) found in {}:\n{}\n".format(
                code_dir_abspath, "\n".join(sorted(os.listdir(code_dir_abspath))[0:100])
            )

            error_mes += all_files_message
            self.logger.error(error_mes)
            raise DrumCommonException(error_mes)

        def raise_multiple_custom_files(py_paths, r_paths, jl_paths):
            files_found = py_paths + r_paths + jl_paths
            error_mes = (
                "Multiple custom.py/R/jl files were identified in the code directories sub directories.\n"
                "If using the output directory option select a directory that does not contain additional "
                "output directories or code directories.\n\n"
                "The following custom model files were found:\n"
            )
            error_mes += "\n".join([str(path) for path in files_found])
            self.logger.error(error_mes)
            raise DrumCommonException(error_mes)

        code_dir_abspath = os.path.abspath(self.options.code_dir)

        custom_language = None
        run_language = None
        is_py = False

        # check which custom code files present in the code dir
        custom_py_paths = list(Path(code_dir_abspath).rglob("{}.py".format(CUSTOM_FILE_NAME)))
        custom_r_paths = list(Path(code_dir_abspath).rglob("{}.[rR]".format(CUSTOM_FILE_NAME)))
        custom_jl_paths = list(Path(code_dir_abspath).rglob("{}.jl".format(CUSTOM_FILE_NAME)))

        # subdirectories also contain custom py/R files, likely an incorrectly selected output dir.
        if len(custom_py_paths) + len(custom_r_paths) + len(custom_jl_paths) > 1:
            raise_multiple_custom_files(custom_py_paths, custom_r_paths, custom_jl_paths)
        # if only one custom file found, set it:
        elif len(custom_py_paths) == 1:
            custom_language = RunLanguage.PYTHON
        elif len(custom_r_paths) == 1:
            custom_language = RunLanguage.R
        elif len(custom_jl_paths) == 1:
            custom_language = RunLanguage.Julia
        # if no custom files, look for any other python file to use
        elif len(custom_py_paths) + len(custom_r_paths) == 0:
            other_py = list(Path(code_dir_abspath).rglob("*.py"))

            other_r = list(Path(code_dir_abspath).rglob("*.r")) + list(
                Path(code_dir_abspath).rglob("*.R")
            )

            # if we find any py files and no R files set python, otherwise raise
            if len(other_py) > 0 and len(other_r) == 0:
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
            if self.options.docker and (
                self.run_mode not in (RunMode.PUSH, RunMode.PERF_TEST, RunMode.VALIDATION)
            ):
                ret = self._run_inside_docker(self.options, self.run_mode, self.raw_arguments)
                if ret:
                    raise DrumCommonException("Error from docker process: {}".format(ret))
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
            self._run_predictions_pipelines_in_mlpiper()
        elif self.run_mode == RunMode.FIT:
            self.run_fit()
        elif self.run_mode == RunMode.PERF_TEST:
            CMRunTests(self.options, self.run_mode).performance_test()
        elif self.run_mode == RunMode.VALIDATION:
            CMRunTests(self.options, self.run_mode, self.target_type).validation_test()
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

    def terminate(self):
        """It is being called upon shutdown in server mode"""
        if self._pipeline_executor:
            self._pipeline_executor.cleanup_pipeline()

    def _get_fit_function(self, cli_adapter: DrumFitAdapter) -> Callable:
        """
        Parameters
        ----------
        cli_adapter: DrumFitAdapter

        Returns
        -------
        Callable
            Function that will run the custom task's fit
        """
        # TODO: Decouple check_artifacts_and_get_run_language from CLI, add it as part of validate in DrumCLIAdapter
        run_language = self._check_artifacts_and_get_run_language()
        model_adapter: AbstractModelAdapter

        if run_language == RunLanguage.PYTHON:
            model_adapter = PythonModelAdapter(
                model_dir=cli_adapter.custom_task_folder_path, target_type=cli_adapter.target_type
            )
        elif run_language == RunLanguage.R:
            model_adapter = RModelAdapter(
                custom_task_folder_path=cli_adapter.custom_task_folder_path,
                target_type=cli_adapter.target_type,
            )
        else:
            raise ValueError("drum fit only supports Python and R")

        model_adapter.load_custom_hooks()

        def _run_fit():
            # TODO: Sampling should be baked into X, y, row_weights. We cannot because R samples within the R code
            possible_mount_path = getattr(self.options, "user_secrets_mount_path", None)
            possible_prefix = getattr(self.options, "user_secrets_prefix", None)
            model_adapter.fit(
                X=cli_adapter.sample_data_if_necessary(cli_adapter.X),
                y=cli_adapter.sample_data_if_necessary(cli_adapter.y),
                output_dir=cli_adapter.output_dir,
                row_weights=cli_adapter.sample_data_if_necessary(cli_adapter.weights),
                parameters=cli_adapter.parameters_for_fit,
                class_order=cli_adapter.class_ordering,
                user_secrets_mount_path=possible_mount_path,
                user_secrets_prefix=possible_prefix,
            )

        return _run_fit

    def run_fit(self):
        """Run when run_model is fit.

        Raises
        ------
        DrumCommonException
            Raised when the code directory is also used as the output directory.
        DrumPredException
            Raised when prediction fails.
        DrumSchemaValidationException
            Raised when model metadata validation fails.
        """
        cli_adapter = DrumFitAdapter(
            custom_task_folder_path=self.options.code_dir,
            input_filename=self.options.input,
            target_type=self.target_type,
            target_name=self.options.target,
            target_filename=self.options.target_csv,
            weights_name=self.options.row_weights,
            weights_filename=self.options.row_weights_csv,
            sparse_column_filename=self.options.sparse_column_file,
            positive_class_label=self.options.positive_class_label,
            negative_class_label=self.options.negative_class_label,
            class_labels=self.options.class_labels,
            parameters_file=self.options.parameter_file,
            default_parameter_values=self.options.default_parameter_values,
            output_dir=self.options.output,
            num_rows=self.options.num_rows,
        ).validate()

        # Validate schema target type and input data
        self.schema_validator.validate_type_schema(cli_adapter.target_type)
        self.schema_validator.validate_inputs(cli_adapter.X)

        fit_function = self._get_fit_function(cli_adapter=cli_adapter)

        print("Starting Fit")
        fit_mem_usage = memory_usage(
            fit_function,
            interval=1,
            max_usage=True,
            max_iterations=1,
        )
        print("Fit successful")

        if self.options.verbose:
            print("Maximum fit memory usage: {}MB".format(int(fit_mem_usage)))

        if cli_adapter.persist_output or not self.options.skip_predict:
            create_custom_inference_model_folder(
                code_dir=cli_adapter.custom_task_folder_path, output_dir=cli_adapter.output_dir
            )
        if not self.options.skip_predict:
            # TODO: Use cli_adapter within run_test_predict instead of setting self.options
            # This is assigning things that were computed in DrumFitAdapter, for compatability
            self.options.output = cli_adapter.output_dir
            self.options.positive_class_label = cli_adapter.positive_class_label
            self.options.negative_class_label = cli_adapter.negative_class_label
            self.options.class_labels = cli_adapter.class_labels

            print("Starting Prediction")
            mem_usage = memory_usage(
                self.run_test_predict,
                interval=1,
                max_usage=True,
                max_iterations=1,
            )
            if self.options.verbose:
                print("Maximum server memory usage: {}MB".format(int(mem_usage)))
            if self.options.enable_fit_metadata:
                self._generate_runtime_report_file(fit_mem_usage, mem_usage)
            pred_str = " and predictions can be made on the fit model! \n "
            print("Prediction successful for fit validation")
        else:
            pred_str = "however since you specified --skip-predict, predictions were not made \n"

        if cli_adapter.cleanup_output_directory_if_necessary():
            print(
                "Validation Complete 🎉 Your model can be fit to your data, {}"
                "You're ready to add it to DataRobot. ".format(pred_str)
            )
        else:
            print("Success 🎉")

    def run_test_predict(self):
        """
        Run after fit has completed. Performs various inference checks including prediction
        consistency and output data matching defined schema.

        Raises
        ------
        DrumCommonException
            Raised when the code directory is also used as the output directory.
        DrumPredException
            Raised when prediction fails.
        DrumSchemaValidationException
            Raised when model metadata validation fails.
        """
        self.options.code_dir = self.options.output
        self.options.output = os.devnull
        __target_temp = None

        if self.options.target or self.options.row_weights:
            df = self.input_df
            if self.target_type == TargetType.TRANSFORM and self.options.target:
                target_df = df[self.options.target]
                __target_temp = NamedTemporaryFile()
                target_df.to_csv(__target_temp.name, index=False, lineterminator="\r\n")

            if self.options.target:
                df = df.drop(self.options.target, axis=1)

            if self.options.row_weights:
                df = df.drop(self.options.row_weights, axis=1)

                # If weights was included in the sparse input training data, drop it from the column file
                # TODO: Always force weights + target to be separate files with sparse input
                if self.options.sparse_column_file:
                    sparse_colnames = StructuredInputReadUtils.read_sparse_column_file_as_list(
                        self.options.sparse_column_file
                    )
                    if self.options.row_weights in sparse_colnames:
                        sparse_colnames.remove(self.options.row_weights)

                        with open(self.options.sparse_column_file, "w") as file:
                            file.write("\n".join(sparse_colnames))

            # convert to R-friendly missing fields
            if self._get_fit_run_language() == RunLanguage.R:
                df = handle_missing_colnames(df)

            if is_sparse_dataframe(df):
                __tempfile = NamedTemporaryFile(suffix=".mtx")
                mmwrite(__tempfile.name, df.sparse.to_coo())
            else:
                __tempfile = NamedTemporaryFile(suffix=".csv")
                df.to_csv(__tempfile.name, index=False, lineterminator="\r\n")
            self.options.input = __tempfile.name

        try:
            if self.target_type == TargetType.TRANSFORM:
                CMRunTests(
                    self.options, self.target_type, self.schema_validator
                ).check_transform_server(__target_temp)
            else:
                CMRunTests(
                    self.options, self.target_type, self.schema_validator
                ).check_prediction_side_effects()
        except DrumPredException as e:
            self.logger.warning(e)

    def _generate_template(self):
        CMTemplateGenerator(
            template_type=TemplateType.MODEL,
            language=RunLanguage(self.options.language),
            dir=self.options.code_dir,
        ).generate()

    def _prepare_prediction_server_or_batch_pipeline(self, run_language):
        options = self.options
        functional_pipeline_name = (
            SERVER_PIPELINE if self.run_mode == RunMode.SERVER else PREDICTOR_PIPELINE
        )
        functional_pipeline_filepath = DrumUtils.get_pipeline_filepath(functional_pipeline_name)
        # fields to replace in the pipeline
        replace_data = {
            "positiveClassLabel": options.positive_class_label,
            "negativeClassLabel": options.negative_class_label,
            "classLabels": options.class_labels,
            "customModelPath": os.path.abspath(options.code_dir),
            "run_language": run_language.value,
            "monitor": options.monitor,
            "monitor_embedded": options.monitor_embedded,
            "model_id": options.model_id,
            "deployment_id": options.deployment_id,
            "monitor_settings": options.monitor_settings,
            "external_webserver_url": options.webserver,
            "gpu_predictor": options.gpu_predictor,
            "triton_host": options.triton_host,
            "triton_http_port": int(options.triton_http_port),
            "triton_grpc_port": int(options.triton_grpc_port),
            "api_token": options.api_token,
            "allow_dr_api_access": options.allow_dr_api_access,
            "query_params": '"{}"'.format(options.query)
            if getattr(options, "query", None) is not None
            else "null",
            "content_type": '"{}"'.format(options.content_type)
            if getattr(options, "content_type", None) is not None
            else "null",
            "target_type": self.target_type.value,
            "user_secrets_mount_path": getattr(options, "user_secrets_mount_path", None),
            "user_secrets_prefix": getattr(options, "user_secrets_prefix", None),
        }

        if self.run_mode == RunMode.SCORE:
            replace_data.update(
                {
                    "input_filename": options.input,
                    "output_filename": '"{}"'.format(options.output) if options.output else "null",
                    "sparse_column_file": options.sparse_column_file,
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
                    "show_perf": str(options.show_perf).lower(),
                    "engine_type": "RestModelServing" if options.production else "Generic",
                    "component_type": "uwsgi_serving"
                    if options.production
                    else "prediction_server",
                    "processes": options.max_workers if getattr(options, "max_workers") else "null",
                    "uwsgi_max_workers": options.max_workers
                    if getattr(options, "max_workers") and options.production
                    else "null",
                    "single_uwsgi_worker": (options.max_workers == 1) and options.production,
                    "deployment_config": '"{}"'.format(options.deployment_config)
                    if getattr(options, "deployment_config", None) is not None
                    else "null",
                }
            )

        functional_pipeline_str = DrumUtils.render_file(functional_pipeline_filepath, replace_data)

        if self.run_mode == RunMode.SERVER:
            if options.production:
                pipeline_json = json.loads(functional_pipeline_str)
                # Because of tech debt in MLPiper which requires that the modelFileSourcePath key
                # be filled with something, we're putting in a dummy file path here
                if json_fields.PIPELINE_SYSTEM_CONFIG_FIELD not in pipeline_json:
                    system_config = {"modelFileSourcePath": os.path.abspath(__file__)}
                pipeline_json[json_fields.PIPELINE_SYSTEM_CONFIG_FIELD] = system_config
                functional_pipeline_str = json.dumps(pipeline_json)
        return functional_pipeline_str

    def _run_predictions_pipelines_in_mlpiper(self):
        run_language = self._check_artifacts_and_get_run_language()

        if self.run_mode == RunMode.SERVER:
            # in prediction server mode infra pipeline == prediction server runner pipeline
            infra_pipeline_str = self._prepare_prediction_server_or_batch_pipeline(run_language)
        elif self.run_mode == RunMode.SCORE:
            tmp_output_filename = None
            # if output is not provided, output into tmp file and print
            if not self.options.output:
                # keep object reference so it will be destroyed only in the end of the process
                __tmp_output_file = tempfile.NamedTemporaryFile(mode="w")
                self.options.output = tmp_output_filename = __tmp_output_file.name
            # in batch prediction mode infra pipeline == predictor pipeline
            infra_pipeline_str = self._prepare_prediction_server_or_batch_pipeline(run_language)
        else:
            error_message = "{} mode is not supported here".format(self.run_mode)
            print(error_message)
            raise DrumCommonException(error_message)

        config = ExecutorConfig(
            pipeline=infra_pipeline_str,
            pipeline_file=None,
            run_locally=True,
            comp_root_path=DrumUtils.get_components_repo(),
            mlpiper_jar=None,
            spark_jars=None,
        )

        self._pipeline_executor = (
            Executor(config).standalone(True).set_verbose(self.options.verbose)
        )
        # assign logger with the name drum.mlpiper.Executor to mlpiper Executor
        self._pipeline_executor.set_logger(
            logging.getLogger(LOGGER_NAME_PREFIX + "." + self._pipeline_executor.logger_name())
        )

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
        sc.enable()
        try:
            sc.mark("start")

            self._pipeline_executor.init_pipeline()
            self.runtime.initialization_succeeded = True
            sc.mark("init")

            self._pipeline_executor.run_pipeline(cleanup=False)
            sc.mark("run")
        finally:
            self._pipeline_executor.cleanup_pipeline()
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
                if self.target_type == TargetType.UNSTRUCTURED:
                    with open(tmp_output_filename) as f:
                        print(f.read())
                else:
                    print(pd.read_csv(tmp_output_filename))

    def _prepare_docker_command(self, options, run_mode, raw_arguments):
        """
        Building a docker command line for running the model inside the docker - this command line
        can be used by the user independently of drum.
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
        in_docker_runtime_parameters_file = "/opt/runtime_parameters.yaml"

        docker_cmd = "docker run --rm --init --entrypoint '' --interactive --user {}:{}".format(
            os.getuid(), os.getgid()
        )
        docker_cmd_args = ' -v "{}":{}'.format(options.code_dir, in_docker_model)

        in_docker_cmd_list = raw_arguments
        in_docker_cmd_list[0] = ArgumentsOptions.MAIN_COMMAND
        in_docker_cmd_list[1] = run_mode.value

        # [RAPTOR-5607] Using -cd makes fit fail within docker, but not --code-dir.
        # Hotfix it by replacing -cd with --code-dir
        in_docker_cmd_list = [
            ArgumentsOptions.CODE_DIR if arg == "-cd" else arg for arg in in_docker_cmd_list
        ]

        DrumUtils.delete_cmd_argument(in_docker_cmd_list, ArgumentsOptions.DOCKER)
        DrumUtils.delete_cmd_argument(in_docker_cmd_list, ArgumentsOptions.SKIP_DEPS_INSTALL)
        if options.memory:
            docker_cmd_args += " --memory {mem_size} --memory-swap {mem_size} ".format(
                mem_size=options.memory
            )
            DrumUtils.delete_cmd_argument(in_docker_cmd_list, ArgumentsOptions.MEMORY)

        if options.class_labels and ArgumentsOptions.CLASS_LABELS not in in_docker_cmd_list:
            DrumUtils.delete_cmd_argument(in_docker_cmd_list, ArgumentsOptions.CLASS_LABELS_FILE)
            in_docker_cmd_list.append(ArgumentsOptions.CLASS_LABELS)
            for label in options.class_labels:
                in_docker_cmd_list.append(label)

        DrumUtils.replace_cmd_argument_value(
            in_docker_cmd_list, ArgumentsOptions.CODE_DIR, in_docker_model
        )
        DrumUtils.replace_cmd_argument_value(in_docker_cmd_list, "-cd", in_docker_model)
        DrumUtils.replace_cmd_argument_value(
            in_docker_cmd_list, ArgumentsOptions.INPUT, in_docker_input_file
        )
        DrumUtils.replace_cmd_argument_value(
            in_docker_cmd_list, ArgumentsOptions.OUTPUT, in_docker_output_file
        )

        if run_mode == RunMode.SERVER:
            host_port_list = options.address.split(":", 1)
            if len(host_port_list) == 1:
                raise DrumCommonException(
                    "Error: when using the docker option provide argument --address host:port"
                )
            port = int(host_port_list[1])
            host_port_inside_docker = "{}:{}".format("0.0.0.0", port)
            DrumUtils.replace_cmd_argument_value(
                in_docker_cmd_list, ArgumentsOptions.ADDRESS, host_port_inside_docker
            )
            docker_cmd_args += " -p {port}:{port}".format(port=port)

        if CMRunnerArgsRegistry.get_arg_option(options, ArgumentsOptions.RUNTIME_PARAMS_FILE):
            docker_cmd_args += (
                f' -v "{options.runtime_params_file}":{in_docker_runtime_parameters_file}'
            )
            DrumUtils.delete_cmd_argument(in_docker_cmd_list, ArgumentsOptions.RUNTIME_PARAMS_FILE)
            in_docker_cmd_list.extend(
                [ArgumentsOptions.RUNTIME_PARAMS_FILE, in_docker_runtime_parameters_file]
            )

        if run_mode in [RunMode.SCORE, RunMode.PERF_TEST, RunMode.VALIDATION, RunMode.FIT]:
            docker_cmd_args += ' -v "{}":{}'.format(options.input, in_docker_input_file)

            if run_mode == RunMode.SCORE and options.output:
                output_file = os.path.realpath(options.output)
                if not os.path.exists(output_file):
                    # Creating an empty file so the mount command will mount the file correctly -
                    # otherwise docker create an empty directory
                    open(output_file, "a").close()
                docker_cmd_args += ' -v "{}":{}'.format(output_file, in_docker_output_file)
                DrumUtils.replace_cmd_argument_value(
                    in_docker_cmd_list, ArgumentsOptions.OUTPUT, in_docker_output_file
                )
            elif run_mode == RunMode.FIT:
                if options.output:
                    fit_output_dir = os.path.realpath(options.output)
                    docker_cmd_args += ' -v "{}":{}'.format(
                        fit_output_dir, in_docker_fit_output_dir
                    )
                DrumUtils.replace_cmd_argument_value(
                    in_docker_cmd_list, ArgumentsOptions.OUTPUT, in_docker_fit_output_dir
                )
                if options.target_csv:
                    fit_target_filename = os.path.realpath(options.target_csv)
                    docker_cmd_args += ' -v "{}":{}'.format(
                        fit_target_filename, in_docker_fit_target_filename
                    )
                    DrumUtils.replace_cmd_argument_value(
                        in_docker_cmd_list,
                        ArgumentsOptions.TARGET_CSV,
                        in_docker_fit_target_filename,
                    )
                if options.target and ArgumentsOptions.TARGET in in_docker_cmd_list:
                    DrumUtils.replace_cmd_argument_value(
                        in_docker_cmd_list, ArgumentsOptions.TARGET, f"{options.target}"
                    )
                if options.row_weights_csv:
                    fit_row_weights_filename = os.path.realpath(options.row_weights_csv)
                    docker_cmd_args += ' -v "{}":{}'.format(
                        fit_row_weights_filename, in_docker_fit_row_weights_filename
                    )
                    DrumUtils.replace_cmd_argument_value(
                        in_docker_cmd_list,
                        ArgumentsOptions.WEIGHTS_CSV,
                        in_docker_fit_row_weights_filename,
                    )

        docker_cmd += " {} {}".format(docker_cmd_args, options.docker)
        docker_cmd = shlex.split(docker_cmd)
        docker_cmd += in_docker_cmd_list

        self._print_verbose("docker command: <{}>".format(docker_cmd))
        return docker_cmd

    def _run_inside_docker(self, options, run_mode, raw_arguments):
        self._check_artifacts_and_get_run_language()
        docker_cmd_lst = self._prepare_docker_command(options, run_mode, raw_arguments)

        self._print_verbose("Checking DRUM version in container...")
        result = subprocess.run(
            [
                "docker",
                "run",
                "-it",
                "--entrypoint",
                # provide empty entrypoint value to unset the one that could be set within the image
                "",
                options.docker,
                "sh",
                "-c",
                "drum --version",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        container_drum_version = result.stdout.decode("utf8")
        # remove double spaces and \n\r
        container_drum_version = " ".join(container_drum_version.split())

        host_drum_version = "{} {}".format(ArgumentsOptions.MAIN_COMMAND, drum_version)
        if container_drum_version != host_drum_version:
            print(
                "WARNING: looks like host DRUM version doesn't match container DRUM version. This can lead to unexpected behavior.\n"
                "Host DRUM version: {}\n"
                "Container DRUM version: {}".format(host_drum_version, result.stdout.decode("utf8"))
            )
            err = result.stderr.decode("utf8")
            if len(err):
                print(err)
            time.sleep(0.5)
        else:
            self._print_verbose(
                "Host DRUM version matches container DRUM version: {}".format(host_drum_version)
            )
        self._print_verbose("-" * 20)
        p = subprocess.Popen(docker_cmd_lst)
        try:
            retcode = p.wait()
        except KeyboardInterrupt:
            p.terminate()
            retcode = 0

        self._print_verbose("{bar} retcode: {retcode} {bar}".format(bar="-" * 10, retcode=retcode))
        return retcode

    def _maybe_build_image(self, docker_image_or_directory):
        def _get_requirements_lines(reqs_file_path):
            if not os.path.exists(reqs_file_path):
                return None

            with open(reqs_file_path) as f:
                lines = f.readlines()
                lines = ["'{}'".format(l.strip()) for l in lines if l.strip() != ""]
            return lines

        ret_docker_image = None
        if os.path.isdir(docker_image_or_directory):
            docker_image_or_directory = os.path.abspath(docker_image_or_directory)
            # Set image tag to the dirname/dirname of the docker context.
            # E.g. for two folders:
            # /home/path1/my_env
            # /home/path2/my_env
            # tags will be 'path1/my_env', 'path2/my_env'
            #
            # If tag already exists, older image will be untagged.
            context_path = os.path.abspath(docker_image_or_directory)
            tag = "{}/{}".format(
                os.path.basename(os.path.dirname(context_path)), os.path.basename(context_path)
            ).lower()

            lines = _get_requirements_lines(os.path.join(self.options.code_dir, "requirements.txt"))
            temp_context_dir = None
            if lines is not None and not self.options.skip_deps_install:
                temp_context_dir = tempfile.mkdtemp()
                shutil.rmtree(temp_context_dir)
                shutil.copytree(docker_image_or_directory, temp_context_dir)
                msg = (
                    "Requirements file has been found in the code dir. DRUM will try to install dependencies into a docker image.\n"
                    "Docker context has been copied from: {} to: {}".format(
                        docker_image_or_directory, temp_context_dir
                    )
                )

                print(msg)
                self.logger.debug(msg)
                docker_image_or_directory = temp_context_dir

                with open(os.path.join(temp_context_dir, "Dockerfile"), mode="a") as f:
                    if self.options.language == RunLanguage.PYTHON.value:
                        f.write("\nRUN pip3 install {}".format(" ".join(lines)))
                    elif self.options.language == RunLanguage.R.value:
                        deps_str = ", ".join(lines)
                        l1 = "\nRUN echo \"r <- getOption('repos'); r['CRAN'] <- 'http://cran.rstudio.com/'; options(repos = r);\" > ~/.Rprofile"
                        l2 = '\nRUN Rscript -e "withCallingHandlers(install.packages(c({}), Ncpus=4), warning = function(w) stop(w))"'.format(
                            deps_str
                        )
                        f.write(l1)
                        f.write(l2)
                    else:
                        msg = "Dependencies management is not supported for the '{}' language and will not be installed into an image".format(
                            self.options.language
                        )
                        self.logger.warning(msg)
                        print(msg)

            docker_build_msg = "Building a docker image from directory: {}...".format(
                docker_image_or_directory
            )
            self.logger.info(docker_build_msg)
            self.logger.info("This may take some time")

            try:
                client_docker_low_level = docker.APIClient()
                spinner = Spinner(docker_build_msg + "  ")
                json_lines = []
                # Build docker, rotate spinner according to build progress
                # and save status messages from docker build.
                for line in client_docker_low_level.build(
                    path=docker_image_or_directory, rm=True, tag=tag
                ):
                    line = line.decode("utf-8").strip()
                    json_lines.extend([json.loads(ll) for ll in line.split("\n")])
                    spinner.next()
                spinner.finish()
                # skip a line after spinner
                print()

                image_id = None
                build_error = False
                for line in json_lines:
                    if "error" in line:
                        build_error = True
                        break
                    if "stream" in line:
                        match = re.search(
                            r"(^Successfully built |sha256:)([0-9a-f]+)$", line["stream"]
                        )
                        if match:
                            image_id = match.group(2)
                if image_id is None or build_error:
                    all_lines = "   \n".join([json.dumps(l) for l in json_lines])
                    raise DrumCommonException(
                        "Failed to build a docker image:\n{}".format(all_lines)
                    )

                print("\nImage successfully built; tag: {}; image id: {}".format(tag, image_id))
                print(
                    "It is recommended to use --docker {}, if you don't need to rebuild the image.\n".format(
                        tag
                    )
                )

                ret_docker_image = image_id
            except docker.errors.APIError as e:
                self.logger.exception("Image build failed because of unknown to DRUM reason!")
                raise
            finally:
                if temp_context_dir is not None:
                    shutil.rmtree(temp_context_dir)
            self.logger.info("Done building image!")
        else:
            try:
                client = docker.client.from_env()
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

    def _generate_runtime_report_file(self, fit_mem_usage: float, pred_mem_usage: float) -> None:
        """
        Saves information related to running a fit pipeline.  All data is reported in Mb

        Parameters:
            fit_mem_usage: Memory footprint of running the fit job
            pred_mem_usage: Memory footprint of running the check for prediction side effects
        """
        print(self.options.input, self.options)
        report_information = {
            "fit_memory_usage": fit_mem_usage,
            "prediction_memory_usage": pred_mem_usage,
            "input_dataframe_size": self.input_df.memory_usage(deep=True).sum() / 1e6,
        }
        # in run_predict the code dir is set to the output and output is set to /dev/null
        output_path = Path(self.options.code_dir) / FIT_METADATA_FILENAME
        json.dump(report_information, open(output_path, "w"))


def output_in_code_dir(code_dir, output_dir):
    """Does the code directory house the output directory?"""
    code_abs_path = os.path.abspath(code_dir)
    output_abs_path = os.path.abspath(output_dir)
    return os.path.commonpath([code_dir, output_abs_path]) == code_abs_path


def create_custom_inference_model_folder(code_dir, output_dir):
    readme = """
    This folder was generated by the DRUM tool. It provides functionality for making
    predictions using the model trained by DRUM
    """
    files_in_output = set(glob.glob(output_dir + "/**"))
    if output_in_code_dir(code_dir, output_dir):
        # since the output directory is in the code directory use a tempdir to copy into first and
        # cleanup files and prevent errors related to copying the output into itself.
        with tempfile.TemporaryDirectory() as tempdir:
            copy_tree(code_dir, tempdir)
            # remove the temporary version of the target dir
            shutil.rmtree(os.path.join(tempdir, os.path.relpath(output_dir, code_dir)))
            shutil.rmtree(os.path.join(tempdir, "__pycache__"), ignore_errors=True)
            copied_files = set(copy_tree(tempdir, output_dir))
    else:
        copied_files = set(copy_tree(code_dir, output_dir))
        shutil.rmtree(os.path.join(output_dir, "__pycache__"), ignore_errors=True)
    with open(os.path.join(output_dir, "README.md"), "w") as fp:
        fp.write(readme)
    if files_in_output & copied_files:
        print("Files were overwritten: {}".format(files_in_output & copied_files))


def _get_default_numeric_param_value(param_config: Dict, cast_to_int: bool) -> Union[int, float]:
    """Get default value of numeric parameter."""
    param_default_value = param_config.get("default")
    if param_default_value is None:
        param_default_value = param_config["min"]
    return int(param_default_value) if cast_to_int else float(param_default_value)


def _get_default_string_param_value(param_config: Dict) -> str:
    """Get default value of string parameter."""
    param_default_value = param_config.get("default")
    if param_default_value is not None:
        return param_default_value
    return ""


def _get_default_select_param_value(param_config: Dict) -> str:
    """Get default value of select parameter."""
    param_default_value = param_config.get("default")
    if param_default_value is not None:
        return param_default_value
    return param_config["values"][0]


def _get_default_multi_param_value(param_config: Dict) -> Union[int, float, str]:
    """Get default value of multi parameter."""
    param_default_value = param_config.get("default")
    if param_default_value is not None:
        return param_default_value
    else:
        param_values = param_config["values"]
        if param_values:
            first_component_param_type = sorted(param_values.keys())[0]
            first_component_param = param_values[first_component_param_type]
            if first_component_param_type == ModelMetadataHyperParamTypes.INT:
                return _get_default_numeric_param_value(first_component_param, cast_to_int=True)
            elif first_component_param_type == ModelMetadataHyperParamTypes.FLOAT:
                return _get_default_numeric_param_value(first_component_param, cast_to_int=False)
            elif first_component_param_type == ModelMetadataHyperParamTypes.SELECT:
                return _get_default_select_param_value(first_component_param)


def get_default_parameter_values(model_metadata: Dict) -> Dict[str, Union[int, float, str]]:
    """Retrieve default parameter values from the hyperparameter section of model-metadata.yaml.
    When `default` is provided, return the default value.
    When `default` is not provided:
        - Return `min` value if it is a numeric parameter.
        - Return an empty string if it is a string parameter.
        - Return the first allowed value if it is the select parameter.
        - Return the default value of the first component parameter (sorted by parameter type) if it is a multi
          parameter.
    """
    hyper_param_config = model_metadata.get(ModelMetadataKeys.HYPERPARAMETERS, [])
    default_params = {}
    for param_config in hyper_param_config:
        param_name = param_config["name"]
        param_type = param_config["type"]
        if param_type == ModelMetadataHyperParamTypes.INT:
            default_params[param_name] = _get_default_numeric_param_value(
                param_config, cast_to_int=True
            )
        elif param_type == ModelMetadataHyperParamTypes.FLOAT:
            default_params[param_name] = _get_default_numeric_param_value(
                param_config, cast_to_int=False
            )
        elif param_type == ModelMetadataHyperParamTypes.STRING:
            default_params[param_name] = _get_default_string_param_value(param_config)
        elif param_type == ModelMetadataHyperParamTypes.SELECT:
            default_params[param_name] = _get_default_select_param_value(param_config)
        elif param_type == ModelMetadataHyperParamTypes.MULTI:
            default_param_value = _get_default_multi_param_value(param_config)
            if default_param_value is not None:
                default_params[param_name] = default_param_value
    return default_params
