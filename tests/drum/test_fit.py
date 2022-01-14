"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import shutil
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.io import mmwrite

from datarobot_drum.drum.enum import ArgumentsOptions, InputFormatExtension
from datarobot_drum.drum.utils import handle_missing_colnames, unset_drum_supported_env_vars
from datarobot_drum.resource.utils import (
    _cmd_add_class_labels,
    _create_custom_model_dir,
    _exec_shell_cmd,
)
from .constants import (
    ANOMALY,
    BINARY,
    BINARY_BOOL,
    BINARY_INT,
    BINARY_INT_TARGET,
    BINARY_SPACES,
    BINARY_NUM_ONLY,
    BINARY_TEXT,
    DOCKER_PYTHON_SKLEARN,
    PYTHON_XFORM_ESTIMATOR,
    R_XFORM_ESTIMATOR,
    KERAS,
    MULTICLASS,
    MULTICLASS_BINARY,
    MULTICLASS_NUM_LABELS,
    MULTICLASS_FLOAT_LABELS,
    MULTICLASS_HIGH_CARD,
    PYTHON,
    PYTORCH,
    PYTORCH_REGRESSION,
    PYTORCH_MULTICLASS,
    R_FIT,
    RDS,
    RDS_BINARY,
    RDS_SPARSE,
    REGRESSION,
    REGRESSION_SINGLE_COL,
    SIMPLE,
    SKLEARN,
    SKLEARN_ANOMALY,
    SKLEARN_BINARY,
    SKLEARN_MULTICLASS,
    SKLEARN_PRED_CONSISTENCY,
    SKLEARN_REGRESSION,
    SKLEARN_SPARSE,
    SKLEARN_TRANSFORM,
    SKLEARN_TRANSFORM_NO_HOOK,
    SKLEARN_TRANSFORM_NON_NUMERIC,
    SKLEARN_TRANSFORM_SPARSE_IN_OUT,
    SKLEARN_TRANSFORM_SPARSE_INPUT,
    SKLEARN_TRANSFORM_WITH_Y,
    SPARSE,
    SPARSE_COLUMNS,
    SPARSE_TARGET,
    TARGET_NAME_DUPLICATED_X,
    TARGET_NAME_DUPLICATED_Y,
    TESTS_ROOT_PATH,
    TRANSFORM,
    WEIGHTS_ARGS,
    WEIGHTS_CSV,
    XGB,
    SKLEARN_BINARY_PARAMETERS,
    SKLEARN_BINARY_HYPERPARAMETERS,
    SKLEARN_TRANSFORM_HYPERPARAMETERS,
    SKLEARN_TRANSFORM_PARAMETERS,
    RDS_HYPERPARAMETERS,
    RDS_PARAMETERS,
    PYTHON_TRANSFORM_FAIL_OUTPUT_SCHEMA_VALIDATION,
    R_TRANSFORM_SPARSE_INPUT,
    R_TRANSFORM_SPARSE_IN_OUT,
    R_TRANSFORM_WITH_Y,
    R_TRANSFORM,
    R_TRANSFORM_NO_HOOK,
    R_TRANSFORM_NON_NUMERIC,
    R_ESTIMATOR_SPARSE,
    R_VALIDATE_SPARSE_ESTIMATOR,
    R_TRANSFORM_SPARSE_INPUT_Y_OUTPUT,
    SKLEARN_TRANSFORM_SPARSE_INPUT_Y_OUTPUT,
    REGRESSION_MULTLILINE_TEXT,
    CUSTOM_TASK_INTERFACE_TRANSFORM,
    CUSTOM_TASK_INTERFACE_REGRESSION,
    CUSTOM_TASK_INTERFACE_ANOMALY,
    CUSTOM_TASK_INTERFACE_BINARY,
    CUSTOM_TASK_INTERFACE_MULTICLASS,
    CUSTOM_TASK_INTERFACE_PYTORCH_BINARY,
    CUSTOM_TASK_INTERFACE_PYTORCH_MULTICLASS,
    CUSTOM_TASK_INTERFACE_KERAS_REGRESSION,
    CUSTOM_TASK_INTERFACE_XGB_REGRESSION,
)


class TestFit:
    @staticmethod
    def _add_weights_cmd(weights, df, input_name, r_fit=False):
        colname = "some-weights"
        weights_data = pd.Series(np.random.randint(1, 3, len(df)))
        ext = os.path.splitext(input_name)[1]
        if weights == WEIGHTS_ARGS:
            __keep_this_around = NamedTemporaryFile("w", suffix=ext)
            df[colname] = weights_data
            if r_fit:
                df = handle_missing_colnames(df)
            if ext == ".mtx":
                with open(input_name.replace(".mtx", ".columns")) as f:
                    sparse_colnames = [col.rstrip() for col in f]
                sparse_colnames.append(colname)
                tmp_colname_file = __keep_this_around.name.replace(".mtx", ".columns")
                with open(tmp_colname_file, "w") as f:
                    f.write("\n".join(sparse_colnames))
                df[colname] = pd.arrays.SparseArray(
                    df[colname], dtype=pd.SparseDtype(np.float64, 0)
                )
                mmwrite(__keep_this_around.name, sp.csr_matrix(df.to_numpy()))
            else:
                df.to_csv(__keep_this_around.name, index=False, line_terminator="\r\n")
            return " --row-weights " + colname, __keep_this_around.name, __keep_this_around
        elif weights == WEIGHTS_CSV:
            __keep_this_around = NamedTemporaryFile("w")
            weights_data.to_csv(__keep_this_around.name, index=False, line_terminator="\r\n")
            return " --row-weights-csv " + __keep_this_around.name, input_name, __keep_this_around

        __keep_this_around = NamedTemporaryFile("w")
        return "", input_name, __keep_this_around

    @pytest.mark.parametrize("framework", [XGB, RDS])
    @pytest.mark.parametrize("problem", [REGRESSION])
    @pytest.mark.parametrize("docker", [DOCKER_PYTHON_SKLEARN, None])
    @pytest.mark.parametrize("weights", [None])
    @pytest.mark.parametrize("use_output", [True, False])
    @pytest.mark.parametrize("nested", [True, False])
    def test_fit_for_use_output_and_nested(
        self, resources, framework, problem, docker, weights, use_output, tmp_path, nested,
    ):
        if docker and framework != SKLEARN:
            return
        if framework == RDS:
            language = R_FIT
        else:
            language = PYTHON

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language, is_training=True, nested=nested
        )

        input_dataset = resources.datasets(framework, problem)
        input_df = resources.input_data(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_df, input_dataset, r_fit=language == R_FIT
        )

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, problem, custom_model_dir, input_dataset
        )
        if problem != ANOMALY:
            cmd += ' --target "{}"'.format(resources.targets(problem))

        if use_output:
            cmd += " --output {}".format(output)
        if problem == BINARY:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type=problem
            )
        if docker:
            cmd += " --docker {} ".format(docker)

        cmd += weights_cmd

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    @pytest.mark.parametrize(
        "framework, problem, docker",
        [
            # (R_XFORM_ESTIMATOR, REGRESSION, None),
            # (PYTHON_XFORM_ESTIMATOR, REGRESSION, None),
            # (SKLEARN_SPARSE, SPARSE, None),
            # (RDS_SPARSE, SPARSE, None),
            # (RDS, BINARY_BOOL, None),
            # (RDS_BINARY, BINARY_INT, None),
            # (RDS, BINARY_TEXT, None),
            # (RDS, REGRESSION, None),
            # (RDS, REGRESSION_SINGLE_COL, None),
            # (RDS, MULTICLASS_FLOAT_LABELS, None),
            # (RDS, MULTICLASS, None),
            # (RDS, MULTICLASS_BINARY, None),
            # (SKLEARN_BINARY, BINARY_TEXT, DOCKER_PYTHON_SKLEARN),
            # (SKLEARN_REGRESSION, REGRESSION, DOCKER_PYTHON_SKLEARN),
            # (SKLEARN_ANOMALY, ANOMALY, DOCKER_PYTHON_SKLEARN),
            # (SKLEARN_MULTICLASS, MULTICLASS, DOCKER_PYTHON_SKLEARN),
            # (SKLEARN_BINARY, BINARY_TEXT, None),
            # (SKLEARN_BINARY, BINARY_SPACES, None),
            # (SKLEARN_REGRESSION, REGRESSION, None),
            # (SKLEARN_REGRESSION, REGRESSION_MULTLILINE_TEXT, None),
            # (SKLEARN_ANOMALY, ANOMALY, None),
            # (SKLEARN_MULTICLASS, MULTICLASS, None),
            # (SKLEARN_MULTICLASS, MULTICLASS_BINARY, None),
            # (SKLEARN_MULTICLASS, MULTICLASS_NUM_LABELS, None),
            # (PYTORCH_MULTICLASS, MULTICLASS_HIGH_CARD, None),
            # (XGB, REGRESSION, None),
            # (KERAS, REGRESSION, None),
            # (PYTORCH, BINARY_TEXT, None),
            # (PYTORCH_MULTICLASS, MULTICLASS, None),
            # (PYTORCH_MULTICLASS, MULTICLASS_BINARY, None),
            (CUSTOM_TASK_INTERFACE_BINARY, BINARY_TEXT, DOCKER_PYTHON_SKLEARN),
            (CUSTOM_TASK_INTERFACE_REGRESSION, REGRESSION, DOCKER_PYTHON_SKLEARN),
            (CUSTOM_TASK_INTERFACE_ANOMALY, ANOMALY, DOCKER_PYTHON_SKLEARN),
            (CUSTOM_TASK_INTERFACE_MULTICLASS, MULTICLASS, DOCKER_PYTHON_SKLEARN),
            (CUSTOM_TASK_INTERFACE_BINARY, BINARY_NUM_ONLY, None),
            (CUSTOM_TASK_INTERFACE_BINARY, BINARY_SPACES, None),
            (CUSTOM_TASK_INTERFACE_REGRESSION, REGRESSION, None),
            (CUSTOM_TASK_INTERFACE_ANOMALY, ANOMALY, None),
            (CUSTOM_TASK_INTERFACE_MULTICLASS, MULTICLASS, None),
            (CUSTOM_TASK_INTERFACE_MULTICLASS, MULTICLASS_BINARY, None),
            (CUSTOM_TASK_INTERFACE_MULTICLASS, MULTICLASS_NUM_LABELS, None),
            (CUSTOM_TASK_INTERFACE_PYTORCH_BINARY, BINARY_NUM_ONLY, None),
            (CUSTOM_TASK_INTERFACE_PYTORCH_MULTICLASS, MULTICLASS, None),
            (CUSTOM_TASK_INTERFACE_PYTORCH_MULTICLASS, MULTICLASS_HIGH_CARD, None),
            (CUSTOM_TASK_INTERFACE_PYTORCH_MULTICLASS, MULTICLASS_BINARY, None),
            (CUSTOM_TASK_INTERFACE_KERAS_REGRESSION, REGRESSION, None),
            (CUSTOM_TASK_INTERFACE_XGB_REGRESSION, REGRESSION, None),
            # (SKLEARN_SPARSE, SPARSE, None),
        ],
    )
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    def test_fit(
        self, resources, framework, problem, docker, weights, tmp_path,
    ):
        if framework in {RDS, RDS_BINARY, RDS_SPARSE, R_XFORM_ESTIMATOR}:
            language = R_FIT
        else:
            language = PYTHON

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language, is_training=True,
        )

        input_dataset = resources.datasets(framework, problem)
        input_df = resources.input_data(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_df, input_dataset, r_fit=language == R_FIT
        )

        target_type = resources.target_types(problem)

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose --show-stacktrace --disable-strict-validation".format(
            ArgumentsOptions.MAIN_COMMAND, target_type, custom_model_dir, input_dataset
        )
        if problem not in {ANOMALY, SPARSE, BINARY_INT}:
            cmd += ' --target "{}"'.format(resources.targets(problem))

        if problem == SPARSE:
            input_dir = tmp_path / "input_dir"
            input_dir.mkdir(parents=True, exist_ok=True)
            target_file = os.path.join(input_dir, "y.csv")
            shutil.copyfile(resources.datasets(None, SPARSE_TARGET), target_file)
            sparse_column_file = input_dataset.replace(".mtx", ".columns")
            cmd += " --sparse-column-file {} --target-csv {}".format(
                sparse_column_file, target_file
            )
        if problem == BINARY_INT:
            # target-csv will result in target dtype int instead of str
            target_dataset = resources.datasets(None, BINARY_INT_TARGET)
            cmd += " --target-csv {}".format(target_dataset)

        if problem in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type=target_type
            )

        if docker:
            cmd += " --docker {} ".format(docker)

        cmd += weights_cmd

        _, stdout, _ = _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
        )
        assert "Starting Fit" in stdout
        assert "Starting Prediction" in stdout

    @pytest.mark.parametrize(
        "framework, problem, docker, parameters",
        [
            (SKLEARN_BINARY_HYPERPARAMETERS, BINARY_TEXT, None, SKLEARN_BINARY_PARAMETERS),
            (SKLEARN_BINARY_HYPERPARAMETERS, BINARY_SPACES, None, SKLEARN_BINARY_PARAMETERS),
            (SKLEARN_TRANSFORM_HYPERPARAMETERS, REGRESSION, None, SKLEARN_TRANSFORM_PARAMETERS),
            (RDS_HYPERPARAMETERS, BINARY_TEXT, None, RDS_PARAMETERS),
        ],
    )
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    def test_fit_hyperparameters(
        self, resources, framework, problem, docker, parameters, weights, tmp_path,
    ):
        if framework == RDS_HYPERPARAMETERS:
            language = R_FIT
        else:
            language = PYTHON

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language, is_training=True,
        )

        input_dataset = resources.datasets(framework, problem)
        parameter_file = resources.datasets(framework, parameters)
        input_df = resources.input_data(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_df, input_dataset, r_fit=language == R_FIT
        )

        target_type = resources.target_types(problem) if "transform" not in framework else TRANSFORM

        cmd = "{} fit --target-type {} --code-dir {} --input {} --parameter-file {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND,
            target_type,
            custom_model_dir,
            input_dataset,
            parameter_file,
        )
        if problem != ANOMALY:
            cmd += ' --target "{}"'.format(resources.targets(problem))

        if problem in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type=target_type
            )
        if docker:
            cmd += " --docker {} ".format(docker)

        cmd += weights_cmd

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    @pytest.mark.parametrize(
        "framework, language, is_framework_directory",
        [
            (SKLEARN_TRANSFORM, SKLEARN_TRANSFORM, False),
            (SKLEARN_TRANSFORM_NO_HOOK, SKLEARN_TRANSFORM_NO_HOOK, False),
            (R_TRANSFORM, R_TRANSFORM, False),
            (R_TRANSFORM_NO_HOOK, R_TRANSFORM_NO_HOOK, False),
            (CUSTOM_TASK_INTERFACE_TRANSFORM, PYTHON, True),
        ],
    )
    @pytest.mark.parametrize("problem", [REGRESSION, BINARY, ANOMALY])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    def test_transform_fit(
        self, resources, framework, is_framework_directory, language, problem, weights, tmp_path,
    ):
        # TODO: [RAPTOR-6175] Improve the test utils for custom tasks
        # the is_training parameter should not make assumptions of whether the framework is a single file or directory
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language=language,
            is_training=is_framework_directory,
        )

        input_dataset = resources.datasets(framework, problem)
        input_df = resources.input_data(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_df, input_dataset, r_fit=framework in [R_TRANSFORM, R_TRANSFORM_NO_HOOK],
        )

        target_type = TRANSFORM

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, target_type, custom_model_dir, input_dataset
        )
        if problem != ANOMALY:
            cmd += ' --target "{}"'.format(resources.targets(problem))

        if problem in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type=target_type
            )

        cmd += weights_cmd

        _, stdout, _ = _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

        # Ensure the default type schema is used since we do not disable strict validation
        assert "WARNING: No type schema provided. For transforms, we" in stdout

    @pytest.mark.parametrize(
        "framework, language",
        [(SKLEARN_TRANSFORM_NON_NUMERIC, PYTHON), (R_TRANSFORM_NON_NUMERIC, R_FIT),],
    )
    @pytest.mark.parametrize("problem", [REGRESSION, BINARY, ANOMALY])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    @pytest.mark.parametrize("strict", [True, False])
    def test_transform_fit_fails_default(
        self, resources, framework, language, problem, weights, tmp_path, strict
    ):
        """Test that with strict validation non numeric transforms fail, but pass when strict validation is disabled."""
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language=framework,
        )

        input_dataset = resources.datasets(framework, problem)
        input_df = resources.input_data(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_df, input_dataset, r_fit=language == R_FIT
        )

        target_type = TRANSFORM

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            target_type,
            custom_model_dir,
            input_dataset,
            "" if strict else "--disable-strict-validation",
        )
        if problem != ANOMALY:
            cmd += ' --target "{}"'.format(resources.targets(problem))

        if problem in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type=target_type
            )

        cmd += weights_cmd

        if strict:
            with pytest.raises(AssertionError):
                _, stdout, _ = _exec_shell_cmd(
                    cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
                )
                assert "expected NUM" in stdout
        else:
            _, stdout, _ = _exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )
            assert "WARNING: No type schema provided. For transforms, we" not in stdout

    @pytest.mark.parametrize(
        "framework, language",
        [
            (SKLEARN_TRANSFORM_WITH_Y, PYTHON),
            # disabling R, there is a bug where the Y value is only passed in for sparse
            # (R_TRANSFORM, R_FIT),
        ],
    )
    @pytest.mark.parametrize("problem", [REGRESSION, BINARY, ANOMALY])
    def test_transform_fit_disallow_y_output(
        self, resources, tmp_path, framework, language, problem
    ):

        input_dataset = resources.datasets(framework, problem)
        target_type = TRANSFORM
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language=framework,
        )

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            target_type,
            custom_model_dir,
            input_dataset,
            "--disable-strict-validation",
        )
        _, stdout, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert "Transformation of the target variable is not supported by DRUM." in stdout

    @pytest.mark.parametrize(
        "framework", [SKLEARN_TRANSFORM_SPARSE_INPUT_Y_OUTPUT, R_TRANSFORM_SPARSE_INPUT_Y_OUTPUT]
    )
    def test_sparse_transform_fit_disallow_y_output(
        self, framework, resources, tmp_path,
    ):
        input_dataset = resources.datasets(None, SPARSE)
        target_dataset = resources.datasets(None, SPARSE_TARGET)

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, REGRESSION, language=framework,
        )
        columns = resources.datasets(framework, SPARSE_COLUMNS)

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose --target-csv {} --sparse-column-file {} --disable-strict-validation".format(
            ArgumentsOptions.MAIN_COMMAND,
            TRANSFORM,
            custom_model_dir,
            input_dataset,
            target_dataset,
            columns,
        )
        _, stdout, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert "Transformation of the target variable is not supported by DRUM." in stdout

    @pytest.mark.parametrize(
        "framework",
        [
            SKLEARN_TRANSFORM_SPARSE_IN_OUT,
            SKLEARN_TRANSFORM_SPARSE_INPUT,
            R_TRANSFORM_SPARSE_IN_OUT,
            R_TRANSFORM_SPARSE_INPUT,
        ],
    )
    def test_sparse_transform_fit(
        self, framework, resources, tmp_path,
    ):
        input_dataset = resources.datasets(None, SPARSE)
        target_dataset = resources.datasets(None, SPARSE_TARGET)

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, REGRESSION, language=framework,
        )
        columns = resources.datasets(framework, SPARSE_COLUMNS)

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose --target-csv {} --sparse-column-file {} --disable-strict-validation".format(
            ArgumentsOptions.MAIN_COMMAND,
            TRANSFORM,
            custom_model_dir,
            input_dataset,
            target_dataset,
            columns,
        )

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    def _create_fit_input_data_dir(
        self, get_target, get_dataset_filename, input_dir, problem, weights, is_sparse=False
    ):
        input_dir.mkdir(parents=True, exist_ok=True)

        # Training data
        if is_sparse:
            X_file = os.path.join(input_dir, "X.mtx")
            input_dataset = get_dataset_filename(None, SPARSE)
            shutil.copyfile(input_dataset, X_file)
        else:
            X_file = os.path.join(input_dir, "X.csv")
            input_dataset = get_dataset_filename(None, problem)
            with open(X_file, "w+") as fp:
                df = pd.read_csv(input_dataset)
                if problem == ANOMALY or is_sparse:
                    feature_df = df
                else:
                    feature_df = df.loc[:, df.columns != get_target(problem)]
                feature_df.to_csv(fp, index=False, line_terminator="\r\n")

        if problem != ANOMALY:
            # Target data
            target_file = os.path.join(input_dir, "y.csv")
            if not is_sparse:
                with open(target_file, "w+") as fp:
                    target_series = df[get_target(problem)]
                    target_series.to_csv(fp, index=False, header="Target", line_terminator="\r\n")
            if is_sparse:
                shutil.copyfile(get_dataset_filename(None, SPARSE_TARGET), target_file)

        if is_sparse:
            columns = get_dataset_filename(None, SPARSE_COLUMNS)
            shutil.copyfile(columns, input_dir / "X.colnames")

        # Weights data
        if weights:
            df = pd.read_csv(input_dataset)
            weights_data = pd.Series(np.random.randint(1, 3, len(df)))
            with open(os.path.join(input_dir, "weights.csv"), "w+") as fp:
                weights_data.to_csv(fp, index=False)

    @pytest.mark.parametrize(
        "framework, problem, parameters",
        [
            (SKLEARN_BINARY, BINARY_TEXT, None),
            (SKLEARN_BINARY, BINARY, None),
            (SKLEARN_BINARY_HYPERPARAMETERS, BINARY, SKLEARN_BINARY_PARAMETERS),
            (SKLEARN_ANOMALY, ANOMALY, None),
            (SKLEARN_MULTICLASS, MULTICLASS, None),
            (SKLEARN_SPARSE, REGRESSION, None),
            (XGB, REGRESSION, None),
            (KERAS, REGRESSION, None),
        ],
    )
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, None])
    def test_fit_sh(
        self, resources, framework, problem, parameters, weights, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, PYTHON, is_training=True,
        )

        env = os.environ
        fit_sh = os.path.join(
            TESTS_ROOT_PATH,
            "..",
            "public_dropin_environments/{}_{}/fit.sh".format(
                PYTHON,
                framework
                if framework
                not in [
                    SKLEARN_ANOMALY,
                    SKLEARN_BINARY,
                    SKLEARN_MULTICLASS,
                    SKLEARN_SPARSE,
                    SKLEARN_BINARY_HYPERPARAMETERS,
                ]
                else SKLEARN,
            ),
        )

        input_dir = tmp_path / "input_dir"
        self._create_fit_input_data_dir(
            resources.targets,
            resources.datasets,
            input_dir,
            problem,
            weights,
            is_sparse=framework == SKLEARN_SPARSE,
        )

        output = tmp_path / "output"
        output.mkdir()

        unset_drum_supported_env_vars()

        env["CODEPATH"] = str(custom_model_dir)
        env["INPUT_DIRECTORY"] = str(input_dir)
        env["ARTIFACT_DIRECTORY"] = str(output)
        env["TARGET_TYPE"] = problem if problem != BINARY_TEXT else BINARY
        if framework == SKLEARN_SPARSE:
            env["TRAINING_DATA_EXTENSION"] = InputFormatExtension.MTX
        else:
            env["TRAINING_DATA_EXTENSION"] = InputFormatExtension.CSV

        if problem in [BINARY, BINARY_TEXT]:
            labels = resources.class_labels(framework, problem)
            env["NEGATIVE_CLASS_LABEL"] = labels[0]
            env["POSITIVE_CLASS_LABEL"] = labels[1]
        elif problem == MULTICLASS:
            labels = resources.class_labels(framework, problem)
            with open(os.path.join(tmp_path, "class_labels.txt"), mode="w") as f:
                f.write("\n".join(labels))
                env["CLASS_LABELS_FILE"] = f.name

        if parameters:
            parameter_file = resources.datasets(framework, parameters)
            parameter_input_file = os.path.join(input_dir, "parameters.json")
            shutil.copyfile(parameter_file, parameter_input_file)

        _exec_shell_cmd(fit_sh, "Failed cmd {}".format(fit_sh), env=env)

        # clear env vars as it may affect next test cases
        unset_drum_supported_env_vars()

    @pytest.mark.parametrize("skip_predict", [True, False])
    def test_fit_simple(self, resources, tmp_path, skip_predict):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, SIMPLE, REGRESSION, PYTHON, is_training=True, nested=True,
        )

        input_dataset = resources.datasets(SKLEARN, REGRESSION)

        output = tmp_path / "output"
        output.mkdir()

        cmd = '{} fit --target-type {} --code-dir {} --target "{}" --input {} --verbose'.format(
            ArgumentsOptions.MAIN_COMMAND,
            REGRESSION,
            custom_model_dir,
            resources.targets(REGRESSION),
            input_dataset,
        )
        if skip_predict:
            cmd += " --skip-predict"
        _, stdout, _ = _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        if skip_predict:
            assert "Prediction started" not in stdout
            assert "predictions can be made on the fit model" not in stdout

    @pytest.mark.parametrize(
        "framework, problem, language, is_framework_directory",
        [
            (SKLEARN_SPARSE, SPARSE, PYTHON, True),
            (PYTORCH_REGRESSION, SPARSE, PYTHON, True),
            (
                R_ESTIMATOR_SPARSE,
                REGRESSION,
                R_FIT,
                True,
            ),  # Tests the R spare regression template (w/schema)
            (
                R_VALIDATE_SPARSE_ESTIMATOR,
                REGRESSION,
                R_VALIDATE_SPARSE_ESTIMATOR,
                False,
            ),  # Asserts data is sparse
        ],
    )
    def test_fit_sparse(
        self, resources, tmp_path, framework, problem, language, is_framework_directory
    ):
        # TODO: [RAPTOR-6175] Improve the test utils for custom tasks
        # the is_training parameter should not make assumptions of whether the framework is a single file or directory
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language=language,
            is_training=is_framework_directory,
        )

        input_dataset = resources.datasets(framework, SPARSE)
        target_dataset = resources.datasets(framework, SPARSE_TARGET)
        columns = resources.datasets(framework, SPARSE_COLUMNS)

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --code-dir {} --input {} --target-type {} --verbose --sparse-column-file {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, REGRESSION, columns
        )

        cmd += " --target-csv " + target_dataset
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    @pytest.mark.parametrize("framework, problem", [(SKLEARN_PRED_CONSISTENCY, BINARY_BOOL)])
    def test_prediction_consistency(self, resources, tmp_path, framework, problem):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, SPARSE, language=PYTHON, is_training=True,
        )

        input_dataset = resources.datasets(framework, problem)

        if problem in [BINARY_TEXT, BINARY_BOOL]:
            target_type = BINARY
        else:
            target_type = problem

        cmd = '{} fit --target-type {} --code-dir {} --target "{}" --input {} --verbose '.format(
            ArgumentsOptions.MAIN_COMMAND,
            target_type,
            custom_model_dir,
            resources.targets(problem),
            input_dataset,
        )

        if target_type in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type
            )

        _, stdout, stderr = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=True,
        )

        # we should throw a warning, not an error
        assert "Your predictions were different when we tried to predict twice." in stderr
        # but don't error out
        assert (
            "Your model can be fit to your data,  and predictions can be made on the fit model!"
            in stdout
        )
        # clean up
        sample_dir = stderr.split(":")[-1]
        if sample_dir.endswith("\n"):
            sample_dir = sample_dir[:-1]
        os.remove(sample_dir.strip())

    def test_duplicate_target_name(self, resources, tmp_path):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, SKLEARN_REGRESSION, SPARSE, language=PYTHON, is_training=True,
        )

        input_dataset = resources.datasets(SKLEARN_REGRESSION, TARGET_NAME_DUPLICATED_X)
        target_dataset = resources.datasets(SKLEARN_REGRESSION, TARGET_NAME_DUPLICATED_Y)

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --code-dir {} --input {} --target-type {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, REGRESSION
        )

        cmd += " --target-csv " + target_dataset
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    def test_fit_schema_validation(self, resources, tmp_path):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            SKLEARN_BINARY,
            BINARY,
            PYTHON,
            is_training=True,
            include_metadata=True,
        )

        input_dataset = resources.datasets(SKLEARN, BINARY)

        output = tmp_path / "output"
        output.mkdir()

        cmd = '{} fit --target-type {} --code-dir {} --target "{}" --input {} --verbose'.format(
            ArgumentsOptions.MAIN_COMMAND,
            BINARY,
            custom_model_dir,
            resources.targets(BINARY),
            input_dataset,
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    @pytest.mark.parametrize(
        "framework, problem, language, error_in_predict_server",
        [
            (PYTORCH, BINARY, PYTHON, False),
            (PYTHON_TRANSFORM_FAIL_OUTPUT_SCHEMA_VALIDATION, TRANSFORM, PYTHON, True),
        ],
    )
    def test_fit_schema_failure(
        self, resources, framework, problem, language, error_in_predict_server, tmp_path
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
            is_training=True,
            include_metadata=True,
        )

        input_dataset = resources.datasets(SKLEARN, BINARY_TEXT)
        output = tmp_path / "output"
        output.mkdir()

        cmd = '{} fit --target-type {} --code-dir {} --target "{}" --input {} --verbose'.format(
            ArgumentsOptions.MAIN_COMMAND,
            problem,
            custom_model_dir,
            resources.targets(BINARY_TEXT),
            input_dataset,
        )
        _, stdout, stderr = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        if error_in_predict_server:
            assert (
                "Schema validation found mismatch between output dataset and the supplied schema"
                in stdout
            )
        else:
            assert (
                "Schema validation found mismatch between input dataset and the supplied schema"
                in stderr
            )
