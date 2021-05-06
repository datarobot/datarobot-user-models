import os
import shutil
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

from datarobot_drum.drum.common import ArgumentsOptions
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
    BINARY_SPACES,
    BINARY_TEXT,
    DOCKER_PYTHON_SKLEARN,
    KERAS,
    MULTICLASS,
    MULTICLASS_BINARY,
    MULTICLASS_NUM_LABELS,
    PYTHON,
    PYTORCH,
    PYTORCH_MULTICLASS,
    R_FIT,
    RDS,
    REGRESSION,
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
    SKLEARN_BINARY_SCHEMA_VALIDATION,
)


class TestFit:
    @staticmethod
    def _add_weights_cmd(weights, input_csv, r_fit=False):
        df = pd.read_csv(input_csv)
        colname = "some-colname"
        weights_data = pd.Series(np.random.randint(1, 3, len(df)))
        __keep_this_around = NamedTemporaryFile("w")
        if weights == WEIGHTS_ARGS:
            df[colname] = weights_data
            if r_fit:
                df = handle_missing_colnames(df)
            df.to_csv(__keep_this_around.name)
            return " --row-weights " + colname, __keep_this_around.name, __keep_this_around
        elif weights == WEIGHTS_CSV:
            weights_data.to_csv(__keep_this_around.name)
            return " --row-weights-csv " + __keep_this_around.name, input_csv, __keep_this_around

        return "", input_csv, __keep_this_around

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

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_dataset, r_fit=language == R_FIT
        )

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, problem, custom_model_dir, input_dataset
        )
        if problem != ANOMALY:
            cmd += " --target {}".format(resources.targets(problem))

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
            (RDS, BINARY_BOOL, None),
            (RDS, BINARY_TEXT, None),
            (RDS, REGRESSION, None),
            (RDS, MULTICLASS, None),
            (RDS, MULTICLASS_BINARY, None),
            (SKLEARN_BINARY, BINARY_TEXT, DOCKER_PYTHON_SKLEARN),
            (SKLEARN_REGRESSION, REGRESSION, DOCKER_PYTHON_SKLEARN),
            (SKLEARN_ANOMALY, ANOMALY, DOCKER_PYTHON_SKLEARN),
            (SKLEARN_MULTICLASS, MULTICLASS, DOCKER_PYTHON_SKLEARN),
            (SKLEARN_BINARY, BINARY_TEXT, None),
            (SKLEARN_BINARY, BINARY_SPACES, None),
            (SKLEARN_REGRESSION, REGRESSION, None),
            (SKLEARN_ANOMALY, ANOMALY, None),
            (SKLEARN_MULTICLASS, MULTICLASS, None),
            (SKLEARN_MULTICLASS, MULTICLASS_BINARY, None),
            (SKLEARN_MULTICLASS, MULTICLASS_NUM_LABELS, None),
            (XGB, BINARY_TEXT, None),
            (XGB, REGRESSION, None),
            (XGB, MULTICLASS, None),
            (XGB, MULTICLASS_BINARY, None),
            (KERAS, BINARY_TEXT, None),
            (KERAS, REGRESSION, None),
            (KERAS, MULTICLASS, None),
            (KERAS, MULTICLASS_BINARY, None),
            (PYTORCH, BINARY_TEXT, None),
            (PYTORCH, REGRESSION, None),
            (PYTORCH_MULTICLASS, MULTICLASS, None),
            (PYTORCH_MULTICLASS, MULTICLASS_BINARY, None),
        ],
    )
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    def test_fit(
        self, resources, framework, problem, docker, weights, tmp_path,
    ):
        if framework == RDS:
            language = R_FIT
        else:
            language = PYTHON

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language, is_training=True,
        )

        input_dataset = resources.datasets(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_dataset, r_fit=language == R_FIT
        )

        target_type = resources.target_types(problem)

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, target_type, custom_model_dir, input_dataset
        )
        if problem != ANOMALY:
            cmd += " --target {}".format(resources.targets(problem))

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

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_dataset, r_fit=language == R_FIT
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
            cmd += " --target {}".format(resources.targets(problem))

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
        "framework",
        [
            SKLEARN_TRANSFORM,
            SKLEARN_TRANSFORM_WITH_Y,
            SKLEARN_TRANSFORM_NO_HOOK,
            SKLEARN_TRANSFORM_NON_NUMERIC,
        ],
    )
    @pytest.mark.parametrize("problem", [REGRESSION, BINARY, ANOMALY])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    def test_transform_fit(
        self, resources, framework, problem, weights, tmp_path,
    ):
        language = PYTHON
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language=framework,
        )

        input_dataset = resources.datasets(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_dataset, r_fit=language == R_FIT
        )

        target_type = TRANSFORM

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, target_type, custom_model_dir, input_dataset
        )
        if problem != ANOMALY:
            cmd += " --target {}".format(resources.targets(problem))

        if problem in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type=target_type
            )

        cmd += weights_cmd

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    @pytest.mark.parametrize(
        "framework", [SKLEARN_TRANSFORM_SPARSE_IN_OUT, SKLEARN_TRANSFORM_SPARSE_INPUT,],
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

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose --target-csv {} --sparse-column-file {}".format(
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
                feature_df.to_csv(fp, index=False)

        if problem != ANOMALY:
            # Target data
            target_file = os.path.join(input_dir, "y.csv")
            if not is_sparse:
                with open(target_file, "w+") as fp:
                    target_series = df[get_target(problem)]
                    target_series.to_csv(fp, index=False, header="Target")
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
                weights_data.to_csv(fp)

    @pytest.mark.parametrize(
        "framework, problem, parameters",
        [
            (SKLEARN_BINARY, BINARY_TEXT, None),
            (SKLEARN_BINARY, BINARY, None),
            (SKLEARN_BINARY_HYPERPARAMETERS, BINARY, SKLEARN_BINARY_PARAMETERS),
            (SKLEARN_ANOMALY, ANOMALY, None),
            (SKLEARN_MULTICLASS, MULTICLASS, None),
            (SKLEARN_SPARSE, REGRESSION, None),
            (XGB, BINARY_TEXT, None),
            (XGB, BINARY, None),
            (XGB, MULTICLASS, None),
            (KERAS, BINARY_TEXT, None),
            (KERAS, BINARY, None),
            (KERAS, MULTICLASS, None),
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
            env["TRAINING_DATA_EXTENSION"] = ".mtx"
        else:
            env["TRAINING_DATA_EXTENSION"] = ".csv"

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

    def test_fit_simple(
        self, resources, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, SIMPLE, REGRESSION, PYTHON, is_training=True, nested=True,
        )

        input_dataset = resources.datasets(SKLEARN, REGRESSION)

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --target-type {} --code-dir {} --target {} --input {} --verbose".format(
            ArgumentsOptions.MAIN_COMMAND,
            REGRESSION,
            custom_model_dir,
            resources.targets(REGRESSION),
            input_dataset,
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    @pytest.mark.parametrize(
        "framework", [SKLEARN_SPARSE, PYTORCH, RDS,],
    )
    def test_fit_sparse(self, resources, tmp_path, framework):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            SPARSE,
            language=R_FIT if framework == RDS else PYTHON,
            is_training=True,
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

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, target_type, custom_model_dir, input_dataset
        )
        cmd += " --target {}".format(resources.targets(problem))

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
            SKLEARN_BINARY_SCHEMA_VALIDATION,
            BINARY,
            PYTHON,
            is_training=True,
            include_metadata=True,
        )

        input_dataset = resources.datasets(SKLEARN, BINARY)

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --target-type {} --code-dir {} --target {} --input {} --verbose".format(
            ArgumentsOptions.MAIN_COMMAND,
            BINARY,
            custom_model_dir,
            resources.targets(BINARY),
            input_dataset,
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    def test_fit_schema_failure(self, resources, tmp_path):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            SKLEARN_BINARY_SCHEMA_VALIDATION,
            BINARY,
            PYTHON,
            is_training=True,
            include_metadata=True,
        )

        input_dataset = resources.datasets(SKLEARN, BINARY_TEXT)
        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --target-type {} --code-dir {} --target {} --input {} --verbose".format(
            ArgumentsOptions.MAIN_COMMAND,
            BINARY,
            custom_model_dir,
            resources.targets(BINARY_TEXT),
            input_dataset,
        )
        with pytest.raises(AssertionError):
            _, _, stderr = _exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )
            assert "DrumSchemaValidationException" in stderr
