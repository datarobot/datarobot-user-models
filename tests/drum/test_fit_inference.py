import json
import os
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
import requests

from datarobot_drum.drum.common import ArgumentsOptions

from .constants import TESTS_ROOT_PATH
from .utils import (
    _exec_shell_cmd,
    _cmd_add_class_labels,
    _create_custom_model_dir,
)
from .drum_server_utils import DrumServerRun


from .constants import (
    XGB,
    KERAS,
    SKLEARN,
    SIMPLE,
    PYTORCH,
    PYPMML,
    SKLEARN_ANOMALY,
    RDS,
    CODEGEN,
    MOJO,
    POJO,
    MULTI_ARTIFACT,
    REGRESSION,
    REGRESSION_INFERENCE,
    BINARY,
    ANOMALY,
    PYTHON,
    NO_CUSTOM,
    PYTHON_LOAD_MODEL,
    R,
    R_FIT,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    DOCKER_PYTHON_SKLEARN,
    RESPONSE_PREDICTIONS_KEY,
    WEIGHTS_ARGS,
    WEIGHTS_CSV,
)


class TestFitInference:
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (KERAS, REGRESSION, PYTHON, None),
            (KERAS, BINARY, PYTHON, None),
            (XGB, REGRESSION, PYTHON, None),
            (XGB, BINARY, PYTHON, None),
            (XGB, BINARY, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None),
            (PYTORCH, REGRESSION, PYTHON, None),
            (PYTORCH, BINARY, PYTHON, None),
            (RDS, REGRESSION, R, None),
            (RDS, BINARY, R, None),
            (CODEGEN, REGRESSION, NO_CUSTOM, None),
            (CODEGEN, BINARY, NO_CUSTOM, None),
            (POJO, REGRESSION, NO_CUSTOM, None),
            (POJO, BINARY, NO_CUSTOM, None),
            (MOJO, REGRESSION, NO_CUSTOM, None),
            (MOJO, BINARY, NO_CUSTOM, None),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None),
            (PYPMML, REGRESSION, NO_CUSTOM, None),
            (PYPMML, BINARY, NO_CUSTOM, None),
        ],
    )
    def test_custom_models_with_drum(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output
        )
        if problem == BINARY:
            cmd = _cmd_add_class_labels(cmd, resources.class_labels(framework, problem))
        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = pd.read_csv(input_dataset)
        out_data = pd.read_csv(output)
        assert in_data.shape[0] == out_data.shape[0]

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (KERAS, REGRESSION, PYTHON, None),
            (KERAS, BINARY, PYTHON, None),
            (XGB, REGRESSION, PYTHON, None),
            (XGB, BINARY, PYTHON, None),
            (PYTORCH, REGRESSION, PYTHON, None),
            (PYTORCH, BINARY, PYTHON, None),
            (RDS, REGRESSION, R, None),
            (RDS, BINARY, R, None),
            (CODEGEN, REGRESSION, NO_CUSTOM, None),
            (CODEGEN, BINARY, NO_CUSTOM, None),
            (MOJO, REGRESSION, NO_CUSTOM, None),
            (MOJO, BINARY, NO_CUSTOM, None),
            (POJO, REGRESSION, NO_CUSTOM, None),
            (POJO, BINARY, NO_CUSTOM, None),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None),
            (PYPMML, REGRESSION, NO_CUSTOM, None),
            (PYPMML, BINARY, NO_CUSTOM, None),
        ],
    )
    def test_custom_models_with_drum_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            resources.class_labels(framework, problem), custom_model_dir, docker
        ) as run:
            input_dataset = resources.datasets(framework, problem)

            # do predictions
            response = requests.post(
                run.url_server_address + "/predict/", files={"X": open(input_dataset)}
            )

            print(response.text)
            assert response.ok
            actual_num_predictions = len(json.loads(response.text)[RESPONSE_PREDICTIONS_KEY])
            in_data = pd.read_csv(input_dataset)
            assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
        ],
    )
    def test_custom_models_with_drum_nginx_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
            nginx=True,
        ) as run:
            input_dataset = resources.datasets(framework, problem)

            # do predictions
            response = requests.post(
                run.url_server_address + "/predict/", files={"X": open(input_dataset)}
            )

            assert response.ok
            actual_num_predictions = len(json.loads(response.text)[RESPONSE_PREDICTIONS_KEY])
            in_data = pd.read_csv(input_dataset)
            assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN), (SKLEARN, BINARY, PYTHON, None)],
    )
    def test_custom_models_drum_prediction_server_response(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            resources.class_labels(framework, problem), custom_model_dir, docker
        ) as run:
            input_dataset = resources.datasets(framework, problem)

            # do predictions
            response = requests.post(
                run.url_server_address + "/predict/", files={"X": open(input_dataset)}
            )

            assert response.ok
            response_json = json.loads(response.text)
            assert isinstance(response_json, dict)
            assert RESPONSE_PREDICTIONS_KEY in response_json
            predictions_list = response_json[RESPONSE_PREDICTIONS_KEY]
            assert isinstance(predictions_list, list)
            assert len(predictions_list)
            prediction_item = predictions_list[0]
            if problem == BINARY:
                assert isinstance(prediction_item, dict)
                assert len(prediction_item) == 2
                assert all([isinstance(x, str) for x in prediction_item.keys()])
                assert all([isinstance(x, float) for x in prediction_item.values()])
            elif problem == REGRESSION:
                assert isinstance(prediction_item, float)

    @staticmethod
    def _add_weights_cmd(weights, input_csv):
        df = pd.read_csv(input_csv)
        colname = "some-colname"
        weights_data = pd.Series(np.random.randint(1, 3, len(df)))
        __keep_this_around = NamedTemporaryFile("w")
        if weights == WEIGHTS_ARGS:
            df[colname] = weights_data
            df.to_csv(__keep_this_around.name)
            return " --row-weights " + colname, __keep_this_around.name, __keep_this_around
        elif weights == WEIGHTS_CSV:
            weights_data.to_csv(__keep_this_around.name)
            return " --row-weights-csv " + __keep_this_around.name, input_csv, __keep_this_around

        return "", input_csv, __keep_this_around

    @pytest.mark.parametrize("framework", [SKLEARN_ANOMALY, RDS, SKLEARN, XGB, KERAS, PYTORCH])
    @pytest.mark.parametrize("problem", [ANOMALY, BINARY, REGRESSION])
    @pytest.mark.parametrize("docker", [DOCKER_PYTHON_SKLEARN, None])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    @pytest.mark.parametrize("use_output", [True, False])
    @pytest.mark.parametrize("nested", [True, False])
    def test_fit(
        self,
        resources,
        framework,
        problem,
        docker,
        weights,
        use_output,
        tmp_path,
        nested,
    ):
        if docker and framework != SKLEARN:
            return
        if framework == RDS:
            language = R_FIT
        else:
            language = PYTHON

        # don't try to run unsupervised problem in supervised framework and vice versa
        # TODO: check for graceful failure for these cases
        if (framework == SKLEARN_ANOMALY and problem != ANOMALY) or (
            problem == ANOMALY and framework != SKLEARN_ANOMALY
        ):
            return

        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
            is_training=True,
            nested=nested if language == PYTHON else False,  # TODO: support nested R files
        )

        input_dataset = resources.datasets(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_dataset
        )

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --code-dir {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset
        )
        if problem == ANOMALY:
            cmd += " --unsupervised"
        else:
            cmd += " --target {}".format(resources.targets(problem))

        if use_output:
            cmd += " --output {}".format(output)
        if problem == BINARY:
            cmd = _cmd_add_class_labels(cmd, resources.class_labels(framework, problem))
        if docker:
            cmd += " --docker {} ".format(docker)

        cmd += weights_cmd

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    def _create_fit_input_data_dir(
        self, get_target, get_dataset_filename, input_dir, problem, weights
    ):
        input_dir.mkdir(parents=True, exist_ok=True)

        input_dataset = get_dataset_filename(None, problem)
        df = pd.read_csv(input_dataset)

        # Training data
        with open(os.path.join(input_dir, "X.csv"), "w+") as fp:
            if problem == ANOMALY:
                feature_df = df
            else:
                feature_df = df.loc[:, df.columns != get_target(problem)]
            feature_df.to_csv(fp, index=False)

        if problem != ANOMALY:
            # Target data
            with open(os.path.join(input_dir, "y.csv"), "w+") as fp:
                target_series = df[get_target(problem)]
                target_series.to_csv(fp, index=False, header="Target")

        # Weights data
        if weights:
            df = pd.read_csv(input_dataset)
            weights_data = pd.Series(np.random.randint(1, 3, len(df)))
            with open(os.path.join(input_dir, "weights.csv"), "w+") as fp:
                weights_data.to_csv(fp, header=False)

    @pytest.mark.parametrize("framework", [SKLEARN, XGB, KERAS, SKLEARN_ANOMALY])
    @pytest.mark.parametrize("problem", [BINARY, REGRESSION, ANOMALY])
    @pytest.mark.parametrize("language", [PYTHON])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, None])
    def test_fit_sh(
        self,
        resources,
        framework,
        problem,
        language,
        weights,
        tmp_path,
    ):

        if (framework == SKLEARN_ANOMALY and problem != ANOMALY) or (
            problem == ANOMALY and framework != SKLEARN_ANOMALY
        ):
            return

        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
            is_training=True,
        )

        env = os.environ
        fit_sh = os.path.join(
            TESTS_ROOT_PATH,
            "..",
            "public_dropin_environments/{}_{}/fit.sh".format(
                language, framework if framework != SKLEARN_ANOMALY else SKLEARN
            ),
        )

        input_dir = tmp_path / "input_dir"
        self._create_fit_input_data_dir(
            resources.targets, resources.datasets, input_dir, problem, weights
        )

        output = tmp_path / "output"
        output.mkdir()

        env["CODEPATH"] = str(custom_model_dir)
        env["INPUT_DIRECTORY"] = str(input_dir)
        env["ARTIFACT_DIRECTORY"] = str(output)

        if problem == BINARY:
            labels = resources.class_labels(framework, problem)
            env["NEGATIVE_CLASS_LABEL"] = labels[0]
            env["POSITIVE_CLASS_LABEL"] = labels[1]
        else:
            if os.environ.get("NEGATIVE_CLASS_LABEL"):
                del os.environ["NEGATIVE_CLASS_LABEL"]
                del os.environ["POSITIVE_CLASS_LABEL"]

        if problem == ANOMALY:
            env["UNSUPERVISED"] = "true"
        elif os.environ.get("UNSUPERVISED"):
            del os.environ["UNSUPERVISED"]

        _exec_shell_cmd(fit_sh, "Failed cmd {}".format(fit_sh), env=env)

    def test_fit_simple(
        self,
        resources,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            SIMPLE,
            REGRESSION,
            PYTHON,
            is_training=True,
            nested=True,
        )

        input_dataset = resources.datasets(SKLEARN, REGRESSION)

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --code-dir {} --target {} --input {} --verbose".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            resources.targets(REGRESSION),
            input_dataset,
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
