import json

import pandas as pd
import pytest
import requests

from datarobot_drum.drum.common import ArgumentsOptions
from .constants import (
    BINARY,
    CODEGEN,
    DOCKER_PYTHON_SKLEARN,
    KERAS,
    MOJO,
    MULTI_ARTIFACT,
    NO_CUSTOM,
    POJO,
    PYPMML,
    PYTHON,
    PYTHON_LOAD_MODEL,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    PYTORCH,
    R,
    RDS,
    REGRESSION,
    REGRESSION_INFERENCE,
    RESPONSE_PREDICTIONS_KEY,
    SKLEARN,
    XGB,
)
from .drum_server_utils import DrumServerRun
from .utils import _cmd_add_class_labels, _create_custom_model_dir, _exec_shell_cmd


class TestInference:
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

        cmd = "{} score --code-dir {} --input {} --output {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
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
            resources.target_types(problem),
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
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
