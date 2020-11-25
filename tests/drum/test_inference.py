import json
from tempfile import NamedTemporaryFile

import pandas as pd
import pyarrow
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
    MULTICLASS,
    NO_CUSTOM,
    POJO,
    PYPMML,
    PYTHON,
    PYTHON_LOAD_MODEL,
    PYTHON_TRANSFORM,
    PYTHON_TRANSFORM_DENSE,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    PYTORCH,
    R,
    RDS,
    REGRESSION,
    REGRESSION_INFERENCE,
    RESPONSE_PREDICTIONS_KEY,
    RESPONSE_TRANSFORM_KEY,
    SKLEARN,
    SKLEARN_TRANSFORM,
    SKLEARN_TRANSFORM_DENSE,
    TRANSFORM,
    XGB,
)
from .drum_server_utils import DrumServerRun
from .utils import _cmd_add_class_labels, _create_custom_model_dir, _exec_shell_cmd


class TestInference:
    @pytest.fixture
    def temp_file(self):
        with NamedTemporaryFile() as f:
            yield f

    @pytest.mark.parametrize(
        "framework, problem, language, docker, use_labels_file",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None, False),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN, False),
            (SKLEARN, BINARY, PYTHON, None, False),
            (SKLEARN, MULTICLASS, PYTHON, None, False),
            (SKLEARN, MULTICLASS, PYTHON, None, True),
            (SKLEARN, MULTICLASS, PYTHON, DOCKER_PYTHON_SKLEARN, False),
            (SKLEARN, MULTICLASS, PYTHON, DOCKER_PYTHON_SKLEARN, True),
            (KERAS, REGRESSION, PYTHON, None, False),
            (KERAS, BINARY, PYTHON, None, False),
            (KERAS, MULTICLASS, PYTHON, None, False),
            (XGB, REGRESSION, PYTHON, None, False),
            (XGB, BINARY, PYTHON, None, False),
            (XGB, BINARY, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None, False),
            (XGB, MULTICLASS, PYTHON, None, False),
            (XGB, MULTICLASS, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None, False),
            (PYTORCH, REGRESSION, PYTHON, None, False),
            (PYTORCH, BINARY, PYTHON, None, False),
            (PYTORCH, MULTICLASS, PYTHON, None, False),
            (RDS, REGRESSION, R, None, False),
            (RDS, BINARY, R, None, False),
            (RDS, MULTICLASS, R, None, False),
            (CODEGEN, REGRESSION, NO_CUSTOM, None, False),
            (CODEGEN, BINARY, NO_CUSTOM, None, False),
            (CODEGEN, MULTICLASS, NO_CUSTOM, None, False),
            (POJO, REGRESSION, NO_CUSTOM, None, False),
            (POJO, BINARY, NO_CUSTOM, None, False),
            (POJO, MULTICLASS, NO_CUSTOM, None, False),
            (MOJO, REGRESSION, NO_CUSTOM, None, False),
            (MOJO, BINARY, NO_CUSTOM, None, False),
            (MOJO, MULTICLASS, NO_CUSTOM, None, False),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None, False),
            (PYPMML, REGRESSION, NO_CUSTOM, None, False),
            (PYPMML, BINARY, NO_CUSTOM, None, False),
            (PYPMML, MULTICLASS, NO_CUSTOM, None, False),
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
        use_labels_file,
        temp_file,
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

        cmd = '{} score --code-dir {} --input "{}" --output {} --target-type {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )
        if problem in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, problem),
                multiclass_label_file=temp_file if use_labels_file else None,
            )
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
            (SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM, None),
            (SKLEARN_TRANSFORM_DENSE, TRANSFORM, PYTHON_TRANSFORM, None),
            (SKLEARN, TRANSFORM, PYTHON_TRANSFORM_DENSE, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, MULTICLASS, PYTHON, None),
            (KERAS, REGRESSION, PYTHON, None),
            (KERAS, BINARY, PYTHON, None),
            (KERAS, MULTICLASS, PYTHON, None),
            (XGB, REGRESSION, PYTHON, None),
            (XGB, BINARY, PYTHON, None),
            (XGB, MULTICLASS, PYTHON, None),
            (PYTORCH, REGRESSION, PYTHON, None),
            (PYTORCH, BINARY, PYTHON, None),
            (PYTORCH, MULTICLASS, PYTHON, None),
            (RDS, REGRESSION, R, None),
            (RDS, BINARY, R, None),
            (RDS, MULTICLASS, R, None),
            (CODEGEN, REGRESSION, NO_CUSTOM, None),
            (CODEGEN, BINARY, NO_CUSTOM, None),
            (CODEGEN, MULTICLASS, NO_CUSTOM, None),
            (MOJO, REGRESSION, NO_CUSTOM, None),
            (MOJO, BINARY, NO_CUSTOM, None),
            (MOJO, MULTICLASS, NO_CUSTOM, None),
            (POJO, REGRESSION, NO_CUSTOM, None),
            (POJO, BINARY, NO_CUSTOM, None),
            (POJO, MULTICLASS, NO_CUSTOM, None),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None),
            (PYPMML, REGRESSION, NO_CUSTOM, None),
            (PYPMML, BINARY, NO_CUSTOM, None),
            (PYPMML, MULTICLASS, NO_CUSTOM, None),
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
            endpoint = "/transform/" if problem == TRANSFORM else "/predict/"
            # do predictions
            response = requests.post(
                run.url_server_address + endpoint, files={"X": open(input_dataset)}
            )

            print(response.text)
            assert response.ok
            response_key = (
                RESPONSE_TRANSFORM_KEY if problem == TRANSFORM else RESPONSE_PREDICTIONS_KEY
            )
            actual_num_predictions = len(json.loads(response.text)[response_key])
            in_data = pd.read_csv(input_dataset)
            assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, TRANSFORM, PYTHON_TRANSFORM, DOCKER_PYTHON_SKLEARN),
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
            endpoint = "/transform/" if problem == TRANSFORM else "/predict/"

            # do predictions
            response = requests.post(
                run.url_server_address + endpoint, files={"X": open(input_dataset)}
            )

            assert response.ok
            response_key = (
                RESPONSE_TRANSFORM_KEY if problem == TRANSFORM else RESPONSE_PREDICTIONS_KEY
            )
            actual_num_predictions = len(json.loads(response.text)[response_key])
            in_data = pd.read_csv(input_dataset)
            assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, MULTICLASS, PYTHON, None),
        ],
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
            if problem in [BINARY, MULTICLASS]:
                assert isinstance(prediction_item, dict)
                assert len(prediction_item) == len(resources.class_labels(framework, problem))
                assert all([isinstance(x, str) for x in prediction_item.keys()])
                assert all([isinstance(x, float) for x in prediction_item.values()])
            elif problem == REGRESSION:
                assert isinstance(prediction_item, float)

    @pytest.mark.parametrize(
        "framework, problem, language, supported_payload_formats",
        [
            (SKLEARN, REGRESSION, PYTHON, {"csv": None, "arrow": "0.14.1"}),
            (RDS, REGRESSION, R, {"csv": None}),
            (CODEGEN, REGRESSION, NO_CUSTOM, {"csv": None}),
        ],
    )
    def test_predictors_supported_payload_formats(
        self,
        resources,
        framework,
        problem,
        language,
        supported_payload_formats,
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
        ) as run:
            response = requests.get(run.url_server_address + "/capabilities/")

            assert response.ok
            assert response.json() == {"supported_payload_formats": supported_payload_formats}

    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (SKLEARN, REGRESSION, PYTHON),
        ],
    )
    @pytest.mark.parametrize("nginx", [False, True])
    def test_predictions_using_arrow(
        self,
        resources,
        framework,
        problem,
        language,
        nginx,
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
            nginx=nginx,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            df = pd.read_csv(input_dataset)
            dataset_buf = pyarrow.ipc.serialize_pandas(df, preserve_index=False).to_pybytes()

            # do predictions
            response = requests.post(
                run.url_server_address + "/predict/", files={"X": ("X.arrow", dataset_buf)}
            )

            assert response.ok
            actual_num_predictions = len(json.loads(response.text)[RESPONSE_PREDICTIONS_KEY])
            in_data = pd.read_csv(input_dataset)
            assert in_data.shape[0] == actual_num_predictions
