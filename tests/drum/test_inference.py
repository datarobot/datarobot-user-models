"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
from tempfile import NamedTemporaryFile
from textwrap import dedent

import io
import os
import pandas as pd
import pyarrow
import pytest
import requests
import scipy
from scipy.sparse import csr_matrix

from datarobot_drum.drum.enum import (
    X_TRANSFORM_KEY,
    Y_TRANSFORM_KEY,
    MODEL_CONFIG_FILENAME,
    PredictionServerMimetypes,
    ModelInfoKeys,
    ArgumentsOptions,
    TargetType,
)
from datarobot_drum.drum.description import version as drum_version
from datarobot_drum.resource.transform_helpers import (
    read_arrow_payload,
    read_mtx_payload,
    read_csv_payload,
    parse_multi_part_response,
)
from .constants import (
    BINARY,
    CODEGEN,
    DOCKER_PYTHON_SKLEARN,
    KERAS,
    MOJO,
    MULTI_ARTIFACT,
    MULTICLASS,
    MULTICLASS_BINARY,
    NO_CUSTOM,
    ONNX,
    POJO,
    PYPMML,
    PYTHON,
    PYTHON_LOAD_MODEL,
    PYTHON_PREDICT_SPARSE,
    PYTHON_TRANSFORM,
    PYTHON_TRANSFORM_DENSE,
    PYTHON_TRANSFORM_SPARSE,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    PYTORCH,
    R,
    R_FAIL_CLASSIFICATION_VALIDATION_HOOKS,
    R_PREDICT_SPARSE,
    RDS,
    RDS_SPARSE,
    REGRESSION,
    REGRESSION_INFERENCE,
    RESPONSE_PREDICTIONS_KEY,
    SKLEARN,
    SKLEARN_TRANSFORM,
    SKLEARN_TRANSFORM_DENSE,
    SPARSE,
    SPARSE_TRANSFORM,
    TRANSFORM,
    XGB,
    JULIA,
    MLJ,
    R_TRANSFORM_WITH_Y,
    R_TRANSFORM_SPARSE_INPUT,
    R_TRANSFORM_SPARSE_OUTPUT,
    R_VALIDATE_SPARSE_ESTIMATOR,
)
from datarobot_drum.resource.drum_server_utils import DrumServerRun
from datarobot_drum.resource.utils import (
    _cmd_add_class_labels,
    _create_custom_model_dir,
    _exec_shell_cmd,
)

from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars
from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils


class TestInference:
    @pytest.fixture
    def temp_file(self):
        with NamedTemporaryFile() as f:
            yield f

    @pytest.mark.parametrize(
        "framework, problem, language, docker, use_labels_file",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN, False),
            (SKLEARN, MULTICLASS, PYTHON, DOCKER_PYTHON_SKLEARN, False),
            (SKLEARN, MULTICLASS, PYTHON, DOCKER_PYTHON_SKLEARN, True),
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
        capitalize_artifact_extension=False,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
            capitalize_artifact_extension=capitalize_artifact_extension,
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
        if problem == SPARSE:
            cmd += " --sparse-column-file {}".format(input_dataset.replace(".mtx", ".columns"))
        if resources.target_types(problem) in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, problem),
                target_type=resources.target_types(problem),
                multiclass_label_file=temp_file if use_labels_file else None,
            )
        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = resources.input_data(framework, problem)
        out_data = pd.read_csv(output)
        assert in_data.shape[0] == out_data.shape[0]

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),],
    )
    @pytest.mark.parametrize("pass_args_as_env_vars", [False])
    def test_custom_models_with_drum_prediction_server(
        self, resources, framework, problem, language, docker, pass_args_as_env_vars, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        unset_drum_supported_env_vars()
        with DrumServerRun(
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
            pass_args_as_env_vars=pass_args_as_env_vars,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": open(input_dataset)}},
                    {"data": open(input_dataset, "rb")},
                ]:
                    if problem == SPARSE:
                        if "data" in post_args:
                            continue
                        input_colnames = input_dataset.replace(".mtx", ".columns")
                        post_args["files"]["X.colnames"] = open(input_colnames)
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    print(response.text)
                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = resources.input_data(framework, problem)
                    assert in_data.shape[0] == actual_num_predictions
            # test model info
            response = requests.get(run.url_server_address + "/info/")

            assert response.ok
            response_dict = response.json()
            for key in ModelInfoKeys.REQUIRED:
                assert key in response_dict
            assert response_dict[ModelInfoKeys.TARGET_TYPE] == resources.target_types(problem)
            # Don't verify code dir when running with Docker.
            # Local code dir is mapped into user-defined location within docker.
            if docker is None:
                assert response_dict[ModelInfoKeys.CODE_DIR] == str(custom_model_dir)
            assert response_dict[ModelInfoKeys.DRUM_SERVER] == "flask"
            assert response_dict[ModelInfoKeys.DRUM_VERSION] == drum_version

            if resources.target_types(problem) == TargetType.BINARY.value:
                assert ModelInfoKeys.POSITIVE_CLASS_LABEL in response_dict
                assert ModelInfoKeys.NEGATIVE_CLASS_LABEL in response_dict
            elif resources.target_types(problem) == TargetType.MULTICLASS.value:
                assert ModelInfoKeys.CLASS_LABELS in response_dict

            if framework == SKLEARN and problem == REGRESSION:
                assert ModelInfoKeys.MODEL_METADATA in response_dict

        unset_drum_supported_env_vars()

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),],
    )
    @pytest.mark.parametrize("pass_args_as_env_vars", [True])
    def test_custom_models_with_drum_prediction_server_with_args_passed_as_env_vars(
        self, resources, framework, problem, language, docker, pass_args_as_env_vars, tmp_path,
    ):
        self.test_custom_models_with_drum_prediction_server(
            resources, framework, problem, language, docker, pass_args_as_env_vars, tmp_path
        )

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),],
    )
    def test_custom_models_with_drum_nginx_prediction_server(
        self, resources, framework, problem, language, docker, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
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
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": open(input_dataset)}},
                    {"data": open(input_dataset, "rb")},
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = pd.read_csv(input_dataset)
                    assert in_data.shape[0] == actual_num_predictions

            # test model info
            response = requests.get(run.url_server_address + "/info/")

            assert response.ok
            response_dict = response.json()
            for key in ModelInfoKeys.REQUIRED:
                assert key in response_dict
            assert response_dict[ModelInfoKeys.TARGET_TYPE] == resources.target_types(problem)
            assert response_dict[ModelInfoKeys.DRUM_SERVER] == "nginx + uwsgi"
            assert response_dict[ModelInfoKeys.DRUM_VERSION] == drum_version

            assert ModelInfoKeys.MODEL_METADATA in response_dict

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM, DOCKER_PYTHON_SKLEARN),],
    )
    def test_custom_transforms_with_drum_nginx_prediction_server(
        self, resources, framework, problem, language, docker, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
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
                run.url_server_address + "/transform/", files={"X": open(input_dataset)}
            )

            assert response.ok

            in_data = pd.read_csv(input_dataset)

            parsed_response = parse_multi_part_response(response)

            transformed_mat = read_mtx_payload(parsed_response, X_TRANSFORM_KEY)
            actual_num_predictions = transformed_mat.shape[0]
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
        self, resources, framework, problem, language, docker, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        with DrumServerRun(
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
        ) as run:
            input_dataset = resources.datasets(framework, problem)

            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": open(input_dataset)}},
                    {"data": open(input_dataset, "rb")},
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

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
                        assert len(prediction_item) == len(
                            resources.class_labels(framework, problem)
                        )
                        assert all([isinstance(x, str) for x in prediction_item.keys()])
                        assert all([isinstance(x, float) for x in prediction_item.values()])
                    elif problem == REGRESSION:
                        assert isinstance(prediction_item, float)

    @pytest.mark.parametrize(
        "framework, problem, language, supported_payload_formats",
        [
            (SKLEARN, REGRESSION, PYTHON, {"csv": None, "mtx": None, "arrow": pyarrow.__version__}),
            (RDS, REGRESSION, R, {"csv": None, "mtx": None}),
            (CODEGEN, REGRESSION, NO_CUSTOM, {"csv": None}),
        ],
    )
    def test_predictors_supported_payload_formats(
        self, resources, framework, problem, language, supported_payload_formats, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
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
        "framework, problem, language", [(SKLEARN, REGRESSION_INFERENCE, PYTHON),],
    )
    # Don't run this test case with nginx as it is still running from the prev test case.
    @pytest.mark.parametrize("nginx", [False])
    def test_predictions_python_arrow_mtx(
        self, resources, framework, problem, language, nginx, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        with DrumServerRun(
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            nginx=nginx,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            df = pd.read_csv(input_dataset)
            arrow_dataset_buf = pyarrow.ipc.serialize_pandas(df, preserve_index=False).to_pybytes()

            sink = io.BytesIO()
            scipy.io.mmwrite(sink, scipy.sparse.csr_matrix(df.values))
            mtx_dataset_buf = sink.getvalue()

            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": ("X.arrow", arrow_dataset_buf)}},
                    {"files": {"X": ("X.mtx", mtx_dataset_buf)}},
                    {
                        "data": arrow_dataset_buf,
                        "headers": {
                            "Content-Type": "{};".format(
                                PredictionServerMimetypes.APPLICATION_X_APACHE_ARROW_STREAM
                            )
                        },
                    },
                    {
                        "data": mtx_dataset_buf,
                        "headers": {
                            "Content-Type": "{};".format(PredictionServerMimetypes.TEXT_MTX)
                        },
                    },
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = pd.read_csv(input_dataset)
                    assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language", [(RDS_SPARSE, REGRESSION, R_VALIDATE_SPARSE_ESTIMATOR),],
    )
    @pytest.mark.parametrize("nginx", [False, True])
    def test_predictions_r_mtx(
        self, resources, framework, problem, language, nginx, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        with DrumServerRun(
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            nginx=nginx,
        ) as run:
            input_dataset = resources.datasets(framework, SPARSE)

            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": ("X.mtx", open(input_dataset))}},
                    {
                        "data": open(input_dataset, "rb"),
                        "headers": {
                            "Content-Type": "{};".format(PredictionServerMimetypes.TEXT_MTX)
                        },
                    },
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = StructuredInputReadUtils.read_structured_input_file_as_df(
                        input_dataset
                    )
                    assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, language, target_type",
        [(None, R_FAIL_CLASSIFICATION_VALIDATION_HOOKS, BINARY,)],
    )
    def test_classification_validation_fails(
        self, resources, framework, language, target_type, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(resources, tmp_path, framework, None, language,)

        input_dataset = resources.datasets(framework, BINARY)

        cmd = "{} score --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, target_type
        )

        if resources.target_types(target_type) in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, target_type),
                target_type=resources.target_types(target_type),
            )

        _, stdo, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        assert "Your prediction probabilities do not add up to 1." in str(stdo)
