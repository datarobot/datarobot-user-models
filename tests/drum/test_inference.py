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
    EnvVarNames,
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
    POJO,
    PYPMML,
    PYTHON,
    PYTHON_LOAD_MODEL,
    PYTHON_PREDICT_SPARSE,
    PYTHON_TRANSFORM_NO_Y,
    PYTHON_TRANSFORM_NO_Y_DENSE,
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
    R_TRANSFORM,
    R_TRANSFORM_SPARSE_INPUT,
    R_TRANSFORM_SPARSE_OUTPUT,
    R_VALIDATE_SPARSE_ESTIMATOR,
    UNSTRUCTURED,
)
from datarobot_drum.resource.drum_server_utils import DrumServerRun
from datarobot_drum.resource.utils import (
    _cmd_add_class_labels,
    _create_custom_model_dir,
    _exec_shell_cmd,
)

from datarobot_drum.drum.utils import StructuredInputReadUtils, unset_drum_supported_env_vars


class TestInference:
    @pytest.fixture
    def temp_file(self):
        with NamedTemporaryFile() as f:
            yield f

    @pytest.mark.parametrize(
        "framework, problem, language, docker, use_labels_file",
        [
            (SKLEARN, SPARSE, PYTHON_PREDICT_SPARSE, None, False),
            (RDS_SPARSE, SPARSE, R_PREDICT_SPARSE, None, False),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None, False),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN, False),
            (SKLEARN, BINARY, PYTHON, None, False),
            (SKLEARN, MULTICLASS, PYTHON, None, False),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, None, False),
            (SKLEARN, MULTICLASS, PYTHON, None, True),
            (SKLEARN, MULTICLASS, PYTHON, DOCKER_PYTHON_SKLEARN, False),
            (SKLEARN, MULTICLASS, PYTHON, DOCKER_PYTHON_SKLEARN, True),
            (KERAS, REGRESSION, PYTHON, None, False),
            (KERAS, BINARY, PYTHON, None, False),
            (KERAS, MULTICLASS, PYTHON, None, False),
            (KERAS, MULTICLASS_BINARY, PYTHON, None, False),
            (XGB, REGRESSION, PYTHON, None, False),
            (XGB, BINARY, PYTHON, None, False),
            (XGB, BINARY, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None, False),
            (XGB, MULTICLASS, PYTHON, None, False),
            (XGB, MULTICLASS, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None, False),
            (XGB, MULTICLASS_BINARY, PYTHON, None, False),
            (PYTORCH, REGRESSION, PYTHON, None, False),
            (PYTORCH, BINARY, PYTHON, None, False),
            (PYTORCH, MULTICLASS, PYTHON, None, False),
            (PYTORCH, MULTICLASS_BINARY, PYTHON, None, False),
            (RDS, REGRESSION, R, None, False),
            (RDS, BINARY, R, None, False),
            (RDS, MULTICLASS, R, None, False),
            (RDS, MULTICLASS_BINARY, R, None, False),
            (CODEGEN, REGRESSION, NO_CUSTOM, None, False),
            (CODEGEN, BINARY, NO_CUSTOM, None, False),
            (CODEGEN, MULTICLASS, NO_CUSTOM, None, False),
            (CODEGEN, MULTICLASS_BINARY, NO_CUSTOM, None, False),
            (POJO, REGRESSION, NO_CUSTOM, None, False),
            (POJO, BINARY, NO_CUSTOM, None, False),
            (POJO, MULTICLASS, NO_CUSTOM, None, False),
            (POJO, MULTICLASS_BINARY, NO_CUSTOM, None, False),
            (MOJO, REGRESSION, NO_CUSTOM, None, False),
            (MOJO, BINARY, NO_CUSTOM, None, False),
            (MOJO, MULTICLASS, NO_CUSTOM, None, False),
            (MOJO, MULTICLASS_BINARY, NO_CUSTOM, None, False),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None, False),
            (PYPMML, REGRESSION, NO_CUSTOM, None, False),
            (PYPMML, BINARY, NO_CUSTOM, None, False),
            (PYPMML, MULTICLASS, NO_CUSTOM, None, False),
            (PYPMML, MULTICLASS_BINARY, NO_CUSTOM, None, False),
            (MLJ, REGRESSION, JULIA, None, False),
            (MLJ, BINARY, JULIA, None, False),
            (MLJ, MULTICLASS, JULIA, None, False),
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
        "framework, problem, language, docker, use_labels_file",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None, False),
            (KERAS, REGRESSION, PYTHON, None, False),
            (XGB, REGRESSION, PYTHON, None, False),
            (PYTORCH, REGRESSION, PYTHON, None, False),
            (RDS, REGRESSION, R, None, False),
            (CODEGEN, REGRESSION, NO_CUSTOM, None, False),
            # POJO is not a relevant case. POJO artifacet is a `.java` file which allowed to be only lowercase
            (MOJO, REGRESSION, NO_CUSTOM, None, False),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None, False),
            (PYPMML, REGRESSION, NO_CUSTOM, None, False),
            (MLJ, REGRESSION, JULIA, None, False),
        ],
    )
    def test_custom_models_with_drum_capitalize_artifact_extensions(
        self, resources, framework, problem, language, docker, tmp_path, use_labels_file, temp_file,
    ):
        self.test_custom_models_with_drum(
            resources,
            framework,
            problem,
            language,
            docker,
            tmp_path,
            use_labels_file,
            temp_file,
            capitalize_artifact_extension=True,
        )
        print(os.listdir(os.path.join(tmp_path, "custom_model")))

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, SPARSE, PYTHON_PREDICT_SPARSE, None),
            (RDS_SPARSE, SPARSE, R_PREDICT_SPARSE, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, REGRESSION, PYTHON, None),
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, MULTICLASS, PYTHON, None),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, None),
            (KERAS, REGRESSION, PYTHON, None),
            (KERAS, BINARY, PYTHON, None),
            (KERAS, MULTICLASS, PYTHON, None),
            (KERAS, MULTICLASS_BINARY, PYTHON, None),
            (XGB, REGRESSION, PYTHON, None),
            (XGB, BINARY, PYTHON, None),
            (XGB, MULTICLASS, PYTHON, None),
            (XGB, MULTICLASS_BINARY, PYTHON, None),
            (PYTORCH, REGRESSION, PYTHON, None),
            (PYTORCH, BINARY, PYTHON, None),
            (PYTORCH, MULTICLASS, PYTHON, None),
            (PYTORCH, MULTICLASS_BINARY, PYTHON, None),
            (RDS, REGRESSION, R, None),
            (RDS, BINARY, R, None),
            (RDS, MULTICLASS, R, None),
            (RDS, MULTICLASS_BINARY, R, None),
            (CODEGEN, REGRESSION, NO_CUSTOM, None),
            (CODEGEN, BINARY, NO_CUSTOM, None),
            (CODEGEN, MULTICLASS, NO_CUSTOM, None),
            (CODEGEN, MULTICLASS_BINARY, NO_CUSTOM, None),
            (MOJO, REGRESSION, NO_CUSTOM, None),
            (MOJO, BINARY, NO_CUSTOM, None),
            (MOJO, MULTICLASS, NO_CUSTOM, None),
            (MOJO, MULTICLASS_BINARY, NO_CUSTOM, None),
            (POJO, REGRESSION, NO_CUSTOM, None),
            (POJO, BINARY, NO_CUSTOM, None),
            (POJO, MULTICLASS, NO_CUSTOM, None),
            (POJO, MULTICLASS_BINARY, NO_CUSTOM, None),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None),
            (PYPMML, REGRESSION, NO_CUSTOM, None),
            (PYPMML, BINARY, NO_CUSTOM, None),
            (PYPMML, MULTICLASS, NO_CUSTOM, None),
            (PYPMML, MULTICLASS_BINARY, NO_CUSTOM, None),
            (MLJ, BINARY, JULIA, None),
            (MLJ, REGRESSION, JULIA, None),
            (MLJ, MULTICLASS, JULIA, None),
        ],
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

    @pytest.mark.sequential
    @pytest.mark.parametrize(
        "problem, class_labels",
        [(REGRESSION, None), (BINARY, ["no", "yes"]), (UNSTRUCTURED, None),],
    )
    # current test case returns hardcoded predictions:
    # - for regression: [1, 2, .., N samples]
    # - for binary: [{0.3, 0,7}, {0.3, 0.7}, ...]
    # - for unstructured: "10"
    def test_custom_model_with_custom_java_predictor(
        self, resources, class_labels, problem,
    ):
        unset_drum_supported_env_vars()
        cur_file_dir = os.path.dirname(os.path.abspath(__file__))
        # have to point model dir to a folder with jar, so drum could detect the language
        model_dir = os.path.join(cur_file_dir, "custom_java_predictor")
        os.environ[
            EnvVarNames.DRUM_JAVA_CUSTOM_PREDICTOR_CLASS
        ] = "com.datarobot.test.TestCustomPredictor"
        os.environ[EnvVarNames.DRUM_JAVA_CUSTOM_CLASS_PATH] = os.path.join(model_dir, "*")
        with DrumServerRun(resources.target_types(problem), class_labels, model_dir,) as run:
            input_dataset = resources.datasets(None, problem)
            # do predictions
            post_args = {"data": open(input_dataset, "rb")}
            if problem == UNSTRUCTURED:
                response = requests.post(
                    run.url_server_address + "/predictUnstructured", **post_args
                )
            else:
                response = requests.post(run.url_server_address + "/predict", **post_args)
                print(response.text)
                assert response.ok
                predictions = json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                actual_num_predictions = len(predictions)
                in_data = pd.read_csv(input_dataset)
                assert in_data.shape[0] == actual_num_predictions
            if problem == REGRESSION:
                assert list(range(1, actual_num_predictions + 1)) == predictions
            elif problem == UNSTRUCTURED:
                assert response.content.decode("UTF-8") == "10"
            else:
                single_prediction = {"yes": 0.7, "no": 0.3}
                assert [single_prediction] * actual_num_predictions == predictions

        unset_drum_supported_env_vars()

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, MULTICLASS, PYTHON, None),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, None),
        ],
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
        "framework, problem, language, docker, use_arrow",
        [
            # The following 3 tests produce Y transform values and are being temporarily removed until y transform
            # validation is added
            # (SKLEARN_TRANSFORM_DENSE, TRANSFORM, PYTHON_TRANSFORM_DENSE, None, True),
            # (SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM, None, False),
            # (SKLEARN_TRANSFORM_DENSE, TRANSFORM, PYTHON_TRANSFORM_DENSE, None, False),
            (SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM_NO_Y, None, True),
            (SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM_NO_Y, None, False),
            (SKLEARN_TRANSFORM_DENSE, TRANSFORM, PYTHON_TRANSFORM_NO_Y_DENSE, None, False),
            (R_TRANSFORM, TRANSFORM, R_TRANSFORM, None, False),
            (SKLEARN_TRANSFORM, SPARSE_TRANSFORM, PYTHON_TRANSFORM_SPARSE, None, False),
            (R_TRANSFORM_SPARSE_INPUT, SPARSE_TRANSFORM, R_TRANSFORM_SPARSE_INPUT, None, False),
            (R_TRANSFORM_SPARSE_OUTPUT, TRANSFORM, R_TRANSFORM_SPARSE_OUTPUT, None, False),
        ],
    )
    @pytest.mark.parametrize("pass_target", [False])
    def test_custom_transform_server(
        self, resources, framework, problem, language, docker, tmp_path, use_arrow, pass_target,
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
            in_data = resources.input_data(framework, problem)

            files = {"X": open(input_dataset)}
            if pass_target:
                target_dataset = resources.targets(problem)
                files["y"] = open(target_dataset)
            if input_dataset.endswith(".mtx"):
                files["X.colnames"] = open(input_dataset.replace(".mtx", ".columns"))

            if use_arrow:
                files["arrow_version"] = ".2"

            response = requests.post(run.url_server_address + "/transform/", files=files)
            assert response.ok

            parsed_response = parse_multi_part_response(response)

            if framework in [SKLEARN_TRANSFORM_DENSE, R_TRANSFORM, R_TRANSFORM_SPARSE_INPUT]:
                if use_arrow:
                    transformed_out = read_arrow_payload(parsed_response, X_TRANSFORM_KEY)
                    if pass_target:
                        target_out = read_arrow_payload(parsed_response, Y_TRANSFORM_KEY)
                    assert parsed_response["X.format"] == "arrow"
                    if pass_target:
                        assert parsed_response["y.format"] == "arrow"
                else:
                    transformed_out = read_csv_payload(parsed_response, X_TRANSFORM_KEY)
                    if pass_target:
                        target_out = read_csv_payload(parsed_response, Y_TRANSFORM_KEY)
                    assert parsed_response["X.format"] == "csv"
                    if pass_target:
                        assert parsed_response["y.format"] == "csv"
                actual_num_predictions = transformed_out.shape[0]
            else:
                transformed_out = read_mtx_payload(parsed_response, X_TRANSFORM_KEY)
                colnames = parsed_response["X.colnames"].decode("utf-8").split("\n")
                assert len(colnames) == transformed_out.shape[1]
                assert colnames == [f"feature_{i}" for i in range(transformed_out.shape[1])]
                if pass_target:
                    # this shouldn't be sparse even though features are
                    if use_arrow:
                        target_out = read_arrow_payload(parsed_response, Y_TRANSFORM_KEY)
                        if pass_target:
                            assert parsed_response["y.format"] == "arrow"
                    else:
                        target_out = read_csv_payload(parsed_response, Y_TRANSFORM_KEY)
                        if pass_target:
                            assert parsed_response["y.format"] == "csv"
                actual_num_predictions = transformed_out.shape[0]
                assert parsed_response["X.format"] == "sparse"

            if framework == R_TRANSFORM_SPARSE_INPUT:
                assert type(transformed_out) == pd.DataFrame
                assert transformed_out.shape[1] == 162
            elif problem == SPARSE_TRANSFORM:
                assert type(transformed_out) == csr_matrix
                assert transformed_out.shape[1] == 162
            elif framework == SKLEARN_TRANSFORM:
                assert type(transformed_out) == csr_matrix
                assert transformed_out.shape[1] == 714
            elif framework == SKLEARN_TRANSFORM_DENSE:
                assert type(transformed_out) == pd.DataFrame
                assert transformed_out.shape[1] == 10
            elif framework == R_TRANSFORM:
                assert type(transformed_out) == pd.DataFrame
                assert transformed_out.shape[1] == 80

            if pass_target:
                assert all(pd.read_csv(target_dataset) == target_out)
            assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM_NO_Y, DOCKER_PYTHON_SKLEARN),],
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
    # Don't run this test case with nginx as it still running from the prev test case.
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

    @pytest.fixture
    def inference_metadata_yaml(self):
        return dedent(
            """
            name: custom_model
            type: inference
            targetType: {target_type}
            inferenceModel:
              targetName: this field is not used for inference
            """
        )

    @pytest.mark.parametrize(
        "framework, problem, language, use_labels_file",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, False),
            (SKLEARN, BINARY, PYTHON, False),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, False),
            (SKLEARN, MULTICLASS, PYTHON, False),
            (SKLEARN, MULTICLASS, PYTHON, True),
        ],
    )
    def test_custom_models_with_drum_with_model_yaml_labels(
        self,
        resources,
        framework,
        problem,
        language,
        tmp_path,
        use_labels_file,
        temp_file,
        inference_metadata_yaml,
    ):
        config_yaml = inference_metadata_yaml.format(target_type=resources.target_types(problem))
        resolved_target_type = resources.target_types(problem)
        if resolved_target_type in [BINARY, MULTICLASS]:
            labels = resources.class_labels(framework, problem)
            if resolved_target_type == BINARY:
                config_yaml += "\n  positiveClassLabel: {}\n  negativeClassLabel: {}".format(
                    *labels
                )
            else:
                if use_labels_file:
                    for label in labels:
                        temp_file.write(label.encode("utf-8"))
                        temp_file.write("\n".encode("utf-8"))
                    temp_file.flush()
                    config_yaml += "\n  classLabelsFile: {}".format(temp_file.name)
                else:
                    config_yaml += "\n  classLabels:"
                    for label in labels:
                        config_yaml += "\n    - {}".format(label)

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        with open(os.path.join(custom_model_dir, MODEL_CONFIG_FILENAME), mode="w") as f:
            f.write(config_yaml)

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {}'.format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output,
        )

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = pd.read_csv(input_dataset)
        out_data = pd.read_csv(output)
        assert in_data.shape[0] == out_data.shape[0]

    @pytest.mark.parametrize(
        "framework, problem, language, use_labels_file, save_yaml, add_to_cmd",
        [
            # No yaml, no class labels in cli arguments
            (SKLEARN, BINARY, PYTHON, False, False, "--target-type binary"),
            # labels in cli and yaml are provided but don't match
            (
                SKLEARN,
                BINARY,
                PYTHON,
                False,
                True,
                " --positive-class-label random1 --negative-class-label random2",
            ),
            # No yaml, no class labels in cli arguments
            (SKLEARN, MULTICLASS, PYTHON, False, False, "--target-type multiclass"),
            # labels in cli and yaml are provided but don't match
            (SKLEARN, MULTICLASS, PYTHON, False, True, "--class-labels a b c"),
            (SKLEARN, MULTICLASS, PYTHON, True, True, "--class-labels a b c"),
        ],
    )
    def test_custom_models_with_drum_with_model_yaml_labels_negative(
        self,
        resources,
        framework,
        problem,
        language,
        tmp_path,
        use_labels_file,
        save_yaml,
        add_to_cmd,
        temp_file,
        inference_metadata_yaml,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {}'.format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output,
        )

        if add_to_cmd is not None:
            cmd += " " + add_to_cmd

        config_yaml = inference_metadata_yaml.format(target_type=resources.target_types(problem))
        resolved_target_type = resources.target_types(problem)

        if resolved_target_type in [BINARY, MULTICLASS]:
            labels = resources.class_labels(framework, problem)
            if resolved_target_type == BINARY:
                config_yaml += "\n  positiveClassLabel: {}\n  negativeClassLabel: {}".format(
                    *labels
                )
            else:
                if use_labels_file:
                    for label in labels:
                        temp_file.write(label.encode("utf-8"))
                        temp_file.write("\n".encode("utf-8"))
                    temp_file.flush()
                    config_yaml += "\n  classLabelsFile: {}".format(temp_file.name)
                else:
                    config_yaml += "\n  classLabels:"
                    for label in labels:
                        config_yaml += "\n    - {}".format(label)

        if save_yaml:
            with open(os.path.join(custom_model_dir, MODEL_CONFIG_FILENAME), mode="w") as f:
                f.write(config_yaml)

        p, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        stdo_stde = str(stdo) + str(stde)
        case_1 = (
            "Positive/negative class labels are missing. They must be provided with either one: --positive-class-label/--negative-class-label arguments, environment variables, model config file."
            in stdo_stde
        )
        case_2 = (
            "Positive/negative class labels provided with command arguments or environment variable don't match values from model config file. Use either one of them or make them match."
            in stdo_stde
        )
        case_3 = (
            "Class labels are missing. They must be provided with either one: --class-labels/--class-labels-file arguments, environment variables, model config file."
            in stdo_stde
        )
        case_4 = (
            "Class labels provided with command arguments or environment variable don't match values from model config file. Use either one of them or make them match."
            in stdo_stde
        )
        assert any([case_1, case_2, case_3, case_4])

    @pytest.mark.parametrize(
        "framework, problem, language, target_type_test_case",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, "no target type, no yaml"),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, "no target type in yaml"),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, "target types don't match"),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, "target type only in yaml"),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, "target type only in cmd"),
        ],
    )
    def test_custom_models_with_drum_with_model_yaml_target(
        self,
        resources,
        framework,
        problem,
        language,
        tmp_path,
        target_type_test_case,
        temp_file,
        inference_metadata_yaml,
    ):

        config_yaml = dedent(
            """
            name: custom_model
            type: inference
            """
        )

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {}'.format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output,
        )

        def _write_yaml(conf_yaml):
            with open(os.path.join(custom_model_dir, MODEL_CONFIG_FILENAME), mode="w") as f:
                f.write(config_yaml)

        assert_on_failure = False
        if target_type_test_case == "no target type, no yaml":
            pass
        elif target_type_test_case == "no target type in yaml":
            _write_yaml(config_yaml)
        elif target_type_test_case == "target types don't match":
            config_yaml += "targetType: {}".format(resources.target_types(problem))
            _write_yaml(config_yaml)
            cmd += " --target-type binary"
        elif target_type_test_case == "target type only in yaml":
            config_yaml += "targetType: {}".format(resources.target_types(problem))
            _write_yaml(config_yaml)
            assert_on_failure = True
        elif target_type_test_case == "target type only in cmd":
            cmd += " --target-type {}".format(resources.target_types(problem))
            assert_on_failure = True
        else:
            assert False

        _, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=assert_on_failure,
        )

        # positive path: target type is in either cmd or yaml
        if assert_on_failure:
            in_data = pd.read_csv(input_dataset)
            out_data = pd.read_csv(output)
            assert in_data.shape[0] == out_data.shape[0]
        else:
            stdo_stde = str(stdo) + str(stde)
            case_1 = (
                "Target type is missing. It must be provided in --target-type argument, TARGET_TYPE env var or model config file."
                in stdo_stde
            )
            case_2 = "required key(s) 'targetType' not found" in stdo_stde
            case_3 = (
                "Target type provided in --target-type argument doesn't match target type from model config file. Use either one of them or make them match."
                in stdo_stde
            )
            assert any([case_1, case_2, case_3])

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
