"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import io
import json
import os
import re
from json import JSONDecoder
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch

import pandas as pd
import pytest
import requests
from scipy.sparse import csr_matrix

from datarobot_drum.drum.description import version as drum_version
from datarobot_drum.drum.enum import (
    MODEL_CONFIG_FILENAME,
    X_TRANSFORM_KEY,
    Y_TRANSFORM_KEY,
    ArgumentsOptions,
    ModelInfoKeys,
    PredictionServerMimetypes,
    TargetType,
)
from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars
from datarobot_drum.drum.utils.structured_input_read_utils import (
    StructuredInputReadUtils,
)
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun, wait_for_server
from datarobot_drum.drum.root_predictors.transform_helpers import (
    parse_multi_part_response,
    read_csv_payload,
    read_mtx_payload,
)
from datarobot_drum.drum.root_predictors.utils import (
    _cmd_add_class_labels,
    _create_custom_model_dir,
    _exec_shell_cmd,
)
from tests.conftest import skip_if_framework_not_in_env, skip_if_keys_not_in_env
from tests.constants import (
    BINARY,
    CODEGEN,
    GPU_NIM,
    GPU_TRITON,
    GPU_VLLM,
    JULIA,
    KERAS,
    MLJ,
    MODEL_TEMPLATES_PATH,
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
    PYTHON_TEXT_GENERATION,
    PYTHON_GEO_POINT,
    PYTHON_TRANSFORM,
    PYTHON_TRANSFORM_DENSE,
    PYTHON_TRANSFORM_SPARSE,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    PYTORCH,
    R_FAIL_CLASSIFICATION_VALIDATION_HOOKS,
    R_PREDICT_SPARSE,
    R_TRANSFORM_SPARSE_INPUT,
    # R_TRANSFORM_SPARSE_OUTPUT,
    R_TRANSFORM_WITH_Y,
    R_VALIDATE_SPARSE_ESTIMATOR,
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
    TESTS_DATA_PATH,
    TEXT_GENERATION,
    GEO_POINT,
    TRANSFORM,
    XGB,
    R,
    PYTHON_VECTOR_DATABASE,
    VECTOR_DATABASE,
)


class TestInference:
    @pytest.fixture
    def temp_file(self):
        with NamedTemporaryFile() as f:
            yield f

    @pytest.mark.parametrize(
        "framework, problem, language, docker, use_labels_file",
        [
            (SKLEARN, SPARSE, PYTHON_PREDICT_SPARSE, None, False),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None, False),
            (SKLEARN, BINARY, PYTHON, None, False),
            (SKLEARN, MULTICLASS, PYTHON, None, False),
            (SKLEARN, MULTICLASS, PYTHON, None, True),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, None, False),
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
            (ONNX, REGRESSION, PYTHON, None, False),
            (ONNX, BINARY, PYTHON, None, False),
            (ONNX, MULTICLASS, PYTHON, None, False),
            (ONNX, MULTICLASS_BINARY, PYTHON, None, False),
            (RDS, REGRESSION, R, None, False),
            (RDS, BINARY, R, None, False),
            (RDS, MULTICLASS, R, None, False),
            (RDS, MULTICLASS_BINARY, R, None, False),
            (RDS_SPARSE, SPARSE, R_PREDICT_SPARSE, None, False),
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
            (PYTHON_TEXT_GENERATION, TEXT_GENERATION, PYTHON_TEXT_GENERATION, None, False),
            (PYTHON_GEO_POINT, GEO_POINT, PYTHON_GEO_POINT, None, False),
            (PYTHON_VECTOR_DATABASE, VECTOR_DATABASE, PYTHON_VECTOR_DATABASE, None, False),
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
        framework_env,
        capitalize_artifact_extension=False,
    ):
        skip_if_framework_not_in_env(framework, framework_env)

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

        env_vars = {}
        if problem == TEXT_GENERATION:
            env_vars = {"TARGET_NAME": "Response"}
        elif problem == VECTOR_DATABASE:
            env_vars = {"TARGET_NAME": "relevant"}

        with patch.dict(os.environ, env_vars):
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
            (ONNX, REGRESSION, PYTHON, None, False),
            (RDS, REGRESSION, R, None, False),
            (CODEGEN, REGRESSION, NO_CUSTOM, None, False),
            # POJO is not a relevant case. POJO artifact is a `.java` file which allowed to be only lowercase
            (MOJO, REGRESSION, NO_CUSTOM, None, False),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None, False),
            (PYPMML, REGRESSION, NO_CUSTOM, None, False),
            (MLJ, REGRESSION, JULIA, None, False),
        ],
    )
    def test_custom_models_with_drum_capitalize_artifact_extensions(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
        use_labels_file,
        temp_file,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        self.test_custom_models_with_drum(
            resources,
            framework,
            problem,
            language,
            docker,
            tmp_path,
            use_labels_file,
            temp_file,
            framework_env,
            capitalize_artifact_extension=True,
        )
        print(os.listdir(os.path.join(tmp_path, "custom_model")))

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, SPARSE, PYTHON_PREDICT_SPARSE, None),
            (RDS_SPARSE, SPARSE, R_PREDICT_SPARSE, None),
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
            (ONNX, REGRESSION, PYTHON, None),
            (ONNX, BINARY, PYTHON, None),
            (ONNX, MULTICLASS, PYTHON, None),
            (ONNX, MULTICLASS_BINARY, PYTHON, None),
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
            (PYTHON_TEXT_GENERATION, TEXT_GENERATION, PYTHON_TEXT_GENERATION, None),
            (PYTHON_GEO_POINT, GEO_POINT, PYTHON_GEO_POINT, None),
            (PYTHON_VECTOR_DATABASE, VECTOR_DATABASE, PYTHON_VECTOR_DATABASE, None),
        ],
    )
    @pytest.mark.parametrize("pass_args_as_env_vars", [False])
    @pytest.mark.parametrize("max_workers", [None, 2])
    def test_custom_models_with_drum_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        pass_args_as_env_vars,
        tmp_path,
        framework_env,
        endpoint_prediction_methods,
        max_workers,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        if framework == KERAS and max_workers and max_workers > 1:
            pytest.skip(
                "Current Keras integration is not multi-processing safe so we skip test when max_workers > 1"
            )
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        unset_drum_supported_env_vars()

        env_vars = {}
        if problem == TEXT_GENERATION:
            env_vars = {"TARGET_NAME": "Response"}
        elif problem == VECTOR_DATABASE:
            env_vars = {"TARGET_NAME": "relevant"}

        with patch.dict(os.environ, env_vars), DrumServerRun(
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
            pass_args_as_env_vars=pass_args_as_env_vars,
            max_workers=max_workers,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            # do predictions
            for endpoint in endpoint_prediction_methods:
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
        [
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, MULTICLASS, PYTHON, None),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, None),
        ],
    )
    @pytest.mark.parametrize("pass_args_as_env_vars", [True])
    @pytest.mark.parametrize("max_workers", [1])
    def test_custom_models_with_drum_prediction_server_with_args_passed_as_env_vars(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        pass_args_as_env_vars,
        tmp_path,
        framework_env,
        endpoint_prediction_methods,
        max_workers,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        self.test_custom_models_with_drum_prediction_server(
            resources,
            framework,
            problem,
            language,
            docker,
            pass_args_as_env_vars,
            tmp_path,
            framework_env,
            endpoint_prediction_methods,
            max_workers,
        )

    # The fitting code for the sklearn transform is in task_templates/1_transforms/3_python3_sklearn_transform
    # To retrain sklearn_transform.pkl / sklearn_transform_dense.pkl artifacts adjust code to use sparse / dense
    # Run DRUM using command:
    # drum fit --code-dir task_templates/1_transforms/3_python3_sklearn_transform --input tests/testdata/10k_diabetes_sample.csv --target-type transform --target readmitted --output <some dir>

    # To fit code for (SKLEARN_TRANSFORM, SPARSE_TRANSFORM, PYTHON_TRANSFORM_SPARSE, None, False):
    # Use the same pipeline ^, do:
    # * in the file:  task_templates/1_transforms/3_python3_sklearn_transform/create_transform_pipeline.py make StandardScaler(with_mean=False):
    # * drum fit --code-dir task_templates/1_transforms/3_python3_sklearn_transform --input tests/testdata/sparse.mtx --target-type transform --target-csv tests/testdata/sparse_target.csv --output output_dir --disable-strict-validation --sparse-column-file tests/testdata/sparse.columns
    # * replace transform_sparse.pkl with the new generated artifact.
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM, None),
            (SKLEARN_TRANSFORM_DENSE, TRANSFORM, PYTHON_TRANSFORM_DENSE, None),
            (SKLEARN_TRANSFORM, SPARSE_TRANSFORM, PYTHON_TRANSFORM_SPARSE, None),
        ],
    )
    @pytest.mark.parametrize("pass_target", [False])
    def test_custom_transform_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
        pass_target,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
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
            in_data = resources.input_data(framework, problem)

            files = {"X": open(input_dataset)}
            if pass_target:
                target_dataset = resources.targets(problem)
                files["y"] = open(target_dataset)
            if input_dataset.endswith(".mtx"):
                files["X.colnames"] = open(input_dataset.replace(".mtx", ".columns"))

            response = requests.post(run.url_server_address + "/transform/", files=files)
            assert response.ok

            parsed_response = parse_multi_part_response(response)

            if framework in [SKLEARN_TRANSFORM_DENSE, R_TRANSFORM_WITH_Y, R_TRANSFORM_SPARSE_INPUT]:
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
            elif framework == R_TRANSFORM_WITH_Y:
                assert type(transformed_out) == pd.DataFrame
                assert transformed_out.shape[1] == 80

            if pass_target:
                assert all(pd.read_csv(target_dataset) == target_out)
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
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
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
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with open(os.path.join(custom_model_dir, MODEL_CONFIG_FILENAME), mode="w") as f:
            f.write(config_yaml)

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
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
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
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
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        config_yaml = dedent(
            """
            name: custom_model
            type: inference
            """
        )

        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
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
        "framework, problem, language, supported_payload_formats",
        [
            (SKLEARN, REGRESSION, PYTHON, {"csv": None, "mtx": None}),
            (RDS, REGRESSION, R, {"csv": None, "mtx": None}),
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
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
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
            assert response.json() == {
                "supported_payload_formats": supported_payload_formats,
                "supported_methods": {"chat": False},
            }

    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (RDS_SPARSE, REGRESSION, R_VALIDATE_SPARSE_ESTIMATOR),
        ],
    )
    @pytest.mark.parametrize("production", [False, True])
    def test_predictions_r_mtx(
        self,
        resources,
        framework,
        problem,
        language,
        production,
        tmp_path,
        framework_env,
        endpoint_prediction_methods,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
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
            production=production,
        ) as run:
            input_dataset = resources.datasets(framework, SPARSE)

            # do predictions
            for endpoint in endpoint_prediction_methods:
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
        [
            (
                None,
                R_FAIL_CLASSIFICATION_VALIDATION_HOOKS,
                BINARY,
            )
        ],
    )
    def test_classification_validation_fails(
        self, resources, framework, language, target_type, tmp_path, framework_env
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            None,
            language,
        )

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

    @pytest.mark.parametrize(
        "framework, target_type, model_template_dir",
        [
            (
                GPU_TRITON,
                TargetType.UNSTRUCTURED,
                "triton_onnx_unstructured",
            ),
        ],
    )
    def test_triton_predictor(
        self, framework, target_type, model_template_dir, resources, tmp_path, framework_env
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        input_dataset = os.path.join(TESTS_DATA_PATH, "triton_densenet_onnx.bin")
        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, model_template_dir)

        run_triton_server_in_background = (
            f"tritonserver --model-repository={custom_model_dir}/model_repository"
        )
        _exec_shell_cmd(
            run_triton_server_in_background,
            "failed to start triton server",
            assert_if_fail=False,
            capture_output=False,
        )
        wait_for_server("http://localhost:8000/v2/health/ready", 60)

        with DrumServerRun(
            target_type=target_type.value,
            labels=None,
            custom_model_dir=custom_model_dir,
            production=False,
            gpu_predictor=framework,
            wait_for_server_timeout=600,
        ) as run:
            headers = {
                "Content-Type": f"{PredictionServerMimetypes.APPLICATION_OCTET_STREAM};charset=UTF-8"
            }
            response = requests.post(
                f"{run.url_server_address}/predictUnstructured/",
                data=open(input_dataset, "rb"),
                headers=headers,
            )

            assert response.ok, response.content

            response_text = response.content.decode("utf-8")
            json, header_length = JSONDecoder().raw_decode(response_text)
            assert json["model_name"] == "densenet_onnx"
            assert "INDIGO FINCH" in response_text[header_length:]


class TestNIM:
    @pytest.fixture(scope="class")
    def nim_predictor(self, framework_env):
        skip_if_framework_not_in_env(GPU_NIM, framework_env)
        skip_if_keys_not_in_env(["GPU_COUNT", "NGC_API_KEY"])

        os.environ["MLOPS_RUNTIME_PARAM_NGC_API_KEY"] = json.dumps(
            {
                "type": "credential",
                "payload": {
                    "credentialType": "apiToken",
                    "apiToken": os.environ["NGC_API_KEY"],
                },
            }
        )

        # the Runtime Parameters used for prediction requests
        os.environ[
            "MLOPS_RUNTIME_PARAM_prompt_column_name"
        ] = '{"type":"string","payload":"user_prompt"}'
        os.environ["MLOPS_RUNTIME_PARAM_max_tokens"] = '{"type": "numeric", "payload": 256}'
        os.environ["MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS"] = '{"type": "numeric", "payload": 2}'

        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, "gpu_nim_textgen")

        with DrumServerRun(
            target_type=TargetType.TEXT_GENERATION.value,
            labels=None,
            custom_model_dir=custom_model_dir,
            with_error_server=True,
            production=False,
            logging_level="info",
            gpu_predictor=GPU_NIM,
            target_name="response",
            wait_for_server_timeout=600,
        ) as run:
            response = requests.get(run.url_server_address)
            if not response.ok:
                raise RuntimeError("Server failed to start")
            yield run

    def test_predict(self, nim_predictor):
        data = io.StringIO("user_prompt\ntell me a joke")
        headers = {"Content-Type": f"{PredictionServerMimetypes.TEXT_CSV};charset=UTF-8"}
        response = requests.post(
            f"{nim_predictor.url_server_address}/predict/",
            data=data,
            headers=headers,
        )
        assert response.ok
        response_data = response.json()
        assert response_data
        assert "predictions" in response_data, response_data
        assert len(response_data["predictions"]) == 1
        assert "What do you call a fake noodle?" in response_data["predictions"][0], response_data

    @pytest.mark.parametrize("streaming", [False, True], ids=["sync", "streaming"])
    @pytest.mark.parametrize(
        "nchoices",
        [
            1,
            pytest.param(
                3, marks=pytest.mark.xfail(reason="NIM doesn't support more than one choice")
            ),
        ],
    )
    def test_chat_api(self, nim_predictor, streaming, nchoices):
        from openai import OpenAI

        client = OpenAI(
            base_url=nim_predictor.url_server_address, api_key="not-required", max_retries=0
        )

        completion = client.chat.completions.create(
            model="any name works",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Describe the city of Boston"},
            ],
            n=nchoices,
            stream=streaming,
            temperature=0.1,
        )

        if streaming:
            collected_messages = []
            for chunk in completion:
                assert len(chunk.choices) == nchoices
                chunk_message = chunk.choices[0].delta.content
                if chunk_message is not None:
                    collected_messages.append(chunk_message)
            llm_response = "".join(collected_messages)
        else:
            assert len(completion.choices) == nchoices
            llm_response = completion.choices[0].message.content

        assert "Boston! One of the oldest and most historic cities" in llm_response


class TestVLLM:
    @pytest.fixture(scope="class")
    def vllm_predictor(self, framework_env):
        skip_if_framework_not_in_env(GPU_VLLM, framework_env)
        skip_if_keys_not_in_env(["GPU_COUNT"])

        # Override default params from example model to use a smaller model
        # TODO: remove this when we can inject runtime params correctly.
        os.environ["MLOPS_RUNTIME_PARAM_model"] = json.dumps(
            {
                "type": "string",
                "payload": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            }
        )
        os.environ[
            "MLOPS_RUNTIME_PARAM_prompt_column_name"
        ] = '{"type":"string","payload":"user_prompt"}'
        os.environ["MLOPS_RUNTIME_PARAM_max_tokens"] = '{"type": "numeric", "payload": 30}'
        os.environ["MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS"] = '{"type": "numeric", "payload": 2}'

        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, "gpu_vllm_textgen")
        with open(os.path.join(custom_model_dir, "engine_config.json"), "w") as f:
            # Allows this model to run on Tesla T4 GPU
            json.dump({"args": ["--dtype=half"]}, f, indent=2)

        with DrumServerRun(
            target_type=TargetType.TEXT_GENERATION.value,
            labels=None,
            custom_model_dir=custom_model_dir,
            with_error_server=True,
            production=False,
            logging_level="info",
            gpu_predictor=GPU_VLLM,
            target_name="response",
            wait_for_server_timeout=360,
        ) as run:
            response = requests.get(run.url_server_address)
            if not response.ok:
                raise RuntimeError("Server failed to start")
            yield run

    def test_predict(self, vllm_predictor):
        data = io.StringIO("user_prompt\nDescribe the city of Boston.")
        headers = {"Content-Type": f"{PredictionServerMimetypes.TEXT_CSV};charset=UTF-8"}
        response = requests.post(
            f"{vllm_predictor.url_server_address}/predict/",
            data=data,
            headers=headers,
        )
        assert response.ok
        response_data = response.json()
        assert response_data
        assert "predictions" in response_data, response_data
        assert len(response_data["predictions"]) == 1
        assert (
            "Boston is a vibrant, historic city" in response_data["predictions"][0]
        ), response_data

    @pytest.mark.parametrize("streaming", [False, True], ids=["sync", "streaming"])
    @pytest.mark.parametrize("nchoices", [1, 3])
    def test_chat_api(self, vllm_predictor, streaming, nchoices):
        from openai import OpenAI

        if streaming and nchoices > 1:
            pytest.xfail("vLLM doesn't support multiple choices in streaming mode")

        client = OpenAI(
            base_url=vllm_predictor.url_server_address, api_key="not-required", max_retries=0
        )

        completion = client.chat.completions.create(
            model="any name works",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Describe the city of Boston"},
            ],
            n=nchoices,
            stream=streaming,
            temperature=0.1,
        )

        if streaming:
            collected_messages = []
            for chunk in completion:
                assert len(chunk.choices) == nchoices
                chunk_message = chunk.choices[0].delta.content
                if chunk_message is not None:
                    collected_messages.append(chunk_message)
            llm_response = "".join(collected_messages)
        else:
            assert len(completion.choices) == nchoices
            llm_response = completion.choices[0].message.content

        assert re.search(r"is a (vibrant and historic|bustling) city", llm_response)
