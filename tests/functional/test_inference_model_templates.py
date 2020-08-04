from __future__ import absolute_import

import os
import pytest
import datarobot as dr

BASE_MODEL_TEMPLATES_DIR = "model_templates"
BASE_FIXTURE_DIR = "tests/fixtures"
BASE_DATASET_DIR = "tests/testdata"
ARTIFACT_DIR = "drop_in_model_artifacts"
CUSTOM_PREDICT_PY_PATH = os.path.join(BASE_FIXTURE_DIR, "custom.py")
CUSTOM_PREDICT_R_PATH = os.path.join(BASE_FIXTURE_DIR, "custom.R")
CUSTOM_LOAD_PREDICT_PY_PATH = os.path.join(BASE_FIXTURE_DIR, "load_model_custom.py")
CUSTOM_LOAD_PREDICT_R_PATH = os.path.join(BASE_FIXTURE_DIR, "load_model_custom.R")


class TestInferenceModelTemplates(object):
    @pytest.mark.parametrize(
        "model_template, language, env, dataset, target, pos_label, neg_label",
        [
            (
                "inference/java_codegen",
                "java",
                "java_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/h2o_pojo/regression",
                "java",
                "java_h2o_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_keras",
                "python",
                "keras_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_keras",
                "python",
                "keras_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_keras_joblib",
                "python",
                "keras_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_keras_vizai_joblib",
                "python",
                "keras_drop_in_env",
                "binary_vizai_testing_data",
                "class",
                "dogs",
                "cats",
            ),
            (
                "inference/python3_pytorch",
                "python",
                "pytorch_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_sklearn",
                "python",
                "sklearn_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_xgboost",
                "python",
                "xgboost_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/r_lang",
                "r",
                "r_drop_in_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_pmml",
                "python",
                "pmml_drop_in_env",
                "binary_testing_data",
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
            ),
            (
                "inference/h2o_pojo/binary",
                "java",
                "java_h2o_drop_in_env",
                "binary_testing_data",
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
            ),
            (
                "inference/h2o_mojo/binary",
                "java",
                "java_h2o_drop_in_env",
                "binary_testing_data",
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
            ),
        ],
    )
    def test_inference_model_templates(
        self, request, model_template, language, env, dataset, target, pos_label, neg_label
    ):
        env_id, env_version_id = request.getfixturevalue(env)
        test_data_id = request.getfixturevalue(dataset)

        dr_target_type = dr.TARGET_TYPE.REGRESSION
        if pos_label is not None and neg_label is not None:
            dr_target_type = dr.TARGET_TYPE.BINARY
        model = dr.CustomInferenceModel.create(
            name=model_template,
            target_type=dr_target_type,
            target_name=target,
            description=model_template,
            language=language,
            positive_class_label=pos_label,
            negative_class_label=neg_label,
        )

        model_version = dr.CustomModelVersion.create_clean(
            custom_model_id=model.id,
            base_environment_id=env_id,
            folder_path=os.path.join(BASE_MODEL_TEMPLATES_DIR, model_template),
        )

        test = dr.CustomModelTest.create(
            custom_model_id=model.id,
            custom_model_version_id=model_version.id,
            environment_id=env_id,
            environment_version_id=env_version_id,
            dataset_id=test_data_id,
        )

        assert test.overall_status == "succeeded"
