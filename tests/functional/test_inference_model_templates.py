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
        "model_template, language, env, target",
        [
            ("java_codegen", "java", "java_drop_in_env", "MEDV"),
            ("python3_keras_inference", "python", "keras_drop_in_env", "MEDV"),
            ("python3_keras_inference_joblib", "python", "keras_drop_in_env", "MEDV"),
            ("python3_pytorch_inference", "python", "pytorch_drop_in_env", "MEDV"),
            ("python3_sklearn_inference", "python", "sklearn_drop_in_env", "MEDV"),
            ("python3_xgboost_inference", "python", "xgboost_drop_in_env", "MEDV"),
            ("r_lang", "r", "r_drop_in_env"),
        ],
    )
    def test_inference_model_templates(self, request, model_template, language, env, target):
        env_id, env_version_id = request.getfixturevalue(env)
        test_data_id = request.getfixturevalue("regression_testing_data")

        model = dr.CustomInferenceModel.create(
            name=model_template,
            target_type=dr.TARGET_TYPE.REGRESSION,
            target_name=target,
            description=model_template,
            language=language,
        )

        model_version = dr.CustomModelVersion.create_clean(
            custom_model_id=model.id,
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
