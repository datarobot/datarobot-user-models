from __future__ import absolute_import

import os
import pytest
import datarobot as dr

BASE_MODEL_TEMPLATES_DIR = "model_templates"
BASE_DEV_ENV_DIR = "dev_env"


@pytest.fixture(scope="session")
def java_nginx_env():
    env_dir = os.path.join(BASE_DEV_ENV_DIR, "java_nginx_env")
    environment = dr.ExecutionEnvironment.create(name="java_nginx_env")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def rlang_nginx_env():
    env_dir = os.path.join(BASE_DEV_ENV_DIR, "rlang_nginx_env")
    environment = dr.ExecutionEnvironment.create(name="rlang_nginx_env")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def python3_nginx_env():
    env_dir = os.path.join(BASE_DEV_ENV_DIR, "python3_nginx_env")
    environment = dr.ExecutionEnvironment.create(name="python3_nginx_env")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


class TestInferenceModelTemplates(object):
    @pytest.mark.parametrize(
        "model_template, language, env, dataset, target, pos_label, neg_label",
        [
            (
                "inference/java_codegen",
                "java",
                "java_nginx_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/h2o_pojo/regression",
                "java",
                "java_nginx_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/h2o_pojo/binary",
                "java",
                "java_nginx_env",
                "binary_testing_data",
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
            ),
            (
                "inference/h2o_mojo/binary",
                "java",
                "java_nginx_env",
                "binary_testing_data",
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
            ),
            (
                "inference/python3_pmml",
                "python",
                "java_nginx_env",
                "binary_testing_data",
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
            ),
            (
                "inference/r_lang",
                "r",
                "rlang_nginx_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_keras",
                "python",
                "python3_nginx_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_keras_joblib",
                "python",
                "python3_nginx_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_keras_vizai_joblib",
                "python",
                "python3_nginx_env",
                "binary_vizai_testing_data",
                "class",
                "dogs",
                "cats",
            ),
            (
                "inference/python3_pytorch",
                "python",
                "python3_nginx_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_sklearn",
                "python",
                "python3_nginx_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
            ),
            (
                "inference/python3_xgboost",
                "python",
                "python3_nginx_env",
                "regression_testing_data",
                "MEDV",
                None,
                None,
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
