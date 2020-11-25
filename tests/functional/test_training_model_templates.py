from __future__ import absolute_import

import os
import pytest
import datarobot as dr

from datarobot._experimental import CustomTrainingBlueprint, CustomTrainingModel

BASE_MODEL_TEMPLATES_DIR = "model_templates"
BASE_DATASET_DIR = "tests/testdata"


class TestTrainingModelTemplates(object):
    @pytest.fixture(scope="session")
    def project_regression_boston(self):
        proj = dr.Project.create(sourcedata=os.path.join(BASE_DATASET_DIR, "boston_housing.csv"))
        proj.set_target(target="MEDV", mode=dr.AUTOPILOT_MODE.MANUAL)
        return proj.id

    @pytest.fixture(scope="session")
    def project_binary_iris(self):
        proj = dr.Project.create(
            sourcedata=os.path.join(BASE_DATASET_DIR, "iris_binary_training.csv")
        )
        proj.set_target(target="Species", mode=dr.AUTOPILOT_MODE.MANUAL)
        return proj.id

    @pytest.fixture(scope="session")
    def project_binary_cats_dogs(self):
        proj = dr.Project.create(
            sourcedata=os.path.join(BASE_DATASET_DIR, "cats_dogs_small_training.csv")
        )
        proj.set_target(target="class", mode=dr.AUTOPILOT_MODE.MANUAL)
        return proj.id

    @pytest.fixture(scope="session")
    def project_binary_diabetes(self):
        proj = dr.Project.create(sourcedata=os.path.join(BASE_DATASET_DIR, "10k_diabetes.csv"))
        proj.set_target(target="readmitted", mode=dr.AUTOPILOT_MODE.MANUAL)
        return proj.id

    @pytest.fixture(scope="session")
    def project_multiclass_skyserver(self):
        proj = dr.Project.create(
            sourcedata=os.path.join(BASE_DATASET_DIR, "skyserver_sql2_27_2018_6_51_39_pm.csv")
        )
        proj.set_target(target="class", mode=dr.AUTOPILOT_MODE.MANUAL)
        return proj.id

    @pytest.mark.parametrize(
        "model_template, proj, env, target_type",
        [
            (
                "training/python3_pytorch",
                "project_binary_diabetes",
                "pytorch_drop_in_env",
                "binary",
            ),
            (
                "training/python3_pytorch",
                "project_binary_iris",
                "pytorch_drop_in_env",
                "binary",
            ),
            (
                "training/python3_pytorch",
                "project_regression_boston",
                "pytorch_drop_in_env",
                "regression",
            ),
            (
                "training/python3_pytorch_multiclass",
                "project_multiclass_skyserver",
                "pytorch_drop_in_env",
                "multiclass",
            ),
            (
                "training/python3_keras_joblib",
                "project_regression_boston",
                "keras_drop_in_env",
                "regression",
            ),
            (
                "training/python3_keras_joblib",
                "project_binary_iris",
                "keras_drop_in_env",
                "binary",
            ),
            (
                "training/python3_keras_joblib",
                "project_multiclass_skyserver",
                "keras_drop_in_env",
                "multiclass",
            ),
            (
                "training/python3_keras_vizai_joblib",
                "project_binary_cats_dogs",
                "keras_drop_in_env",
                "binary",
            ),
            (
                "training/python3_xgboost",
                "project_regression_boston",
                "xgboost_drop_in_env",
                "regression",
            ),
            (
                "training/python3_xgboost",
                "project_binary_iris",
                "xgboost_drop_in_env",
                "binary",
            ),
            (
                "training/python3_xgboost",
                "project_multiclass_skyserver",
                "xgboost_drop_in_env",
                "multiclass",
            ),
            (
                "training/python3_sklearn_regression",
                "project_regression_boston",
                "sklearn_drop_in_env",
                "regression",
            ),
            (
                "training/python3_sklearn_binary",
                "project_binary_iris",
                "sklearn_drop_in_env",
                "binary",
            ),
            (
                "training/python3_sklearn_multiclass",
                "project_multiclass_skyserver",
                "sklearn_drop_in_env",
                "multiclass",
            ),
            (
                "training/r_lang",
                "project_regression_boston",
                "r_drop_in_env",
                "regression",
            ),
            (
                "training/r_lang",
                "project_binary_iris",
                "r_drop_in_env",
                "binary",
            ),
            (
                "training/r_lang",
                "project_multiclass_skyserver",
                "r_drop_in_env",
                "multiclass",
            ),
        ],
    )
    def test_training_model_templates(self, request, model_template, proj, env, target_type):
        env_id, env_version_id = request.getfixturevalue(env)
        proj_id = request.getfixturevalue(proj)
        if target_type == "regression":
            dr_target_type = dr.TARGET_TYPE.REGRESSION
        elif target_type == "binary":
            dr_target_type = dr.TARGET_TYPE.BINARY
        elif target_type == "multiclass":
            dr_target_type = dr.TARGET_TYPE.MULTICLASS
        else:
            raise ValueError("Unkown target type {}".format(target_type))

        model = CustomTrainingModel.create(name="training model", target_type=dr_target_type)
        model_version = dr.CustomModelVersion.create_clean(
            custom_model_id=model.id,
            base_environment_id=env_id,
            folder_path=os.path.join(BASE_MODEL_TEMPLATES_DIR, model_template),
        )
        proj = dr.Project.get(proj_id)

        # TODO: Update this once the datarobot client is updated
        payload = dict(custom_mode_version_id=model_version.id)
        response = dr.client.get_client().post("customTrainingBlueprints/", data=payload)
        user_blueprint_id = response.json()["user_blueprint_id"]

        payload = dict(project_id=proj_id, user_blueprint_ids=[user_blueprint_id])
        response = dr.client.get_client().post("userBlueprints/addToMenu/", data=payload)
        blueprint_id = response.json()[user_blueprint_id]

        job_id = proj.train(blueprint_id)

        job = dr.ModelJob.get(proj_id, job_id)
        test_passed = False
        res = job.get_result_when_complete(max_wait=900)
        if isinstance(res, dr.Model):
            test_passed = True

        assert test_passed, "Job result is the object: " + str(res)
