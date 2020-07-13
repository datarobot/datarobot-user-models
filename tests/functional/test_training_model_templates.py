from __future__ import absolute_import

import os
import pytest
import datarobot as dr

from datarobot._experimental import CustomTrainingBlueprint

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

    @pytest.mark.parametrize(
        "model_template, proj, env, target_type",
        [
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
            ("training/python3_xgboost", "project_binary_iris", "xgboost_drop_in_env", "binary",),
            (
                "training/python3_sklearn",
                "project_regression_boston",
                "sklearn_drop_in_env",
                "regression",
            ),
            ("training/python3_sklearn", "project_binary_iris", "sklearn_drop_in_env", "binary",),
        ],
    )
    def test_training_model_templates(self, request, model_template, proj, env, target_type):
        env_id, env_version_id = request.getfixturevalue(env)
        proj_id = request.getfixturevalue(proj)
        dr_target_type = (
            dr.TARGET_TYPE.REGRESSION if target_type == "regression" else dr.TARGET_TYPE.BINARY
        )

        blueprint = CustomTrainingBlueprint.create_from_dropin(
            model_name="training model",
            dropin_env_id=env_id,
            target_type=dr_target_type,
            folder_path=os.path.join(BASE_MODEL_TEMPLATES_DIR, model_template),
        )
        proj = dr.Project.get(proj_id)

        job_id = proj.train(blueprint)

        job = dr.ModelJob.get(proj_id, job_id)
        test_passed = False
        res = job.get_result_when_complete()
        if isinstance(res, dr.Model):
            test_passed = True

        assert test_passed, "Job result is the object: " + str(res)
