from __future__ import absolute_import

import os
import pytest
import datarobot as dr

from .tmp_training_fixtures import CustomTrainingBlueprint, CustomTrainingModel

BASE_MODEL_TEMPLATES_DIR = "model_templates"
BASE_DATASET_DIR = "tests/testdata"


class TestTrainingModelTemplates(object):
    @classmethod
    def setup_class(cls):
        dr.Client(endpoint="http://localhost/api/v2", token=os.environ["DATAROBOT_API_TOKEN"])

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

    @pytest.mark.parametrize(
        "model_template, proj, target_type, env",
        [
            (
                "python3_keras_training_joblib",
                "project_regression_boston",
                "regression",
                "keras_drop_in_env",
            ),
            (
                "python3_xgboost_training",
                "project_regression_boston",
                "regression",
                "xgboost_drop_in_env",
            ),
            (
                "python3_sklearn_training",
                "project_regression_boston",
                "regression",
                "sklearn_drop_in_env",
            ),
        ],
    )
    def test_training_model_templates(self, request, model_template, proj, target_type, env):
        env_id, env_version_id = request.getfixturevalue(env)
        proj_id = request.getfixturevalue(proj)
        dr_target_type = (
            dr.TARGET_TYPE.REGRESSION if target_type == "regression" else dr.TARGET_TYPE.BINARY
        )

        model = CustomTrainingModel.create(
            name=model_template, target_type=dr_target_type, description=model_template
        )

        model_version = dr.CustomModelVersion.create_clean(
            model.id, folder_path=os.path.join(BASE_MODEL_TEMPLATES_DIR, model_template)
        )

        blueprint = CustomTrainingBlueprint.create(
            model.id, env_id, model_version.id, env_version_id
        )
        proj = dr.Project.get(proj_id)

        raw_featurelist_id = next(
            fl.id for fl in proj.get_featurelists() if fl.name == "Raw Features"
        )
        job_id = proj.train(blueprint, featurelist_id=raw_featurelist_id)

        job = dr.ModelJob.get(proj_id, job_id)
        test_passed = False
        try:
            res = job.get_result_when_complete()
            if isinstance(res, dr.Model):
                test_passed = True
        except:
            pass

        assert test_passed
