from __future__ import absolute_import

import os
import shutil
from tempfile import TemporaryDirectory

import pytest
import datarobot as dr
import yaml
from datarobot_bp_workshop import Workshop

BASE_PIPELINE_TASK_TEMPLATES_DIR = "task_templates/pipelines"
BASE_TRANSFORM_TASK_TEMPLATES_DIR = "task_templates/transforms"
BASE_DATASET_DIR = "tests/testdata"


class TestCustomTaskTemplates(object):
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
        "template_type, model_template, proj, env, target_type",
        [
            (
                "pipeline",
                "python3_pytorch",
                "project_binary_diabetes",
                "pytorch_drop_in_env",
                "binary",
            ),
            (
                "pipeline",
                "python3_pytorch",
                "project_binary_iris",
                "pytorch_drop_in_env",
                "binary",
            ),
            (
                "pipeline",
                "python3_pytorch",
                "project_regression_boston",
                "pytorch_drop_in_env",
                "regression",
            ),
            (
                "pipeline",
                "python3_pytorch_multiclass",
                "project_multiclass_skyserver",
                "pytorch_drop_in_env",
                "multiclass",
            ),
            (
                "pipeline",
                "python3_keras_joblib",
                "project_regression_boston",
                "keras_drop_in_env",
                "regression",
            ),
            (
                "pipeline",
                "python3_keras_joblib",
                "project_binary_iris",
                "keras_drop_in_env",
                "binary",
            ),
            (
                "pipeline",
                "python3_keras_joblib",
                "project_multiclass_skyserver",
                "keras_drop_in_env",
                "multiclass",
            ),
            # This test currently fails, because it uses image features, which isn't one of the
            # Allowed by default data types for Custom Tasks. We can re-enable this
            # Test if we add image features in the fixture to the allowed data types.
            # (
            #     "pipeline",
            #     "python3_keras_vizai_joblib",
            #     "project_binary_cats_dogs",
            #     "keras_drop_in_env",
            #     "binary",
            # ),
            (
                "pipeline",
                "python3_xgboost",
                "project_regression_boston",
                "xgboost_drop_in_env",
                "regression",
            ),
            (
                "pipeline",
                "python3_xgboost",
                "project_binary_iris",
                "xgboost_drop_in_env",
                "binary",
            ),
            (
                "pipeline",
                "python3_xgboost",
                "project_multiclass_skyserver",
                "xgboost_drop_in_env",
                "multiclass",
            ),
            (
                "pipeline",
                "python3_sklearn_regression",
                "project_regression_boston",
                "sklearn_drop_in_env",
                "regression",
            ),
            (
                "pipeline",
                "python3_sklearn_binary",
                "project_binary_iris",
                "sklearn_drop_in_env",
                "binary",
            ),
            (
                "pipeline",
                "python3_sklearn_multiclass",
                "project_multiclass_skyserver",
                "sklearn_drop_in_env",
                "multiclass",
            ),
            ("pipeline", "r_lang", "project_regression_boston", "r_drop_in_env", "regression",),
            ("pipeline", "r_lang", "project_binary_iris", "r_drop_in_env", "binary",),
            ("pipeline", "r_lang", "project_multiclass_skyserver", "r_drop_in_env", "multiclass",),
            (
                "transform",
                "python3_sklearn_transform",
                "project_binary_diabetes",
                "sklearn_drop_in_env",
                "transform",
            ),
        ],
    )
    def test_custom_task_templates(
        self, request, template_type, model_template, proj, env, target_type
    ):
        env_id, env_version_id = request.getfixturevalue(env)
        proj_id = request.getfixturevalue(proj)
        folder_base_path = BASE_PIPELINE_TASK_TEMPLATES_DIR

        if template_type == "transform":
            folder_base_path = BASE_TRANSFORM_TASK_TEMPLATES_DIR

        dr_target_type = {
            "regression": dr.enums.CUSTOM_TASK_TARGET_TYPE.REGRESSION,
            "binary": dr.enums.CUSTOM_TASK_TARGET_TYPE.BINARY,
            "multiclass": dr.enums.CUSTOM_TASK_TARGET_TYPE.MULTICLASS,
            "transform": dr.enums.CUSTOM_TASK_TARGET_TYPE.TRANSFORM,
        }[target_type]

        custom_task = dr.CustomTask.create(name="estimator", target_type=dr_target_type)
        with TemporaryDirectory() as temp_dir:
            code_dir = os.path.join(temp_dir, "code")
            shutil.copytree(os.path.join(folder_base_path, model_template), code_dir)
            metadata_filename = os.path.join(code_dir, "model-metadata.yaml")
            if os.path.isfile(metadata_filename):
                # Set the target type in the metadata file sent to DataRobot to the correct type.
                metadata = yaml.load(open(metadata_filename))
                metadata["targetType"] = target_type
                yaml.dump(metadata, open(metadata_filename, "w"))

            custom_task_version = dr.CustomTaskVersion.create_clean(
                custom_task_id=str(custom_task.id),
                base_environment_id=env_id,
                folder_path=code_dir,
            )

        w = Workshop()
        bp = w.CustomTask(custom_task_version.custom_task_id, version=str(custom_task_version.id))(
            w.TaskInputs.ALL
        )
        if dr_target_type == dr.enums.CUSTOM_TASK_TARGET_TYPE.TRANSFORM:
            bp = w.Tasks.LR1()(bp)
        user_blueprint = w.BlueprintGraph(bp).save()
        bp_id = user_blueprint.add_to_repository(proj_id)

        proj = dr.Project.get(proj_id)
        job_id = proj.train(bp_id)

        job = dr.ModelJob.get(proj_id, job_id)
        test_passed = False
        res = job.get_result_when_complete(max_wait=900)
        if isinstance(res, dr.Model):
            test_passed = True

        assert test_passed, "Job result is the object: " + str(res)
