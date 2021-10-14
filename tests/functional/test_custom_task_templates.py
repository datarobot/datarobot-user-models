from __future__ import absolute_import

import pytest
import os
import tarfile
import shutil
from tempfile import TemporaryDirectory
import tempfile
import yaml
import datarobot as dr
from datarobot.errors import AsyncProcessUnsuccessfulError
from datarobot_bp_workshop import Workshop

BASE_PIPELINE_TASK_TEMPLATES_DIR = "task_templates/pipelines"
BASE_ESTIMATOR_TASK_TEMPLATES_DIR = "task_templates/estimators"
BASE_TRANSFORM_TASK_TEMPLATES_DIR = "task_templates/transforms"
BASE_FIXTURE_TASK_TEMPLATES_DIR = "tests/fixtures"

BASE_DATASET_DIR = "tests/testdata"


class TestCustomTaskTemplates(object):
    @staticmethod
    def get_template_base_path(template_type):
        if template_type == "pipeline":
            return BASE_PIPELINE_TASK_TEMPLATES_DIR
        if template_type == "estimator":
            return BASE_ESTIMATOR_TASK_TEMPLATES_DIR
        if template_type == "transform":
            return BASE_TRANSFORM_TASK_TEMPLATES_DIR
        if template_type == "fixture":
            return BASE_FIXTURE_TASK_TEMPLATES_DIR
        raise ValueError(f"Invalid template type {template_type}")

    @pytest.fixture(scope="session")
    def project_regression_juniors_grade(self):
        proj = dr.Project.create(
            sourcedata=os.path.join(BASE_DATASET_DIR, "juniors_3_year_stats_regression.csv")
        )
        proj.set_target(target="Grade 2014", mode=dr.AUTOPILOT_MODE.MANUAL)
        return proj.id

    @pytest.fixture(scope="session")
    def project_anomaly_juniors_grade(self):
        proj = dr.Project.create(
            sourcedata=os.path.join(BASE_DATASET_DIR, "juniors_3_year_stats_regression.csv")
        )
        proj.set_target(unsupervised_mode=True, mode=dr.AUTOPILOT_MODE.MANUAL)
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
    def project_binary_diabetes_no_text(self):
        proj = dr.Project.create(
            sourcedata=os.path.join(BASE_DATASET_DIR, "10k_diabetes_no_text.csv")
        )
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
                "project_binary_iris",
                "pytorch_drop_in_env",
                "binary",
            ),
            (
                "pipeline",
                "python3_pytorch_regression",
                "project_regression_juniors_grade",
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
                "project_regression_juniors_grade",
                "keras_drop_in_env",
                "regression",
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
                "project_regression_juniors_grade",
                "xgboost_drop_in_env",
                "regression",
            ),
            (
                "pipeline",
                "python3_sklearn_regression",
                "project_regression_juniors_grade",
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
            (
                "pipeline",
                "r_lang",
                "project_regression_juniors_grade",
                "r_drop_in_env",
                "regression",
            ),
            (
                "pipeline",
                "r_lang",
                "project_binary_iris",
                "r_drop_in_env",
                "binary",
            ),
            (
                "pipeline",
                "r_lang",
                "project_multiclass_skyserver",
                "r_drop_in_env",
                "multiclass",
            ),
            (
                "transform",
                "python3_sklearn_transform",
                "project_binary_diabetes_no_text",
                "sklearn_drop_in_env",
                "transform",
            ),
            (
                "transform",
                "r_transform_recipe",
                "project_binary_iris",
                "r_drop_in_env",
                "transform",
            ),
            (
                "transform",
                "r_transform_simple",
                "project_binary_iris",
                "r_drop_in_env",
                "transform",
            ),
            (
                "estimator",
                "r_anomaly_detection",
                "project_anomaly_juniors_grade",
                "r_drop_in_env",
                "anomaly",
            ),
        ],
    )
    def test_custom_task_templates(
        self, request, template_type, model_template, proj, env, target_type
    ):
        env_id, env_version_id = request.getfixturevalue(env)
        proj_id = request.getfixturevalue(proj)
        folder_base_path = self.get_template_base_path(template_type)

        dr_target_type = {
            "regression": dr.enums.CUSTOM_TASK_TARGET_TYPE.REGRESSION,
            "binary": dr.enums.CUSTOM_TASK_TARGET_TYPE.BINARY,
            "multiclass": dr.enums.CUSTOM_TASK_TARGET_TYPE.MULTICLASS,
            "transform": dr.enums.CUSTOM_TASK_TARGET_TYPE.TRANSFORM,
            "anomaly": dr.enums.CUSTOM_TASK_TARGET_TYPE.ANOMALY,
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
            if "image" in model_template:
                # Image example outputs an image, but we need a numeric for the final estimator
                bp = w.Tasks.IMG_GRAYSCALE_DOWNSCALED_IMAGE_FEATURIZER()(bp)
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

        models = proj.get_models()
        if len(models) > 1:
            if target_type == "regression":
                xgboost_mdl = [
                    m
                    for m in models
                    if m.model_type
                    == "eXtreme Gradient Boosted Trees Regressor with Early Stopping"
                ][0]
                dummy_mdl = [m for m in models if m.model_type == "Mean Response Regressor"][0]
            elif target_type == "anomaly":
                dm_mdl = [
                    m
                    for m in models
                    if m.model_type
                    == "Double Median Absolute Deviation Anomaly Detection with Calibration"
                ][0]
            else:
                xgboost_mdl = [
                    m
                    for m in models
                    if m.model_type
                    == "eXtreme Gradient Boosted Trees Classifier with Early Stopping"
                ][0]
                dummy_mdl = [m for m in models if m.model_type == "Majority Class Classifier"][0]
        else:
            blueprints = proj.get_blueprints()
            if target_type == "regression":
                xgboost_bp = [
                    bp
                    for bp in blueprints
                    if bp.model_type
                    == "eXtreme Gradient Boosted Trees Regressor with Early Stopping"
                ][0]
                dummy_bp = [bp for bp in blueprints if bp.model_type == "Mean Response Regressor"][
                    0
                ]
            elif target_type == "anomaly":
                dm_bp = [
                    bp
                    for bp in blueprints
                    if bp.model_type
                    == "Double Median Absolute Deviation Anomaly Detection with Calibration"
                ][0]
            else:
                xgboost_bp = [
                    bp
                    for bp in blueprints
                    if bp.model_type
                    == "eXtreme Gradient Boosted Trees Classifier with Early Stopping"
                ][0]
                dummy_bp = [
                    bp for bp in blueprints if bp.model_type == "Majority Class Classifier"
                ][0]

            if target_type == "anomaly":
                job_id = proj.train(dm_bp)
                job = dr.ModelJob.get(proj_id, job_id)
                dm_mdl = job.get_result_when_complete(max_wait=900)
            else:
                job_id = proj.train(xgboost_bp)
                job = dr.ModelJob.get(proj_id, job_id)
                xgboost_mdl = job.get_result_when_complete(max_wait=900)

                job_id = proj.train(dummy_bp)
                job = dr.ModelJob.get(proj_id, job_id)
                dummy_mdl = job.get_result_when_complete(max_wait=900)

        if target_type == "anomaly":
            # Require Synthetic AUC so dummy model score can be set to 0.5
            assert proj.metric == "Synthetic AUC"
            dm_score = dm_mdl.metrics[proj.metric]["validation"]
            threshold_score = (dm_score + 0.5) / 2
        else:
            xgboost_score = xgboost_mdl.metrics[proj.metric]["validation"]
            dummy_score = dummy_mdl.metrics[proj.metric]["validation"]
            threshold_score = (xgboost_score + dummy_score) / 2

        custom_task_score = res.metrics[proj.metric]["validation"]
        if target_type == "anomaly":
            metric_asc = False
        else:
            metric_details = proj.get_metrics(proj.target)["metric_details"]
            metric_asc = [m for m in metric_details if m["metric_name"] == proj.metric][0][
                "ascending"
            ]
        if metric_asc:
            assert (
                custom_task_score < threshold_score
            ), f"Accuracy check failed: {custom_task_score} > {threshold_score}"
        else:
            assert (
                custom_task_score > threshold_score
            ), f"Accuracy check failed: {custom_task_score} < {threshold_score}"

    @pytest.mark.parametrize(
        "template_type, model_template, proj, env, target_type, expected_msgs",
        [
            (
                "fixture",
                "validate_transform_fail_input_schema_validation",
                "project_binary_diabetes",
                "sklearn_drop_in_env",
                "transform",
                [
                    "schema validation failed for input",
                    "Datatypes incorrect. Data has types: NUM, but expected types to exactly match: CAT",
                ],
            ),
            (
                "fixture",
                "validate_transform_fail_output_schema_validation",
                "project_binary_diabetes",
                "sklearn_drop_in_env",
                "transform",
                [
                    "Schema validation found mismatch between output dataset and the supplied schema",
                    "Datatypes incorrect. Data has types: NUM, but expected types to exactly match: CAT",
                ],
            ),
        ],
    )
    def test_custom_task_templates_fail__schema_validation(
        self, request, template_type, model_template, proj, env, target_type, expected_msgs
    ):
        env_id, env_version_id = request.getfixturevalue(env)
        proj_id = request.getfixturevalue(proj)
        folder_base_path = self.get_template_base_path(template_type)

        target_type = dr.enums.CUSTOM_TASK_TARGET_TYPE.TRANSFORM

        custom_task = dr.CustomTask.create(name="transform", target_type=target_type)
        custom_task_version = dr.CustomTaskVersion.create_clean(
            custom_task_id=str(custom_task.id),
            base_environment_id=env_id,
            folder_path=os.path.join(folder_base_path, model_template),
        )

        # Only use numeric input features
        w = Workshop()
        bp = w.CustomTask(custom_task_version.custom_task_id, version=str(custom_task_version.id))(
            w.TaskInputs.NUM
        )
        bp = w.Tasks.LR1()(bp)

        user_blueprint = w.BlueprintGraph(bp).save()
        bp_id = user_blueprint.add_to_repository(proj_id)

        proj = dr.Project.get(proj_id)
        job_id = proj.train(bp_id)

        job = dr.ModelJob.get(proj_id, job_id)

        try:
            job.get_result_when_complete(max_wait=900)
        except AsyncProcessUnsuccessfulError:
            pass

        # TODO: [RAPTOR-5948] Create public-api-client support to download logs from model
        response = dr.client.get_client().get(f"projects/{proj_id}/models/{job.model_id}/logs")
        assert response.status_code == 200

        with tempfile.NamedTemporaryFile() as tar_f:
            tar_f.write(response.content)
            tar_f.flush()
            tar_f.seek(0)

            with tarfile.open(tar_f.name, mode="r:gz") as tar:
                files = tar.getnames()

                # The last log should contain the error
                fit_log = tar.extractfile(files[-1]).read().decode("utf-8")
                for expected_msg in expected_msgs:
                    assert expected_msg in fit_log
