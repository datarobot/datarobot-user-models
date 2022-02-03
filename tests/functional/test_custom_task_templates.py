"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from __future__ import absolute_import
from typing import List, Tuple, Union

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

BASE_PIPELINE_TASK_TEMPLATES_DIR = "task_templates/3_pipelines"
BASE_ESTIMATOR_TASK_TEMPLATES_DIR = "task_templates/2_estimators"
BASE_TRANSFORM_TASK_TEMPLATES_DIR = "task_templates/1_transforms"
BASE_FIXTURE_TASK_TEMPLATES_DIR = "tests/fixtures"

BASE_DATASET_DIR = "tests/testdata"

ModelLike = Union[dr.models.Model, dr.models.Blueprint]


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

    @staticmethod
    def get_model_by_type(models: List[ModelLike], type_name: str) -> ModelLike:
        return [m for m in models if m.model_type == type_name][0]

    @staticmethod
    def get_accuracy_threshold(proj: dr.models.Project, target_type: str) -> Tuple[float, bool]:
        """ Compute accuracy threshold by running a known accurate model and dummy model
        and averaging their accuracy scores """
        models = proj.get_models()
        if len(models) > 1:
            # get existing models if project was created by another run of this test
            if target_type == "regression":
                xgboost_mdl = TestCustomTaskTemplates.get_model_by_type(
                    models, "eXtreme Gradient Boosted Trees Regressor with Early Stopping"
                )
                dummy_mdl = TestCustomTaskTemplates.get_model_by_type(
                    models, "Mean Response Regressor"
                )
            elif target_type == "anomaly":
                dm_mdl = TestCustomTaskTemplates.get_model_by_type(
                    models, "Double Median Absolute Deviation Anomaly Detection with Calibration"
                )
            else:
                xgboost_mdl = TestCustomTaskTemplates.get_model_by_type(
                    models, "eXtreme Gradient Boosted Trees Classifier with Early Stopping"
                )
                dummy_mdl = TestCustomTaskTemplates.get_model_by_type(
                    models, "Majority Class Classifier"
                )
        else:
            # run models for new projects
            blueprints = proj.get_blueprints()
            if target_type == "regression":
                xgboost_bp = TestCustomTaskTemplates.get_model_by_type(
                    blueprints, "eXtreme Gradient Boosted Trees Regressor with Early Stopping"
                )
                dummy_bp = TestCustomTaskTemplates.get_model_by_type(
                    blueprints, "Mean Response Regressor"
                )
            elif target_type == "anomaly":
                dm_bp = TestCustomTaskTemplates.get_model_by_type(
                    blueprints,
                    "Double Median Absolute Deviation Anomaly Detection with Calibration",
                )
            else:
                xgboost_bp = TestCustomTaskTemplates.get_model_by_type(
                    blueprints, "eXtreme Gradient Boosted Trees Classifier with Early Stopping"
                )
                dummy_bp = TestCustomTaskTemplates.get_model_by_type(
                    blueprints, "Majority Class Classifier"
                )

            if target_type == "anomaly":
                job_id = proj.train(dm_bp)
                job = dr.ModelJob.get(proj.id, job_id)
                dm_mdl = job.get_result_when_complete(max_wait=900)
            else:
                job_id = proj.train(xgboost_bp)
                job = dr.ModelJob.get(proj.id, job_id)
                xgboost_mdl = job.get_result_when_complete(max_wait=900)

                job_id = proj.train(dummy_bp)
                job = dr.ModelJob.get(proj.id, job_id)
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

        if target_type == "anomaly":
            metric_asc = False
        else:
            metric_details = proj.get_metrics(proj.target)["metric_details"]
            metric_asc = [m for m in metric_details if m["metric_name"] == proj.metric][0][
                "ascending"
            ]
        return threshold_score, metric_asc

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
    def project_multiclass_iris(self):
        proj = dr.Project.create(
            sourcedata=os.path.join(BASE_DATASET_DIR, "iris_with_spaces_full.csv")
        )
        proj.set_target(target="Species", mode=dr.AUTOPILOT_MODE.MANUAL)
        return proj.id

    @pytest.fixture(scope="session")
    def project_weight_test(self):
        proj = dr.Project.create(sourcedata=os.path.join(BASE_DATASET_DIR, "weight_test.csv"))
        advanced_options = dr.helpers.AdvancedOptions(weights="weights")
        proj.set_target(
            target="target", mode=dr.AUTOPILOT_MODE.MANUAL, advanced_options=advanced_options
        )
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

    @pytest.fixture(scope="session")
    def project_skyserver_manual_partition(self):
        # This dataset has a "partition" column where partition V1 has classes QSO & GALAXY and
        # partition V2 has STAR & GALAXY
        proj = dr.Project.create(
            sourcedata=os.path.join(BASE_DATASET_DIR, "skyserver_manual_partition.csv")
        )
        proj.set_target(
            target="class",
            mode=dr.AUTOPILOT_MODE.MANUAL,
            partitioning_method=dr.UserCV("partition", "H"),
        )
        return proj.id

    @pytest.mark.parametrize(
        "template_type, model_template, proj, env, target_type, pre_processing",
        [
            (
                "pipeline",
                "3_r_lang",
                "project_regression_juniors_grade",
                "r_drop_in_env",
                "regression",
                None,
            ),
            ("pipeline", "3_r_lang", "project_binary_iris", "r_drop_in_env", "binary", None),
            (
                "pipeline",
                "3_r_lang",
                "project_multiclass_skyserver",
                "r_drop_in_env",
                "multiclass",
                None,
            ),
            (
                "transform",
                "4_r_transform_recipe",
                "project_binary_iris",
                "r_drop_in_env",
                "transform",
                None,
            ),
            (
                "transform",
                "2_r_transform_simple",
                "project_binary_iris",
                "r_drop_in_env",
                "transform",
                None,
            ),
            (
                "estimator",
                "8_r_anomaly_detection",
                "project_anomaly_juniors_grade",
                "r_drop_in_env",
                "anomaly",
                None,
            ),
            (
                "fixture",
                "custom_task_interface_binary",
                "project_binary_iris",
                "sklearn_drop_in_env",
                "binary",
                None,
            ),
            (
                "fixture",
                "custom_task_interface_regression",
                "project_regression_juniors_grade",
                "sklearn_drop_in_env",
                "regression",
                None,
            ),
            (
                "fixture",
                "custom_task_interface_regression",
                "project_regression_juniors_grade",
                "sklearn_drop_in_env",
                "regression",
                "dr_numeric_impute",
            ),
            (
                "fixture",
                "custom_task_interface_transform_missing_values",
                "project_binary_iris",
                "sklearn_drop_in_env",
                "transform",
                None,
            ),
            (
                "fixture",
                "custom_task_interface_multiclass",
                "project_multiclass_skyserver",
                "sklearn_drop_in_env",
                "multiclass",
                None,
            ),
        ],
    )
    def test_custom_task_templates(
        self, request, template_type, model_template, proj, env, target_type, pre_processing
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
                metadata = yaml.safe_load(open(metadata_filename))
                metadata["targetType"] = target_type
                yaml.dump(metadata, open(metadata_filename, "w"), default_flow_style=False)

            custom_task_version = dr.CustomTaskVersion.create_clean(
                custom_task_id=str(custom_task.id),
                base_environment_id=env_id,
                folder_path=code_dir,
            )

        w = Workshop()
        bp = w.TaskInputs.ALL
        if pre_processing == "dr_numeric_impute":
            bp = w.Tasks.PNI2()(bp)
        else:
            assert not pre_processing, f"Pre-processing {pre_processing} not supported."
        bp = w.CustomTask(custom_task_version.custom_task_id, version=str(custom_task_version.id))(
            bp
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
        res = job.get_result_when_complete(max_wait=1200)
        if isinstance(res, dr.Model):
            test_passed = True

        assert test_passed, "Job result is the object: " + str(res)

        custom_task_score = res.metrics[proj.metric]["validation"]
        threshold_score, metric_asc = self.get_accuracy_threshold(proj, target_type)
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
                    "Schema validation found mismatch between input dataset and the supplied schema",
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
            job.get_result_when_complete(max_wait=1200)
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
