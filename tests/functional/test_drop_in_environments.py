from __future__ import absolute_import

import os
import time
import pytest
import datarobot as dr

BASE_FIXTURE_DIR = "tests/fixtures"
ARTIFACT_DIR = "drop_in_model_artifacts"
CUSTOM_PREDICT_PY_PATH = os.path.join(BASE_FIXTURE_DIR, "custom.py")
CUSTOM_PREDICT_R_PATH = os.path.join(BASE_FIXTURE_DIR, "custom.R")
CUSTOM_LOAD_PREDICT_PY_PATH = os.path.join(BASE_FIXTURE_DIR, "load_model_custom.py")
CUSTOM_LOAD_PREDICT_R_PATH = os.path.join(BASE_FIXTURE_DIR, "load_model_custom.R")


class TestDropInEnvironments(object):
    def make_custom_model(
        self,
        artifact_names,
        base_environment_id,
        positive_class_label=None,
        negative_class_label=None,
        class_labels=None,
        custom_predict_path=None,
        other_file_names=None,
        artifact_only=False,
        target_name="target",
    ):
        """

        Parameters
        ----------
        artifact_names: Union[List[str], str]
            a singular filename or list of artifact filenames from the
            drop_in_model_artifacts directory to include
        base_environment_id: str
            The base environment for the model version
        positive_class_label: str or None
        negative_class_label: str or None
        class_labels: List[str] or None
        custom_predict_path: str or None
            path to the custom.py or custom.R file to include
        other_file_names: List[str] or None
            a list of non-artifact files to include from drop_in_model_artifacts
        artifact_only: bool
            if true, the custom model will be uploaded as a single uncompressed artifact file
            This is not compatible with a list of artifacts, other_file_names,
            or custom_predict_path

        Returns
        -------
        custom_model_id, custom_model_version_id
        """
        if artifact_only and (
            isinstance(artifact_names, list) or custom_predict_path or other_file_names
        ):
            raise ValueError(
                "artifact_only=True cannot be used with a list of artifact_names, "
                "a custom_predict_path, or other_file_names"
            )

        if not isinstance(artifact_names, list):
            artifact_names = [artifact_names]

        if positive_class_label and negative_class_label:
            target_type = dr.TARGET_TYPE.BINARY
        elif class_labels:
            target_type = dr.TARGET_TYPE.MULTICLASS
        else:
            target_type = dr.TARGET_TYPE.REGRESSION

        custom_model = dr.CustomInferenceModel.create(
            name=artifact_names[0],
            target_type=target_type,
            target_name=target_name,
            positive_class_label=positive_class_label,
            negative_class_label=negative_class_label,
            class_labels=class_labels,
        )

        items = []

        if custom_predict_path:
            _, extension = os.path.splitext(custom_predict_path)
            items.append((custom_predict_path, "custom{}".format(extension)))
        if other_file_names:
            for other_file_name in other_file_names:
                other_file_path = os.path.join(BASE_FIXTURE_DIR, ARTIFACT_DIR, other_file_name)
                items.append((other_file_path, other_file_name))
        for artifact_name in artifact_names:
            artifact_path = os.path.join(BASE_FIXTURE_DIR, ARTIFACT_DIR, artifact_name)
            items.append((artifact_path, artifact_name))

        model_version = dr.CustomModelVersion.create_clean(
            custom_model_id=custom_model.id, base_environment_id=base_environment_id, files=items
        )

        return custom_model.id, model_version.id

    @pytest.fixture(scope="session")
    def sklearn_binary_custom_model(self, sklearn_drop_in_env):
        env_id, _ = sklearn_drop_in_env
        return self.make_custom_model(
            "sklearn_bin.pkl",
            env_id,
            "Iris-setosa",
            "Iris-versicolor",
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
            target_name="Species",
        )

    @pytest.fixture(scope="session")
    def sklearn_regression_custom_model(self, sklearn_drop_in_env):
        env_id, _ = sklearn_drop_in_env
        return self.make_custom_model(
            "sklearn_reg.pkl",
            env_id,
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
            target_name="MEDV",
        )

    @pytest.fixture(scope="session")
    def sklearn_multiclass_custom_model(self, sklearn_drop_in_env):
        env_id, _ = sklearn_drop_in_env
        return self.make_custom_model(
            "sklearn_multi.pkl",
            env_id,
            class_labels=["GALAXY", "QSO", "STAR"],
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
            target_name="class",
        )

    @pytest.fixture(scope="session")
    def keras_binary_custom_model(self, keras_drop_in_env):
        env_id, _ = keras_drop_in_env
        return self.make_custom_model(
            "keras_bin.h5",
            env_id,
            "yes",
            "no",
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def keras_regression_custom_model(self, keras_drop_in_env):
        env_id, _ = keras_drop_in_env
        return self.make_custom_model(
            "keras_reg.h5", env_id, custom_predict_path=CUSTOM_PREDICT_PY_PATH
        )

    @pytest.fixture(scope="session")
    def torch_binary_custom_model(self, pytorch_drop_in_env):
        env_id, _ = pytorch_drop_in_env
        return self.make_custom_model(
            "torch_bin.pth",
            env_id,
            "yes",
            "no",
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
            other_file_names=["PyTorch.py"],
        )

    @pytest.fixture(scope="session")
    def torch_regression_custom_model(self, pytorch_drop_in_env):
        env_id, _ = pytorch_drop_in_env
        return self.make_custom_model(
            "torch_reg.pth",
            env_id,
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
            other_file_names=["PyTorch.py"],
        )

    @pytest.fixture(scope="session")
    def xgb_binary_custom_model(self, xgboost_drop_in_env):
        env_id, _ = xgboost_drop_in_env
        return self.make_custom_model(
            "xgb_bin.pkl",
            env_id,
            "yes",
            "no",
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def xgb_regression_custom_model(self, xgboost_drop_in_env):
        env_id, _ = xgboost_drop_in_env
        return self.make_custom_model(
            "xgb_reg.pkl", env_id, custom_predict_path=CUSTOM_PREDICT_PY_PATH
        )

    @pytest.fixture(scope="session")
    def python_multi_artifact_regression_custom_model(self, sklearn_drop_in_env):
        env_id, _ = sklearn_drop_in_env
        return self.make_custom_model(
            ["sklearn_bin.pkl", "sklearn_reg.pkl"],
            env_id,
            custom_predict_path=CUSTOM_LOAD_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def bad_python_multi_artifact_binary_custom_model(self, sklearn_drop_in_env):
        env_id, _ = sklearn_drop_in_env
        return self.make_custom_model(
            ["sklearn_bin.pkl", "sklearn_reg.pkl"],
            env_id,
            "Iris-setosa",
            "Iris-versicolor",
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def java_binary_custom_model(self, java_drop_in_env):
        env_id, _ = java_drop_in_env
        return self.make_custom_model(
            "java_bin.jar",
            env_id,
            "Iris-setosa",
            "Iris-versicolor",
            artifact_only=True,
            target_name="Species",
        )

    @pytest.fixture(scope="session")
    def java_regression_custom_model(self, java_drop_in_env):
        env_id, _ = java_drop_in_env
        return self.make_custom_model(
            "java_reg.jar", env_id, artifact_only=True, target_name="MEDV"
        )

    @pytest.fixture(scope="session")
    def r_binary_custom_model(self, r_drop_in_env):
        env_id, _ = r_drop_in_env
        return self.make_custom_model(
            "r_bin.rds",
            env_id,
            "Iris-setosa",
            "Iris-versicolor",
            custom_predict_path=CUSTOM_PREDICT_R_PATH,
            target_name="Species",
        )

    @pytest.fixture(scope="session")
    def r_regression_custom_model(self, r_drop_in_env):
        env_id, _ = r_drop_in_env
        return self.make_custom_model(
            "r_reg.rds",
            env_id,
            custom_predict_path=CUSTOM_PREDICT_R_PATH,
            target_name="MEDV",
        )

    @pytest.fixture(scope="session")
    def r_multi_artifact_binary_custom_model(self, r_drop_in_env):
        env_id, _ = r_drop_in_env
        return self.make_custom_model(
            ["r_bin.rds", "r_reg.rds"],
            env_id,
            "Iris-setosa",
            "Iris-versicolor",
            custom_predict_path=CUSTOM_LOAD_PREDICT_R_PATH,
        )

    @pytest.fixture(scope="session")
    def bad_r_multi_artifact_binary_custom_model(self, r_drop_in_env):
        env_id, _ = r_drop_in_env
        return self.make_custom_model(
            ["r_bin.rds", "r_reg.rds"],
            env_id,
            "Iris-setosa",
            "Iris-versicolor",
            custom_predict_path=CUSTOM_PREDICT_R_PATH,
        )

    @pytest.mark.parametrize(
        "env, model, test_data_id",
        [
            ("java_drop_in_env", "java_binary_custom_model", "binary_testing_data"),
            ("sklearn_drop_in_env", "sklearn_binary_custom_model", "binary_testing_data"),
            ("keras_drop_in_env", "keras_binary_custom_model", "binary_testing_data"),
            ("pytorch_drop_in_env", "torch_binary_custom_model", "binary_testing_data"),
            ("xgboost_drop_in_env", "xgb_binary_custom_model", "binary_testing_data"),
            ("r_drop_in_env", "r_binary_custom_model", "binary_testing_data"),
            ("java_drop_in_env", "java_regression_custom_model", "regression_testing_data"),
            ("sklearn_drop_in_env", "sklearn_regression_custom_model", "regression_testing_data"),
            ("keras_drop_in_env", "keras_regression_custom_model", "regression_testing_data"),
            ("pytorch_drop_in_env", "torch_regression_custom_model", "regression_testing_data"),
            ("xgboost_drop_in_env", "xgb_regression_custom_model", "regression_testing_data"),
            ("r_drop_in_env", "r_regression_custom_model", "regression_testing_data"),
            (
                "sklearn_drop_in_env",
                "python_multi_artifact_regression_custom_model",
                "regression_testing_data",
            ),
            ("r_drop_in_env", "r_multi_artifact_binary_custom_model", "binary_testing_data"),
            ("sklearn_drop_in_env", "sklearn_multiclass_custom_model", "multiclass_testing_data"),
        ],
    )
    def test_drop_in_environments(self, request, env, model, test_data_id):
        env_id, env_version_id = request.getfixturevalue(env)
        model_id, model_version_id = request.getfixturevalue(model)
        test_data_id = request.getfixturevalue(test_data_id)

        test = dr.CustomModelTest.create(
            custom_model_id=model_id,
            custom_model_version_id=model_version_id,
            dataset_id=test_data_id,
        )

        assert test.overall_status == "succeeded"

    @pytest.mark.parametrize(
        "env, model, test_data_id",
        [
            ("java_drop_in_env", "java_binary_custom_model", "binary_testing_data"),
            ("java_drop_in_env", "java_regression_custom_model", "regression_testing_data"),
            ("sklearn_drop_in_env", "sklearn_binary_custom_model", "binary_testing_data"),
            ("sklearn_drop_in_env", "sklearn_regression_custom_model", "regression_testing_data"),
            ("r_drop_in_env", "r_binary_custom_model", "binary_testing_data"),
            ("r_drop_in_env", "r_regression_custom_model", "regression_testing_data"),
        ],
    )
    def test_feature_impact(self, request, env, model, test_data_id):
        env_id, env_version_id = request.getfixturevalue(env)
        model_id, model_version_id = request.getfixturevalue(model)
        test_data_id = request.getfixturevalue(test_data_id)

        model_image = dr.CustomInferenceImage.create(
            model_id, model_version_id, env_id, env_version_id
        )
        model = dr.CustomInferenceModel.get(model_id)
        model.assign_training_data(test_data_id)
        model_image.calculate_feature_impact()

        test_passed = False
        error_message = ""
        for i in range(300):
            try:
                model_image.get_feature_impact()
                test_passed = True
                break
            except (dr.errors.ClientError, dr.errors.ClientError) as e:
                error_message = "get_feature_impact() response: " + str(e)
            time.sleep(1)

        assert test_passed, "Feature impact test has timed out. " + error_message

    @pytest.mark.parametrize(
        "env, model, test_data_id",
        [
            (
                "sklearn_drop_in_env",
                "bad_python_multi_artifact_binary_custom_model",
                "binary_testing_data",
            ),
            ("r_drop_in_env", "bad_r_multi_artifact_binary_custom_model", "binary_testing_data"),
        ],
    )
    def test_fail_multi_artifacts(self, request, env, model, test_data_id):
        env_id, env_version_id = request.getfixturevalue(env)
        model_id, model_version_id = request.getfixturevalue(model)
        test_data_id = request.getfixturevalue(test_data_id)

        test = dr.CustomModelTest.create(
            custom_model_id=model_id,
            custom_model_version_id=model_version_id,
            environment_id=env_id,
            environment_version_id=env_version_id,
            dataset_id=test_data_id,
        )
        assert test.overall_status == "failed"
