from __future__ import absolute_import

import os
import time

import datarobot.dse as dse
import pytest
from datarobot.dse.enums import CustomModelType, DatasetCategory
from datarobot.dse.errors import AsyncTaskFailureError

BASE_TEMPLATE_ENV_DIR = "public_dropin_environments"
BASE_FIXTURE_DIR = "tests/fixtures"
BASE_DATASET_DIR = "tests/testdata"
ARTIFACT_DIR = "drop_in_model_artifacts"
CUSTOM_PREDICT_PY_PATH = os.path.join(BASE_FIXTURE_DIR, "custom.py")
CUSTOM_PREDICT_R_PATH = os.path.join(BASE_FIXTURE_DIR, "custom.R")
CUSTOM_LOAD_PREDICT_PY_PATH = os.path.join(BASE_FIXTURE_DIR, "load_model_custom.py")
CUSTOM_LOAD_PREDICT_R_PATH = os.path.join(BASE_FIXTURE_DIR, "load_model_custom.R")


class TestDropInEnvironments(object):
    def make_custom_model(
        self,
        client,
        artifact_names,
        positive_class_label=None,
        negative_class_label=None,
        custom_predict_path=None,
        other_file_names=None,
        artifact_only=False,
        target_name="target",
    ):
        """

        Parameters
        ----------
        client: dse.DSEClient
        artifact_names: Union[List[str], str]
            a singular filename or list of artifact filenames from the
            drop_in_model_artifacts directory to include
        positive_class_label: str or None
        negative_class_label: str or None
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
        custom_models_url = "http://localhost/api/v2/customModels/"

        if artifact_only and (
            isinstance(artifact_names, list) or custom_predict_path or other_file_names
        ):
            raise ValueError(
                "artifact_only=True cannot be used with a list of artifact_names, "
                "a custom_predict_path, or other_file_names"
            )

        if not isinstance(artifact_names, list):
            artifact_names = [artifact_names]

        model_args = dict(
            customModelType=CustomModelType.INFERENCE.value,
            name=artifact_names[0],
            targetName=target_name,
        )

        if positive_class_label and negative_class_label:
            model_args["positiveClassLabel"] = positive_class_label
            model_args["negativeClassLabel"] = negative_class_label
            model_args["supportsBinaryClassification"] = True
        else:
            model_args["supportsRegression"] = True

        response = client.post(custom_models_url, json=model_args)
        custom_model_id = response.json()["id"]
        version_url = "{}/{}/versions".format(custom_models_url, custom_model_id)

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

        file_paths = []
        files_actual = []
        for file_actual, file_path in items:
            file_paths.append(("filePath", file_path))
            files_actual.append(("file", open(file_actual, "rb")))

        version = client.post(version_url, data=file_paths, files=files_actual)

        return custom_model_id, version.json()["id"]

    @pytest.fixture(scope="session")
    def dse_client(self, pytestconfig):
        client = dse.DSEClient(
            base_url="http://localhost/api/v2",
            token=os.environ["DATAROBOT_API_TOKEN"],
            username=pytestconfig.user_username,
        )
        return client

    @pytest.fixture(scope="session")
    def java_drop_in_env(self, dse_client):
        env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "java_codegen")
        environment = dse_client.execution_environments.create("java_drop_in")
        environment_version = environment.versions.sync_create(environment_path=env_dir)
        return environment.id, environment_version.id

    @pytest.fixture(scope="session")
    def sklearn_drop_in_env(self, dse_client):
        env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_sklearn")
        environment = dse_client.execution_environments.create("python3_sklearn")
        environment_version = environment.versions.sync_create(environment_path=env_dir)
        return environment.id, environment_version.id

    @pytest.fixture(scope="session")
    def xgboost_drop_in_env(self, dse_client):
        env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_xgboost")
        environment = dse_client.execution_environments.create("python3_xgboost")
        environment_version = environment.versions.sync_create(environment_path=env_dir)
        return environment.id, environment_version.id

    @pytest.fixture(scope="session")
    def pytorch_drop_in_env(self, dse_client):
        env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_pytorch")
        environment = dse_client.execution_environments.create("python3_pytorch")
        environment_version = environment.versions.sync_create(environment_path=env_dir)
        return environment.id, environment_version.id

    @pytest.fixture(scope="session")
    def keras_drop_in_env(self, dse_client):
        env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_keras")
        environment = dse_client.execution_environments.create("python3_keras")
        environment_version = environment.versions.sync_create(environment_path=env_dir)
        return environment.id, environment_version.id

    @pytest.fixture(scope="session")
    def r_drop_in_env(self, dse_client):
        env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "r_lang")
        environment = dse_client.execution_environments.create("r_drop_in")
        environment_version = environment.versions.sync_create(
            environment_path=env_dir, timeout=5400,  # R dependency management is an abomination
        )
        return environment.id, environment_version.id

    @pytest.fixture(scope="session")
    def sklearn_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            "sklearn_bin.pkl",
            "Iris-setosa",
            "Iris-versicolor",
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
            target_name="Species",
        )

    @pytest.fixture(scope="session")
    def sklearn_regression_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            "sklearn_reg.pkl",
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
            target_name="MEDV",
        )

    @pytest.fixture(scope="session")
    def keras_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client, "keras_bin.h5", "yes", "no", custom_predict_path=CUSTOM_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def keras_regression_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client, "keras_reg.h5", custom_predict_path=CUSTOM_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def torch_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client, "torch_bin.pth", "yes", "no", CUSTOM_PREDICT_PY_PATH, ["PyTorch.py"],
        )

    @pytest.fixture(scope="session")
    def torch_regression_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            "torch_reg.pth",
            custom_predict_path=CUSTOM_PREDICT_PY_PATH,
            other_file_names=["PyTorch.py"],
        )

    @pytest.fixture(scope="session")
    def xgb_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client, "xgb_bin.pkl", "yes", "no", custom_predict_path=CUSTOM_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def xgb_regression_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client, "xgb_reg.pkl", custom_predict_path=CUSTOM_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def python_multi_artifact_regression_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            ["sklearn_bin.pkl", "sklearn_reg.pkl"],
            custom_predict_path=CUSTOM_LOAD_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def bad_python_multi_artifact_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            ["sklearn_bin.pkl", "sklearn_reg.pkl"],
            "Iris-setosa",
            "Iris-versicolor",
            CUSTOM_PREDICT_PY_PATH,
        )

    @pytest.fixture(scope="session")
    def java_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            "java_bin.jar",
            "Iris-setosa",
            "Iris-versicolor",
            artifact_only=True,
            target_name="Species",
        )

    @pytest.fixture(scope="session")
    def java_regression_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client, "java_reg.jar", artifact_only=True, target_name="MEDV",
        )

    @pytest.fixture(scope="session")
    def r_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            "r_bin.rds",
            "Iris-setosa",
            "Iris-versicolor",
            CUSTOM_PREDICT_R_PATH,
            target_name="Species",
        )

    @pytest.fixture(scope="session")
    def r_regression_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client, "r_reg.rds", custom_predict_path=CUSTOM_PREDICT_R_PATH, target_name="MEDV",
        )

    @pytest.fixture(scope="session")
    def r_multi_artifact_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            ["r_bin.rds", "r_reg.rds"],
            "Iris-setosa",
            "Iris-versicolor",
            CUSTOM_LOAD_PREDICT_R_PATH,
        )

    @pytest.fixture(scope="session")
    def bad_r_multi_artifact_binary_custom_model(self, dse_client):
        return self.make_custom_model(
            dse_client,
            ["r_bin.rds", "r_reg.rds"],
            "Iris-setosa",
            "Iris-versicolor",
            CUSTOM_PREDICT_R_PATH,
        )

    @pytest.fixture(scope="session")
    def binary_testing_data(self, dse_client):
        dataset = dse_client.datasets.sync_create(
            dataset_path=os.path.join(BASE_DATASET_DIR, "iris_binary_training.csv"),
            categories=DatasetCategory.PREDICTION,
        )
        return dataset.dataset_id

    @pytest.fixture(scope="session")
    def regression_testing_data(self, dse_client):
        dataset = dse_client.datasets.sync_create(
            dataset_path=os.path.join(BASE_DATASET_DIR, "boston_housing.csv"),
            categories=DatasetCategory.PREDICTION,
        )
        return dataset.dataset_id

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
        ],
    )
    def test_drop_in_environments(self, dse_client, request, env, model, test_data_id):
        env_id, env_version_id = request.getfixturevalue(env)
        model_id, model_version_id = request.getfixturevalue(model)
        test_data_id = request.getfixturevalue(test_data_id)

        payload = {
            "custom_model_id": model_id,
            "dataset_id": test_data_id,
            "custom_model_version_id": model_version_id,
            "environment_id": env_id,
            "environment_version_id": env_version_id,
        }
        test = dse_client.custom_model_tests.sync_create(timeout=600, **payload)

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
    def test_feature_impact(self, dse_client, request, env, model, test_data_id):
        env_id, env_version_id = request.getfixturevalue(env)
        model_id, model_version_id = request.getfixturevalue(model)
        test_data_id = request.getfixturevalue(test_data_id)

        attrs = {
            "custom_model_id": model_id,
            "custom_model_version_id": model_version_id,
            "environment_id": env_id,
            "environment_version_id": env_version_id,
        }
        model_image = dse_client.custom_model_images.create(**attrs)
        dataset = dse_client.datasets.get(test_data_id)
        custom_model = dse_client.custom_models.get(model_id)

        # training dataset assignment triggers feature impact test, because model image already exists
        custom_model.assign_training_data(dataset)

        url = "http://localhost/api/v2/customInferenceImages/{}/featureImpact/".format(
            model_image.id
        )
        test_passed = False
        status_404_count = 0
        for i in range(600):
            response = dse_client.get(url)
            if response.ok:
                test_passed = True
                break
            elif response.status_code == 404:
                status_404_count += 1
                assert status_404_count < 30, (
                    "Feature impact test has failed with response: " + response.text
                )
            time.sleep(1)

        assert test_passed, "Feature impact test has timed out"

    @pytest.mark.parametrize(
        "env, model, test_data_id",
        [
            (
                "sklearn_drop_in_env",
                "bad_python_multi_artifact_binary_custom_model",
                "binary_testing_data",
            ),
            ("r_drop_in_env", "bad_r_multi_artifact_binary_custom_model", "binary_testing_data",),
        ],
    )
    def test_fail_multi_artifacts(self, dse_client, request, env, model, test_data_id):
        env_id, env_version_id = request.getfixturevalue(env)
        model_id, model_version_id = request.getfixturevalue(model)
        test_data_id = request.getfixturevalue(test_data_id)

        payload = {
            "custom_model_id": model_id,
            "dataset_id": test_data_id,
            "custom_model_version_id": model_version_id,
            "environment_id": env_id,
            "environment_version_id": env_version_id,
        }
        with pytest.raises(AsyncTaskFailureError):
            dse_client.custom_model_tests.sync_create(timeout=600, **payload)
