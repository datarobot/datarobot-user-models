import os
import re
import responses
import strictyaml
from datarobot_drum.drum.exceptions import DrumCommonException
import pytest
from tempfile import NamedTemporaryFile

import pandas as pd

from datarobot_drum.drum.drum import possibly_intuit_order
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.push import push_inference, push_training, schema


class TestOrderIntuition(object):
    tests_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
    binary_filename = os.path.join(tests_data_path, "iris_binary_training.csv")
    regression_filename = os.path.join(tests_data_path, "boston_housing.csv")
    one_target_filename = os.path.join(tests_data_path, "one_target.csv")

    def test_colname(self):
        classes = possibly_intuit_order(self.binary_filename, target_col_name="Species")
        assert set(classes) == {"Iris-versicolor", "Iris-setosa"}

    def test_colfile(self):
        with NamedTemporaryFile() as target_file:
            df = pd.read_csv(self.binary_filename)
            with open(target_file.name, "w") as f:
                target_series = df["Species"]
                target_series.to_csv(f, index=False, header="Target")

            classes = possibly_intuit_order(self.binary_filename, target_data_file=target_file.name)
            assert set(classes) == {"Iris-versicolor", "Iris-setosa"}

    def test_regression(self):
        classes = possibly_intuit_order(self.regression_filename, target_col_name="MEDV")
        assert set(classes) == {None, None}

    def test_badfile(self):
        with pytest.raises(DrumCommonException):
            possibly_intuit_order(self.one_target_filename, target_col_name="Species")


class TestValidatePredictions(object):
    def test_add_to_one_happy(self):
        positive_label = "poslabel"
        negative_label = "neglabel"
        adapter = PythonModelAdapter(model_dir=None)
        df = pd.DataFrame({positive_label: [0.1, 0.2, 0.3], negative_label: [0.9, 0.8, 0.7]})
        adapter._validate_predictions(
            to_validate=df,
            positive_class_label=positive_label,
            negative_class_label=negative_label,
        )

    def test_add_to_one_sad(self):
        positive_label = "poslabel"
        negative_label = "neglabel"
        adapter = PythonModelAdapter(model_dir=None)
        df = pd.DataFrame({positive_label: [1, 1, 1], negative_label: [-1, 0, 0]})
        with pytest.raises(ValueError):
            adapter._validate_predictions(
                to_validate=df,
                positive_class_label=positive_label,
                negative_class_label=negative_label,
            )


modelID = "5f1f15a4d6111f01cb7f91f"
environmentID = "5e8c889607389fe0f466c72d"
projectID = "abc123"

inference_metadata_yaml = """
name: drumpush-regression
type: inference
targetType: regression
modelID: {modelID}
environmentID: {environmentID}
inferenceModel:
  targetName: MEDV
validation:
  inputData: hi
  targetName: MEDV
""".format(
    modelID=modelID, environmentID=environmentID
)

training_metadata_yaml = """
name: drumpush-regression
type: training
targetType: regression
modelID: {modelID}
environmentID: {environmentID}
validation:
  inputData: hi
  targetName: MEDV
""".format(
    modelID=modelID, environmentID=environmentID
)


training_metadata_yaml_with_proj = """
name: drumpush-regression
type: training
targetType: regression
modelID: {modelID}
environmentID: {environmentID}
trainingModel:
  trainOnProject: {projectID}
validation:
  inputData: hi
  targetName: MEDV
""".format(
    modelID=modelID, environmentID=environmentID, projectID=projectID
)


version_response = {
    "id": "1",
    "custom_model_id": "1",
    "version_minor": 1,
    "version_major": 1,
    "is_frozen": False,
    "items": [{"id": "1", "file_name": "hi", "file_path": "hi", "file_source": "hi"}],
}


def version_mocks():
    responses.add(
        responses.GET,
        "http://yess/version/",
        json={"major": 2, "versionString": "2.21", "minor": 21},
        status=200,
    )
    responses.add(
        responses.POST,
        "http://yess/customModels/{}/versions/".format(modelID),
        json=version_response,
        status=200,
    )


def mock_get_model(model_type="training"):
    responses.add(
        responses.GET,
        "http://yess/customModels/{}/".format(modelID),
        json={
            "customModelType": model_type,
            "id": "1",
            "name": "1",
            "description": "1",
            "supports_binary_classification": False,
            "supports_regression": True,
            "deployments_count": "1",
            "created_by": "1",
            "updated": "1",
            "created": "1",
            "latestVersion": version_response,
        },
    )


def mock_post_blueprint():
    responses.add(
        responses.POST,
        "http://yess/customTrainingBlueprints/",
        json={
            "blueprint_id": "1",
            "custom_model": {"id": "1", "name": "1"},
            "custom_model_version": {"id": "1", "label": "1"},
            "execution_environment": {"id": "1", "name": "1"},
            "execution_environment_version": {"id": "1", "label": "1"},
            "training_history": [],
        },
    )


def mock_get_env():
    responses.add(
        responses.GET,
        "http://yess/executionEnvironments/{}/".format(environmentID),
        json={
            "id": "1",
            "name": "hi",
            "latestVersion": {"id": "hii", "environment_id": environmentID, "build_status": "yes"},
        },
    )


def mock_train_model():
    responses.add(
        responses.POST,
        "http://yess/projects/{}/models/".format(projectID),
        json={},
        adding_headers={"Location": "the/moon"},
    )


@responses.activate
@pytest.mark.parametrize(
    "config_yaml",
    [training_metadata_yaml, training_metadata_yaml_with_proj, inference_metadata_yaml,],
)
def test_push(config_yaml):
    config = strictyaml.load(config_yaml, schema).data

    version_mocks()
    mock_post_blueprint()
    mock_get_model(model_type=config["type"])
    mock_get_env()
    mock_train_model()
    push_fn = push_training if config["type"] == "training" else push_inference
    push_fn(config, code_dir="", endpoint="http://Yess", token="okay")

    calls = responses.calls
    assert (
        calls[1].request.path_url == "/customModels/{}/versions/".format(modelID)
        and calls[1].request.method == "POST"
    )
    if push_fn == push_training:
        assert (
            calls[2].request.path_url == "/customModels/{}/".format(modelID)
            and calls[2].request.method == "GET"
        )
        assert (
            calls[3].request.path_url == "/customTrainingBlueprints/"
            and calls[3].request.method == "POST"
        )
        if "trainingModel" in config:
            assert (
                calls[4].request.path_url == "/projects/abc123/models/"
                and calls[4].request.method == "POST"
            )
            assert len(calls) == 5
        else:
            assert len(calls) == 4
    else:
        assert len(calls) == 2
