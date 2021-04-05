import json
import os
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import dedent

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pyarrow
import pytest
import responses

from datarobot_drum.drum.drum import (
    possibly_intuit_order,
    output_in_code_dir,
    create_custom_inference_model_folder,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.push import _push_inference, _push_training
from datarobot_drum.drum.common import (
    read_model_metadata_yaml,
    MODEL_CONFIG_FILENAME,
    TargetType,
    validate_config_fields,
    ModelMetadataKeys,
)
from datarobot_drum.drum.utils import StructuredInputReadUtils


class TestOrderIntuition:
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

    def test_badfile(self):
        with pytest.raises(DrumCommonException):
            possibly_intuit_order(self.one_target_filename, target_col_name="Species")

    def test_unsupervised(self):
        classes = possibly_intuit_order(
            self.regression_filename, target_col_name="MEDV", is_anomaly=True
        )
        assert classes is None


class TestValidatePredictions:
    def test_add_to_one_happy(self):
        positive_label = "poslabel"
        negative_label = "neglabel"
        adapter = PythonModelAdapter(model_dir=None, target_type=TargetType.BINARY)
        df = pd.DataFrame({positive_label: [0.1, 0.2, 0.3], negative_label: [0.9, 0.8, 0.7]})
        adapter._validate_predictions(
            to_validate=df, class_labels=[positive_label, negative_label],
        )

    def test_add_to_one_sad(self):
        positive_label = "poslabel"
        negative_label = "neglabel"
        adapter = PythonModelAdapter(model_dir=None, target_type=TargetType.BINARY)
        df = pd.DataFrame({positive_label: [1, 1, 1], negative_label: [-1, 0, 0]})
        with pytest.raises(ValueError):
            adapter._validate_predictions(
                to_validate=df, class_labels=[positive_label, negative_label],
            )


modelID = "5f1f15a4d6111f01cb7f91f"
environmentID = "5e8c889607389fe0f466c72d"
projectID = "abc123"


@pytest.fixture
def inference_metadata_yaml():
    return dedent(
        """
        name: drumpush-regression
        type: inference
        targetType: regression
        environmentID: {environmentID}
        inferenceModel:
          targetName: MEDV
        validation:
          input: hello
        """
    ).format(environmentID=environmentID)


@pytest.fixture
def multiclass_labels():
    return ["GALAXY", "QSO", "STAR"]


@pytest.fixture
def inference_multiclass_metadata_yaml(multiclass_labels):
    return dedent(
        """
        name: drumpush-multiclass
        type: inference
        targetType: multiclass
        environmentID: {}
        inferenceModel:
          targetName: class
          classLabels:
            - {}
            - {}
            - {}
        validation:
          input: hello
        """
    ).format(environmentID, *multiclass_labels)


@pytest.fixture
def inference_multiclass_metadata_yaml_label_file(multiclass_labels):
    with NamedTemporaryFile(mode="w+") as f:
        f.write("\n".join(multiclass_labels))
        f.flush()
        yield dedent(
            """
            name: drumpush-multiclass
            type: inference
            targetType: multiclass
            environmentID: {}
            inferenceModel:
              targetName: class
              classLabelsFile: {}
            validation:
              input: hello
            """
        ).format(environmentID, f.name)


@pytest.fixture
def training_metadata_yaml():
    return dedent(
        """
        name: drumpush-regression
        type: training
        targetType: regression
        environmentID: {environmentID}
        validation:
           input: hello 
        """
    ).format(environmentID=environmentID)


@pytest.fixture
def training_metadata_yaml_with_proj():
    return dedent(
        """
        name: drumpush-regression
        type: training
        targetType: regression
        environmentID: {environmentID}
        trainingModel:
            trainOnProject: {projectID}
        validation:
            input: hello 
        """
    ).format(environmentID=environmentID, projectID=projectID)


@pytest.fixture
def custom_predictor_metadata_yaml():
    return dedent(
        """
        name: model-with-custom-java-predictor
        type: inference
        targetType: regression
        customPredictor:
           arbitraryField: This info is read directly by a custom predictor
        """
    )


version_response = {
    "id": "1",
    "custom_model_id": "1",
    "version_minor": 1,
    "version_major": 1,
    "is_frozen": False,
    "items": [{"id": "1", "file_name": "hi", "file_path": "hi", "file_source": "hi"}],
}


@pytest.mark.parametrize(
    "config_yaml",
    [
        "custom_predictor_metadata_yaml",
        "training_metadata_yaml",
        "training_metadata_yaml_with_proj",
        "inference_metadata_yaml",
        "inference_multiclass_metadata_yaml",
        "inference_multiclass_metadata_yaml_label_file",
    ],
)
@pytest.mark.parametrize("existing_model_id", [None])
def test_yaml_metadata(request, config_yaml, existing_model_id, tmp_path):
    config_yaml = request.getfixturevalue(config_yaml)
    if existing_model_id:
        config_yaml = config_yaml + "\nmodelID: {}".format(existing_model_id)

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)
    read_model_metadata_yaml(tmp_path)


def test_yaml_metadata_missing_fields(custom_predictor_metadata_yaml, tmp_path):
    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(custom_predictor_metadata_yaml)
    conf = read_model_metadata_yaml(tmp_path)
    with pytest.raises(
        DrumCommonException, match="Missing keys: \['validation', 'environmentID'\]"
    ):
        validate_config_fields(
            conf,
            ModelMetadataKeys.CUSTOM_PREDICTOR,
            ModelMetadataKeys.VALIDATION,
            ModelMetadataKeys.ENVIRONMENT_ID,
        )


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


def mock_get_model(model_type="training", target_type="Regression"):
    body = {
        "customModelType": model_type,
        "id": modelID,
        "name": "1",
        "description": "1",
        "targetType": target_type,
        "deployments_count": "1",
        "created_by": "1",
        "updated": "1",
        "created": "1",
        "latestVersion": version_response,
    }
    if model_type == "inference":
        body["language"] = "Python"
        body["trainingDataAssignmentInProgress"] = False
    responses.add(
        responses.GET, "http://yess/customModels/{}/".format(modelID), json=body,
    )
    responses.add(
        responses.POST, "http://yess/customModels/".format(modelID), json=body,
    )


def mock_post_blueprint():
    responses.add(
        responses.POST,
        "http://yess/customTrainingBlueprints/",
        json={
            "userBlueprintId": "2",
            "custom_model": {"id": "1", "name": "1"},
            "custom_model_version": {"id": "1", "label": "1"},
            "execution_environment": {"id": "1", "name": "1"},
            "execution_environment_version": {"id": "1", "label": "1"},
            "training_history": [],
        },
    )


def mock_post_add_to_repository():
    responses.add(
        responses.POST,
        "http://yess/projects/{}/blueprints/fromUserBlueprint/".format(projectID),
        json={"2": "1"},
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
    responses.add(
        responses.GET,
        "http://yess/projects/{}/modelJobs/the/".format(projectID),
        json={
            "is_blocked": False,
            "id": "55",
            "processes": [],
            "model_type": "fake",
            "project_id": projectID,
            "blueprint_id": "1",
        },
    )


@responses.activate
@pytest.mark.parametrize(
    "config_yaml",
    [
        "training_metadata_yaml",
        "training_metadata_yaml_with_proj",
        "inference_metadata_yaml",
        "inference_multiclass_metadata_yaml",
        "inference_multiclass_metadata_yaml_label_file",
    ],
)
@pytest.mark.parametrize("existing_model_id", [None, modelID])
def test_push(request, config_yaml, existing_model_id, multiclass_labels, tmp_path):
    config_yaml = request.getfixturevalue(config_yaml)
    if existing_model_id:
        config_yaml = config_yaml + "\nmodelID: {}".format(existing_model_id)

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)
    config = read_model_metadata_yaml(tmp_path)

    version_mocks()
    mock_post_blueprint()
    mock_post_add_to_repository()
    mock_get_model(model_type=config["type"], target_type=config["targetType"].capitalize())
    mock_get_env()
    mock_train_model()
    push_fn = _push_training if config["type"] == "training" else _push_inference
    push_fn(config, code_dir="", endpoint="http://Yess", token="okay")

    calls = responses.calls
    if existing_model_id is None:
        assert calls[1].request.path_url == "/customModels/" and calls[1].request.method == "POST"
        if config["targetType"] == TargetType.MULTICLASS.value:
            sent_labels = json.loads(calls[1].request.body)["classLabels"]
            assert sent_labels == multiclass_labels
        call_shift = 1
    else:
        call_shift = 0
    assert (
        calls[call_shift + 1].request.path_url == "/customModels/{}/versions/".format(modelID)
        and calls[call_shift + 1].request.method == "POST"
    )
    if push_fn == _push_training:
        assert (
            calls[call_shift + 2].request.path_url == "/customTrainingBlueprints/"
            and calls[call_shift + 2].request.method == "POST"
        )
        if "trainingModel" in config:
            assert (
                calls[call_shift + 3].request.path_url
                == "/projects/{}/blueprints/fromUserBlueprint/".format(projectID)
                and calls[call_shift + 3].request.method == "POST"
            )
            assert (
                calls[call_shift + 4].request.path_url == "/projects/abc123/models/"
                and calls[call_shift + 4].request.method == "POST"
            )
            assert len(calls) == 6 + call_shift
        else:
            assert len(calls) == 3 + call_shift
    else:
        assert len(calls) == 2 + call_shift


def test_output_in_code_dir():
    code_dir = "/test/code/is/here"
    output_other = "/test/not/code"
    output_code_dir = "/test/code/is/here/output"
    assert not output_in_code_dir(code_dir, output_other)
    assert output_in_code_dir(code_dir, output_code_dir)


def test_output_dir_copy():
    with tempfile.TemporaryDirectory() as tempdir:
        # setup
        file = Path(tempdir, "test.py")
        file.touch()
        Path(tempdir, "__pycache__").mkdir()
        out_dir = Path(tempdir, "out")
        out_dir.mkdir()

        # test
        create_custom_inference_model_folder(tempdir, str(out_dir))
        assert Path(out_dir, "test.py").exists()
        assert not Path(out_dir, "__pycache__").exists()
        assert not Path(out_dir, "out").exists()


def test_read_structured_input_arrow_csv_na_consistency(tmp_path):
    """
    Test that N/A values (None, numpy.nan) are handled consistently when using
    CSV vs Arrow as a prediction payload format.
    1. Make CSV and Arrow prediction payloads from the same dataframe
    2. Read both payloads
    3. Assert the resulting dataframes are equal
    """

    # arrange
    df = pd.DataFrame({"col_int": [1, np.nan, None], "col_obj": ["a", np.nan, None]})

    csv_filename = os.path.join(tmp_path, "X.csv")
    with open(csv_filename, "w") as f:
        f.write(df.to_csv(index=False))

    arrow_filename = os.path.join(tmp_path, "X.arrow")
    with open(arrow_filename, "wb") as f:
        f.write(pyarrow.ipc.serialize_pandas(df).to_pybytes())

    # act
    csv_df = StructuredInputReadUtils.read_structured_input_file_as_df(csv_filename)
    arrow_df = StructuredInputReadUtils.read_structured_input_file_as_df(arrow_filename)

    # assert
    is_nan = lambda x: isinstance(x, float) and np.isnan(x)
    is_none = lambda x: x is None

    assert_frame_equal(csv_df, arrow_df)
    # `assert_frame_equal` doesn't make a difference between None and np.nan.
    # To do an exact comparison, compare None and np.nan "masks".
    assert_frame_equal(csv_df.applymap(is_nan), arrow_df.applymap(is_nan))
    assert_frame_equal(csv_df.applymap(is_none), arrow_df.applymap(is_none))


class TestJavaPredictor:
    # Verifying that correct code branch is taken depending on the data size.
    # As jp object is not properly configured, just check for the expected error message.
    @pytest.mark.parametrize(
        "data_size, error_message",
        [(2, "object has no attribute 'predict'"), (40000, "object has no attribute 'predictCSV'")],
    )
    def test_java_predictor_py4j_data(self, data_size, error_message):
        from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import (
            JavaPredictor,
        )

        jp = JavaPredictor()
        with pytest.raises(AttributeError, match=error_message):
            jp.predict(binary_data=b"d" * data_size)
