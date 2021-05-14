import json
import os
import tempfile
from itertools import permutations
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import List, Union

import numpy as np
import pandas as pd
import scipy.sparse
import yaml
from pandas.testing import assert_frame_equal
import pyarrow
import pytest
import responses
from strictyaml import load, YAMLValidationError

from datarobot_drum.drum.drum import (
    possibly_intuit_order,
    output_in_code_dir,
    create_custom_inference_model_folder,
)
from datarobot_drum.drum.exceptions import DrumCommonException, DrumSchemaValidationException
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.push import _push_inference, _push_training, drum_push
from datarobot_drum.drum.common import (
    read_model_metadata_yaml,
    MODEL_CONFIG_FILENAME,
    TargetType,
    validate_config_fields,
    ModelMetadataKeys,
)
from datarobot_drum.drum.utils import StructuredInputReadUtils

from datarobot_drum.drum.typeschema_validation import (
    get_type_schema_yaml_validator,
    revalidate_typeschema,
    Conditions,
    Values,
    Fields,
    SchemaValidator,
)


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
def inference_binary_metadata_yaml_no_target_name():
    return dedent(
        """
        name: drumpush-binary
        type: inference
        targetType: binary
        environmentID: {environmentID}
        inferenceModel:
          positiveClassLabel: yes
          negativeClassLabel: no
        validation:
          input: hello
        """
    ).format(environmentID=environmentID)


@pytest.fixture
def inference_binary_metadata_no_label():
    return dedent(
        """
        name: drumpush-binary
        type: inference
        targetType: binary
        inferenceModel:
          positiveClassLabel: yes
        """
    )


@pytest.fixture
def multiclass_labels():
    return ["GALAXY", "QSO", "STAR"]


@pytest.fixture
def inference_multiclass_metadata_yaml_no_labels():
    return dedent(
        """
        name: drumpush-multiclass
        type: inference
        targetType: multiclass
        environmentID: {}
        inferenceModel:
          targetName: class
        validation:
          input: hello
        """
    ).format(environmentID)


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
def inference_multiclass_metadata_yaml_labels_and_label_file(multiclass_labels):
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
              classLabels:
                - {}
                - {}
                - {}
            validation:
              input: hello
            """
        ).format(environmentID, f.name, *multiclass_labels)


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


@pytest.mark.parametrize(
    "config_yaml, test_case_number",
    [
        ("custom_predictor_metadata_yaml", 1),
        ("inference_binary_metadata_no_label", 2),
        ("inference_multiclass_metadata_yaml_no_labels", 3),
        ("inference_multiclass_metadata_yaml_labels_and_label_file", 4),
        ("inference_multiclass_metadata_yaml", 100),
        ("inference_multiclass_metadata_yaml_label_file", 100),
    ],
)
def test_yaml_metadata_missing_fields(tmp_path, config_yaml, request, test_case_number):
    config_yaml = request.getfixturevalue(config_yaml)
    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)

    if test_case_number == 1:
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
    elif test_case_number == 2:
        with pytest.raises(DrumCommonException, match="Missing keys: \['negativeClassLabel'\]"):
            read_model_metadata_yaml(tmp_path)
    elif test_case_number == 3:
        with pytest.raises(
            DrumCommonException,
            match="Error - for multiclass classification, either the class labels or a class labels file must be provided in model-metadata.yaml file",
        ):
            read_model_metadata_yaml(tmp_path)
    elif test_case_number == 4:
        with pytest.raises(
            DrumCommonException,
            match="Error - for multiclass classification, either the class labels or a class labels file should be provided in model-metadata.yaml file, but not both",
        ):
            read_model_metadata_yaml(tmp_path)
    elif test_case_number == 100:
        read_model_metadata_yaml(tmp_path)


def test_read_model_metadata_properly_casts_typeschema(tmp_path, training_metadata_yaml):
    config_yaml = training_metadata_yaml + dedent(
        """
        typeSchema:
           input_requirements:
           - field: number_of_columns
             condition: IN
             value:
               - 1
               - 2
           - field: data_types
             condition: EQUALS
             value:
               - NUM
               - TXT
           output_requirements:
           - field: number_of_columns
             condition: IN
             value: 2
           - field: data_types
             condition: EQUALS
             value: NUM
        """
    )
    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)

    yaml_conf = read_model_metadata_yaml(tmp_path)
    output_reqs = yaml_conf["typeSchema"]["output_requirements"]
    input_reqs = yaml_conf["typeSchema"]["input_requirements"]

    value_key = "value"
    expected_as_int_list = next(
        (el for el in input_reqs if el["field"] == "number_of_columns")
    ).get(value_key)
    expected_as_str_list = next((el for el in input_reqs if el["field"] == "data_types")).get(
        value_key
    )
    expected_as_int = next((el for el in output_reqs if el["field"] == "number_of_columns")).get(
        value_key
    )
    expected_as_str = next((el for el in output_reqs if el["field"] == "data_types")).get(value_key)

    assert all(isinstance(el, int) for el in expected_as_int_list)
    assert all(isinstance(el, str) for el in expected_as_str_list)
    assert isinstance(expected_as_str_list, list)

    assert isinstance(expected_as_int, int)
    assert isinstance(expected_as_str, str)


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
        json={"id": "1"},
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


@responses.activate
@pytest.mark.parametrize(
    "config_yaml", ["inference_binary_metadata_yaml_no_target_name",],
)
def test_push_no_target_name_in_yaml(request, config_yaml, tmp_path):
    config_yaml = request.getfixturevalue(config_yaml)
    config_yaml = config_yaml + "\nmodelID: {}".format(modelID)

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)
    config = read_model_metadata_yaml(tmp_path)

    from argparse import Namespace

    options = Namespace(code_dir=tmp_path, model_config=config)
    with pytest.raises(DrumCommonException, match="Missing keys: \['targetName'\]"):
        drum_push(options)


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


def input_requirements_yaml(
    field: Fields, condition: Conditions, values: List[Union[int, Values]]
) -> str:
    yaml_dict = get_yaml_dict(condition, field, values, top_requirements="input_requirements")
    return yaml.dump(yaml_dict)


def output_requirements_yaml(
    field: Fields, condition: Conditions, values: List[Union[int, Values]]
) -> str:
    yaml_dict = get_yaml_dict(condition, field, values, top_requirements="output_requirements")
    return yaml.dump(yaml_dict)


def get_yaml_dict(condition, field, values, top_requirements) -> dict:
    def _get_val(value):
        if isinstance(value, Values):
            return str(value)
        return value

    if len(values) == 1:
        new_vals = _get_val(values[0])
    else:
        new_vals = [_get_val(el) for el in values]
    yaml_dict = {
        top_requirements: [{"field": str(field), "condition": str(condition), "value": new_vals}]
    }
    return yaml_dict


def get_data(dataset_name: str) -> pd.DataFrame:
    tests_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
    return pd.read_csv(os.path.join(tests_data_path, dataset_name))


CATS_AND_DOGS = get_data("cats_dogs_small_training.csv")
TEN_K_DIABETES = get_data("10k_diabetes.csv")
IRIS_BINARY = get_data("iris_binary_training.csv")
LENDING_CLUB = get_data("lending_club_reduced.csv")


@pytest.fixture
def lending_club():
    return LENDING_CLUB.copy()


@pytest.fixture
def iris_binary():
    return IRIS_BINARY.copy()


@pytest.fixture
def ten_k_diabetes():
    return TEN_K_DIABETES.copy()


@pytest.fixture
def cats_and_dogs():
    return CATS_AND_DOGS.copy()


class TestSchemaValidator:
    tests_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))

    @pytest.fixture
    def data(self, iris_binary):
        yield iris_binary

    @pytest.fixture
    def missing_data(self, data):
        df = data.copy(deep=True)
        for col in df.columns:
            df.loc[df.sample(frac=0.1).index, col] = pd.np.nan
        yield df

    @pytest.fixture
    def sparse_df(self):
        yield pd.DataFrame.sparse.from_spmatrix(scipy.sparse.eye(10))

    @pytest.fixture
    def dense_df(self):
        yield pd.DataFrame(np.zeros((10, 10)))

    @staticmethod
    def yaml_str_to_schema_dict(yaml_str: str) -> dict:
        """this emulates how we cast a yaml to a dict for validation in
        `datarobot_drum.drum.common.read_model_metadata_yaml` and these assumptions
        are tested in: `tests.drum.test_units.test_read_model_metadata_properly_casts_typeschema` """
        schema = load(yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(schema)
        return schema.data

    @pytest.mark.parametrize(
        "condition, value, passing_dataset, passing_target, failing_dataset, failing_target",
        [
            (
                Conditions.IN,
                [Values.CAT, Values.NUM],
                "iris_binary",
                "SepalLengthCm",
                "ten_k_diabetes",
                "readmitted",
            ),
            (
                Conditions.EQUALS,
                [Values.NUM],
                "iris_binary",
                "Species",
                "ten_k_diabetes",
                "readmitted",
            ),
            (
                Conditions.NOT_IN,
                [Values.TXT],
                "iris_binary",
                "SepalLengthCm",
                "ten_k_diabetes",
                "readmitted",
            ),
            (
                Conditions.NOT_EQUALS,
                [Values.CAT],
                "iris_binary",
                "Species",
                "lending_club",
                "is_bad",
            ),
            (
                Conditions.EQUALS,
                [Values.IMG],
                "cats_and_dogs",
                "class",
                "ten_k_diabetes",
                "readmitted",
            ),
        ],
        ids=lambda x: str([str(el) for el in x]) if isinstance(x, list) else str(x),
    )
    def test_data_types(
        self,
        condition,
        value,
        passing_dataset,
        passing_target,
        failing_dataset,
        failing_target,
        request,
    ):
        yaml_str = input_requirements_yaml(Fields.DATA_TYPES, condition, value)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        # good_data = pd.read_csv(os.path.join(self.tests_data_path, passing_dataset))
        good_data = request.getfixturevalue(passing_dataset)
        good_data.drop(passing_target, inplace=True, axis=1)
        assert validator.validate_inputs(good_data)

        # bad_data = pd.read_csv(os.path.join(self.tests_data_path, failing_dataset))
        bad_data = request.getfixturevalue(failing_dataset)
        bad_data.drop(failing_target, inplace=True, axis=1)
        with pytest.raises(DrumSchemaValidationException):
            validator.validate_inputs(bad_data)

    def test_data_types_raises_error_if_all_type_in_in_are_not_present(self, iris_binary):

        # TODO is this really correct behavior? if the dataset is missing any one
        #  of the data types then it's wrong?
        condition = Conditions.IN
        value = Values.data_values()

        yaml_str = input_requirements_yaml(Fields.DATA_TYPES, condition, value)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        with pytest.raises(DrumSchemaValidationException):
            validator.validate_inputs(iris_binary)

    @pytest.mark.parametrize(
        "single_value_condition",
        [
            Conditions.EQUALS,
            Conditions.NOT_EQUALS,
            Conditions.GREATER_THAN,
            Conditions.NOT_GREATER_THAN,
            Conditions.LESS_THAN,
            Conditions.NOT_LESS_THAN,
        ],
    )
    def test_instantiating_validator_raises_error_for_too_many_values(
        self, single_value_condition, iris_binary
    ):
        yaml_str = input_requirements_yaml(Fields.NUMBER_OF_COLUMNS, single_value_condition, [1, 2])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        with pytest.raises(DrumSchemaValidationException):
            SchemaValidator(schema_dict)

    @pytest.mark.parametrize(
        "condition, value, fail_expected",
        [
            (Conditions.EQUALS, [6], False),
            (Conditions.EQUALS, [3], True),
            (Conditions.IN, [2, 4, 6], False),
            (Conditions.IN, [1, 2, 3], True),
            (Conditions.LESS_THAN, [7], False),
            (Conditions.LESS_THAN, [3], True),
            (Conditions.GREATER_THAN, [4], False),
            (Conditions.GREATER_THAN, [10], True),
            (Conditions.NOT_EQUALS, [5], False),
            (Conditions.NOT_EQUALS, [6], True),
            (Conditions.NOT_IN, [1, 2, 3], False),
            (Conditions.NOT_IN, [2, 4, 6], True),
            (Conditions.NOT_GREATER_THAN, [6], False),
            (Conditions.NOT_GREATER_THAN, [2], True),
            (Conditions.NOT_LESS_THAN, [3], False),
            (Conditions.NOT_LESS_THAN, [100], True),
        ],
        ids=lambda x: str(x),
    )
    def test_num_columns(self, data, condition, value, fail_expected):
        yaml_str = input_requirements_yaml(Fields.NUMBER_OF_COLUMNS, condition, value)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)
        if fail_expected:
            with pytest.raises(DrumSchemaValidationException):
                validator.validate_inputs(data)
        else:
            assert validator.validate_inputs(data)

    @pytest.mark.parametrize(
        "value, missing_ok", [(Values.FORBIDDEN, False), (Values.SUPPORTED, True)]
    )
    def test_missing_input(self, data, missing_data, value, missing_ok):
        yaml_str = input_requirements_yaml(Fields.CONTAINS_MISSING, Conditions.EQUALS, [value])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        assert validator.validate_inputs(data)
        if missing_ok:
            assert validator.validate_inputs(missing_data)
        else:
            with pytest.raises(DrumSchemaValidationException):
                validator.validate_inputs(missing_data)

    @pytest.mark.parametrize("value, missing_ok", [(Values.NEVER, False), (Values.DYNAMIC, True)])
    def test_missing_output(self, data, missing_data, value, missing_ok):
        yaml_str = output_requirements_yaml(Fields.CONTAINS_MISSING, Conditions.EQUALS, [value])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        assert validator.validate_outputs(data)
        if missing_ok:
            assert validator.validate_outputs(missing_data)
        else:
            with pytest.raises(DrumSchemaValidationException):
                validator.validate_outputs(missing_data)

    @pytest.mark.parametrize(
        "value, sparse_ok, dense_ok",
        [
            (Values.FORBIDDEN, False, True),
            (Values.SUPPORTED, True, True),
            (Values.REQUIRED, True, False),
        ],
    )
    def test_sparse_input(self, sparse_df, dense_df, value, sparse_ok, dense_ok):
        yaml_str = input_requirements_yaml(Fields.SPARSE, Conditions.EQUALS, [value])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        self._assert_validation(validator.validate_inputs, sparse_df, should_pass=sparse_ok)
        self._assert_validation(validator.validate_inputs, dense_df, should_pass=dense_ok)

    @pytest.mark.parametrize(
        "value, sparse_ok, dense_ok",
        [
            (Values.NEVER, False, True),
            (Values.DYNAMIC, True, True),
            (Values.ALWAYS, True, False),
            (Values.IDENTITY, False, True),
        ],
    )
    def test_sparse_output(self, sparse_df, dense_df, value, sparse_ok, dense_ok):
        yaml_str = output_requirements_yaml(Fields.SPARSE, Conditions.EQUALS, [value])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        self._assert_validation(validator.validate_outputs, sparse_df, should_pass=sparse_ok)
        self._assert_validation(validator.validate_outputs, dense_df, should_pass=dense_ok)

    @pytest.mark.parametrize(
        "value, sparse_ok, dense_ok",
        [(Values.FORBIDDEN, False, True), (Values.REQUIRED, True, False),],
    )
    def test_multiple_input_requirements(self, sparse_df, dense_df, value, sparse_ok, dense_ok):
        yaml_str = input_requirements_yaml(Fields.SPARSE, Conditions.EQUALS, [value])
        num_input = input_requirements_yaml(
            Fields.DATA_TYPES, Conditions.EQUALS, [Values.NUM]
        ).replace("input_requirements:\n", "")
        random_output = output_requirements_yaml(
            Fields.NUMBER_OF_COLUMNS, Conditions.EQUALS, [10000]
        )
        yaml_str += num_input
        yaml_str += random_output
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        self._assert_validation(validator.validate_inputs, sparse_df, should_pass=sparse_ok)
        self._assert_validation(validator.validate_inputs, dense_df, should_pass=dense_ok)

    @pytest.mark.parametrize(
        "value, sparse_ok, dense_ok", [(Values.NEVER, False, True), (Values.ALWAYS, True, False),],
    )
    def test_multiple_output_requirements(self, sparse_df, dense_df, value, sparse_ok, dense_ok):
        yaml_str = output_requirements_yaml(Fields.SPARSE, Conditions.EQUALS, [value])
        num_output = output_requirements_yaml(
            Fields.DATA_TYPES, Conditions.EQUALS, [Values.NUM]
        ).replace("output_requirements:\n", "")
        random_input = input_requirements_yaml(Fields.NUMBER_OF_COLUMNS, Conditions.EQUALS, [10000])
        yaml_str += num_output
        yaml_str += random_input
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        self._assert_validation(validator.validate_outputs, sparse_df, should_pass=sparse_ok)
        self._assert_validation(validator.validate_outputs, dense_df, should_pass=dense_ok)

    @staticmethod
    def _assert_validation(validator_method, data_frame, should_pass):
        if should_pass:
            assert validator_method(data_frame)
        else:
            with pytest.raises(DrumSchemaValidationException):
                validator_method(data_frame)


class TestRevalidateTypeSchemaDataTypes:
    field = Fields.DATA_TYPES

    @pytest.mark.parametrize("condition", Conditions.non_numeric())
    def test_datatypes_allowed_conditions(self, condition):
        values = [Values.NUM, Values.TXT]
        input_data_type_str = input_requirements_yaml(self.field, condition, values)
        output_data_type_str = output_requirements_yaml(self.field, condition, values)

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("condition", list(set(Conditions) - set(Conditions.non_numeric())))
    def test_datatypes_unallowed_conditions(self, condition):
        values = [Values.NUM, Values.TXT]
        input_data_type_str = input_requirements_yaml(self.field, condition, values)
        output_data_type_str = output_requirements_yaml(self.field, condition, values)

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", Values.data_values())
    def test_datatyped_allowed_values(self, value):
        condition = Conditions.EQUALS
        input_data_type_str = input_requirements_yaml(self.field, condition, [value])
        output_data_type_str = output_requirements_yaml(self.field, condition, [value])

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", list(set(Values) - set(Values.data_values())))
    def test_datatypes_unallowed_values(self, value):
        condition = Conditions.EQUALS
        input_data_type_str = input_requirements_yaml(self.field, condition, [value])
        output_data_type_str = output_requirements_yaml(self.field, condition, [value])

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)

    def test_datatypes_multiple_values(self):
        condition = Conditions.IN
        values = Values.data_values()
        input_data_type_str = input_requirements_yaml(self.field, condition, values)
        output_data_type_str = output_requirements_yaml(self.field, condition, values)

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize(
        "permutation",
        [[Values.CAT, Values.NUM], [Values.NUM, Values.CAT]],
        ids=lambda x: str([str(el) for el in x]),
    )
    def test_regression_test_datatypes_multi_values(self, permutation):
        corner_case = input_requirements_yaml(Fields.DATA_TYPES, Conditions.IN, permutation)
        parsed_yaml = load(corner_case, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    def test_datatypes_mix_allowed_and_unallowed_values(self):
        values = [Values.NUM, Values.REQUIRED]
        condition = Conditions.EQUALS
        input_data_type_str = input_requirements_yaml(self.field, condition, values)
        output_data_type_str = output_requirements_yaml(self.field, condition, values)

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)


class TestRevalidateTypeSchemaSparse:
    field = Fields.SPARSE

    @pytest.mark.parametrize("value", Values.input_values())
    def test_sparsity_input_allowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", list(set(Values) - set(Values.input_values())))
    def test_sparsity_input_disallowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_sparsity_input_only_single_value(self):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, Values.input_values())

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", Values.output_values())
    def test_sparsity_output_allowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", list(set(Values) - set(Values.output_values())))
    def test_sparsity_output_disallowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_sparsity_output_only_single_value(self):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, Values.output_values())

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("condition", list(set(Conditions) - {Conditions.EQUALS}))
    def test_sparsity_input_output_disallows_conditions(self, condition):
        sparse_yaml_input_str = input_requirements_yaml(self.field, condition, [Values.REQUIRED])
        sparse_yaml_output_str = output_requirements_yaml(self.field, condition, [Values.ALWAYS])
        for yaml_str in (sparse_yaml_input_str, sparse_yaml_output_str):
            parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)


class TestRevalidateTypeSchemaContainsMissing:
    field = Fields.CONTAINS_MISSING

    @pytest.mark.parametrize("value", [Values.FORBIDDEN, Values.SUPPORTED])
    def test_contains_missing_input_allowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", list(set(Values) - {Values.FORBIDDEN, Values.SUPPORTED}))
    def test_contains_missing_input_disallowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_contains_missing_input_only_single_value(self):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(
            self.field, condition, [Values.FORBIDDEN, Values.SUPPORTED]
        )

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", [Values.NEVER, Values.DYNAMIC])
    def test_contains_missing_output_allowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", list(set(Values) - {Values.NEVER, Values.DYNAMIC}))
    def test_contains_missing_output_disallowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_contains_missing_output_only_single_value(self):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(
            self.field, condition, [Values.NEVER, Values.DYNAMIC]
        )

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("condition", list(set(Conditions) - {Conditions.EQUALS}))
    def test_contains_missing_input_output_disallows_conditions(self, condition):
        sparse_yaml_input_str = input_requirements_yaml(self.field, condition, [Values.REQUIRED])
        sparse_yaml_output_str = output_requirements_yaml(self.field, condition, [Values.ALWAYS])
        for yaml_str in (sparse_yaml_input_str, sparse_yaml_output_str):
            parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)


class TestRevalidateTypeSchemaNumberOfColumns:
    field = Fields.NUMBER_OF_COLUMNS

    @pytest.mark.parametrize("condition", list(Conditions))
    def test_number_of_columns_can_use_all_conditions(self, condition):
        sparse_yaml_input_str = input_requirements_yaml(self.field, condition, [1])
        sparse_yaml_output_str = output_requirements_yaml(self.field, condition, [1])
        for yaml_str in (sparse_yaml_input_str, sparse_yaml_output_str):
            parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
            revalidate_typeschema(parsed_yaml)

    def test_number_of_columns_can_have_multiple_ints(self):
        yaml_str = input_requirements_yaml(self.field, Conditions.EQUALS, [1, 0, -1])
        parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", list(Values))
    def test_number_of_columns_cannot_use_other_values(self, value):
        yaml_str = input_requirements_yaml(self.field, Conditions.EQUALS, [value])
        parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_revalidate_typescehma_mutates_yaml_num_columns_to_int(self):
        yaml_single_int = input_requirements_yaml(self.field, Conditions.EQUALS, [1])
        yaml_int_list = input_requirements_yaml(self.field, Conditions.EQUALS, [1, 2])
        parsed_single_int = load(yaml_single_int, get_type_schema_yaml_validator())
        parsed_int_list = load(yaml_int_list, get_type_schema_yaml_validator())

        def get_value(yaml):
            return yaml["input_requirements"][0]["value"].data

        assert isinstance(get_value(parsed_single_int), str)
        assert isinstance(get_value(parsed_int_list)[0], str)

        revalidate_typeschema(parsed_single_int)
        revalidate_typeschema(parsed_int_list)

        assert isinstance(get_value(parsed_single_int), int)
        assert isinstance(get_value(parsed_int_list)[0], int)


class TestRevalidateTypeSchemaMixedCases:
    @pytest.fixture
    def passing_yaml_string(self):
        yield dedent(
            """
            input_requirements:
            - field: data_types
              condition: IN
              value:
                - NUM
            - field: sparse
              condition: EQUALS
              value: FORBIDDEN
            output_requirements:
            - field: data_types
              condition: EQUALS
              value: NUM
            - field: sparse
              condition: EQUALS
              value: NEVER
            """
        )

    def test_happy_path(self, passing_yaml_string):
        parsed_yaml = load(passing_yaml_string, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("requirements_key", ["input_requirements", "output_requirements"])
    def test_failing_on_bad_requirements_key(self, requirements_key, passing_yaml_string):
        bad_yaml = passing_yaml_string.replace(requirements_key, "oooooops")
        with pytest.raises(YAMLValidationError):
            load(bad_yaml, get_type_schema_yaml_validator())

    def test_failing_on_bad_field(self, passing_yaml_string):
        bad_yaml = passing_yaml_string.replace("sparse", "oooooops")
        with pytest.raises(YAMLValidationError):
            load(bad_yaml, get_type_schema_yaml_validator())

    def test_failing_on_bad_condition(self, passing_yaml_string):
        bad_yaml = passing_yaml_string.replace("EQUALS", "oooooops")
        parsed_yaml = load(bad_yaml, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_failing_on_bad_value(self, passing_yaml_string):
        bad_yaml = passing_yaml_string.replace("NUM", "oooooops")
        parsed_yaml = load(bad_yaml, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)
