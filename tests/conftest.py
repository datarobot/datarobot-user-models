import os
import pytest
import json

from tests.drum.constants import (
    TESTS_ARTIFACTS_PATH,
    TESTS_DATA_PATH,
    TESTS_FIXTURES_PATH,
    TRAINING_TEMPLATES_PATH,
    TRAINING,
    INFERENCE,
    XGB,
    KERAS,
    KERAS_JOBLIB,
    SKLEARN,
    SIMPLE,
    PYTORCH,
    PYPMML,
    SKLEARN_ANOMALY,
    RDS,
    CODEGEN,
    MOJO,
    POJO,
    MULTI_ARTIFACT,
    CODEGEN_AND_SKLEARN,
    REGRESSION,
    REGRESSION_INFERENCE,
    BINARY,
    ANOMALY,
    PYTHON,
    NO_CUSTOM,
    PYTHON_ALL_HOOKS,
    PYTHON_LOAD_MODEL,
    R,
    R_ALL_HOOKS,
    R_FIT,
    JAVA,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
)

_datasets = {
    # If specific dataset should be defined for a framework, use (framework, problem) key.
    # Otherwise default dataset is used (None, problem)
    (None, REGRESSION): os.path.join(TESTS_DATA_PATH, "boston_housing.csv"),
    (PYPMML, REGRESSION): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, REGRESSION_INFERENCE): os.path.join(TESTS_DATA_PATH, "boston_housing_inference.csv"),
    (None, BINARY): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, ANOMALY): os.path.join(TESTS_DATA_PATH, "boston_housing.csv"),
}

_training_models_paths = {
    (PYTHON, SKLEARN): os.path.join(TRAINING_TEMPLATES_PATH, "python3_sklearn"),
    (PYTHON, SIMPLE): os.path.join(TRAINING_TEMPLATES_PATH, "simple"),
    (PYTHON, KERAS): os.path.join(TRAINING_TEMPLATES_PATH, "python3_keras_joblib"),
    (PYTHON, XGB): os.path.join(TRAINING_TEMPLATES_PATH, "python3_xgboost"),
    (R_FIT, RDS): os.path.join(TRAINING_TEMPLATES_PATH, "r_lang"),
    (PYTHON, PYTORCH): os.path.join(TRAINING_TEMPLATES_PATH, "python3_pytorch"),
    (PYTHON, SKLEARN_ANOMALY): os.path.join(TRAINING_TEMPLATES_PATH, "python3_anomaly_detection"),
}

_targets = {BINARY: "Species", REGRESSION: "MEDV"}

_class_labels = {
    (SKLEARN, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (XGB, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (KERAS, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (RDS, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (PYPMML, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (CODEGEN, BINARY): ["yes", "no"],
    (MOJO, BINARY): ["yes", "no"],
    (POJO, BINARY): ["yes", "no"],
}

_artifacts = {
    (None, REGRESSION): None,
    (None, BINARY): None,
    (SKLEARN, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_reg.pkl"),
    (SKLEARN, REGRESSION_INFERENCE): os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_reg.pkl"),
    (MULTI_ARTIFACT, REGRESSION): [
        os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_reg.pkl"),
        os.path.join(TESTS_ARTIFACTS_PATH, "keras_reg.h5"),
    ],
    (CODEGEN_AND_SKLEARN, REGRESSION): [
        os.path.join(TESTS_ARTIFACTS_PATH, "java_reg.jar"),
        os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_reg.pkl"),
    ],
    (SKLEARN, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_bin.pkl"),
    (KERAS, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "keras_reg.h5"),
    (KERAS, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "keras_bin.h5"),
    (XGB, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "xgb_reg.pkl"),
    (XGB, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "xgb_bin.pkl"),
    (PYTORCH, REGRESSION): [
        os.path.join(TESTS_ARTIFACTS_PATH, "torch_reg.pth"),
        os.path.join(TESTS_ARTIFACTS_PATH, "PyTorch.py"),
    ],
    (PYTORCH, BINARY): [
        os.path.join(TESTS_ARTIFACTS_PATH, "torch_bin.pth"),
        os.path.join(TESTS_ARTIFACTS_PATH, "PyTorch.py"),
    ],
    (RDS, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "r_reg.rds"),
    (RDS, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "r_bin.rds"),
    (CODEGEN, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "java_reg.jar"),
    (CODEGEN, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "java_bin.jar"),
    (POJO, REGRESSION): os.path.join(
        TESTS_ARTIFACTS_PATH,
        "pojo_reg",
        "drf_887c2e5b_0941_40b7_ae26_cae274c4b424.java",
    ),
    (POJO, BINARY): os.path.join(
        TESTS_ARTIFACTS_PATH,
        "pojo_bin",
        "XGBoost_grid__1_AutoML_20200717_163214_model_159.java",
    ),
    (MOJO, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_reg.zip"),
    (MOJO, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_bin.zip"),
    (PYPMML, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "iris_reg.pmml"),
    (PYPMML, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "iris_bin.pmml"),
}

_custom_filepaths = {
    PYTHON: (os.path.join(TESTS_FIXTURES_PATH, "custom.py"), "custom.py"),
    NO_CUSTOM: (None, None),
    PYTHON_ALL_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "all_hooks_custom.py"),
        "custom.py",
    ),
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION: (
        os.path.join(TESTS_FIXTURES_PATH, "pred_validation_custom.py"),
        "custom.py",
    ),
    PYTHON_LOAD_MODEL: (
        os.path.join(TESTS_FIXTURES_PATH, "load_model_custom.py"),
        "custom.py",
    ),
    R: (os.path.join(TESTS_FIXTURES_PATH, "custom.R"), "custom.R"),
    R_ALL_HOOKS: (os.path.join(TESTS_FIXTURES_PATH, "all_hooks_custom.R"), "custom.R"),
    R_FIT: (os.path.join(TESTS_FIXTURES_PATH, "fit_custom.R"), "custom.R"),
}


class Resource:
    def __init__(self):
        self.datasets = None
        self.training_models = None
        self.targets = None
        self.class_labels = None
        self.artifacts = None
        self.custom = None


@pytest.fixture(scope="session")
def get_dataset_filename():
    def _foo(framework, problem):
        framework_key = framework
        problem_key = problem
        # if specific dataset for framework was not defined,
        # use default dataset for this problem, e.g. (None, problem)
        framework_key = None if (framework_key, problem_key) not in _datasets else framework_key
        return _datasets[(framework_key, problem_key)]

    return _foo


@pytest.fixture(scope="session")
def get_paths_to_training_models():
    def _foo(language, framework):
        return _training_models_paths[(language, framework)]

    return _foo


@pytest.fixture(scope="session")
def get_target():
    def _foo(problem):
        return _targets[problem]

    return _foo


@pytest.fixture(scope="session")
def get_class_labels():
    def _foo(framework, problem):
        return _class_labels.get((framework, problem), None)

    return _foo


@pytest.fixture(scope="session")
def get_artifacts():
    def _foo(framework, problem):
        return _artifacts[(framework, problem)]

    return _foo


@pytest.fixture(scope="session")
def get_custom():
    def _foo(language):
        return _custom_filepaths[language]

    return _foo


@pytest.fixture(scope="session")
def resources(
    get_dataset_filename,
    get_paths_to_training_models,
    get_target,
    get_class_labels,
    get_artifacts,
    get_custom,
):
    resource = Resource()
    resource.datasets = get_dataset_filename
    resource.training_models = get_paths_to_training_models
    resource.targets = get_target
    resource.class_labels = get_class_labels
    resource.artifacts = get_artifacts
    resource.custom = get_custom
    return resource


# fixtures for variety data tests
with open(os.path.join(TESTS_DATA_PATH, "variety_samples/variety_data_key.json")) as jsonfile:
    variety_data_dict = json.load(jsonfile)

variety_data_names = [*variety_data_dict]


@pytest.fixture(scope="session", params=variety_data_names)
def variety_data_names(request):
    return request.param


class VarietyDataResource:
    def __init__(self):
        self.dataset = None
        self.target = None
        self.class_labels = None
        self.problem = None


@pytest.fixture(scope="session")
def get_variety_dataset():
    def _foo(data_name):
        return TESTS_DATA_PATH + "/variety_samples/" + data_name

    return _foo


@pytest.fixture(scope="session")
def get_variety_target():
    def _foo(data_name):
        return variety_data_dict[data_name]["target"]

    return _foo


@pytest.fixture(scope="session")
def get_variety_problem():
    def _foo(data_name):
        return variety_data_dict[data_name]["problem"]

    return _foo


@pytest.fixture(scope="session")
def get_variety_classes_labels():
    def _foo(data_name):
        return variety_data_dict[data_name].get("classes")

    return _foo


@pytest.fixture(scope="session")
def variety_resources(
    get_variety_dataset, get_variety_target, get_variety_problem, get_variety_classes_labels
):
    resource = VarietyDataResource()
    resource.dataset = get_variety_dataset
    resource.problem = get_variety_problem
    resource.target = get_variety_target
    resource.class_labels = get_variety_classes_labels
    return resource
