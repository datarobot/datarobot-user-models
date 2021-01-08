import os

import pytest
import yaml

from tests.drum.constants import (
    ANOMALY,
    BINARY,
    BINARY_BOOL,
    BINARY_TEXT,
    CODEGEN,
    CODEGEN_AND_SKLEARN,
    KERAS,
    MOJO,
    MULTI_ARTIFACT,
    MULTICLASS,
    MULTICLASS_NUM_LABELS,
    MULTICLASS_BINARY,
    NO_CUSTOM,
    POJO,
    PYPMML,
    PYTHON,
    PYTHON_ALL_PREDICT_STRUCTURED_HOOKS,
    PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS,
    PYTHON_LOAD_MODEL,
    PYTHON_TRANSFORM,
    PYTHON_TRANSFORM_DENSE,
    PYTHON_UNSTRUCTURED,
    PYTHON_UNSTRUCTURED_PARAMS,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    PYTORCH,
    PYTORCH_MULTICLASS,
    R,
    R_ALL_PREDICT_STRUCTURED_HOOKS,
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS,
    R_FIT,
    R_UNSTRUCTURED,
    R_UNSTRUCTURED_PARAMS,
    RDS,
    RDS_SPARSE,
    REGRESSION,
    REGRESSION_INFERENCE,
    SIMPLE,
    SKLEARN,
    SKLEARN_ANOMALY,
    SKLEARN_BINARY,
    SKLEARN_MULTICLASS,
    SKLEARN_REGRESSION,
    SKLEARN_SPARSE,
    SKLEARN_TRANSFORM,
    SKLEARN_TRANSFORM_DENSE,
    SKLEARN_TRANSFORM_WITH_Y,
    PYTHON_TRANSFORM_NO_Y,
    PYTHON_TRANSFORM_NO_Y_DENSE,
    SKLEARN_TRANSFORM_NO_HOOK,
    SKLEARN_TRANSFORM_SPARSE_INPUT,
    SKLEARN_TRANSFORM_NON_NUMERIC,
    SKLEARN_PRED_CONSISTENCY,
    SPARSE,
    SPARSE_TARGET,
    TESTS_ARTIFACTS_PATH,
    TESTS_DATA_PATH,
    TESTS_FIXTURES_PATH,
    TRAINING_TEMPLATES_PATH,
    TRANSFORM,
    UNSTRUCTURED,
    XGB,
)

_datasets = {
    # If specific dataset should be defined for a framework, use (framework, problem) key.
    # Otherwise default dataset is used (None, problem)
    (None, REGRESSION): os.path.join(TESTS_DATA_PATH, "boston_housing.csv"),
    (None, BINARY_TEXT): os.path.join(TESTS_DATA_PATH, "telecomms_churn.csv"),
    (PYPMML, REGRESSION): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, REGRESSION_INFERENCE): os.path.join(TESTS_DATA_PATH, "boston_housing_inference.csv"),
    (None, BINARY): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, ANOMALY): os.path.join(TESTS_DATA_PATH, "boston_housing.csv"),
    (None, UNSTRUCTURED): os.path.join(TESTS_DATA_PATH, "unstructured_data.txt"),
    (None, MULTICLASS): os.path.join(TESTS_DATA_PATH, "skyserver_sql2_27_2018_6_51_39_pm.csv"),
    (None, MULTICLASS_NUM_LABELS): os.path.join(
        TESTS_DATA_PATH, "skyserver_sql2_27_2018_6_51_39_pm_num_class.csv"
    ),
    (None, MULTICLASS_BINARY): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, SPARSE): os.path.join(TESTS_DATA_PATH, "sparse.mtx"),
    (None, SPARSE_TARGET): os.path.join(TESTS_DATA_PATH, "sparse_target.csv"),
    (None, BINARY_BOOL): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (SKLEARN, TRANSFORM): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (SKLEARN_TRANSFORM, TRANSFORM): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (SKLEARN_TRANSFORM_DENSE, TRANSFORM): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (SKLEARN_PRED_CONSISTENCY, BINARY_BOOL): os.path.join(
        TESTS_DATA_PATH, "10k_diabetes_sample.csv"
    ),
}

_training_models_paths = {
    (PYTHON, SKLEARN_BINARY): os.path.join(TRAINING_TEMPLATES_PATH, "python3_sklearn_binary"),
    (PYTHON, SKLEARN_REGRESSION): os.path.join(
        TRAINING_TEMPLATES_PATH, "python3_sklearn_regression"
    ),
    (PYTHON, SKLEARN_MULTICLASS): os.path.join(
        TRAINING_TEMPLATES_PATH, "python3_sklearn_multiclass"
    ),
    (PYTHON, SIMPLE): os.path.join(TRAINING_TEMPLATES_PATH, "simple"),
    (PYTHON, SKLEARN_SPARSE): os.path.join(TRAINING_TEMPLATES_PATH, "python3_sparse"),
    (PYTHON, KERAS): os.path.join(TRAINING_TEMPLATES_PATH, "python3_keras_joblib"),
    (PYTHON, XGB): os.path.join(TRAINING_TEMPLATES_PATH, "python3_xgboost"),
    (R_FIT, RDS): os.path.join(TRAINING_TEMPLATES_PATH, "r_lang"),
    (PYTHON, PYTORCH): os.path.join(TRAINING_TEMPLATES_PATH, "python3_pytorch"),
    (PYTHON, SKLEARN_ANOMALY): os.path.join(TRAINING_TEMPLATES_PATH, "python3_anomaly_detection"),
    (PYTHON, PYTORCH_MULTICLASS): os.path.join(
        TRAINING_TEMPLATES_PATH, "python3_pytorch_multiclass"
    ),
    (PYTHON, SKLEARN_PRED_CONSISTENCY): os.path.join(
        TESTS_FIXTURES_PATH, "custom_pred_consistency"
    ),
}

_targets = {
    BINARY: "Species",
    REGRESSION: "MEDV",
    BINARY_TEXT: "Churn",
    MULTICLASS: "class",
    MULTICLASS_BINARY: "Species",
    MULTICLASS_NUM_LABELS: "class",
    SPARSE: "my_target",
    BINARY_BOOL: "readmitted",
    ANOMALY: None,
    TRANSFORM: os.path.join(TESTS_DATA_PATH, "transform_target.csv"),
}

_target_types = {
    BINARY: "binary",
    BINARY_TEXT: "binary",
    REGRESSION: "regression",
    REGRESSION_INFERENCE: "regression",
    ANOMALY: "anomaly",
    UNSTRUCTURED: "unstructured",
    MULTICLASS: "multiclass",
    MULTICLASS_BINARY: "multiclass",
    MULTICLASS_NUM_LABELS: "multiclass",
    BINARY_BOOL: "binary",
    TRANSFORM: "transform",
}

_class_labels = {
    (SKLEARN_BINARY, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (SKLEARN, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (XGB, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (KERAS, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (RDS, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (PYPMML, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (PYTORCH, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (CODEGEN, BINARY): ["yes", "no"],
    (MOJO, BINARY): ["yes", "no"],
    (POJO, BINARY): ["yes", "no"],
    (SKLEARN, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (SKLEARN_MULTICLASS, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (SKLEARN_MULTICLASS, MULTICLASS_NUM_LABELS): ["0", "1", "2"],
    (SKLEARN_MULTICLASS, MULTICLASS_NUM_LABELS): [0, 1, 2],
    (XGB, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (KERAS, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (RDS, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (PYPMML, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (PYTORCH_MULTICLASS, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (PYTORCH, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (CODEGEN, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (SKLEARN_BINARY, BINARY_TEXT): ["False", "True"],
    (XGB, BINARY_TEXT): ["False", "True"],
    (KERAS, BINARY_TEXT): ["False", "True"],
    (POJO, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (MOJO, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (SKLEARN_BINARY, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (SKLEARN, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (XGB, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (KERAS, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (RDS, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (PYPMML, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (PYTORCH, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (CODEGEN, MULTICLASS_BINARY): ["yes", "no"],
    (MOJO, MULTICLASS_BINARY): ["yes", "no"],
    (POJO, MULTICLASS_BINARY): ["yes", "no"],
    (SKLEARN_PRED_CONSISTENCY, BINARY_BOOL): ["False", "True"],
}

_artifacts = {
    (None, None): None,
    (None, REGRESSION): None,
    (None, BINARY): None,
    (None, UNSTRUCTURED): None,
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
    (RDS_SPARSE, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "r_reg_sparse.rds"),
    (RDS, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "r_bin.rds"),
    (CODEGEN, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "java_reg.jar"),
    (CODEGEN, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "java_bin.jar"),
    (POJO, REGRESSION): os.path.join(
        TESTS_ARTIFACTS_PATH,
        "drf_887c2e5b_0941_40b7_ae26_cae274c4b424.java",
    ),
    (POJO, BINARY): os.path.join(
        TESTS_ARTIFACTS_PATH,
        "XGBoost_grid__1_AutoML_20200717_163214_model_159.java",
    ),
    (POJO, MULTICLASS): os.path.join(
        TESTS_ARTIFACTS_PATH,
        "XGBoost_3_AutoML_20201016_143029.java",
    ),
    (MOJO, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_reg.zip"),
    (MOJO, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_bin.zip"),
    (MOJO, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_multi.zip"),
    (PYPMML, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "iris_reg.pmml"),
    (PYPMML, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "iris_bin.pmml"),
    (PYPMML, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "iris_multi.pmml"),
    (SKLEARN, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_multi.pkl"),
    (XGB, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "xgb_multi.pkl"),
    (KERAS, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "keras_multi.h5"),
    (RDS, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "r_multi.rds"),
    (PYTORCH, MULTICLASS): [
        os.path.join(TESTS_ARTIFACTS_PATH, "torch_multi.pth"),
        os.path.join(TESTS_ARTIFACTS_PATH, "PyTorch.py"),
    ],
    (CODEGEN, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "java_multi.jar"),
    (SKLEARN_TRANSFORM, TRANSFORM): os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_transform.pkl"),
    (SKLEARN_TRANSFORM_DENSE, TRANSFORM): os.path.join(
        TESTS_ARTIFACTS_PATH, "sklearn_transform_dense.pkl"
    ),
    (SKLEARN, MULTICLASS_BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_bin.pkl"),
    (KERAS, MULTICLASS_BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "keras_bin.h5"),
    (XGB, MULTICLASS_BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "xgb_bin.pkl"),
    (PYTORCH, MULTICLASS_BINARY): [
        os.path.join(TESTS_ARTIFACTS_PATH, "torch_bin.pth"),
        os.path.join(TESTS_ARTIFACTS_PATH, "PyTorch.py"),
    ],
    (RDS, MULTICLASS_BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "r_bin.rds"),
    (CODEGEN, MULTICLASS_BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "java_bin.jar"),
    (POJO, MULTICLASS_BINARY): os.path.join(
        TESTS_ARTIFACTS_PATH,
        "XGBoost_grid__1_AutoML_20200717_163214_model_159.java",
    ),
    (MOJO, MULTICLASS_BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_bin.zip"),
    (PYPMML, MULTICLASS_BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "iris_bin.pmml"),
    (SKLEARN_TRANSFORM_WITH_Y, REGRESSION): None,
    (SKLEARN_TRANSFORM_WITH_Y, BINARY): None,
    (SKLEARN_TRANSFORM_WITH_Y, ANOMALY): None,
    (SKLEARN_TRANSFORM_NO_HOOK, REGRESSION): None,
    (SKLEARN_TRANSFORM_NO_HOOK, BINARY): None,
    (SKLEARN_TRANSFORM_NO_HOOK, ANOMALY): None,
    (SKLEARN_TRANSFORM, REGRESSION): None,
    (SKLEARN_TRANSFORM, BINARY): None,
    (SKLEARN_TRANSFORM, ANOMALY): None,
    (SKLEARN_TRANSFORM_SPARSE_INPUT, REGRESSION): None,
    (SKLEARN_TRANSFORM_SPARSE_INPUT, BINARY): None,
    (SKLEARN_TRANSFORM_SPARSE_INPUT, ANOMALY): None,
    (SKLEARN_TRANSFORM_NON_NUMERIC, REGRESSION): None,
    (SKLEARN_TRANSFORM_NON_NUMERIC, BINARY): None,
    (SKLEARN_TRANSFORM_NON_NUMERIC, ANOMALY): None,
}

_custom_filepaths = {
    PYTHON: (os.path.join(TESTS_FIXTURES_PATH, "custom.py"), "custom.py"),
    NO_CUSTOM: (None, None),
    PYTHON_ALL_PREDICT_STRUCTURED_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "all_predict_structured_hooks_custom.py"),
        "custom.py",
    ),
    PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "all_predict_unstructured_hooks_custom.py"),
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
    R_ALL_PREDICT_STRUCTURED_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "all_predict_structured_hooks_custom.R"),
        "custom.R",
    ),
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "all_predict_unstructured_hooks_custom.R"),
        "custom.R",
    ),
    R_FIT: (os.path.join(TESTS_FIXTURES_PATH, "fit_custom.R"), "custom.R"),
    PYTHON_UNSTRUCTURED: (os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom.py"), "custom.py"),
    R_UNSTRUCTURED: (os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom.R"), "custom.R"),
    PYTHON_UNSTRUCTURED_PARAMS: (
        os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom_params.py"),
        "custom.py",
    ),
    R_UNSTRUCTURED_PARAMS: (
        os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom_params.R"),
        "custom.R",
    ),
    PYTHON_TRANSFORM: (os.path.join(TESTS_FIXTURES_PATH, "transform_custom.py"), "custom.py"),
    PYTHON_TRANSFORM_DENSE: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_custom.py"),
        "custom.py",
    ),
    PYTHON_TRANSFORM_NO_Y: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_custom_no_y.py"),
        "custom.py",
    ),
    PYTHON_TRANSFORM_NO_Y_DENSE: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_custom_no_y.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_WITH_Y: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_NO_HOOK: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_no_hook.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_no_y.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_SPARSE_INPUT: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_sparse_input.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_NON_NUMERIC: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_non_numeric.py"),
        "custom.py",
    ),
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
def get_target_type():
    def _foo(problem):
        return _target_types.get(problem, problem)

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
    get_target_type,
    get_class_labels,
    get_artifacts,
    get_custom,
):
    resource = Resource()
    resource.datasets = get_dataset_filename
    resource.training_models = get_paths_to_training_models
    resource.targets = get_target
    resource.target_types = get_target_type
    resource.class_labels = get_class_labels
    resource.artifacts = get_artifacts
    resource.custom = get_custom
    return resource


# fixtures for variety data tests
with open(os.path.join(TESTS_DATA_PATH, "variety_samples/variety_data_key.yaml")) as yamlfile:
    variety_data_dict = yaml.load(yamlfile)

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
