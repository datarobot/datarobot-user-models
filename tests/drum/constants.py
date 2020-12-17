import os

TESTS_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_FIXTURES_PATH = os.path.join(TESTS_ROOT_PATH, "fixtures")
TESTS_ARTIFACTS_PATH = os.path.join(TESTS_FIXTURES_PATH, "drop_in_model_artifacts")
TESTS_DATA_PATH = os.path.join(TESTS_ROOT_PATH, "testdata")
TRAINING_TEMPLATES_PATH = os.path.join(TESTS_ROOT_PATH, "..", "model_templates", "training")


TRAINING = "training"
INFERENCE = "inference"

# Framework keywords
XGB = "xgboost"
KERAS = "keras"
KERAS_JOBLIB = "keras_joblib"
SKLEARN = "sklearn"
SKLEARN_BINARY = "sklearn_binary"
SKLEARN_REGRESSION = "sklearn_regression"
SKLEARN_MULTICLASS = "sklearn_multiclass"
SKLEARN_SPARSE = "sparse"
SKLEARN_TRANSFORM = "sklearn_transform"
SKLEARN_TRANSFORM_DENSE = "sklearn_transform_dense"
SKLEARN_PRED_CONSISTENCY = "sklearn_pred_consistency"
SKLEARN_TRANSFORM_WITH_Y = "sklearn_transform_with_y"
SKLEARN_TRANSFORM_NO_HOOK = "sklearn_transform_no_hook"
SKLEARN_TRANSFORM_SPARSE = "sklearn_transform_sparse"
SIMPLE = "simple"
PYTORCH = "pytorch"
PYTORCH_MULTICLASS = "pytorch_multiclass"
PYPMML = "pypmml"
SKLEARN_ANOMALY = "sklearn_anomaly_detection"

RDS = "rds"
CODEGEN = "jar"
## adding h2o pojo and mojo
MOJO = "zip"
POJO = "java"
##
MULTI_ARTIFACT = "multiartifact"
CODEGEN_AND_SKLEARN = "codegen_and_sklearn"
# Problem keywords, used to mark datasets
REGRESSION = "regression"
BINARY_TEXT = "bintxt"
BINARY_BOOL = "binary_bool"
REGRESSION_INFERENCE = "regression_inference"
BINARY = "binary"
ANOMALY = "anomaly"
UNSTRUCTURED = "unstructured"
MULTICLASS = "multiclass"
TRANSFORM = "transform"
MULTICLASS_NUM_LABELS = "multiclass_num_labels"
MULTICLASS_BINARY = "multiclass_binary"  # special case for testing multiclass with only 2 classes
SPARSE = "sparse"
SPARSE_TARGET = "sparse_target"

# Language keywords
PYTHON = "python3"
NO_CUSTOM = "no_custom"
PYTHON_ALL_PREDICT_STRUCTURED_HOOKS = "python_all_predict_structured_hooks"
PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS = "python_all_predict_unstructured_hooks"
PYTHON_LOAD_MODEL = "python_load_model"
PYTHON_TRANSFORM = "python_transform"
PYTHON_TRANSFORM_DENSE = "python_transform_dense"
PYTHON_TRANSFORM_NO_Y = "python_transform_no_y"
PYTHON_TRANSFORM_NO_Y_DENSE = "python_transform_no_y_dense"
R = "R"
R_ALL_PREDICT_STRUCTURED_HOOKS = "R_all_predict_structured_hooks"
R_ALL_PREDICT_UNSTRUCTURED_HOOKS = "R_all_predict_unstructured_hooks"
R_FIT = "R_fit"
JAVA = "java"
PYTHON_UNSTRUCTURED = "python_unstructured"
PYTHON_UNSTRUCTURED_PARAMS = "python_unstructured_params"
R_UNSTRUCTURED = "r_unstructured"
R_UNSTRUCTURED_PARAMS = "r_unstructured_params"
PYTHON_XGBOOST_CLASS_LABELS_VALIDATION = "predictions_and_class_labels_validation"

DOCKER_PYTHON_SKLEARN = "cmrunner_test_env_python_sklearn"

RESPONSE_PREDICTIONS_KEY = "predictions"
RESPONSE_TRANSFORM_KEY = "transformations"

WEIGHTS_ARGS = "weights-args"
WEIGHTS_CSV = "weights-csv"
