"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os

REPO_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TESTS_ROOT_PATH = os.path.join(REPO_ROOT_PATH, "tests")
TESTS_FIXTURES_PATH = os.path.join(TESTS_ROOT_PATH, "fixtures")
TESTS_ARTIFACTS_PATH = os.path.join(TESTS_FIXTURES_PATH, "drop_in_model_artifacts")
TESTS_DATA_PATH = os.path.join(TESTS_ROOT_PATH, "testdata")
TESTS_DEPLOYMENT_CONFIG_PATH = os.path.join(TESTS_DATA_PATH, "deployment_config")
TRANSFORM_TEMPLATES_PATH = os.path.join(TESTS_ROOT_PATH, "..", "task_templates", "1_transforms")
ESTIMATORS_TEMPLATES_PATH = os.path.join(TESTS_ROOT_PATH, "..", "task_templates", "2_estimators")
TRAINING_TEMPLATES_PATH = os.path.join(TESTS_ROOT_PATH, "..", "task_templates", "3_pipelines")

MODEL_TEMPLATES_PATH = os.path.join(REPO_ROOT_PATH, "model_templates")
PUBLIC_DROPIN_ENVS_PATH = os.path.join(REPO_ROOT_PATH, "public_dropin_environments")


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
SKLEARN_TRANSFORM_SPARSE_INPUT = "sklearn_transform_sparse_input"
SKLEARN_TRANSFORM_SPARSE_INPUT_Y_OUTPUT = "sklearn_transform_sparse_input_y_output"
SKLEARN_TRANSFORM_SPARSE_IN_OUT = "sklearn_transform_sparse_in_out"
SKLEARN_TRANSFORM_NON_NUMERIC = "sklearn_transform_non_numeric"
PYTORCH = "pytorch"
PYTORCH_REGRESSION = "pytorch_regression"
PYTORCH_MULTICLASS = "pytorch_multiclass"
ONNX = "onnx"
PYPMML = "pypmml"
SKLEARN_ANOMALY = "sklearn_anomaly_detection"
SKLEARN_BINARY_HYPERPARAMETERS = "sklearn_binary_hyperparameters"
SKLEARN_TRANSFORM_HYPERPARAMETERS = "sklearn_transform_hyperparameters"
R_TRANSFORM_WITH_Y = "R_transform"
R_TRANSFORM = "R_transform_no_y"
R_TRANSFORM_NO_HOOK = "R_transform_no_hook"
R_TRANSFORM_SPARSE_INPUT = "R_transform_sparse_input"
R_TRANSFORM_SPARSE_OUTPUT = "R_transform_sparse_output"
R_TRANSFORM_SPARSE_IN_OUT = "R_transform_sparse_in_out"
R_TRANSFORM_SPARSE_INPUT_Y_OUTPUT = "R_transform_sparse_input_y_output"
R_TRANSFORM_NON_NUMERIC = "R_transform_non_numeric"
R_ESTIMATOR_SPARSE = "R_estimator_sparse"
R_VALIDATE_SPARSE_ESTIMATOR = "R_validate_sparse_estimator"
PYTHON_XFORM_ESTIMATOR = "python_xform_estimator"
R_XFORM_ESTIMATOR = "r_xform_estimator"
CUSTOM_TASK_INTERFACE_BINARY = "custom_task_interface_binary"
CUSTOM_TASK_INTERFACE_MULTICLASS = "custom_task_interface_multiclass"
CUSTOM_TASK_INTERFACE_MULTICLASS = "custom_task_interface_multiclass"
CUSTOM_TASK_INTERFACE_REGRESSION = "custom_task_interface_regression"
CUSTOM_TASK_INTERFACE_ANOMALY = "custom_task_interface_anomaly"
CUSTOM_TASK_INTERFACE_TRANSFORM = "custom_task_interface_transform_missing_values"
CUSTOM_TASK_INTERFACE_PYTORCH_BINARY = "custom_task_interface_pytorch_binary"
CUSTOM_TASK_INTERFACE_PYTORCH_MULTICLASS = "custom_task_interface_pytorch_multiclass"
CUSTOM_TASK_INTERFACE_KERAS_REGRESSION = "custom_task_interface_keras_regression"
CUSTOM_TASK_INTERFACE_XGB_REGRESSION = "custom_task_interface_xgboost_regression"

MLJ = "mlj"
MLJ_BINARY = "mlj_binary"
MLJ_REGRESSION = "mlj_regression"
MLJ_MULTICLASS = "mlj_multiclass"

RDS = "rds"
RDS_BINARY = "rds_binary"
RDS_SPARSE = "rds_sparse"
RDS_HYPERPARAMETERS = "rds_hyperparameters"
CODEGEN = "jar"
## adding h2o pojo and mojo
MOJO = "zip"
POJO = "java"
##
## adding julia
JLSO = "jlso"
##
MULTI_ARTIFACT = "multiartifact"
CODEGEN_AND_SKLEARN = "codegen_and_sklearn"
# Problem keywords, used to mark datasets
REGRESSION = "regression"
REGRESSION_SINGLE_COL = "regression_single_col"
REGRESSION_MULTLILINE_TEXT = "regression_multiline_text"
BINARY_TEXT = "bintxt"
BINARY_NUM_ONLY = "bin_num_only"
BINARY_NUM_TARGET = "binary_num_target"
BINARY_BOOL = "binary_bool"
BINARY_INT = "binary_int"
BINARY_INT_TARGET = "binary_int_target"
BINARY_SPACES = "binary_spaces"
REGRESSION_INFERENCE = "regression_inference"
BINARY = "binary"
ANOMALY = "anomaly"
UNSTRUCTURED = "unstructured"
MULTICLASS = "multiclass"
TRANSFORM = "transform"
MULTICLASS_NUM_LABELS = "multiclass_num_labels"
MULTICLASS_FLOAT_LABELS = "multiclass_float_labels"
MULTICLASS_HIGH_CARD = "multiclass_high_card"
MULTICLASS_BINARY = "multiclass_binary"  # special case for testing multiclass with only 2 classes
SPARSE = "sparse"
SPARSE_TRANSFORM = "sparse_transform"
SPARSE_COLUMNS = "sparse_columns"
SPARSE_TARGET = "sparse_target"
TARGET_NAME_DUPLICATED_X = "target_name_duplicated_x"
TARGET_NAME_DUPLICATED_Y = "target_name_duplicated_y"
SKLEARN_BINARY_PARAMETERS = "sklearn_binary_parameters"
SKLEARN_TRANSFORM_PARAMETERS = "sklearn_transform_parameters"
RDS_PARAMETERS = "r_parameters"
MULTICLASS_LABEL_SPACES = "iris_with_spaces_full"

# Language keywords
PYTHON = "python3"
NO_CUSTOM = "no_custom"
PYTHON_ALL_PREDICT_STRUCTURED_HOOKS = "python_all_predict_structured_hooks"
PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS = "python_all_predict_unstructured_hooks"
PYTHON_NO_ARTIFACT_REGRESSION_HOOKS = "python_no_artifact_regression_hooks"
PYTHON_LOAD_MODEL = "python_load_model"
PYTHON_PREDICT_SPARSE = "python_predict_sparse"
PYTHON_TRANSFORM_WITH_Y = "python_transform"
PYTHON_TRANSFORM_DENSE_WITH_Y = "python_transform_dense"
PYTHON_TRANSFORM_SPARSE = "python_transform_sparse"
PYTHON_TRANSFORM = "python_transform_no_y"
PYTHON_TRANSFORM_DENSE = "python_transform_no_y_dense"
PYTHON_TRANSFORM_FAIL_OUTPUT_SCHEMA_VALIDATION = "python_transform_fail_output_schema_validation"
R = "R"
R_ALL_PREDICT_STRUCTURED_HOOKS = "R_all_predict_structured_hooks"
R_ALL_PREDICT_STRUCTURED_HOOKS_LOWERCASE_R = "R_all_predict_structured_hooks_lowercase_r"
R_FAIL_CLASSIFICATION_VALIDATION_HOOKS = "R_fail_classification_validation_hooks"
R_ALL_PREDICT_UNSTRUCTURED_HOOKS = "R_all_predict_unstructured_hooks"
R_ALL_PREDICT_UNSTRUCTURED_HOOKS_LOWERCASE_R = "R_all_predict_unstructured_hooks_lowercase_r"
R_INT_COLNAMES_BINARY = "R_int_colnames_binary"
R_INT_COLNAMES_MULTICLASS = "R_int_colnames_multiclass"
R_FIT = "R_fit"
R_PREDICT_SPARSE = "R_predict_sparse"
JAVA = "java"
JULIA = "julia"
PYTHON_UNSTRUCTURED = "python_unstructured"
PYTHON_UNSTRUCTURED_MLOPS = "python_unstructured_mlops"
PYTHON_UNSTRUCTURED_PARAMS = "python_unstructured_params"
R_UNSTRUCTURED = "r_unstructured"
R_UNSTRUCTURED_PARAMS = "r_unstructured_params"
PYTHON_XGBOOST_CLASS_LABELS_VALIDATION = "predictions_and_class_labels_validation"

DOCKER_PYTHON_SKLEARN = "cmrunner_test_env_python_sklearn"

RESPONSE_PREDICTIONS_KEY = "predictions"
RESPONSE_TRANSFORM_KEY = "transformations"

WEIGHTS_ARGS = "weights-args"
WEIGHTS_CSV = "weights-csv"
