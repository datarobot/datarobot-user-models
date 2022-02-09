"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import io
import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml
from scipy.io import mmread

from tests.drum.constants import (
    ANOMALY,
    BINARY,
    BINARY_BOOL,
    BINARY_INT,
    BINARY_INT_TARGET,
    SPARSE_COLUMNS,
    BINARY_TEXT,
    BINARY_NUM_ONLY,
    BINARY_SPACES,
    CODEGEN,
    CODEGEN_AND_SKLEARN,
    PYTHON_XFORM_ESTIMATOR,
    R_XFORM_ESTIMATOR,
    KERAS,
    MOJO,
    MULTI_ARTIFACT,
    MULTICLASS,
    MULTICLASS_NUM_LABELS,
    MULTICLASS_FLOAT_LABELS,
    MULTICLASS_HIGH_CARD,
    MULTICLASS_BINARY,
    NO_CUSTOM,
    POJO,
    PYPMML,
    PYTHON,
    PYTHON_ALL_PREDICT_STRUCTURED_HOOKS,
    PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS,
    PYTHON_NO_ARTIFACT_REGRESSION_HOOKS,
    PYTHON_LOAD_MODEL,
    PYTHON_PREDICT_SPARSE,
    PYTHON_TRANSFORM_WITH_Y,
    PYTHON_TRANSFORM_DENSE_WITH_Y,
    PYTHON_TRANSFORM_SPARSE,
    PYTHON_TRANSFORM_FAIL_OUTPUT_SCHEMA_VALIDATION,
    PYTHON_UNSTRUCTURED,
    PYTHON_UNSTRUCTURED_PARAMS,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    PYTORCH,
    PYTORCH_REGRESSION,
    PYTORCH_MULTICLASS,
    R,
    R_ALL_PREDICT_STRUCTURED_HOOKS,
    R_ALL_PREDICT_STRUCTURED_HOOKS_LOWERCASE_R,
    R_FAIL_CLASSIFICATION_VALIDATION_HOOKS,
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS,
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS_LOWERCASE_R,
    R_FIT,
    R_PREDICT_SPARSE,
    R_UNSTRUCTURED,
    R_UNSTRUCTURED_PARAMS,
    RDS,
    RDS_BINARY,
    RDS_SPARSE,
    REGRESSION,
    REGRESSION_SINGLE_COL,
    REGRESSION_INFERENCE,
    REGRESSION_MULTLILINE_TEXT,
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
    PYTHON_TRANSFORM,
    PYTHON_TRANSFORM_DENSE,
    SKLEARN_TRANSFORM_NO_HOOK,
    SKLEARN_TRANSFORM_SPARSE_INPUT,
    SKLEARN_TRANSFORM_SPARSE_IN_OUT,
    SKLEARN_TRANSFORM_NON_NUMERIC,
    SKLEARN_PRED_CONSISTENCY,
    R_TRANSFORM_WITH_Y,
    R_TRANSFORM,
    R_TRANSFORM_NO_HOOK,
    R_TRANSFORM_SPARSE_INPUT,
    R_TRANSFORM_SPARSE_OUTPUT,
    R_TRANSFORM_SPARSE_IN_OUT,
    R_TRANSFORM_NON_NUMERIC,
    R_ESTIMATOR_SPARSE,
    R_VALIDATE_SPARSE_ESTIMATOR,
    SPARSE,
    SPARSE_TARGET,
    SPARSE_TRANSFORM,
    TESTS_ARTIFACTS_PATH,
    TESTS_DATA_PATH,
    TESTS_FIXTURES_PATH,
    TRAINING_TEMPLATES_PATH,
    TRANSFORM_TEMPLATES_PATH,
    ESTIMATORS_TEMPLATES_PATH,
    TRANSFORM,
    UNSTRUCTURED,
    XGB,
    TARGET_NAME_DUPLICATED_X,
    TARGET_NAME_DUPLICATED_Y,
    SKLEARN_BINARY_PARAMETERS,
    SKLEARN_BINARY_HYPERPARAMETERS,
    SKLEARN_TRANSFORM_PARAMETERS,
    SKLEARN_TRANSFORM_HYPERPARAMETERS,
    RDS_PARAMETERS,
    RDS_HYPERPARAMETERS,
    MLJ,
    JLSO,
    JULIA,
    R_INT_COLNAMES_BINARY,
    R_INT_COLNAMES_MULTICLASS,
    R_TRANSFORM_SPARSE_INPUT_Y_OUTPUT,
    SKLEARN_TRANSFORM_SPARSE_INPUT_Y_OUTPUT,
    PYTHON_UNSTRUCTURED_MLOPS,
    CUSTOM_TASK_INTERFACE_BINARY,
    CUSTOM_TASK_INTERFACE_REGRESSION,
    CUSTOM_TASK_INTERFACE_ANOMALY,
    CUSTOM_TASK_INTERFACE_MULTICLASS,
    CUSTOM_TASK_INTERFACE_TRANSFORM,
    CUSTOM_TASK_INTERFACE_PYTORCH_BINARY,
    CUSTOM_TASK_INTERFACE_PYTORCH_MULTICLASS,
    CUSTOM_TASK_INTERFACE_KERAS_REGRESSION,
    CUSTOM_TASK_INTERFACE_XGB_REGRESSION,
    BINARY_NUM_TARGET,
    MULTICLASS_LABEL_SPACES,
)
from datarobot_drum.drum.model_adapter import PythonModelAdapter


_datasets = {
    # If specific dataset should be defined for a framework, use (framework, problem) key.
    # Otherwise default dataset is used (None, problem)
    (None, REGRESSION): os.path.join(TESTS_DATA_PATH, "juniors_3_year_stats_regression.csv"),
    (None, REGRESSION_INFERENCE): os.path.join(
        TESTS_DATA_PATH, "juniors_3_year_stats_regression_inference.csv"
    ),
    (None, REGRESSION_SINGLE_COL): os.path.join(TESTS_DATA_PATH, "regression_single_col.csv"),
    (None, REGRESSION_MULTLILINE_TEXT): os.path.join(
        TESTS_DATA_PATH, "de_reviews_small_multiline.csv"
    ),
    (None, BINARY_TEXT): os.path.join(TESTS_DATA_PATH, "telecomms_churn.csv"),
    (None, BINARY_NUM_ONLY): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, BINARY_NUM_TARGET): os.path.join(TESTS_DATA_PATH, "iris_with_binary.csv"),
    (PYPMML, REGRESSION): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, BINARY): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, ANOMALY): os.path.join(TESTS_DATA_PATH, "juniors_3_year_stats_regression_inference.csv"),
    (None, UNSTRUCTURED): os.path.join(TESTS_DATA_PATH, "unstructured_data.txt"),
    (None, MULTICLASS): os.path.join(TESTS_DATA_PATH, "skyserver_sql2_27_2018_6_51_39_pm.csv"),
    (None, MULTICLASS_NUM_LABELS): os.path.join(
        TESTS_DATA_PATH, "skyserver_sql2_27_2018_6_51_39_pm_num_class.csv"
    ),
    (None, MULTICLASS_FLOAT_LABELS): os.path.join(TESTS_DATA_PATH, "telecomms_churn.csv"),
    (None, MULTICLASS_HIGH_CARD): os.path.join(
        TESTS_DATA_PATH, "skyserver_sql2_27_2018_6_51_39_pm_num_class.csv"
    ),
    (None, MULTICLASS_BINARY): os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv"),
    (None, SPARSE): os.path.join(TESTS_DATA_PATH, "sparse.mtx"),
    (None, SPARSE_TRANSFORM): os.path.join(TESTS_DATA_PATH, "sparse.mtx"),
    (None, SPARSE_TARGET): os.path.join(TESTS_DATA_PATH, "sparse_target.csv"),
    (None, SPARSE_COLUMNS): os.path.join(TESTS_DATA_PATH, "sparse.columns"),
    (None, SKLEARN_BINARY_PARAMETERS): os.path.join(
        TESTS_DATA_PATH, "sklearn_binary_parameters.json"
    ),
    (None, SKLEARN_TRANSFORM_PARAMETERS): os.path.join(
        TESTS_DATA_PATH, "sklearn_transform_parameters.json"
    ),
    (None, RDS_PARAMETERS): os.path.join(TESTS_DATA_PATH, "r_parameters.json"),
    (None, BINARY_BOOL): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (None, BINARY_INT): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (None, BINARY_INT_TARGET): os.path.join(TESTS_DATA_PATH, "int_target.csv"),
    (SKLEARN, TRANSFORM): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (SKLEARN_TRANSFORM, TRANSFORM): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (SKLEARN_TRANSFORM_DENSE, TRANSFORM): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (SKLEARN_PRED_CONSISTENCY, BINARY_BOOL): os.path.join(
        TESTS_DATA_PATH, "10k_diabetes_sample.csv"
    ),
    (None, BINARY_SPACES): os.path.join(TESTS_DATA_PATH, "iris_with_spaces.csv"),
    (SKLEARN_REGRESSION, TARGET_NAME_DUPLICATED_X): os.path.join(
        TESTS_DATA_PATH, "target_name_duplicated_X.csv"
    ),
    (SKLEARN_REGRESSION, TARGET_NAME_DUPLICATED_Y): os.path.join(
        TESTS_DATA_PATH, "target_name_duplicated_y.csv"
    ),
    (R_TRANSFORM_WITH_Y, TRANSFORM): os.path.join(TESTS_DATA_PATH, "10k_diabetes_sample.csv"),
    (R_TRANSFORM_SPARSE_OUTPUT, TRANSFORM): os.path.join(
        TESTS_DATA_PATH, "10k_diabetes_sample.csv"
    ),
    (None, MULTICLASS_LABEL_SPACES): os.path.join(TESTS_DATA_PATH, "iris_with_spaces_full.csv"),
}

_training_models_paths = {
    (PYTHON, PYTHON_XFORM_ESTIMATOR): os.path.join(TESTS_FIXTURES_PATH, "python_xform_estimator"),
    (PYTHON, SKLEARN_BINARY): os.path.join(TRAINING_TEMPLATES_PATH, "5_python3_sklearn_binary"),
    (PYTHON, SKLEARN_BINARY_HYPERPARAMETERS): os.path.join(
        TESTS_FIXTURES_PATH, "python3_sklearn_binary_hyperparameters"
    ),
    (PYTHON, SKLEARN_REGRESSION): os.path.join(
        TRAINING_TEMPLATES_PATH, "2_python3_sklearn_regression"
    ),
    (PYTHON, SKLEARN_MULTICLASS): os.path.join(
        TRAINING_TEMPLATES_PATH, "6_python3_sklearn_multiclass"
    ),
    (PYTHON, SKLEARN_TRANSFORM_HYPERPARAMETERS): os.path.join(
        TESTS_FIXTURES_PATH, "python3_sklearn_transform_hyperparameters"
    ),
    (PYTHON, SIMPLE): os.path.join(TRAINING_TEMPLATES_PATH, "1_simple"),
    (PYTHON, SKLEARN_SPARSE): os.path.join(TESTS_FIXTURES_PATH, "validate_sparse_columns"),
    (PYTHON, KERAS): os.path.join(TRAINING_TEMPLATES_PATH, "14_python3_keras_joblib"),
    (PYTHON, XGB): os.path.join(TRAINING_TEMPLATES_PATH, "4_python3_xgboost"),
    (R_FIT, RDS): os.path.join(TRAINING_TEMPLATES_PATH, "3_r_lang"),
    (R_FIT, RDS_BINARY): os.path.join(TESTS_FIXTURES_PATH, "r_binary_classifier"),
    (R_FIT, RDS_HYPERPARAMETERS): os.path.join(TESTS_FIXTURES_PATH, "r_lang_hyperparameters"),
    (R_FIT, R_ESTIMATOR_SPARSE): os.path.join(ESTIMATORS_TEMPLATES_PATH, "3_r_sparse_regression"),
    (R_FIT, RDS_SPARSE): os.path.join(TESTS_FIXTURES_PATH, "r_sparse_validation"),
    (R_FIT, R_XFORM_ESTIMATOR): os.path.join(TESTS_FIXTURES_PATH, "r_xform_estimator"),
    (PYTHON, PYTORCH): os.path.join(TRAINING_TEMPLATES_PATH, "12_python3_pytorch"),
    (PYTHON, PYTORCH_REGRESSION): os.path.join(
        TRAINING_TEMPLATES_PATH, "11_python3_pytorch_regression"
    ),
    (PYTHON, SKLEARN_ANOMALY): os.path.join(TRAINING_TEMPLATES_PATH, "7_python3_anomaly_detection"),
    (PYTHON, PYTORCH_MULTICLASS): os.path.join(
        TRAINING_TEMPLATES_PATH, "13_python3_pytorch_multiclass"
    ),
    (PYTHON, SKLEARN_PRED_CONSISTENCY): os.path.join(
        TESTS_FIXTURES_PATH, "custom_pred_consistency"
    ),
    (PYTHON, PYTHON_TRANSFORM_FAIL_OUTPUT_SCHEMA_VALIDATION): os.path.join(
        TESTS_FIXTURES_PATH, "validate_transform_fail_output_schema_validation"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_REGRESSION): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_regression"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_ANOMALY): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_anomaly"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_BINARY): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_binary"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_MULTICLASS): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_multiclass"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_TRANSFORM): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_transform_missing_values"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_KERAS_REGRESSION): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_keras_regression"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_XGB_REGRESSION): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_xgboost_regression"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_PYTORCH_BINARY): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_pytorch_binary"
    ),
    (PYTHON, CUSTOM_TASK_INTERFACE_PYTORCH_MULTICLASS): os.path.join(
        TESTS_FIXTURES_PATH, "custom_task_interface_pytorch_multiclass"
    ),
}

_targets = {
    BINARY: "Species",
    REGRESSION: "Grade 2014",
    REGRESSION_SINGLE_COL: "target",
    BINARY_TEXT: "Churn",
    BINARY_NUM_ONLY: "Species",
    BINARY_NUM_TARGET: "target",
    MULTICLASS: "class",
    MULTICLASS_BINARY: "Species",
    MULTICLASS_NUM_LABELS: "class",
    MULTICLASS_FLOAT_LABELS: "Total.intl.minutes",
    MULTICLASS_HIGH_CARD: "run",
    SPARSE: "my_target",
    BINARY_BOOL: "readmitted",
    BINARY_INT: "is_bad",
    ANOMALY: None,
    TRANSFORM: os.path.join(TESTS_DATA_PATH, "transform_target.csv"),
    BINARY_SPACES: "Species",
    REGRESSION_MULTLILINE_TEXT: "rating",
    MULTICLASS_LABEL_SPACES: "Species",
}

_target_types = {
    BINARY: "binary",
    BINARY_TEXT: "binary",
    BINARY_NUM_ONLY: "binary",
    BINARY_NUM_TARGET: "binary",
    BINARY_SPACES: "binary",
    REGRESSION: "regression",
    REGRESSION_SINGLE_COL: "regression",
    REGRESSION_MULTLILINE_TEXT: "regression",
    REGRESSION_INFERENCE: "regression",
    ANOMALY: "anomaly",
    UNSTRUCTURED: "unstructured",
    SPARSE: "regression",
    SPARSE_TRANSFORM: "transform",
    MULTICLASS: "multiclass",
    MULTICLASS_BINARY: "multiclass",
    MULTICLASS_NUM_LABELS: "multiclass",
    MULTICLASS_FLOAT_LABELS: "multiclass",
    MULTICLASS_HIGH_CARD: "multiclass",
    BINARY_BOOL: "binary",
    BINARY_INT: "binary",
    TRANSFORM: "transform",
    MULTICLASS_LABEL_SPACES: "multiclass",
}

_class_labels = {
    (SKLEARN_BINARY, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (CUSTOM_TASK_INTERFACE_BINARY, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (SKLEARN_BINARY_HYPERPARAMETERS, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (SKLEARN_BINARY, BINARY_SPACES): ["Iris setosa", "Iris versicolor"],
    (CUSTOM_TASK_INTERFACE_BINARY, BINARY_SPACES): ["Iris setosa", "Iris versicolor"],
    (SKLEARN, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (XGB, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (KERAS, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (RDS, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (RDS_BINARY, BINARY_INT): ["0", "1"],
    (PYPMML, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (PYTORCH, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (CODEGEN, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (MOJO, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (POJO, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (SKLEARN, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (SKLEARN_MULTICLASS, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (SKLEARN_MULTICLASS, MULTICLASS_NUM_LABELS): ["0", "1", "2"],
    (SKLEARN_MULTICLASS, MULTICLASS_NUM_LABELS): [0, 1, 2],
    (SKLEARN_MULTICLASS, MULTICLASS_FLOAT_LABELS): [
        10.0,
        13.7,
        12.2,
        6.6,
        10.1,
        6.3,
        7.5,
        7.1,
        8.7,
        11.2,
        12.7,
        9.1,
        12.3,
        13.1,
        5.4,
        13.8,
        8.1,
        13.0,
        10.6,
        5.7,
        9.5,
        7.7,
        10.3,
        15.5,
        14.7,
        11.1,
        14.2,
        12.6,
        11.8,
        8.3,
        14.5,
        10.5,
        9.4,
        14.6,
        9.2,
        3.5,
        8.5,
        13.2,
        7.4,
        8.8,
        11.0,
        7.8,
        6.8,
        11.4,
        9.3,
        9.7,
        10.2,
        8.0,
        5.8,
        12.1,
        12.0,
        11.6,
        8.2,
        6.2,
        7.3,
        6.1,
        11.7,
        15.0,
        9.8,
        12.4,
        8.6,
        10.9,
        13.9,
        8.9,
        7.9,
        5.3,
    ],
    (PYTORCH_MULTICLASS, MULTICLASS_HIGH_CARD): [
        "752"
        "756"
        "308"
        "727"
        "745"
        "1035"
        "1045"
        "1140"
        "1231"
        "1332"
        "1334"
        "1302"
        "1239"
        "1119"
        "1331"
        "1345"
        "1350"
        "1404"
        "1412"
        "1336"
        "1402"
        "1411"
        "1356",
    ],
    (XGB, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (KERAS, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (RDS, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (PYPMML, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (PYTORCH_MULTICLASS, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (PYTORCH, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (CODEGEN, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (SKLEARN_BINARY, BINARY_TEXT): ["False", "True"],
    (CUSTOM_TASK_INTERFACE_BINARY, BINARY_NUM_ONLY): ["Iris-setosa", "Iris-versicolor"],
    (CUSTOM_TASK_INTERFACE_BINARY, BINARY_NUM_TARGET): [0, 1],
    (XGB, BINARY_TEXT): ["False", "True"],
    (KERAS, BINARY_TEXT): ["False", "True"],
    (POJO, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (MOJO, MULTICLASS): ["GALAXY", "QSO", "STAR"],
    (SKLEARN_BINARY, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (CUSTOM_TASK_INTERFACE_BINARY, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (SKLEARN, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (XGB, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (KERAS, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (RDS, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (PYPMML, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (PYTORCH, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (CODEGEN, MULTICLASS_BINARY): ["yes", "no"],
    (MOJO, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (POJO, MULTICLASS_BINARY): ["Iris-setosa", "Iris-versicolor"],
    (SKLEARN_PRED_CONSISTENCY, BINARY_BOOL): ["False", "True"],
    (MLJ, BINARY): ["Iris-setosa", "Iris-versicolor"],
    (MLJ, MULTICLASS): ["GALAXY", "QSO", "STAR"],
}

_artifacts = {
    (None, None): None,
    (None, REGRESSION): None,
    (None, BINARY): None,
    (None, UNSTRUCTURED): None,
    (SKLEARN, SPARSE): os.path.join(TESTS_ARTIFACTS_PATH, "sklearn_dtr_sparse.pkl"),
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
    (RDS_SPARSE, SPARSE): os.path.join(TESTS_ARTIFACTS_PATH, "r_reg_sparse.rds"),
    (RDS, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "r_bin.rds"),
    (CODEGEN, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "java_reg.jar"),
    (CODEGEN, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "java_bin.jar"),
    (POJO, REGRESSION): os.path.join(
        TESTS_ARTIFACTS_PATH, "DRF_model_python_1633108695693_1.java",
    ),
    (POJO, BINARY): os.path.join(
        TESTS_ARTIFACTS_PATH, "XGBoost_grid__1_AutoML_20200717_163214_model_159.java",
    ),
    (POJO, MULTICLASS): os.path.join(
        TESTS_ARTIFACTS_PATH, "XGBoost_3_AutoML_20201016_143029.java",
    ),
    (MOJO, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_reg.zip"),
    (MOJO, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_bin.zip"),
    (MOJO, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "mojo_multi.zip"),
    (MLJ, REGRESSION): os.path.join(TESTS_ARTIFACTS_PATH, "grade_regression.jlso"),
    (MLJ, BINARY): os.path.join(TESTS_ARTIFACTS_PATH, "iris_binary.jlso"),
    (MLJ, MULTICLASS): os.path.join(TESTS_ARTIFACTS_PATH, "galaxy.jlso"),
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
    (SKLEARN_TRANSFORM, SPARSE_TRANSFORM): os.path.join(
        TESTS_ARTIFACTS_PATH, "transform_sparse.pkl"
    ),
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
        TESTS_ARTIFACTS_PATH, "XGBoost_grid__1_AutoML_20200717_163214_model_159.java",
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
    (SKLEARN_TRANSFORM_SPARSE_INPUT_Y_OUTPUT, REGRESSION): None,
    (SKLEARN_TRANSFORM_SPARSE_INPUT_Y_OUTPUT, BINARY): None,
    (SKLEARN_TRANSFORM_SPARSE_INPUT_Y_OUTPUT, ANOMALY): None,
    (SKLEARN_TRANSFORM_SPARSE_IN_OUT, REGRESSION): None,
    (SKLEARN_TRANSFORM_SPARSE_IN_OUT, BINARY): None,
    (SKLEARN_TRANSFORM_SPARSE_IN_OUT, ANOMALY): None,
    (SKLEARN_TRANSFORM_NON_NUMERIC, REGRESSION): None,
    (SKLEARN_TRANSFORM_NON_NUMERIC, BINARY): None,
    (SKLEARN_TRANSFORM_NON_NUMERIC, ANOMALY): None,
    (R_TRANSFORM_WITH_Y, TRANSFORM): os.path.join(TESTS_ARTIFACTS_PATH, "r_transform.rds"),
    (R_TRANSFORM_WITH_Y, REGRESSION): None,
    (R_TRANSFORM_WITH_Y, BINARY): None,
    (R_TRANSFORM_WITH_Y, ANOMALY): None,
    (R_TRANSFORM, REGRESSION): None,
    (R_TRANSFORM, BINARY): None,
    (R_TRANSFORM, ANOMALY): None,
    (R_TRANSFORM_NO_HOOK, REGRESSION): None,
    (R_TRANSFORM_NO_HOOK, BINARY): None,
    (R_TRANSFORM_NO_HOOK, ANOMALY): None,
    (R_TRANSFORM_SPARSE_INPUT, REGRESSION): None,
    (R_TRANSFORM_SPARSE_INPUT, BINARY): None,
    (R_TRANSFORM_SPARSE_INPUT, ANOMALY): None,
    (R_TRANSFORM_SPARSE_INPUT, SPARSE_TRANSFORM): os.path.join(
        TESTS_ARTIFACTS_PATH, "r_sparse_transform.rds"
    ),
    (R_TRANSFORM_SPARSE_OUTPUT, TRANSFORM): os.path.join(
        TESTS_ARTIFACTS_PATH, "r_sparse_transform.rds"
    ),
    (R_TRANSFORM_SPARSE_INPUT_Y_OUTPUT, REGRESSION): None,
    (R_TRANSFORM_SPARSE_INPUT_Y_OUTPUT, BINARY): None,
    (R_TRANSFORM_SPARSE_INPUT_Y_OUTPUT, ANOMALY): None,
    (R_TRANSFORM_SPARSE_IN_OUT, REGRESSION): None,
    (R_TRANSFORM_SPARSE_IN_OUT, BINARY): None,
    (R_TRANSFORM_SPARSE_IN_OUT, ANOMALY): None,
    (R_TRANSFORM_NON_NUMERIC, REGRESSION): None,
    (R_TRANSFORM_NON_NUMERIC, BINARY): None,
    (R_TRANSFORM_NON_NUMERIC, ANOMALY): None,
    (R_ESTIMATOR_SPARSE, REGRESSION): None,
    (R_VALIDATE_SPARSE_ESTIMATOR, REGRESSION): None,
    (CUSTOM_TASK_INTERFACE_BINARY, BINARY): None,
    (CUSTOM_TASK_INTERFACE_REGRESSION, REGRESSION): None,
    (CUSTOM_TASK_INTERFACE_ANOMALY, ANOMALY): None,
    (CUSTOM_TASK_INTERFACE_TRANSFORM, TRANSFORM): None,
    (CUSTOM_TASK_INTERFACE_TRANSFORM, BINARY): None,
    (CUSTOM_TASK_INTERFACE_MULTICLASS, MULTICLASS): None,
    (CUSTOM_TASK_INTERFACE_TRANSFORM, REGRESSION): None,
    (CUSTOM_TASK_INTERFACE_TRANSFORM, MULTICLASS): None,
    (CUSTOM_TASK_INTERFACE_PYTORCH_BINARY, BINARY): None,
    (CUSTOM_TASK_INTERFACE_PYTORCH_MULTICLASS, MULTICLASS): None,
    (CUSTOM_TASK_INTERFACE_KERAS_REGRESSION, REGRESSION): None,
    (CUSTOM_TASK_INTERFACE_XGB_REGRESSION, REGRESSION): None,
}

_custom_filepaths = {
    PYTHON: (os.path.join(TESTS_FIXTURES_PATH, "custom.py"), "custom.py"),
    JULIA: (os.path.join(TESTS_FIXTURES_PATH, "custom.jl"), "custom.jl"),
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
    PYTHON_LOAD_MODEL: (os.path.join(TESTS_FIXTURES_PATH, "load_model_custom.py"), "custom.py",),
    PYTHON_NO_ARTIFACT_REGRESSION_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "no_artifact_regression_custom.py"),
        "custom.py",
    ),
    # sparse_predict.* for sparse.mtx, checks that data is sparse matrix and contains column names
    R_PREDICT_SPARSE: (os.path.join(TESTS_FIXTURES_PATH, "sparse_predict.R"), "custom.R",),
    PYTHON_PREDICT_SPARSE: (os.path.join(TESTS_FIXTURES_PATH, "sparse_predict.py"), "custom.py",),
    R: (os.path.join(TESTS_FIXTURES_PATH, "custom.R"), "custom.R"),
    R_ALL_PREDICT_STRUCTURED_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "all_predict_structured_hooks_custom.R"),
        "custom.R",
    ),
    R_ALL_PREDICT_STRUCTURED_HOOKS_LOWERCASE_R: (
        os.path.join(TESTS_FIXTURES_PATH, "all_predict_structured_hooks_custom_lowercase_r.r"),
        "custom.r",
    ),
    R_FAIL_CLASSIFICATION_VALIDATION_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "fail_classification_validation_custom.R"),
        "custom.R",
    ),
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS: (
        os.path.join(TESTS_FIXTURES_PATH, "all_predict_unstructured_hooks_custom.R"),
        "custom.R",
    ),
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS_LOWERCASE_R: (
        os.path.join(TESTS_FIXTURES_PATH, "all_predict_unstructured_hooks_custom_lowercase_r.r"),
        "custom.r",
    ),
    R_FIT: (os.path.join(TESTS_FIXTURES_PATH, "fit_custom.R"), "custom.R"),
    PYTHON_UNSTRUCTURED: (os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom.py"), "custom.py"),
    R_UNSTRUCTURED: (os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom.R"), "custom.R"),
    PYTHON_UNSTRUCTURED_MLOPS: (
        os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom_mlops.py"),
        "custom.py",
    ),
    PYTHON_UNSTRUCTURED_PARAMS: (
        os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom_params.py"),
        "custom.py",
    ),
    R_UNSTRUCTURED_PARAMS: (
        os.path.join(TESTS_FIXTURES_PATH, "unstructured_custom_params.R"),
        "custom.R",
    ),
    R_INT_COLNAMES_BINARY: (os.path.join(TESTS_FIXTURES_PATH, "int_colnames_binary.R"), "custom.R"),
    R_INT_COLNAMES_MULTICLASS: (
        os.path.join(TESTS_FIXTURES_PATH, "int_colnames_multiclass.R"),
        "custom.R",
    ),
    PYTHON_TRANSFORM_WITH_Y: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_custom_with_y.py"),
        "custom.py",
    ),
    PYTHON_TRANSFORM_DENSE_WITH_Y: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_custom_with_y.py"),
        "custom.py",
    ),
    PYTHON_TRANSFORM: (os.path.join(TESTS_FIXTURES_PATH, "transform_custom.py"), "custom.py",),
    PYTHON_TRANSFORM_DENSE: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_custom.py"),
        "custom.py",
    ),
    PYTHON_TRANSFORM_SPARSE: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_sparse_input.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_WITH_Y: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_with_y.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_NO_HOOK: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_no_hook.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM: (os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom.py"), "custom.py",),
    SKLEARN_TRANSFORM_SPARSE_INPUT: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_sparse_input.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_SPARSE_INPUT_Y_OUTPUT: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_sparse_input_y_output.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_SPARSE_IN_OUT: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_sparse_in_out.py"),
        "custom.py",
    ),
    SKLEARN_TRANSFORM_NON_NUMERIC: (
        os.path.join(TESTS_FIXTURES_PATH, "transform_fit_custom_non_numeric.py"),
        "custom.py",
    ),
    R_TRANSFORM_WITH_Y: (
        os.path.join(TESTS_FIXTURES_PATH, "r_transform_custom_with_y.R"),
        "custom.R",
    ),
    R_TRANSFORM: (os.path.join(TESTS_FIXTURES_PATH, "r_transform_custom.R"), "custom.R",),
    R_TRANSFORM_NO_HOOK: (
        os.path.join(TESTS_FIXTURES_PATH, "r_transform_fit_custom_no_hook.R"),
        "custom.R",
    ),
    R_TRANSFORM_SPARSE_INPUT: (
        os.path.join(TESTS_FIXTURES_PATH, "r_transform_fit_custom_sparse_input.R"),
        "custom.R",
    ),
    R_TRANSFORM_SPARSE_OUTPUT: (
        os.path.join(TESTS_FIXTURES_PATH, "r_transform_custom_sparse_output.R"),
        "custom.R",
    ),
    R_TRANSFORM_SPARSE_INPUT_Y_OUTPUT: (
        os.path.join(TESTS_FIXTURES_PATH, "r_transform_fit_custom_sparse_input_y_output.R"),
        "custom.R",
    ),
    R_TRANSFORM_SPARSE_IN_OUT: (
        os.path.join(TESTS_FIXTURES_PATH, "r_transform_fit_custom_sparse_in_out.R"),
        "custom.R",
    ),
    R_TRANSFORM_NON_NUMERIC: (
        os.path.join(TESTS_FIXTURES_PATH, "r_transform_fit_custom_non_numeric.R"),
        "custom.R",
    ),
    R_VALIDATE_SPARSE_ESTIMATOR: (
        os.path.join(TESTS_FIXTURES_PATH, "r_validate_sparse_estimator.R"),
        "custom.R",
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
def get_input_data(get_dataset_filename):
    def _foo(framework, problem):
        dataset_path = get_dataset_filename(framework, problem)
        column_file = dataset_path.replace(".mtx", ".columns")
        if problem in {SPARSE, SPARSE_TRANSFORM}:
            with open(column_file) as f:
                columns = [c.rstrip() for c in f]
            with open(dataset_path, "rb") as f:
                return pd.DataFrame.sparse.from_spmatrix(
                    mmread(io.BytesIO(f.read())), columns=columns
                )
        else:
            return pd.read_csv(dataset_path)

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
    get_input_data,
):
    resource = Resource()
    resource.datasets = get_dataset_filename
    resource.training_models = get_paths_to_training_models
    resource.targets = get_target
    resource.target_types = get_target_type
    resource.class_labels = get_class_labels
    resource.artifacts = get_artifacts
    resource.custom = get_custom
    resource.input_data = get_input_data
    return resource


# fixtures for variety data tests
with open(os.path.join(TESTS_DATA_PATH, "variety_samples/variety_data_key.yaml")) as yamlfile:
    variety_data_dict = yaml.safe_load(yamlfile)

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


@pytest.fixture
def essential_language_predictor_init_params():
    return {
        "__custom_model_path__": "custom_model_path",
        "monitor": False,
    }


@pytest.fixture
def mock_python_model_adapter_predict():
    with patch.object(PythonModelAdapter, "predict") as mock_predict:
        mock_predict.return_value = None, None
        yield mock_predict


@pytest.fixture
def mock_python_model_adapter_load_model_from_artifact():
    with patch.object(
        PythonModelAdapter, "load_model_from_artifact"
    ) as mock_load_model_from_artifact:
        mock_load_model_from_artifact.return_value = Mock()
        yield mock_load_model_from_artifact
