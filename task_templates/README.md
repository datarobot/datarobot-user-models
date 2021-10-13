# Templates for Custom Tasks

In this directory are three folders - estimators, transforms, and pipelines

* estimators - meant to be used as the prediction step in a blueprint
* transforms - meant to be used as a preprocessing step in a blueprint
* pipelines - meant to be added as a single task blueprint, these have preprocessing and an
  estimator built into a single task

Our estimator examples are also different from the pipelines examples because these templates have
score hooks implemented. The pipelines examples do not have score hooks to demonstrate the automatic
scoring functionality of the DRUM tool middleware layer. This functionality works if your estimator
inherits from sklearn, pytorch, keras, or xgboost, and doesn't have any additional functionality.

If a template is tagged as "Verified", this means we have automated tests which guarantee
integration functionality with the DataRobot platform

## Data format

When working with structured models DRUM supports data as files of `csv`, `sparse`, or `arrow` format.   
DRUM doesn't perform sanitation of missing or strange(containing parenthesis, slash, etc symbols) column names.

## Transforms

1. `python_missing_values` - Good starter python transform. It uses builtin pandas functionality to impute
  missing values
2. `r_transform_simple` - Good starter R transform. It imputes missing values with a median
2. `python3_sklearn_transform` - Verified - This is a python template implementing a
   preprocessing-only SKLearn pipeline, handling categorical and numeric variables using
   an external helper file
3. `r_transform_recipe` - Demonstrates how to use caret style recipe in R transform
4. `custom_class_python` - Shows how to define your own transformation class without relying on
   external libraries
5. `python3_image_transform` - Shows how to handle input and output of image data type with DataRobot
6. `python3_sklearn_transform_hyperparameters` - Hyperparameters functionality is currently still in
  development

## Estimators

1. `python_regression` - Implements a linear regressor with SGD training
2. `r_regression` - Implements a GLM regressor in R
3. `r_sparse_regression` - Demonstrates how to use sparse input with R
4. `python_binary_classification` - Shows how to create an estimator for binary
  classification problems
5. `r_binary_classification` - Implements a GLM binary classifier in R
6. `python_multiclass_classification` - Implements a linear classifier with SGD training
7. `python_anomaly` - Shows how to create an anomaly estimator
8. `r_anomaly_detection`  - Shows how to create an anomaly estimator in R
9. `python_regression_with_custom_class` - Uses a custom python class, CustomCalibrator, which
  implements fit and score from scratch with no external libraries
10. `python_calibrator` - Shows how to create an estimator task for doing prediction calibration,
  which is usually done in DataRobot as an additional estimator step after the main estimator

## Pipelines

Except for where noted, our pipelines are verified within our functional test framework.

1. `simple` - A pipeline where you don't even need a custom.py file. The drum_autofit function will
    mark the pipeline object so that DRUM knows that this is the object you want to use to train your
    model
2. `python3_sklearn_regression` - Preprocessing with numeric, categorical and text, then SVD, with a
    Ridge regression estimator at the end
3. `r_lang` - This R pipeline can support either binary classification or regression out of the box.
4. `python3_xgboost` - XGBoost also has a DRUM predictor provided that knows how to predict all
    XGBoost models
5. `python3_sklearn_binary` - Preprocessing with numeric, categorical and text, then SVD, with a
    linear model estimator at the end
6. `python3_sklearn_multiclass` - Handles text, numeric and categorical inputs. Relies on sklearn
     dropin env and predictor. 
7. `python3_anomaly_detection` - Provides an uncalibrated Sklearn anomaly pipeline
8. `python3_calibrated_anomaly_detection` - Provides a calibrated Sklearn anomaly pipeline
9. `python3_sklearn_with_custom_classes` - This pipeline shows you how to build an estimator
    independent from any library [Not Verified]
10. `python3_sparse` - A pipeline that intakes sparse data (csr_matrix) from an MTX file
11. `python3_pytorch_regression` - Provides a pytorch pipeline. Also uses the transform hook
12. `python3_pytorch` - Provides a pytorch pipeline. Also uses the transform hook
13. `python3_pytorch_multiclass` - Parses the passed in class_labels file. Also uses transform, and
     also relies on the internal PyTorch prediction implementation
14. `python3_keras_joblib` - Contains the option for both binary classification and regression.
    Serializes to the h5 format
15. `python3_keras_vizai_joblib` - Trains a keras model on base64 encoded images.

There is also a growing repository of re-usable tasks that might address your use case. You can find
it here: https://github.com/datarobot-community/custom-models/tree/master/custom_tasks