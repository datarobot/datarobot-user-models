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

* `python_missing_values` - Good starter template. It uses builtin pandas functionality to impute
  missing values
* `custom_class_python` - Shows how to define your own transformation class without relying on
  external libraries
* `python3_sklearn_transform_hyperparameters` - Hyperparameters functionality is currently still in
  development
* `python3_sklearn_transform` - Verified - This is a python template implementing a
  preprocessing-only SKLearn pipeline, handling categorical and numeric variables
* `python3_image_transform` - Shows how to handle input and output of image data type with DataRobot

## Estimators

* `python_anomaly` - Shows how to create an anomaly estimator
* `python_binary_classification` - Shows how to create an estimator for binary
  classification problems
* `python_multiclass_classification` - Implements a linear classifier with SGD training
* `python_regression` - Implements a linear regressor with SGD training
* `python_regression_with_custom_class` - Uses a custom python class, CustomCalibrator, which
  implements fit and score from scratch with no external libraries
* `python_calibrator` - Shows how to create an estimator task for doing prediction calibration,
  which is usually done in DataRobot as an additional estimator step after the main estimator
* `r_classification` - Implements a GLM binary classifier in R
* `r_regression` - Implements a GLM regressor in R

## Pipelines

Except for where noted, our pipelines are verified within our functional test framework.

* `python3_anomaly_detection` - Provides an uncalibrated Sklearn anomaly pipeline
* `python3_pytorch` - Provides a pytorch pipeline. Also uses the transform hook
* `python3_sklearn_binary_schema_validation` - Shows how to supply a model-metadata.yaml file to
  dictate which data types are allowed in this task.
* `python3_calibrated_anomaly_detection` - Provides a calibrated Sklearn anomaly pipeline
* `python3_pytorch_multiclass` - Parses the passed in class_labels file. Also uses transform, and
  also relies on the internal PyTorch prediction implementation
* `python3_sklearn_multiclass` - Handles text, numeric and categorical inputs. Relies on sklearn
  dropin env and predictor.
* `python3_sklearn_with_custom_classes` - This pipeline shows you how to build an estimator
  independent from any library [Not Verified]
* `python3_keras_joblib` - Contains the option for both binary classification and regression.
  Serializes to the h5 format
* `python3_sklearn_binary` - Preprocessing with numeric, categorical and text, then SVD, with a
  linear model estimator at the end
* `python3_sklearn_regression` - Preprocessing with numeric, categorical and text, then SVD, with a
  Ridge regression estimator at the end
* `python3_sparse` - A pipeline that intakes sparse data (csr_matrix) from an MTX file
* `python3_keras_vizai_joblib` - Trains a keras model on base64 encoded images.
* `python3_xgboost` - XGBoost also has a DRUM predictor provided that knows how to predict all
  XGBoost models
* `r_lang` - This R pipeline can support either binary classification or regression out of the box.
* `simple` - A pipeline where you don't even need a custom.py file. The drum_autofit function will
  mark the pipeline object so that DRUM knows that this is the object you want to use to train your
  model

There is also a growing repository of re-usable tasks that might address your use case. You can find
it here: https://github.com/datarobot-community/custom-models/tree/master/custom_tasks