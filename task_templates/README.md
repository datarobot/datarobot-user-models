# Templates for Custom Tasks

In this directory are three folders - estimators, transforms, and pipelines

* estimators - meant to be used as the prediction step in a blueprint
* transforms - meant to be used as a preprocessing step
* pipelines - meant to be added as a single task blueprint, these have some preprocessing and an
  estimator built in

Our estimator examples are also different than the pipelines examples because, we decided to
implement the score hooks for all the estimator templates, and for the pipelines templates we did
not. You only don't have to use a score hook if your model inherits from sklearn, pytorch, keras, or
xgboost and doesn't need any additional functionality

Below is an enumeration of the templates we provide

## Transforms

* `custom_class_python` - Shows how to define your own transformation class without relying on
  external libraries
* `python_missing_values` - Uses builtin pandas functionality to impute missing values
* `python3_sklearn_transform_hyperparameters` - Hyperparameters functionality is currently still in
  development
* `python3_sklearn_transform` - Verified- This is a python transformer template with preproccessing
  that handles categorical and numeric variables

## Estimators

* `python_calibrator` - Shows how to create an estimator task for doing prediction calibration,
  which is usually done in DataRobot as an additional estimator step after the main estimator
* `python_anomaly` - Shows how to create an anomaly estimator
* `python_binary_classification` - Shows how to use sklearn to create an estimator for binary
  classification problems
* `python_multiclass_classification` - Implements a linear classifier with SGD training
* `python_regression` - Implements a linear regressor with SGD training
* `python_regression_with_custom_class` - Uses a custom python class, CustomCalibrator, so that we
  can store the complex state inside the object and then re-use it during scoring.
* `r_classification` - Implements a GLM binary classifier in R
* `r_regression` - Implements a GLM regressor in R

## Pipelines

Except for where noted, pipelines are also verified within our functional test framework.

* `python3_anomaly_detection` - Provides an uncalibrated Sklearn anomaly pipeline
* `python3_pytorch` - Provides a pytorch pipeline. Also uses the transform hook
* `python3_sklearn_binary_schema_validation` - Shows how to supply a model-metadata.yaml file to
  dictate which data types are allowed in this bp
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
* `r_lang` - This R model can support either binary classification or regression out of the box.
* `simple` - A pipeline where you don't even need a custom.py file. The drum_autofit function will
  mark the pipeline object so that DRUM knows that this is the object you want to use to train your
  model

