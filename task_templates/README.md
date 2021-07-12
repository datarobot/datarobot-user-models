# Templates for Custom Tasks
In this directory are three folders - estimators, transforms, and pipelines

* estimators - meant to be used as the prediction step in a blueprint
* transforms - meant to be used as a preprocessing step
* pipelines - meant to be added as a single task blueprint, these have some preprocessing and an estimator built in

Our estimator examples are also different than the pipelines examples because, we decided to implement the score
hooks for all the estimator templates, and for the pipelines templates we did not. You only don't have to use a score
hook if your model inherits from sklearn, pytorch, keras, or xgboost and doesn't need any additional functionality

Below is an enumeration of the templates we provide

## Transforms
* `custom_class_python` - Shows how to define your own transformation class without relying on external libraries
* `python_missing_values` - Uses builtin pandas functionality to impute missing values
 * `python3_sklearn_transform_hyperparameters` - Verified
 * `python3_sklearn_transform` - Verified

## Estimators 
* `python_calibrator` - Shows how to create an estimator task for doing prediction calibration, which
is usually done in DataRobot as an additional estimator step after the main estimator
* `python_anomaly` - Shows how to use sklearn to create an anomaly estimator
* `python_binary_classification` - Shows how to use sklearn to create an estimator for binary classification problems
* `python_multiclass_classification` 
* `python_regression`
* `python_regression_with_custom_class` - Does not inherit sklearn for the estimator
* `r_classification`
* `r_regression`

## Pipelines
Except for where noted, pipelines are also verified within our functional test framework. 

 * `python3_anomaly_detection`
 * `python3_pytorch`
 * `python3_sklearn_binary_schema_validation`
 * `python3_calibrated_anomaly_detection`
 * `python3_pytorch_multiclass`
 * `python3_sklearn_multiclass`
 * `python3_sklearn_with_custom_classes` - This pipeline shows you how to not use one of the main libraries (untested)
 * `python3_keras_joblib`
 * `python3_sklearn_binary`
 * `python3_sklearn_regression`
 * `python3_sparse`
 * `python3_keras_vizai_joblib`
 * `python3_sklearn_binary_hyperparameters`
 * `python3_xgboost`
 * `r_lang`
 * `r_lang_hyperparameters`
 * `simple`
