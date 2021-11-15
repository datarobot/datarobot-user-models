# Templates for Custom Tasks

In this directory there are three folders: transforms, estimators, and pipelines. 

These are collectively known as "tasks". Each of these folders
contain code snippets, i.e. tasks, that can be uploaded to DataRobot and incorporated into a blueprint. 
DataRobot will then run this custom code as part of a blueprint both during model training and 
model inference (i.e. scoring on new data). To make sure DataRobot can correctly execute this code, 
there are a few requirements:

1. There must be a custom.py or custom.R file (depending on which language you use)
2. The custom.py or custom.R file must contain one or multiple hooks.
   Note: the exact hooks you will use depend on both the task type (estimator or transform) 
   and the language (R or Python). 
   DataRobot will call these hooks during either model training or model inference (i.e scoring with new data). 
   DataRobot will pass in the appropriate data and artifacts (e.g. serialized model) to each hook 
   as parameters so that they are available to your custom code
   
There are also several recommended best practices, although these are not requirements:
1. You can include a model-metadata.yaml file. This will define what problems your custom task can be used on 
   (e.g. regression, classification, etc.) and what input it will accept (e.g. sparse vs. dense input). 
   See detailed documentation on model-metadata 
   [here](https://github.com/datarobot/datarobot-user-models/blob/master/MODEL-METADATA.md)
   or in the examples linked below 
2. You can also include multiple supporting files as needed, e.g. helper files to clean up your code.
    You can see several examples of this is in the folders linked below

Once uploaded each of these three custom task types (transforms, estimators, pipelines) will appear as a single 
box in the Blueprint. Please refer to the Composable ML documentation to learn more about how tasks work.

The primary difference between the 3 custom task types are their output and their scope:

* transforms - transform the input data, e.g. one hot encoding, numeric scaling, etc. 
  Their output is always a dataframe to be used by other transforms or estimators. 
  Note that the output also includes headers / column names. 
* estimators - predict (i.e. estimate) a new value using the input data, e.g. Logistic Regression or SVM. 
  In DataRobot the final task, i.e. box, in any Blueprint must be an estimator. Note: a single Blueprint
  can, and often does, contain multiple estimators.
* pipelines - allow a user to create a single task, i.e. box, in the Blueprint that incorporates
  multiple transforms and/or estimators. This is useful if you have a fully developed model pipeline
  with preprocessing and just wants to upload the entire functionality to DataRobot. Often 
  blueprints utilizing pipelines will only have one task, which is connected to all the 
  input data types, and handles preprocessing and prediction. One advantage of pipelines is that you can 
  then download the entire trained model from DataRobot as one file. The disadvantage is the component 
  transforms / estimators in a pipeline can't be used independently by other blueprints, e.g. if 
  you create a custom missing logic imputation that you want every blueprint to use. 
  

**Note:** We recommend that you use transforms and estimators instead of pipelines when possible 
to promote reusable tasks. Transforms and estimators can often be used across multiple blueprints, 
e.g. a categorical encoding transform might be added to both a linear model and a neural network. 
Pipelines in contrast are more difficult to reuse because they often tightly couple the transforms 
and estimators. 

**To summarize**: the key difference between transforms, estimators, and pipelines
is the "hooks", i.e. functions that are automatically called by DataRobot. Both transforms and estimators support
the init (mostly for tasks written in R to load libraries), fit, and 
load_model (used if the model is serialized in a non-standard format) hooks. 
The difference is that transforms also support the transform hook (to transform input data) while estimators
support the score hook (to generate predictions at inference time).
As you would expect, the pipelines support all of the above hooks because they can incorporate 
both transforms and estimators.

Note: Some estimator and pipeline examples below do not have score hooks. This is to demonstrate that DataRobot
can automatically apply the correct scoring functionality if your estimator or pipeline uses the default
sklearn, pytorch, keras, or xgboost scoring functions. E.g. if you have a sklearn multiclass estimator such 
as a DecisionTreeClassifer and don't have a score hook defined, then DataRobot will automatically call 
model.predict_proba() and output the results.

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