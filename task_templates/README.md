# Templates for Custom Tasks

In this directory there are three folders - estimators, transforms, and pipelines. 
These are collectively known as "tasks". Each of these folders
contain code snippets, i.e. tasks, that can be uploaded to DataRobot and incorporated into a blueprint. 
The key "hooks" (see below) are impemented in the main file called custom.py, custom.R, etc. You 
can use multiple supporting files as well as needed, e.g. to clean up your code with helper functions.  
Once uploaded each tasks appears as a single box in the Blueprint. 
The primary difference between these three are their output and their scope:

* transforms - transform the input data, e.g. one hot encoding, numeric scaling, etc. 
  Their output is always a dataframe to be used by other transforms or estimators.
* estimators - predict (i.e. estimate) a new value using the input data, e.g. Logistic Regression or SVM. 
  In DataRobot the final task, i.e. box, in any Blueprint must be an estimator. Note: a single Blueprint
  can, and often does, contain multiple estimators.
* pipelines - allow a user to create a single task, i.e. box, in the Blueprint that incorporates
multiple transforms and/or estimators. This is useful if a user has a fully developed model pipeline
  with preprocessing and just wants to upload the entire functionality to DataRobot. Often 
  blueprints utilizing pipelines will only have one task, which is connected to all the 
  input data types, and handles preprocessing and prediction. One advantage of pipelines is that you can 
  then download the entire trained model from DataRobot as one file. The disadvantage is the component 
  transforms / estimators in a pipeline can't be used independently by other blueprints, e.g. if 
  you create a custom missing logic imputation that you want every blueprint to use. 

As a simple example, let's say we have custom logic to impute missing values. We look at the 
example template found in the transforms folders (e.g. https://github.com/datarobot/datarobot-user-models/blob/master/task_templates/transforms/python_missing_values/custom.py) 
and code our imputation logic into the 
fit hook (note: the "fit" and "transform" hooks are always required for transform tasks). Notice how in the transform hook we apply
the median value imputation we learned in fit. This is because the transform hook is called at both trainging and scoring time
while the fit hook is ONLY called during training. 
Now we have a "custom missing value imputation" transform. Now let's say we want to also create a custom task to apply a weight of evidence 
(see https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html) encoding to 
our categorical features. We also use a template from the transforms as a base, but this time
we use the fit hook to learn the encoding and then use the transform hook to apply it. 
Finally we use the template in estimators to build out a 
Multivariate Adaptive Regression Splines (MARS) estimator (see https://github.com/scikit-learn-contrib/py-earth) . We add the model.fit() logic to the fit 
hook, ensuring that we output the model to a file, e.g. a pickle file for python. 
We then add the model.predict() logic to the score hook, optionally using the load_model ONLY IF our model
is stored in a non-default format (see https://github.com/datarobot/datarobot-user-models#built-in-model-support)
Don't forget to add the pyearth package (which provides the MARS algorithm) to the requirements.txt file 
so DataRobot can import it!

Now we can upload each of our two transforms and our MARS estimator to DataRobot and build our own blueprint. This 
blueprint can leverage not only our custom estimators / transforms but also DataRobot's library of built in tasks!
Note that we could have instead built out one pipeline and included ALL of the functionality of the transforms
and MARS estimator. The drawback is that we can't then use our custom transformers independently. 

As you hopefully saw in this example, the key difference between transforms, estimators, and pipelines
is the "hooks", i.e. functions that are automatically called by DataRobot. The transforms support the 
init (mostly for R to load libraries), fit, load_model (only if model is stored in non-standard format), 
and most importantly transform hooks. The estimators support the init, fit, load-model, and score 
hooks. As you would expect, the pipeline estimators support all of the hooks for estimators and transforms.


Note: The pipelines examples do not have score hooks to demonstrate the automatic
scoring functionality of the DRUM tool middleware layer. This functionality works if your estimator
inherits from sklearn, pytorch, keras, or xgboost, and doesn't have any additional functionality.

If a template is tagged as "Verified", this means we have automated tests which guarantee
integration functionality with the DataRobot platform

## Transforms

* `python_missing_values` - Good starter template. It uses builtin pandas functionality to impute
  missing values
* `custom_class_python` - Shows how to define your own transformation class without relying on
  external libraries
* `python3_sklearn_transform_hyperparameters` - Hyperparameters functionality is currently still in
  development
* `python3_sklearn_transform` - Verified - This is a python template implementing a
  preprocessing-only SKLearn pipeline, handling categorical and numeric variables

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