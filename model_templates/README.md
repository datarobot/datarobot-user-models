# Custom Inference Model Templates
This folder provides templates for building and deploying custom models in DataRobot. Use the templates as an example structure for your own custom models.

### DataRobot User Model Runner
The examples in this repository use the DataRobot User Model Runner (DRUM).  For more information on how to use and write models with DRUM, reference the [readme](./custom_model_runner/README.md).

### Sample Models
The [model_templates](model_templates) folder contains sample models that work with the provided template environments. For more information about each model, reference the readme for every example:

##### Inference Models
* [Scikit-Learn sample model](model_templates/python3_sklearn)
* [Scikit-Learn sample unsupervised anomaly detection model](model_templates/python3_sklearn_anomaly)
* [PyTorch sample model](model_templates/python3_pytorch)
* [XGBoost sample model](model_templates/python3_xgboost)
* [Keras sample model](model_templates/python3_keras)
* [Keras sample model + Joblib artifact](model_templates/python3_keras_joblib)
* [PyPMML sample model](model_templates/python3_pmml)
* [R sample model](model_templates/r_lang)
* [Java sample model](model_templates/java_codegen)
* [Julia sample models](model_templates/julia)

##### Training Models
* [Scikit-Learn sample regression model](task_templates/pipelines/python3_sklearn_regression)
* [Scikit-Learn sample binary model](task_templates/pipelines/python3_sklearn_binary)
* [Scikit-Learn sample unsupervised anomaly detection model](task_templates/pipelines/python3_anomaly_detection)
> Note: Unsupervised support is limited to anomaly detection models as of release 1.1.5
* [Scikit-Learn sample transformer](task_templates/transforms/python3_sklearn_transform)
* [XGBoost sample model](task_templates/pipelines/python3_xgboost)
* [Keras sample model + Joblib artifact](task_templates/pipelines/python3_keras_joblib)
* [R sample model](task_templates/pipelines/r_lang)

__