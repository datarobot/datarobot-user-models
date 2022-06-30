 # Custom Environment Templates
The [public_dropin_environments](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments)
contains templates for the base environments used in DataRobot.
Dependency requirements can be applied to the base environment to create a
runtime environment for both custom tasks and/or custom inference models.
A custom environment defines the runtime environment for either a custom task 
or custom inference model.
In this repository, we provide several example environments that you can use and modify:
* [Python 3 + sklearn](python3_sklearn)
* [Python 3 + PyTorch](python3_pytorch)
* [Python 3 + xgboost](python3_xgboost)
* [Python 3 + keras/tensorflow](python3_keras)
* [Python 3 + ONNX](python3_onnx)
* [Python 3 + pmml](python3_pmml)
* [R + caret](r_lang)
* [Java Scoring Code](java_codegen)
* [Julia + MLJ](../example_dropin_environments/julia_mlj)

These sample environments each define the libraries available in the environment 
and are designed to allow for simple custom inference models to be made that 
consist solely of your model's artifacts and an optional custom code file, if necessary.

For detailed information on how to create models that work in these environments, 
reference the links above for each environment.