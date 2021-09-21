 # Custom Environment Templates
The [public_dropin_environments](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments)
contains templates for the base environments used in DataRobot.
Dependency requirements can be applied to the base environment to create a
runtime environment for both custom tasks and/or custom inference models.
A custom environment defines the runtime environment for either a custom task 
or custom inference model.
In this repository, we provide several example environments that you can use and modify:
* [Python 3 + sklearn](public_dropin_environments/python3_sklearn)
* [Python 3 + PyTorch](public_dropin_environments/python3_pytorch)
* [Python 3 + xgboost](public_dropin_environments/python3_xgboost)
* [Python 3 + keras/tensorflow](public_dropin_environments/python3_keras)
* [Python 3 + pmml](public_dropin_environments/python3_pmml)
* [R + caret](public_dropin_environments/r_lang)
* [Java Scoring Code](public_dropin_environments/java_codegen)
* [Julia + MLJ](public_dropin_environments/julia_mlj)

These sample environments each define the libraries available in the environment 
and are designed to allow for simple custom tasks and/or custom inference models to be made that 
consist solely of your model's artifacts and an optional custom code file, if necessary.

For detailed information on how to create models that work in these environments, 
reference the links above for each environment.

