# Python 3 PMML Drop-In Template Environment

This template environment can be used to create artifact-only PMML custom models.
Your custom model directory needs only contain your model artifact if you use the
environment correctly.

## Supported Libraries

This environment has built for Python 3 and has support for the following libraries:
For specific version information, see [requirements](requirements.txt).

- pypmml
- numpy
- pandas
- scikit-learn

## Instructions

1. From the terminal, run `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/python3_pmml/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created
in step 1.

### Creating models for this environment

To use this environment, your custom model archive must contain a single serialized model artifact
with `.pmml` file extension as well as any other custom code needed to use your serialized model.


This environment makes the following assumption about your serialized model:
- The data sent to custom model can be used to make predictions without
additional pre-processing
- Regression models return a single floating point per row of prediction data
- Binary classification models return one floating point value <= 1.0 or two floating point values that sum to 1.0 per row of prediction data.
  - Single value output is assumed to be the positive class probability
  - Multi value it is assumed that the first value is the negative class probability, the second is the positive class probability
- There is a single `pmml` file present

If these assumptions are incorrect for your model, you should make a copy of [custom.py](https://github.com/datarobot/datarobot-user-models/blob/master/model_templates/python3_pmml/custom.py), modify it as needed, and include in your custom model archive.

The structure of your custom model archive should look like:

- custom_model.tar.gz
  - artifact.pmml
  - custom.py (if needed)

Please read [datarobot-cmrunner](../../custom_model_runner/README.md) documentation on how to assemble **custom.py**.
