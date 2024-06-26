# Python Sklearn Inference Model with Runtime Parameters (example)

This model is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../public_dropin_environments/python3_sklearn/).
The supplied pkl file is a scikit-learn model trained on [juniors_3_year_stats_regression.csv](../../tests/testdata/juniors_3_year_stats_regression.csv)
with a `Grade 2014` as the target (regression), though any binary or regression model trained using the libraries
outlined in [Python 3 Scikit-Learn Drop-In Environment](../../public_dropin_environments/python3_sklearn) will work.

In this example, the `custom.py` file contains a dummy `transform()` function that demonstrates how
to access runtime parameter values set in a _Custom Inference Model_ in the DataRobot platform.

> :warning: Runtime Parameters is currently a _Beta Feature_ that will need to be enabled by your administrator.

## Instructions

This example contains a `model-metadata.yaml` file with a `runtimeParameterDefinitions` section where you must declare the runtime parameters.
The `custom.py` script uses the helper functions from the `datarobot_drum.RuntimeParameters` class, available in `datarobot-drum` package version 1.10 and newer (included in the
latest pre-built drop-in environments).

Create a new custom model with these files and the Python Drop-In Environment. Ensure that you upload the `model-metadata.yaml` file to the top-level of the model's file structure. After you add the
metadata file, DataRobot reads the parameter definitions and updates the **Assembly** tab
UI to display the default values, allowing you to override them and provide any required credentials.

## Run locally with `drum`

Create a _values file_ to set overrides for the parameters defined in this
example. An example could look as follows:

```yaml
# option1 will use its default value (ABCD 123) if you don't override it below
#option1: Hello World

# set a value for option2
option2: goodbye!

# option3 has no default and if we don't override it below, it will simply have an implicit
# default value of `None`.
#option3: NULL

# Credential type params are stored as a mapping. The `credentialType` key will always be present
# and the value will inform what specific credential type is being injected. The other key/values
# vary with credential type:
#   https://docs.datarobot.com/en/docs/api/reference/public-api/credentials.html#properties_3
encryption_key:
  credentialType: rsa
  rsaPrivateKey: xfzRmbBoIT/89/QdACH0f8PI6Idq55JOnkzy9nkMXu2uyt3VHZWm6krbLtInBvc+
```

Paths are relative to `./datarobot-user-models`:

```sh
drum score --code-dir model_templates/python3_sklearn_runtime_params --target-type regression --input tests/testdata/juniors_3_year_stats_regression.csv --runtime-params-file <path_to_values_file>
```
