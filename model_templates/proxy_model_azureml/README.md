# AzureML Proxy Model Example

This model is intended to work with any Python based drop-in environment. Any additional
dependencies needed to connect to the remote model are listed in the `requirements.txt`
file.

This sample proxy model shows how the custom model infrastructure can be used to connect (e.g. proxy) data back and forth between the DataRobot MLOps platform and a remote model hosted
in AzureML.

## Instructions

First, you will need to create a new **online endpoint** in AzureML from a trained model. You should
compare the Python code snippet in the AzureML UI with the code in the `custom.py` file to
confirm no adjustments are necessary (this example was specifically tested against an AutoML
model trained on a simple regression dataset: [juniors_3_year_stats_regression.csv](../../tests/testdata/juniors_3_year_stats_regression.csv)).

We recommend configuring the endpoint with _Key-based authentication_ because it doesn't expire.
Please create a new credential in the DataRobot platform of type `Basic` and enter any value for the `Username` and input one of the authentication keys of the endpoint into the `Password` field.

Next, create a new _Custom Inference Model_ in the DataRobot platform of type `Proxy` and select
the `Target type` and `Target name` to match the model trained in AzureML.

Finally, upload the files in this example in the Assemble tab and select public network access in the
resource settings. In addition, there will be several runtime parameters that will need to be
filled in with the appropriate information.

## To run locally using 'drum'

You'll need to create a _values file_ that sets overrides for the parameters defined in this
example. An example could look as follows:

```yaml
# This is the name of your endpoint
endpoint: demo-endpoint

# Override the default value if your endpoint does not reside in the `eastus` region
#region: eastus

API_KEY:
  credentialType: basic
  username: robot
  password: 4TZ0EdUVP8uoUXbx04Ve7yn44TiMl485
```

Paths are relative to `./datarobot-user-models`:

```sh
drum score --logging-level info --code-dir model_templates/proxy_model_azure --target-type <target_type> --input <path_to_inference_dataset> --runtime-params-file <path_to_values_file>
```

_**Note:** additional CLI flags may be needed depending on your model's target type_
