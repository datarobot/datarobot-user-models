# AzureML Proxy Model Example

This model is intended to work with any Python-based drop-in environment. Any additional
dependencies needed to connect to the remote model are listed in the `requirements.txt`
file.

This sample proxy model illustrates how you can use custom model infrastructure as a proxy between the DataRobot MLOps platform and a remote model hosted in AzureML.

> :warning: Proxy Models is currently a _Beta Feature_ that will need to be enabled by your administrator.

## Instructions

First, you create a new _online endpoint_ in AzureML from a trained model. You should
compare the Python code snippet in the AzureML UI with the code in the `custom.py` file to
confirm no adjustments are necessary (this example was tested with an AutoML
model trained on a simple regression dataset: [juniors_3_year_stats_regression.csv](../../tests/testdata/juniors_3_year_stats_regression.csv)).

DataRobot recommends configuring the endpoint with _Key-based authentication_ because it doesn't expire.
Create a new, **Basic** credential in the DataRobot platform, enter any value for the **Username**, and enter one of the endpoint's authentication keys in the **Password** field.

Next, create a new _Custom Inference Model_ in the DataRobot platform. Select the **Proxy** model type and
enter a **Target type** and **Target name** matching the model trained in AzureML.

Finally, on the **Assemble** tab, upload the files in this example and select public network access in the
**Resource Settings**. After the example files are uploaded, you can configure the **Runtime Parameters**
with the appropriate information.

## Run locally with `drum`

Create a _values file_ to set overrides for the parameters defined in this
example. An example could look as follows:

```yaml
# This is the name of your endpoint
endpoint: demo-endpoint

# Override the default value if your endpoint does not reside in the `eastus` region
#region: eastus

# This is the API key you can get from the `Consume` tab in the Endpoint's UI. We
# will structure the data the same as the DataRobot Platform will when you associate
# the runtime parameter with a credential stored in the Credential Manager.
API_KEY:
  credentialType: basic
  username: robot
  password: 4TZ0EdUVP8uoUXbx04Ve7yn44TiMl485
```

Paths are relative to `./datarobot-user-models`:

```sh
drum score --logging-level info --code-dir model_templates/proxy_model_azure --target-type <target_type> --input <path_to_inference_dataset> --runtime-params-file <path_to_values_file>
```

> **Note**: Additional CLI flags may be required depending on your model's target type.
