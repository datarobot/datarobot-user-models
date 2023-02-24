# DataRobot Proxy Model Example

This model is intended to work with any Python-based drop-in environment. Any additional
dependencies needed to connect to the remote model are listed in the `requirements.txt`
file.

This sample proxy model illustrates how you can use custom model infrastructure as a proxy between two DataRobot clusters or deployments in the same cluster.

## Instructions

First, you will need to have an existing deployment in the DataRobot MLOps platform.

Next, create a new _Custom Inference Model_ in the DataRobot platform of type `Proxy` and select
the `Target type` and `Target name` to match the model trained in AzureML.

Finally, upload the files in this example in the Assemble tab and select public network access in the
resource settings. In addition, there will be several runtime parameters that will need to be
filled in with the appropriate information.

## Run locally with `drum`

Create a _values file_ to set overrides for the parameters defined in this
example. An example could look as follows:

```yaml
# Enter in the base URL of your DataRobot cluster if it is not the default
#DATAROBOT_ENDPOINT: https://app.datarobot.com

# Input the ID of the deployment you want to proxy to
deploymentID: 63f6dc613f1d2368177b9659

# This is the API key you can get from the `Developer Tools` tab in your user profile. We
# will structure the data the same as the DataRobot Platform will when you associate
# the runtime parameter with a credential stored in the Credential Manager.
DATAROBOT_API_KEY:
  credentialType: basic
  username: robot
  password: NjNmNmRjODRkYjk5OGZhZjNhODY2NGUwOkhpamlWTGRrbUNBRUVhaFhFRlNQb1dhU3FxQ0U3a3pKMGR0S3h6
```

Paths are relative to `./datarobot-user-models`:

```sh
drum score --logging-level info --code-dir model_templates/proxy_model_datarobot --target-type <target_type> --input <path_to_inference_dataset> --runtime-params-file <path_to_values_file>
```

> **Note:** Additional CLI flags may be needed depending on your model's target type.
