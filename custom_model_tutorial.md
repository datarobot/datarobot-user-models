Building, Testing, and Deploying a Custom Model
===============================================

This notebook walks through the general workflow for building, testing, and
deploying a custom inference model on a custom environment. 

Note
----

To finish this tutorial, you must have access to either Cloud DataRobot or
On-Site Deploy of DataRobot.

This tutorial is using Cloud DataRobot (app.datarobot.com).

Agenda
------

In this tutorial, we'll learn:

1. How to use the client to create an environment
2. How to check the status of an environment build
3. How to create a custom model
4. How to iteratively test and debug a custom model on a custom environment
5. How to deploy and run predictions on a tested custom model.


Setup and Requirements
----------------------

This tutorial assumes a few things about your filepath and prior work. 


**Firstly, you need a feature flag enabled:**

- Enable MLOps

Secondly, you should have a folder at the path `~/datarobot-user-models/`. If
you put the folder in a different location, make sure you update the
`TESTING_PATH` variable. This folder should contain 4 things:

1. A folder containing your properly configured custom environment.     
  In this example, it's named `public_dropin_environments/python3_pytorch/`

2. A folder containing your properly-configured custom model.     
  In this example, it's named `model_templates/python3_pytorch/`

3. The current version of the DataRobot Python Client.
  - Installation instructions for the client can be found here: [DataRobot Python Client Docs](https://datarobot-public-api-client.readthedocs-hosted.com/en/v2.20.0/setup/getting_started.html#installation)
  - Full documentation for the client can be found here: [DataRobot Python Client Docs](https://datarobot-public-api-client.readthedocs-hosted.com/en/v2.20.0/index.html)

4. A test dataset that you can use to test predictions from your custom model.
    In this example, it's stored at `tests/testdata/juniors_3_year_stats_regression.csv`

It also assumes that you have access to app.datarobot.com.

If you use another version of DataRobot - use appropriate credentials and URL.


Configuring Models and Environments
-----------------------------------

For more information on how to properly configure custom models and
environments, read the README of our [DataRobot User Models
repository](https://github.com/datarobot/datarobot-user-models).

Imports
-------

First, we need to make the proper imports. Make sure the `TESTING_PATH` is correct and pointing to the right folder:

```
%load_ext autoreload
%autoreload 2
import sys
import os
import requests
from pprint import pprint
```

This is where you save the `TESTING_PATH` that contains the relevant folders.

```
# Save the path to the custom model testing folder, and add it to the PYTHONPATH so we can import the client
TESTING_PATH = os.getcwd() + '/'
sys.path.append(TESTING_PATH)

import datarobot as dr
```

Configuring User Credentials
----------------------------

Make sure to fill in your username and API token from app.datarobot.com.

Also ensure that all the paths are correct!

```
## Save user credentials ##
TOKEN = ''
USERNAME = ''
DATAROBOT_KEY = ''  # required to make predictions against deployments

## Save path to environment ##
environment_folder = TESTING_PATH + 'public_dropin_environments/python3_pytorch/'

## Save path to custom model ##
custom_model_folder = TESTING_PATH + 'model_templates/python3_pytorch/'

## Save test dataset path ##
test_dataset = TESTING_PATH + 'tests/testdata/juniors_3_year_stats_regression.csv'
```

Loading the API client
----------------------

This command initializes the API client. **You shouldn't need to change
anything in this block if you configured your credentials properly!**

```
# Configure client
client = dr.Client(
    endpoint='https://app.datarobot.com/api/v2',
    token=TOKEN,
)
```

Creating a Custom Environment
-----------------------------

This command creates a custom environment! When you run the command, it uploads
your Docker context and we attempt to build the Docker Image (the container
that your model will eventually run in). 

Depending on the environment and the libraries you want to download, this
process can take a while (10-30 minutes)! This command sets the wait time to 1
hour, but if it fails with a AsyncTimeoutError, it's possible that the
environment is still processing and could still succeed.



### Custom Environment Templates

Custom environment templates can be found here: [environment
templates](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments)


You'll find templates for Python 3, Java and R environments.

```
## Create the environment, which will eventually contain versions  ##
execution_environment = dr.ExecutionEnvironment.create(
    name="Python3 PyTorch Environment",
    description="This environment contains Python3 pytorch library.",
)

## Create the environment version ##
environment_version = dr.ExecutionEnvironmentVersion.create(
    execution_environment.id,
    environment_folder,
    max_wait=3600,  # 1 hour timeout
)
```


Creating a Custom Model
-----------------------

Once the Custom Environment is successfully built, now it's time to build the
Custom Model. You will need to define details about your custom model in this
command, depending on the type of model.


### Required fields:

* `model_path` : string containing the path to the model folder
* `name` : string that defines the name of the model
* `target_name` : string that defines the name of the target column that the model was trained on
* `target_type` : boolean that describes the target type. Supported target types are "Binary" (`datarobot.TARGET_TYPE.BINARY`) and "Regression" (`datarobot.TARGET_TYPE.REGRESSION`).
* `positive_class_label` : string that defines the "positive class". Only required for Binary Classification models
* `negative_class_label` : string that defines the "negative class". Only required for Binary Classification models


### Optional Fields:

`prediction_threshold` : a float that defines the prediction threshold for binary classification. This value is used for features and charts in MMM.
`description` : a string that describe the model. User can input whatever they want for the description.
`language` : a string that details the language the model uses. User can input whatever they want for the language.

```
## Create the custom model ##
custom_model = dr.CustomInferenceModel.create(
    name='Python 3 PyTorch Custom Model',
    target_type=dr.TARGET_TYPE.REGRESSION,
    target_name='Grade 2014',
    description='This is a Python3-based custom model. It has a simple PyTorch model built on juniors_3_year_stats_regression dataset',
    language='python'
)

## Create the custom model version ##
model_version = dr.CustomModelVersion.create_clean(
    custom_model_id=custom_model.id,
    folder_path=custom_model_folder,
    base_environment_id=execution_environment.id
)
```

The Model Testing Workflow
--------------------------

Just because you created an environment and a model doesn't mean that it will
actually work in production! There are all sorts of things that can go wrong,
whether on the engineering side or the data science side. Bad code, an
environment with the wrong versions of libraries, or even a model that can't
handle missing values in the inference data can all lead to a model that will
break in production.

With this in mind, we created an easy way to ensure that a custom inference
model will work in production: You can actually test your model with a specific
environment using sample inference data before deploying the model. 

Model Testing
-------------

### Step 1: Run the Test

To run a custom model test, you upload and save a test dataset from the sample
inference data. Then, you simply select the appropriate model and environment
(as well as version) IDs, and test it on that dataset.

Depending on the k8s cluster and the model itself, it may take a few minutes to
test the model. Once the test is finished, it will have a status property to
let you know whether the test passed. If it failed, it will contain an `error`
property that contains the relevant error!



An important note: As of right now, the only available test is an error check,
where we simply ensure the model can return predictions. In the future, we will
add more tests to that suite: prediction consistency, missing value handling,
and more.

```
dataset = dr.Dataset.create_from_file(file_path=test_dataset)
```


```
# Perform custom model test
custom_model_test = dr.CustomModelTest.create(
    custom_model_id=custom_model.id, 
    custom_model_version_id=model_version.id,
    dataset_id=dataset.id,
    max_wait=3600,  # 1 hour timeout
)

print("Overall testing status: {}".format(custom_model_test.overall_status))

if any(test['status'] == 'failed' for test in custom_model_test.detailed_status.values()):
    print('Test log:\n')
    print(custom_model_test.get_log())
```

### Step 2: Iterate

If the test passed, then congratulations! You can skip this test; your model is
ready to be deployed. If it failed the test however, it's easy to iterate. 

First, check the error from the custom model test. Then, fix any errors in the
code that you uploaded. Finally, upload a new version of the model using the
updated code, and test it again!


```
# Add new version of custom model. Repeat these last two blocks until the model passes testing!
model_version = dr.CustomModelVersion.create_clean(
    custom_model_id=custom_model.id,
    folder_path=custom_model_folder,
    base_environment_id=execution_environment.id
)

model_version.update(description='Fixing errors from testing')
# Perform custom model test... again

custom_model_test = dr.CustomModelTest.create(
    custom_model_id=custom_model.id, 
    custom_model_version_id=model_version.id,
    dataset_id=dataset.id,
    max_wait=3600,  # 1 hour timeout
)

print("Overall testing status: {}".format(custom_model_test.overall_status))

if any(test['status'] == 'failed' for test in custom_model_test.detailed_status.values()):
    print('Test log:\n')
    print(custom_model_test.get_log())
```

```
# This command shows all tests that have been run on the model
model_tests = dr.CustomModelTest.list(custom_model_id=custom_model.id)
print(model_tests)
```

## Deploying the model

To deploy an inference model, you create something called a
`custom_model_image`, which saves the custom model code with a _specific_
environment. This will make it easy to see which custom models have been tested
or deployed on specific environments.



Once you have the desired custom model image, simply call the
`dr.Deployment.create_from_custom_model_image()` method, inputting the model
image's id, the prediction server's `default_prediction_server_id`, and the
desired deployment label.

```
# Ensure that the client is using the correct prediction server to deploy the model. 
# This uses the prediction server for testing on Cloud DataRobot.

available_prediction_server_urls = [
    "https://datarobot-predictions.orm.datarobot.com",
]


prediction_server = None

for pred_server in dr.PredictionServer.list():
    if pred_server.url in available_prediction_server_urls:
        prediction_server = pred_server
        break
else:
    raise Exception("no suitable prediction server found")
```

```
deployment = dr.Deployment.create_from_custom_model_version(
    model_version.id,
    label='Test client deployment',
    # instance id is only required for Cloud DataRobot App
    # ignore for on-premises Platform installations.
    default_prediction_server_id=prediction_server.id,
    max_wait=3600,  # 1 hour timeout
)
```

### Making predictions on a deployed custom inference model

Predictions look exactly the same for a custom inference model and a native DR
model. If training data was assigned to the model, then we can also provide
predictions explanations and all MMM features, deeply integrated with the
custom model.

```
# Make predictions on the custom model deployment

url = '{}/predApi/v1.0/deployments/{}/predictions'.format(prediction_server.url, deployment.id)

import pandas as pd

predictions_dataset = pd.read_csv(test_dataset)
predictions_data = predictions_dataset.to_json(orient='records')

headers = dr.client.get_client().headers
headers['datarobot-key'] = DATAROBOT_KEY
headers['Content-Type'] = 'application/json'

response = requests.post(url, headers=headers, data=predictions_data)

predictions = response.json()
pprint(predictions)
```
