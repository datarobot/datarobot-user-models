"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import json
import logging
import ssl
import urllib.request
from types import SimpleNamespace

import pandas as pd

from datarobot_drum import RuntimeParameters

logger = logging.getLogger(__name__)


def _test_connectivity(endpoint, region, api_key):
    # The root path of the endpoint can be used for health checks
    url = f"https://{endpoint}.{region}.inference.ml.azure.com/"
    logger.info("Checking liveness of endpoint: %s", url)

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as error:
        logger.error(
            "Failed to connect to %s status_code=%s\n%s",
            endpoint,
            error.code,
            error.read().decode("utf8", "ignore"),
        )
        raise


# We override the load_model hook because there is no pickle file to load. Instead
# we use this time to load the runtime params and test that we have connectivity
# to the remote model.
def load_model(code_dir):
    logger.info("Loading Runtime Parameters...")
    api_key = RuntimeParameters.get("API_KEY")["password"]

    endpoint = RuntimeParameters.get("endpoint")
    region = RuntimeParameters.get("region")
    deployment = RuntimeParameters.get("deployment")
    url = f"https://{endpoint}.{region}.inference.ml.azure.com/score"
    verify_ssl = RuntimeParameters.get("verifySSL").lower() == "true"

    if verify_ssl:
        allowSelfSignedHttps()
    _test_connectivity(endpoint, region, api_key)

    # Can return any object as a placeholder for a model that we can
    # then use again in the `score()` function.
    return SimpleNamespace(**locals())


def score(data, model, **kwargs):
    # The default format that an endpoint for an Azure AutoML model expects
    # is of the form:
    #   {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
    #    'data': [[1, 0.5], [2, 0.75]]}
    payload = {"input_data": data.to_dict(orient="split")}
    response = make_remote_prediction_request(
        json.dumps(payload).encode("utf-8"),
        model.url,
        model.api_key,
        deployment=model.deployment,
    )

    # convert the prediction request response to the required data structure
    predictions_data = pd.DataFrame({"Predictions": response})
    return predictions_data


### The code below was adapted from the snippet provided in the AzureML UI
def allowSelfSignedHttps():
    ssl._create_default_https_context = ssl._create_unverified_context


def make_remote_prediction_request(payload, url, api_key, deployment=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if deployment:
        # The azureml-model-deployment header will force the request to go to a specific deployment.
        headers["azureml-model-deployment"] = deployment

    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError as error:
        logger.error(
            "The request failed with status_code=%s; headers=%s;\n%s",
            error.code,
            error.info(),
            error.read().decode("utf8", "ignore"),
        )
        raise

    try:
        return json.load(response)
    except json.JSONDecodeError as error:
        logger.error("Response from server was not JSON: %s", error)
        raise
