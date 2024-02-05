"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from typing import Any, Dict

import os
import pandas as pd

import datarobot as dr
from dr_custom_metrics import DRCustomMetricInfo, DRCustomMetric


def get_container_memory_usage():
    try:
        # Read the memory usage from the cgroup file
        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as file:
            usage_in_bytes = int(file.read().strip())

        # Convert the usage to megabytes for better readability
        usage_in_megabytes = usage_in_bytes / (1024 ** 2)

        return usage_in_megabytes
    except FileNotFoundError:
        print("Error: Could not find the memory usage file.")
        return None


def get_datarobot_endpoint():
    value = os.environ['DATAROBOT_ENDPOINT']
    if value.endswith("/api/v2"):
        value = value[:-len("/api/v2")]
    return value + "/api/v2"


def create_if_not_exists_used_mem_metric(dr_client, deployment_id, baseline, name="used_mem"):
    used_mem_metric = DRCustomMetric(dr_client=dr_client, deployment_id=deployment_id)
    metric_config_yaml = f"""
       customMetrics:
         - name: {name}
           nickName: {name}
           description: Used memory out of total
           type: gauge
           timeStep: hour
           units: MB
           directionality: lowerIsBetter
           isModelSpecific: no
           baselineValue: {baseline}
       """
    used_mem_metric.set_config(config_yaml=metric_config_yaml)
    used_mem_metric.sync()
    return used_mem_metric


def load_model(code_dir: str) -> Any:
    model = {}

    used_mem = get_container_memory_usage()
    print("\n\nInitial memory usage (MBs):")
    print(used_mem)

    deployment_id = os.environ.get('MLOPS_DEPLOYMENT_ID')

    if deployment_id:
        # create a DR client
        api_token = os.environ.get('DATAROBOT_API_TOKEN')
        dr_client = dr.Client(endpoint=get_datarobot_endpoint(), token=api_token)

        # create a metric to track memory usage
        # name: "used_mem".
        # In order to see it, you can check "Custom Metrics" section of your deployment
        used_mem_metric = create_if_not_exists_used_mem_metric(dr_client, deployment_id, used_mem)
        model["used_mem_metric"] = used_mem_metric

        # print metric details in logs
        cm_list = used_mem_metric.get_list_of_dr_custom_metrics()
        print('Custom metrics created:')
        print(cm_list)

    return model


def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    This hook is only needed if you would like to use **drum** with a framework not natively
    supported by the tool.

    Note: While best practice is to include the score hook, if the score hook is not present
    DataRobot will add a score hook and call the default predict method for the library
    See https://github.com/datarobot/datarobot-user-models#built-in-model-support for details

    This dummy implementation returns a dataframe with all rows having value 42 in the
    "Predictions" column, regardless of the provided input dataset.

    Parameters
    ----------
    data : is the dataframe to make predictions against. If `transform` is supplied,
    `data` will be the transformed data.
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
    kwargs : additional keyword arguments to the method

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Regression: must have a single column called `Predictions` with numerical values
    """
    preds = pd.DataFrame([42 for _ in range(data.shape[0])], columns=["Predictions"])

    # report memory usage
    if "used_mem_metric" in model:
        used_mem = get_container_memory_usage()
        model["used_mem_metric"].report_value("used_mem", used_mem)

    return preds
