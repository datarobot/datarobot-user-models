"""
A class to manage custom metrics object on the DR side

"""
from __future__ import absolute_import

from schema import Schema, SchemaError, Or, Optional
import yaml
import json
from datetime import datetime
import logging
import datarobot as dr
from datetime import datetime
from datetime import timezone
import requests
from requests.exceptions import ConnectionError
from requests.exceptions import HTTPError
from urllib3.exceptions import MaxRetryError

from datarobot.utils.waiters import wait_for_async_resolution


"""
customMetrics:
  - name: Model Cost in USD
    nickName: cost
    description: Blabla
    aggregator: sum
    y-axis: bla
    defaultInterval: Hour
    baseline:
    metricDirection: higher
    isModelSpecific: no
"""


class DRCustomMetrics:

    allowed_type_values = ["sum", "average", "gauge"]
    allowed_directionality = ["higherIsBetter", "lowerIsBetter"]
    config_schema = Schema({
        "customMetrics": [
            {
                "name": str,
                "nickName": str,
                "description": str,
                "isModelSpecific": bool,
                "directionality": Or(*allowed_directionality),
                "type": Or(*allowed_type_values),
                "units": str,
                "timeStep": str,
                Optional("baselineValue"): Or(int, float)
            }
        ]
    })

    def __init__(self, dr_client=None,
                 dr_url=None,
                 dr_api_key=None,
                 deployment_id=None,
                 model_package_id=None):
        """
        :param dr_url: DataRobot app url
        :param dr_api_key: API Key to access public API
        :param deployment_id: Deployment ID to report custom metrics for
        :param model_package_id: Model package id is required in case of reporting model specific
           metrics
        """
        self._logger = logging.getLogger(__name__)

        if dr_client:
            self._dr_client = dr_client
        elif dr_url and dr_api_key:
            self._dr_url = dr_url
            self._dr_api_key = dr_api_key,
            self._dr_client = dr.Client(token=self._dr_api_key, endpoint=self._dr_url + "/api/v2")

        if not deployment_id:
            raise Exception("Must provide deployment id")
        self._deployment_id = deployment_id

        self._model_package_id = model_package_id
        self._metrics_config = None
        self._metric_by_name = None

    def set_config_file(self, file_path):
        """
        Read a file JSON/YAML with definitions of required custom metrics
        :param file_path:
        :return:
        """

        #TODO Read file and call set_config
        return self

    def create_custom_segments(self, names, max_wait=600):
        url = f"deployments/{self._deployment_id}/settings/"
        payload = {
            "segment_analysis": {
                "enabled": True,
                "custom_attributes": names,
            }
        }
        response = self._dr_client.patch(url, data=payload)
        wait_for_async_resolution(self._dr_client, response.headers["Location"], max_wait)

    @staticmethod
    def _has_unique_values(input_list, key):
        seen_values = set()

        for item in input_list:
            value = item.get(key)
            if value in seen_values:
                return value
            seen_values.add(value)

        return None

    def set_config(self, config_dict=None, config_yaml=None):
        """
        Get definition of custom metrics from a config dict
        :param config_yaml:
        :param config_dict:
        :return:
        """

        # Read the config in multiple formats
        if config_yaml:
            parsed_dict = yaml.safe_load(config_yaml)
            self._logger.debug(parsed_dict)
            self._metrics_config = parsed_dict
        elif config_dict:
            self._metrics_config = config_dict
        else:
            raise Exception("only YAML or dict are supported for now")

        try:
            self.config_schema.validate(self._metrics_config)
            self._logger.debug("Configuration is valid.")
        except SchemaError as se:
            raise se

        # Removing the section and pointing directly to the list
        self._metrics_config = self._metrics_config["customMetrics"]

        # Validating that name and nickNames are unique
        non_unique = self._has_unique_values(self._metrics_config, "name")
        if non_unique:
            raise Exception(f"Found a non unique name field in customMetrics list {non_unique}")
        non_unique = self._has_unique_values(self._metrics_config, "nickName")
        if non_unique:
            raise Exception(f"Found non unique nickName field in customMetrics list {non_unique}")

        return self

    def get_list_of_dr_custom_metrics(self):
        url = f"deployments/{self._deployment_id}/customMetrics/"
        self._logger.debug(f"Getting cm list: {url}")
        res = self._dr_client.get(url)
        cm_dict = res.json()
        self._logger.info(cm_dict)
        if cm_dict["count"] != cm_dict["totalCount"]:
            raise Exception("Too many custom metrics in this deployment - not supported")

        cm_list = cm_dict["data"]
        return cm_list

    def _has_config(self):
        if self._metrics_config is None:
            raise Exception("Must provide custom metrics configuration first")

    def nick2id(self, nick):
        """
        Translate nickname to a metric id
        :param nick:
        :return:
        """
        self._has_config()
        if nick in self._metric_by_name:
            return self._metric_by_name[nick]["id"]
        return None

    def _build_unified_list_of_cm(self, dr_cm_list):
        self._logger.debug(f"dr_cm_list, {dr_cm_list}")
        self._logger.debug(f"metrics_config: {self._metrics_config}")

        dr_by_name = {item["name"]: item for item in dr_cm_list}
        self._logger.debug(dr_by_name)

        # Comparing the lists
        missing_in_dr = 0
        for local_metric in self._metrics_config:
            self._logger.debug("Checking local metric: {}".format(local_metric["name"]))
            name = local_metric["name"]
            if name in dr_by_name:
                self._logger.debug("Found metric in dr")
                local_metric["id"] = dr_by_name[name]["id"]
            else:
                self._logger.debug("Metric is NOT in dr")
                missing_in_dr += 1
        self._logger.debug(f"Done comparison: missing_in_dr {missing_in_dr}")
        self._logger.debug(self._metrics_config)

        metric_by_name = {item["name"]: item for item in self._metrics_config}
        self._logger.debug("Metric by Name")
        self._logger.debug(metric_by_name)
        # create the missing metrics

        for name in metric_by_name:
            if "id" not in metric_by_name[name]:
                self._logger.debug(f"Creating metric {name} in dr")

        for name in dr_by_name:
            if name not in metric_by_name:
                self._logger.debug(f"Metric from dr {name} is not in local - adding")
                metric_by_name[name] = {"id": dr_by_name[name]["id"]}

        self._metric_by_name = metric_by_name
        self._logger.debug(self._metric_by_name)

    def _create_missing_dr_cm(self):
        self._logger.debug("Creating missing custom metrics in DR")
        metrics_created = 0
        for metric in self._metrics_config:
            if "id" not in metric:
                self._logger.debug("Creating {} in dr".format(metric["name"]))
                # description: Blabla
                # type: sum
                # timeStep: Hour
                # directionality: higherIsBetter
                # isModelSpecific: no
                metric_id = self.create_cm(name=metric["name"],
                                    directionality=metric["directionality"],
                                    aggregation_type=metric["type"],
                                    time_step=metric["timeStep"],
                                    units=metric["units"],
                                    baseline_value=metric.get("baselineValue"),
                                    is_model_specific=metric["isModelSpecific"])
                metric["id"] = metric_id
                metrics_created +=1
        self._logger.debug(f"Created {metrics_created} metrics in DR")

    def _validate_dr_cm(self, dr_cm_list):
        self._logger.debug("Validating DataRobot custom metrics")
        non_unique = self._has_unique_values(dr_cm_list, "name")
        if non_unique:
            raise Exception(f"Found non unique custom metric name {non_unique} on DataRobot side")

    def sync(self):
        """
        Sync DR deployment custom metrics from a definition of custom metrics
        :param cm_dict:
        :return:
        """
        self._has_config()
        dr_cm_list = self.get_list_of_dr_custom_metrics()
        self._validate_dr_cm(dr_cm_list)
        self._build_unified_list_of_cm(dr_cm_list)
        self._create_missing_dr_cm()

    def report_value(self, nick_name, value, segments=None):
        """
        Report a value for a custom metric given the nick name. Avoid using the id
        :param nick_name:
        :param value:
        :return:
        """
        self._has_config()

        metric_id = self.nick2id(nick_name)
        if metric_id is None:
            raise Exception(f"Failed translating nick name {nick_name} to metric id")

        api_url = 'deployments/{}/customMetrics/{}/fromJSON/'
        ts = datetime.utcnow()
        segments = segments or []

        rows = [{'timestamp': ts.isoformat(), 'value': value}]

        json_payload = {'buckets': rows}
        if self._model_package_id:
            json_payload["modelPackageId"] = self._model_package_id

        if segments:
            json_payload["segments"] = [{"name": n, "value": v} for (n, v) in segments]

        self._logger.debug(json_payload)
        response = self._dr_client.post(
            api_url.format(self._deployment_id, metric_id),
            json=json_payload,
        )
        print(response.content)
        response.raise_for_status()

    def create_cm(self, name, directionality, units, aggregation_type,
                  is_model_specific, time_step="hour", baseline_value=None):

        # optional parameters, used in ingestion from dataset
        timestamp = {"columnName": None, "timeFormat": None}
        value = {"columnName": None}
        sample_count = {"columnName": None}
        batch = {"columnName": None}

        url = "deployments/{}/customMetrics/".format(self._deployment_id)
        payload = {
            "name": name,
            "directionality": directionality,
            "units": units,
            "type": aggregation_type,
            "timeStep": time_step,
            "isModelSpecific": is_model_specific,
            "timestamp": timestamp,
            "value": value,
            "sampleCount": sample_count,
            "batch": batch,
        }
        baselines = []
        if baseline_value is not None:
            baselines.append({"value": baseline_value})

        payload['baselineValues'] = baselines
        response = self._dr_client.post(url, json=payload)
        if response.status_code != 201:
            raise Exception("Error creating custom metric {}")
        custom_metric_id = response.json()["id"]
        self._logger.debug(f"created metric {name} with id: {custom_metric_id}")
        return custom_metric_id
