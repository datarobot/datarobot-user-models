#
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc. Confidential.
#
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
#
# The copyright notice above does not evidence any actual or intended
# publication of such source code.
import json
import logging
import math
import os
import traceback
from datetime import datetime

import datarobot as dr
import numpy as np
import pytz
import requests
import yaml
from datarobot.errors import ClientError
from dmm.custom_metric import CustomMetricAggregationType
from dmm.custom_metric import CustomMetricDirectionality
from dmm.datarobot_api_client import DataRobotApiClient

from datarobot_dome.async_http_client import AsyncHTTPClient
from datarobot_dome.constants import CUSTOM_METRIC_DESCRIPTION_SUFFIX
from datarobot_dome.constants import DEFAULT_PROMPT_COLUMN_NAME
from datarobot_dome.constants import LOGGER_NAME_PREFIX
from datarobot_dome.constants import GuardAction
from datarobot_dome.constants import GuardOperatorType
from datarobot_dome.constants import GuardStage
from datarobot_dome.constants import OOTBType
from datarobot_dome.guard import GuardFactory
from datarobot_dome.guard import ModelGuard
from datarobot_dome.guard import NeMoGuard
from datarobot_dome.guard import OOTBGuard
from datarobot_dome.guard import moderation_config_trafaret
from datarobot_dome.guard_helpers import get_rouge_1_scorer

CUSTOM_METRICS_BULK_UPLOAD_API_PREFIX = "deployments"
CUSTOM_METRICS_BULK_UPLOAD_API_SUFFIX = "customMetrics/bulkUpload/"


def get_stage_str(stage):
    return "Prompts" if stage == GuardStage.PROMPT else "Responses"


def get_blocked_custom_metric(stage):
    return {
        "name": f"Blocked {get_stage_str(stage)}",
        "directionality": CustomMetricDirectionality.LOWER_IS_BETTER,
        "units": "count",
        "type": CustomMetricAggregationType.SUM,
        "baselineValue": 0,
        "isModelSpecific": True,
        "timeStep": "hour",
        "description": (
            f"Number of blocked {get_stage_str(stage)}.  {CUSTOM_METRIC_DESCRIPTION_SUFFIX}"
        ),
    }


def get_total_custom_metric(stage):
    return {
        "name": f"Total {get_stage_str(stage)}",
        "directionality": CustomMetricDirectionality.HIGHER_IS_BETTER,
        "units": "count",
        "type": CustomMetricAggregationType.SUM,
        "baselineValue": 0,
        "isModelSpecific": True,
        "timeStep": "hour",
        "description": (
            f"Total Number of {get_stage_str(stage)}.  {CUSTOM_METRIC_DESCRIPTION_SUFFIX}"
        ),
    }


prescore_guard_latency_custom_metric = {
    "name": "Prescore Guard Latency",
    "directionality": CustomMetricDirectionality.LOWER_IS_BETTER,
    "units": "seconds",
    "type": CustomMetricAggregationType.AVERAGE,
    "baselineValue": 0,
    "isModelSpecific": True,
    "timeStep": "hour",
    "description": f"Latency to execute prescore guards.  {CUSTOM_METRIC_DESCRIPTION_SUFFIX}",
}

postscore_guard_latency_custom_metric = {
    "name": "Postscore Guard Latency",
    "directionality": CustomMetricDirectionality.LOWER_IS_BETTER,
    "units": "seconds",
    "type": CustomMetricAggregationType.AVERAGE,
    "baselineValue": 0,
    "isModelSpecific": True,
    "timeStep": "hour",
    "description": f"Latency to execute postscore guards.  {CUSTOM_METRIC_DESCRIPTION_SUFFIX}",
}

score_latency = {
    "name": "LLM Score Latency",
    "directionality": CustomMetricDirectionality.LOWER_IS_BETTER,
    "units": "seconds",
    "type": CustomMetricAggregationType.AVERAGE,
    "baselineValue": 0,
    "isModelSpecific": True,
    "timeStep": "hour",
    "description": f"Latency of actual LLM Score.  {CUSTOM_METRIC_DESCRIPTION_SUFFIX}",
}


class Pipeline:
    def __init__(self, guards_config_filename):
        self._logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + self.__class__.__name__)
        self._pre_score_guards = []
        self._post_score_guards = []
        self._custom_metric = {}
        self._dr_custom_metric_api_client = None
        self._deployment = None
        self._association_id_column_name = None
        self._prompt_column_name = None
        self._response_column_name = None
        self._datarobot_url = None
        self._datarobot_api_token = None
        self.dr_client = None
        self._headers = None
        self._deployment_id = None
        self._custom_model_dir = None
        self._model_id = None
        self._custom_metrics_bulk_upload_url = None
        self._assoc_id_specific_custom_metric_ids = list()
        self._aggregate_custom_metric = None
        self._custom_metric_map = dict()
        # List of custom metrics names which do not need the association id while reporting
        self._custom_metrics_no_association_ids = list()
        self._modifier_guard_seen = {stage: None for stage in GuardStage.ALL}
        self.async_http_client = None

        self._query_self_deployment()
        self._setup_custom_metrics()

        self.rouge_scorer = get_rouge_1_scorer()

        with open(guards_config_filename) as f:
            input_moderation_config = yaml.safe_load(f)

        moderation_config = moderation_config_trafaret.check(input_moderation_config)
        self._add_default_custom_metrics()
        for guard_config in moderation_config["guards"]:
            if isinstance(guard_config["stage"], list):
                for stage in guard_config["stage"]:
                    self._set_guard(guard_config, stage=stage)
            else:
                self._set_guard(guard_config)

        self.guard_timeout_sec = moderation_config["timeout_sec"]
        self.guard_timeout_action = moderation_config["timeout_action"]

        if self._datarobot_url and self._datarobot_api_token:
            self.async_http_client = AsyncHTTPClient(self.guard_timeout_sec)

        self._create_custom_metric_if_any()
        self._run_llm_in_parallel_with_pre_score_guards = False

    def _get_average_score_metric_definition(self, guard):
        metric_definition = guard.get_average_score_metric(guard.stage)
        if not guard.intervention:
            return metric_definition

        if guard.intervention.comparator not in [
            GuardOperatorType.GREATER_THAN,
            GuardOperatorType.LESS_THAN,
        ]:
            # For all other guard types, its not possible to define baseline value
            return metric_definition

        metric_definition["baselineValue"] = guard.intervention.threshold
        if guard.intervention.comparator == GuardOperatorType.GREATER_THAN:
            # if threshold is "greater", lower is better and vice-a-versa
            metric_definition["directionality"] = CustomMetricDirectionality.LOWER_IS_BETTER
        else:
            metric_definition["directionality"] = CustomMetricDirectionality.HIGHER_IS_BETTER

        return metric_definition

    def _set_guard(self, guard_config, stage=None):
        guard = GuardFactory().create(guard_config, stage=stage)

        guard_stage = stage if stage else guard.stage
        intervention_action = guard.get_intervention_action()

        if intervention_action == GuardAction.REPLACE:
            if self._modifier_guard_seen[guard_stage]:
                modifier_guard = self._modifier_guard_seen[guard_stage]
                raise ValueError(
                    "Cannot configure more than 1 modifier guards in the "
                    f"{guard_config['stage']} stage, "
                    f"guard {modifier_guard.name} already present"
                )
            else:
                self._modifier_guard_seen[guard_stage] = guard
        self._add_guard_to_pipeline(guard)
        guard.set_pipeline(self)

        if isinstance(guard, NeMoGuard):
            pass  # No average score metric for NeMo Guard
        elif isinstance(guard, ModelGuard) and guard.model_info.target_type == "Multiclass":
            pass  # No average score metric for Multiclass model guards
        else:
            self._custom_metric_map[guard.get_average_score_custom_metric_name(guard_stage)] = {
                "metric_definition": self._get_average_score_metric_definition(guard)
            }

        if isinstance(guard, OOTBGuard) and guard.ootb_type == OOTBType.TOKEN_COUNT:
            # No latency metric for Token count
            pass
        else:
            self._custom_metric_map[guard.get_latency_custom_metric_name()] = {
                "metric_definition": guard.get_latency_custom_metric()
            }

        if intervention_action:
            # Enforced metric for all kinds of guards, as long as they have intervention
            # action defined - even for token count
            self._custom_metric_map[
                guard.get_guard_enforced_custom_metric_name(guard_stage, intervention_action)
            ] = {
                "metric_definition": guard.get_enforced_custom_metric(
                    guard_stage, intervention_action
                )
            }
        self._custom_metrics_no_association_ids.append(guard.get_latency_custom_metric_name())

    def _add_default_custom_metrics(self):
        """Default custom metrics"""
        metric_list = [
            get_total_custom_metric(GuardStage.PROMPT),
            get_total_custom_metric(GuardStage.RESPONSE),
            prescore_guard_latency_custom_metric,
            postscore_guard_latency_custom_metric,
            score_latency,
        ]
        # Metric list so far does not need association id for reporting
        for metric in metric_list:
            self._custom_metrics_no_association_ids.append(metric["name"])

        metric_list.append(get_blocked_custom_metric(GuardStage.PROMPT))
        metric_list.append(get_blocked_custom_metric(GuardStage.RESPONSE))
        for metric in metric_list:
            self._custom_metric_map[metric["name"]] = {"metric_definition": metric}

    def _query_self_deployment(self):
        """
        Query the details of the deployment (LLM) that this pipeline is running
        moderations for
        :return:
        """
        common_message = "Custom Metrics and deployment settings will not be available"

        # This URL and Token is where the custom LLM model is running.
        self._datarobot_url = os.environ.get("DATAROBOT_ENDPOINT", None)
        if self._datarobot_url is None:
            self._logger.warning(f"Missing DataRobot endpoint, {common_message}")
            return

        self._datarobot_api_token = os.environ.get("DATAROBOT_API_TOKEN", None)
        if self._datarobot_api_token is None:
            self._logger.warning(f"Missing DataRobot API Token, {common_message}")
            return

        # This is regular / default DataRobot Client
        self.dr_client = dr.Client(endpoint=self._datarobot_url, token=self._datarobot_api_token)

        self._deployment_id = os.environ.get("MLOPS_DEPLOYMENT_ID", None)
        if self._deployment_id is None:
            self._logger.warning(f'Custom Model workshop "test" mode?, {common_message}')
            return

        # Get the model id from environ variable, because moderation lib cannot
        # query deployment each time there is scoring data.
        self._model_id = os.environ.get("MLOPS_MODEL_ID", None)
        self._logger.info(f"Model ID from env variable {self._model_id}")

        try:
            self._deployment = dr.Deployment.get(deployment_id=self._deployment_id)
            self._prompt_column_name = self._deployment.model.get("prompt")
            self._response_column_name = self._deployment.model["target_name"]
            self._logger.info(f"Model ID set on the deployment {self._deployment.model['id']}")
        except Exception as e:
            self._logger.warning(f"Couldn't query the deployment Exception: {e}, {common_message}")

        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._datarobot_api_token}",
        }

    def _query_association_id_column_name(self):
        self._logger.info(f"Deployment ID: {self._deployment_id}")
        if self._deployment is None:
            return

        self._logger.info(f"Check Association ID Column name: {self._association_id_column_name}")
        # Apparently, the pipeline.init() is called only once when deployment is created.
        # If the association id column name is not set (we cannot set it during creation
        # of the deployment), moderation library never gets it.  So, moderation library
        # is going to query it one time during prediction request
        if self._association_id_column_name:
            return

        try:
            association_id_settings = self._deployment.get_association_id_settings()
            self._logger.debug(f"Association id settings: {association_id_settings}")
            column_names = association_id_settings.get("column_names")
            if column_names and len(column_names) > 0:
                self._association_id_column_name = column_names[0]
        except Exception as e:
            self._logger.warning(
                f"Couldn't query the association id settings, "
                f"custom metrics will not be available {e}"
            )
            self._logger.error(traceback.format_exc())

        if self._association_id_column_name is None:
            self._logger.warning(
                "Association ID column is not set on the deployment, "
                "data quality analysis will not be available"
            )
        else:
            self._logger.info(f"Association ID column name: {self._association_id_column_name}")
            self._custom_metrics_bulk_upload_url = (
                CUSTOM_METRICS_BULK_UPLOAD_API_PREFIX
                + "/"
                + self._deployment_id
                + "/"
                + CUSTOM_METRICS_BULK_UPLOAD_API_SUFFIX
            )
            self._logger.info(f"URL: {self._custom_metrics_bulk_upload_url}")

    def _setup_custom_metrics(self):
        if self.dr_client is None or self._deployment_id is None:
            return

        self._dr_custom_metric_api_client = DataRobotApiClient(client=self.dr_client)

    def _create_custom_metric_if_any(self):
        if self._dr_custom_metric_api_client is None:
            return

        cleanup_metrics_list = list()
        for index, (metric_name, custom_metric) in enumerate(self._custom_metric_map.items()):
            metric_definition = custom_metric["metric_definition"]
            try:
                # We create metrics one by one, instead of using a library call.  This gives
                # us control over which are duplicates, if max limit reached etc and we can
                # take appropriate actions accordingly.  Performance wise it is same, because
                # library also runs a loop to create custom metrics one by one
                custom_metric_id = self._dr_custom_metric_api_client.create_custom_metric(
                    deployment_id=self._deployment_id,
                    name=metric_name,
                    directionality=metric_definition["directionality"],
                    aggregation_type=metric_definition["type"],
                    time_step=metric_definition["timeStep"],
                    units=metric_definition["units"],
                    baseline_value=metric_definition["baselineValue"],
                    is_model_specific=metric_definition["isModelSpecific"],
                )
                custom_metric["id"] = str(custom_metric_id)
                custom_metric["requires_association_id"] = self._requires_association_id(
                    metric_name
                )
            except ClientError as e:
                if e.status_code == 409:
                    if "not unique for deployment" in e.json["message"]:
                        # Duplicate entry nothing to worry - just continue
                        self._logger.error(f"Metric '{metric_name}' already exists, skipping")
                        continue
                    elif e.json["message"].startswith("Maximum number of custom metrics reached"):
                        # Reached the limit - we can't create more
                        self._logger.error(f"Failed to create metric '{metric_name}'")
                        self._logger.error("Maximum number of custom metrics reached")
                        cleanup_metrics_list = list(self._custom_metric_map.keys())[index:]
                        self._logger.error(
                            f"Cannot create rest of the metrics: {cleanup_metrics_list}"
                        )
                        # Lets not raise the exception, for now - break the loop and
                        # consolidate valid custom metrics
                        break
                # Else raise it and catch in next block
                raise
            except Exception as e:
                self._logger.error(f"Failed to create custom metrics: {e}")
                self._logger.error(traceback.format_exc())
                self._logger.error(f"Custom Metric definition: {custom_metric}")
                cleanup_metrics_list.append(metric_name)
                # Lets again not raise exception
                continue

        # Now query all the metrics and get their custom metric ids.  Specifically,
        # required in case a metric is duplicated, in which case, we don't have its
        # id in the loop above
        #
        # We have to go through pagination - dmm list_custom_metrics does not implement
        # pagination
        custom_metrics_list = []
        offset, limit = 0, 50
        while True:
            response_list = self.dr_client.get(
                f"deployments/{self._deployment_id}/customMetrics/?offset={offset}&limit={limit}"
            ).json()
            custom_metrics_list.extend(response_list["data"])
            offset += response_list["count"]
            if response_list["next"] is None:
                break

        for metric in custom_metrics_list:
            metric_name = metric["name"]
            if metric_name not in self._custom_metric_map:
                self._logger.error(f"Metric '{metric_name}' exists at DR but not in moderation")
                continue
            self._custom_metric_map[metric_name]["id"] = metric["id"]
            self._custom_metric_map[metric_name]["requires_association_id"] = (
                self._requires_association_id(metric_name)
            )

        # These are the metrics we couldn't create - so, don't track them
        for metric_name in cleanup_metrics_list:
            if not self._custom_metric_map[metric_name].get("id"):
                self._logger.error(f"Skipping metric creation: {metric_name}")
                del self._custom_metric_map[metric_name]

    def _requires_association_id(self, metric_name):
        return metric_name not in self._custom_metrics_no_association_ids

    def _add_guard_to_pipeline(self, guard):
        if guard.stage == GuardStage.PROMPT:
            self._pre_score_guards.append(guard)
        elif guard.stage == GuardStage.RESPONSE:
            self._post_score_guards.append(guard)
        else:
            print("Ignoring invalid guard stage", guard.stage)

    def report_stage_total_inputs(self, stage, num_rows):
        if self._dr_custom_metric_api_client is None or self._aggregate_custom_metric is None:
            return

        entry = self._aggregate_custom_metric[f"Total {get_stage_str(stage)}"]
        self._set_custom_metrics_aggregate_entry(entry, num_rows)

    def get_prescore_guards(self):
        return self._pre_score_guards

    def get_postscore_guards(self):
        return self._post_score_guards

    def report_stage_latency(self, latency_in_sec, stage):
        if self._dr_custom_metric_api_client is None or self._aggregate_custom_metric is None:
            return

        if stage == GuardStage.PROMPT:
            metric_name = prescore_guard_latency_custom_metric["name"]
        else:
            metric_name = postscore_guard_latency_custom_metric["name"]
        entry = self._aggregate_custom_metric[metric_name]
        self._set_custom_metrics_aggregate_entry(entry, latency_in_sec)

    def report_guard_latency(self, guard, latency_in_sec):
        if (
            guard is None
            or self._dr_custom_metric_api_client is None
            or self._aggregate_custom_metric is None
        ):
            return

        guard_latency_name = guard.get_latency_custom_metric_name()
        entry = self._aggregate_custom_metric[guard_latency_name]
        self._set_custom_metrics_aggregate_entry(entry, latency_in_sec)

    def report_score_latency(self, latency_in_sec):
        if self._dr_custom_metric_api_client is None or self._aggregate_custom_metric is None:
            return

        entry = self._aggregate_custom_metric[score_latency["name"]]
        self._set_custom_metrics_aggregate_entry(entry, latency_in_sec)

    @property
    def prediction_url(self):
        return self._datarobot_url

    @property
    def api_token(self):
        return self._datarobot_api_token

    def get_input_column(self, stage):
        if stage == GuardStage.PROMPT:
            return (
                self._prompt_column_name if self._prompt_column_name else DEFAULT_PROMPT_COLUMN_NAME
            )
        else:
            # DRUM ensures that TARGET_NAME is always set as environment variable, but
            # TARGET_NAME comes in double quotes, remove those
            return (
                self._response_column_name
                if self._response_column_name
                else (os.environ.get("TARGET_NAME").replace('"', ""))
            )

    def set_model_dir(self, model_dir):
        self._custom_model_dir = model_dir

    def get_model_dir(self):
        return self._custom_model_dir

    def get_association_id_column_name(self):
        return self._association_id_column_name

    def get_new_metrics_payload(self):
        self._query_association_id_column_name()

        if self._deployment is None:
            return

        self._aggregate_custom_metric = dict()
        for metric_name, metric_info in self._custom_metric_map.items():
            if not metric_info["requires_association_id"]:
                self._aggregate_custom_metric[metric_name] = {
                    "customMetricId": str(metric_info["id"])
                }

    def _set_custom_metrics_aggregate_entry(self, entry, value):
        if isinstance(value, np.generic):
            entry["value"] = value.item()
        else:
            entry["value"] = value
        entry["timestamp"] = str(datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat())
        entry["sampleSize"] = 1

    def _set_custom_metrics_individual_entry(self, metric_id, value, association_id):
        if isinstance(value, bool):
            _value = 1.0 if value else 0.0
        elif isinstance(value, np.bool_):
            _value = 1.0 if value.item() else 0.0
        elif isinstance(value, np.generic):
            _value = value.item()
        else:
            _value = value
        return {
            "customMetricId": str(metric_id),
            "value": _value,
            "associationId": str(association_id),
            "sampleSize": 1,
            "timestamp": str(datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat()),
        }

    def get_enforced_column_name(self, guard, stage):
        input_column = self.get_input_column(stage)
        intervention_action = guard.get_intervention_action()
        if intervention_action == GuardAction.REPLACE:
            return f"{guard.name}_replaced_{input_column}"
        else:
            return f"{guard.name}_{intervention_action}ed_{input_column}"

    def get_guard_specific_custom_metric_names(self, guard):
        intervention_action = guard.get_intervention_action()
        metric_list = []
        if isinstance(guard, NeMoGuard):
            pass  # No average score metric for NeMo Guard
        elif isinstance(guard, ModelGuard) and guard.model_info.target_type == "Multiclass":
            pass  # No average score metric for Multiclass model guards
        else:
            metric_list = [
                (
                    guard.get_average_score_custom_metric_name(guard.stage),
                    guard.get_metric_column_name(guard.stage),
                )
            ]
        if intervention_action:
            metric_list.append(
                (
                    guard.get_guard_enforced_custom_metric_name(guard.stage, intervention_action),
                    self.get_enforced_column_name(guard, guard.stage),
                )
            )
        return metric_list

    def _add_guard_specific_custom_metrics(self, row, guards):
        if len(guards) == 0:
            return []

        association_id = row[self._association_id_column_name]

        buckets = []
        for guard in guards:
            for metric_name, column_name in self.get_guard_specific_custom_metric_names(guard):
                if column_name not in row:
                    # It is possible metric column is missing if there is exception
                    # executing the guard.  Just continue with rest
                    self._logger.warning(
                        f"Missing {column_name} in result for guard {guard.name} "
                        f"Not reporting the value with association id {association_id}"
                    )
                    continue
                if math.isnan(row[column_name]):
                    self._logger.warning(
                        f"{column_name} in result is NaN for guard {guard.name} "
                        f"Not reporting the value with association id {association_id}"
                    )
                    continue
                custom_metric_id = self._custom_metric_map[metric_name].get("id")
                if custom_metric_id is None:
                    self._logger.warning(f"No metric id for '{metric_name}', not reporting")
                    continue
                bucket = self._set_custom_metrics_individual_entry(
                    custom_metric_id, row[column_name], association_id
                )
                buckets.append(bucket)
        return buckets

    def _get_blocked_column_name_from_result_df(self, stage):
        input_column_name = self.get_input_column(stage)
        return f"blocked_{input_column_name}"

    def _set_individual_custom_metrics_entries(self, result_df, payload):
        for index, row in result_df.iterrows():
            association_id = row[self._association_id_column_name]
            for stage in GuardStage.ALL:
                blocked_metric_name = f"Blocked {get_stage_str(stage)}"
                blocked_column_name = self._get_blocked_column_name_from_result_df(stage)
                if blocked_metric_name not in self._custom_metric_map:
                    continue
                if blocked_column_name not in result_df.columns:
                    continue
                if math.isnan(row[blocked_column_name]):
                    # If prompt is blocked, response will be NaN, so don't report it
                    continue
                custom_metric_id = self._custom_metric_map[blocked_metric_name].get("id")
                if custom_metric_id is None:
                    self._logger.warning(f"No metric id for '{blocked_metric_name}', not reporting")
                    continue
                bucket = self._set_custom_metrics_individual_entry(
                    custom_metric_id, row[blocked_column_name], association_id
                )
                payload["buckets"].append(bucket)

            buckets = self._add_guard_specific_custom_metrics(row, self.get_prescore_guards())
            payload["buckets"].extend(buckets)
            buckets = self._add_guard_specific_custom_metrics(row, self.get_postscore_guards())
            payload["buckets"].extend(buckets)

    def report_custom_metrics(self, result_df):
        if self._association_id_column_name is None:
            return

        payload = {"buckets": []}

        if self._association_id_column_name in result_df.columns:
            # Custom metrics are reported only if the association id column
            # is defined and is "present" in result_df
            self._set_individual_custom_metrics_entries(result_df, payload)

        # Ensure that "Total Prompts" and "Total Responses" are set properly too.
        for stage in GuardStage.ALL:
            entry = self._aggregate_custom_metric[f"Total {get_stage_str(stage)}"]
            if "value" not in entry:
                if stage == GuardStage.PROMPT:
                    # If No prompt guards, then all entries are in Total Prompts
                    self._set_custom_metrics_aggregate_entry(entry, result_df.shape[0])
                    latency_entry = self._aggregate_custom_metric[
                        prescore_guard_latency_custom_metric["name"]
                    ]
                    self._set_custom_metrics_aggregate_entry(latency_entry, 0.0)
                else:
                    # Prompt guards might have blocked some, so remaining will be
                    # Total Responses
                    blocked_column_name = self._get_blocked_column_name_from_result_df(
                        GuardStage.PROMPT
                    )
                    value = result_df.shape[0] - ((result_df[blocked_column_name]).sum())
                    self._set_custom_metrics_aggregate_entry(entry, value)
                    latency_entry = self._aggregate_custom_metric[
                        postscore_guard_latency_custom_metric["name"]
                    ]
                    self._set_custom_metrics_aggregate_entry(latency_entry, 0.0)

        self._bulk_upload_custom_metrics(payload)

    def _bulk_upload_custom_metrics(self, payload):
        if self._model_id:
            payload["modelId"] = self._model_id

        for metric_name, metric_value in self._aggregate_custom_metric.items():
            if "value" not in metric_value:
                # Different exception paths - especially with asyncio can
                # end up not adding values for some aggregated custom metrics
                # Capturing them for future fixes
                self._logger.warning(f"No value for custom metric {metric_name}")
                continue
            if not math.isnan(metric_value["value"]):
                payload["buckets"].append(metric_value)

        self._logger.debug(f"Payload: {payload}")

        if len(payload["buckets"]) == 0:
            self._logger.warning("No custom metrics to report, empty payload")
            return

        url = self._datarobot_url + "/" + self._custom_metrics_bulk_upload_url
        try:
            response = requests.post(url, data=json.dumps(payload), headers=self._headers)
            if response.status_code != 202:
                raise Exception(
                    f"Error uploading custom metrics: Status Code: {response.status_code}"
                    f"Message: {response.text}"
                )
            self._logger.info("Successfully uploaded custom metrics")
        except Exception as e:
            self._logger.error(f"Failed to upload custom metrics: {e}")
            self._logger.error(traceback.format_exc())
            self._logger.error(f"Payload: {payload}")
            # Lets not raise the exception, just walk off

    @property
    def custom_metrics(self):
        return {
            metric_name: metric_info for metric_name, metric_info in self._custom_metric_map.items()
        }
