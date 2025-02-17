"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import logging
import traceback

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import math
import numpy as np
import pandas as pd
import requests
import tiktoken

from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    VectorDatabaseMetrics,
)


logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


def get_deployment_url(host: str, deployment_id: str, *args) -> str:
    """Get the URL for the deployment sub-path."""
    # remove the api/v2 from the host, since it is now in the urls
    _host = host.replace("/api/v2", "")
    parts = [_host, "api/v2/deployments", deployment_id] + list(args)
    return "/".join([_.strip("/") for _ in parts]) + "/"


def get_token_count(value: str) -> int:
    """Get the token count for the input."""
    if value is None:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(value), disallowed_special=()))


def get_citation_columns(columns: pd.Index) -> list:
    """
    Ensure that citation columns are returned in the order 0, 1, 2, etc
    Order matters
    """
    index = 0
    citation_columns = []
    while True:
        column_name = f"CITATION_CONTENT_{index}"
        if column_name not in columns:
            break
        citation_columns.append(column_name)
        index += 1

    return citation_columns


class AggregatedMetricEvaluator(ABC):
    """Abstract base class for metrics that report a value per prompt."""

    def __init__(self, metric_name: str, metric_id: str):
        self.metric_name = metric_name
        self.metric_id = metric_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metric_id={self.metric_id})"

    @abstractmethod
    def score(self, df: pd.DataFrame) -> float:
        """
        This method must be implemented by subclasses. It evaluates the data in the DataFrame.
        """


class CitationTokenCount(AggregatedMetricEvaluator):
    """Counts the total tokens in all citation values."""

    def score(self, df: pd.DataFrame) -> float:
        total = 0
        columns = get_citation_columns(df.columns)
        for column in columns:
            total += sum(get_token_count(v) for v in df[column].values)

        return total


class CitationTokenAverage(AggregatedMetricEvaluator):
    """Counts the average numer of tokens per citation."""

    def score(self, df: pd.DataFrame) -> float:
        average = 0.0
        total = 0
        count = 0
        columns = get_citation_columns(df.columns)
        for column in columns:
            total += sum(get_token_count(v) for v in df[column].values)
            count += sum(v != "" for v in df[column].values)
            average = total / count

        return average


class DocumentCount(AggregatedMetricEvaluator):
    """Counts tne number of non-blank citations"""

    def score(self, df: pd.DataFrame) -> float:
        # not sure it is necessary to check if blank...
        non_blank = 0
        columns = get_citation_columns(df.columns)
        for column in columns:
            non_blank += sum(v != "" for v in df[column].values)

        return non_blank


class DocumentAverage(AggregatedMetricEvaluator):
    """Counts tne number of non-blank citations"""

    def score(self, df: pd.DataFrame) -> float:
        # not sure it is necessary to check if blank...
        non_blank = 0
        columns = get_citation_columns(df.columns)
        for column in columns:
            non_blank += sum(v != "" for v in df[column].values)

        return non_blank


class CustomMetricsProcessor:
    """
    This implements the processing of the VDB metrics.

    For each call to `process_predictions()`, the metrics will be calculated and reported
    to the deployment's bulk-upload.
    """

    def __init__(
        self,
        host: str,
        headers: dict[str, Any],
        deployment_id: str,
        model_id: Optional[str],
        model_package_id: Optional[str],
        metrics: list[AggregatedMetricEvaluator],
    ):
        self.deployment_id = deployment_id
        self.model_id = model_id
        self.model_package_id = model_package_id
        self._aggregated_metrics = metrics
        self._bulk_upload_url = get_deployment_url(host, deployment_id, "customMetrics/bulkUpload")
        self._headers = headers
        self._logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + self.__class__.__name__)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(deployment_id={self.deployment_id}, "
            ""
            f"metrics={self._aggregated_metrics})"
        )

    @staticmethod
    def _create_metric_payload(metric_id: str, value: Any, timestamp: str):
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
            "sampleSize": 1,
            "timestamp": timestamp,
        }

    def _create_custom_metrics_buckets(self, result_df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        For each row in the result_df (typically only one), create a custom metric entry for each
        custom metric.
        """

        buckets = []

        # one timestamp for all the metrics in the row
        timestamp = str(datetime.now(timezone.utc).isoformat())

        for metric in self._aggregated_metrics:
            value = metric.score(result_df)
            if math.isnan(value):
                continue

            bucket = self._create_metric_payload(metric.metric_id, value, timestamp)
            buckets.append(bucket)

        return buckets

    def create_custom_metrics_bulk_payload(self, result_df: pd.DataFrame) -> dict[str, Any]:
        """Creates the bulkUpload payload for the metrics generated by processing the dataframe."""
        payload = {}
        if self.model_id:
            payload["modelId"] = self.model_id

        if self.model_package_id:
            payload["modelPackageId"] = self.model_package_id

        buckets = self._create_custom_metrics_buckets(result_df)
        payload["buckets"] = buckets

        return payload

    def process_predictions(self, result_df: pd.DataFrame) -> None:
        """
        Processes the predictions in the provided dataframe.

        Includes the calculations, and the reporting of those metrics using the buld upload.
        """
        payload = self.create_custom_metrics_bulk_payload(result_df)
        if len(payload["buckets"]) == 0:
            self._logger.warning("No custom metrics to report, empty payload")
            return

        self._logger.debug("Payload: {}".format(payload))

        try:
            response = requests.post(
                self._bulk_upload_url, data=json.dumps(payload), headers=self._headers
            )
            if response.status_code != 202:
                raise Exception(
                    f"Error uploading custom metrics: Status Code: {response.status_code}"
                    f"Message: {response.text}"
                )
            self._logger.info("Successfully uploaded custom metrics")
        except Exception as e:
            title = "Failed to upload custom metrics"
            message = f"Exception: {e} Payload: {payload}"
            self._logger.error(title + " " + message)
            self._logger.error(traceback.format_exc())
            # TODO: send event
            # Let's not raise the exception, just walk off


METRIC_NAME_TO_CLASS_MAP = {
    VectorDatabaseMetrics.TOTAL_CITATION_TOKENS.value: CitationTokenCount,
    VectorDatabaseMetrics.AVERAGE_CITATION_TOKENS.value: CitationTokenAverage,
    VectorDatabaseMetrics.TOTAL_DOCUMENTS.value: DocumentCount,
    VectorDatabaseMetrics.AVERAGE_DOCUMENTS.value: DocumentAverage,
}


def metric_factory(name: VectorDatabaseMetrics, metric_id: str) -> AggregatedMetricEvaluator:
    """Creates a new "processor" for the provided metric name."""
    return METRIC_NAME_TO_CLASS_MAP[name](name, metric_id)


def fetch_deployment_custom_metrics(
    host: str, headers: dict[str, Any], deployment_id: str
) -> dict[str, Any]:
    """
    Fetches all the custom-metrics for a given deployment, and organizes them into a map by name.
    """
    url = get_deployment_url(host, deployment_id, "customMetrics/")
    offset = 0
    limit = 100
    deployment_metrics = {}
    params = {"offset": offset, "limit": limit}
    while True:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # not sure if this is needed
        except Exception as e:
            logger.warning(f"Could not load vdb metrics: {e}")
            break

        items = response.json().get("data", [])
        offset += limit

        # create a map by name to avoid looping below
        deployment_metrics.update({item.get("name"): item for item in items})

        # keep going until we don't get a full page
        if len(items) < limit:
            break

    return deployment_metrics


def create_vdb_metric_pipeline(
    host: str,
    api_token: str,
    deployment_id: str,
    model_id: Optional[str],
    model_package_id: Optional[str],
) -> Optional[CustomMetricsProcessor]:
    """
    Create a VDB metric processor when appropriate.

    Fetches the deployment's custom-metrics, and tries to align (by name) with the
    VDB metrics. If no overlap is found, it does NOT create a `CustomMetricsProcessor`.
    If overlap is found, it creates the `CustomMetricsProcessor` with the metrics
    that were found to overlap, so each dataframe can be processed.
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_token}"}

    # get list of custom-metrics for the deployment, and use the name try to reconcile with
    # the VectorDatabaseMetrics.
    deployment_metrics = fetch_deployment_custom_metrics(host, headers, deployment_id)
    metric_ids = {}
    for metric_type in VectorDatabaseMetrics:
        metric = deployment_metrics.get(metric_type)
        if not metric:
            logger.warning(f"No metric found for {metric_type}")
            continue

        metric_ids[metric_type] = metric.get("id")

    # until the processor searches for custom-metrics again, no need to create it
    if not metric_ids:
        logger.error("No custom metric matches found")
        return None

    evaluators = [metric_factory(name, metric_id) for name, metric_id in metric_ids.items()]
    return CustomMetricsProcessor(
        host=host,
        headers=headers,
        deployment_id=deployment_id,
        model_id=model_id,
        model_package_id=model_package_id,
        metrics=evaluators,
    )
