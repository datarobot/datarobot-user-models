#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from typing import Dict, Any
from unittest.mock import patch

import pytest

import pandas as pd

from custom_model_runner.datarobot_drum.drum.adapters.model_adapters.custom_metrics import (
    create_vdb_metric_pipeline,
)
from custom_model_runner.datarobot_drum.drum.enum import VectorDatabaseMetrics

CUSTOM_METRIC_MODULE = (
    "custom_model_runner.datarobot_drum.drum.adapters.model_adapters.custom_metrics"
)
FETCH_METRIC_FUNCTION = f"{CUSTOM_METRIC_MODULE}.fetch_deployment_custom_metrics"

METRIC_ID_MAP = {name: str(index) * 8 for index, name in enumerate(VectorDatabaseMetrics)}


def no_citations_df() -> pd.DataFrame:
    """Single row, no citations"""
    columns = ["foo", "bar"]
    data = [("messed up", "beyond all recognition")]
    return pd.DataFrame(data, columns=columns)


def single_citation_df() -> pd.DataFrame:
    columns = ["sna", "foo", "CITATION_CONTENT_0"]
    data = [("situation normal all", "messed up", "read it in some book in the library")]
    return pd.DataFrame(data, columns=columns)


def multiple_citation_df() -> pd.DataFrame:
    columns = ["prompt", "CITATION_CONTENT_0", "CITATION_CONTENT_1", "CITATION_CONTENT_2"]
    data = [
        ("some prompt data", "this is a citation", "read it in some book", "another small citation")
    ]
    return pd.DataFrame(data, columns=columns)


def full_citation_df() -> pd.DataFrame:
    columns = ["prompt", "CITATION_CONTENT_0", "CITATION_CONTENT_1"]
    data = [
        ("some prompt data", "read it in some book", "another small citation"),
        ("another row", "read it on the Internet", "Abraham Lincoln says it is true"),
    ]
    return pd.DataFrame(data, columns=columns)


def empty_citations_df() -> pd.DataFrame:
    columns = ["prompt", "CITATION_CONTENT_0", "CITATION_CONTENT_1"]
    data = [
        ("some prompt data", "", "small citation"),
        ("another row", "read it on Internet", ""),
    ]
    return pd.DataFrame(data, columns=columns)


def full_custom_metrics_response() -> Dict[str, Any]:
    return {name: {"id": identifier, "name": name} for name, identifier in METRIC_ID_MAP.items()}


def expected_map(
    total_citations: int, average_citations: float, total_docs: int, average_docs: float
) -> dict[str, Any]:
    return {
        METRIC_ID_MAP[VectorDatabaseMetrics.TOTAL_CITATION_TOKENS]: total_citations,
        METRIC_ID_MAP[VectorDatabaseMetrics.AVERAGE_CITATION_TOKENS]: average_citations,
        METRIC_ID_MAP[VectorDatabaseMetrics.TOTAL_DOCUMENTS]: total_docs,
        METRIC_ID_MAP[VectorDatabaseMetrics.AVERAGE_DOCUMENTS]: average_docs,
    }


@pytest.mark.parametrize(
    ["dataframe", "expected"],
    [
        pytest.param(no_citations_df(), expected_map(0, 0, 0, 0), id="none"),
        pytest.param(single_citation_df(), expected_map(8, 8, 1, 1), id="single"),
        pytest.param(multiple_citation_df(), expected_map(12, 4, 3, 3), id="multi"),
        pytest.param(full_citation_df(), expected_map(20, 5, 4, 4), id="full"),
        pytest.param(empty_citations_df(), expected_map(6, 3, 2, 2), id="empty"),
    ],
)
def test_custom_metrics_processor(dataframe: pd.DataFrame, expected: dict[str, Any]) -> None:
    deployment_id = "1234567890123456"
    with (patch(FETCH_METRIC_FUNCTION, return_value=full_custom_metrics_response()),):
        processor = create_vdb_metric_pipeline(
            host="localhost",
            api_token="<TOKEN>",
            deployment_id=deployment_id,
            model_id=None,
            model_package_id=None,
        )
        payload = processor.create_custom_metrics_bulk_payload(result_df=dataframe)

        # remap to buckets to be a dict by id
        reported = {p.get("customMetricId"): p for p in payload["buckets"]}
        assert set(reported.keys()) == set(expected.keys())
        for identifier, report in reported.items():
            assert expected[identifier] == report["value"]
