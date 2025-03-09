#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import json
from datetime import datetime, timezone
from typing import Any

import pytest

import pandas as pd
import responses
import tiktoken

from datarobot_drum.drum.adapters.model_adapters.custom_metrics import (
    create_vdb_metric_pipeline,
    fetch_deployment_custom_metrics,
)
from datarobot_drum.drum.enum import VectorDatabaseMetrics

MODEL_ID = "abadface"
MODEL_PKG_ID = "c0ffee"

METRIC_ID_MAP = {name: str(index) * 8 for index, name in enumerate(VectorDatabaseMetrics)}

# NOTE: force loading the encodings BEFORE responses snags the request
encoder = tiktoken.get_encoding("cl100k_base")


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


@pytest.fixture
def mock_server_address() -> str:
    return "http://my-local-host/api/v2"


@pytest.fixture
def deployment_id() -> str:
    return "abcdefghijk"


def list_to_body(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    return {
        "next": None,
        "previous": None,
        "data": items,
        "count": count,
        "totalCount": count,
    }


def full_metric(name: str, identifier: str) -> dict[str, Any]:
    return {
        # only real important parts are the name, and identifier
        "name": name,
        "id": identifier,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "createdBy": {"id": "556dd9677adc72e95f7b5013"},
        "type": "sum",
        "units": "score",
        "isModelSpecific": True,
        "directionality": "higherIsBetter",
        "timeStep": "hour",
        "baselineValues": [],
        "timestamp": {"columnName": "timestamp", "timeFormat": None},
        "value": {"columnName": "value"},
        "sampleCount": {"columnName": "sample_count"},
        "batch": {"columnName": "batch"},
        "associationId": {"columnName": "association_id"},
        "description": "",
        "displayChart": True,
        "categories": None,
        "metadata": {"columnName": "metadata"},
        "isGeospatial": False,
        "geospatialSegmentAttribute": None,
    }


@pytest.fixture
def full_custom_metrics_response(mock_server_address, deployment_id):
    responses.add(
        responses.GET,
        f"{mock_server_address}/deployments/{deployment_id}/customMetrics/",
        body=json.dumps(
            list_to_body(
                [full_metric(name, identifier) for name, identifier in METRIC_ID_MAP.items()]
            )
        ),
    )
    yield


@pytest.fixture
def bulk_upload_success(mock_server_address, deployment_id):
    responses.add(
        responses.POST,
        f"{mock_server_address}/deployments/{deployment_id}/customMetrics/bulkUpload/",
        status=202,
    )
    yield


@pytest.fixture
def bulk_upload_failure(mock_server_address, deployment_id):
    responses.add(
        responses.POST,
        f"{mock_server_address}/deployments/{deployment_id}/customMetrics/bulkUpload/",
        status=400,
    )
    yield


def expected_map(
    total_citations: int, average_citations: float, total_docs: int, average_docs: float
) -> dict[str, Any]:
    return {
        METRIC_ID_MAP[VectorDatabaseMetrics.TOTAL_CITATION_TOKENS]: total_citations,
        METRIC_ID_MAP[VectorDatabaseMetrics.AVERAGE_CITATION_TOKENS]: average_citations,
        METRIC_ID_MAP[VectorDatabaseMetrics.TOTAL_DOCUMENTS]: total_docs,
        METRIC_ID_MAP[VectorDatabaseMetrics.AVERAGE_DOCUMENTS]: average_docs,
    }


@responses.activate
@pytest.mark.usefixtures("full_custom_metrics_response", "bulk_upload_success")
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
def test_custom_metrics_processor_success(
    mock_server_address, deployment_id, dataframe: pd.DataFrame, expected: dict[str, Any]
) -> None:
    processor = create_vdb_metric_pipeline(
        host=mock_server_address,
        api_token="<TOKEN>",
        deployment_id=deployment_id,
        model_id=MODEL_ID,
        model_package_id=MODEL_PKG_ID,
    )
    processor.process_predictions(dataframe)

    payload = json.loads(responses.calls[1].request.body)
    assert MODEL_ID == payload["modelId"]
    assert MODEL_PKG_ID == payload["modelPackageId"]

    # remap to buckets to be a dict by id
    reported = {p.get("customMetricId"): p for p in payload["buckets"]}
    assert set(reported.keys()) == set(expected.keys())
    for identifier, report in reported.items():
        assert expected[identifier] == report["value"]


@responses.activate
@pytest.mark.usefixtures("full_custom_metrics_response", "bulk_upload_failure")
def test_custom_metrics_processor_failure(mock_server_address, deployment_id):
    dataframe = single_citation_df()
    processor = create_vdb_metric_pipeline(
        host=mock_server_address,
        api_token="<TOKEN>",
        deployment_id=deployment_id,
        model_id=None,
        model_package_id=None,
    )
    processor.process_predictions(dataframe)

    payload = json.loads(responses.calls[1].request.body)
    assert "modelId" not in payload
    assert "modelPackageId" not in payload

    # remap to buckets to be a dict by id
    reported = {p.get("customMetricId"): p for p in payload["buckets"]}
    assert set(reported.keys()) == set(METRIC_ID_MAP.values())


def generate_custom_metrics(count: int, start: int = 0) -> dict[str, Any]:
    return list_to_body(
        [
            full_metric(f"Unittest Metric {index + start + 1}", f"11111111{index + start:8}")
            for index in range(count)
        ]
    )


@pytest.fixture
def custom_metrics_one_page(mock_server_address, deployment_id):
    responses.add(
        responses.GET,
        f"{mock_server_address}/deployments/{deployment_id}/customMetrics/",
        body=json.dumps(generate_custom_metrics(count=3)),
    )
    yield


@responses.activate
@pytest.mark.usefixtures("custom_metrics_one_page")
def test_fetch_deployment_custom_metrics_simple(mock_server_address, deployment_id):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    metrics = fetch_deployment_custom_metrics(mock_server_address, headers, deployment_id)
    assert len(metrics) == 3


@pytest.fixture
def custom_metrics_two_pages(mock_server_address, deployment_id):
    responses.add(
        responses.GET,
        f"{mock_server_address}/deployments/{deployment_id}/customMetrics/",
        body=json.dumps(generate_custom_metrics(count=20)),
        match=[responses.matchers.query_param_matcher({"offset": "0", "limit": "20"})],
    )
    responses.add(
        responses.GET,
        f"{mock_server_address}/deployments/{deployment_id}/customMetrics/?offset=20&limit=20",
        body=json.dumps(generate_custom_metrics(count=2, start=20)),
        match=[responses.matchers.query_param_matcher({"offset": "20", "limit": "20"})],
    )
    yield


@responses.activate
@pytest.mark.usefixtures("custom_metrics_two_pages")
def test_fetch_deployment_custom_metrics_two_pages(mock_server_address, deployment_id):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    metrics = fetch_deployment_custom_metrics(mock_server_address, headers, deployment_id)
    assert len(metrics) == 22


@responses.activate
def test_fetch_deployment_custom_metrics_not_found(mock_server_address, deployment_id):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    metrics = fetch_deployment_custom_metrics(mock_server_address, headers, deployment_id)
    assert len(metrics) == 0
