import json
import pytest

import datarobot_drum.drum.root_predictors.prediction_server as ps


class DummySpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.status = None
        self.status_description = None

    def set_attributes(self, attrs: dict):
        self.attributes.update(attrs)

    def set_status(self, status_code, description=None):
        self.status = status_code
        self.status_description = description


class DummyOtelContext:
    def __init__(self, span_name, span_store):
        self._span = DummySpan(span_name)
        self._span_store = span_store

    def __enter__(self):
        self._span_store.append(self._span)
        return self._span

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def span_store():
    return []


@pytest.fixture
def patch_otel_context(monkeypatch, span_store):
    def fake_otel_context(tracer, span_name, request_headers):
        return DummyOtelContext(span_name, span_store)

    monkeypatch.setattr(ps, "otel_context", fake_otel_context)
    return span_store


@pytest.fixture
def patch_chat_attr_extractor(monkeypatch):
    def fake_extract_chat_request_attributes(body):
        return {
            "gen_ai.request.model": "patched-model",
            "gen_ai.request.tokens": 42,
        }

    monkeypatch.setattr(ps, "extract_chat_request_attributes", fake_extract_chat_request_attributes)


@pytest.fixture
def patch_do_chat(monkeypatch):
    def fake_do_chat(self, logger):
        return {
            "id": "dummy",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        }, 200

    monkeypatch.setattr(ps.PredictionServer, "do_chat", fake_do_chat)


@pytest.fixture
def patch_do_predict_unstructured(monkeypatch):
    def fake_do_predict_unstructured(self, logger):
        return {"message": "ok"}, 200

    monkeypatch.setattr(
        ps.PredictionServer, "do_predict_unstructured", fake_do_predict_unstructured
    )


@pytest.fixture
def prediction_client(test_flask_app, prediction_server):
    return test_flask_app.test_client()


@pytest.mark.usefixtures("patch_otel_context", "patch_do_predict_unstructured")
def test_predict_unstructured_span_includes_consumer_headers(prediction_client, span_store):
    headers = {
        "X-DataRobot-Consumer-Id": "abc123",
        "X-DataRobot-Consumer-Type": "external",
        "Content-Type": "application/json",
    }
    resp = prediction_client.post("/predictUnstructured/", data=json.dumps({}), headers=headers)
    assert resp.status_code == 200
    assert span_store, "No span captured"
    span = span_store[-1]
    assert span.name == "drum.predictUnstructured"
    assert span.attributes.get("gen_ai.request.consumer_id") == "abc123"
    assert span.attributes.get("gen_ai.request.consumer_type") == "external"


@pytest.mark.usefixtures("patch_otel_context", "patch_chat_attr_extractor", "patch_do_chat")
def test_chat_span_combines_chat_and_header_attributes(prediction_client, span_store):
    headers = {
        "X-DataRobot-Consumer-Id": "chat-user-7",
        "X-DataRobot-Consumer-Type": "service",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "ignored",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = prediction_client.post("/v1/chat/completions", data=json.dumps(payload), headers=headers)
    assert resp.status_code == 200
    assert span_store, "No span captured"

    span = span_store[-1]
    assert span.name == "drum.chat.completions"
    assert span.attributes["gen_ai.request.model"] == "patched-model"
    assert span.attributes["gen_ai.request.tokens"] == 42
    assert span.attributes.get("gen_ai.request.consumer_id") == "chat-user-7"
    assert span.attributes.get("gen_ai.request.consumer_type") == "service"


@pytest.mark.usefixtures("patch_otel_context")
def test_direct_access_span_includes_consumer_headers(prediction_client, span_store):
    headers = {
        "X-DataRobot-Consumer-Id": "direct-1",
        "X-DataRobot-Consumer-Type": "agent",
    }
    resp = prediction_client.get("/nim/some/path", headers=headers)
    assert resp.status_code == 400
    assert span_store, "No span captured"
    span = span_store[-1]
    assert span.name == "drum.directAccess"
    assert span.attributes["gen_ai.request.consumer_id"] == "direct-1"
    assert span.attributes["gen_ai.request.consumer_type"] == "agent"


@pytest.mark.usefixtures("patch_otel_context")
def test_transform_without_consumer_headers(prediction_client, span_store):
    headers = {"Content-Type": "application/json"}
    resp = prediction_client.post("/transform/", data=json.dumps({}), headers=headers)
    assert resp.status_code == 422
    assert span_store, "No span captured"
    span = span_store[-1]
    assert span.name == "drum.transform"
    assert "gen_ai.request.consumer_id" not in span.attributes
    assert "gen_ai.request.consumer_type" not in span.attributes


@pytest.mark.usefixtures("patch_otel_context")
def test_invocations_span_includes_consumer_headers(prediction_client, span_store):
    headers = {
        "X-DataRobot-Consumer-Id": "abc123",
        "X-DataRobot-Consumer-Type": "external",
        "Content-Type": "application/json",
    }
    resp = prediction_client.post("/invocations", data=json.dumps({}), headers=headers)
    assert resp.status_code == 422
    assert span_store, "No span captured"
    span = span_store[-1]
    assert span.name == "drum.invocations"
    assert span.attributes.get("gen_ai.request.consumer_id") == "abc123"
    assert span.attributes.get("gen_ai.request.consumer_type") == "external"
