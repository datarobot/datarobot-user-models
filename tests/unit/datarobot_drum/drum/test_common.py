import os

from unittest.mock import patch

import json

import types
from unittest import mock
import pytest

from datarobot_drum.drum.common import setup_otel
from datarobot_drum.drum.common import extract_chat_request_attributes
from datarobot_drum.drum.common import extract_chat_response_attributes
from datarobot_drum.drum.common import reconstruct_chat_response_from_sse
from datarobot_drum.drum.common import iter_stream_with_span

from datarobot_drum import RuntimeParameters

from datarobot_drum.runtime_parameters.runtime_parameters_schema import RuntimeParameterTypes


@pytest.fixture
def otel_sdk_enabled():
    namespaced_runtime_param_name = RuntimeParameters.namespaced_param_name("OTEL_SDK_DISABLED")
    runtime_param_env_value = json.dumps(
        {"type": RuntimeParameterTypes.BOOLEAN.value, "payload": False}
    )
    os.environ[namespaced_runtime_param_name] = runtime_param_env_value
    yield
    del os.environ[namespaced_runtime_param_name]


@pytest.fixture
def otel_sdk_disabled():
    namespaced_runtime_param_name = RuntimeParameters.namespaced_param_name("OTEL_SDK_DISABLED")
    runtime_param_env_value = json.dumps(
        {"type": RuntimeParameterTypes.BOOLEAN.value, "payload": True}
    )
    os.environ[namespaced_runtime_param_name] = runtime_param_env_value
    yield
    del os.environ[namespaced_runtime_param_name]


class TestOtel:
    @staticmethod
    def make_options(max_workers=1):
        return types.SimpleNamespace(max_workers=max_workers)

    @staticmethod
    def make_options_no_max_workers():
        # Return an object without max_workers attribute
        return types.SimpleNamespace()

    def test_setup_otel_disabled(self, monkeypatch, otel_sdk_disabled):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        options = self.make_options()
        result = setup_otel(RuntimeParameters, options)
        assert result == (None, None, None)

    def test_setup_otel_not_configured(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        options = self.make_options()
        result = setup_otel(RuntimeParameters, options)
        assert result == (None, None, None)

    def test_setup_otel_configured_with_max_workers(self, monkeypatch, otel_sdk_enabled):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        options = self.make_options(max_workers=2)
        with mock.patch(
            "datarobot_drum.drum.common._setup_otel_tracing", return_value="tracer"
        ), mock.patch(
            "datarobot_drum.drum.common._setup_otel_logging", return_value="logger"
        ), mock.patch(
            "datarobot_drum.drum.common._setup_otel_metrics", return_value="metrics"
        ):
            result = setup_otel(RuntimeParameters, options)
            assert result == ("tracer", "metrics", "logger")

    def test_setup_otel_configured_without_max_workers(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        options = self.make_options_no_max_workers()
        with mock.patch(
            "datarobot_drum.drum.common._setup_otel_tracing", return_value="tracer"
        ), mock.patch(
            "datarobot_drum.drum.common._setup_otel_logging", return_value="logger"
        ), mock.patch(
            "datarobot_drum.drum.common._setup_otel_metrics", return_value="metrics"
        ):
            result = setup_otel(RuntimeParameters, options)
            assert result == ("tracer", "metrics", "logger")


class TestChatRequestAttributes:
    def test_extract_chat_request_attributes_text_messages(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }

        attrs = extract_chat_request_attributes(payload)

        assert attrs["gen_ai.request.model"] == "gpt-4o"
        assert attrs["gen_ai.prompt.0.role"] == "system"
        assert attrs["gen_ai.prompt.1.role"] == "user"
        assert attrs["gen_ai.prompt"] == "Hello"
        assert json.loads(attrs["gen_ai.input.messages"]) == [
            {
                "role": "system",
                "parts": [{"type": "text", "content": "You are helpful."}],
            },
            {
                "role": "user",
                "parts": [{"type": "text", "content": "Hello"}],
            },
        ]

    def test_extract_chat_request_attributes_structured_content(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Weather in Paris?"},
                        {"type": "image_url", "image_url": {"url": "https://img"}},
                    ],
                }
            ],
        }

        attrs = extract_chat_request_attributes(payload)

        assert json.loads(attrs["gen_ai.input.messages"]) == [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "Weather in Paris?"},
                    {"type": "image_url", "image_url": {"url": "https://img"}},
                ],
            }
        ]


class TestChatResponseAttributes:
    def test_extract_chat_response_attributes_text_messages(self):
        response = {
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there"},
                    "finish_reason": "stop",
                }
            ],
        }

        attrs = extract_chat_response_attributes(response)

        assert attrs["gen_ai.response.model"] == "gpt-4o"
        assert attrs["gen_ai.completion.0.role"] == "assistant"
        assert attrs["gen_ai.completion.0.content"] == "Hi there"
        assert attrs["gen_ai.completion"] == "Hi there"
        assert json.loads(attrs["gen_ai.output.messages"]) == [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Hi there"}],
                "finish_reason": "stop",
            }
        ]

    def test_extract_chat_response_attributes_structured_content(self):
        response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Result:"},
                            {"type": "tool_call", "name": "get_weather", "arguments": {}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        attrs = extract_chat_response_attributes(response)

        assert json.loads(attrs["gen_ai.output.messages"]) == [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "Result:"},
                    {"type": "tool_call", "name": "get_weather", "arguments": {}},
                ],
                "finish_reason": "tool_calls",
            }
        ]


class TestStreamingChatHelpers:
    def test_reconstruct_chat_response_from_sse(self):
        chunks = [
            'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
            "event: ping\n",
            "data: not-json\n",
            "data: [DONE]\n\n",
        ]

        response = reconstruct_chat_response_from_sse(chunks)

        assert response == {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
        }

    def test_iter_stream_with_span_sets_attributes_and_closes_span(self):
        chunks = [
            'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        parent_span = mock.Mock()
        tracer = mock.Mock()
        stream_span = mock.Mock()
        span_cm = mock.MagicMock()
        context_token = object()

        tracer.start_as_current_span.return_value = span_cm
        span_cm.__enter__.return_value = stream_span

        with patch(
            "datarobot_drum.drum.common.trace.set_span_in_context",
            return_value=context_token,
        ) as set_span_in_context:
            streamed = list(iter_stream_with_span(tracer, parent_span, chunks))

        assert streamed == chunks
        set_span_in_context.assert_called_once_with(parent_span)
        tracer.start_as_current_span.assert_called_once_with(
            "drum.chat.completions.stream", context=context_token
        )
        stream_span.set_attributes.assert_called_once_with(
            {
                "gen_ai.response.model": "gpt-4o",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "Hi",
                "gen_ai.completion": "Hi",
                "gen_ai.output.messages": json.dumps(
                    [
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": "Hi"}],
                            "finish_reason": "stop",
                        }
                    ]
                ),
            }
        )

    def test_iter_stream_with_span_finalizes_on_iterable_error(self):
        class BrokenIterable:
            def __iter__(self):
                yield 'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Par"},"finish_reason":null}]}\n\n'
                raise RuntimeError("stream failed")

        parent_span = mock.Mock()
        tracer = mock.Mock()
        stream_span = mock.Mock()
        span_cm = mock.MagicMock()

        tracer.start_as_current_span.return_value = span_cm
        span_cm.__enter__.return_value = stream_span

        with pytest.raises(RuntimeError, match="stream failed"):
            list(iter_stream_with_span(tracer, parent_span, BrokenIterable()))

        stream_span.set_attributes.assert_called_once()
        response_attrs = stream_span.set_attributes.call_args[0][0]
        assert response_attrs["gen_ai.response.model"] == "gpt-4o"
        assert response_attrs["gen_ai.completion.0.content"] == "Par"

    def test_iter_stream_with_span_finalizes_on_generator_exit(self):
        chunks = [
            'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        parent_span = mock.Mock()
        tracer = mock.Mock()
        stream_span = mock.Mock()
        span_cm = mock.MagicMock()

        tracer.start_as_current_span.return_value = span_cm
        span_cm.__enter__.return_value = stream_span

        stream = iter_stream_with_span(tracer, parent_span, chunks)
        assert next(stream) == chunks[0]
        stream.close()

        # response attributes reconstructed from the single chunk seen before close
        stream_span.set_attributes.assert_called_once()
        response_attrs = stream_span.set_attributes.call_args[0][0]
        assert response_attrs["gen_ai.response.model"] == "gpt-4o"
        assert response_attrs["gen_ai.completion.0.content"] == "Hi"
