"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import json
import logging
import os
import sys
import trafaret as t
from contextvars import ContextVar
from urllib.parse import urlparse, urlunparse

from contextlib import contextmanager
from pathlib import Path

from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    MODEL_CONFIG_FILENAME,
    PredictionServerMimetypes,
    PayloadFormat,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from opentelemetry import trace, context, metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

from opentelemetry.sdk.metrics import Counter
from opentelemetry.sdk.metrics import Histogram
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics import ObservableCounter
from opentelemetry.sdk.metrics.export import AggregationTemporality
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

ctx_request_id = ContextVar("request_id")


@contextmanager
def reroute_stdout_to_stderr():
    keep = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = keep


@contextmanager
def verbose_stdout(verbose):
    new_target = sys.stdout
    old_target = sys.stdout
    if not verbose:
        new_target = open(os.devnull, "w")
        sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def request_id_filter(record: logging.LogRecord):
    """
    A filter which injects context-specific information into logs and ensures
    that only information for a specific webapp is included in its log
    """
    request_id = ctx_request_id.get(None)
    record.context_data = {"request_id": request_id} if request_id else ""
    return True


def config_logging():
    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(request_id_filter)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(name)s:  %(message)s %(context_data)s",
        handlers=[stream_handler],
    )


def get_drum_logger(logger_name):
    """Provides a logger with `drum.` prefix."""
    return logging.getLogger(".".join([LOGGER_NAME_PREFIX, logger_name]))


def get_metadata(options):
    code_dir = Path(options.code_dir)
    if options.model_config is None:
        raise DrumCommonException(
            "You must have a file with the name {} in the directory {}. \n"
            "You don't. \nWhat you do have is these files: \n{} ".format(
                MODEL_CONFIG_FILENAME, code_dir, os.listdir(code_dir)
            )
        )
    return options.model_config


class SupportedPayloadFormats:
    def __init__(self):
        self._formats = {}
        self._mimetype_to_payload_format = {
            None: PayloadFormat.CSV,
            PredictionServerMimetypes.EMPTY: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_CSV: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_PLAIN: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_MTX: PayloadFormat.MTX,
        }

    def add(self, payload_format, format_version=None):
        self._formats[payload_format] = format_version

    def is_mimetype_supported(self, mimetype):
        payload_format = self._mimetype_to_payload_format.get(mimetype)
        if payload_format is None:
            return False

        return payload_format in self._formats

    def __iter__(self):
        for payload_format, format_version in self._formats.items():
            yield payload_format, format_version


def to_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return t.ToBool().check(value)


FIT_METADATA_FILENAME = "fit_runtime_data.json"


def make_otel_endpoint(datarobot_endpoint):
    parsed_url = urlparse(datarobot_endpoint)
    stripped_url = (parsed_url.scheme, parsed_url.netloc, "otel", "", "", "")
    result = urlunparse(stripped_url)
    return result


class _ExcludeOtelLogsFilter(logging.Filter):
    """A logging filter to exclude logs from the opentelemetry library."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("opentelemetry")


def _setup_otel_logging(resource, multiprocessing=False):
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    exporter = OTLPLogExporter()
    if multiprocessing:
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    else:
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
    # Remove own logs to avoid infinite recursion if endpoint is not available
    handler.addFilter(_ExcludeOtelLogsFilter())
    logging.getLogger().addHandler(handler)
    return logger_provider


def _setup_otel_metrics(resource):
    # OTEL SDK default termporarity is CUMULATIVE, but this is rarely what users
    # actualy want to work with, so here we switch default. Also in case of delta
    # PeriodicExportingMetricReader does not spam collector with same data.
    preferred_temporality = {
        Counter: AggregationTemporality.DELTA,
        Histogram: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
    }
    metric_exporter = OTLPMetricExporter(preferred_temporality=preferred_temporality)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    metric_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
    metrics.set_meter_provider(metric_provider)
    return metric_provider


def _setup_otel_tracing(resource, multiprocessing=False):
    otlp_exporter = OTLPSpanExporter()
    trace_provider = TracerProvider(resource=resource)
    if multiprocessing:
        trace_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
    else:
        trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(trace_provider)
    return trace_provider


def setup_otel(runtime_parameters, options):
    """Setups OTEL tracer.

    OTEL is configured by OTEL_EXPORTER_OTLP_ENDPOINT and
    OTEL_EXPORTER_OTLP_HEADERS env vars set by DR.

    Parameters
    ----------
    runtime_parameters: Type[RuntimeParameters] class handles runtime parameters for custom modes
        used to check if OTEL configuration from user.
    options: argparse.Namespace: object obtained from argparser filled with user supplied
        command argumetns
    Returns
    -------
    (TracerProvider, MetricProvider)
    """
    log = get_drum_logger("setup_otel")

    # Can be used to disable OTEL reporting from env var parameters
    # https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
    if runtime_parameters.has("OTEL_SDK_DISABLED") and runtime_parameters.get("OTEL_SDK_DISABLED"):
        log.info("OTEL explictly disabled")
        return (None, None, None)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        log.info("OTEL is not configured")
        return (None, None, None)

    # In case of NIM flask server is configured to run in multiprocessing
    # mode that uses fork. Since BatchSpanProcessor start background thread
    # with bunch of locks, OTEL simply deadlocks and does not offlooad any
    # spans. Even if we start BatchSpanProcessor per fork, batches often
    # missing due to process exits before all data offloaded. In forking
    # case we use SimpleSpanProcessor (mostly NIMs) otherwise BatchSpanProcessor
    # (most frequent case)
    multiprocessing = False
    if hasattr(options, "max_workers") and options.max_workers is not None:
        multiprocessing = options.max_workers > 1

    resource = Resource.create()
    trace_provider = _setup_otel_tracing(resource=resource, multiprocessing=multiprocessing)
    logger_provider = _setup_otel_logging(resource=resource, multiprocessing=multiprocessing)
    metric_provider = _setup_otel_metrics(resource=resource)

    log.info(f"OTEL is configured with endpoint: {endpoint}")
    return trace_provider, metric_provider, logger_provider


@contextmanager
def otel_context(tracer, span_name, carrier):
    ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    token = context.attach(ctx)
    try:
        with tracer.start_as_current_span(span_name) as span:
            yield span
    finally:
        context.detach(token)


def _normalize_chat_content_to_parts(content):
    parts = []
    if isinstance(content, str):
        parts.append({"type": "text", "content": content})
    elif isinstance(content, list):
        for content_part in content:
            if not isinstance(content_part, dict):
                continue
            content_part_type = content_part.get("type")
            if content_part_type == "text" and "text" in content_part:
                parts.append({"type": "text", "content": content_part["text"]})
            else:
                parts.append(content_part)
    elif content is not None:
        parts.append({"type": "text", "content": str(content)})
    return parts


def extract_chat_request_attributes(completion_params):
    """Extracts otel related attributes from chat request payload

    Used to populate span with relevant monitoring attriubtes.
    """
    completion_params = completion_params or {}
    logger = get_drum_logger(__name__)

    attributes = {}
    attributes["gen_ai.request.model"] = completion_params.get("model")
    gen_ai_input_messages = []
    for i, m in enumerate(completion_params.get("messages", [])):
        if not isinstance(m, dict):
            continue

        role = m.get("role")
        content = m.get("content")

        attributes[f"gen_ai.prompt.{i}.role"] = role
        attributes[f"gen_ai.prompt.{i}.content"] = content
        # last prompt wins
        attributes["gen_ai.prompt"] = content

        try:
            parts = _normalize_chat_content_to_parts(content)
        except Exception:
            logger.exception(f"Error normalizing chat content for span attributes")
            continue

        message = {"role": role}
        if parts:
            message["parts"] = parts
        gen_ai_input_messages.append(message)

    # Spans do not always support native structured values, so serialize to JSON.
    if gen_ai_input_messages:
        attributes["gen_ai.input.messages"] = json.dumps(gen_ai_input_messages)

    return attributes


def extract_request_headers(request_headers):
    attributes = {}
    consumer_id = request_headers.get("X-DataRobot-Consumer-Id")
    consumer_type = request_headers.get("X-DataRobot-Consumer-Type")

    if consumer_id:
        attributes["gen_ai.request.consumer_id"] = consumer_id

    if consumer_type:
        attributes["gen_ai.request.consumer_type"] = consumer_type

    return attributes


def extract_chat_response_attributes(response):
    """Extracts otel related attributes from chat response.

    Used to populate span with relevant monitoring attriubtes.
    """
    response = response or {}

    attributes = {}
    attributes["gen_ai.response.model"] = response.get("model")
    gen_ai_output_messages = []
    logger = get_drum_logger(__name__)

    for i, c in enumerate(response.get("choices", [])):
        if not isinstance(c, dict):
            continue

        m = c.get("message", {})
        if not isinstance(m, dict):
            m = {}

        role = m.get("role")
        content = m.get("content")

        attributes[f"gen_ai.completion.{i}.role"] = role
        attributes[f"gen_ai.completion.{i}.content"] = content
        # last completion wins
        attributes["gen_ai.completion"] = content

        try:
            parts = _normalize_chat_content_to_parts(content)
        except Exception:
            logger.exception(f"Error normalizing chat content for span attributes")
            continue

        message = {"role": role}
        if parts:
            message["parts"] = parts

        finish_reason = c.get("finish_reason")
        if finish_reason is not None:
            message["finish_reason"] = finish_reason

        gen_ai_output_messages.append(message)

    # Spans do not always support native structured values, so serialize to JSON.
    if gen_ai_output_messages:
        attributes["gen_ai.output.messages"] = json.dumps(gen_ai_output_messages)

    return attributes


def reconstruct_chat_response_from_sse(chunks):
    """Reconstruct a chat completion response dict from collected SSE event chunks."""
    model = None
    choices_content = {}  # index -> accumulated content
    choices_role = {}  # index -> role
    choices_finish_reason = {}  # index -> finish_reason

    for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                parsed = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            if model is None:
                model = parsed.get("model")
            for choice in parsed.get("choices", []):
                idx = choice.get("index", 0)
                delta = choice.get("delta", {})
                role = delta.get("role")
                if role:
                    choices_role[idx] = role
                content = delta.get("content")
                if content:
                    choices_content[idx] = choices_content.get(idx, "") + content
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    choices_finish_reason[idx] = finish_reason

    all_indices = sorted(
        set(list(choices_content) + list(choices_role) + list(choices_finish_reason))
    )
    choices = [
        {
            "message": {
                "role": choices_role.get(idx),
                "content": choices_content.get(idx),
            },
            "finish_reason": choices_finish_reason.get(idx),
        }
        for idx in all_indices
    ]
    return {"model": model, "choices": choices}


def iter_stream_with_span(tracer, parent_span, iterable):
    """Yield chunks from a streaming response inside a dedicated child span.

    The child span uses the request span as parent so stream lifecycle and
    response attributes are recorded separately from the initial request work.
    Attributes are set in a finally block so they are always recorded regardless
    of how the generator terminates (normal exhaustion, error, or GeneratorExit
    on client disconnect).
    """
    parent_context = trace.set_span_in_context(parent_span)
    with tracer.start_as_current_span(
        "drum.chat.completions.stream", context=parent_context
    ) as stream_span:
        chunks = []
        try:
            for chunk in iterable:
                chunks.append(chunk)
                yield chunk
        finally:
            try:
                reconstructed = reconstruct_chat_response_from_sse(chunks)
                stream_span.set_attributes(extract_chat_response_attributes(reconstructed))
            except Exception:
                logger = get_drum_logger(__name__)
                logger.exception(f"Error reconstructing chat response for span attributes")
