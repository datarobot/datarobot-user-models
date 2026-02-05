"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

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
    """Setup OpenTelemetry logging with gevent-compatible thread handling.

    When gevent monkey patches threading, BatchLogRecordProcessor's background
    threads become greenlets. In Python 3.12, this causes KeyError during shutdown
    when greenlets try to remove themselves from threading._active dict.

    Solution: Temporarily restore real threading.Thread during processor creation.
    """
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    exporter = OTLPLogExporter()

    if multiprocessing:
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    else:
        # Use real OS thread for BatchLogRecordProcessor background thread
        # to prevent KeyError during shutdown in Python 3.12 + gevent
        processor = _create_processor_with_real_thread(lambda: BatchLogRecordProcessor(exporter))
        logger_provider.add_log_record_processor(processor)

    handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
    # Remove own logs to avoid infinite recursion if endpoint is not available
    handler.addFilter(_ExcludeOtelLogsFilter())
    logging.getLogger().addHandler(handler)
    return logger_provider


def _setup_otel_metrics(resource):
    """Setup OpenTelemetry metrics with gevent-compatible thread handling.

    PeriodicExportingMetricReader creates a background thread. When gevent
    monkey patches threading, this becomes a greenlet, causing KeyError
    during shutdown in Python 3.12.

    Solution: Temporarily restore real threading.Thread during reader creation.
    """
    # OTEL SDK default termporarity is CUMULATIVE, but this is rarely what users
    # actualy want to work with, so here we switch default. Also in case of delta
    # PeriodicExportingMetricReader does not spam collector with same data.
    preferred_temporality = {
        Counter: AggregationTemporality.DELTA,
        Histogram: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
    }
    metric_exporter = OTLPMetricExporter(preferred_temporality=preferred_temporality)

    # Use real OS thread for PeriodicExportingMetricReader background thread
    # to prevent KeyError during shutdown in Python 3.12 + gevent
    metric_reader = _create_processor_with_real_thread(
        lambda: PeriodicExportingMetricReader(metric_exporter)
    )

    metric_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
    metrics.set_meter_provider(metric_provider)
    return metric_provider


def _setup_otel_tracing(resource, multiprocessing=False):
    """Setup OpenTelemetry tracing with gevent-compatible thread handling.

    BatchSpanProcessor creates a background thread. When gevent monkey patches
    threading, this becomes a greenlet, causing KeyError during shutdown in
    Python 3.12.

    Solution: Temporarily restore real threading.Thread during processor creation.
    """
    otlp_exporter = OTLPSpanExporter()
    trace_provider = TracerProvider(resource=resource)

    if multiprocessing:
        trace_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
    else:
        # Use real OS thread for BatchSpanProcessor background thread
        # to prevent KeyError during shutdown in Python 3.12 + gevent
        processor = _create_processor_with_real_thread(lambda: BatchSpanProcessor(otlp_exporter))
        trace_provider.add_span_processor(processor)

    trace.set_tracer_provider(trace_provider)
    return trace_provider


def _create_processor_with_real_thread(factory_func):
    """Create OTEL processor/reader using real OS threads instead of gevent greenlets.

    OpenTelemetry's BatchSpanProcessor, BatchLogRecordProcessor, and
    PeriodicExportingMetricReader create background threads during __init__.
    When gevent has monkey-patched threading, these become greenlets.

    In Python 3.12, greenlets trying to clean up as threads cause:
        KeyError: <greenlet_id> in threading._delete()

    This function temporarily restores the real threading.Thread class
    during processor/reader creation to ensure background threads are real
    OS threads, preventing the KeyError.

    Args:
        factory_func: Callable that creates the processor/reader

    Returns:
        The created processor/reader with real OS thread
    """
    import threading

    # Check if gevent has monkey-patched threading
    try:
        from gevent import monkey

        if not monkey.is_module_patched("threading"):
            # Not patched, just create normally
            return factory_func()
    except ImportError:
        # gevent not installed, create normally
        return factory_func()

    # Gevent is active - temporarily restore real threading.Thread
    original_thread_class = threading.Thread

    try:
        # Get the real (unpatched) Thread class
        real_thread = monkey.get_original("threading", "Thread")
        if not real_thread:
            # Unable to get original Thread class, log warning and create normally
            # This shouldn't happen, but better to be safe
            import logging

            log = logging.getLogger(__name__)
            log.warning(
                "Could not get original threading.Thread from gevent. "
                "OpenTelemetry background threads may be greenlets, "
                "which could cause KeyError during shutdown in Python 3.12."
            )
            return factory_func()

        # Replace with real Thread during processor creation
        threading.Thread = real_thread

        # Create the processor - its background thread will be a real OS thread
        return factory_func()

    finally:
        # Always restore gevent's patched Thread class
        threading.Thread = original_thread_class


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


def extract_chat_request_attributes(completion_params):
    """Extracts otel related attributes from chat request payload

    Used to populate span with relevant monitoring attriubtes.
    """
    attributes = {}
    attributes["gen_ai.request.model"] = completion_params.get("model")
    for i, m in enumerate(completion_params.get("messages", [])):
        attributes[f"gen_ai.prompt.{i}.role"] = m.get("role")
        attributes[f"gen_ai.prompt.{i}.content"] = m.get("content")
        # last promt wins
        attributes["gen_ai.prompt"] = m.get("content")
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
    attributes = {}
    attributes["gen_ai.response.model"] = response.get("model")
    for i, c in enumerate(response.get("choices", [])):
        m = c.get("message", {})
        attributes[f"gen_ai.completion.{i}.role"] = m.get("role")
        attributes[f"gen_ai.completion.{i}.content"] = m.get("content")
        # last completion wins
        attributes["gen_ai.completion"] = m.get("content")
    return attributes
