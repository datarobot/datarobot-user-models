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
from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParametersLoader
from opentelemetry import trace, context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
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


def setup_tracer(runtime_parameters, options):
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
    TracerProvider
    """
    log = get_drum_logger("setup_tracer")

    # Can be used to disable OTEL reporting from env var parameters
    # https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
    if runtime_parameters.has("OTEL_SDK_DISABLED") and os.environ.get("OTEL_SDK_DISABLED"):
        log.info("Tracing explictly disabled")
        return

    main_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    trace_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not main_endpoint and not trace_endpoint:
        log.info("Tracing is not configured")
        return

    resource = Resource.create()
    otlp_exporter = OTLPSpanExporter()
    provider = TracerProvider(resource=resource)

    # In case of NIM flask server is configured to run in multiprocessing
    # mode that uses fork. Since BatchSpanProcessor start background thread
    # with bunch of locks, OTEL simply deadlocks and does not offlooad any
    # spans. Even if we start BatchSpanProcessor per fork, batches often
    # missing due to process exits before all data offloaded. In forking
    # case we use SimpleSpanProcessor (mostly NIMs) otherwise BatchSpanProcessor
    # (most frequent case)
    if options.max_workers > 1:
        provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
    else:
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)

    endpoint = main_endpoint or trace_endpoint
    log.info(f"Tracing is configured with endpoint: {endpoint}")
    return provider


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


def setup_required_environment_variables(options):
    if "runtime_params_file" in options and options.runtime_params_file:
        loader = RuntimeParametersLoader(options.runtime_params_file, options.code_dir)
        loader.setup_environment_variables()

    if "lazy_loading_file" in options and options.lazy_loading_file:
        LazyLoadingHandler.setup_environment_variables_from_values_file(options.lazy_loading_file)
