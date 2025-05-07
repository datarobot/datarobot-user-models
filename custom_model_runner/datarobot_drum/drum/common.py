"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os
import sys
from contextvars import ContextVar
from distutils.util import strtobool
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
from opentelemetry import trace, context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
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
    return strtobool(value)


FIT_METADATA_FILENAME = "fit_runtime_data.json"


def make_otel_endpoint(datarobot_endpoint):
    parsed_url = urlparse(datarobot_endpoint)
    stripped_url = (parsed_url.scheme, parsed_url.netloc, "otel", "", "", "")
    result = urlunparse(stripped_url)
    return result


def setup_tracer(runtime_parameters):
    # OTEL disabled by default for now.
    if not (
        runtime_parameters.has("OTEL_SDK_ENABLED") and runtime_parameters.get("OTEL_SDK_ENABLED")
    ):
        return
    # if deployment_id is not found, most likely this is custom model
    # testing
    deployment_id = os.environ.get("MLOPS_DEPLOYMENT_ID", os.environ.get("DEPLOYMENT_ID"))
    if not deployment_id:
        return

    service_name = f"deployment-{deployment_id}"
    resource = Resource.create(
        {
            "service.name": service_name,
            "datarobot.deployment_id": deployment_id,
        }
    )
    key = os.environ.get("DATAROBOT_API_TOKEN")
    datarobot_endpoint = os.environ.get("DATAROBOT_ENDPOINT")
    if not key or not datarobot_endpoint:
        return
    endpoint = make_otel_endpoint(datarobot_endpoint)

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    headers = {
        "Authorization": f"Bearer {key}",
        "X-DataRobot-Entity-Id": f"entity=deployment; id={deployment_id};",
    }
    otlp_exporter = OTLPSpanExporter(headers=headers)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(provider)


@contextmanager
def otel_context(tracer, span_name, carrier):
    ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    token = context.attach(ctx)
    try:
        with tracer.start_as_current_span(span_name) as span:
            yield span
    finally:
        context.detach(token)
