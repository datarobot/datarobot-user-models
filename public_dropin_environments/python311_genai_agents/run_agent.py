# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, TextIO, cast
from urllib.parse import urlparse, urlunparse

from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.root_predictors.drum_inline_utils import drum_inline_predictor
from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase,
)
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Span, use_span
from pydantic import TypeAdapter

# Set up tracer provider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

root = logging.getLogger()

CURRENT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_LOG_PATH = CURRENT_DIR / "output.log"
DEFAULT_OUTPUT_JSON_PATH = CURRENT_DIR / "output.json"
ENABLE_STDOUT_REDIRECT = str(os.environ.get("ENABLE_STDOUT_REDIRECT", 0)).lower() in [
    1,
    "1",
    "true",
    "True",
]


def argparse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chat_completion",
        type=str,
        required=True,
        help="OpenAI ChatCompletion dict as json string",
    )
    parser.add_argument(
        "--custom_model_dir",
        type=str,
        required=True,
        help="directory containing custom.py location",
    )
    parser.add_argument(
        "--default_headers",
        type=str,
        default="{}",
        help="OpenAI default_headers as json string",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="json output file location"
    )
    parser.add_argument(
        "--otel_entity_id",
        type=str,
        default=None,
        help="Entity ID, necessary for OpenTelemetry tracing authorization in DataRobot. Format: <entity_type>-<entity_id>",
    )
    parser.add_argument(
        "--otel_attributes",
        type=str,
        default=None,
        help="Custom attributes for tracing. Should be a JSON dictionary.",
    )
    args = parser.parse_args()
    return args


def get_open_port() -> int:
    """Get an open port on the local machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return int(port)


def setup_logging(
    logger: logging.Logger,
    stream: TextIO = sys.stderr,
    log_level: int = logging.INFO,
) -> None:
    logger.setLevel(log_level)

    handler_stream = logging.StreamHandler(stream)
    handler_stream.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler_stream.setFormatter(formatter)

    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    logger.addHandler(handler_stream)


def setup_otel_env_variables(entity_id: str) -> None:
    # do not override if already set
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_HEADERS"
    ):
        root.info(
            "OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_HEADERS already set, skipping"
        )
        return

    datarobot_endpoint = os.environ.get("DATAROBOT_ENDPOINT", "")
    datarobot_api_token = os.environ.get("DATAROBOT_API_TOKEN")
    otlp_endpoint = os.environ.get("DATAROBOT_OTEL_COLLECTOR_BASE_URL", "")

    if not (datarobot_endpoint or otlp_endpoint):
        root.warning(
            "DATAROBOT_ENDPOINT or DATAROBOT_OTEL_COLLECTOR_BASE_URL not set, tracing is disabled"
        )
        return

    if not datarobot_api_token:
        root.warning("DATAROBOT_API_TOKEN not set, tracing is disabled")
        return

    if not otlp_endpoint:
        assert datarobot_endpoint is not None  # mypy
        parsed_url = urlparse(datarobot_endpoint)
        stripped_url = (parsed_url.scheme, parsed_url.netloc, "otel", "", "", "")
        otlp_endpoint = urlunparse(stripped_url)

    otlp_headers = (
        f"X-DataRobot-Api-Key={datarobot_api_token},X-DataRobot-Entity-Id={entity_id}"
    )
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = otlp_headers
    root.info(
        f"Using OTEL_EXPORTER_OTLP_ENDPOINT: {otlp_endpoint} with X-DataRobot-Entity-Id {entity_id}"
    )


def setup_otel_exporter() -> None:
    otlp_exporter = OTLPSpanExporter()
    span_processor = SimpleSpanProcessor(otlp_exporter)  # Do not use batch processor
    trace.get_tracer_provider().add_span_processor(span_processor)  # type: ignore[attr-defined]


def set_otel_attributes(span: Span, attributes: str) -> None:
    try:
        attributes_dict = json.loads(attributes)
    except Exception as e:
        root.error(f"Error parsing OTEL attributes: {e}")
        return

    for key, value in attributes_dict.items():
        span.set_attribute(key, value)


def setup_otel(args: Any) -> Span:
    """
    Setup OTEL tracing and return a span to be parent for the agent run.
    """
    # Setup tracing
    if args.otel_entity_id:
        root.info("Setting up tracing")
        setup_otel_env_variables(args.otel_entity_id)
    else:
        root.info("No OTEL entity ID provided, skipping tracing setup")

    if "OTEL_EXPORTER_OTLP_ENDPOINT" in os.environ:
        root.info("Setting up OTEL exporter")
        setup_otel_exporter()

    span = tracer.start_span("run_agent")

    if args.otel_attributes:
        root.info("Setting up custom OTEL attributes")
        set_otel_attributes(span, args.otel_attributes)

    return span


def execute_drum_inline(
    chat_completion: CompletionCreateParamsBase,
    custom_model_dir: Path,
) -> ChatCompletion:
    root.info("Executing agent as [chat] endpoint. DRUM Inline Executor.")

    root.info("Starting DRUM runner.")
    with drum_inline_predictor(
        target_type=TargetType.AGENTIC_WORKFLOW.value,
        custom_model_dir=custom_model_dir,
        target_name="response",
    ) as predictor:
        root.info("Executing Agent")
        completion = predictor.chat(chat_completion)

    return cast(ChatCompletion, completion)


def construct_prompt(chat_completion: str) -> CompletionCreateParamsBase:
    chat_completion_dict = json.loads(chat_completion)
    model = chat_completion_dict.get("model")
    if model is None or len(str(model)) == 0:
        chat_completion_dict["model"] = "unknown"
    validator = TypeAdapter(CompletionCreateParamsBase)
    validator.validate_python(chat_completion_dict)
    completion_create_params: CompletionCreateParamsBase = CompletionCreateParamsBase(
        **chat_completion_dict  # type: ignore[typeddict-item]
    )
    return completion_create_params


def store_result(result: ChatCompletion, trace_id: str, output_path: Path) -> None:
    root.info(f"Storing result: {output_path}")
    with open(output_path, "w") as fp:
        result_dict = result.model_dump()
        result_dict["trace_id"] = trace_id
        fp.write(json.dumps(result_dict))


def run_agent_procedure(args: Any) -> None:
    # Parse input to fail early if it's not valid
    chat_completion = construct_prompt(args.chat_completion)
    default_headers = json.loads(args.default_headers)
    root.info(f"Chat completion: {chat_completion}")
    root.info(f"Default headers keys: {default_headers.keys()}")

    span = setup_otel(args)
    with use_span(span, end_on_exit=True):
        trace_id = f"{span.context.trace_id:32x}".strip()  # type: ignore[attr-defined]
        root.info(f"Trace id: {trace_id}")

        root.info(f"Executing request in directory {args.custom_model_dir}")
        result = execute_drum_inline(
            chat_completion=chat_completion,
            custom_model_dir=args.custom_model_dir,
        )
        store_result(
            result,
            trace_id,
            Path(args.output_path) if args.output_path else DEFAULT_OUTPUT_JSON_PATH,
        )


def main_stdout_redirect() -> Any:
    """
    This is a wrapper around the main function that redirects stdout and stderr to a file.
    This is used to ensure that logs are written to a file even if the process fails.
    Mainly used in when running the agent in a codespace.
    """
    with open(DEFAULT_OUTPUT_LOG_PATH, "w") as f:
        setup_logging(logger=root, stream=f, log_level=logging.INFO)
        sys.stdout = f
        sys.stderr = f

        print("Parsing args")
        try:
            args = argparse_args()
        except Exception as e:
            root.exception(f"Error parsing arguments: {e}")
            raise
        finally:
            # flush stdout and stderr to ensure all logs are written
            f.flush()

    output_log_path = str(
        Path(args.output_path + ".log") if args.output_path else DEFAULT_OUTPUT_LOG_PATH
    )
    with open(output_log_path, "a") as f:
        # setup logging again: we have a new stream in stderr, so we need a new handler
        setup_logging(logger=root, stream=f, log_level=logging.INFO)
        sys.stdout = f
        sys.stderr = f

        try:
            run_agent_procedure(args)
        except Exception as e:
            root.exception(f"Error executing agent: {e}")
            raise
        finally:
            # flush stdout and stderr to ensure all logs are written
            f.flush()


def main() -> Any:
    setup_logging(logger=root, log_level=logging.INFO)
    root.info("Parsing args")
    args = argparse_args()
    run_agent_procedure(args)
    # flush stdout and stderr to ensure all output is returned to the caller
    sys.stdout.flush()
    sys.stderr.flush()


# Agent execution
if __name__ == "__main__":
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        if ENABLE_STDOUT_REDIRECT:
            main_stdout_redirect()
        else:
            main()
    except Exception:
        pass
    finally:
        # Return to original stdout and stderr otherwise the kernel will fail to flush and
        # hang
        sys.stdout = stdout
        sys.stderr = stderr
