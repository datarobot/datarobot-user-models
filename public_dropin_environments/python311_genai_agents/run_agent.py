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
import sys
from pathlib import Path
from typing import Any, Optional, Union, cast
from urllib.parse import urlparse, urlunparse

import requests
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase,
)
from pydantic import TypeAdapter

root = logging.getLogger()

CURRENT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_LOG_PATH = CURRENT_DIR / "output.log"
DEFAULT_OUTPUT_JSON_PATH = CURRENT_DIR / "output.json"


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
        "--otlp_entity_id", type=str, default=None, help="Entity ID for tracing"
    )
    args = parser.parse_args()
    return args


def setup_logging(
    logger: logging.Logger,
    output_path: Optional[Union[Path, str]] = DEFAULT_OUTPUT_LOG_PATH,
    log_level: Optional[int] = logging.INFO,
    update: Optional[bool] = False,
) -> None:
    log_level = cast(int, log_level)
    output_path = str(output_path)

    logger.setLevel(log_level)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler_stream.setFormatter(formatter)

    if os.path.exists(output_path):
        os.remove(output_path)
    handler_file = logging.FileHandler(output_path)
    handler_file.setLevel(log_level)
    formatter_file = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler_file.setFormatter(formatter_file)

    if update:
        logger.handlers.clear()

    logger.addHandler(handler_stream)
    logger.addHandler(handler_file)


def setup_otlp_env_variables(entity_id: str | None = None) -> None:
    # do not override if already set
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_HEADERS"
    ):
        root.info(
            "OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_HEADERS already set, skipping"
        )
        return

    datarobot_endpoint = os.environ.get("DATAROBOT_ENDPOINT")
    datarobot_api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if not datarobot_endpoint or not datarobot_api_token:
        root.warning(
            "DATAROBOT_ENDPOINT or DATAROBOT_API_TOKEN not set, tracing is disabled"
        )
        return

    parsed_url = urlparse(datarobot_endpoint)
    stripped_url = (parsed_url.scheme, parsed_url.netloc, "otel", "", "", "")
    otlp_endpoint = urlunparse(stripped_url)
    otlp_headers = f"X-DataRobot-Api-Key={datarobot_api_token}"
    if entity_id:
        otlp_headers += f",X-DataRobot-Entity-Id={entity_id}"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = otlp_headers
    root.info(f"Using OTEL_EXPORTER_OTLP_ENDPOINT: {otlp_endpoint}")


def execute_drum(
    chat_completion: CompletionCreateParamsBase,
    default_headers: dict[str, str],
    custom_model_dir: Path,
) -> ChatCompletion:
    root.info("Executing agent as [chat] endpoint. DRUM Executor.")
    root.info("Starting DRUM server.")
    with DrumServerRun(
        target_type=TargetType.TEXT_GENERATION.value,
        labels=None,
        custom_model_dir=custom_model_dir,
        with_error_server=True,
        production=False,
        verbose=True,
        logging_level="info",
        target_name="response",
        wait_for_server_timeout=360,
        port=8191,
        stream_output=True,
    ) as drum_runner:
        root.info("Verifying DRUM server")
        response = requests.get(drum_runner.url_server_address)
        if not response.ok:
            root.error("Server failed to start")
            try:
                root.error(response.text)
            finally:
                raise RuntimeError("Server failed to start")

        # Use a standard OpenAI client to call the DRUM server. This mirrors the behavior of a deployed agent.
        # Using the `chat.completions.create` method ensures the parameters are OpenAI compatible.
        root.info("Executing Agent")
        client = OpenAI(
            base_url=drum_runner.url_server_address,
            api_key="not-required",
            default_headers=default_headers,
            max_retries=0,
        )
        completion = client.chat.completions.create(**chat_completion)
    # Continue outside the context manager to ensure the server is stopped and logs
    # are flushed before we write the output
    return completion


def construct_prompt(chat_completion: str) -> CompletionCreateParamsBase:
    chat_completion_dict = json.loads(chat_completion)
    if "model" not in chat_completion_dict:
        chat_completion_dict["model"] = "unknown"
    validator = TypeAdapter(CompletionCreateParamsBase)
    validator.validate_python(chat_completion_dict)
    completion_create_params: CompletionCreateParamsBase = CompletionCreateParamsBase(
        **chat_completion_dict  # type: ignore[typeddict-item]
    )
    return completion_create_params


def store_result(result: ChatCompletion, output_path: Path) -> None:
    root.info(f"Storing result: {output_path}")
    with open(output_path, "w") as fp:
        fp.write(result.to_json())


def main() -> Any:
    # During failures logs will be dumped to the default output log path
    setup_logging(logger=root, log_level=logging.INFO)
    try:
        root.info("Parsing args")
        try:
            # Attempt to parse arguments and setup logging
            args = argparse_args()

            root.info("Setting up logging")
            output_log_path = str(
                Path(args.output_path + ".log")
                if args.output_path
                else DEFAULT_OUTPUT_LOG_PATH
            )
            setup_logging(
                logger=root,
                output_path=output_log_path,
                log_level=logging.INFO,
                update=True,
            )
        except Exception as e:
            root.exception(f"Error parsing arguments: {e}")
            sys.exit(1)

        # Parse input to fail early if it's not valid
        chat_completion = construct_prompt(args.chat_completion)
        default_headers = json.loads(args.default_headers)
        root.info(f"Chat completion: {chat_completion}")
        root.info(f"Default headers: {default_headers}")

        # Setup tracing
        root.info("Setting up tracing")
        setup_otlp_env_variables(args.otlp_entity_id)

        root.info(f"Executing request in directory {args.custom_model_dir}")
        result = execute_drum(
            chat_completion=chat_completion,
            default_headers=default_headers,
            custom_model_dir=args.custom_model_dir,
        )
        root.info(f"Result: {result}")
        store_result(
            result,
            Path(args.output_path) if args.output_path else DEFAULT_OUTPUT_JSON_PATH,
        )
    except Exception as e:
        root.exception(f"Error executing agent: {e}")


# Agent execution
if __name__ == "__main__":
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        main()
    except Exception:
        pass
    finally:
        # Return to original stdout and stderr otherwise the kernel will fail to flush and
        # hang
        sys.stdout = stdout
        sys.stderr = stderr
