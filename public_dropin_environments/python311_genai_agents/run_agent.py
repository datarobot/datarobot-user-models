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
from typing import cast

import requests
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun
from openai import OpenAI
from openai.types.chat import ChatCompletion

root = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--chat_completion",
    type=str,
    default="{}",
    help="OpenAI ChatCompletion dict as json string",
)
parser.add_argument(
    "--default_headers",
    type=str,
    default="{}",
    help="OpenAI default_headers as json string",
)
parser.add_argument(
    "--custom_model_dir",
    type=str,
    default="",
    help="directory containing custom.py location",
)
parser.add_argument(
    "--output_path", type=str, default="", help="json output file location"
)
args = parser.parse_args()


def setup_logging(
    logger: logging.Logger, output_path: str, log_level: int = logging.INFO
) -> None:
    if len(output_path) == 0:
        output_path = "output.log"
    else:
        output_path = f"{output_path}.log"

    logger.setLevel(log_level)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setLevel(log_level)
    formatter = logging.Formatter("%(message)s")
    handler_stream.setFormatter(formatter)

    if os.path.exists(output_path):
        os.remove(output_path)
    handler_file = logging.FileHandler(output_path)
    handler_file.setLevel(log_level)
    formatter_file = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler_file.setFormatter(formatter_file)

    logger.addHandler(handler_stream)
    logger.addHandler(handler_file)


def execute_drum(
    chat_completion: str, default_headers: str, custom_model_dir: str, output_path: str
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
                root.error(response.json())
            finally:
                raise RuntimeError("Server failed to start")

        root.info("Parsing OpenAI request")
        completion_params = json.loads(chat_completion)
        header_params = json.loads(default_headers)

        # Use a standard OpenAI client to call the DRUM server. This mirrors the behavior of a deployed agent.
        # Using the `chat.completions.create` method ensures the parameters are OpenAI compatible.
        root.info("Executing Agent")
        client = OpenAI(
            base_url=drum_runner.url_server_address,
            api_key="not-required",
            default_headers=header_params,
            max_retries=0,
        )
        completion = client.chat.completions.create(**completion_params)

    # Continue outside the context manager to ensure the server is stopped and logs
    # are flushed before we write the output
    root.info(f"Storing result: {output_path}")
    if len(output_path) == 0:
        output_path = os.path.join(custom_model_dir, "output.json")
    with open(output_path, "w") as fp:
        fp.write(completion.to_json())

    root.info(completion.to_json())
    return cast(ChatCompletion, completion)


# Agent execution
if len(args.custom_model_dir) == 0:
    args.custom_model_dir = os.path.join(os.getcwd(), "custom_model")
setup_logging(logger=root, output_path=args.output_path, log_level=logging.INFO)
result = execute_drum(
    chat_completion=args.chat_completion,
    default_headers=args.default_headers,
    custom_model_dir=args.custom_model_dir,
    output_path=args.output_path,
)
