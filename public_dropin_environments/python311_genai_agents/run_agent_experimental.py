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

# pylint: skip-file

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, cast

import requests
import datarobot_drum
from datarobot_drum.drum.common import setup_tracer
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParameters
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.enum import TargetType
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)

root = logging.getLogger()

CURRENT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_PATH = CURRENT_DIR / "output.log"

parser = argparse.ArgumentParser()
parser.add_argument("--chat_completion", type=str, default="", help="json string of chat completion")
parser.add_argument(
    "--custom_model_dir",
    type=str,
    default="",
    help="directory containing custom.py location",
)
parser.add_argument("--output_path", type=str, default="", help="json output file location")


def setup_logging(logger: logging.Logger, log_level: int = logging.INFO) -> None:
    logger.setLevel(log_level)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setLevel(log_level)
    formatter_stream = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler_stream.setFormatter(formatter_stream)

    logger.addHandler(handler_stream)


def construct_prompt(chat_completion: str) -> Any:
    chat_completion = json.loads(chat_completion) if chat_completion else {}
    completion_create_params = CompletionCreateParamsNonStreaming(
        **chat_completion
    )
    return completion_create_params


def execute_drum(
    chat_completion: str, custom_model_dir: str, output_path: str
) -> ChatCompletion:
    #root.info("Setting up tracer")
    #setup_tracer(RuntimeParameters)
    root.info("Setting up model adapter")
    os.environ["TARGET_NAME"] = "response"
    model_adapter = PythonModelAdapter(custom_model_dir, target_type=TargetType.AGENTIC_WORKFLOW)
    root.info("Model adapter set up. Loading hooks.")
    model_adapter.load_custom_hooks()
    root.info("Hooks loaded.")

    # Use a standard OpenAI client to call the DRUM server. This mirrors the behavior of a deployed agent.
    root.info("Building prompt.")
    completion_create_params = construct_prompt(chat_completion)

    root.info("Executing Agent.")
    completion = model_adapter.chat(completion_create_params, model=None, association_id=None)

    # Continue outside the context manager to ensure the server is stopped and logs
    # are flushed before we write the output
    root.info(f"Storing result: {output_path}")
    if len(output_path) == 0:
        output_path = os.path.join(custom_model_dir, "output.json")
    with open(output_path, "w") as fp:
        fp.write(completion.to_json())

    root.info(completion.to_json())
    return cast(ChatCompletion, completion)


if __name__ == "__main__":
    with open(DEFAULT_OUTPUT_PATH, "a") as f:
        sys.stdout = f
        sys.stderr = f 
        print("Parsing args")
        args = parser.parse_args()
    
    output_log_path = args.output_path + ".log" if args.output_path else DEFAULT_OUTPUT_PATH
    with open(output_log_path, "a") as f:
        sys.stdout = f
        sys.stderr = f
        
        print("Setting up logging")
        setup_logging(logger=root, log_level=logging.INFO)
        if len(args.custom_model_dir) == 0:
            args.custom_model_dir = CURRENT_DIR / "custom_model"
        # Agent execution
        root.info(f"Executing agent at {args.custom_model_dir}")
        try:
            result = execute_drum(
                chat_completion=args.chat_completion,
                    custom_model_dir=args.custom_model_dir,
                    output_path=args.output_path,
                )
        except Exception as e:
            root.exception(f"Error executing agent: {e}")
            sys.exit(1)
    
    sys.exit(0)
