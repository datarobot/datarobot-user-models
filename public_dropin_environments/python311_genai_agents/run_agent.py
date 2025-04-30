# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# pylint: skip-file

import argparse
import json
import os

import requests
from custom import chat
from custom import load_model
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

parser = argparse.ArgumentParser()
parser.add_argument("--store_output", action="store_true", help="Store output in a file")
parser.add_argument(
    "--use_drum", action="store_true", help="Use DRUM for execution instead of direct execution"
)
parser.add_argument("--user_prompt", type=str, default="", help="user_prompt for chat endpoint")
parser.add_argument("--extra_body", type=str, default="", help="extra_body for chat endpoint")
args = parser.parse_args()


class RunAgent:
    """
    This class is responsible for running the agent. It can run in two modes:
        - :code:`execute_direct`: Directly using the chat endpoint.
        - :code:`execute_drum`: Using the DRUM server.
    """

    @property
    def code_dir(self):
        return "/home/notebooks/storage/"

    @staticmethod
    def construct_prompt(user_prompt, extra_body, merge_extra_body=True):
        extra_body_params = json.loads(extra_body) if extra_body else {}
        completion_create_params = CompletionCreateParamsNonStreaming(
            model="datarobot-deployed-llm",
            messages=[
                ChatCompletionSystemMessageParam(
                    content="You are a helpful assistant",
                    role="system",
                ),
                ChatCompletionUserMessageParam(
                    content=user_prompt,
                    role="user",
                ),
            ],
            n=1,
            temperature=0.01,
            extra_body=extra_body_params,
        )
        if merge_extra_body:
            completion_create_params.update(**extra_body_params)
        return completion_create_params

    def execute_direct(self, user_prompt, extra_body):
        print("Executing agent as [chat] endpoint. Local Executor.")
        completion_create_params = self.construct_prompt(user_prompt, extra_body)

        # Use direct execution of the agent. This is more straightforward to debug agent code related issues.
        model = load_model(code_dir=self.code_dir)
        chat_result = chat(completion_create_params, model)

        return chat_result

    def execute_drum(self, user_prompt, extra_body):
        print("Executing agent as [chat] endpoint. DRUM Executor.")
        print("NOTE: Realtime logging will be delayed in terminal and displayed after execution.")

        # This logic spools an ephemeral DRUM server to run the agent
        os.environ[
            "MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS"
        ] = '{"type": "numeric", "payload": 10}'
        os.environ["MLOPS_RUNTIME_PARAM_model"] = json.dumps(
            {
                "type": "string",
                "payload": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            }
        )
        os.environ[
            "MLOPS_RUNTIME_PARAM_prompt_column_name"
        ] = '{"type":"string","payload":"user_prompt"}'
        os.environ[
            "MLOPS_RUNTIME_PARAM_system_prompt"
        ] = '{"type":"string","payload":"You are a helpful assistant"}'
        os.environ["MLOPS_RUNTIME_PARAM_temperature"] = '{"type":"numeric","payload":0.01}'
        with DrumServerRun(
            target_type=TargetType.TEXT_GENERATION.value,
            labels=None,
            custom_model_dir=self.code_dir,
            with_error_server=True,
            production=False,
            logging_level="info",
            target_name="response",
            wait_for_server_timeout=360,
            port=8191,
        ) as drum_runner:
            response = requests.get(drum_runner.url_server_address)
            if not response.ok:
                raise RuntimeError("Server failed to start")

            # Use a standard OpenAI client to call the DRUM server. This mirrors the behavior of a deployed agent.
            client = OpenAI(
                base_url=drum_runner.url_server_address, api_key="not-required", max_retries=0
            )
            completion_create_params = self.construct_prompt(
                user_prompt, extra_body, merge_extra_body=False
            )
            completion = client.chat.completions.create(**completion_create_params)

            return completion

    def store_output(self, chat_result):
        with open(os.path.join(self.code_dir, "output.json"), "w") as fp:
            fp.write(chat_result.to_json())


# This is the main entry point for the script
runner = RunAgent()
if args.use_drum:
    result = runner.execute_drum(user_prompt=args.user_prompt, extra_body=args.extra_body)
else:
    result = runner.execute_direct(user_prompt=args.user_prompt, extra_body=args.extra_body)

# Store results to file if requested
if args.store_output:
    runner.store_output(result)
