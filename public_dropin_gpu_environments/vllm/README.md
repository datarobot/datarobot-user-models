# vLLM for LLMs Drop-In Environment

This drop-in environment contains the vLLM OpenAI compatible inference server with support for large language models (LLMs).

## Instructions

1. From the terminal, run `tar -czvf vllm_dropin.tar.gz -C /path/to/public_dropin_environments/vllm/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created in step 1.

### Creating models for this environment

This environment makes the following assumption about your serialized model:
- The data sent to custom model can be used to make predictions without additional pre-processing
- There is a single model and model version present
- Model is expected to have the `textGeneration` target type.

### Supported Runtime Parameters

| Parameter Name | Required | Description |
| --- | --- | --- |
| `model` | No | Name of the model to load from HuggingFace. This param is not required if your `custom.py:load_model` hook is downloading the model artifacts. |
| `HuggingFaceToken` | No | Auth used to download model artifacts if downloading from HuggingFace. |
| `max_model_len` | No | Model context length. If unspecified, will be automatically derived from the model config. |
| `gpu_memory_utilization` | No | The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use the default value of 0.9. |
| `trust_remote_code` | No | Trust remote code from HuggingFace. |
| `system_prompt` | No | Prompt to assign as the `system` role to the LLM. |
| `prompt_column_name` | No | Name of the input column we expect to find the LLM prompt. |
| `max_tokens` | No | The maximum number of tokens that can be generated in the chat completion. This value can be used to control costs for text generated via API. |
| `n` | No | How many chat completion choices to generate for each input message. Note that you will be charged based on the number of generated tokens across all of the choices. Keep n as 1 to minimize costs. |
| `temperature` | No | The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit. |
| `verifySSL` | No | If we need to verify TLS whe communicating status back to DataRobot |

#### Additional configuration

The vLLM OpenAI Inference server supports a multitude of command line arguments. The most commonly used ones have been exposed as runtime parameters; however, for all other options, you can pass them by including an `engine_config.json` file in the root of your custom model. The contents of the file must conform to the following schema:
```yaml
$schema: http://json-schema.org/schema#
title: DataRobot vLLM Engine Config
description: Schema for vLLM config file
type: object
additionalProperties: false

properties:
  args:
    type: array
    description: list of command line flags to be passed to vllm.entrypoints.openai.api_server
```
If arguments are present in this file that have corresponding runtime parameters, the values in this file **take precedence** over the runtime parameters.

### Air Gapped Deployment
It is possible to use this environment in a cluster disconnected from the internet. The environment supports the presence of a `custom.py` file that contains a `load_model` hook. This hook can perform any operation (such as downloading the model artifacts from a local blob storage) and as long as it saves the files to `${CODE_DIR}/vllm/` then the server will load those artifacts instead of attempting to fetch files from HuggingFace.

### Example model-metadata.yaml

```yaml
name: vLLM Server Example
type: inference
targetType: textgeneration

runtimeParameterDefinitions:
  - fieldName: model
    type: string
    description: Name of the model to download from HuggingFace (or local path to pre-downloaded model).

  - fieldName: HuggingFaceToken
    type: credential
    credentialType: api_token
    description: |-
      Access Token from HuggingFace (https://huggingface.co/settings/tokens). Only
      required if the model was not downloaded in the `custom.py:load_model` function.

  - fieldName: max_model_len
    type: string
    description: Model context length. If unspecified, will be automatically derived from the model config.

  - fieldName: max_tokens
    type: numeric
    defaultValue: 1024
    minValue: 1
    description: max number of symbols in response

  - fieldName: system_prompt
    type: string
    defaultValue: You are a helpful AI assistant. Keep short answers of no more than 2 sentences.
    description: instructions to the model, to set the tone of model completions

  - fieldName: prompt_column_name
    type: string
    defaultValue: promptText
    description: column with user's prompt (each row is a separate completion request)
```

### Example engine_config.json
```json
{
  "args": ["--model", "tiiuae/falcon-7b-instruct", "--chat-template", "/opt/code/template_falcon.jinja"]

}
```
