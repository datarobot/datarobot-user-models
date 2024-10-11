# NVIDIA NIM for LLMs Drop-In Environment

This drop-in environment contains the NVIDIA NIM server with support for large language models (LLMs).

## Instructions

1. Build the image, run `docker build -t nim:latest /path/to/public_dropin_environments/nim_llm/`
   - You will need to authenticate to fetch the base image. See the [official instructions](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#generate-an-api-key) for help.
2. Export the image, run `docker save nim:latest | gzip -9 > pre-built-nim_latest.tar.gz`
3. Create the docker context archive, run `tar -czvf nim_dropin.tar.gz -C /path/to/public_dropin_environments/nim_llm/ .`
4. Using either the API or from the UI create a new Custom Environment with the tarballs created in step 1 and 3.
   - Be sure to upload **both** the context and pre-built image when creating a new version.

### Creating models for this environment

This environment makes the following assumption about your serialized model:
- The data sent to custom model can be used to make predictions without additional pre-processing
- There is a single model and model version present
- Model is expected to have the `textgeneration` target type.

### Supported Runtime Parameters

| Parameter Name | Required | Description |
| --- | --- | --- |
| `NGC_API_KEY` | Yes | Auth used to download model artifacts (if not using air-gapped solution). |
| `NIM_MODEL_PROFILE` | No | Override the NIM optimization profile. |
| `NIM_LOG_LEVEL` | No | Override default logging level. |
| `NIM_MAX_MODEL_LEN` | No | If running in vLLM mode, the model context length. |
| `system_prompt` | No | Prompt to assign as the `system` role to the LLM. |
| `prompt_column_name` | No | Name of the input column we expect to find the LLM prompt. |
| `max_tokens` | No | The maximum number of tokens that can be generated in the chat completion. This value can be used to control costs for text generated via API. |
| `n` | No | How many chat completion choices to generate for each input message. Note that you will be charged based on the number of generated tokens across all of the choices. Keep n as 1 to minimize costs. |
| `temperature` | No | The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit. |
| `verifySSL` | No | If we need to verify TLS whe communicating status back to DataRobot |

### Air Gapped Deployment
NVIDIA has official [documentation](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#serving-models-from-local-assets) regarding the process to pre-download the model artifacts
that you should read first. From there, you can include a `load_model` hook in a `custom.py` file
that fetches the artifacts from the desired local cache location. The only caveat with DRUM is
the destination of the files must not be `/opt/nim/.cache` as described in the official docs but
instead should be the value of `${NIM_CACHE_PATH}` environment variable.

Conversely, for the _local model directory route_, you **must** download/store your model artifacts to `${CODE_DIR}/model-repo` as part of the `load_model` hook.

### Example model-metadata.yaml

```yaml
name: NIM Server Example
type: inference
targetType: textgeneration

runtimeParameterDefinitions:
  - fieldName: NGC_API_KEY
    type: credential
    credentialType: api_token
    description: |-
      Access token to connect to NGC (for downloading optimized model artifacts). You must set this
      variable to the value of your personal NGC API key.

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

  - fieldName: NIM_MODEL_PROFILE
    type: string
    description: |-
      Override the NIM optimization profile that is automatically selected by specifying a profile
      ID from the manifest located at `/etc/nim/config/model_manifest.yaml`. If not specified, NIM
      will attempt to select an optimal profile compatible with available GPUs. A list of the
      compatible profiles can be obtained by appending `list-model-profiles` at the end of the
      docker run command. Using the profile name `default` will select a profile that is maximally
      compatible and may not be optimal for your hardware.

  - fieldName: NIM_LOG_LEVEL
    type: string
    defaultValue: DEFAULT
    description: |-
      Log level of NIM for LLMs service. Possible values of the variable are `DEFAULT`, `TRACE`, `DEBUG`,
      `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Mostly, the effect of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
      is described in Python 3 logging docs. `TRACE` log level enables printing of diagnostic
      information for debugging purposes in TRT-LLM and in uvicorn. When `NIM_LOG_LEVEL` is `DEFAULT`
      sets all log levels to `INFO` except for TRT-LLM log level which equals `ERROR`. When
      `NIM_LOG_LEVEL` is `CRITICAL` TRT-LLM log level is `ERROR`.

  - fieldName: NIM_MAX_MODEL_LEN
    type: numeric
    description: |-
      Model context length. If unspecified, will be automatically derived from the model configuration.
      Note that this setting has an effect on only models running on the vLLM backend.
```
