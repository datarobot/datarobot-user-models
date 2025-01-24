## NIM Inference Server Text Generation Template

This text generation example uses Llama-3.1-8b model pre-compiled for one of the [supported](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html#llama-3-1-8b-instruct) GPUs.

## Requirements
- NIM LLM [Execution Environment](../../public_dropin_gpu_environments/nim_llama_8b/)
- NIM Models currently only available via Developer program:
https://developer.nvidia.com/developer-program
- Generate a valid NVIDIA NGC Registry API Key:
https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html#optionally-generate-an-ngc-key
API Key is used to build a docker image and pull LLM models from NGC Registry.
- Install NGC CLI tool https://org.ngc.nvidia.com/setup/installers/cli

## Instructions

- Create a new custom model version with the contents of `model_templates/gpu_nim_textgen`.
- Update Runtime Parameters of the model:
  - The `NGC_API_KEY` parameter:
    - [Create a new credential](https://docs.datarobot.com/en/docs/data/connect-data/stored-creds.html#credentials-management) of type `API Token`:
    - Add the NGC Registry API Key ([how to generate API Key](https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html#optionally-generate-an-ngc-key))
- In custom model resources, select an appropriate GPU bundle (at least a `GPU - XL` or a bundle that matches one of the [supported configurations](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html#llama-3-1-8b-instruct)).
- Register a new model version and deploy.

### To run locally using Docker

1. Build an image:
```shell
export NGC_CLI_API_KEY=<INSERT NGC API KEY HERE>
docker login --username="\$oauthtoken" --password="${NGC_CLI_API_KEY}" nvcr.io
cd ~/datarobot-user-models/public_dropin_gpu_environments/nim_llama_8b
cp ~/datarobot-user-models/model_templates/gpu_nim_textgen/* .
docker build -t nim_llm .
```

2. Run:
```shell
docker run -p8080:8080 \
  --gpus 'all' \
  --net=host \
  --shm-size=16GB \
  -e DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2 \
  -e DATAROBOT_API_TOKEN=${DATAROBOT_API_TOKEN} \
  -e MLOPS_DEPLOYMENT_ID=${DATAROBOT_DEPLOYMENT_ID} \
  -e TARGET_TYPE=textgeneration \
  -e TARGET_NAME=completions \
  -e MLOPS_RUNTIME_PARAM_NGC_API_KEY="{\"type\": \"credential\", \"payload\": {\"credentialType\": \"api_token\", \"apiToken\": \"${NGC_CLI_API_KEY}\"}}" \
  nim_llm
```

Note: The `--shm-size` argument is only needed if you are trying to utilize multiple GPUs to run your LLM and your GPUs are not connected with NVLink.
