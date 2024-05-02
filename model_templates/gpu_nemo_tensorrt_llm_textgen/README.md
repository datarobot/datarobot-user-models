## NeMo Inference Server Text Generation Template

This text generation example uses Llama-2-7b model pre-compiled for A100 GPU.

## Requirements
- NeMo Inference Models currently only available via Early Access program:
https://developer.nvidia.com/nemo-llm-service-early-access/join
- Generate a valid NVIDIA NGC Registry API Key:
https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html#optionally-generate-an-ngc-key
API Key is used to build a docker image and pull LLM models from NGC Registry.
- Install NGC CLI tool https://org.ngc.nvidia.com/setup/installers/cli
- Most of NVIDIA models are pre-compiled for TensorRT-LLM support on A100 / V100 GPUs. See the official documentation:
https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#supported-model-and-gpus
- NOTE: it's possible to re-compile some models for lower tier GPUs, e.g. the Llama-2-7b model can run on A10:
https://developer.nvidia.com/docs/nemo-microservices/inference/playbooks/nmi_nonprebuilt_playbook.html

## Instructions

- Create a new custom model version with the contents of `model_templates/gpu_nemo_tensorrt_llm_textgen`.
- Update Runtime Parameters of the model:
  - The `ngcCredential` parameter:
    - [Create a new credential](https://docs.datarobot.com/en/docs/data/connect-data/stored-creds.html#credentials-management) of type `API Token`:
    - Add the NGC Registry API Key ([how to generate API Key](https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html#optionally-generate-an-ngc-key))
  - The `ngcRegistryUrl` parameter:
    - defaults to `ohlfw0olaadg/ea-participants/llama-2-7b:LLAMA-2-7B-4K-FP16-1-A100.24.01`
    - to list all the available model use
        ```shell
        export NGC_CLI_API_KEY=<INSERT NGC API KEY HERE>
        ngc registry image list "ohlfw0olaadg/ea-participants/nim_llm"
        ```
- In custom model resources, select an appropriate GPU bundle.
- Register a new model version and deploy.


### To run locally using Docker

1. Build an image:
```shell
export NGC_CLI_API_KEY=<INSERT NGC API KEY HERE>
docker login --username="\$oauthtoken" --password="${NGC_CLI_API_KEY}" nvcr.io
cd ~/datarobot-user-models/public_dropin_gpu_environments/nim_llm
cp ~/datarobot-user-models/model_templates/gpu_nemo_tensorrt_llm_textgen/* .
docker build -t nim_llm .
```

2. Choose a LLM model to run from NGC Registry:
```shell
ngc registry image list "ohlfw0olaadg/ea-participants/nim_llm"
```
For example, lets select a Llama-2-7b model: `ohlfw0olaadg/ea-participants/llama-2-7b:LLAMA-2-7B-4K-FP16-1-A100.24.01`

3. Run:
```shell
docker run -p8080:8080 \
  --gpus 'all' \
  --net=host \
  -e DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2 \
  -e DATAROBOT_API_TOKEN=${DATAROBOT_API_TOKEN} \
  -e MLOPS_DEPLOYMENT_ID=${DATAROBOT_DEPLOYMENT_ID} \
  -e TARGET_TYPE=textgeneration \
  -e TARGET_NAME=completions \
  -e MLOPS_RUNTIME_PARAM_ngcRegistryUrl='{"type": "string", "payload": "ohlfw0olaadg/ea-participants/llama-2-7b:LLAMA-2-7B-4K-FP16-1-A100.24.01"}' \
  -e MLOPS_RUNTIME_PARAM_ngcCredentials='{"type": "credential", "payload": {"credentialType": "api_token", "apiToken": ${NGC_CLI_API_KEY}}}' \
  nim_llm
```
