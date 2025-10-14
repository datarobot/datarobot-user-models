## vLLM Inference Server Text Generation Template

This text generation example uses Llama-3.1-8b model by default but can be changed by modifying the [engine_config.json](engine_config.json) file.

## Requirements
- vLLM [Execution Environment](../../public_dropin_gpu_environments/vllm/)
- Generate a valid HuggingFace API Key:
https://huggingface.co/docs/hub/en/security-tokens
API Key is only needed to access models that are gated (i.e. Llama)
- If you run locally, you need a machine with a GPU and Docker installed.

## Instructions

- Create a new custom model version with the contents of `model_templates/gpu_vllm_textgen`.
- Update Runtime Parameters of the model:
  - The `HuggingFaceToken` parameter:
    - [Create an account](https://huggingface.co/join).
    - [Create a user access token](https://huggingface.co/docs/hub/en/security-tokens) with at least `READ` permission.
- In custom model resources, select an appropriate GPU bundle (at least `GPU - L` or a GPU bundle with at least 24GB of VRAM).
- Register a new model version and deploy.

### To run locally using Docker

1. Build an image:
```shell
export HF_TOKEN=<INSERT HUGGINGFACE TOKEN HERE>
cd ~/datarobot-user-models/public_dropin_gpu_environments/vllm
cp ~/datarobot-user-models/model_templates/gpu_vllm_textgen/* .
docker build -t vllm .
```



2. Run:
```shell
docker run -p8080:8080 \
  --gpus 'all' \
  --net=host \
  --shm-size=8GB \
  -e DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2 \
  -e DATAROBOT_API_TOKEN=${DATAROBOT_API_TOKEN} \
  -e MLOPS_DEPLOYMENT_ID=${DATAROBOT_DEPLOYMENT_ID} \
  -e TARGET_TYPE=textgeneration \
  -e TARGET_NAME=completions \
  -e MLOPS_RUNTIME_PARAM_HuggingFaceToken="{\"type\": \"credential\", \"payload\": {\"credentialType\": \"api_token\", \"apiToken\": \"${HF_TOKEN}\"}}" \
  vllm
```
- You can get the values for `DATAROBOT_API_TOKEN` and `MLOPS_DEPLOYMENT_ID` from the the DataRobot UI.If you use staging environment, you also need to set `DATAROBOT_ENDPOINT` to `https://staging.datarobot.com/api/v2`.
- Note: The `--shm-size` argument is only needed if you are trying to utilize multiple GPUs to run your LLM.
