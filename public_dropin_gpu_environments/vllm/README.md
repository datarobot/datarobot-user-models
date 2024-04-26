# vLLM for LLMs Drop-In Environment

This drop-in environment contains the vLLM OpenAI compatible inference server with support for large language models (LLM).

## Instructions

1. From the terminal, run `tar -czvf vllm_dropin.tar.gz -C /path/to/public_dropin_environments/vllm/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created in step 1.

### Creating models for this environment

To use this environment, your custom model archive must contain at least one of the following:

1. Download an OSS LLM directly from [HuggingFace](https://huggingface.co) via setting the following Runtime Parameters:
  - `model`: name of the HuggingFace model (i.e. `meta-llama/Llama-2-7b-chat-hf`)
  - `HuggingFaceToken`: a credential of type API Token that

2. Download an OSS LLM via a user defined means via providing a `load_model` hook in a `custom.py` file that downloads the model artifacts to `/opt/code/vllm/`.

3. Provide a `vllm/` directory in the custom models assembly process that contains a supported model.


This environment makes the following assumption about your serialized model:

- The data sent to custom model can be used to make predictions without additional pre-processing
- There is a single model and model version present
- Model is expected to have the `TextGeneration` target type.
