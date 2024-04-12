# vLLM for LLMs Drop-In Environment

This drop-in environment contains the vLLM OpenAI compatible inference server with support for large language models (LLM).

## Instructions

1. From the terminal, run `tar -czvf vllm_dropin.tar.gz -C /path/to/public_dropin_environments/vllm/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created in step 1.

### Creating models for this environment

To use this environment, your custom model archive must contain a model repository directory structure
which is expected by vLLM server. Refer to


This environment makes the following assumption about your serialized model:

- The data sent to custom model can be used to make predictions without additional pre-processing
- There is a single model and model version present
- Model is expected to have the `TextGeneration` target type.
