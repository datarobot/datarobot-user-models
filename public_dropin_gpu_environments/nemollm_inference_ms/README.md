# NVIDIA NIM for LLMs Drop-In Environment

This drop-in environment contains the NVIDIA NIM server with support for large language models (LLM).

## Instructions

1. From the terminal, run `tar -czvf nim_dropin.tar.gz -C /path/to/public_dropin_environments/nemollm_inference_ms/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created in step 1.

### Creating models for this environment

To use this environment, your custom model archive must contain a model repository directory structure
which is expected by NIM server. Refer to
https://developer.nvidia.com/docs/nemo-microservices/inference/model-repo-generator.html#model-repo-generator

Expected model repository structure example:
```
model-store/
├── ensemble
│   ├── 1
│   │   ├── model_config.yaml
│   │   └── model.py
│   └── config.pbtxt
└── trt_llm
    ├── 1
    │   └── engine_dir
    │       ├── config.json
    │       ├── model.cache
    │       └── llama_float16_tp1_rank0.engine
    └── config.pbtxt
```

This environment makes the following assumption about your serialized model:
- The data sent to custom model can be used to make predictions without
additional pre-processing
- There is a single model and model version present
- Model is expected to have the `Textgeneration` target type. 