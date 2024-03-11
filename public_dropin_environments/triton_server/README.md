# NVIDIA Triton Inference Server Drop-In Template Environment

This template environment can be used to create artifact-only custom models with 
the following Triton Inference Server backends: ONNX, vLLM, TensorRT-LLM.
Your custom model directory needs only contain your model artifact if you use the
environment correctly.


## Instructions

1. From the terminal, run `tar -czvf triton_dropin.tar.gz -C /path/to/public_dropin_environments/triton_server/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created in step 1.

### Creating models for this environment

To use this environment, your custom model archive must contain a model repository directory structure
which is expected by Triton Inference server.
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html

Expected model repository structure:
```
model_repository/
  <model-name>/
    config.pbtxt
    <output-labels-file>
    <version>/
      <model-definition-file>
```

Example:
```
model_repository/
  densenet_onnx/
    config.pbtxt
    densenet_labels.txt
    1/
      model.onnx
```

This environment makes the following assumption about your serialized model:
- The data sent to custom model can be used to make predictions without
additional pre-processing
- There is a single model and model version present
- Model is expected to have one of the target types: `Unstructured` or `TextGeneration` 