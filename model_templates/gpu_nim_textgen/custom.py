# Save your NIM engine files here
MODEL_DIR = "/opt/code/model-repo/"


def load_model(code_dir: str):
    # print(f"Downloading model to {MODEL_DIR}...")

    # Here is where you can put code that downloads the model artifacts
    # from an internal source. See the official documentation for more details:
    #   https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#air-gap-deployment-local-model-directory-route
    # The only requirement is that the files must be downloaded to `/opt/code/model-repo`.
    return "succeeded"  # a non-empty response is required to signal that load_model succeeded

def score_unstructured(model, data, **kwargs):
    # Hook to make inference requests to NIM model deployed as target_type: unstructured
    # See https://docs.datarobot.com/en/docs/mlops/deployment/custom-models/custom-model-assembly/unstructured-custom-models.html

    return "succeeded", None
