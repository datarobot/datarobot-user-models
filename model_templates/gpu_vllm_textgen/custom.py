# Save your OSS LLM to this directory to be loaded by vLLM.
MODEL_DIR = "/opt/code/vllm/"


def load_model(code_dir: str):
    print(f"Downloading model to {MODEL_DIR}...")
    # Add custom code to download supported OSS LLM here, otherwise we will
    # download the weights from the HuggingFace Hub based on the model name
    # specified in the runtime parameters.
    return True
