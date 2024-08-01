MODEL_FILE_SINK_PATH_KEY = "modelFileSinkPath"
MODEL_FILE_SOURCE_PATH_KEY = "modelFileSourcePath"

TAGS = {
    "model_dir": MODEL_FILE_SINK_PATH_KEY,
    "input_model_path": MODEL_FILE_SOURCE_PATH_KEY,
}

# The reserved keys ensures that the given attributes will not be manipulated by
# user's components
RESERVED_KEYS = {k: "__{}__tag__".format(k) for (k, v) in TAGS.items()}
