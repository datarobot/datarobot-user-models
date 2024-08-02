# Provides information about the model file path that should be fetched and used by the pipeline
METADATA_FILENAME = "metadata.json"

# A helper file that is used to signal the uWSGI workers about new models
SYNC_FILENAME = "sync"

# A dedicated extension that is used to avoid model file paths collisions between the
# pipeline model fetch and the agent
PIPELINE_MODEL_EXT = ".last_approved"
