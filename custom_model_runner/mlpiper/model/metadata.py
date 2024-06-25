import json


class Metadata(object):
    FILEPATH_KEY_NAME = "model"

    def __init__(self, model_filepath=None):
        self._model_filepath = model_filepath

    @property
    def model_filepath(self):
        return self._model_filepath

    def save(self, metadata_filepath):
        with open(metadata_filepath, "w") as f:
            json.dump(self._serialize(), f)

    def load(self, metadata_filepath):
        with open(metadata_filepath, "r") as f:
            content = json.load(f)
            self._model_filepath = content[Metadata.FILEPATH_KEY_NAME]
        return self

    def _serialize(self):
        return {Metadata.FILEPATH_KEY_NAME: self._model_filepath}
