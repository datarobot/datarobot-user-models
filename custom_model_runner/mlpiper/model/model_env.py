import os

from mlpiper.model import constants


class ModelEnv(object):
    def __init__(self, model_filepath, standalone=False):
        self._model_filepath = model_filepath
        self._standalone = standalone
        if not self._standalone and not self._model_filepath.endswith(
            constants.PIPELINE_MODEL_EXT
        ):
            self._model_filepath += constants.PIPELINE_MODEL_EXT
        self._model_root_dir = os.path.dirname(model_filepath)
        self._metadata_filepath = os.path.join(
            self._model_root_dir, constants.METADATA_FILENAME
        )
        self._sync_filepath = os.path.join(
            self._model_root_dir, constants.SYNC_FILENAME
        )

    @property
    def model_filepath(self):
        return self._model_filepath

    @property
    def standalone(self):
        return self._standalone

    @property
    def model_root_dir(self):
        return self._model_root_dir

    @property
    def sync_filepath(self):
        if not self._standalone and not os.path.isfile(self._sync_filepath):
            self.touch_sync()

        return self._sync_filepath

    @property
    def metadata_filepath(self):
        return self._metadata_filepath

    def touch_sync(self):
        try:
            os.utime(self._sync_filepath, None)
        except OSError:
            open(self._sync_filepath, "a").close()
