import os
import traceback
import uuid

from mlpiper.common.bg_actor import BgActor
from mlpiper.model.metadata import Metadata
from mlpiper.model.model_env import ModelEnv


class ModelFetcher(BgActor):
    POLLING_INTERVAL_SEC = 10.0

    def __init__(self, mlops, model_filepath, ml_engine):
        super(ModelFetcher, self).__init__(
            mlops, ml_engine, ModelFetcher.POLLING_INTERVAL_SEC
        )

        self._uuid = uuid.uuid4()
        self._current_model = self._mlops.current_model()
        self._logger.info("Current model: {}".format(self._current_model))
        self._model_env = None
        self._setup_model_env(model_filepath)

    def _setup_model_env(self, model_filepath):
        self._logger.info("Setup model env with: {}".format(model_filepath))
        self._model_env = ModelEnv(model_filepath)
        if os.path.isfile(model_filepath):
            self._logger.info(
                "Rename model file path to {}".format(self._model_env.model_filepath)
            )
            os.rename(model_filepath, self._model_env.model_filepath)
        Metadata(self._model_env.model_filepath).save(self._model_env.metadata_filepath)

    # Overloaded function
    def _do_repetitive_work(self):
        try:
            last_approved_model = self._mlops.get_last_approved_model()
            self._logger.debug(
                "Last approved model: {}, uuid: {}".format(
                    last_approved_model.get_id() if last_approved_model else None,
                    self._uuid,
                )
            )
            if last_approved_model and (
                not self._current_model or last_approved_model != self._current_model
            ):
                self._current_model = last_approved_model
                self._download_and_signal()
        except Exception as ex:
            self._logger.error(
                "Failed fetch last approved model from server! {}\n{}".format(
                    traceback.format_exc(), ex
                )
            )

    def _download_and_signal(self):
        self._logger.info(
            "New model is about to be downloaded: {}".format(self._current_model)
        )
        self._current_model.download(self._model_env.model_filepath)
        Metadata(self._model_env.model_filepath).save(self._model_env.metadata_filepath)
        self._signal()

    def _signal(self):
        self._model_env.touch_sync()
