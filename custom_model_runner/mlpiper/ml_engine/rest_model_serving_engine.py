from mlpiper.ml_engine.python_engine import PythonEngine
from mlpiper.model.model_fetcher import ModelFetcher
from mlpiper.pipeline import java_mapping
from mlpiper.pipeline import json_fields


class RestModelServingEngine(PythonEngine):
    """
    Implementing the MLEngine API for a RestModelServing engine.
    """

    def __init__(self, pipeline, mlpiper_jar=None, standalone=False):
        super(RestModelServingEngine, self).__init__(pipeline, mlpiper_jar, standalone)
        self._model_fetcher = None

    def run(self, mlops, pipeline):
        if not self.standalone:
            self._start_background_model_fetcher(mlops, pipeline)

    def _start_background_model_fetcher(self, mlops, pipeline):
        system_conf = pipeline[json_fields.PIPELINE_SYSTEM_CONFIG_FIELD]
        self._model_fetcher = ModelFetcher(
            mlops, system_conf[java_mapping.MODEL_FILE_SOURCE_PATH_KEY], self
        ).start()

    def stop(self):
        if self._model_fetcher:
            self._model_fetcher.stop_gracefully()
