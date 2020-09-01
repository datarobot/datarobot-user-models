import logging
import sys
import pprint
import time

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
)
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

mlops_loaded = False
mlops_import_error = None
try:
    from datarobot.mlops.mlops import MLOps

    mlops_loaded = True
except ImportError as e:
    mlops_import_error = "Error importing MLOps python module: {}".format(e)


class PythonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(PythonPredictor, self).__init__()
        self._model_adapter = None
        self._mlops = None

    def configure(self, params):
        super(PythonPredictor, self).configure(params)

        self._model_adapter = PythonModelAdapter(model_dir=self._custom_model_path)

        sys.path.append(self._custom_model_path)
        self._model_adapter.load_custom_hooks()
        self._model = self._model_adapter.load_model_from_artifact()
        if self._model is None:
            raise Exception("Failed to load model")

<<<<<<< HEAD
    def predict(self, input_filename):
=======
        if self._params["monitor"]:
            if not mlops_loaded:
                raise Exception("MLOps module was not imported: {}".format(mlops_import_error))
            self._mlops = (
                MLOps()
                .set_model_id(self._params["model_id"])
                .set_deployment_id(self._params["deployment_id"])
                .set_channel_config(self._params["monitor_settings"])
                .init()
            )
            # TODO: if server use async, if bash, use sync

    def predict(self, df):
        kwargs = {}
        if self._positive_class_label and self._negative_class_label:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self._positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self._negative_class_label

        pprint.pprint(self._params)
        start_predict = time.time()
        predictions = self._model_adapter.predict(data=df, model=self._model, **kwargs)
        end_predict = time.time()

        if self._params["monitor"]:
            print("Reporting predictions")
            execution_time_ms = (end_predict - start_predict) * 1000
            self._mlops.report_deployment_stats(
                num_predictions=len(predictions), execution_time_ms=execution_time_ms
            )

            # TODO: Need to convert predictions to a proper format
            # TODO: or add report_predictions_data that can handle a df directly..
            # TODO: need to handle associds correctly

            # mlops.report_predictions_data expect the prediction data in the following format:
            # Regression: [10, 12, 13]
            # Classification: [[0.5, 0.5], [0.7, 03]]
            # In case of classifcation, class names are also required
            class_names = None
            if len(predictions.columns) == 1:
                print(predictions.columns[0])
                mlops_predictions = predictions[predictions.columns[0]].tolist()
            else:
                mlops_predictions = predictions.values.tolist()
                if self._positive_class_label and self._negative_class_label:
                    class_names = [self._negative_class_label, self._positive_class_label]
            self._mlops.report_predictions_data(
                features_df=df, predictions=mlops_predictions, class_names=class_names
            )
            # pprint.pprint(predictions)
        return predictions
