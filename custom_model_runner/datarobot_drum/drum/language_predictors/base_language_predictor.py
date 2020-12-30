import logging

from abc import ABC, abstractmethod


from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    TargetType,
    StructuredDtoKeys,
)
from datarobot_drum.drum.utils import StructuredInputReadUtils

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

mlops_loaded = False
mlops_import_error = None
try:
    from datarobot.mlops.mlops import MLOps

    mlops_loaded = True
except ImportError as e:
    mlops_import_error = "Error importing MLOps python module: {}".format(e)


class BaseLanguagePredictor(ABC):
    def __init__(self):
        self._model = None
        self._positive_class_label = None
        self._negative_class_label = None
        self._class_labels = None
        self._custom_model_path = None
        self._params = None
        self._mlops = None

    def configure(self, params):
        self._custom_model_path = params["__custom_model_path__"]
        self._positive_class_label = params.get("positiveClassLabel")
        self._negative_class_label = params.get("negativeClassLabel")
        self._class_labels = params.get("classLabels")
        self._target_type = TargetType(params.get("target_type"))
        self._params = params

        if self._params["monitor"] == "True":
            if not mlops_loaded:
                raise Exception("MLOps module was not imported: {}".format(mlops_import_error))
            # TODO: if server use async, if batch, use sync etc.. some way of passing params
            self._mlops = (
                MLOps()
                .set_model_id(self._params["model_id"])
                .set_deployment_id(self._params["deployment_id"])
                .set_channel_config(self._params["monitor_settings"])
                .init()
            )

    def monitor(self, kwargs, predictions, predict_time_ms):
        if self._params["monitor"] == "True":
            self._mlops.report_deployment_stats(
                num_predictions=len(predictions), execution_time_ms=predict_time_ms
            )

            # TODO: Need to convert predictions to a proper format
            # TODO: or add report_predictions_data that can handle a df directly..
            # TODO: need to handle associds correctly

            # mlops.report_predictions_data expect the prediction data in the following format:
            # Regression: [10, 12, 13]
            # Classification: [[0.5, 0.5], [0.7, 03]]
            # In case of classification, class names are also required
            class_names = self._class_labels
            if len(predictions.columns) == 1:
                mlops_predictions = predictions[predictions.columns[0]].tolist()
            else:
                mlops_predictions = predictions.values.tolist()
                if (
                    self._positive_class_label is not None
                    and self._negative_class_label is not None
                ):
                    class_names = [self._negative_class_label, self._positive_class_label]

            df = StructuredInputReadUtils.read_structured_input_data_as_df(
                kwargs.get(StructuredDtoKeys.BINARY_DATA),
                kwargs.get(StructuredDtoKeys.MIMETYPE),
            )
            self._mlops.report_predictions_data(
                features_df=df, predictions=mlops_predictions, class_names=class_names
            )

    @abstractmethod
    def predict(self, **kwargs):
        """ Predict on input_filename or binary_data """
        pass

    @abstractmethod
    def transform(self, **kwargs):
        """ Predict on input_filename or binary_data """
        pass

    @abstractmethod
    def has_read_input_data_hook(self):
        """ Check if read_input_data hook defined in predictor """
        pass
