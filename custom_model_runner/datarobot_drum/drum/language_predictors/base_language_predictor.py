import logging
import time

from abc import ABC, abstractmethod
import numpy as np

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    TargetType,
    StructuredDtoKeys,
    ModelInfoKeys,
    read_model_metadata_yaml,
)
from datarobot_drum.drum.typeschema_validation import SchemaValidator
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
        self._code_dir = None
        self._params = None
        self._mlops = None

    def configure(self, params):
        self._code_dir = params["__custom_model_path__"]
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

        model_metadata = read_model_metadata_yaml(self._code_dir)
        self._schema_validator = (
            SchemaValidator(model_metadata.get("typeSchema", {})) if model_metadata else None
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
                kwargs.get(StructuredDtoKeys.BINARY_DATA), kwargs.get(StructuredDtoKeys.MIMETYPE),
            )
            self._mlops.report_predictions_data(
                features_df=df, predictions=mlops_predictions, class_names=class_names
            )

    def validate_output(self, output_df):
        if self._target_type.value in TargetType.CLASSIFICATION.value:
            try:
                added_probs = output_df.sum(axis=1)
                np.testing.assert_array_almost_equal(added_probs, 1)
            except AssertionError:
                raise ValueError(
                    "Your prediction probabilities do not add up to 1. \n{}".format(output_df)
                )
        if self._target_type.value == TargetType.TRANSFORM.value:
            if self._schema_validator:
                self._schema_validator.validate_outputs(output_df)

    def predict(self, **kwargs):
        start_predict = time.time()
        predictions = self._predict(**kwargs)
        self.validate_output(predictions)
        end_predict = time.time()
        execution_time_ms = (end_predict - start_predict) * 1000
        self.monitor(kwargs, predictions, execution_time_ms)
        return predictions

    @abstractmethod
    def _predict(self, **kwargs):
        """ Predict on input_filename or binary_data """
        pass

    def transform(self, **kwargs):
        output = self._transform(**kwargs)
        output_X = output[0]
        # TODO: [RAPTOR-5765] validate output_y
        self.validate_output(output_X)
        return output

    @abstractmethod
    def _transform(self, **kwargs):
        """ Predict on input_filename or binary_data """
        pass

    @abstractmethod
    def has_read_input_data_hook(self):
        """ Check if read_input_data hook defined in predictor """
        pass

    def model_info(self):
        model_info = {
            ModelInfoKeys.TARGET_TYPE: self._target_type.value,
            ModelInfoKeys.CODE_DIR: self._code_dir,
        }

        if self._target_type == TargetType.BINARY:
            model_info.update({ModelInfoKeys.POSITIVE_CLASS_LABEL: self._positive_class_label})
            model_info.update({ModelInfoKeys.NEGATIVE_CLASS_LABEL: self._negative_class_label})
        elif self._target_type == TargetType.MULTICLASS:
            model_info.update({ModelInfoKeys.CLASS_LABELS: self._class_labels})

        return model_info
