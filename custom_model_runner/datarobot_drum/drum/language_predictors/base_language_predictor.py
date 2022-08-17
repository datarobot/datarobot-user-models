"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, List

from datarobot_drum.drum.adapters.cli.shared.drum_class_label_adapter import DrumClassLabelAdapter
from datarobot_drum.drum.common import read_model_metadata_yaml, to_bool
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    ModelInfoKeys,
    StructuredDtoKeys,
    TargetType,
)
from datarobot_drum.drum.typeschema_validation import SchemaValidator
from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils
from datarobot_drum.drum.data_marshalling import marshal_predictions

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

mlops_loaded = False
mlops_import_error = None
try:
    from datarobot.mlops.mlops import MLOps

    mlops_loaded = True
except ImportError as e:
    mlops_import_error = "Error importing MLOps python module: {}".format(e)


class BaseLanguagePredictor(DrumClassLabelAdapter, ABC):
    def __init__(
        self,
        target_type: TargetType = None,
        positive_class_label: Optional[str] = None,
        negative_class_label: Optional[str] = None,
        class_labels: Optional[List[str]] = None,
    ):
        # TODO: Only use init, and do not initialize using mlpiper configure
        DrumClassLabelAdapter.__init__(
            self,
            target_type=target_type,
            positive_class_label=positive_class_label,
            negative_class_label=negative_class_label,
            class_labels=class_labels,
        )
        self._model = None
        self._code_dir = None
        self._params = None
        self._mlops = None
        self._schema_validator = None

    def mlpiper_configure(self, params):
        """
        Set class instance variables based in mlpiper input.
        TODO: Remove this function entirely, and have MLPiper init variables using the actual class init.
        """
        # DrumClassLabelAdapter fields
        self.positive_class_label = params.get("positiveClassLabel")
        self.negative_class_label = params.get("negativeClassLabel")
        self.class_labels = params.get("classLabels")
        self.target_type = TargetType(params.get("target_type"))

        self._code_dir = params["__custom_model_path__"]
        self._params = params
        self._validate_mlops_monitoring_requirements(self._params)

        if to_bool(params.get("monitor")):
            # TODO: if server use async, if batch, use sync etc.. some way of passing params
            self._mlops = (
                MLOps()
                .set_model_id(self._params["model_id"])
                .set_deployment_id(self._params["deployment_id"])
                .set_channel_config(self._params["monitor_settings"])
                .init()
            )

        model_metadata = read_model_metadata_yaml(self._code_dir)
        if model_metadata:
            self._schema_validator = SchemaValidator(model_metadata.get("typeSchema", {}))

    @staticmethod
    def _validate_mlops_monitoring_requirements(params):
        if (
            to_bool(params.get("monitor")) or to_bool(params.get("monitor_embedded"))
        ) and not mlops_loaded:
            # Note that for the case of monitoring from environment variable for the java
            # this package is not really needed, but it'll anyway be available
            raise Exception("MLOps module was not imported: {}".format(mlops_import_error))

    @staticmethod
    def _validate_expected_env_variables(*args):
        for env_var in args:
            if not os.environ.get(env_var):
                raise Exception(f"A valid environment variable '{env_var}' is missing!")

    def monitor(self, kwargs, predictions, predict_time_ms):
        if to_bool(self._params.get("monitor")):
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
            class_names = None
            if len(predictions.columns) == 1:
                mlops_predictions = predictions[predictions.columns[0]].tolist()
            else:
                mlops_predictions = predictions.values.tolist()
                class_names = list(predictions.columns)

            df = StructuredInputReadUtils.read_structured_input_data_as_df(
                kwargs.get(StructuredDtoKeys.BINARY_DATA), kwargs.get(StructuredDtoKeys.MIMETYPE),
            )
            self._mlops.report_predictions_data(
                features_df=df, predictions=mlops_predictions, class_names=class_names
            )

    def predict(self, **kwargs):
        start_predict = time.time()
        predictions, labels_in_predictions = self._predict(**kwargs)
        predictions = marshal_predictions(
            request_labels=self.class_ordering,
            predictions=predictions,
            target_type=self.target_type,
            model_labels=labels_in_predictions,
        )
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
        if self.target_type.value == TargetType.TRANSFORM.value and self._schema_validator:
            self._schema_validator.validate_outputs(output_X)
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
            ModelInfoKeys.TARGET_TYPE: self.target_type.value,
            ModelInfoKeys.CODE_DIR: self._code_dir,
        }

        if self.target_type == TargetType.BINARY:
            model_info.update({ModelInfoKeys.POSITIVE_CLASS_LABEL: self.positive_class_label})
            model_info.update({ModelInfoKeys.NEGATIVE_CLASS_LABEL: self.negative_class_label})
        elif self.target_type == TargetType.MULTICLASS:
            model_info.update({ModelInfoKeys.CLASS_LABELS: self.class_labels})

        return model_info
