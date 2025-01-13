"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import datetime
import glob
import io
import logging
import os
import re
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd
from datarobot_predict import TimeSeriesType

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import (
    RawPredictResponse, RUNNING_LANG_MSG,
)
from datarobot_drum.drum.common import to_bool, SupportedPayloadFormats
from datarobot_drum.drum.data_marshalling import get_request_labels
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
    CLASS_LABELS_ARG_KEYWORD,
    TARGET_TYPE_ARG_KEYWORD,
    TargetType, JavaArtifacts, PayloadFormat,
)
from datarobot_drum.drum.exceptions import DrumCommonException, DrumSerializationError
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_predict.scoring_code import ScoringCodeModel

from datarobot_drum.drum.utils.dataframe import split_to_predictions_and_extra_model_output

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class ScoringCodePredictor(BaseLanguagePredictor):
    def __init__(self):
        super(ScoringCodePredictor, self).__init__()
        self.custom_model_path = None
        self.model_artifact_extension = None
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self._model_adapter = None
        self._mlops_spool_dir = None
        self.target_type = None

    def configure(self, params):
        super(ScoringCodePredictor, self).configure(params)
        self.target_type = TargetType(params.get("target_type"))
        code_dir = params["__custom_model_path__"]

        sys.path.append(code_dir)

        self.custom_model_path = params["__custom_model_path__"]
        files_list = sorted(os.listdir(self.custom_model_path))
        files_list_str = " | ".join(files_list).lower()

        self.logger.debug("files in custom model path: ".format(files_list_str))
        reg_exp = r"(\{})".format(JavaArtifacts.JAR_EXTENSION)
        ext_re = re.findall(reg_exp, files_list_str)

        if len(ext_re) == 0:
            raise DrumCommonException(
                "\n\n{}\n"
                "Could not find model artifact file in: {} supported by default predictors.\n"
                "They support filenames with the following extensions {}.\n"
                "List of retrieved files are: {}".format(
                    RUNNING_LANG_MSG, self.custom_model_path, JavaArtifacts.JAR_EXTENSION, files_list_str
                )
            )
        self.logger.debug("relevant artifact extensions {}".format(", ".join(ext_re)))

        self.model_artifact_extension = ext_re[0]
        self.logger.debug("model artifact extension: {}".format(self.model_artifact_extension))

        jars = glob.glob(os.path.join(self.custom_model_path, "*{}".format(self.model_artifact_extension)))
        self.logger.debug("Detected jars: {}".format(jars))

        self._model_adapter = ScoringCodeModel(jars[0])

    def _should_enable_mlops(self):
        return super()._should_enable_mlops() or to_bool(self._params.get("monitor_embedded"))

    @staticmethod
    def _dr_api_url(endpoint):
        if not endpoint.endswith("api/v2"):
            endpoint = f"{endpoint}/api/v2"
        return endpoint

    def supports_chat(self):
        return False

    def _configure_mlops_for_non_chat(self):
        monitor_settings = self._params.get("monitor_settings")

        if not monitor_settings:
            self._mlops_spool_dir = tempfile.mkdtemp()
            monitor_settings = (
                "spooler_type=FILESYSTEM;directory={};max_files=5;file_max_size=10485760".format(
                    self._mlops_spool_dir
                )
            )

        self._mlops.set_channel_config(monitor_settings)

        if to_bool(self._params["monitor_embedded"]):
            self._mlops.agent(
                mlops_service_url=self._params["external_webserver_url"],
                mlops_api_token=self._params["api_token"],
            )

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        return formats

    def model_info(self):
        model_info = super(ScoringCodePredictor, self).model_info()
        if hasattr(self._model_adapter, "model_info") and self._model_adapter.model_info is not None:
            model_info.update(self._model_adapter.model_info)
        return model_info


    def _predict(self, **kwargs) -> RawPredictResponse:
        kwargs[TARGET_TYPE_ARG_KEYWORD] = self.target_type
        if self.positive_class_label is not None and self.negative_class_label is not None:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self.positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self.negative_class_label
        if self.class_labels:
            kwargs[CLASS_LABELS_ARG_KEYWORD] = self.class_labels

        data = kwargs.get("binary_data")
        query = kwargs.get('query')

        positive_class_label = kwargs.get(POSITIVE_CLASS_LABEL_ARG_KEYWORD)
        negative_class_label = kwargs.get(NEGATIVE_CLASS_LABEL_ARG_KEYWORD)
        if self.target_type in {TargetType.BINARY, TargetType.MULTICLASS}:
            request_labels = get_request_labels(
                kwargs.get(CLASS_LABELS_ARG_KEYWORD),
                positive_class_label,
                negative_class_label,
            )
        else:
            request_labels = None

        if request_labels is not None:
            assert all(isinstance(label, str) for label in request_labels)

        df = pd.read_csv(io.StringIO(data.decode(kwargs.get('charset'))), dtype=str)  # use str type for all columns

        predictions_df = self._model_adapter.predict(df, **self._query_convert(query))

        if bool(self.model_info() and self.model_info().get('Is-Time-Series', False)):
            predictions_df.rename(columns={'PREDICTION': 'Predictions'}, inplace=True)

        if self.target_type in {TargetType.BINARY, TargetType.MULTICLASS} and isinstance(
                self._model_adapter, ScoringCodeModel
        ):
            labels_map = {
                f"target_{label}_PREDICTION": label for label in request_labels
            }
            predictions_df.rename(columns=labels_map, inplace=True)
        predictions_df, extra_model_output = split_to_predictions_and_extra_model_output(
            predictions_df, request_labels
        )
        predictions = predictions_df.values
        model_labels = predictions_df.columns

        return RawPredictResponse(predictions, model_labels, extra_model_output)

    def _query_convert(self, kwargs: dict) -> dict[str, Any]:
        return {
            'max_explanations': self._convert_value(kwargs.get('max_explanations', '0'), int),
            'threshold_high': self._convert_value(kwargs.get('threshold_high', 'None'), float) if kwargs.get(
                'threshold_high') is not None else None,
            'threshold_low': self._convert_value(kwargs.get('threshold_low', 'None'), float) if kwargs.get(
                'threshold_low') is not None else None,
            'time_series_type': self._convert_value(kwargs.get('time_series_type', 'FORECAST'), TimeSeriesType),
            'forecast_point': self._convert_value(kwargs.get('forecast_point', 'None'),
                                                  datetime.datetime) if kwargs.get(
                'forecast_point') is not None else None,
            'predictions_start_date': self._convert_value(kwargs.get('predictions_start_date', 'None'),
                                                          datetime.datetime) if kwargs.get(
                'predictions_start_date') is not None else None,
            'predictions_end_date': self._convert_value(kwargs.get('predictions_end_date', 'None'),
                                                        datetime.datetime) if kwargs.get(
                'predictions_end_date') is not None else None,
            'prediction_intervals_length': self._convert_value(kwargs.get('prediction_intervals_length', 'None'),
                                                               int) if kwargs.get(
                'prediction_intervals_length') is not None else None,
            'passthrough_columns': self._convert_value(kwargs.get('passthrough_columns', 'None'), set) if kwargs.get(
                'passthrough_columns') is not None else None
        }

    def _convert_value(self, value: str, target_type):
        if target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == datetime.datetime:
            return datetime.datetime.fromisoformat(value)
        elif target_type == set:
            return set(value.split(',')) if value else None
        elif target_type == TimeSeriesType:
            if value.isdigit():
                return TimeSeriesType(int(value))
            else:
                return TimeSeriesType[value.upper()]
        else:
            return value

    def terminate(self):
        if self._mlops:
            self._mlops.shutdown()
            if self._mlops_spool_dir:
                shutil.rmtree(self._mlops_spool_dir)

    def _transform(self, **kwargs):
        return self._model_adapter.transform(model=self._model, **kwargs)

    def has_read_input_data_hook(self):
        return self._model_adapter.has_read_input_data_hook()
