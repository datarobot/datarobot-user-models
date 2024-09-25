"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import itertools
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

from datarobot_drum.drum.adapters.cli.shared.drum_class_label_adapter import DrumClassLabelAdapter
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import to_bool
from datarobot_drum.drum.model_metadata import read_model_metadata_yaml
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    ModelInfoKeys,
    StructuredDtoKeys,
    TargetType,
)
from datarobot_drum.drum.typeschema_validation import SchemaValidator
from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils
from datarobot_drum.drum.data_marshalling import marshal_predictions
from datarobot_drum.resource.chat_helpers import is_streaming_response

import datarobot as dr

DEFAULT_PROMPT_COLUMN_NAME = "promptText"

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

mlops_loaded = False
mlops_import_error = None
MLOps = None
try:
    from datarobot_mlops.mlops import MLOps
    from datarobot_mlops.common.exception import DRCommonException

    mlops_loaded = True
except ImportError as e:
    mlops_import_error = "Error importing MLOps python module(new path): {}".format(e)
    try:
        from datarobot.mlops.mlops import MLOps
        from datarobot.mlops.common.exception import DRCommonException

        mlops_loaded = True
    except ImportError as e:
        mlops_import_error += "\n\tError importing MLOps python module(old path): {}".format(e)


@dataclass
class PredictResponse:
    predictions: pd.DataFrame
    extra_model_output: Optional[pd.DataFrame] = None

    @property
    def combined_dataframe(self):
        return (
            self.predictions
            if self.extra_model_output is None
            else self.predictions.join(self.extra_model_output)
        )


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
        self._prompt_column_name = DEFAULT_PROMPT_COLUMN_NAME

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

        if self._should_enable_mlops():
            self._init_mlops()

        model_metadata = read_model_metadata_yaml(self._code_dir)
        if model_metadata:
            self._schema_validator = SchemaValidator(model_metadata.get("typeSchema", {}))

    def _should_enable_mlops(self):
        return to_bool(self._params.get("monitor")) or self._supports_chat()

    def _supports_chat(self):
        return False

    def _init_mlops(self):
        self._mlops = MLOps()

        if self._params.get("deployment_id", None):
            self._mlops.set_deployment_id(self._params["deployment_id"])

        if self._params.get("model_id", None):
            self._mlops.set_model_id(self._params["model_id"])

        if self._supports_chat():
            self._configure_mlops_for_chat()
        else:
            self._configure_mlops_for_non_chat()

        self._mlops.init()

    def _configure_mlops_for_chat(self):
        self._mlops.set_channel_config("spooler_type=API")

        self._prompt_column_name = self._get_prompt_column_name()
        logger.debug("Prompt column name: %s", self._prompt_column_name)

    def _get_prompt_column_name(self):
        if not self._params.get("deployment_id", None):
            logger.error(
                "No deployment ID found while configuring mlops for chat. "
                f"Fallback to default prompt column name ('{DEFAULT_PROMPT_COLUMN_NAME}')"
            )
            return DEFAULT_PROMPT_COLUMN_NAME

        try:
            deployment = dr.Deployment.get(self._params["deployment_id"])
            return deployment.model["prompt"]
        except Exception:
            logger.exception(
                "Failed to get prompt column name from deployment. "
                f"Fallback to default prompt column name ('{DEFAULT_PROMPT_COLUMN_NAME}')"
            )

        return DEFAULT_PROMPT_COLUMN_NAME

    def _configure_mlops_for_non_chat(self):
        self._mlops.set_channel_config(self._params["monitor_settings"])

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
                kwargs.get(StructuredDtoKeys.BINARY_DATA),
                kwargs.get(StructuredDtoKeys.MIMETYPE),
            )
            self._mlops.report_predictions_data(
                features_df=df, predictions=mlops_predictions, class_names=class_names
            )

    def predict(self, **kwargs) -> PredictResponse:
        start_predict = time.time()
        raw_predict_response = self._predict(**kwargs)
        predictions_df = marshal_predictions(
            request_labels=self.class_ordering,
            predictions=raw_predict_response.predictions,
            target_type=self.target_type,
            model_labels=raw_predict_response.columns,
        )
        end_predict = time.time()
        execution_time_ms = (end_predict - start_predict) * 1000
        self.monitor(kwargs, predictions_df, execution_time_ms)
        return PredictResponse(predictions_df, raw_predict_response.extra_model_output)

    @abstractmethod
    def _predict(self, **kwargs) -> RawPredictResponse:
        """Predict on input_filename or binary_data"""
        pass

    def chat(self, completion_create_params):
        start_time = time.time()
        try:
            response = self._chat(completion_create_params)
            response = self._validate_chat_response(response)
        except Exception as e:
            self._mlops_report_error(start_time)
            raise e

        if not is_streaming_response(response):
            self._mlops_report_chat_prediction(
                completion_create_params, start_time, response.choices[0].message.content
            )
            return response
        else:

            def generator():
                message_content = ""
                try:
                    for chunk in response:
                        message_content += (
                            chunk.choices[0].delta.content
                            if chunk.choices and chunk.choices[0].delta.content
                            else ""
                        )
                        yield chunk
                except Exception:
                    self._mlops_report_error(start_time)
                    raise

                self._mlops_report_chat_prediction(
                    completion_create_params, start_time, message_content
                )

            return generator()

    def _chat(self, completion_create_params):
        raise NotImplementedError("Chat is not implemented ")

    def _mlops_report_chat_prediction(self, completion_create_params, start_time, message_content):
        execution_time_ms = (time.time() - start_time) * 1000

        self._mlops.report_deployment_stats(num_predictions=1, execution_time_ms=execution_time_ms)

        latest_message = completion_create_params["messages"][-1]["content"]
        features_df = pd.DataFrame([{self._prompt_column_name: latest_message}])

        predictions = [message_content]
        try:
            self._mlops.report_predictions_data(
                features_df,
                predictions,
                association_ids=[str(uuid.uuid4())],
            )
        except DRCommonException:
            logger.exception("Failed to report predictions data")

    def _mlops_report_error(self, start_time):
        execution_time_ms = (time.time() - start_time) * 1000

        self._mlops.report_deployment_stats(num_predictions=0, execution_time_ms=execution_time_ms)

    @staticmethod
    def _validate_chat_response(response):
        if getattr(response, "object", None) == "chat.completion":
            return response

        try:
            # Take a peek at the first object in the iterable to make sure that this is indeed a chat completion chunk.
            # This should catch cases where hook returns an iterable of a different type early on.
            response_iter = iter(response)
            first_chunk = next(response_iter)

            if type(first_chunk).__name__ == "ChatCompletionChunk":
                # Return a new iterable where the peeked object is included in the beginning
                return itertools.chain([first_chunk], response_iter)
            else:
                raise Exception(
                    f"First chunk does not look like chat completion chunk. str(chunk): '{first_chunk}'"
                )
        except StopIteration:
            return iter(())
        except Exception as e:
            raise Exception(
                f"Expected response to be ChatCompletion or Iterable[ChatCompletionChunk]. response type: {type(response)}."
                f"response(str): {str(response)}"
            ) from e

    def transform(self, **kwargs):
        output = self._transform(**kwargs)
        output_X = output[0]
        if self.target_type.value == TargetType.TRANSFORM.value and self._schema_validator:
            self._schema_validator.validate_outputs(output_X)
        return output

    @abstractmethod
    def _transform(self, **kwargs):
        """Predict on input_filename or binary_data"""
        pass

    @abstractmethod
    def has_read_input_data_hook(self):
        """Check if read_input_data hook defined in predictor"""
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
