"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import itertools
import logging
import shutil
import sys
import tempfile
import time
import uuid

import datarobot as dr
import pandas as pd
from datarobot_mlops.common.exception import DRCommonException

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import (
    PythonModelAdapter,
    RawPredictResponse,
)
from datarobot_drum.drum.common import to_bool
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
    CLASS_LABELS_ARG_KEYWORD,
    TARGET_TYPE_ARG_KEYWORD,
    CustomHooks,
)
from datarobot_drum.drum.exceptions import DrumCommonException, DrumSerializationError
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.drum.language_predictors.base_language_predictor import MLOps
from datarobot_drum.resource.chat_helpers import is_streaming_response

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PythonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(PythonPredictor, self).__init__()
        self._model_adapter = None
        self._mlops_spool_dir = None

    def mlpiper_configure(self, params):
        super(PythonPredictor, self).mlpiper_configure(params)

        if to_bool(params.get("allow_dr_api_access")):
            logger.info("Initializing DataRobot Python client.")
            dr_api_endpoint = self._dr_api_url(endpoint=params["external_webserver_url"])
            dr.Client(token=params["api_token"], endpoint=dr_api_endpoint)

        self._model_adapter = PythonModelAdapter(
            model_dir=self._code_dir, target_type=self.target_type
        )

        sys.path.append(self._code_dir)
        self._model_adapter.load_custom_hooks()

        if to_bool(params.get("monitor_embedded")) or self._model_adapter.has_custom_hook(
            CustomHooks.CHAT
        ):
            self._init_mlops(params)

        try:
            self._model = self._model_adapter.load_model_from_artifact(
                user_secrets_mount_path=params.get("user_secrets_mount_path"),
                user_secrets_prefix=params.get("user_secrets_prefix"),
            )
        except Exception as e:
            raise DrumSerializationError(f"An error occurred when loading your artifact: {str(e)}")
        if self._model is None:
            raise Exception("Failed to load model")

    @staticmethod
    def _dr_api_url(endpoint):
        if not endpoint.endswith("api/v2"):
            endpoint = f"{endpoint}/api/v2"
        return endpoint

    def _init_mlops(self, params):
        has_chat_hook = self._model_adapter.has_custom_hook(CustomHooks.CHAT)

        if self._mlops:
            if has_chat_hook:
                logger.warning(
                    "MLOps already initialized. Probably because --monitor was enabled. Can't configure "
                    "mlops with default settings for chat."
                )
            return

        self._mlops = MLOps()

        if params.get("deployment_id", None):
            self._mlops.set_deployment_id(params["deployment_id"])

        if params.get("model_id", None):
            self._mlops.set_model_id(params["model_id"])

        monitor_settings = self._params.get("monitor_settings")

        if not monitor_settings:
            if has_chat_hook:
                monitor_settings = "spooler_type=API"
            else:
                self._mlops_spool_dir = tempfile.mkdtemp()
                monitor_settings = "spooler_type=FILESYSTEM;directory={};max_files=5;file_max_size=10485760".format(
                    self._mlops_spool_dir
                )

        if not has_chat_hook:
            self._mlops.agent(
                mlops_service_url=params["external_webserver_url"],
                mlops_api_token=params["api_token"],
            )

        self._mlops.set_channel_config(monitor_settings)
        self._mlops.init()

    @property
    def supported_payload_formats(self):
        return self._model_adapter.supported_payload_formats

    def model_info(self):
        model_info = super(PythonPredictor, self).model_info()
        model_info.update(self._model_adapter.model_info())
        return model_info

    def has_read_input_data_hook(self):
        return self._model_adapter.has_read_input_data_hook()

    def _predict(self, **kwargs) -> RawPredictResponse:
        kwargs[TARGET_TYPE_ARG_KEYWORD] = self.target_type
        if self.positive_class_label is not None and self.negative_class_label is not None:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self.positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self.negative_class_label
        if self.class_labels:
            kwargs[CLASS_LABELS_ARG_KEYWORD] = self.class_labels

        return self._model_adapter.predict(model=self._model, **kwargs)

    def _transform(self, **kwargs):
        return self._model_adapter.transform(model=self._model, **kwargs)

    def predict_unstructured(self, data, **kwargs):
        if self._mlops:
            kwargs["mlops"] = self._mlops
        str_or_tuple = self._model_adapter.predict_unstructured(
            model=self._model, data=data, **kwargs
        )
        if isinstance(str_or_tuple, (str, bytes, type(None))):
            ret = str_or_tuple, None
        elif isinstance(str_or_tuple, tuple):
            ret = str_or_tuple
        else:
            raise DrumCommonException(
                "Wrong type returned in unstructured mode: {}".format(type(str_or_tuple))
            )
        return ret

    def chat(self, completion_create_params):
        start_time = time.time()
        try:
            response = self._model_adapter.chat(completion_create_params, self._model)
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

    def _mlops_report_chat_prediction(self, completion_create_params, start_time, message_content):
        execution_time_ms = (time.time() - start_time) * 1000

        self._mlops.report_deployment_stats(num_predictions=1, execution_time_ms=execution_time_ms)

        latest_message = completion_create_params["messages"][-1]["content"]
        features_df = pd.DataFrame([{"prompt": latest_message}])

        predictions = [message_content]
        try:
            self._mlops.report_predictions_data(
                features_df,
                predictions,
                association_ids=[str(uuid.uuid4())],
                skip_drift_tracking=True,
                skip_accuracy_tracking=True,
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

            if getattr(first_chunk, "object", None) == "chat.completion.chunk":
                # Return a new iterable where the peeked object is included in the beginning
                return itertools.chain([first_chunk], response_iter)
        except StopIteration:
            return iter(())
        except Exception:
            pass

        raise Exception(
            f"Expected response to be ChatCompletion or Iterable[ChatCompletionChunk]. response type: {type(response)}."
            f"response(str): {str(response)}"
        )

    def terminate(self):
        if self._mlops:
            self._mlops.shutdown()
            if self._mlops_spool_dir:
                shutil.rmtree(self._mlops_spool_dir)
