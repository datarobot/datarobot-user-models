"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import urllib
from typing import Optional

import werkzeug
from datarobot_drum.drum.adapters.cli.drum_score_adapter import DrumScoreAdapter
from datarobot_drum.drum.enum import GPU_PREDICTORS
from datarobot_drum.drum.enum import TARGET_TYPE_ARG_KEYWORD
from datarobot_drum.drum.enum import RunLanguage
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.enum import UnstructuredDtoKeys
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.root_predictors.unstructured_helpers import (
    _resolve_incoming_unstructured_data,
)
from datarobot_drum.drum.root_predictors.unstructured_helpers import (
    _resolve_outgoing_unstructured_data,
)


class GenericPredictorComponent:
    def __init__(self, params: dict):
        self._params = params
        self._run_language = RunLanguage(params.get("run_language"))
        self._gpu_predictor_type = params.get("gpu_predictor")
        self.cli_adapter = DrumScoreAdapter(
            custom_task_folder_path=params["__custom_model_path__"],
            input_filename=params["input_filename"],
            sparse_column_filename=params.get("sparse_column_file"),
            target_type=TargetType(params[TARGET_TYPE_ARG_KEYWORD]),
            positive_class_label=params.get("positiveClassLabel"),
            negative_class_label=params.get("negativeClassLabel"),
            class_labels=params.get("classLabels"),
        )
        self._predictor = self._setup_predictor()

    def _setup_predictor(self):
        if self._run_language == RunLanguage.PYTHON:
            from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
                PythonPredictor,
            )

            predictor = PythonPredictor()
        elif self._run_language == RunLanguage.JAVA:
            from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import (
                JavaPredictor,
            )

            predictor = JavaPredictor()
        elif self._run_language == RunLanguage.JULIA:
            from datarobot_drum.drum.language_predictors.julia_predictor.julia_predictor import (
                JlPredictor,
            )

            predictor = JlPredictor()
        elif self._run_language == RunLanguage.R:
            # this import is here, because RPredictor imports rpy library,
            # which is not installed for Java and Python cases.
            from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import RPredictor

            predictor = RPredictor()
        elif self._gpu_predictor_type and self._gpu_predictor_type == GPU_PREDICTORS.TRITON:
            from datarobot_drum.drum.gpu_predictors.triton_predictor import (
                TritonPredictor,
            )

            predictor = TritonPredictor()
        elif self._gpu_predictor_type and self._gpu_predictor_type == GPU_PREDICTORS.NIM:
            from datarobot_drum.drum.gpu_predictors.nim_predictor import (
                NIMPredictor,
            )

            predictor = NIMPredictor()
        elif self._gpu_predictor_type and self._gpu_predictor_type == GPU_PREDICTORS.VLLM:
            from datarobot_drum.drum.gpu_predictors.vllm_predictor import (
                VllmPredictor,
            )

            predictor = VllmPredictor()
        else:
            raise DrumCommonException(
                "Prediction server doesn't support language: {} ".format(self._run_language)
            )

        predictor.configure(self._params)
        return predictor

    def materialize(self):
        output_filename = self._params.get("output_filename")

        if self.cli_adapter.target_type == TargetType.UNSTRUCTURED:
            # TODO: add support to use cli_adapter for unstructured
            return self._materialize_unstructured(
                input_filename=self._params["input_filename"],
                output_filename=output_filename,
            )

        if self.cli_adapter.target_type == TargetType.TRANSFORM:
            transformed_output = self._predictor.transform(
                binary_data=self.cli_adapter.input_binary_data,
                mimetype=self.cli_adapter.input_binary_mimetype,
                # TODO: add sparse colnames
            )
            transformed_df = transformed_output[0]
            transformed_df.to_csv(output_filename, index=False)
        else:
            predict_response = self._predictor.predict(
                binary_data=self.cli_adapter.input_binary_data,
                mimetype=self.cli_adapter.input_binary_mimetype,
                sparse_colnames=self.cli_adapter.sparse_column_names,
            )
            predict_response.combined_dataframe.to_csv(output_filename, index=False)
        return []

    def _materialize_unstructured(self, input_filename, output_filename):
        kwargs_params = {}
        query_params = dict(urllib.parse.parse_qsl(self._params.get("query_params")))
        mimetype, content_type_params_dict = werkzeug.http.parse_options_header(
            self._params.get("content_type")
        )
        charset = content_type_params_dict.get("charset")

        with open(input_filename, "rb") as f:
            data_binary = f.read()

        data_binary_or_text, mimetype, charset = _resolve_incoming_unstructured_data(
            data_binary,
            mimetype,
            charset,
        )
        kwargs_params[UnstructuredDtoKeys.MIMETYPE] = mimetype
        if charset is not None:
            kwargs_params[UnstructuredDtoKeys.CHARSET] = charset
        kwargs_params[UnstructuredDtoKeys.QUERY] = query_params

        ret_data, ret_kwargs = self._predictor.predict_unstructured(
            data_binary_or_text, **kwargs_params
        )
        _, _, response_charset = _resolve_outgoing_unstructured_data(ret_data, ret_kwargs)

        # only for screen printout convenience we take pred data directly from unstructured_response
        if isinstance(ret_data, bytes):
            with open(output_filename, "wb") as f:
                f.write(ret_data)
        else:
            if ret_data is None:
                ret_data = "Return value from prediction is: None (NULL in R)"
            with open(output_filename, "w", encoding=response_charset) as f:
                f.write(ret_data)
        return []

    def terminate(self):
        terminate_op = getattr(self._predictor, "terminate", None)
        if callable(terminate_op):
            terminate_op()
