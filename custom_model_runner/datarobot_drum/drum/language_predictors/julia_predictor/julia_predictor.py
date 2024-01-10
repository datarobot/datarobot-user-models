"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    PayloadFormat,
    StructuredDtoKeys,
    UnstructuredDtoKeys,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JL_SCORE_PATH = os.path.join(CUR_DIR, "score.jl")
JL_SYSIMAGE_PATH = os.environ.get("JULIA_SYS_IMAGE")
JL_PROJECT = os.environ.get("JULIA_PROJECT", CUR_DIR)
JL_COMMON_PATH = os.path.abspath(
    os.path.join(
        CUR_DIR,
        "..",
        "julia_common_code",
        "common.jl",
    )
)
JL_INIT = ["--history-file=no"]
logger.info(f"Julia project director set as {JL_PROJECT}")

try:
    from julia.api import Julia
    from julia.api import LibJulia, JuliaInfo
except ImportError:
    error_message = (
        "julia package is not installed."
        "Install julia using 'pip install julia==0.5.6'"
        "Available for Python>=3.4 and Python <= 3.8"
    )
    logger.error(error_message)
    raise DrumCommonException(error_message)

## need better invocation here
try:
    jl = Julia(sysimage=JL_SYSIMAGE_PATH, init_julia=JL_INIT)
except Exception as error_message:
    logger.error(error_message)
    jl = Julia(init_julia=JL_INIT)
logger.info("Julia ready!")
from julia import Base

logger.info(f"julia was started with {Base.julia_cmd()}")
jl.eval(f'using Pkg; Pkg.activate("{JL_PROJECT}"); Pkg.instantiate()')


class JlPredictor(BaseLanguagePredictor):
    def __init__(
        self,
    ):
        super(JlPredictor, self).__init__()

    def mlpiper_configure(self, params):
        super(JlPredictor, self).mlpiper_configure(params)
        logger.info(f"loading {JL_SCORE_PATH}")
        jl.eval(f'include("{JL_SCORE_PATH}")')
        logger.info(f"{JL_SCORE_PATH} loaded")
        from julia import Main

        Main.init(self._code_dir, self.target_type.value)
        self._model = Main.load_serialized_model(self._code_dir)

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        # formats.add(PayloadFormat.MTX)
        return formats

    def has_read_input_data_hook(self):
        return Main.defined_hooks["read_input_data"]

    def _predict(self, **kwargs) -> RawPredictResponse:
        from julia import Main

        input_binary_data = kwargs.get(StructuredDtoKeys.BINARY_DATA)
        mimetype = kwargs.get(StructuredDtoKeys.MIMETYPE)

        predictions = Main.outer_predict(
            self.target_type.value,
            binary_data=input_binary_data,
            mimetype=mimetype,
            model=self._model,
            positive_class_label=self.positive_class_label,
            negative_class_label=self.negative_class_label,
            class_labels=self.class_labels,
        )

        return RawPredictResponse(predictions.values, predictions.columns)

    # # TODO: check test coverage for all possible cases: return None/str/bytes, and casting.
    def predict_unstructured(self, data, **kwargs):
        from julia import Main

        ## TODO still need to do a validation on unstructured predict
        ## TODO still need to do a validation on the data
        data_binary_or_text = data
        mimetype = kwargs.get(UnstructuredDtoKeys.MIMETYPE)
        query = kwargs.get(UnstructuredDtoKeys.QUERY)
        charset = kwargs.get(UnstructuredDtoKeys.CHARSET)
        ret = Main.predict_unstructured(
            data_binary_or_text, model=self._model, mimetype=mimetype, query=query, charset=charset
        )
        if isinstance(ret, (str, bytes, type(None))):
            ret = ret, None
        elif isinstance(ret, tuple):
            ret = ret
        return ret

    def _transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported for Julia yet")
