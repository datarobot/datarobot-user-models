import logging
import numpy
import os
import pandas as pd
import time

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    TargetType,
    CustomHooks,
    REGRESSION_PRED_COLUMN,
    UnstructuredDtoKeys,
    PayloadFormat,
    SupportedPayloadFormats,
    StructuredDtoKeys,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, StrVector
    from rpy2.robjects.conversion import localconverter

except ImportError:
    error_message = (
        "rpy2 package is not installed."
        "Install datarobot-drum using 'pip install datarobot-drum[R]'"
        "Available for Python>=3.6"
    )
    logger.error(error_message)
    raise DrumCommonException(error_message)


pandas2ri.activate()
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
R_SCORE_PATH = os.path.join(CUR_DIR, "score.R")
R_COMMON_PATH = os.path.abspath(
    os.path.join(
        CUR_DIR,
        "..",
        "r_common_code",
        "common.R",
    )
)

r_handler = ro.r


class RPredictor(BaseLanguagePredictor):
    def __init__(
        self,
    ):
        super(RPredictor, self).__init__()

    def configure(self, params):
        super(RPredictor, self).configure(params)

        if self._positive_class_label is None:
            self._positive_class_label = ro.rinterface.NULL
        if self._negative_class_label is None:
            self._negative_class_label = ro.rinterface.NULL
        if self._class_labels is None:
            self._class_labels = ro.rinterface.NULL
        else:
            self._class_labels = StrVector(self._class_labels)

        r_handler.source(R_COMMON_PATH)
        r_handler.source(R_SCORE_PATH)
        r_handler.init(self._custom_model_path, self._target_type.value)
        if self._target_type == TargetType.UNSTRUCTURED:
            for hook_name in [
                CustomHooks.LOAD_MODEL,
                CustomHooks.SCORE_UNSTRUCTURED,
            ]:
                if not hasattr(r_handler, hook_name):
                    raise DrumCommonException(
                        "In '{}' mode hook '{}' must be provided.".format(
                            TargetType.UNSTRUCTURED.value, hook_name
                        )
                    )

        self._model = r_handler.load_serialized_model(self._custom_model_path)

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        formats.add(PayloadFormat.MTX)
        return formats

    def has_read_input_data_hook(self):
        return bool(r_handler.has_read_input_data_hook()[0])

    def predict(self, **kwargs):
        input_binary_data = kwargs.get(StructuredDtoKeys.BINARY_DATA)
        mimetype = kwargs.get(StructuredDtoKeys.MIMETYPE)
        start_predict = time.time()
        predictions = r_handler.outer_predict(
            self._target_type.value,
            binary_data=ro.rinterface.NULL
            if input_binary_data is None
            else ro.vectors.ByteVector(input_binary_data),
            mimetype=ro.rinterface.NULL if mimetype is None else mimetype,
            model=self._model,
            positive_class_label=self._positive_class_label,
            negative_class_label=self._negative_class_label,
            class_labels=self._class_labels,
        )
        with localconverter(ro.default_converter + pandas2ri.converter):
            py_data_object = ro.conversion.rpy2py(predictions)

        # in case of regression, array is returned
        if isinstance(py_data_object, numpy.ndarray):
            py_data_object = pd.DataFrame({REGRESSION_PRED_COLUMN: py_data_object})

        end_predict = time.time()
        execution_time_ms = (end_predict - start_predict) * 1000

        if not isinstance(py_data_object, pd.DataFrame):
            error_message = (
                "Expected predictions type: {}, actual: {}. "
                "Are you trying to run binary classification without class labels provided?".format(
                    pd.DataFrame, type(py_data_object)
                )
            )
            logger.error(error_message)
            raise DrumCommonException(error_message)

        self.monitor(kwargs, py_data_object, execution_time_ms)
        return py_data_object

    # TODO: check test coverage for all possible cases: return None/str/bytes, and casting.
    def predict_unstructured(self, data, **kwargs):
        def _r_is_character(r_val):
            _is_character = ro.r("is.character")
            return bool(_is_character(r_val))

        def _r_is_raw(r_val):
            _is_raw = ro.r("is.raw")
            return bool(_is_raw(r_val))

        def _r_is_null(r_val):
            return r_val == ro.rinterface.NULL

        def _cast_r_to_py(r_val):
            # TODO: consider checking type against rpy2 proxy object like: isinstance(list_data_kwargs, ro.vectors.ListVector)
            # instead of calling R interpreter
            if _r_is_null(r_val):
                return None
            elif _r_is_raw(r_val):
                return bytes(r_val)
            elif _r_is_character(r_val):
                # Any scalar value is returned from R as one element vector,
                # so get this value.
                return str(r_val[0])
            else:
                raise DrumCommonException(
                    "Can not convert R value {} type {}".format(r_val, type(r_val))
                )

        def _rlist_to_dict(rlist):
            if _r_is_null(rlist):
                return None
            return {str(k): _cast_r_to_py(v) for k, v in rlist.items()}

        data_binary_or_text = data

        if UnstructuredDtoKeys.QUERY in kwargs:
            kwargs[UnstructuredDtoKeys.QUERY] = ro.ListVector(kwargs[UnstructuredDtoKeys.QUERY])

        # if data_binary_or_text is str it will be auto converted into R character type;
        # otherwise if it is bytes, manually convert it into byte vector (raw)
        r_data_binary_or_text = data_binary_or_text
        if isinstance(data_binary_or_text, bytes):
            r_data_binary_or_text = ro.vectors.ByteVector(data_binary_or_text)

        kwargs_filtered = {k: v for k, v in kwargs.items() if v is not None}
        list_data_kwargs = r_handler.predict_unstructured(
            model=self._model, data=r_data_binary_or_text, **kwargs_filtered
        )
        if isinstance(list_data_kwargs, ro.vectors.ListVector):
            ret = _cast_r_to_py(list_data_kwargs[0]), _rlist_to_dict(list_data_kwargs[1])
        else:
            raise DrumCommonException(
                "Wrong type returned in unstructured mode: {}".format(type(list_data_kwargs))
            )

        return ret

    def transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported for R")
