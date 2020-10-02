import logging
import numpy
import os
import pandas as pd

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    TargetType,
    CustomHooks,
    REGRESSION_PRED_COLUMN,
    UnstructuredDtoKeys,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
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

    def predict(self, input_filename):
        predictions = r_handler.outer_predict(
            input_filename,
            self._target_type.value,
            model=self._model,
            positive_class_label=self._positive_class_label,
            negative_class_label=self._negative_class_label,
        )
        with localconverter(ro.default_converter + pandas2ri.converter):
            py_data_object = ro.conversion.rpy2py(predictions)

        if self._target_type == TargetType.UNSTRUCTURED:
            py_data_object = str(py_data_object)
        else:
            # in case of regression, array is returned
            if isinstance(py_data_object, numpy.ndarray):
                py_data_object = pd.DataFrame({REGRESSION_PRED_COLUMN: py_data_object})

            if not isinstance(py_data_object, pd.DataFrame):
                error_message = (
                    "Expected predictions type: {}, actual: {}. "
                    "Are you trying to run binary classification without class labels provided?".format(
                        pd.DataFrame, type(py_data_object)
                    )
                )
                logger.error(error_message)
                raise DrumCommonException(error_message)
        return py_data_object

    def predict_unstructured(self, **kwargs):
        def _r_is_character(r_val):
            _is_character = ro.r("is.character")
            return bool(_is_character(r_val))

        def _rlist_to_dict(rlist):

            dd = {}
            for k, v in rlist.items():
                k = str(k)
                if k == UnstructuredDtoKeys.DATA:
                    if v == ro.rinterface.NULL:
                        dd[k] = None
                    else:
                        dd[k] = str(v[0]) if _r_is_character(v) else bytes(v)
                else:
                    # Any scalar value is returned from R as one element vector,
                    # so get this value.
                    dd[k] = str(v[0]) if v != ro.rinterface.NULL else None
            return dd

        data_binary_or_text = kwargs.get(UnstructuredDtoKeys.DATA, None)

        if UnstructuredDtoKeys.DATA in kwargs:
            kwargs.pop(UnstructuredDtoKeys.DATA)

        # if data_binary_or_text is str it will be auto converted into R character type;
        # otherwise if it is bytes, manually convert it into byte vector (raw)
        r_data_binary_or_text = data_binary_or_text
        if isinstance(data_binary_or_text, bytes):
            r_data_binary_or_text = ro.rinterface.ByteSexpVector(data_binary_or_text)

        kwargs_filtered = {k: v for k, v in kwargs.items() if v is not None}
        kwargs_filtered[UnstructuredDtoKeys.DATA] = r_data_binary_or_text
        ret_rlist = r_handler.predict_unstructured(model=self._model, **kwargs_filtered)
        ret_dict = _rlist_to_dict(ret_rlist)

        return ret_dict
