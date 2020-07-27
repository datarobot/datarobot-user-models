import logging
import numpy
import os
import pandas as pd

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
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
R_COMMON_PATH = os.path.abspath(os.path.join(CUR_DIR, "..", "r_common_code", "common.R",))

r_handler = ro.r


class RPredictor(BaseLanguagePredictor):
    def __init__(self,):
        super(RPredictor, self).__init__()

    def configure(self, params):
        super(RPredictor, self).configure(params)

        if self._positive_class_label is None:
            self._positive_class_label = ro.rinterface.NULL
        if self._negative_class_label is None:
            self._negative_class_label = ro.rinterface.NULL

        r_handler.source(R_COMMON_PATH)
        r_handler.source(R_SCORE_PATH)
        r_handler.init(self._custom_model_path)
        self._model = r_handler.load_serialized_model(self._custom_model_path)

    def predict(self, df):
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)

        predictions = r_handler.outer_predict(
            r_df,
            model=self._model,
            positive_class_label=self._positive_class_label,
            negative_class_label=self._negative_class_label,
        )
        with localconverter(ro.default_converter + pandas2ri.converter):
            py_df = ro.conversion.rpy2py(predictions)

        if isinstance(py_df, numpy.ndarray):
            py_df = pd.DataFrame({"Predictions": py_df})

        if not isinstance(py_df, pd.DataFrame):
            error_message = (
                "Expected predictions type: {}, actual: {}. "
                "Are you trying to run binary classification without class labels provided?".format(
                    pd.DataFrame, type(py_df)
                )
            )
            logger.error(error_message)
            raise DrumCommonException(error_message)

        return py_df
