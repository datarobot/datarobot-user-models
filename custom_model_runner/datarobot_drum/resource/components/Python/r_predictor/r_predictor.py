import logging
import numpy
import os
import pandas as pd

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
from mlpiper.components.connectable_component import ConnectableComponent

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
    exit(1)


pandas2ri.activate()
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
R_SCORE_PATH = os.path.join(CUR_DIR, "score.R")

r_handler = ro.r


class RPredictor(ConnectableComponent):
    def __init__(self, engine):
        super(RPredictor, self).__init__(engine)
        self.model = None
        self.positive_class_label = ro.rinterface.NULL
        self.negative_class_label = ro.rinterface.NULL
        self.custom_model_path = None

    def configure(self, params):
        super(RPredictor, self).configure(params)
        self.custom_model_path = self._params["__custom_model_path__"]
        pos_cl_label = self._params.get("positiveClassLabel")
        neg_cl_label = self._params.get("negativeClassLabel")
        if pos_cl_label:
            self.positive_class_label = pos_cl_label
        if neg_cl_label:
            self.negative_class_label = neg_cl_label

        r_handler.source(R_SCORE_PATH)
        r_handler.init(self.custom_model_path)
        self.model = r_handler.load_serialized_model(self.custom_model_path)

    def _materialize(self, parent_data_objs, user_data):
        # This class is a ConnectableComponent utilizing mlpiper infrastructure.
        # Items existance in parent_data_objs is guaranteed by pipeline DAG verification by mlpiper
        # and by implementation of the previous step in pipeline.
        df = parent_data_objs[0]
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)

        predictions = r_handler.outer_predict(
            r_df,
            model=self.model,
            positive_class_label=self.positive_class_label,
            negative_class_label=self.negative_class_label,
        )
        with localconverter(ro.default_converter + pandas2ri.converter):
            py_df = ro.conversion.rpy2py(predictions)

        if isinstance(py_df, numpy.ndarray):
            py_df = pd.DataFrame({"Predictions": py_df})

        if not isinstance(py_df, pd.DataFrame):
            logger.error(
                "Expected predictions type: {}, actual: {}. "
                "Are you trying to run binary classification without class labels provided?".format(
                    pd.DataFrame, type(py_df)
                )
            )
            exit(1)

        return [py_df]
