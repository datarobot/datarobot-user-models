import logging
import os

from mlpiper.components.connectable_component import ConnectableComponent

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
from datarobot_drum.drum.utils import shared_fit_preprocessing, make_sure_artifact_is_small

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
R_FIT_PATH = os.path.join(CUR_DIR, "fit.R")
R_COMMON_PATH = os.path.abspath(
    os.path.join(
        CUR_DIR,
        "..",
        "..",
        "..",
        "..",
        "drum",
        "language_predictors",
        "r_common_code",
        "common.R",
    )
)

r_handler = ro.r


class RFit(ConnectableComponent):
    def __init__(self, engine):
        super(RFit, self).__init__(engine)
        self.target_name = None
        self.output_dir = None
        self.estimator = None
        self.positive_class_label = None
        self.negative_class_label = None
        self.class_labels = None
        self.custom_model_path = None
        self.input_filename = None
        self.weights = None
        self.weights_filename = None
        self.target_filename = None
        self.num_rows = None

    def configure(self, params):
        super(RFit, self).configure(params)
        self.custom_model_path = self._params["__custom_model_path__"]
        self.input_filename = self._params["inputFilename"]
        self.target_name = self._params.get("targetColumn")
        self.output_dir = self._params["outputDir"]
        self.positive_class_label = self._params.get("positiveClassLabel")
        self.negative_class_label = self._params.get("negativeClassLabel")
        self.class_labels = self._params.get("classLabels")
        self.weights = self._params["weights"]
        self.weights_filename = self._params["weightsFilename"]
        self.target_filename = self._params.get("targetFilename")
        self.num_rows = self._params["numRows"]

        r_handler.source(R_COMMON_PATH)
        r_handler.source(R_FIT_PATH)
        r_handler.init(self.custom_model_path)

    def _materialize(self, parent_data_objs, user_data):
        # make sure our output dir ends with a slash
        if self.output_dir[-1] != "/":
            self.output_dir += "/"

        if self.weights:
            weights = self.weights.replace("-", ".")
        else:
            weights = ro.NULL

        if self.target_name:
            target_name = self.target_name.replace("-", ".").replace("_", ".")
        else:
            target_name = ro.NULL

        r_handler.outer_fit(
            self.output_dir,
            self.input_filename,
            self.target_filename or ro.NULL,
            target_name,
            self.num_rows,
            self.weights_filename or ro.NULL,
            weights,
            self.positive_class_label or ro.NULL,
            self.negative_class_label or ro.NULL,
            ro.StrVector(self.class_labels) if self.class_labels else ro.NULL,
        )

        make_sure_artifact_is_small(self.output_dir)
        return []
