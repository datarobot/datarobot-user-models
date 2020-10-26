import logging
import sys
import os

from mlpiper.components.connectable_component import ConnectableComponent

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.utils import shared_fit_preprocessing, make_sure_artifact_is_small

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PythonFit(ConnectableComponent):
    def __init__(self, engine):
        super(PythonFit, self).__init__(engine)
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
        self._model_adapter = None
        self.num_rows = None

    def configure(self, params):
        super(PythonFit, self).configure(params)
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

        self._model_adapter = PythonModelAdapter(self.custom_model_path)
        sys.path.append(self.custom_model_path)
        self._model_adapter.load_custom_hooks()

    def _materialize(self, parent_data_objs, user_data):

        X, y, class_order, row_weights = shared_fit_preprocessing(self)

        self._model_adapter.fit(
            X, y, output_dir=self.output_dir, class_order=class_order, row_weights=row_weights
        )

        make_sure_artifact_is_small(self.output_dir)
        return []
