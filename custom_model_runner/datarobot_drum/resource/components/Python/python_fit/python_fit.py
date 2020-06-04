import logging
import sys
from pathlib import Path

import pandas as pd
from mlpiper.components.connectable_component import ConnectableComponent

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
from datarobot_drum.drum.model_adapter import PythonModelAdapter

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PythonFit(ConnectableComponent):
    def __init__(self, engine):
        super(PythonFit, self).__init__(engine)
        self.target_name = None
        self.output_dir = None
        self.estimator = None
        self.positive_class_label = None
        self.negative_class_label = None
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
        self.weights = self._params["weights"]
        self.weights_filename = self._params["weightsFilename"]
        self.target_filename = self._params.get("targetFilename")
        self.num_rows = self._params["numRows"]

        self._model_adapter = PythonModelAdapter(self.custom_model_path)
        sys.path.append(self.custom_model_path)
        self._model_adapter.load_custom_hooks()

    def _materialize(self, parent_data_objs, user_data):
        df = pd.read_csv(self.input_filename)
        if self.num_rows == "ALL":
            self.num_rows = len(df)
        else:
            self.num_rows = int(self.num_rows)
        if self.target_filename:
            X = df.head(self.num_rows)
            y = pd.read_csv(self.target_filename, index_col=False).head(self.num_rows)
            assert len(y.columns) == 1
            assert len(X) == len(y)
            y = y.iloc[:, 0]
        else:
            X = df.drop(self.target_name, axis=1).head(self.num_rows)
            y = df[self.target_name].head(self.num_rows)

        if self.weights_filename:
            row_weights = pd.read_csv(self.weights_filename).head(self.num_rows)
        elif self.weights:
            if self.weights not in X.columns:
                raise ValueError(
                    "The column name {} is not one of the columns in "
                    "your training data".format(self.weights)
                )
            row_weights = X[self.weights]
        else:
            row_weights = None

        class_order = (
            [self.negative_class_label, self.positive_class_label]
            if self.negative_class_label
            else None
        )
        self._model_adapter.fit(
            X, y, output_dir=self.output_dir, class_order=class_order, row_weights=row_weights,
        )

        make_sure_artifact_is_small(self.output_dir)
        return []


def make_sure_artifact_is_small(output_dir):
    MEGABYTE = 1024 * 1024
    GIGABYTE = 1024 * MEGABYTE
    root_directory = Path(output_dir)
    dir_size = sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())
    logger.info("Artifact directory has been filled to {} Megabytes".format(dir_size / MEGABYTE))
    assert dir_size < 10 * GIGABYTE
