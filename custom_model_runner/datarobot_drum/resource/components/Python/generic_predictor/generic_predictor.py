import logging

import pandas as pd

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
from datarobot_drum.drum.common import RunLanguage
from datarobot_drum.drum.exceptions import DrumCommonException

from mlpiper.components.connectable_component import ConnectableComponent


class GenericPredictorComponent(ConnectableComponent):
    def __init__(self, engine):
        super(GenericPredictorComponent, self).__init__(engine)
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self._run_language = None
        self._predictor = None

    def configure(self, params):
        super(GenericPredictorComponent, self).configure(params)
        self._run_language = RunLanguage(params.get("run_language"))

        if self._run_language == RunLanguage.PYTHON:
            from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
                PythonPredictor,
            )

            self._predictor = PythonPredictor()
        elif self._run_language == RunLanguage.JAVA:
            from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import (
                JavaPredictor,
            )

            self._predictor = JavaPredictor()
        elif self._run_language == RunLanguage.R:
            # this import is here, because RPredictor imports rpy library,
            # which is not installed for Java and Python cases.
            from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import RPredictor

            self._predictor = RPredictor()
        else:
            raise DrumCommonException(
                "Prediction server doesn't support language: {} ".format(self._run_language)
            )

        self._predictor.configure(params)

    def _materialize(self, parent_data_objs, user_data):
        input_filename = self._params.get("input_filename", "default-string-value")
        df = pd.read_csv(input_filename)
        predictions = self._predictor.predict(df)
        output_filename = self._params.get("output_filename")
        predictions.to_csv(output_filename, index=False)
        return [predictions]
