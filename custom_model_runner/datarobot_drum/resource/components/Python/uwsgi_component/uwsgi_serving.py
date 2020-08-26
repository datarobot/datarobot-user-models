import logging
import os
import pandas as pd
import sys

from flask import request

from mlpiper.components.restful.flask_route import FlaskRoute
from mlpiper.components.restful_component import RESTfulComponent

from datarobot_drum.drum.common import RunLanguage, URL_PREFIX_ENV_VAR_NAME, REGRESSION_PRED_COLUMN

from datarobot_drum.drum.server import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_ENTITY,
)


class UwsgiServing(RESTfulComponent):
    def __init__(self, engine):
        super(UwsgiServing, self).__init__(engine)
        self._params = {}
        self._verbose = self._logger.isEnabledFor(logging.DEBUG)
        self._run_language = None

        self.info_json = {
            "python": "{}.{}.{}".format(
                sys.version_info[0], sys.version_info[1], sys.version_info[2]
            ),
            "worker_id": self.get_wid(),
        }
        self._predictor = None

    def configure(self, params):
        """
        @brief      It is called in within the 'deputy' context
        """
        super(UwsgiServing, self).configure(params)
        self._logger.info(
            "Configure component with input params, name: {}, params: {}".format(
                self.name(), params
            )
        )
        self._run_language = RunLanguage(params.get("run_language"))

    def load_model_callback(self, model_path, stream, version):
        self._logger.info(self.info_json)
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
            from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import RPredictor

            self._predictor = RPredictor()
        self._predictor.configure(self._params)

    @FlaskRoute("{}/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")), methods=["GET"])
    def ping(self, url_params, form_params):
        return 200, {"message": "OK"}

    @FlaskRoute("{}/health/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")), methods=["GET"])
    def health(self, url_params, form_params):
        return 200, {"message": "OK"}

    @FlaskRoute("{}/predict/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")), methods=["POST"])
    def predict(self, url_params, form_params):
        response_status = HTTP_200_OK
        file_key = "X"
        filename = request.files[file_key] if file_key in request.files else None

        if not filename:
            wrong_key_error_message = "Samples should be provided as a csv file under `{}` key.".format(
                file_key
            )
            response_status = HTTP_422_UNPROCESSABLE_ENTITY
            return response_status, {"message": "ERROR: " + wrong_key_error_message}

        in_df = pd.read_csv(filename)
        out_df = self._predictor.predict(in_df)

        num_columns = len(out_df.columns)
        # float32 is not JSON serializable, so cast to float, which is float64
        out_df = out_df.astype("float")
        if num_columns == 1:
            # df.to_json() is much faster.
            # But as it returns string, we have to assemble final json using strings.
            df_json = out_df[REGRESSION_PRED_COLUMN].to_json(orient="records")
            response_json = '{{"predictions":{df_json}}}'.format(df_json=df_json)
        elif num_columns == 2:
            # df.to_json() is much faster.
            # But as it returns string, we have to assemble final json using strings.
            df_json_str = out_df.to_json(orient="records")
            response_json = '{{"predictions":{df_json}}}'.format(df_json=df_json_str)
        else:
            ret_str = (
                "Predictions dataframe has {} columns; "
                "Expected: 1 - for regression, 2 - for binary classification.".format(num_columns)
            )
            response_json = {"message": "ERROR: " + ret_str}
            response_status = HTTP_422_UNPROCESSABLE_ENTITY

        return response_status, response_json
