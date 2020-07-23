import logging
import pandas as pd
import sys

from flask import request

from mlpiper.components.restful.flask_route import FlaskRoute
from mlpiper.components.restful_component import RESTfulComponent

from datarobot_drum.drum.common import RunLanguage

from datarobot_drum.drum.server import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_ENTITY,
)


class UwsgiServing(RESTfulComponent):
    JSON_KEY_NAME = "data"

    def __init__(self, engine):
        super(UwsgiServing, self).__init__(engine)
        self._model = None
        self._model_loading_error = None
        self._params = {}
        self._verbose = self._logger.isEnabledFor(logging.DEBUG)
        self._run_language = None

        self.info_json = {
            "sample_keyword": UwsgiServing.JSON_KEY_NAME,
            "python": "{}.{}.{}".format(
                sys.version_info[0], sys.version_info[1], sys.version_info[2]
            ),
            "worker": self.get_wid(),
        }
        self._predictor = None

    def configure(self, params):
        """
        @brief      It is called in within the 'deputy' context
        """
        self._logger.info(
            "Configure component with input params, name: {}, params: {}".format(
                self.name(), params
            )
        )
        self._params = params
        self._run_language = RunLanguage(params.get("run_language"))

    def load_model_callback(self, model_path, stream, version):
        self._logger.info(sys.version_info)
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

    def _empty_predict(self):
        model_loaded = True if self._model else False

        result_json = {
            "message": "got empty predict",
            "expected_input_format": '{{"data":[<vector>]}}',
            "model_loaded": model_loaded,
            "model_class": str(type(self._model)),
        }

        if model_loaded is False and self._model_loading_error:
            result_json["model_load_error"] = self._model_loading_error

        if self._model:
            if hasattr(self._model, "n_features_"):
                result_json["n_features"] = self._model.n_features_
                result_json[
                    "expected_input_format"
                ] += ", where vector has {} comma separated values".format(self._model.n_features_)

        result_json.update(self.info_json)

        return result_json

    @FlaskRoute("/")
    def ping(self, url_params, form_params):
        return 200, "Hello World"

    @FlaskRoute("/predict/")
    def predict(self, url_params, form_params):
        response_status = HTTP_200_OK
        file_key = "X"
        # logger.debug("Entering predict() endpoint")
        REGRESSION_PRED_COLUMN = "Predictions"
        filename = request.files[file_key] if file_key in request.files else None
        # logger.debug("Filename provided under X key: {}".format(filename))

        if not filename:
            wrong_key_error_message = "Samples should be provided as a csv file under `{}` key.".format(
                file_key
            )
            # logger.error(wrong_key_error_message)
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
