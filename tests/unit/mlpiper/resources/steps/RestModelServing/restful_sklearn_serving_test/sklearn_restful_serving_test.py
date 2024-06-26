# flake8: noqa E402
import logging
import numpy as np
import os
import pickle
import pprint
import random
import sklearn
import sys
import warnings

uwsgi_loaded = False
try:
    import uwsgi

    uwsgi_loaded = True
except ImportError:
    pass

from mlpiper.components.restful.flask_route import FlaskRoute
from mlpiper.components.restful_component import RESTfulComponent
from mlpiper.components.restful.metric import Metric
from mlpiper.components.restful.metric import MetricType
from mlpiper.components.restful.metric import MetricRelation


class SklearnRESTfulServingTest(RESTfulComponent):
    JSON_KEY_NAME = "data"

    def __init__(self, engine):
        super(SklearnRESTfulServingTest, self).__init__(engine)
        self._model = None
        self._loading_error = None
        self._params = {}
        self._verbose = self._logger.isEnabledFor(logging.DEBUG)

        self._metric1 = None
        self._metric2 = None
        self._metric3 = None
        self._metric4 = None

        self.info_json = {
            "sample_keyword": SklearnRESTfulServingTest.JSON_KEY_NAME,
            "python": "{}.{}.{}".format(
                sys.version_info[0], sys.version_info[1], sys.version_info[2]
            ),
            "numpy": np.version.version,
            "sklearn": sklearn.__version__,
        }

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

        self._metric1 = Metric(
            "requests.per.win.time",
            hidden=True,
            metric_type=MetricType.COUNTER_PER_TIME_WINDOW,
        )

        self._metric2 = Metric(
            name="distance.per.req",
            title="Avg Distance / time-window [per-reqs]",
            metric_type=MetricType.COUNTER_PER_TIME_WINDOW,
            value_type=float,
            metric_relation=MetricRelation.AVG_PER_REQUEST,
        )

        self._metric3 = Metric(
            name="distance.per.counter",
            title="Avg Distance / time-window [counter.per.reqs]",
            metric_type=MetricType.COUNTER_PER_TIME_WINDOW,
            value_type=float,
            metric_relation=MetricRelation.DIVIDE_BY,
            related_metric=self._metric1,
        )

        self._metric4 = Metric(
            name="classification",
            title="Prediction Distribution",
            metric_type=MetricType.COUNTER_PER_TIME_WINDOW,
            metric_relation=MetricRelation.BAR_GRAPH,
            related_metric=[(self._metric2, "metric2"), (self._metric3, "metric3")],
        )

    def load_model_callback(self, model_path, stream, version):
        self._logger.info(sys.version_info)

        self._logger.info(
            "Model is loading, wid: {}, path: {}".format(self.get_wid(), model_path)
        )
        self._logger.info("params: {}".format(pprint.pformat(self._params)))
        model = None

        with warnings.catch_warnings(record=True) as warns:
            try:
                with open(model_path, "rb") as f:
                    self._loading_error = None
                    model = (
                        pickle.load(f)
                        if sys.version_info[0] < 3
                        else pickle.load(f, encoding="latin1")
                    )

                    if self._verbose:
                        self._logger.debug("Un-pickled model: {}".format(self._model))
                    self._logger.debug("Model loaded successfully!")

            except Exception as e:
                warn_str = ""
                if len(warns) > 0:
                    warn_str = "{}".format(warns[-1].message)
                self._logger.error(
                    "Model loading warning: {}; Model loading error: {}".format(
                        warn_str, e
                    )
                )

                # Not sure we want to throw exception only to move to a non model mode
                if self._params.get("ignore-incompatible-model", True):
                    self._logger.info(
                        "New model could not be loaded, due to error: {}".format(e)
                    )
                    if self._model is None:
                        self._loading_error = (
                            "Model loading warning: {}; "
                            "Model loading error: {}".format(warn_str, str(e))
                        )
                    else:
                        raise Exception(
                            "Model loading warning: {}; Model loading error: {}".format(
                                warn_str, e
                            )
                        )

        # This line should be reached only if
        #  a) model loaded successfully
        #  b) model loading failed but it can be ignored
        if model is not None:
            self._model = model

    def _empty_predict(self):
        model_loaded = True if self._model else False

        result_json = {
            "message": "got empty predict",
            "expected_input_format": '{{"data":[<vector>]}}',
            "model_loaded": model_loaded,
            "model_class": str(type(self._model)),
        }

        if model_loaded is False and self._loading_error:
            result_json["model_load_error"] = self._loading_error

        if self._model:
            if hasattr(self._model, "n_features_"):
                result_json["n_features"] = self._model.n_features_
                result_json[
                    "expected_input_format"
                ] += ", where vector has {} comma separated values".format(
                    self._model.n_features_
                )

        result_json.update(self.info_json)

        return result_json

    @FlaskRoute("/predict")
    def predict(self, url_params, form_params):

        if len(form_params) == 0:
            return 200, self._empty_predict()

        elif not self._model:
            if self._loading_error:
                return_json = {
                    "error": "Failed loading model: {}".format(self._loading_error)
                }
            else:
                return_json = {"error": "Model not loaded yet - please set a model"}
            return_json.update(self.info_json)
            return 404, return_json

        elif SklearnRESTfulServingTest.JSON_KEY_NAME not in form_params:
            msg = "Unexpected json format for prediction! Missing '{}' key in: {}".format(
                SklearnRESTfulServingTest.JSON_KEY_NAME, form_params
            )
            self._logger.error(msg)
            error_json = {"error": msg}
            error_json.update(self.info_json)
            return 404, error_json
        else:
            try:
                two_dim_array = np.array(
                    [form_params[SklearnRESTfulServingTest.JSON_KEY_NAME]]
                )
                prediction = self._model.predict(two_dim_array)
                if self._verbose:
                    self._logger.debug(
                        "predict, url_params: {}, form_params: {}".format(
                            url_params, form_params
                        )
                    )
                    self._logger.debug(
                        "type<form_params>: {}\n{}".format(
                            type(form_params), form_params
                        )
                    )
                    self._logger.debug(
                        "type(two_dim_array): {}\n{}".format(
                            type(two_dim_array), two_dim_array
                        )
                    )
                    self._logger.debug(
                        "prediction: {}, type: {}".format(
                            prediction[0], type(prediction[0])
                        )
                    )
                return 200, {"prediction": prediction[0]}
            except Exception as e:
                error_json = {"error": "Error performing prediction: {}".format(e)}
                error_json.update(self.info_json)
                return 404, error_json

    @FlaskRoute("/metric-test")
    def metric_test(self, url_params, form_params):
        try:
            self._metric1.increase(1)

            confident_num = random.random()

            # The values in the graphs are supposed to be the same
            self._metric2.increase(confident_num)
            self._metric3.increase(confident_num * 2)
            return 200, {"response": "ok"}
        except Exception as ex:
            return 404, {"message": str(ex)}


if __name__ == "__main__":
    import argparse

    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to listen on")
    parser.add_argument("input_model", help="Path of input model to create")
    parser.add_argument(
        "--log_level", choices=log_levels.keys(), default="info", help="Logging level"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_model):
        raise Exception("Model file {} does not exists".format(args.input_model))

    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s"
    )
    logging.getLogger("mlpiper").setLevel(log_levels[args.log_level])

    SklearnRESTfulServingTest.run(port=args.port, model_path=args.input_model)
