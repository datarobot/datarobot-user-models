# flake8: noqa E402
import logging
import numpy as np
import os
import sklearn
import sys

uwsgi_loaded = False
try:
    import uwsgi

    uwsgi_loaded = True
except ImportError:
    pass

from mlpiper.components.restful.flask_route import FlaskRoute
from mlpiper.components.restful_component import RESTfulComponent


logger = logging.getLogger(__name__)


class EchoRESTfulServingTest(RESTfulComponent):
    JSON_KEY_NAME = "data"

    def __init__(self, engine):
        super(EchoRESTfulServingTest, self).__init__(engine)
        self._loading_error = None
        self._params = {}
        self._verbose = self._logger.isEnabledFor(logging.DEBUG)

        self._metric1 = None
        self._metric2 = None
        self._metric3 = None
        self._metric4 = None

        self.info_json = {
            "sample_keyword": EchoRESTfulServingTest.JSON_KEY_NAME,
            "python": "{}.{}.{}".format(
                sys.version_info[0], sys.version_info[1], sys.version_info[2]
            ),
            "numpy": np.version.version,
            "sklearn": sklearn.__version__,
        }

    def load_model_callback(self, model_path, stream, version):
        pass

    @FlaskRoute("/predict")
    def predict(self, url_params, form_params):
        if len(form_params) == 0:
            return 200, self._empty_predict()

        try:
            arr = np.array(form_params[EchoRESTfulServingTest.JSON_KEY_NAME])
            logger.info("Array: {}".format(arr))
            return 200, {"prediction": np.sum(arr)}
        except Exception as e:
            error_json = {"error": "Error performing prediction: {}".format(e)}
            error_json.update(self.info_json)
            return 404, error_json


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

    EchoRESTfulServingTest.run(port=args.port, model_path=args.input_model)
