import logging

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    RunLanguage,
    TargetType,
    TARGET_TYPE_ARG_KEYWORD,
    UnstructuredDtoKeys,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.resource.unstructured_helpers import (
    _resolve_incoming_unstructured_data,
    _resolve_outgoing_unstructured_data,
    _is_text_mimetype,
)
from datarobot_drum.drum.utils import split_params_to_dict

from mlpiper.components.connectable_component import ConnectableComponent


class GenericPredictorComponent(ConnectableComponent):
    def __init__(self, engine):
        super(GenericPredictorComponent, self).__init__(engine)
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self._run_language = None
        self._predictor = None
        self._target_type = None

    def configure(self, params):
        super(GenericPredictorComponent, self).configure(params)
        self._run_language = RunLanguage(params.get("run_language"))
        self._target_type = TargetType(params[TARGET_TYPE_ARG_KEYWORD])

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
        input_filename = self._params["input_filename"]
        output_filename = self._params.get("output_filename")

        additional_params = self._params.get("params")

        def _resolve_params(params):
            params_dict = split_params_to_dict(params)

            for key in [UnstructuredDtoKeys.DATA]:
                if key in params_dict:
                    raise DrumCommonException(
                        "Parameter name '{}' is not allowed in the additional params".format(key)
                    )
            return params_dict

        if self._target_type == TargetType.UNSTRUCTURED:
            kwargs_params = {}

            params = _resolve_params(additional_params)

            with open(input_filename, "rb") as f:
                data_binary = f.read()

            data_binary_or_text, mimetype, charset = _resolve_incoming_unstructured_data(
                data_binary,
                params.get(UnstructuredDtoKeys.MIMETYPE, None),
                params.get(UnstructuredDtoKeys.CHARSET, None),
            )

            kwargs_params[UnstructuredDtoKeys.DATA] = data_binary_or_text

            kwargs_params.update(params)

            unstructured_response = self._predictor.predict_unstructured(**kwargs_params)
            _, response_mimetype, response_charset = _resolve_outgoing_unstructured_data(
                unstructured_response
            )

            # only for screen printout convenience we take pred data directly from unstructured_response
            pred_data = unstructured_response.get(UnstructuredDtoKeys.DATA, None)
            if isinstance(pred_data, bytes):
                with open(output_filename, "wb") as f:
                    f.write(pred_data)
            else:
                if pred_data is None:
                    pred_data = "Return value from prediction is: None (NULL in R)"
                with open(output_filename, "w") as f:
                    f.write(pred_data)

        else:
            predictions = self._predictor.predict(input_filename)
            predictions.to_csv(output_filename, index=False)
        return []
