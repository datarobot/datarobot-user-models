"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os

import pandas as pd
from scipy.sparse import coo_matrix

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import (
    CustomHooks,
    LOGGER_NAME_PREFIX,
    PayloadFormat,
    StructuredDtoKeys,
    TargetType,
    UnstructuredDtoKeys,
    PRED_COLUMN,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.drum.utils.dataframe import extract_additional_columns
from datarobot_drum.drum.utils.stacktraces import capture_R_traceback_if_errors

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, StrVector
    from rpy2.robjects.conversion import localconverter

except ImportError:
    error_message = (
        "rpy2 package is not installed."
        "Install datarobot-drum using 'pip install datarobot-drum[R]'"
        "Available for Python>=3.6"
    )
    logger.error(error_message)
    raise DrumCommonException(error_message)


pandas2ri.activate()
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
R_SCORE_PATH = os.path.join(CUR_DIR, "score.R")
R_COMMON_PATH = os.path.abspath(
    os.path.join(
        CUR_DIR,
        "..",
        "r_common_code",
        "common.R",
    )
)

r_handler = ro.r


class RPredictor(BaseLanguagePredictor):
    def mlpiper_configure(self, params):
        super(RPredictor, self).mlpiper_configure(params)

        r_handler.source(R_COMMON_PATH)
        r_handler.source(R_SCORE_PATH)
        r_handler.init(self._code_dir, self.target_type.value)
        if self.target_type == TargetType.UNSTRUCTURED:
            for hook_name in [
                CustomHooks.LOAD_MODEL,
                CustomHooks.SCORE_UNSTRUCTURED,
            ]:
                if not hasattr(r_handler, hook_name):
                    raise DrumCommonException(
                        "In '{}' mode hook '{}' must be provided.".format(
                            TargetType.UNSTRUCTURED.value, hook_name
                        )
                    )

        self._model = r_handler.load_serialized_model(self._code_dir, self.target_type.value)

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        formats.add(PayloadFormat.MTX)
        return formats

    def has_read_input_data_hook(self):
        return bool(r_handler.has_read_input_data_hook()[0])

    @staticmethod
    def _get_sparse_colnames(kwargs):
        sparse_colnames = kwargs.get(StructuredDtoKeys.SPARSE_COLNAMES)
        if sparse_colnames:
            return ro.vectors.StrVector(sparse_colnames)
        return ro.NULL

    def _replace_sanitized_class_names(self, predictions):
        """Match prediction data labels to project class labels.
        Note that this contains only logic specific to R name
        sanitization and relies on marshal_predictions() for
        language neutral cases like matching floats and bools"""

        # get class labels
        if not self.class_ordering:
            raise DrumCommonException("Class labels not available for classification.")

        extra_model_output = None
        prediction_labels = predictions.columns

        # if the labels match then do nothing
        if set(prediction_labels) == set(self.class_ordering):
            return predictions, extra_model_output

        # check for match after make.names is applied to class labels
        sanitized_request_labels = ro.r["make.names"](self.class_ordering)
        if set(sanitized_request_labels) <= set(prediction_labels):
            if len(set(sanitized_request_labels)) != len(sanitized_request_labels):
                raise DrumCommonException("Class label names are ambiguous.")
            if len(sanitized_request_labels) < len(prediction_labels):
                # Extract the additional columns and consider them as the extra model output
                predictions, extra_model_output = extract_additional_columns(
                    predictions, sanitized_request_labels
                )
            label_map = dict(zip(sanitized_request_labels, self.class_ordering))
            # return class labels in the same order as prediction labels
            ordered_labels = [label_map[l] for l in prediction_labels if l in label_map]
            predictions.columns = ordered_labels
            return predictions, extra_model_output

        def floatify(f):
            try:
                return float(f)
            except ValueError:
                return f

        # check for match after sanitized float strings (e.g. X7.1) are converted to plain floats
        if any(isinstance(l, str) and l.startswith("X") for l in prediction_labels):
            relevant_prediction_labels = [
                l for l in prediction_labels if isinstance(l, str) and l.startswith("X")
            ]
            float_pred_labels = [floatify(f[1:]) for f in relevant_prediction_labels]
            float_request_labels = [floatify(f) for f in self.class_ordering]
            if set(float_request_labels) <= set(float_pred_labels):
                if len(float_request_labels) < len(prediction_labels):
                    # Extract the additional columns and consider them as the extra model output
                    predictions, extra_model_output = extract_additional_columns(
                        predictions, relevant_prediction_labels
                    )
                label_map = dict(zip(float_request_labels, self.class_ordering))
                # return class labels in the same order as prediction labels
                ordered_labels = [label_map[l] for l in float_pred_labels]
                predictions.columns = ordered_labels
                return predictions, extra_model_output

        return predictions, extra_model_output

    def _predict(self, **kwargs) -> RawPredictResponse:
        input_binary_data = kwargs.get(StructuredDtoKeys.BINARY_DATA)
        mimetype = kwargs.get(StructuredDtoKeys.MIMETYPE)
        with capture_R_traceback_if_errors(r_handler, logger):
            predictions = r_handler.outer_predict(
                self.target_type.value,
                binary_data=ro.vectors.ByteVector(input_binary_data),
                mimetype=ro.NULL if mimetype is None else mimetype,
                model=self._model,
                positive_class_label=self.positive_class_label or ro.NULL,
                negative_class_label=self.negative_class_label or ro.NULL,
                class_labels=ro.StrVector(self.class_labels) if self.class_labels else ro.NULL,
                sparse_colnames=self._get_sparse_colnames(kwargs),
            )

        with localconverter(ro.default_converter + pandas2ri.converter):
            predictions = ro.conversion.rpy2py(predictions)

        if not isinstance(predictions, pd.DataFrame):
            error_message = (
                "Expected predictions type: {}, actual: {}. "
                "Are you trying to run binary classification without class labels provided?".format(
                    pd.DataFrame, type(predictions)
                )
            )
            logger.error(error_message)
            raise DrumCommonException(error_message)

        extra_model_output = None
        if self.target_type.is_classification():
            predictions, extra_model_output = self._replace_sanitized_class_names(predictions)
        elif self.target_type == TargetType.REGRESSION:
            predictions, extra_model_output = extract_additional_columns(predictions, [PRED_COLUMN])
        return RawPredictResponse(predictions.values, predictions.columns, extra_model_output)

    # TODO: check test coverage for all possible cases: return None/str/bytes, and casting.
    def predict_unstructured(self, data, **kwargs):
        def _r_is_character(r_val):
            return isinstance(r_val, ro.vectors.StrVector)

        def _r_is_raw(r_val):
            return isinstance(r_val, ro.vectors.ByteVector)

        def _r_is_null(r_val):
            return r_val == ro.NULL

        def _cast_r_to_py(r_val):
            if _r_is_null(r_val):
                return None
            elif _r_is_raw(r_val):
                return bytes(r_val)
            elif _r_is_character(r_val):
                # Any scalar value is returned from R as one element vector,
                # so get this value.
                return str(r_val[0])
            else:
                raise DrumCommonException(
                    "Can not convert R value {} type {}".format(r_val, type(r_val))
                )

        def _rlist_to_dict(rlist):
            if _r_is_null(rlist):
                return None
            return {str(k): _cast_r_to_py(v) for k, v in rlist.items()}

        data_binary_or_text = data

        if UnstructuredDtoKeys.QUERY in kwargs:
            kwargs[UnstructuredDtoKeys.QUERY] = ro.vectors.ListVector(
                kwargs[UnstructuredDtoKeys.QUERY]
            )
        if UnstructuredDtoKeys.HEADERS in kwargs:
            kwargs[UnstructuredDtoKeys.HEADERS] = ro.vectors.ListVector(
                kwargs[UnstructuredDtoKeys.HEADERS]
            )

        # if data_binary_or_text is str it will be auto converted into R character type;
        # otherwise if it is bytes, manually convert it into byte vector (raw)
        r_data_binary_or_text = data_binary_or_text
        if isinstance(data_binary_or_text, bytes):
            r_data_binary_or_text = ro.vectors.ByteVector(data_binary_or_text)

        kwargs_filtered = {k: v for k, v in kwargs.items() if v is not None}
        with capture_R_traceback_if_errors(r_handler, logger):
            list_data_kwargs = r_handler.predict_unstructured(
                model=self._model, data=r_data_binary_or_text, **kwargs_filtered
            )

        if isinstance(list_data_kwargs, ro.vectors.ListVector):
            ret = _cast_r_to_py(list_data_kwargs[0]), _rlist_to_dict(list_data_kwargs[1])
        else:
            raise DrumCommonException(
                "Wrong type returned in unstructured mode: {}".format(type(list_data_kwargs))
            )

        return ret

    def _transform(self, **kwargs):
        input_binary_data = kwargs.get(StructuredDtoKeys.BINARY_DATA)
        target_binary_data = kwargs.get(StructuredDtoKeys.TARGET_BINARY_DATA)
        mimetype = kwargs.get(StructuredDtoKeys.MIMETYPE)
        with capture_R_traceback_if_errors(r_handler, logger):
            transformations = r_handler.outer_transform(
                binary_data=ro.vectors.ByteVector(input_binary_data),
                target_binary_data=ro.NULL
                if target_binary_data is None
                else ro.vectors.ByteVector(target_binary_data),
                mimetype=ro.NULL if mimetype is None else mimetype,
                transformer=self._model,
                sparse_colnames=self._get_sparse_colnames(kwargs),
            )

        if not isinstance(transformations, ro.vectors.ListVector) or len(transformations) != 3:
            error_message = "Expected transform to return a three-element list containing X, y and colnames, got {}. ".format(
                type(transformations)
            )
            raise DrumCommonException(error_message)

        with localconverter(ro.default_converter + pandas2ri.converter):
            output_X = ro.conversion.rpy2py(transformations[0])
            output_y = (
                ro.conversion.rpy2py(transformations[1])
                if transformations[1] is not ro.NULL
                else None
            )
            colnames = (
                ro.conversion.rpy2py(transformations[2])
                if transformations[2] is not ro.NULL
                else None
            )

        if not isinstance(output_X, pd.DataFrame):
            error_message = "Expected transform output type: {}, actual: {}.".format(
                ro.vectors.DataFrame, type(transformations[0])
            )
            raise DrumCommonException(error_message)

        # If the column names contain this set of magic values, it implies the output data is sparse, so construct
        # a sparse coo_matrix out of it. TODO: [RAPTOR-6209] propagate column names when R output data is sparse
        if list(output_X.columns) == ["__DR__i", "__DR__j", "__DR__x"]:
            # The last row will contain the number of rows and cols, so get that and then drop it from output_X
            num_rows, num_cols, _ = output_X.iloc[-1]
            num_rows, num_cols = int(num_rows), int(num_cols)
            output_X = output_X[:-1]

            # R is 1-based indexing whereas python is 0, so adjust here
            row = output_X["__DR__i"] - 1
            col = output_X["__DR__j"] - 1
            data = output_X["__DR__x"]

            output_X = pd.DataFrame.sparse.from_spmatrix(
                coo_matrix((data, (row, col)), shape=(num_rows, num_cols))
            )
            if colnames is not None:
                assert (
                    len(colnames) == output_X.shape[1]
                ), "Supplied count of column names does not match number of target classes."
                output_X.columns = colnames

        return output_X, output_y
