import os
import tempfile
from flask import request, Response
import werkzeug


from datarobot_drum.drum.common import (
    REGRESSION_PRED_COLUMN,
    TargetType,
    UnstructuredDtoKeys,
    PredictionServerMimetypes,
)
from datarobot_drum.resource.unstructured_helpers import (
    _resolve_incoming_unstructured_data,
    _resolve_outgoing_unstructured_data,
)


from datarobot_drum.drum.server import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_ENTITY,
)


class PredictMixin:
    """
    This class implements predict flow shared by PredictionServer and UwsgiServing classes.
    This flow assumes endpoints implemented using Flask.

    """

    def do_predict(self, logger=None):
        response_status = HTTP_200_OK
        response = None

        file_key = "X"
        filestorage = request.files.get(file_key)

        if not filestorage:
            wrong_key_error_message = (
                "Samples should be provided as a csv file under `{}` key.".format(file_key)
            )
            if logger is not None:
                logger.error(wrong_key_error_message)
            response_status = HTTP_422_UNPROCESSABLE_ENTITY
            return {"message": "ERROR: " + wrong_key_error_message}, response_status
        else:
            if logger is not None:
                logger.debug("Filename provided under X key: {}".format(filestorage.filename))

        _, file_ext = os.path.splitext(filestorage.filename)
        with tempfile.NamedTemporaryFile(suffix=file_ext) as f:
            filestorage.save(f)
            f.flush()
            out_data = self._predictor.predict(f.name)

        if self._target_type == TargetType.UNSTRUCTURED:
            response = out_data
        else:
            num_columns = len(out_data.columns)
            # float32 is not JSON serializable, so cast to float, which is float64
            out_data = out_data.astype("float")
            if num_columns == 1:
                # df.to_json() is much faster.
                # But as it returns string, we have to assemble final json using strings.
                df_json = out_data[REGRESSION_PRED_COLUMN].to_json(orient="records")
                response = '{{"predictions":{df_json}}}'.format(df_json=df_json)
            else:
                # df.to_json() is much faster.
                # But as it returns string, we have to assemble final json using strings.
                df_json_str = out_data.to_json(orient="records")
                response = '{{"predictions":{df_json}}}'.format(df_json=df_json_str)

        response = Response(response, mimetype=PredictionServerMimetypes.APPLICATION_JSON)

        return response, response_status

    def do_predict_unstructured(self, logger=None):
        def _validate_content_type_header(header):
            ret_mimetype, content_type_params_dict = werkzeug.http.parse_options_header(header)
            ret_charset = content_type_params_dict.get("charset")
            return ret_mimetype, ret_charset

        response_status = HTTP_200_OK
        kwargs_params = {}

        data = request.data
        mimetype, charset = _validate_content_type_header(request.content_type)

        data_binary_or_text, mimetype, charset = _resolve_incoming_unstructured_data(
            data,
            mimetype,
            charset,
        )
        kwargs_params[UnstructuredDtoKeys.MIMETYPE] = mimetype
        if charset is not None:
            kwargs_params[UnstructuredDtoKeys.CHARSET] = charset
        kwargs_params[UnstructuredDtoKeys.QUERY] = request.args

        ret_data, ret_kwargs = self._predictor.predict_unstructured(
            data_binary_or_text, **kwargs_params
        )

        response_data, response_mimetype, response_charset = _resolve_outgoing_unstructured_data(
            ret_data, ret_kwargs
        )

        response = Response(response_data)

        if response_mimetype is not None:
            content_type = response_mimetype
            if response_charset is not None:
                content_type += "; charset={}".format(response_charset)
            response.headers["Content-Type"] = content_type

        return response, response_status
