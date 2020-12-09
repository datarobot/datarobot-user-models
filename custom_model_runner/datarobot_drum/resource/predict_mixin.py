from flask import request, Response
import werkzeug


from datarobot_drum.drum.common import (
    REGRESSION_PRED_COLUMN,
    TargetType,
    UnstructuredDtoKeys,
    PredictionServerMimetypes,
    X_TRANSFORM_KEY,
)
from datarobot_drum.drum.utils import StructuredInputReadUtils
from datarobot_drum.resource.transform_helpers import (
    make_arrow_payload,
    is_sparse,
    make_mtx_payload,
    make_csv_payload,
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

    @staticmethod
    def _validate_content_type_header(header):
        ret_mimetype, content_type_params_dict = werkzeug.http.parse_options_header(header)
        ret_charset = content_type_params_dict.get("charset")
        return ret_mimetype, ret_charset

    def _predict_or_transform(self, logger=None):
        response_status = HTTP_200_OK

        file_key = "X"
        filestorage = request.files.get(file_key)

        if self._target_type == TargetType.TRANSFORM:
            arrow_key = "arrow_version"
            arrow_version = request.files.get(arrow_key)
            if arrow_version is not None:
                arrow_version = eval(arrow_version.getvalue())
            use_arrow = arrow_version is not None

        charset = None
        if filestorage is not None:
            binary_data = filestorage.stream.read()
            mimetype = StructuredInputReadUtils.resolve_mimetype_by_filename(filestorage.filename)

            if logger is not None:
                logger.debug("Filename provided under X key: {}".format(filestorage.filename))

        # TODO: probably need to return empty response in case of empty request
        elif len(request.data):
            binary_data = request.data
            mimetype, charset = PredictMixin._validate_content_type_header(request.content_type)
        else:
            wrong_key_error_message = (
                "Samples should be provided as: "
                "  - a csv, mtx, or arrow file under `{}` form-data param key."
                "  - binary data".format(file_key)
            )
            if logger is not None:
                logger.error(wrong_key_error_message)
            response_status = HTTP_422_UNPROCESSABLE_ENTITY
            return {"message": "ERROR: " + wrong_key_error_message}, response_status

        if self._target_type == TargetType.TRANSFORM:
            out_data = self._predictor.transform(
                binary_data=binary_data, mimetype=mimetype, charset=charset
            )
        else:
            out_data = self._predictor.predict(
                binary_data=binary_data, mimetype=mimetype, charset=charset
            )

        if self._target_type == TargetType.UNSTRUCTURED:
            response = out_data
        elif self._target_type == TargetType.TRANSFORM:
            if is_sparse(out_data):
                mtx_payload = make_mtx_payload(out_data)
                response = (
                    '{{"{transform_key}":{mtx_payload}, "out.format":"{out_format}"}}'.format(
                        transform_key=X_TRANSFORM_KEY, mtx_payload=mtx_payload, out_format="sparse"
                    )
                )
            else:
                if use_arrow:
                    arrow_payload = make_arrow_payload(out_data, arrow_version)
                    response = (
                        '{{"{transform_key}":{arrow_payload}, "out.format":"{out_format}"}}'.format(
                            transform_key=X_TRANSFORM_KEY,
                            arrow_payload=arrow_payload,
                            out_format="arrow",
                        )
                    )
                else:
                    csv_payload = make_csv_payload(out_data)
                    response = (
                        '{{"{transform_key}":{csv_payload}, "out.format":"{out_format}"}}'.format(
                            transform_key=X_TRANSFORM_KEY, csv_payload=csv_payload, out_format="csv"
                        )
                    )
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

    def do_predict(self, logger=None):
        if self._target_type == TargetType.TRANSFORM:
            wrong_target_type_error_message = (
                "This project has target type {}, "
                "use the /transform/ endpoint.".format(self._target_type)
            )
            if logger is not None:
                logger.error(wrong_target_type_error_message)
            response_status = HTTP_422_UNPROCESSABLE_ENTITY
            return {"message": "ERROR: " + wrong_target_type_error_message}, response_status

        return self._predict_or_transform(logger=logger)

    def do_predict_unstructured(self, logger=None):
        response_status = HTTP_200_OK
        kwargs_params = {}

        data = request.data
        mimetype, charset = PredictMixin._validate_content_type_header(request.content_type)

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

    def do_transform(self, logger=None):
        if self._target_type != TargetType.TRANSFORM:
            endpoint = (
                "predictUnstructured" if self._target_type == TargetType.UNSTRUCTURED else "predict"
            )
            wrong_target_type_error_message = (
                "This project has target type {}, "
                "use the /{}/ endpoint.".format(self._target_type, endpoint)
            )
            if logger is not None:
                logger.error(wrong_target_type_error_message)
            response_status = HTTP_422_UNPROCESSABLE_ENTITY
            return {"message": "ERROR: " + wrong_target_type_error_message}, response_status

        return self._predict_or_transform(logger=logger)
