import tempfile
from flask import request


from datarobot_drum.drum.common import REGRESSION_PRED_COLUMN, TargetType

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

        with tempfile.NamedTemporaryFile() as f:
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
            elif num_columns == 2:
                # df.to_json() is much faster.
                # But as it returns string, we have to assemble final json using strings.
                df_json_str = out_data.to_json(orient="records")
                response = '{{"predictions":{df_json}}}'.format(df_json=df_json_str)
            else:
                ret_str = (
                    "Predictions dataframe has {} columns; "
                    "Expected: 1 - for regression, 2 - for binary classification.".format(
                        num_columns
                    )
                )
                response = {"message": "ERROR: " + ret_str}
                response_status = HTTP_422_UNPROCESSABLE_ENTITY
        return response, response_status
