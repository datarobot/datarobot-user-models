"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import datetime
import flask
import os
import uuid
from flask import Flask, Blueprint
from flask import request

from datarobot_drum.drum.common import ctx_request_id
from datarobot_drum.drum.common import get_drum_logger
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.enum import URL_PREFIX_ENV_VAR_NAME

HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400
HTTP_404_NOT_FOUND = 404
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_513_DRUM_PIPELINE_ERROR = 513

HEADER_REQUEST_ID = "X_Request_ID"


logger = get_drum_logger(LOGGER_NAME_PREFIX)


def get_flask_app(api_blueprint):
    app = create_flask_app()
    url_prefix = os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)
    return app


def base_api_blueprint(termination_hook=None, predictor=None):
    model_api = Blueprint("model_api", __name__)

    @model_api.route("/", methods=["GET"])
    @model_api.route("/ping/", methods=["GET"], strict_slashes=False)
    def ping():
        """This route is used to ensure that server has started"""
        if hasattr(predictor, "liveness_probe"):
            return predictor.liveness_probe()

        return {"message": "OK"}, HTTP_200_OK

    return model_api


def empty_api_blueprint(termination_hook=None):
    return Blueprint("model_api", __name__)


def before_request():
    flask.g.request_start_time = datetime.datetime.now()

    # always take the request_id from the request
    request_id = flask.request.environ.get("HTTP_X_REQUEST_ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    flask.g.request_id_token = ctx_request_id.set(request_id)


def after_request(response):
    request_string = "{method} {path}".format(method=request.method, path=request.full_path)

    request_id = ctx_request_id.get(None)
    response.headers[HEADER_REQUEST_ID] = request_id

    if flask.has_request_context():
        if hasattr(flask.g, "request_id_token"):
            ctx_request_id.reset(flask.g.request_id_token)
            delattr(flask.g, "request_id_token")

    if response.status_code >= 400:
        request_start_time = getattr(flask.g, "request_start_time", None)
        request_time = "unknown"
        if request_start_time is not None:
            total_time = datetime.datetime.now() - request_start_time
            request_time = total_time.total_seconds()

        logger.info(
            "API [%s] request time: %s sec, request_id: %s",
            request_string,
            request_time,
            request_id,
        )
    return response


def create_flask_app():
    flask_app = Flask(__name__)
    flask_app.before_request(before_request)
    flask_app.after_request(after_request)
    return flask_app
