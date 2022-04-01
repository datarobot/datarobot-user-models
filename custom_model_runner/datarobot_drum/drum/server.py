"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from flask import Flask, Blueprint, request
import os

from datarobot_drum.drum.enum import URL_PREFIX_ENV_VAR_NAME

HTTP_200_OK = 200
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_513_DRUM_PIPELINE_ERROR = 513


def get_flask_app(api_blueprint):
    app = Flask(__name__)
    url_prefix = os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)
    return app


def base_api_blueprint(termination_hook=None):
    model_api = Blueprint("model_api", __name__)

    @model_api.route("/", methods=["GET"])
    @model_api.route("/ping/", methods=["GET"])
    def ping():
        """This route is used to ensure that server has started"""
        return {"message": "OK"}, HTTP_200_OK

    return model_api
