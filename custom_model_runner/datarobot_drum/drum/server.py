from flask import Flask, Blueprint, request
import os

from datarobot_drum.drum.common import URL_PREFIX_ENV_VAR_NAME

HTTP_200_OK = 200
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_513_DRUM_PIPELINE_ERROR = 513


def get_flask_app(api_blueprint):
    app = Flask(__name__)
    url_prefix = os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)
    return app


def base_api_blueprint():
    model_api = Blueprint("model_api", __name__)

    @model_api.route("/shutdown/", methods=["POST"])
    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        func()
        return "Server shutting down...", HTTP_200_OK

    @model_api.route("/", methods=["GET"])
    def ping():
        """This route is used to ensure that server has started"""
        return {"message": "OK"}, 200

    return model_api
