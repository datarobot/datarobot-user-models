"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from flask import request, jsonify


def init_app(app):
    @app.before_request
    def check_header():
        # Allow ping route with no Auth otherwise test setup would fail
        if request.endpoint != "model_api.ping":
            try:
                token = request.headers["X-Auth"]
            except KeyError:
                return jsonify({"message": "Missing X-Auth header"}), 401
            else:
                if token != "t0k3n":
                    return jsonify({"message": "Auth token is invalid"}), 401
