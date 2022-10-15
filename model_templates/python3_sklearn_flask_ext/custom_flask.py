"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging

from flask import request
from flask_httpauth import HTTPTokenAuth

logger = logging.getLogger(__name__)

token_auth = HTTPTokenAuth()

# Hard-code users for demo purposes but in a real setup this data would be
# fetched from a database, for example.
AUTHORIZED_TOKEN = "dummy_demo_token"


@token_auth.verify_token
def verify(token):
    if token == AUTHORIZED_TOKEN:
        return "dummy_user"


def init_app(app):
    """
    Below is a sample hook that illustrates how to add simple token based
    authentication to most of the routes served by the custom model runner.

    Parameters
    ----------
    app: Flask
    """
    # Health check endpoints shouldn't require auth
    no_auth_endpoints = {"model_api.ping", "model_api.health"}
    logger.info("Setting up authentication on all routes except ping routes")

    @app.before_request
    def check_auth():
        auth = token_auth.get_auth()

        # Flask normally handles OPTIONS requests on its own, but in
        # the case it is configured to forward those to the
        # application, we need to ignore authentication headers and
        # let the request through to avoid unwanted interactions with
        # CORS.
        if request.method != 'OPTIONS' and request.endpoint not in no_auth_endpoints:
            user = token_auth.authenticate(auth, None)
            if user in (False, None):
                return token_auth.auth_error_callback(401)

    logger.info('Please authenticate to the server:\n\t`curl -H "Authorization: Bearer %s" ...`',
                AUTHORIZED_TOKEN)
