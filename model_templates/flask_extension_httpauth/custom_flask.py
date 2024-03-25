"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import logging

from flask import request

# To show the flexibility of this hook, this example utilizes an open-source, third-party
# extension but you can also just as easily customize the Flask server without using any
# additional dependencies if desired.
from flask_httpauth import HTTPTokenAuth

logger = logging.getLogger(__name__)

token_auth = HTTPTokenAuth()

AUTHENTICATION_TOKEN = "DUMMY_TOKEN_123"


@token_auth.verify_token
def verify(token):
    # Hard-code users for demo purposes but in a real setup this data would be
    # fetched from a database or secure key vault, for example.
    if token == AUTHENTICATION_TOKEN:
        # flask_httpauth requires this function to return a username on successful authentication
        # so it can be used for authorization (but we aren't implementing that for this sample).
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

        # Flask normally handles OPTIONS requests on its own, but in the case it is configured to
        # forward those to the application, we need to ignore authentication headers and let the
        # request through to avoid unwanted interactions with CORS.
        if request.method != "OPTIONS" and request.endpoint not in no_auth_endpoints:
            user = token_auth.authenticate(auth, None)
            if user in (False, None):
                return token_auth.auth_error_callback(401)

    logger.info(
        'Please authenticate to the server:\n\t`curl -H "Authorization: Bearer %s" ...`',
        AUTHENTICATION_TOKEN,
    )
