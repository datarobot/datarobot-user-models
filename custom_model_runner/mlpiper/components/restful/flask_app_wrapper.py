"""
For internal use only. These are wrappers for the Flask building blocks
"""
import inspect
from flask import Flask
from flask import json
from flask import request
from flask import Response
from flask_cors import CORS

from mlpiper.common.mlpiper_exception import MLPiperException


class EndpointAction(object):
    """
    A wrapper over a Flask handler. It is called on each incoming request and therefore
    should be treated with caution and should be optimized on each part of it.
    """

    def __init__(self, handler, raw):
        if not inspect.ismethod(handler):
            raise MLPiperException(
                "Invalid REST endpoint handler! Should be a component's method with the "
                "following prototype: <handler>(self, url_params, form_params), given: {}".format(
                    handler
                )
            )

        self._handler = handler
        self._raw = raw

    def __call__(self):
        try:
            if self._raw:
                # This option is intended to serve py4j handlers
                status, response = self._handler(
                    request.query_string.decode(), request.get_data(as_text=True)
                )
            else:
                status, response = self._handler(
                    request.args.to_dict(),
                    request.get_json() if request.is_json else request.form.to_dict(),
                )
        except ValueError:
            raise MLPiperException(
                "Invalid returned type from endpoint handler: '{}', ".format(
                    self._handler
                )
                + "Expecting for tuple of two elements: (status, response)"
            )

        if isinstance(response, Response):
            return response

        if not isinstance(response, str):
            response = json.dumps(response)

        return Response(response=response, status=status, mimetype="application/json")


class FlaskAppWrapper(object):
    """
    A wrapper over the Flask application.
    """

    app = None

    def __init__(self, pipeline_name):
        FlaskAppWrapper.app = Flask(pipeline_name)
        CORS(FlaskAppWrapper.app)

    def run(self, host, port):
        FlaskAppWrapper.app.run(host, port)

    def add_endpoint(self, url_rule, endpoint, handler, options, raw):
        endpoint = handler.__name__ if not endpoint else endpoint
        FlaskAppWrapper.app.add_url_rule(
            url_rule, endpoint, EndpointAction(handler, raw), None, **options
        )
