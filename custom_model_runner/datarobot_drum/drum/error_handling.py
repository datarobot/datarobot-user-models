import os
from flask import Flask, request

HTTP_200_OK = 200
HTTP_503_SERVICE_UNAVAILABLE = 503


class DrumErrorHandler:
    def __init__(self, ctx):
        self._ctx = ctx

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not exc_type:
            # no exception, just return
            return True

        if not self._ctx.options:
            # exception occurred before args were parsed
            # propagate exception further
            return False

        if self._ctx.initialization_succeeded:
            # pipeline initialization was successful.
            # exceptions that occur during pipeline running
            # must be propagated further
            return False

        if self._ctx.options.force_start_internal:
            host_port_list = self._ctx.options.address.split(":", 1)
            host = host_port_list[0]
            port = int(host_port_list[1]) if len(host_port_list) == 2 else None

            run_error_server(host, port, exc_value)

        # NOTE: exception is propagated further
        return False


def run_error_server(host, port, exc_value):
    app = Flask(__name__)
    url_prefix = os.environ.get("URL_PREFIX", "")

    @app.route("{}/".format(url_prefix))
    def ping():
        """This route is used to ensure that server has started"""
        return "Server is up!\n", HTTP_200_OK

    @app.route("{}/shutdown/".format(url_prefix), methods=["POST"])
    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        func()
        return "Server shutting down...", HTTP_200_OK

    @app.route("{}/predict/".format(url_prefix), methods=["POST"])
    def predict():
        return {"message": "ERROR: {}".format(exc_value)}, HTTP_503_SERVICE_UNAVAILABLE

    app.run(host, port)
