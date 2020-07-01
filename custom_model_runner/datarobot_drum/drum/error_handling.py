import os
from flask import Flask, request


class DrumErrorHandler:
    def __init__(self, ctx):
        self._ctx = ctx

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not exc_type:
            return True

        if not self._ctx.options:
            return False

        force_start_server = self._ctx.options.force_start_internal

        if not self._ctx.is_server_running and force_start_server:
            host_port_list = self._ctx.options.address.split(":", 1)
            host = host_port_list[0]
            port = int(host_port_list[1]) if len(host_port_list) == 2 else None

            run_error_server(host, port, exc_type, exc_value, exc_traceback)

            # NOTE: exception is propagated further
            return False


def run_error_server(host, port, exc_type, exc_value, exc_traceback):
    app = Flask(__name__)
    url_prefix = os.environ.get("URL_PREFIX", "")

    @app.route("{}/".format(url_prefix))
    def ping():
        """This route is used to ensure that server has started"""
        return "Server is up!\n", 200

    @app.route("{}/shutdown/".format(url_prefix), methods=["POST"])
    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        func()
        return "Server shutting down...", 200

    app.run(host, port)
