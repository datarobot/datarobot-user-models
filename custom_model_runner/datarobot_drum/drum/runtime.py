from datarobot_drum.drum.server import (
    base_api_blueprint,
    get_flask_app,
    HTTP_503_SERVICE_UNAVAILABLE,
)


class DrumRuntime:
    def __init__(self):
        self.initialization_succeeded = False
        self.options = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not exc_type:
            # no exception, just return
            return True

        if not self.options:
            # exception occurred before args were parsed
            # propagate exception further
            return False

        if not getattr(self.options, "force_start_internal", False):
            # drum is not run in server mode, or force start is not set
            # propagate exception further
            return False

        if self.initialization_succeeded:
            # pipeline initialization was successful.
            # exceptions that occur during pipeline running
            # must be propagated further
            return False

        # start 'error server'
        host_port_list = self.options.address.split(":", 1)
        host = host_port_list[0]
        port = int(host_port_list[1]) if len(host_port_list) == 2 else None

        run_error_server(host, port, exc_value)

        # NOTE: exception is propagated further
        return False


def run_error_server(host, port, exc_value):
    model_api = base_api_blueprint()

    @model_api.route("/predict/", methods=["POST"])
    def predict():
        return {"message": "ERROR: {}".format(exc_value)}, HTTP_503_SERVICE_UNAVAILABLE

    app = get_flask_app(model_api)
    app.run(host, port)
