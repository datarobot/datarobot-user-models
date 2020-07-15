import logging

from datarobot_drum.drum.server import (
    base_api_blueprint,
    get_flask_app,
    HTTP_513_DRUM_PIPELINE_ERROR,
)

from datarobot_drum.drum.common import RunMode, verbose_stdout


class DrumRuntime:
    def __init__(self):
        self.initialization_succeeded = False
        self.options = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not exc_type:
            return True  # no exception, just return

        if not self.options:
            # exception occurred before args were parsed
            return False  # propagate exception further

        run_mode = RunMode(self.options.subparser_name)
        if run_mode != RunMode.SERVER:
            # drum is not run in server mode
            return False  # propagate exception further

        # TODO: add docker support
        if getattr(self.options, "docker", None):
            # running 'error server' in docker mode is not supported
            return False  # propagate exception further

        if not self.options.force_start_internal:
            # force start is not set
            return False  # propagate exception further

        if self.initialization_succeeded:
            # pipeline initialization was successful.
            # exceptions that occur during pipeline running
            # must be propagated further
            return False  # propagate exception further

        # start 'error server'
        host_port_list = self.options.address.split(":", 1)
        host = host_port_list[0]
        port = int(host_port_list[1]) if len(host_port_list) == 2 else None

        with verbose_stdout(self.options.verbose):
            run_error_server(host, port, exc_value)

        return False  # propagate exception further


def run_error_server(host, port, exc_value):
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    model_api = base_api_blueprint()

    @model_api.route("/predict/", methods=["POST"])
    def predict():
        return {"message": "ERROR: {}".format(exc_value)}, HTTP_513_DRUM_PIPELINE_ERROR

    app = get_flask_app(model_api)
    app.run(host, port)
