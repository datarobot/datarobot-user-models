from datarobot_drum.drum.gunicorn.run_gunicorn import main_gunicorn
from datarobot_drum.drum.main import main
from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.utils.setup import setup_options

from datarobot_drum.drum.enum import ArgumentsOptions


def run_drum_server():
    options = setup_options()
    if (
        options.subparser_name == ArgumentsOptions.SERVER
        and RuntimeParameters.has("DRUM_SERVER_TYPE")
        and str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower() == "gunicorn"
    ):
        main_gunicorn()
    else:
        main()


if __name__ == "__main__":
    run_drum_server()
