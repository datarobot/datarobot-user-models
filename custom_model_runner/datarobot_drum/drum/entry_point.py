import logging

from datarobot_drum.drum.gunicorn.run_gunicorn import main_gunicorn
from datarobot_drum.drum.main import main
from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.utils.setup import setup_options

from datarobot_drum.drum.enum import ArgumentsOptions, DrumServerType


def run_drum_server():
    options = setup_options()
    if options.subparser_name == ArgumentsOptions.SERVER:
        server_type = DrumServerType.DEFAULT
        if RuntimeParameters.has("DRUM_SERVER_TYPE"):
            requested = str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower()
            if requested in DrumServerType.ALL:
                server_type = requested
            else:
                logging.warning(
                    "Server type '%s' is not supported. Supported types: %s. Falling back to %s.",
                    requested,
                    sorted(DrumServerType.ALL),
                    DrumServerType.DEFAULT,
                )
        if server_type == DrumServerType.GUNICORN:
            main_gunicorn()
        else:
            main()
    else:
        main()


if __name__ == "__main__":
    run_drum_server()
