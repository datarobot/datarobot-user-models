from datarobot_drum.drum.common import config_logging, get_drum_logger
from datarobot_drum.drum.gunicorn.run_gunicorn import main_gunicorn
from datarobot_drum.drum.main import main
from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.utils.setup import setup_options

from datarobot_drum.drum.enum import ArgumentsOptions, DrumServerType, LOGGER_NAME_PREFIX

logger = get_drum_logger(LOGGER_NAME_PREFIX)


def run_drum_server():
    config_logging()
    options = setup_options()
    if options.subparser_name == ArgumentsOptions.SERVER:
        server_type = DrumServerType.DEFAULT
        if RuntimeParameters.has("DRUM_SERVER_TYPE"):
            requested = str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower()
            if requested in DrumServerType.ALL:
                server_type = requested
            else:
                logger.warning(
                    "Server type '%s' is not supported. Supported types: %s. Falling back to %s.",
                    requested,
                    sorted(DrumServerType.ALL),
                    DrumServerType.DEFAULT,
                )
        else:
            # TODO(remove after a few releases past 1.17.19): drop this migration notice
            # once users have had time to adjust to the new default.
            logger.warning(
                "Default DRUM server changed from '%s' to '%s' as of DRUM 1.17.19. "
                "Set DRUM_SERVER_TYPE='%s' to keep previous behavior.",
                DrumServerType.WERKZEUG,
                DrumServerType.GUNICORN,
                DrumServerType.WERKZEUG,
            )

        if server_type == DrumServerType.GUNICORN and getattr(options, "docker", None):
            logger.warning(
                "Gunicorn is not supported with --docker: gunicorn binds the host address "
                "before the container starts, which conflicts with the container's own port "
                "publish. Falling back to %s.",
                DrumServerType.WERKZEUG,
            )
            server_type = DrumServerType.WERKZEUG

        if server_type == DrumServerType.GUNICORN:
            main_gunicorn()
        else:
            main()
    else:
        main()


if __name__ == "__main__":
    run_drum_server()
