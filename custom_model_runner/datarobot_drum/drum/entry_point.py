from datarobot_drum.drum.gunicorn.run_gunicorn import main_gunicorn
from datarobot_drum.drum.main import main
import sys
from datarobot_drum import RuntimeParameters


def run_drum_server():
    cmd_args = sys.argv
    is_server = False
    if len(cmd_args) > 1 and cmd_args[1] == "server":
        is_server = True

    if (
        RuntimeParameters.has("DRUM_SERVER_TYPE")
        and str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower() == "gunicorn"
        and is_server
    ):
        main_gunicorn()
    else:
        main()


if __name__ == "__main__":
    run_drum_server()
