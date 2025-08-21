# Import DRUM's WSGI application

from gevent import monkey
monkey.patch_all()

import sys
from datarobot_drum.drum.main import main
from datarobot_drum.drum.server import create_flask_app

sys.argv = [
    "drum","server",  # Program name
    "--sidecar","--gpu-predictor=nim", "--logging-level=info"
]

app = create_flask_app()
main(app)

