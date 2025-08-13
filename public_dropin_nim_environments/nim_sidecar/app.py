# Import DRUM's WSGI application
import sys

from datarobot_drum.drum.main import main
from datarobot_drum.drum.server import create_flask_app

# You may need to configure the app with the same settings
# that would normally be set by command line arguments
import os
#os.environ["DRUM_SIDECAR"] = "true"
#os.environ["DRUM_GPU_PREDICTOR"] = "nim"
'''os.environ["TARGET_TYPE"] = "textgeneration"
os.environ["ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS"] = "true"
os.environ["ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP"] = "0"
os.environ["EXTERNAL_WEB_SERVER_URL"] = "http://127.0.0.1"
os.environ["DATAROBOT_ENDPOINT"] = "http://127.0.0.1/api/v2"
os.environ["MLOPS_DEPLOYMENT_ID"] = "a2fde18c5458caba0267c"
os.environ["MLOPS_MODEL_ID"] = "689a2fae18c5458caba02677"
os.environ["TARGET_NAME"] = "resultText"
os.environ["API_TOKEN"] = "resultText"'''


sys.argv = [
    "drum","server",  # Program name
    "--sidecar","--gpu-predictor=nim", "--logging-level=info"
]
import traceback

app = create_flask_app()
app2 = create_flask_app()
main(app, app)

