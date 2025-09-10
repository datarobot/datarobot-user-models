from datarobot_drum.drum.server import create_flask_app

app = create_flask_app()

_worker_ctx = None

def set_worker_ctx(ctx):
    global _worker_ctx
    _worker_ctx = ctx

def get_worker_ctx():
    return _worker_ctx