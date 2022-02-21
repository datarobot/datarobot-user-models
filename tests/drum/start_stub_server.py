from drum.utils import SimpleCache
from flask import Flask, request
import json
import multiprocessing


init_cache_data = {"actual_version_queries": 0, "actual_pred_requests_queries": 0}



with SimpleCache(init_cache_data) as cache:
    app = Flask(__name__)

    @app.route("/api/v2/version/")
    def version():
        cache.inc_value("actual_version_queries")
        return json.dumps({"major": 2, "minor": 28, "versionString": "2.28.0"}), 200


    @app.route(
        "/api/v2/deployments/<deployment_id>/predictionRequests/fromJSON/", methods=["POST"]
    )
    def post_prediction_requests(deployment_id):
        stats = request.get_json()
        cache.inc_value("actual_pred_requests_queries", value=len(stats["data"]))
        return json.dumps({"message": "ok"}), 202


    proc = multiprocessing.Process(
        target=lambda: app.run(host="localhost", port=13909, debug=True, use_reloader=False)
    )
    proc.start()

    import time
    while True:
        time.sleep(1)