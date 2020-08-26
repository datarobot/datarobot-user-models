# This script creates parallel processes to send predict requests

import argparse
import json
import requests
import time
from multiprocessing import Process, Queue, Pipe

_predict_endpoint = "/predict/"
_health_endpoint = "/health/"
_statsinternal_endpoint = "/statsinternal/"
_stats_endpoint = "/stats/"


class SenderStats:
    def __init__(self, name):
        self.name = name
        self.requests_to_send = None
        self.responses_received = 0
        self.requests_failed = 0
        self.values_range = []
        self.average_request_time = 0


class WorkerStats:
    def __init__(self):
        self.workers = {}
        self.max_mem = 0
        self.total_predictions = 0
        self.max_avg_prediction_time_on_worker = 0


def request_func(queue, params, name):
    stats = SenderStats(name)
    stats.requests_to_send = params["requests"]

    with_dr = False
    if "api-key" in params and "datarobot-key" in params:
        headers = {
            "Content-Type": "text/plain; charset=UTF-8",
        }
        headers.update({"Authorization": "Bearer {}".format(params["api-key"])})
        headers.update({"DataRobot-Key": "{}".format(params["datarobot-key"])})
        with_dr = True

    request_time_sum = 0
    for _ in range(params["requests"]):
        with open(params["input"]) as f:
            start_time = time.time()
            if not with_dr:
                try:
                    response = requests.post(params["url"] + _predict_endpoint, files={"X": f})
                except requests.exceptions.ReadTimeout:
                    pass
            else:
                data = f.read()
                response = requests.post(params["url"], data=data, headers=headers)
            end_time = time.time()
            request_time = end_time - start_time
            request_time_sum += request_time

            if response.ok:
                stats.responses_received += 1
                if not with_dr:
                    response_value = response.json()["predictions"][0]
                    try:
                        response_value = int(response_value)
                        if response_value not in stats.values_range:
                            stats.values_range.append(response_value)
                    except TypeError:
                        pass
                else:
                    response_value = response.json()["data"][0]["prediction"]
                    try:
                        response_value = int(response_value)
                        if response_value not in stats.values_range:
                            stats.values_range.append(response_value)
                    except TypeError:
                        pass
            else:
                stats.requests_failed += 1
    stats.average_request_time = request_time_sum / stats.requests_to_send
    queue.put(stats)


def stats_func(pipe, server_url):
    worker_stats = WorkerStats()
    while True:
        for _ in range(10):
            time.sleep(0.1)
            response = requests.post(server_url + _statsinternal_endpoint)
            if response.ok:
                dd = json.loads(response.text)
                worker_id = dd["sys_info"]["wuuid"]
                predict_calls_count = dd["predict_calls_per_worker"]

                worker_stats.workers[str(worker_id)] = predict_calls_count
                worker_stats.total_predictions = dd["predict_calls_total"]

        response = requests.get(server_url + _stats_endpoint)
        if response.ok:
            dd = json.loads(response.text)
            try:
                worker_stats.max_mem = max(dd["mem_info"]["drum_rss"], worker_stats.max_mem)
            except TypeError:
                pass

            if "time_info" in dd:
                avg_time = dd["time_info"]["run_predictor_total"]["avg"]
                if avg_time:
                    worker_stats.max_avg_prediction_time_on_worker = max(
                        avg_time,
                        worker_stats.max_avg_prediction_time_on_worker,
                    )

        o = ""
        if pipe.poll():
            o = pipe.recv()
        if o == "shutdown":
            pipe.send(worker_stats)
            break

        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel requests sender")
    parser.add_argument("--input", required=True, help="Input dataset")
    parser.add_argument("--api-key", required=False, default=None, help="API key")
    parser.add_argument("--datarobot-key", required=False, default=None, help="Datarobot key")
    parser.add_argument("--requests", default=1, type=int, help="Number of requests")
    parser.add_argument("--threads", default=1, type=int, help="Number of clients")
    parser.add_argument(
        "--address", default=None, required=True, help="Prediction server address with http:\\"
    )

    args = parser.parse_args()

    remainder = args.requests % args.threads
    if remainder:
        requests_per_thread = int(args.requests / args.threads) + 1
    else:
        requests_per_thread = int(args.requests / args.threads)

    params = {"requests": requests_per_thread, "input": args.input, "url": args.address}

    fetch_and_show_uwsgi_stats = True
    if args.api_key and args.datarobot_key:
        params.update({"api-key": args.api_key})
        params.update({"datarobot-key": args.datarobot_key})
        fetch_and_show_uwsgi_stats = False

    processes = []
    q = Queue()

    if fetch_and_show_uwsgi_stats:
        main_conn, worker_stats_conn = Pipe()
        stats_thread = Process(
            target=stats_func,
            args=(
                worker_stats_conn,
                args.address,
            ),
        )
        stats_thread.start()

    for i in range(args.threads):
        p = Process(
            target=request_func,
            args=(
                q,
                params,
                i,
            ),
        )
        processes.append(p)
        p.start()

    start_time = time.time()
    for p in processes:
        p.join()

    if fetch_and_show_uwsgi_stats:
        main_conn.send("shutdown")
        stats_thread.join()
        workers_stats = main_conn.recv()

    total_requests = 0
    total_responses = 0
    total_failed = 0
    for i in range(args.threads):
        stats = q.get()
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Stats from sender: {}".format(stats.name))
        print("    Requests to send: {}".format(stats.requests_to_send))
        print("    Requests succeeded: {}".format(stats.responses_received))
        print("    Requests failed: {}".format(stats.requests_failed))
        print("    Avg. request time: {}".format(stats.average_request_time))
        print("    Response values: {}".format(stats.values_range))
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n")
        total_requests += stats.requests_to_send
        total_responses += stats.responses_received
        total_failed += stats.requests_failed

    print("Summary:")
    print("    Total to send: {}".format(total_requests))
    print("    Total succeeded: {}".format(total_responses))
    print("    Total failed: {}".format(total_failed))
    print("    Total run time: {:0.2f} s".format(time.time() - start_time))
    if fetch_and_show_uwsgi_stats:
        print("    Max mem: {:0.2f} MB".format(workers_stats.max_mem))
        print(
            "    Max avg pred time: {:0.4f} s".format(
                workers_stats.max_avg_prediction_time_on_worker
            )
        )

    if fetch_and_show_uwsgi_stats:
        print("\n\nWorkers stats:")
        total_predicted_on_workers = 0
        for key, value in workers_stats.workers.items():
            if key != "total":
                print("   worker: {}; predicsts: {}".format(key, value))
                total_predicted_on_workers += value
        print("\n")
        print("Total predicted on workers: {}".format(total_predicted_on_workers))
        print(
            "Total predicted on workers (metrics by uwsgi): {}".format(
                workers_stats.total_predictions
            )
        )
