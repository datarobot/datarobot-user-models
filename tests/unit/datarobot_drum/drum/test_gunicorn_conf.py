#
#  Copyright 2026 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

import datarobot_drum.drum.gunicorn as gunicorn_pkg

GUNICORN_CONF = Path(gunicorn_pkg.__file__).parent / "gunicorn.conf.py"


RUNTIME_PARAMS = [
    "DRUM_GUNICORN_KEEP_ALIVE",
    "DRUM_GUNICORN_FIRST_BYTE_TIMEOUT",
    "DRUM_GUNICORN_WORKER_CLASS",
]


def load_conf(monkeypatch, **runtime_params):
    monkeypatch.setenv("ADDRESS", "0.0.0.0:8080")
    for name in RUNTIME_PARAMS:
        monkeypatch.delenv(f"MLOPS_RUNTIME_PARAM_{name}", raising=False)
    for name, (param_type, payload) in runtime_params.items():
        monkeypatch.setenv(
            f"MLOPS_RUNTIME_PARAM_{name}",
            json.dumps({"type": param_type, "payload": payload}),
        )
    return runpy.run_path(str(GUNICORN_CONF))


@pytest.mark.parametrize(
    "param, key, default",
    [
        ("DRUM_GUNICORN_KEEP_ALIVE", "keepalive", 0),
        ("DRUM_GUNICORN_FIRST_BYTE_TIMEOUT", "first_byte_timeout", 2),
    ],
)
@pytest.mark.parametrize(
    "value, expected", [(None, None), (0, 0), (1, 1), (3600, 3600), (-1, None), (3601, None)]
)
def test_numeric_param_falls_back_to_default_when_unset_or_out_of_range(
    monkeypatch, param, key, default, value, expected
):
    conf = load_conf(monkeypatch, **({} if value is None else {param: ("numeric", value)}))
    assert conf[key] == (default if expected is None else expected)


def test_post_fork_passes_first_byte_timeout_to_worker(monkeypatch):
    conf = load_conf(monkeypatch, DRUM_GUNICORN_FIRST_BYTE_TIMEOUT=("numeric", 7))
    worker = SimpleNamespace()
    conf["post_fork"](server=None, worker=worker)
    assert worker.first_byte_timeout == 7


def test_sync_worker_class_by_default(monkeypatch):
    assert load_conf(monkeypatch)["worker_class"] == "sync"


def test_gevent_worker_class_maps_to_drum_gevent_worker(monkeypatch):
    conf = load_conf(monkeypatch, DRUM_GUNICORN_WORKER_CLASS=("string", "gevent"))
    assert conf["worker_class"] == "datarobot_drum.drum.gunicorn.workers.DrumGeventWorker"
