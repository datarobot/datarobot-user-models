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

import pytest

import datarobot_drum.drum.gunicorn as gunicorn_pkg

GUNICORN_CONF = Path(gunicorn_pkg.__file__).parent / "gunicorn.conf.py"


def load_conf(monkeypatch, keepalive=None):
    monkeypatch.setenv("ADDRESS", "0.0.0.0:8080")
    if keepalive is None:
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_DRUM_GUNICORN_KEEP_ALIVE", raising=False)
    else:
        monkeypatch.setenv(
            "MLOPS_RUNTIME_PARAM_DRUM_GUNICORN_KEEP_ALIVE",
            json.dumps({"type": "numeric", "payload": keepalive}),
        )
    return runpy.run_path(str(GUNICORN_CONF))


def test_keepalive_disabled_by_default(monkeypatch):
    assert load_conf(monkeypatch)["keepalive"] == 0


@pytest.mark.parametrize("value", [0, 1, 3600])
def test_keepalive_accepts_values_in_range(monkeypatch, value):
    assert load_conf(monkeypatch, value)["keepalive"] == value


@pytest.mark.parametrize("value", [-1, 3601])
def test_keepalive_falls_back_to_disabled_on_out_of_range_values(monkeypatch, value):
    assert load_conf(monkeypatch, value)["keepalive"] == 0
