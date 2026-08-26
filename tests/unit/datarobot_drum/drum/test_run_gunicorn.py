#
#  Copyright 2026 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from datarobot_drum.drum.gunicorn import run_gunicorn
from datarobot_drum.drum.gunicorn.run_gunicorn import main_gunicorn


class TestMainGunicorn:
    @pytest.fixture(autouse=True)
    def _stub_config_path(self, tmp_path, monkeypatch):
        # main_gunicorn() resolves the config path from __file__; redirect both the
        # parent lookup and the file-existence check at our temp directory.
        config = tmp_path / "gunicorn.conf.py"
        config.write_text("")
        # NB: `parent` is a reserved MagicMock constructor kwarg (it sets the
        # mock's parent, not a `.parent` attribute), so build the chain by
        # assigning `.resolve().parent` explicitly instead.
        path_instance = MagicMock()
        path_instance.resolve.return_value.parent = tmp_path
        monkeypatch.setattr(run_gunicorn, "Path", MagicMock(return_value=path_instance))
        return config

    def test_execs_gunicorn_module(self, monkeypatch, tmp_path):
        execve = MagicMock()
        monkeypatch.setattr(run_gunicorn.os, "execve", execve)

        main_gunicorn()

        execve.assert_called_once()
        prog, argv, _env = execve.call_args.args
        assert prog == sys.executable
        assert argv[:4] == [sys.executable, "-m", "gunicorn", "-c"]
        assert argv[-1] == "app:app"
        assert argv[4] == str(tmp_path / "gunicorn.conf.py")

    def test_exports_drum_args_before_exec(self, monkeypatch):
        monkeypatch.setattr(run_gunicorn.os, "execve", MagicMock())
        monkeypatch.setattr(run_gunicorn.sys, "argv", ["drum", "server", "--code-dir", "/m"])

        main_gunicorn()

        assert os.environ["DRUM_GUNICORN_DRUM_ARGS"] == "drum server --code-dir /m"

    def test_prepends_package_dir_to_pythonpath(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_gunicorn.os, "execve", MagicMock())
        monkeypatch.setenv("PYTHONPATH", "/existing")

        main_gunicorn()

        assert os.environ["PYTHONPATH"] == f"{tmp_path}{os.pathsep}/existing"

    def test_raises_when_config_missing(self, monkeypatch, tmp_path):
        missing = tmp_path / "nope"
        path_instance = MagicMock()
        path_instance.resolve.return_value.parent = missing
        monkeypatch.setattr(run_gunicorn, "Path", MagicMock(return_value=path_instance))

        with pytest.raises(FileNotFoundError):
            main_gunicorn()

    def test_propagates_missing_gunicorn(self, monkeypatch):
        monkeypatch.setattr(run_gunicorn.os, "execve", MagicMock(side_effect=FileNotFoundError))

        with pytest.raises(FileNotFoundError):
            main_gunicorn()
