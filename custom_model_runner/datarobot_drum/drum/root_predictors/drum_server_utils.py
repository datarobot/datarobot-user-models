"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os
from typing import Optional

import requests
import signal
import time
from threading import Thread

from datarobot_drum.drum.server import HTTP_200_OK
from datarobot_drum.drum.server import HTTP_513_DRUM_PIPELINE_ERROR
from datarobot_drum.drum.utils.drum_utils import DrumUtils
from datarobot_drum.drum.enum import ArgumentsOptions, ArgumentOptionsEnvVars
from datarobot_drum.drum.root_predictors.utils import _exec_shell_cmd, _cmd_add_class_labels

logger = logging.getLogger(__name__)


def wait_for_server(url, timeout):
    # waiting for ping to succeed
    while True:
        try:
            response = requests.get(url)
            if response.status_code in [HTTP_200_OK, HTTP_513_DRUM_PIPELINE_ERROR]:
                break
            logger.debug("server is not ready: %s\n%s", response, response.text)
        except Exception:
            pass

        time.sleep(1)
        timeout -= 1
        if timeout <= 0:
            raise TimeoutError("Server failed to start: url: {}".format(url))


def _run_server_thread(cmd, process_obj_holder, verbose=True, stream_output=False):
    _exec_shell_cmd(
        cmd,
        "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
        assert_if_fail=False,
        process_obj_holder=process_obj_holder,
        verbose=verbose,
        stream_output=stream_output,
    )


class DrumServerProcess:
    def __init__(self):
        self.process = None
        self.out_stream = None
        self.err_stream = None

    @property
    def returncode(self):
        return self.process.returncode


class DrumServerRun:
    """
    Utility to help run a local drum prediction server for tests and prediction validation
    """

    def __init__(
        self,
        target_type: str,
        labels,
        custom_model_dir: str,
        docker=None,
        with_error_server=False,
        show_stacktrace=True,
        production=False,
        memory=None,
        fail_on_shutdown_error=True,
        pass_args_as_env_vars=False,
        verbose: bool = True,
        logging_level: Optional[str] = None,
        append_cmd: Optional[str] = None,
        user_secrets_mount_path: Optional[str] = None,
        thread_class=Thread,
        gpu_predictor=None,
        sidecar=False,
        target_name=None,
        wait_for_server_timeout=30,
        max_workers=None,
        cmd_override=None,
        port: Optional[int] = None,
        stream_output: Optional[bool] = False,
    ):
        self.port = port or DrumUtils.find_free_port()
        self.server_address = "localhost:{}".format(self.port)
        url_host = os.environ.get("TEST_URL_HOST", "localhost")

        if docker:
            self.url_server_address = "http://{}:{}".format(url_host, self.port)
        else:
            self.url_server_address = "http://localhost:{}".format(self.port)

        self._process_object_holder = DrumServerProcess()
        self._server_thread = None
        self._fail_on_shutdown_error = fail_on_shutdown_error
        self._verbose = verbose
        self._log_level = logging_level or logging.getLevelName(logging.root.level).lower()

        self._pass_args_as_env_vars = pass_args_as_env_vars
        self._custom_model_dir = custom_model_dir
        self._target_type = target_type
        self._target_name = target_name
        self._labels = labels
        self._docker = docker
        self._memory = memory
        self._with_error_server = with_error_server
        self._production = production
        self._show_stacktrace = show_stacktrace
        self._append_cmd = append_cmd
        self._user_secrets_mount_path = user_secrets_mount_path
        self._thread_class = thread_class
        self._gpu_predictor = gpu_predictor
        self._sidecar = sidecar
        self._wait_for_server_timeout = wait_for_server_timeout
        self._max_workers = max_workers
        self._cmd_override = cmd_override
        self._stream_output = stream_output

    def __enter__(self):
        self._server_thread = self._thread_class(
            name="DRUM Server",
            target=_run_server_thread,
            args=(
                self.get_command(),
                self._process_object_holder,
                self._verbose,
                self._stream_output,
            ),
        )
        self._server_thread.start()
        time.sleep(0.5)
        try:
            wait_for_server(self.url_server_address, timeout=self._wait_for_server_timeout)
        except TimeoutError:
            try:
                self._shutdown_server()
            except TimeoutError as e:
                logger.error("server shutdown failure: %s", e)
            raise

        return self

    def _shutdown_server(self):
        pid = self._process_object_holder.process.pid
        pgid = None
        try:
            pgid = os.getpgid(pid)
            logger.info("Sending signal to ProcessGroup: %s", pgid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            logger.warning("server at pid=%s is already gone", pid)

        self._server_thread.join(timeout=10)
        if self._server_thread.is_alive():
            if pgid is not None:
                logger.warning("Forcefully killing process group: %s", pgid)
                os.killpg(pgid, signal.SIGKILL)
                self._server_thread.join(timeout=5)
            if self._server_thread.is_alive():
                raise TimeoutError("Server failed to shutdown gracefully in allotted time")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # shutdown server
        if self._fail_on_shutdown_error:
            self._shutdown_server()
        else:
            try:
                self._shutdown_server()
            except Exception:
                logger.warning("shutdown failure", exc_info=True)

    @property
    def process(self):
        return self._process_object_holder or None

    @property
    def server_thread(self):
        return self._server_thread

    def get_command(self):
        cmd = f"{ArgumentsOptions.MAIN_COMMAND} server --logging-level={self._log_level}"
        cmd = self._cmd_override or cmd

        if self._pass_args_as_env_vars:
            os.environ[ArgumentOptionsEnvVars.CODE_DIR] = str(self._custom_model_dir)
            os.environ[ArgumentOptionsEnvVars.TARGET_TYPE] = self._target_type
            os.environ[ArgumentOptionsEnvVars.ADDRESS] = self.server_address
        else:
            cmd += " --code-dir {} --target-type {} --address {}".format(
                self._custom_model_dir, self._target_type, self.server_address
            )

        if self._labels:
            cmd = _cmd_add_class_labels(
                cmd,
                self._labels,
                target_type=self._target_type,
                pass_args_as_env_vars=self._pass_args_as_env_vars,
            )
        if self._docker:
            cmd += " --docker {}".format(self._docker)
            if self._memory:
                cmd += " --memory {}".format(self._memory)
        if self._with_error_server:
            if self._pass_args_as_env_vars:
                os.environ[ArgumentOptionsEnvVars.WITH_ERROR_SERVER] = "1"
            else:
                cmd += " --with-error-server"
        if self._show_stacktrace:
            if self._pass_args_as_env_vars:
                os.environ[ArgumentOptionsEnvVars.SHOW_STACKTRACE] = "1"
            else:
                cmd += " --show-stacktrace"
        if self._production:
            if self._pass_args_as_env_vars:
                os.environ[ArgumentOptionsEnvVars.PRODUCTION] = "1"
            else:
                cmd += " --production"
        if self._max_workers:
            if self._pass_args_as_env_vars:
                os.environ[ArgumentOptionsEnvVars.MAX_WORKERS] = str(self._max_workers)
            else:
                cmd += " --max-workers {}".format(self._max_workers)
        elif self._production:
            # The production mode is deprecated and will be removed in the future.
            self._max_workers = min(
                4, os.cpu_count()
            )  # NOTE: does not count for cgroup limits (.e.g k8s)
            cmd += " --max-workers {}".format(self._max_workers)
        if self._verbose:
            cmd += " --verbose"
        if self._gpu_predictor:
            cmd += "  --gpu-predictor {}".format(self._gpu_predictor)
        if self._sidecar:
            cmd += "  --sidecar"
        if self._target_name:
            os.environ["TARGET_NAME"] = self._target_name

        if self._user_secrets_mount_path:
            cmd += f" {ArgumentsOptions.USER_SECRETS_MOUNT_PATH} {self._user_secrets_mount_path}"

        if self._append_cmd is not None:
            cmd += " " + self._append_cmd
        return cmd
