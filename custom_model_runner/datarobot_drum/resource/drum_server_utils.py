import os
import psutil
import requests
import signal
import time
from threading import Thread

from datarobot_drum.drum.utils import CMRunnerUtils
from datarobot_drum.drum.common import ArgumentsOptions
from datarobot_drum.resource.utils import _exec_shell_cmd, _cmd_add_class_labels


def _wait_for_server(url, timeout, process_holder):
    # waiting for ping to succeed
    while True:
        try:
            response = requests.get(url)
            if response.ok:
                break
        except Exception:
            pass

        time.sleep(1)
        timeout -= 1
        if timeout <= 0:
            if process_holder is not None:
                print("Killing subprocess: {}".format(process_holder.process.pid))
                os.killpg(os.getpgid(process_holder.process.pid), signal.SIGTERM)
                time.sleep(0.25)
                os.killpg(os.getpgid(process_holder.process.pid), signal.SIGKILL)

            assert timeout, "Server failed to start: url: {}".format(url)


def _run_server_thread(cmd, process_obj_holder, verbose=True):
    _exec_shell_cmd(
        cmd,
        "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
        assert_if_fail=False,
        process_obj_holder=process_obj_holder,
        verbose=verbose,
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
        target_type,
        labels,
        custom_model_dir,
        docker=None,
        with_error_server=False,
        show_stacktrace=True,
        nginx=False,
        memory=None,
        fail_on_shutdown_error=True,
        verbose=True,
    ):
        port = CMRunnerUtils.find_free_port()
        self.server_address = "localhost:{}".format(port)
        url_host = os.environ.get("TEST_URL_HOST", "localhost")

        if docker:
            self.url_server_address = "http://{}:{}".format(url_host, port)
        else:
            self.url_server_address = "http://localhost:{}".format(port)

        cmd = "{} server --code-dir {} --target-type {} --address {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, target_type, self.server_address
        )
        if labels:
            cmd = _cmd_add_class_labels(cmd, labels, target_type=target_type)
        if docker:
            cmd += " --docker {}".format(docker)
            if memory:
                cmd += " --memory {}".format(memory)
        if with_error_server:
            cmd += " --with-error-server"
        if show_stacktrace:
            cmd += " --show-stacktrace"
        if nginx:
            cmd += " --production"
        self._cmd = cmd

        self._process_object_holder = DrumServerProcess()
        self._server_thread = None
        self._with_nginx = nginx
        self._fail_on_shutdown_error = fail_on_shutdown_error
        self._verbose = verbose

    def __enter__(self):
        self._server_thread = Thread(
            target=_run_server_thread, args=(self._cmd, self._process_object_holder, self._verbose)
        )
        self._server_thread.start()
        time.sleep(0.5)

        _wait_for_server(
            self.url_server_address, timeout=10, process_holder=self._process_object_holder
        )

        return self

    def _shutdown_server(self):
        if not self._with_nginx:
            response = requests.post(self.url_server_address + "/shutdown/")
            assert response.ok
            time.sleep(1)
            self._server_thread.join()
        else:
            # When running with nginx:
            # this test starts drum process with --docker option,
            # that process starts drum server inside docker.
            # nginx server doesn't have shutdown API, so we need to kill it

            # This loop kill all the chain except for docker
            parent = psutil.Process(self._process_object_holder.process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()

            # this kills drum running in the docker
            for proc in psutil.process_iter():
                if "{}".format(ArgumentsOptions.MAIN_COMMAND) in proc.name().lower():
                    print(proc.cmdline())
                    if "{}".format(ArgumentsOptions.SERVER) in proc.cmdline():
                        if "--production" in proc.cmdline():
                            try:
                                proc.terminate()
                                time.sleep(0.3)
                                proc.kill()
                            except psutil.NoSuchProcess:
                                pass
                            break

            self._server_thread.join(timeout=5)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # shutdown server
        if self._fail_on_shutdown_error:
            self._shutdown_server()
        else:
            try:
                self._shutdown_server()
            except Exception:
                pass

    @property
    def process(self):
        return self._process_object_holder or None
