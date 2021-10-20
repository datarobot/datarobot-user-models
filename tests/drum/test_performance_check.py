import pytest
import subprocess
import os
import time

from datarobot_drum.drum.enum import ArgumentsOptions
from datarobot_drum.drum.perf_testing import (
    _find_drum_perf_test_server_process,
    _kill_drum_perf_test_server_process,
)

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
    _cmd_add_class_labels,
    _create_custom_model_dir,
)


from .constants import (
    SKLEARN,
    REGRESSION,
    BINARY,
    MULTICLASS,
    MULTICLASS_BINARY,
    PYTHON,
    DOCKER_PYTHON_SKLEARN,
)


class TestPerformanceCheck:
    @pytest.mark.parametrize(
        "framework, problem, language, docker, timeout",
        [
            (SKLEARN, BINARY, PYTHON, None, None),
            (SKLEARN, BINARY, PYTHON, None, 5),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN, 5),
            (SKLEARN, MULTICLASS, PYTHON, None, None),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, None, None),
        ],
    )
    def test_custom_models_perf_test(
        self, resources, framework, problem, language, docker, timeout, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        input_dataset = resources.datasets(framework, problem)

        cmd = "{} perf-test -i 200 -s 1000 --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            resources.target_types(problem),
        )

        if timeout is not None:
            cmd += " --timeout {}".format(timeout)

        if resources.target_types(problem) in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, problem),
                target_type=resources.target_types(problem),
            )

        if docker:
            cmd += " --docker {}".format(docker)

        _, stdo, _ = _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        if timeout is not None:
            expected_str = "timed out ({}s)".format(timeout)
            assert expected_str in stdo
            assert "NA" in stdo
        else:
            assert "NA" not in stdo

    @pytest.mark.parametrize(
        "framework, problem, language, docker", [(SKLEARN, REGRESSION, PYTHON, None),],
    )
    def test_perf_test_drum_server_kill(
        self, resources, framework, problem, language, docker, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        input_dataset = resources.datasets(framework, problem)

        cmd = "{} perf-test -i 10 -s 10 --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            resources.target_types(problem),
        )

        # wait for drum perf-test server from prev test case to be stopped
        time.sleep(0.5)
        assert _find_drum_perf_test_server_process() is None
        p = subprocess.Popen(cmd, shell=True, env=os.environ, universal_newlines=True,)
        time.sleep(2)
        # kill drum perf-test process, child server should be running
        p.kill()
        pid = _find_drum_perf_test_server_process()
        assert pid is not None
        _kill_drum_perf_test_server_process(pid)
        assert _find_drum_perf_test_server_process() is None
