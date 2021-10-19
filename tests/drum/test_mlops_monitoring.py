import os
import pytest
import pandas as pd

from datarobot_drum.drum.enum import ArgumentsOptions

from .constants import SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, BINARY

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
    _cmd_add_class_labels,
    _create_custom_model_dir,
)


class TestMLOpsMonitoring:
    @staticmethod
    def _drum_with_monitoring(resources, framework, problem, language, docker, tmp_path):
        """
        We expect the run of drum to be ok, since mlops is assumed to be installed.
        """
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        mlops_spool_dir = tmp_path / "mlops_spool"
        os.mkdir(str(mlops_spool_dir))

        input_dataset = resources.datasets(framework, problem)
        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )
        monitor_settings = "spooler_type=filesystem;directory={};max_files=1;file_max_size=1024000".format(
            mlops_spool_dir
        )
        cmd += ' --monitor --model-id 555 --deployment-id 777 --monitor-settings="{}"'.format(
            monitor_settings
        )

        if problem == BINARY:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, problem),
                target_type=resources.target_types(problem),
            )
        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        return cmd, input_dataset, output, mlops_spool_dir

    @pytest.mark.parametrize(
        "framework, problem, language, docker", [(SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),],
    )
    def test_drum_monitoring_with_mlops_installed(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        cmd, input_file, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
            resources, framework, problem, language, docker, tmp_path
        )

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = pd.read_csv(input_file)
        out_data = pd.read_csv(output_file)
        assert in_data.shape[0] == out_data.shape[0]

        print("Spool dir {}".format(mlops_spool_dir))
        assert os.path.isdir(mlops_spool_dir)
        assert os.path.isfile(os.path.join(mlops_spool_dir, "fs_spool.1"))

    @pytest.mark.parametrize(
        "framework, problem, language, docker", [(SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),],
    )
    def test_drum_monitoring_no_mlops_installed(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        """
        We expect the run of drum to fail since the mlops package is assumed to not be installed
        Returns
        -------

        """
        cmd, _, _, _ = TestMLOpsMonitoring._drum_with_monitoring(
            resources, framework, problem, language, docker, tmp_path
        )
        p, _, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert (
            p.returncode != 0
        ), "drum should fail when datarobot-mlops is not installed and monitoring is requested"

    @pytest.mark.parametrize(
        "framework, problem, language, docker", [(SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),],
    )
    def test_drum_monitoring_fails_in_unstructured_mode(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        cmd, input_file, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
            resources, framework, problem, language, docker, tmp_path
        )

        cmd += " --target-type unstructured"
        _, stdo, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        assert str(stdo).find("MLOps monitoring can not be used in unstructured mode") != -1
