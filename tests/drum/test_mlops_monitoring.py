import pytest
import pandas as pd
import os

from test_custom_model import TestCMRunner
from test_custom_model import SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM

from datarobot_drum.drum.common import (
    ArgumentsOptions,
)


class TestMLOpsMonitoring:
    @classmethod
    def setup_class(cls):
        TestCMRunner.setup_class()

    @classmethod
    def teardown_class(cls):
        pass

    @staticmethod
    def _drum_with_monitoring(framework, problem, language, docker, tmp_path):
        """
        We expect the run of drum to be ok, since mlops is assumed to be installed.
        """
        custom_model_dir = tmp_path / "custom_model"
        TestCMRunner._create_custom_model_dir(custom_model_dir, framework, problem, language)

        mlops_spool_dir = tmp_path / "mlops_spool"
        os.mkdir(str(mlops_spool_dir))

        input_dataset = TestCMRunner._get_dataset_filename(framework, problem)
        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output
        )
        monitor_settings = (
            "spooler_type=filesystem;directory={};max_files=1;file_max_size=1024000".format(
                mlops_spool_dir
            )
        )
        cmd += ' --monitor --model-id 555 --deployment-id 777 --monitor-settings="{}"'.format(
            monitor_settings
        )

        cmd = TestCMRunner._cmd_add_class_labels(cmd, framework, problem)
        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        return cmd, input_dataset, output, mlops_spool_dir

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
        ],
    )
    def test_drum_monitoring_with_mlops_installed(
        self, framework, problem, language, docker, tmp_path
    ):
        cmd, input_file, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
            framework, problem, language, docker, tmp_path
        )

        TestCMRunner._exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = pd.read_csv(input_file)
        out_data = pd.read_csv(output_file)
        assert in_data.shape[0] == out_data.shape[0]

        print("Spool dir {}".format(mlops_spool_dir))
        assert os.path.isdir(mlops_spool_dir)
        assert os.path.isfile(os.path.join(mlops_spool_dir, "fs_spool.1"))

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
        ],
    )
    def test_drum_monitoring_no_mlops_installed(
        self, framework, problem, language, docker, tmp_path
    ):
        """
        We expect the run of drum to fail since the mlops package is assumed to not be installed
        Returns
        -------

        """
        cmd, input_file, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
            framework, problem, language, docker, tmp_path
        )
        p, stdo, stde = TestCMRunner._exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert (
            p.returncode != 0
        ), "drum should fail when datarobot-mlops is not installed and monitoring is requested"
