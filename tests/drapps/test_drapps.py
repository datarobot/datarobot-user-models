"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pytest

from datarobot_drum.resource.utils import _exec_shell_cmd


class TestDRApps:
    @pytest.mark.parametrize(
        "script_name",
        ["drapps.py", "drapps"],
    )
    def test_drapps_help(self, script_name):
        cmd = f"{script_name} --help"

        _, stdo, stde = _exec_shell_cmd(
            cmd, "Failed command line! {}".format(cmd), assert_if_fail=False
        )

        stdo_stde = str(stdo) + str(stde)

        assert str(stdo_stde).find(f"Usage: {script_name}") != -1

    @pytest.mark.parametrize(
        "script_name",
        ["drapps.py", "drapps"],
    )
    def test_drapps_no_params_provided(self, script_name):
        cmd = f"{script_name}"

        _, stdo, stde = _exec_shell_cmd(
            cmd, "Failed command line! {}".format(cmd), assert_if_fail=False
        )

        stdo_stde = str(stdo) + str(stde)

        assert str(stdo_stde).find("Missing option '-e'") != -1
