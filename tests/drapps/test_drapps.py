"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot_drum.resource.utils import _exec_shell_cmd


class TestDRApps:
    def test_drapps_help(self):
        cmd = "drapps.py --help"

        _, stdo, stde = _exec_shell_cmd(
            cmd, "Failed command line! {}".format(cmd), assert_if_fail=False
        )

        stdo_stde = str(stdo) + str(stde)

        assert str(stdo_stde).find("Usage: drapps.py") != -1

    def test_drapps_no_params_provided(self):
        cmd = "drapps.py"

        _, stdo, stde = _exec_shell_cmd(
            cmd, "Failed command line! {}".format(cmd), assert_if_fail=False
        )

        stdo_stde = str(stdo) + str(stde)

        assert str(stdo_stde).find("Missing option '-e'") != -1
