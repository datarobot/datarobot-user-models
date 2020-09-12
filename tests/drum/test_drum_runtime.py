import collections
import pytest
from unittest import mock

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import RunMode
from datarobot_drum.drum.runtime import DrumRuntime


class TestDrumRuntime:
    Options = collections.namedtuple(
        "Options",
        "with_error_server {} docker address verbose show_stacktrace".format(
            CMRunnerArgsRegistry.SUBPARSER_DEST_KEYWORD
        ),
        defaults=[RunMode.SERVER, None, "localhost", False, True],
    )

    class StubDrumException(Exception):
        pass

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_no_exceptions(self, mock_run_error_server):
        with DrumRuntime():
            pass

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_no_options(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime():
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_initialization_succeeded(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False)
                runtime.initialization_succeeded = True
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_not_server_mode(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False, RunMode.SCORE)
                runtime.initialization_succeeded = False
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_not_server_mode(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False, RunMode.SERVER, "path_to_image")
                runtime.initialization_succeeded = False
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_no_with_error_server(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False)
                runtime.initialization_succeeded = False
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_with_error_server(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(True)
                runtime.initialization_succeeded = False
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_called()
