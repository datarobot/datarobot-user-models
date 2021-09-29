import argparse
import os
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import ArgumentsOptions, ArgumentOptionsEnvVars
from datarobot_drum.resource.utils import _exec_shell_cmd
from datarobot_drum.drum.utils import unset_drum_supported_env_vars


class TestDrumHelp:
    @pytest.mark.parametrize(
        "cmd",
        [
            "{}".format(ArgumentsOptions.MAIN_COMMAND),
            "{} --help".format(ArgumentsOptions.MAIN_COMMAND),
        ],
    )
    def test_drum_help(self, cmd):
        _, stdo, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert "usage: drum" in str(stdo)

    def test_drum_bad_subparser(self):
        cmd = ("{} some_command".format(ArgumentsOptions.MAIN_COMMAND),)
        _, _, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert "argument subparser_name: invalid choice: 'some_command'" in str(stde)


class TestMulticlassLabelsParser(object):
    @pytest.fixture
    def parser(self):
        test_parser = argparse.ArgumentParser()
        subparsers = test_parser.add_subparsers(dest="command")
        label_parser = subparsers.add_parser("dummy")
        CMRunnerArgsRegistry._reg_arg_multiclass_labels(label_parser)
        return test_parser

    @pytest.fixture
    def class_labels(self):
        return list("ABC")

    @pytest.fixture
    def unordered_class_labels(self):
        return list("BAC")

    @pytest.fixture
    def class_labels_with_spaces(self):
        return ["Label {}".format(i) for i in range(4)]

    @pytest.mark.parametrize(
        "valid_labels", ["class_labels", "unordered_class_labels", "class_labels_with_spaces"]
    )
    @pytest.mark.parametrize("as_file", [True, False])
    def test_valid_class_labels(self, request, valid_labels, as_file, parser):
        valid_labels = request.getfixturevalue(valid_labels)
        with NamedTemporaryFile() as f:
            if as_file:
                f.write("\n".join(valid_labels).encode("utf-8"))
                f.flush()
                args = "dummy --class-labels-file {}".format(f.name).split()
            else:
                args = ["dummy", "--class-labels", *valid_labels]
            options = parser.parse_args(args)

        assert options.class_labels == valid_labels

    @pytest.mark.parametrize("as_file", [True, False])
    def test_too_few_labels(self, as_file, parser):
        labels = list("A")
        with NamedTemporaryFile() as f:
            if as_file:
                f.write("\n".join(labels).encode("utf-8"))
                f.flush()
                args = "dummy --class-labels-file {}".format(f.name).split()
            else:
                args = ["dummy", "--class-labels", *labels]

            with pytest.raises(argparse.ArgumentTypeError, match="at least 2"):
                parser.parse_args(args)


class TestPosNegLabelsParser(object):
    @pytest.fixture
    def parser(self):
        test_parser = argparse.ArgumentParser()
        subparsers = test_parser.add_subparsers(dest="command")
        label_parser = subparsers.add_parser("dummy")
        CMRunnerArgsRegistry._reg_arg_pos_neg_labels(label_parser)
        return test_parser

    @pytest.fixture
    def numeric_class_labels(self):
        return [1.0, 0.0]

    @pytest.fixture
    def bool_class_labels(self):
        return [True, False]

    @pytest.fixture
    def str_class_labels(self):
        return ["Yes", "No"]

    @pytest.mark.parametrize(
        "valid_labels", ["numeric_class_labels", "bool_class_labels", "str_class_labels"]
    )
    def test_valid_class_labels(self, request, valid_labels, parser):
        valid_labels = request.getfixturevalue(valid_labels)
        args = "dummy --positive-class-label {} --negative-class-label {}".format(
            valid_labels[0], valid_labels[1]
        ).split()
        with patch.object(sys, "argv", args):
            options = parser.parse_args(args)

        actual_labels = [options.positive_class_label, options.negative_class_label]
        assert all(isinstance(label, str) for label in actual_labels)
        assert actual_labels == [str(label) for label in valid_labels]


class TestStrictValidationParser(object):
    @pytest.fixture
    def parser(self):
        test_parser = argparse.ArgumentParser()
        subparsers = test_parser.add_subparsers(dest="command")
        label_parser = subparsers.add_parser("dummy")
        CMRunnerArgsRegistry._reg_arg_strict_validation(label_parser)
        return test_parser

    def test_disable_strict_validation(self, parser):
        args = "dummy --disable-strict-validation".split()
        with patch.object(sys, "argv", args):
            options = parser.parse_args(args)

        assert options.disable_strict_validation

    def test_enable_strict_validation(self, parser):
        # Test the strict validation is enabled by default
        args = "dummy".split()
        with patch.object(sys, "argv", args):
            options = parser.parse_args(args)

        assert not options.disable_strict_validation


class TestBooleanArgumentOptions:
    def test_boolean_argument_options_passed_through_env_var(self):
        args = f"{ArgumentsOptions.MAIN_COMMAND} server  --with-error-server --skip-predict".split()
        with patch.object(sys, "argv", args):
            unset_drum_supported_env_vars()
            CMRunnerArgsRegistry.get_arg_parser()
            os.environ[ArgumentOptionsEnvVars.PRODUCTION] = "True"
            os.environ[ArgumentOptionsEnvVars.MONITOR] = "False"
            os.environ[ArgumentOptionsEnvVars.WITH_ERROR_SERVER] = "1"
            os.environ[ArgumentOptionsEnvVars.SKIP_PREDICT] = "False"
            CMRunnerArgsRegistry.extend_sys_argv_with_env_vars()
            assert ArgumentsOptions.PRODUCTION in sys.argv
            assert ArgumentsOptions.MONITOR not in sys.argv
            assert ArgumentsOptions.WITH_ERROR_SERVER in sys.argv
            assert ArgumentsOptions.SKIP_PREDICT in sys.argv
            assert sys.argv.count(ArgumentsOptions.WITH_ERROR_SERVER) == 1
            assert sys.argv.count(ArgumentsOptions.SKIP_PREDICT) == 1
            unset_drum_supported_env_vars()
