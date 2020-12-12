import argparse
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry


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
