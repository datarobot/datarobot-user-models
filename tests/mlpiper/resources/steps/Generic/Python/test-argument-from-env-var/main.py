from __future__ import print_function

import argparse
import os


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--arg1", help="Test argument 1")
    parser.add_argument("--fromEnvVar", help="Argument coming from env var")
    parser.add_argument("--fromEnvVar2", help="Argument coming from env var")
    options = parser.parse_args()
    return options


def main():

    print("Test - argument from env var")
    options = parse_args()
    expected_str_value = format(options.arg1)
    actual_value = format(options.fromEnvVar)

    if expected_str_value == "test-exit-0":
        exit(0)
    if expected_str_value == "test-exit-1":
        exit(1)

    print("arg1: {}".format(options.arg1))
    print("fromEnvVar: {}".format(options.fromEnvVar))

    if expected_str_value != actual_value:
        raise Exception(
            "Actual [{}] != Expected [{}]".format(actual_value, expected_str_value)
        )

    actual_value2 = format(options.fromEnvVar2)
    value_from_env = os.environ.get("TEST_VAR2")

    if value_from_env is not None and actual_value2 == value_from_env:
        raise Exception(
            "Actual [{}] == Value from env [{}]".format(actual_value2, value_from_env)
        )


if __name__ == "__main__":
    main()
