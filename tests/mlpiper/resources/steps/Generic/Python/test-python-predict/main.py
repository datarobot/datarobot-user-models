from __future__ import print_function

import argparse
import os
import sys
import time


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--arg1", help="Test argument 1")
    parser.add_argument("--input-model", help="Path to read input model from")
    parser.add_argument(
        "--model-is-directory", type=int, default=0, help="Whether model is a directory"
    )
    parser.add_argument("--exit-value", type=int, default=0, help="Exit value")
    parser.add_argument(
        "--iter", type=int, default=20, help="How many 1sec iterations to perform"
    )

    options = parser.parse_args()
    return options


def main():

    print("args: {}".format(sys.argv))
    options = parse_args()
    print("- inside test-python-predict Running main.py")
    print("arg1:         {}".format(options.arg1))
    print("input_model: {}".format(options.input_model))
    print("model_is_directory: {}".format(options.model_is_directory))
    print("iter:         {}".format(options.iter))
    print("exit_value:   {}".format(options.exit_value))

    # Model should be present
    if not os.path.exists(options.input_model):
        raise Exception("Model path does not exist: {}".format(options.input_model))

    if options.model_is_directory == 1:
        model_file = os.path.join(options.input_model, "saved_model.pb")
    else:
        model_file = options.input_model

    print("Reading from model file {}".format(model_file))
    with open(model_file) as f:
        model_content = f.readlines()
        print("Model content:\n", "\n".join(model_content))

    # Some output - to test logs
    for idx in range(options.iter):
        print("stdout - Idx {}".format(idx))
        print("stderr- Idx  {}".format(idx), file=sys.stderr)
        time.sleep(1)

    # Exit status
    if options.exit_value >= 0:
        print("About to exit with value: {}".format(options.exit_value))
        sys.exit(options.exit_value)
    else:
        print("About to raise exception: {}".format(options.exit_value))
        raise Exception("Exiting main using exception")


if __name__ == "__main__":
    main()
