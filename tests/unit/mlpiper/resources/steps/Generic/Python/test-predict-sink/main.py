from __future__ import print_function

import argparse
import sys
import time
import os

from mlpiper.components import ConnectableComponent

DEFAULT_MODEL_CONTENT = "model-1234-test-train-python"


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for the do_train
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        model_content = self._params.get("model_content", DEFAULT_MODEL_CONTENT)
        model_output = self._params.get("model_output")
        iter_num = self._params.get("iter")
        exit_value = self._params.get("exit_value")

        do_predict(model_content, model_output, iter_num, exit_value)
        return ["just-a-string-to-connect"]


def do_predict(model_content, input_model, iter_num, exit_value):
    for idx in range(iter_num):
        print("stdout - Idx {}".format(idx))
        print("stderr- Idx  {}".format(idx), file=sys.stderr)
        time.sleep(1)

    # Model should be present
    if not os.path.exists(input_model):
        raise Exception("Model path does not exists: {}".format(input_model))

    with open(input_model) as file:
        model_content = file.readlines()
        print("Model content:\n", "\n".join(model_content))
        # TODO: compare model content to the model obtained

    # Some output - to test logs
    for idx in range(iter_num):
        print("stdout - Idx {}".format(idx))
        print("stderr- Idx  {}".format(idx), file=sys.stderr)
        time.sleep(1)

    if exit_value >= 0:
        print("About to exit with value: {}".format(exit_value))
        sys.exit(exit_value)
    else:
        print("About to raise exception: {}".format(exit_value))
        raise Exception("Exiting main using exception")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-content",
        default=DEFAULT_MODEL_CONTENT,
        help="Model content to generate",
    )
    parser.add_argument("--input-model", help="Path to load model from")
    parser.add_argument("--exit-value", type=int, default=0, help="Exit value")
    parser.add_argument("--iter", type=int, default=20, help="How many 1sec iterations to perform")

    options = parser.parse_args()
    return options


def main():
    print("args: {}".format(sys.argv))
    options = parse_args()
    print("- inside main.py Running main.py")
    print("model_content:    [{}]".format(options.arg1))
    print("output_model:     [{}]".format(options.output_model))
    print("iter:             [{}]".format(options.iter))
    print("exit_value:       [{}]".format(options.exit_value))

    do_predict(options.arg1, options.output_model, options.iter, options.exit_value)


if __name__ == "__main__":
    main()
