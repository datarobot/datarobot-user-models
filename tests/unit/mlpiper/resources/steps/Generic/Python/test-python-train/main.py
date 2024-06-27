from __future__ import print_function

import argparse
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arg1", help="Test argument 1")
    parser.add_argument("--output-model", help="Path to store generated model")
    parser.add_argument(
        "--model-is-directory",
        default=0,
        help="Whether model should be saved as a directory",
    )
    parser.add_argument("--import-tensorflow", default=0, help="Whether to import tensorflow")
    parser.add_argument("--exit-value", type=int, default=0, help="Exit value")
    parser.add_argument("--iter", type=int, default=20, help="How many 1sec iterations to perform")

    # TODO add model size as argument
    # TODO add mlops test as argument

    options = parser.parse_args()
    return options


def main():
    print("args: {}".format(sys.argv))
    options = parse_args()
    print("- inside test-python-train.main.py Running main.py")
    print("arg1:         {}".format(options.arg1))
    print("output_model: {}".format(options.output_model))
    print("model_is_directory: {}".format(options.model_is_directory))
    print("import_tensorflow: {}".format(options.import_tensorflow))
    print("iter:         {}".format(options.iter))
    print("exit_value:   {}".format(options.exit_value))

    for idx in range(options.iter):
        print("stdout - Idx {}".format(idx))
        print("stderr- Idx  {}".format(idx), file=sys.stderr)
        time.sleep(1)

    if options.import_tensorflow:
        import tensorflow as tf

        feature_configs = {"x": tf.FixedLenFeature(shape=[784], dtype=tf.float32)}
        print("feature_configs: {}".format(feature_configs))

    if options.output_model is not None:
        if options.model_is_directory == 0:
            with open(options.output_model, "w") as f:
                f.write("model-1234-test-train-python")
        else:
            os.mkdir(options.output_model)
            filename = os.path.join(options.output_model, "saved_model.pb")
            with open(filename, "a+") as f:
                f.write("model-1234-test-train-tf")

    if options.exit_value >= 0:
        print("About to exit with value: {}".format(options.exit_value))
        sys.exit(options.exit_value)
    else:
        print("About to raise exception: {}".format(options.exit_value))
        raise Exception("Exiting main using exception")


if __name__ == "__main__":
    main()
