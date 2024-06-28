from __future__ import print_function

import argparse
import sys
import pprint
import os

from mlpiper.components import ConnectableComponent

TMP_DIR_ENV = "MCENTER_MODEL_CONNECTOR_TMP_DIR"
DEFAULT_MODEL_CONTENT = "model_1234"


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for the do_train
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        model_file = import_model(self._params)
        return [model_file]


def _get_tmp_dir():
    return os.environ.get(TMP_DIR_ENV, "/tmp")


def import_model(options):
    print("importing model", file=sys.stderr)

    model_path = os.path.join(_get_tmp_dir(), "m1")
    with open(model_path, "w") as model_file:
        model_file.write(DEFAULT_MODEL_CONTENT)
    return model_path


def list_params(options):
    print("listing params", file=sys.stderr)
    raise Exception("Not done yet")


def parse_args():
    parser = argparse.ArgumentParser()

    # The cmd is used to determine how to run the connector (import/list_params)
    parser.add_argument("--cmd", default="import", help="command to perform")
    parser.add_argument("--json-file", default=None, help="JSON file for input output")

    # All arguments below are components arguments
    parser.add_argument("--server", default="localhost", help="Server name")
    parser.add_argument("--username", default=None, help="Username")
    parser.add_argument("--password", default=None, help="Password")
    parser.add_argument("--project-name", default=None, help="Project name")
    parser.add_argument("--model-name", default=None, help="Model name")

    options = parser.parse_args()
    return options


def main():
    print("args: {}".format(sys.argv))
    options = parse_args()
    pprint.pprint(options)
    if options.cmd == "import":
        model_path = import_model(options)
        print(model_path)
    elif options.cmd == "list-params":
        list_params(options)
    else:
        raise Exception("CMD: [{}] is not supported".format(options.cmd))


if __name__ == "__main__":
    main()
