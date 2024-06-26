from __future__ import print_function

import sys
import time
import os

from mlpiper.components import ConnectableComponent


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for the do_train
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        model_output = self._params.get("input_model")
        iter_num = self._params.get("iter")
        exit_value = self._params.get("exit_value")

        do_predict(model_output, iter_num, exit_value)
        return ["just-a-string-to-connect"]


def do_predict(input_model, iter_num, exit_value):

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

    if exit_value != 0:
        print("About to raise exception: {}".format(exit_value))
        raise Exception("Exiting component with error code: {}".format(exit_value))
    else:
        print("Normal component's execution ({})".format(os.path.basename(__file__)))
