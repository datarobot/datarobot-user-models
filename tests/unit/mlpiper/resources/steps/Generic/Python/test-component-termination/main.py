from __future__ import print_function

from mlpiper.components import ConnectableComponent


class DummyComponentToTestTermination(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        print("{}: Component body".format(self.__class__.__name__))

    def terminate(self):
        print("{}: Handle termination".format(self.__class__.__name__))
        with open(self._params["filepath-for-termination-output"], "w") as f:
            f.write('Termination was handled successfully')
