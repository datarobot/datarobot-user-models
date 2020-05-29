import pickle

from mlpiper.components.connectable_component import ConnectableComponent


class SharedMemToDf(ConnectableComponent):
    def __init__(self, engine):
        super(SharedMemToDf, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        mmap_filename = self._params.get("input_filename")
        with open(mmap_filename, "r+b") as f:
            df = pickle.load(f)
        return [df]
