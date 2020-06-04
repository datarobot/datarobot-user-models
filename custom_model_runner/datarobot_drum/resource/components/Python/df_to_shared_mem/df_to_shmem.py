import pickle

from mlpiper.components.connectable_component import ConnectableComponent


class DfToSharedMem(ConnectableComponent):
    def __init__(self, engine):
        super(DfToSharedMem, self).__init__(engine)

    def configure(self, params):
        super(DfToSharedMem, self).configure(params)

    def _materialize(self, parent_data_objs, user_data):
        df = parent_data_objs[0]
        mmap_filename = self._params.get("output_filename")
        with open(mmap_filename, "w+b") as f:
            f.truncate(0)
            f.seek(0)
            pickle.dump(df, f)
            f.flush()

        return []
