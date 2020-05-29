import pandas as pd

from mlpiper.components.connectable_component import ConnectableComponent


class CsvToDf(ConnectableComponent):
    def __init__(self, engine):
        super(CsvToDf, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        input_filename = self._params.get("input_filename", "default-string-value")
        df = pd.read_csv(input_filename)
        return [df]
