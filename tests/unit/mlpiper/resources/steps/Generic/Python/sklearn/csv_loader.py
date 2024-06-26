import pandas as pd
from mlpiper.components import ConnectableComponent


class CsvLoader(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        dataset_path = self._params.get("path")

        data = pd.read_csv(dataset_path)

        self._logger.info(data.head())

        return [data]
