import numpy as np
import pandas as pd

from mlpiper.components import ConnectableComponent


class RandomDataframe(ConnectableComponent):
    """
    Generating a random dataframe. The number of rows and columns is provided as input
    parameters to the component
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        num_rows = self._params.get("num_lines", 100)
        num_cols = self._params.get("num_cols", 5)

        df = pd.DataFrame(np.random.randint(0, 100, size=(num_rows, num_cols)))
        self._logger.info(
            "Generated random dataframe rows: {} cols: {})".format(num_rows, num_cols)
        )
        return [df]
