from mlpiper.components import ConnectableComponent

from sklearn.preprocessing import FunctionTransformer


class CategoricalsConverter(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        X = parent_data_objs[0]
        y = parent_data_objs[1]

        transformer = FunctionTransformer(
            CategoricalsConverter._convert_to_categoricals, validate=False
        )
        X_new = transformer.fit_transform(X, y)

        return [X_new]

    @staticmethod
    def _convert_to_categoricals(X, y=None):
        """Helper function that calls `.categorize()`
        on its arguments before returning them.
        """
        if y is not None:
            return X.astype("category"), y.astype("category")

        return X.astype("category")
