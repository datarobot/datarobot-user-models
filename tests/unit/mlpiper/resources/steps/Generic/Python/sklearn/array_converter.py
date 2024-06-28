from mlpiper.components import ConnectableComponent

# from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder


class ArrayConverter(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        X = parent_data_objs[0]
        y = parent_data_objs[1]

        enc = OrdinalEncoder()
        X_new = enc.fit_transform(X, y)

        # transformer = FunctionTransformer(
        #     ArrayConverter._convert_to_array, validate=True, pass_y=True
        # )

        # X_new = transformer.fit_transform(X, y)

        return [X_new]

    # @staticmethod
    # def _convert_to_array(X, y=None):
    #     """Helper function that converts its arguments
    #     to dask arrays of type `float`
    #     """
    #     if y is not None:
    #         return array.from_array(X.astype(float)), array.from_array(y.astype(float))
    #     return array.from_array(X.astype(float))
