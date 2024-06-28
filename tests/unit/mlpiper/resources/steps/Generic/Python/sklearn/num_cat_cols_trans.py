from mlpiper.components import ConnectableComponent

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class NumCatColsTrans(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        if len(parent_data_objs) != 2:
            raise Exception("Unexpected input dataframes")

        X = parent_data_objs[0]
        y = parent_data_objs[1]

        numerical_features = X.dtypes == "int64"
        categorical_features = ~numerical_features

        transformer = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(), numerical_features),
                ("cat", OrdinalEncoder(), categorical_features),
            ]
        )

        X_new = transformer.fit_transform(X, y)

        return [X_new]
