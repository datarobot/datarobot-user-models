from mlpiper.components import ConnectableComponent

from sklearn.model_selection import train_test_split


class SplitTarget(ConnectableComponent):
    """Given a dataframe containing a column,
    extract that column as a separate dataframe.
    Parameters
    ----------
    df : DataFrame
        Must contain a column whose name is the value of the 'target' argument
    target : str
        Name of the target column.  Default 'readmitted'.
    Returns
    -------
    A tuple of the form (X, y).
    X : DataFrame
        A dataframe with the same contents as 'df'
        except excluding the specified target column
    y : DataFrame
        A dataframe with the contents of the
        specified target column
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        if not parent_data_objs:
            raise Exception("Missing expected inpurt Pandas dataframee")

        train_size = self._params.get("train_size", 0.7)
        test_size = self._params.get("test_size", 0.3)
        random_state = self._params.get("random_state", 42)

        df = parent_data_objs[0]

        df = df.dropna()

        target = self._params.get("target", "readmitted")

        X = df.copy()
        del X[target]

        y = df.loc[:, [target]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=random_state
        )

        return [X_train, X_test, y_train, y_test]
