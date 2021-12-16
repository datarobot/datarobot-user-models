from serializable import Serializable

class TransformerInterface(Serializable):

    def fit(self, X, y, **kwargs):
        """ This hook defines how DataRobot will train this task. Even transform tasks need to be
        trained to learn/store information from training data
        DataRobot runs this hook when the task is being trained inside a blueprint.
        As an output, this hook is expected to create an artifact containg a trained object
        [in this example - median of each numeric column], that is then used to transform new data.
        The input parameters are passed by DataRobot based on project and blueprint configuration.
        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column (None is passed for unsupervised projects).
        Returns
        -------
        TransformerInterface
        """
        return self

    def transform(self, data):
        """ This hook defines how DataRobot will use the trained object from fit() to transform new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the transformed data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.
        Parameters
        -------
        data: pd.DataFrame
            Data that DataRobot passes for transformation.
        Returns
        -------
        pd.DataFrame
            Returns a dataframe with transformed data.
        """
        raise NotImplementedError()

