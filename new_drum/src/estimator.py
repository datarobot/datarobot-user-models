from .serializable import Serializable


class EstimatorInterface(Serializable):

    def fit(self, X, y, row_weights=None, **kwargs):
        """ This hook defines how DataRobot will train this task. Even transform tasks need to be
        trained to learn/store information from training data
        DataRobot runs this hook when the task is being trained inside a blueprint.
        As an output, this hook is expected to create an artifact containg a trained object [in
        this example - median of each numeric column], that is then used to transform new data.
        The input parameters are passed by DataRobot based on project and blueprint configuration.
        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column (None is passed for unsupervised projects).
        row_weights: np.ndarray (optional, default = None)
            A list of weights. DataRobot passes it in case of smart downsampling or when weights
            column is specified in project settings.
        Returns
        -------
        EstimatorInterface
        """
        raise NotImplementedError()

    def score(self, X, **kwargs):
        """ This hook defines how DataRobot will use the trained object from fit() to score new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the scored data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.
        Parameters
        -------
        X: pd.DataFrame
            Data that DataRobot passes for scoring.
        Returns
        -------
        pd.DataFrame
            Returns a dataframe with scored data.
            In case of regression, score() must return a dataframe with a single column with column
            name "Predictions".
        """
        raise NotImplementedError()