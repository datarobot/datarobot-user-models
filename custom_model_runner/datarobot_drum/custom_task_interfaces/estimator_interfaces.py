from datarobot_drum.custom_task_interfaces.custom_task_interface import CustomTaskInterface


class EstimatorInterface(CustomTaskInterface):
    def predict(self, X, **kwargs):
        """
        This hook defines how DataRobot will use the trained object from fit() to predict new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the predicted data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Data that DataRobot passes for scoring.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with predict data.
            In case of regression, predict() must return a dataframe with a single column with column
            name "Predictions".
        """
        raise NotImplementedError()


class BinaryEstimatorInterface(EstimatorInterface):
    def predict(self, X, **kwargs):
        pass

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError()


class RegressionEstimatorInterface(EstimatorInterface):
    pass


class MulticlassEstimatorInterface(EstimatorInterface):
    def predict(self, X, **kwargs):
        pass

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError()


class AnomalyEstimatorInterface(EstimatorInterface):
    pass
