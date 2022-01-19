from datarobot_drum.custom_task_interfaces.custom_task_interface import CustomTaskInterface


class TransformerInterface(CustomTaskInterface):
    def transform(self, X):
        """
        This hook defines how DataRobot will use the trained object from fit() to transform new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the transformed data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Data that DataRobot passes for transformation.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with transformed data.
        """
        raise NotImplementedError()
