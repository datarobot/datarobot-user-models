"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot_drum.custom_task_interfaces import TransformerInterface


class CustomTask(TransformerInterface):
    def fit(self, X, y, **kwargs):
        """This hook defines how DataRobot will train this task. Transformers will typically learn and/or
        store information from training data and then apply it, e.g. learning the median for each column
        and then transforming missing/NaN values to that median.
        Note that when training a blueprint we apply this fit hook. After fit, we run the below transform hook on the
        training data to pass to downstream tasks. In contrast, when scoring a blueprint on validation / holdout data
        we only use the below transform hook.
        The input parameters are passed by DataRobot based on project and blueprint configuration.
        The output is the trained Custom Task instance, which allows a user to easily test, e.g.
        task.fit(...).transform(...) or task.fit(...).save().transform(...)

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column (None is passed for unsupervised projects). This allows users to implement
            transformers that rely on the target values, e.g. target encoding

        Returns
        -------
        CustomTask
            returns an object instance of class CustomTask that can be used in chained method calls
        """

        # Any information derived from the training data (i.e. median values for each column) should be stored to self.
        # Then, in the transform hook below, we use this information to transform any data that passes through this task
        self.medians = X.median(axis=0, numeric_only=True, skipna=True).to_dict()
        return self

    def transform(self, X, **kwargs):
        """This hook defines how DataRobot will either use a trained transform from fit() to transform new data,
        e.g. imputing missing values, or will directly apply a stateless transformation, e.g. changing a data type
        DataRobot runs this hook when after fit is run when the blueprint is training and then by itself
        when the blueprint is used to score validation or holdout data
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.
        The output is the transformed data in tabular format (typically a pandas dataframe)

        Parameters
        -------
        X: pd.DataFrame
            Data that DataRobot passes for transformation.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with transformed data.
        """
        return X.fillna(self.medians)
