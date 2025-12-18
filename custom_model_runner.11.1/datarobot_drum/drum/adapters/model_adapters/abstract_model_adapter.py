#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class AbstractModelAdapter(ABC):
    @abstractmethod
    def load_custom_hooks(self):
        """Loads the customer code and its hooks: load_model, fit, predict, ... either from the deprecated
        top-level methods in custom.py or from the new interface classes:

        - datarobot_drum.custom_task_interfaces.BinaryEstimatorInterface
        - datarobot_drum.custom_task_interfaces.RegressionEstimatorInterface
        - datarobot_drum.custom_task_interfaces.MulticlassEstimatorInterface
        - datarobot_drum.custom_task_interfaces.AnomalyEstimatorInterface

        """
        raise NotImplementedError()

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str,
        class_order: Optional[list],
        row_weights: Optional[pd.Series],
        parameters: Optional[dict],
        user_secrets_mount_path: Optional[str],
        user_secrets_prefix: Optional[str],
    ) -> "AbstractModelAdapter":
        """
        Trains a custom task.

        Parameters
        ----------
        X:
            Training data. Could be sparse or dense
        y:
            Target values
        output_dir:
            Output directory to store the artifact
        class_order:
            Expected order of classification labels
        row_weights:
            Class weights
        parameters:
            Hyperparameter values
        user_secrets_mount_path:
            The directory where mounted secrets would be found
        user_secrets_prefix:
            The prefix for secret env vars
        """
        raise NotImplementedError()
