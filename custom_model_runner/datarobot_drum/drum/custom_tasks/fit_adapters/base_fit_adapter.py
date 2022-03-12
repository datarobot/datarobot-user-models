"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
from abc import ABC

from datarobot_drum.drum.custom_tasks.fit_adapters.classification_labels_util import (
    needs_class_labels,
    infer_class_labels,
)
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX, TargetType

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class BaseFitAdapter(ABC):
    """
    This class acts as a layer between the DRUM CLI and running a custom task's fit method. It will convert the
    cli arguments (mostly paths) into tangible variables to be passed into fit.
    """

    def __init__(
        self,
        custom_task_folder_path,
        input_filename,
        target_type,
        target_name=None,
        target_filename=None,
        weights=None,
        weights_filename=None,
        sparse_column_filename=None,
        positive_class_label=None,
        negative_class_label=None,
        class_labels=None,
        parameters_file=None,
        default_parameter_values=None,
        output_dir=None,
        num_rows=None,
    ):
        """
        Parameters
        ----------
        custom_task_folder_path: str
            Path to the custom task folder
        input_filename: str
            Path to the input training dataset
        target_type: datarobot_drum.drum.enum.TargetType
        target_name: str or None
            Optional. Name of the target column in the input training dataset
        target_filename: str or None
            Optional. Path to the target values if it's not included in the input training dataset
        weights: str or None
            Optional. Name of the weights column in the input training dataset
        weights_filename: str or None
            Optional. Path to the weight values if it's not included in the input training dataset
        sparse_column_filename: str or None
            Optional. Path to the column names if it's not included in the input training dataset
        positive_class_label: str or None
            Optional. Name of the positive class label if target type is binary
        negative_class_label: str or None
            Optional. Name of the negative class label if target type is binary
        class_labels: list[str] or None
            Optional. List of class labels
        parameters_file: str or None
            Optional. Path to the hyperparameter values
        default_parameter_values: dict[str, Any] or None
            Optional. Dict containing default parameter values from the model metadata
        output_dir: str or None
            Optional. Output directory to store the fit artifacts
        num_rows: int or None
            Optional. Number of rows
        """
        self.custom_task_folder_path = custom_task_folder_path
        self.input_filename = input_filename
        self.target_type = target_type
        self.target_name = target_name
        self.target_filename = target_filename
        self.weights = weights
        self.weights_filename = weights_filename
        self.sparse_column_filename = sparse_column_filename
        self.positive_class_label = positive_class_label
        self.negative_class_label = negative_class_label
        self.class_labels = class_labels
        self.parameters_file = parameters_file
        self.default_parameter_values = default_parameter_values
        self.output_dir = output_dir
        self.num_rows = num_rows

    def _infer_class_labels_if_needed(self):
        if needs_class_labels(
            target_type=self.target_type,
            negative_class_label=self.negative_class_label,
            positive_class_label=self.positive_class_label,
            class_labels=self.class_labels,
        ):
            class_labels = infer_class_labels(
                target_type=self.target_type,
                input_filename=self.input_filename,
                target_filename=self.target_filename,
                target_name=self.target_name,
            )

            if self.target_type == TargetType.BINARY:
                self.positive_class_label, self.negative_class_label = class_labels
            elif self.target_type == TargetType.MULTICLASS:
                self.class_labels = class_labels

    def configure(self):
        """
        Configure things before running fit. Should always be called from child classes.
        """
        self._infer_class_labels_if_needed()

    def outer_fit(self):
        raise NotImplementedError("FitAdapters must define outer_fit")
