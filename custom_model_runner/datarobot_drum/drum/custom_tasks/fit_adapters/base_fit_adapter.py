"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
from abc import ABC

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class BaseFitAdapter(ABC):
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
        target_type: str
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
        class_labels: str
        parameters_file: str
        default_parameter_values: str
        output_dir: str
        num_rows: int
        """
        self.custom_task_folder_path = custom_task_folder_path
        self.input_filename = input_filename
        self.target_name = target_name
        self.target_filename = target_filename
        self.weights = weights
        self.weights_filename = weights_filename
        self.sparse_column_filename = sparse_column_filename
        self.target_type = target_type
        self.positive_class_label = positive_class_label
        self.negative_class_label = negative_class_label
        self.class_labels = class_labels
        self.parameters_file = parameters_file
        self.default_parameter_values = default_parameter_values
        self.output_dir = output_dir
        self.num_rows = num_rows

    def configure(self):
        pass

    def outer_fit(self):
        raise NotImplementedError("FitAdapters must define outer_fit")