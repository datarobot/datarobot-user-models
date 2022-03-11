"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import sys
import os

from mlpiper.components.connectable_component import ConnectableComponent

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX, TargetType, TARGET_TYPE_ARG_KEYWORD
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.utils import shared_fit_preprocessing, make_sure_artifact_is_small

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class BaseFitAdapter(object):
    def __init__(
        self,
        custom_task_folder_path=None,
        input_filename=None,
        target_name=None,
        target_filename=None,
        weights=None,
        weights_filename=None,
        sparse_column_filename=None,
        target_type=None,
        positive_class_label=None,
        negative_class_label=None,
        class_labels=None,
        parameters_file=None,
        default_parameter_values=None,
        output_dir=None,
        num_rows=None,
    ):
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
        pass
