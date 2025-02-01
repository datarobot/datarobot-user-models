"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging

from datarobot_drum.drum.adapters.cli.shared.drum_class_label_adapter import DrumClassLabelAdapter
from datarobot_drum.drum.adapters.cli.shared.drum_input_file_adapter import DrumInputFileAdapter
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class DrumScoreAdapter(DrumInputFileAdapter, DrumClassLabelAdapter):
    """
    This class acts as a layer between the CLI and performing `drum score`. It will convert
    the arguments (mostly paths) into tangible variables to be passed into fit or predict.
    """

    def __init__(
        self,
        custom_task_folder_path,
        target_type,
        input_filename=None,
        sparse_column_filename=None,
        positive_class_label=None,
        negative_class_label=None,
        class_labels=None,
        use_datarobot_predict=False,
        forecast_point=None,
        predictions_start_date=None,
        predictions_end_date=None,
    ):
        """
        Parameters
        ----------
        custom_task_folder_path: str
            Path to the custom task folder
        target_type: datarobot_drum.drum.enum.TargetType
        input_filename: str or None
            Path to the input training dataset
        sparse_column_filename: str or None
            Optional. Path to the column names if it's not included in the input training dataset
        positive_class_label: str or None
            Optional. Name of the positive class label if target type is binary
        negative_class_label: str or None
            Optional. Name of the negative class label if target type is binary
        class_labels: list[str] or None
            Optional. List of class labels
        use_datarobot_predict: bool
            Optional. Whether to use datarobot-predict package or not
        forecast_point : str or None
            Optional, Forecast point as timestamp in ISO format
        predictions_start_date : str or None
            Optional, Start of predictions as timestamp in ISO format  
        predictions_end_date : str or None
            Optional, End of predictions as timestamp in ISO format
        """
        DrumInputFileAdapter.__init__(
            self=self,
            target_type=target_type,
            input_filename=input_filename,
            sparse_column_filename=sparse_column_filename,
        )
        DrumClassLabelAdapter.__init__(
            self=self,
            target_type=target_type,
            positive_class_label=positive_class_label,
            negative_class_label=negative_class_label,
            class_labels=class_labels,
        )

        self.use_datarobot_predict = use_datarobot_predict
        self.custom_task_folder_path = custom_task_folder_path
        self.forecast_point = forecast_point
        self.predictions_start_date = predictions_start_date
        self.predictions_end_date = predictions_end_date
