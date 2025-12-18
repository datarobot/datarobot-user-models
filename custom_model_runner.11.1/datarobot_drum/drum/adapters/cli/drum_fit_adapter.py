"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import logging
import shutil
from tempfile import mkdtemp

from datarobot_drum.drum.adapters.cli.shared.drum_class_label_adapter import DrumClassLabelAdapter
from datarobot_drum.drum.adapters.cli.shared.drum_input_file_adapter import DrumInputFileAdapter
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class DrumFitAdapter(DrumInputFileAdapter, DrumClassLabelAdapter):
    """
    This class acts as a layer between DRUM CLI and performing `drum fit`. It will convert
    the arguments (mostly paths) into tangible variables to be passed into fit or predict.
    """

    def __init__(
        self,
        custom_task_folder_path,
        target_type,
        input_filename=None,
        target_name=None,
        target_filename=None,
        weights_name=None,
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
        target_type: datarobot_drum.drum.enum.TargetType
        input_filename: str or None
            Path to the input training dataset
        target_name: str or None
            Optional. Name of the target column in the input training dataset
        target_filename: str or None
            Optional. Path to the target values if it's not included in the input training dataset
        weights_name: str or None
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
        DrumInputFileAdapter.__init__(
            self=self,
            target_type=target_type,
            input_filename=input_filename,
            target_name=target_name,
            target_filename=target_filename,
            weights_name=weights_name,
            weights_filename=weights_filename,
            sparse_column_filename=sparse_column_filename,
        )
        DrumClassLabelAdapter.__init__(
            self=self,
            target_type=target_type,
            positive_class_label=positive_class_label,
            negative_class_label=negative_class_label,
            class_labels=class_labels,
        )

        self.custom_task_folder_path = custom_task_folder_path
        self.parameters_file = parameters_file
        self.default_parameter_values = default_parameter_values
        self.output_dir = output_dir
        self.num_rows = num_rows

        self.persist_output = self.output_dir is not None

    @property
    def parameters(self):
        if self.parameters_file:
            return json.load(open(self.parameters_file))
        return {}

    @property
    def default_parameters(self):
        if self.default_parameter_values:
            return self.default_parameter_values
        return {}

    @property
    def parameters_for_fit(self):
        if self.parameters:
            return self.parameters
        return self.default_parameters

    def _validate_output_dir(self):
        """
        Validate that the output directory is not the same as the input.
        If no output dir exists, then assign a temporary directory to it.

        Returns
        -------
        DrumFitAdapter
            self

        Raises
        ------
        DrumCommonException
            Raised when the code directory is also used as the output directory.
        """
        if self.output_dir == self.custom_task_folder_path:
            raise DrumCommonException("The code directory may not be used as the output directory.")

        if not self.persist_output:
            self.output_dir = mkdtemp()

        return self

    def validate(self):
        """
        After initialization, validate and configure the inputs for fit.

        Returns
        -------
        DrumFitAdapter
            self
        """
        self._validate_output_dir()
        self._infer_class_labels_if_not_provided(
            input_filename=self.input_filename,
            target_filename=self.target_filename,
            target_name=self.target_name,
        )

        return self

    def sample_data_if_necessary(self, data):
        if self.num_rows == "ALL":
            return data

        assert isinstance(self.num_rows, (int, float))
        self.num_rows = int(self.num_rows)
        if self.num_rows > len(self.input_dataframe):
            raise DrumCommonException(
                "Requested number of rows greater than data length {} > {}".format(
                    self.num_rows, len(self.input_dataframe)
                )
            )

        return data.sample(self.num_rows, random_state=1)

    def cleanup_output_directory_if_necessary(self):
        """
        Returns
        -------
        bool
            True if output directory was cleaned up. False otherwise.
        """
        if not self.persist_output:
            shutil.rmtree(self.output_dir)
        return not self.persist_output
