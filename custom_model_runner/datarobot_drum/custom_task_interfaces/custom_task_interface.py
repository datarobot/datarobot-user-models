#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import json
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from datarobot_drum.drum.exceptions import DrumSerializationError


class Serializable(object):
    default_artifact_filename = "drum_artifact.pkl"

    def save(self, artifact_directory):
        """
        Serializes the object and stores it in `artifact_directory`

        Parameters
        ----------
        artifact_directory: str
            Path to the directory to save the serialized artifact(s) to.

        Returns
        -------
        self
        """

        self.save_task(artifact_directory)

        # For use in easy chaining, e.g. CustomTask().fit().save().load()
        return self

    def save_task(self, artifact_directory, exclude=None):
        """
        Helper function that abstracts away pickling the CustomTask object. It also can
        automatically set previously serialized variables to None, e.g. when using keras you likely
        want to serialize self.estimator using model.save() or keras.models.save_model() and then
        pass in exclude='estimator'

        Parameters
        ----------
        artifact_directory: str
            Path to the directory to save the serialized artifact(s) to.
        exclude: List[str]
            Variables on the CustomTask object we want to exclude from serialization by setting to None

        Returns
        -------
        None
        """
        # If any custom task variables are excluded in the pickle, temporarily store them here, set them to None, then
        # restore them back onto the class after serialization
        variables_to_restore = {}

        if exclude:
            for custom_task_variable in exclude:
                try:
                    # Ensure the variable actually exists in the custom task
                    variables_to_restore[custom_task_variable] = getattr(self, custom_task_variable)
                except AttributeError:
                    raise DrumSerializationError(
                        f"The object named {custom_task_variable} passed in exclude= was not found"
                    )

                # Set it to None so it does not get serialized
                setattr(self, custom_task_variable, None)
        with open(
            os.path.join(artifact_directory, Serializable.default_artifact_filename), "wb"
        ) as fp:
            pickle.dump(self, fp)

        for custom_task_variable, value in variables_to_restore.items():
            setattr(self, custom_task_variable, value)

    @classmethod
    def load(cls, artifact_directory):
        """
        Deserializes the object stored within `artifact_directory`.

        Parameters
        ----------
        artifact_directory: str
            Path to the directory to save the serialized artifact(s) to.

        Returns
        -------
        cls
            The deserialized object
        """

        return cls.load_task(artifact_directory)

    @classmethod
    def load_task(cls, artifact_directory):
        """
        Helper method to abstract deserializing the pickle object stored within `artifact_directory` and
        returning the custom task. Any variables that were excluded in `save_task` must be manually loaded
        proceeding this function.

        Parameters
        ----------
        artifact_directory: str
            Path to the directory to save the serialized artifact(s) to.

        Returns
        -------
        cls
            The deserialized object
        """
        with open(
            os.path.join(artifact_directory, Serializable.default_artifact_filename), "rb"
        ) as fp:
            deserialized_object = pickle.load(fp)

        if not isinstance(deserialized_object, cls):
            raise DrumSerializationError(
                "load_task method must return a {} class".format(cls.__name__)
            )
        return deserialized_object


class CustomTaskInterface(Serializable):
    _secrets: Optional[Dict[str, Any]] = None

    @property
    def secrets(self) -> Dict[str, Any]:
        if self._secrets:
            return self._secrets
        return {}

    @secrets.setter
    def secrets(self, secrets: Optional[Dict[str, Any]]):
        self._secrets = secrets

    def fit(self, X, y, parameters=None, row_weights=None, **kwargs):
        """
        This hook defines how DataRobot will train this task. Even transform tasks need to be
        trained to learn/store information from training data
        DataRobot runs this hook when the task is being trained inside a blueprint.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column (None is passed for unsupervised projects).
        parameters: dict (optional, default = None)
            A dictionary of hyperparameters defined in the model-metadata.yaml file for the task.
        row_weights: np.ndarray (optional, default = None)
            A list of weights. DataRobot passes it in case of smart downsampling or when weights
            column is specified in project settings.
        Returns
        -------
        EstimatorInterface
        """
        raise NotImplementedError()

    @staticmethod
    def log_message(message):
        """Prints the message to the logs and then flushes the buffer."""
        print(message)
        sys.stdout.flush()


@contextmanager
def secrets_injection_context(
    interface: CustomTaskInterface, mount_path: Optional[str], env_var_prefix: Optional[str]
):
    interface.secrets = load_secrets(mount_path=mount_path, env_var_prefix=env_var_prefix)
    try:
        yield
    finally:
        interface.secrets = None


def load_secrets(mount_path: Optional[str], env_var_prefix: Optional[str]) -> Dict[str, Any]:
    all_secrets = {}
    env_secrets = _get_environment_secrets(env_var_prefix)
    mounted_secrets = _get_mounted_secrets(mount_path)
    all_secrets.update(env_secrets)
    all_secrets.update(mounted_secrets)
    return all_secrets


def _get_environment_secrets(env_var_prefix):
    if not env_var_prefix:
        return {}

    full_prefix = f"{env_var_prefix}_"
    actual_secrets = [(k, v) for k, v in os.environ.items() if k.startswith(full_prefix)]

    return {key.replace(full_prefix, ""): json.loads(value) for key, value in actual_secrets}


def _get_mounted_secrets(mount_path: str):
    if mount_path is None:
        return {}

    secret_files = [file_path for file_path in Path(mount_path).glob("*") if file_path.is_file()]

    def get_dict(file_path: Path):
        with file_path.open() as fp:
            return json.load(fp)

    return {file_path.name: get_dict(file_path) for file_path in secret_files}
