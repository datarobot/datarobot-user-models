import os
import pickle
import sys

from datarobot_drum.drum.exceptions import DrumCommonException, DrumSerializationError


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
    def fit(self, X, y, row_weights=None, **kwargs):
        """
        This hook defines how DataRobot will train this task. Even transform tasks need to be
        trained to learn/store information from training data
        DataRobot runs this hook when the task is being trained inside a blueprint.
        As an output, this hook is expected to create an artifact containing a trained object,
        that is then used to transform new data.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column (None is passed for unsupervised projects).
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
