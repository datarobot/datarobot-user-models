import os
import pickle


class Serializable(object):
    default_artifact_filename = "artifact.pkl"

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

        with open(os.path.join(artifact_directory, Serializable.default_artifact_filename), "wb") as fp:
            pickle.dump(self, fp)
        return self

    @classmethod
    def load(cls, artifact_directory):
        """
        Deserializes the object stored within `artifact_directory`

        Returns
        -------
        cls
            The deserialized object
        """
        with open(os.path.join(artifact_directory, Serializable.default_artifact_filename), "rb") as fp:
            deserialized_object = pickle.load(fp)

        if not isinstance(deserialized_object, cls):
            raise ValueError("load method must return a {} class".format(cls.__name__))
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
