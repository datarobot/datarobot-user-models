import pickle


class Serializable(object):
    default_filename = "artifact"

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

        with open("{}/{}".format(artifact_directory, Serializable.default_filename), "wb") as fp:
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
        with open("{}/{}".format(artifact_directory, Serializable.default_filename), "rb") as fp:
            deserialized_object = pickle.load(fp)

        if not isinstance(deserialized_object, cls):
            raise ValueError("load method must return a {} class".format(cls.__name__))
        return deserialized_object
