import pickle


class Serializable(object):
    default_filename = 'artifact'

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
        with open(artifact_directory + Serializable.default_filename, 'w') as fp:
            pickle.dump(self, fp)
        return self

    @staticmethod
    def load(artifact_directory):
        """
        Deserializes the object stored within `artifact_directory`

        Returns
        -------
        object
            The deserialized object
        """
        with open(artifact_directory + Serializable.default_filename, 'w') as fp:
            return pickle.load(fp)
