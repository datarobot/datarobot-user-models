class Serializable(object):
    default_filename = 'task'

    def save(self, dir):
        """
        Serializes the object.
        """
        with open(dir + CustomTask.default_filename, 'w') as fp:
            pickle.dump(self, fp)

    def load(self, dir):
        """
        Deserializes the object.
        Returns
        -------
        Self
        """
        with open(dir + CustomTask.default_filename, 'w') as fp:
            self.__set_state__(pickle.load(fp))
            return self