class Word(object):
    def __init__(self, str):
        self._words = str.split()

    @property
    def words(self):
        return self._words
