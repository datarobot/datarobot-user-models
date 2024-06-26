from mlpiper.components import ConnectableComponent

from source_additions.add import source_encode
from util.word import Word


class StringSourceSharedDir(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        str_value = self._params.get("value", "default-string-value")
        str_value = source_encode(str_value)
        self._logger.info("Word: {}".format(Word("Hello World").words))
        return [str_value]
