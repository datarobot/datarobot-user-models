from mlpiper.components import ConnectableComponent

from sink_additions.add import sink_decode
from util.word import Word


class StringSinkSharedDir(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        expected_str_value = self._params.get("expected-value", "default-string-value")
        actual_value = parent_data_objs[0]
        actual_value = sink_decode(actual_value)
        print(
            "String Sink, encoded value: {}; decoded value: {}".format(
                parent_data_objs[0], actual_value
            )
        )
        print("String Sink, Got:[{}] Expected: [{}] ".format(actual_value, expected_str_value))
        self._logger.info("Word: {}".format(Word("Hello World").words))
        if expected_str_value != actual_value:
            raise Exception("Actual [{}] != Expected [{}]".format(actual_value, expected_str_value))
        return []
