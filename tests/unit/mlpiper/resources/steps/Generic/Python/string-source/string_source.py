from mlpiper.components import ConnectableComponent


class StringSource(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        str_value = self._params.get("value", "default-string-value")
        return [str_value]
