from mlpiper.components.connectable_component import ConnectableComponent


class DfToCsv(ConnectableComponent):
    def __init__(self, engine):
        super(DfToCsv, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        output_filename = self._params.get("output_filename")
        df = parent_data_objs[0]
        df.to_csv(output_filename, index=False)
        return []
