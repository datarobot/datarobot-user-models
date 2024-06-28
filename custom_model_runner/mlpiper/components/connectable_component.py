import abc

from mlpiper.components.component import Component


class ConnectableComponent(Component):
    def __init__(self, engine):
        super(ConnectableComponent, self).__init__(engine)

    def materialize(self, parent_data_objs):
        return self._materialize(parent_data_objs, self._ml_engine.user_data)

    def _validate_output(self, objs):
        pass

    def _post_validation(self, objs):
        pass

    @abc.abstractmethod
    def _materialize(self, parent_data_objs, user_data):
        """
        This abstract method is supposed to return a list of any desired python object(s),
        or nothing
        """
        pass

    def terminate(self):
        """Allow a component to handle termination upon pipeline completion"""
        pass
