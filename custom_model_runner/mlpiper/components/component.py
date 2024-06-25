import abc
from future.utils import with_metaclass
import pprint

from mlpiper.common.base import Base


class Component(with_metaclass(abc.ABCMeta, Base)):
    def __init__(self, ml_engine):
        super(Component, self).__init__(ml_engine.get_engine_logger(self.logger_name()))
        self._ml_engine = ml_engine
        self._logger.debug("Creating pipeline component: " + self.name())
        self._params = None

    def configure(self, params):
        self._logger.debug(
            "Configure component with input params, name: {}, params:\n {}".format(
                self.name(), pprint.pformat(params)
            )
        )
        self._params = params

    @abc.abstractmethod
    def materialize(self, parent_data_objs):
        pass

    @abc.abstractmethod
    def _validate_output(self, objs):
        pass

    @abc.abstractmethod
    def _post_validation(self, objs):
        pass
