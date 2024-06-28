import abc
from future.utils import with_metaclass
import threading

from mlpiper.common.base import Base
from mlpiper.common.mlpiper_exception import MLPiperException


class BgActor(with_metaclass(abc.ABCMeta, Base, threading.Thread)):
    def __init__(self, mlops, ml_engine, polling_interval_sec=10.0):
        super(BgActor, self).__init__()
        self.set_logger(ml_engine.get_engine_logger(self.logger_name()))

        if not mlops or not mlops.init_called:
            raise MLPiperException("'mlops' was not setup properly!")

        self._mlops = mlops
        self._polling_interval_sec = polling_interval_sec

        self._condition = threading.Condition()
        self._stop_gracefully = False

    def run(self):
        while True:
            with self._condition:
                self._condition.wait(self._polling_interval_sec)
                if self._mlops.done_called or self._stop_gracefully:
                    break

            self._do_repetitive_work()

        self._logger.warning("Exiting background actor ...")

    def stop_gracefully(self):
        with self._condition:
            self._finalize()
            self._stop_gracefully = True
            self._condition.notify_all()

    @abc.abstractmethod
    def _do_repetitive_work(self):
        """
        Implement any desired repetitive functionality that will be called in a background thread
        every 'polling_interval_sec'
        """
        pass

    def _finalize(self):
        """
        An overridable method, to let the derived class perform final actions before shutting down
        """
        pass
