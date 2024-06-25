import abc
from future.utils import with_metaclass

from mlpiper.common.base import Base
from mlpiper.pipeline import json_fields


class MLEngine(with_metaclass(abc.ABCMeta, Base)):
    """
    An abstract class, which defines the interface of any implemented engine.
     An example for such engines are PySpark, Tensorflow and so on.
    """

    def __init__(self, pipeline, standalone):
        super(MLEngine, self).__init__()
        self._pipeline = pipeline
        self._standalone = standalone
        self._user_data = {}
        self._config = {}
        self._uuid = None

    @property
    def pipeline_name(self):
        return self._pipeline[json_fields.PIPELINE_NAME_FIELD]

    @property
    def standalone(self):
        return self._standalone

    def set_standalone(self, set_standalone):
        self._standalone = set_standalone
        return self

    @property
    def config(self):
        return self._config

    @property
    def user_data(self):
        return self._user_data

    @property
    def session(self):
        """
        Returns a session, which represents a single execution. It is totally dependent
        on to the engine to determine the meaning of it.

        :return:  engine's session
        """
        return self._session()

    @property
    def context(self):
        """
        Returns a context which is relevant to the given engine. (.e.g in the case of
        spark it is the spark context)
        :return:  engine's context
        """
        return self._context()

    def run(self, mlops, pipeline):
        """
        The given engine can safely run after all initializations have been completed
        with success.
        """
        pass

    @abc.abstractmethod
    def finalize(self):
        """
        Will be called only after the component's materialize functions were all called.
        It supposed to perform any engine specific and final code to actually drive the pipeline.
        """
        pass

    def stop(self):
        """
        Will be called after the pipeline's execution is completed and before the mlops is shutdown
        """
        pass

    @abc.abstractmethod
    def cleanup(self):
        """
        Will be called to cleanup remainders after the pipeline's execution is completed,
        either by success or failure
        """
        pass

    @abc.abstractmethod
    def get_engine_logger(self, name):
        """
        Returns an engine's specific logger. The logger can be accessed by the given engine
        components

        :param name:  the logger name
        :return:  engine's logger
        """
        pass

    @abc.abstractmethod
    def _session(self):
        pass

    @abc.abstractmethod
    def _context(self):
        pass

    def set_uuid(self, uuid):
        self._uuid = uuid

    def get_uuid(self):
        return self._uuid
