import logging

from mlpiper.ml_engine.ml_engine import MLEngine


class PythonEngine(MLEngine):
    """
    Implementing the MLEngine API for a python engine.
    """

    def __init__(self, pipeline, mlpiper_jar=None, standalone=False):
        super(PythonEngine, self).__init__(pipeline, standalone)
        self._config = {"mlpiper_jar": mlpiper_jar}

        self.set_logger(self.get_engine_logger(self.logger_name()))

    def finalize(self):
        pass

    def cleanup(self):
        pass

    def get_engine_logger(self, name):
        return logging.getLogger(name)

    def _session(self):
        pass

    def _context(self):
        pass
