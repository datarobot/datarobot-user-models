from mlpiper.common import constants
from mlpiper.common.mlpiper_exception import MLPiperException


class Base(object):
    def __init__(self, logger=None):
        super(Base, self).__init__()
        self._msg_container = []
        self.__logger = logger

    def name(self):
        return self.__class__.__name__

    def logger_name(self):
        return constants.LOGGER_NAME_PREFIX + "." + self.name()

    def set_logger(self, logger):
        self.__logger = logger
        self._print_acc_messages()

    def is_logger_set(self):
        return bool(self.__logger)

    def _print_acc_messages(self):
        if not self.__logger:
            raise MLPiperException("None logger! Invalid internal sequence!")

        if self._msg_container:
            for m in self._msg_container:
                self.__logger.info(m)

    @property
    def _logger(self):
        return self.__logger if self.__logger else self

    def debug(self, msg):
        self._msg_container.append(msg)

    def info(self, msg):
        self._msg_container.append(msg)

    def warning(self, msg):
        self._msg_container.append(msg)

    def error(self, msg):
        self._msg_container.append(msg)

    def critical(self, msg):
        self._msg_container.append(msg)
