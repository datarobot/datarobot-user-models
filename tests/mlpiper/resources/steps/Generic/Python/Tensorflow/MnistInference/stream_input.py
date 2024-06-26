import abc
from future.utils import with_metaclass
from random import randint


class StreamInput(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, total_records, stop_at_record=-1, random=False):
        self._total_records = total_records
        self._stop_at_record = stop_at_record
        self._random = random
        self._records_returned = 0

    def get_next_input_index(self):
        if self._stop_at_record >= 0 and self._records_returned >= self._stop_at_record:
            return -1

        if self._random:
            next_index = randint(0, self._total_records)
        else:
            next_index = self._records_returned % self._total_records
        self._records_returned += 1
        return next_index

    @abc.abstractmethod
    def get_next_input(self):
        pass

    def __del__(self):
        pass
