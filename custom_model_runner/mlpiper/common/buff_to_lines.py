class BufferToLines(object):
    def __init__(self):
        self._acc_buff = ""
        self._last_line = ""
        self._in_middle_of_line = False

    def add(self, buff):
        self._acc_buff += buff.decode()
        self._in_middle_of_line = False if self._acc_buff[-1] == "\n" else True

    def lines(self):
        lines = self._acc_buff.split("\n")
        up_to_index = len(lines) - 2 if self._in_middle_of_line else len(lines) - 1
        self._acc_buff = lines[-1] if self._in_middle_of_line else ""

        for iii in range(up_to_index):
            yield lines[iii]
