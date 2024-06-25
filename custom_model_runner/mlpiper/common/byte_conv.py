class ByteConv(object):
    KB_UNIT = 1024.0

    def __init__(self, size_bytes):
        self._mem_size_bytes = size_bytes

    @classmethod
    def from_bytes(cls, size_bytes):
        return cls(size_bytes)

    @classmethod
    def from_kbytes(cls, size_kbytes):
        return cls(size_kbytes * cls.KB_UNIT)

    @classmethod
    def from_mbytes(cls, size_mbytes):
        return cls(size_mbytes * cls.KB_UNIT * cls.KB_UNIT)

    @classmethod
    def from_gbytes(cls, size_gbytes):
        return cls(size_gbytes * cls.KB_UNIT * cls.KB_UNIT * cls.KB_UNIT)

    @property
    def bytes(self):
        return self._mem_size_bytes

    @property
    def kbytes(self):
        return self._mem_size_bytes / ByteConv.KB_UNIT

    @property
    def mbytes(self):
        return self._mem_size_bytes / ByteConv.KB_UNIT / ByteConv.KB_UNIT

    @property
    def gbytes(self):
        return (
            self._mem_size_bytes
            / ByteConv.KB_UNIT
            / ByteConv.KB_UNIT
            / ByteConv.KB_UNIT
        )
