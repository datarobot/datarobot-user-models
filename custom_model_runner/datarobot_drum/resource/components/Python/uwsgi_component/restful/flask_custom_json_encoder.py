"""
For internal use only. It is a custom json encoder, which enables to serialize
additional types to json, beyond those that are natively supported.
"""
from flask.json import JSONEncoder
import numpy as np


class FlaskCustomJsonEncode(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)

        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, obj)
