"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import json
import tempfile
from contextlib import ContextDecorator
from pathlib import Path


def get_test_data() -> Path:
    top_dir = Path(__file__).parent.parent
    return top_dir / "testdata"


class SimpleCache(ContextDecorator):
    def __init__(self, init_dict_template):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self._cache_filepath = f.name
        self._init_dict_template = init_dict_template

    def __enter__(self):
        with open(self._cache_filepath, "w") as f:
            json.dump(self._init_dict_template, f)
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        Path(self._cache_filepath).unlink()

    def read_cache(self):
        with open(self._cache_filepath) as f:
            return json.load(f)

    def save_cache(self, data):
        with open(self._cache_filepath, "w") as f:
            json.dump(data, f)

    def inc_value(self, key, value=1):
        cache = self.read_cache()
        cache[key] += value
        self.save_cache(cache)


def test_simple_cache():
    cache_data = {"a": 0, "b": 0}
    with SimpleCache(cache_data) as cache:
        cache.inc_value("a")
        cache.inc_value("a", value=2)
        cache.inc_value("b")

        data = cache.read_cache()

    assert data == {"a": 3, "b": 1}
