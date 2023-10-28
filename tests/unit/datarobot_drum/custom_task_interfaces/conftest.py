#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import contextlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from unittest.mock import patch

import pytest


@pytest.fixture
def mounted_secrets_factory():
    with TemporaryDirectory(suffix="-secrets") as dir_name:

        def inner(secrets_dict: Dict[str, dict]):
            top_dir = Path(dir_name)
            for k, v in secrets_dict.items():
                target = top_dir / k
                with target.open("w") as fp:
                    json.dump(v, fp)
            return dir_name

        yield inner


@pytest.fixture
def env_patcher():
    @contextlib.contextmanager
    def inner(prefix, secrets_dict):
        env_dict = {f"{prefix}_{key}": json.dumps(value) for key, value in secrets_dict.items()}
        with patch.dict(os.environ, env_dict):
            yield

    return inner
