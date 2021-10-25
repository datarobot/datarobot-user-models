"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from pathlib import Path


def test_data() -> Path:
    top_dir = Path(__file__).parent.parent
    return top_dir / "testdata"
