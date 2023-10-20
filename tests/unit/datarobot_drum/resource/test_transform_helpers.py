"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd
import numpy as np
import pytest
from scipy.sparse import coo_matrix

from datarobot_drum.resource.transform_helpers import (
    validate_and_convert_column_names_for_serialization,
)


@pytest.mark.parametrize(
    "columns, expected_columns",
    [
        (["a", "b", "c"], ["a", "b", "c"]),
        (
            ["newline strip\n", "    trailing    ", "replace_new\nline"],
            ["newline strip", "trailing", "replace_new\\nline"],
        ),
        (["a", "dupe", "dupe"], ["a", "dupe", "dupe"]),
        (["unicode okay ⏎"], ["unicode okay ⏎"]),
        ([" ", "first col empty should error"], None),
        ([" \n\n ", "first col empty should error"], None),
        ([" \t\t ", "first col empty should error"], None),
    ],
)
@pytest.mark.parametrize("sparse", [True, False])
def test_validate_and_convert_column_names_for_serialization(columns, expected_columns, sparse):
    num_columns = len(columns)
    if sparse:
        df = pd.DataFrame.sparse.from_spmatrix(coo_matrix(np.zeros([10, num_columns])))
    else:
        df = pd.DataFrame(np.zeros([10, num_columns]))

    df.columns = columns
    if not expected_columns:
        with pytest.raises(ValueError):
            validate_and_convert_column_names_for_serialization(df)
    else:
        output_df = validate_and_convert_column_names_for_serialization(df)
        assert list(output_df.columns.values) == expected_columns
