"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd
import numpy as np
import pytest
import types
from scipy.sparse import coo_matrix

from datarobot_drum.drum.root_predictors.transform_helpers import (
    validate_and_convert_column_names_for_serialization,
    parse_multi_part_response,
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


def test_parse_multi_part_response():
    file_content1 = b"value1"
    file_content2 = b"value2"
    boundary = "boundary123"
    content = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="X.format"\r\n\r\n'
            "csv\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="key1"; filename="file1.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode("utf-8")
        + file_content1
        + (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="key2"; filename="file2.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode("utf-8")
        + file_content2
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )

    response = types.SimpleNamespace()
    response.content = content
    response.headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    result = parse_multi_part_response(response)
    assert result["X.format"] == "csv"
    assert result["key1"] == file_content1
    assert result["key2"] == file_content2
