"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import tempfile

import pandas as pd
import numpy as np
import pyarrow
import pytest
from pandas._testing import assert_frame_equal

from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils


class TestStructuredInputReadUtils(object):
    def test_read_structured_input_file_as_df_df_treats_missing_lines_as_nans(self):
        """
        Shared fit preprocessing should interpret a blank row as nan if the CSV contains a single column
        """
        # row 5 should be nan, and should not count the last line as a nan (totalling 7 rows)
        single_col_csv_data = "data\n0\n1\n2\n3\n4\n\n6\n"

        tmp_file = tempfile.NamedTemporaryFile(suffix=".csv")
        with open(tmp_file.name, "w") as f:
            f.write(single_col_csv_data)

        X = StructuredInputReadUtils.read_structured_input_file_as_df(tmp_file.name)
        tmp_file.close()

        assert np.isnan(X["data"][5])
        assert X.shape[0] == 7

    def test_read_structured_input_arrow_csv_na_consistency(self, tmp_path):
        """
        Test that N/A values (None, numpy.nan) are handled consistently when using
        CSV vs Arrow as a prediction payload format.
        1. Make CSV and Arrow prediction payloads from the same dataframe
        2. Read both payloads
        3. Assert the resulting dataframes are equal
        """

        # arrange
        df = pd.DataFrame({"col_int": [1, np.nan, None], "col_obj": ["a", np.nan, None]})

        csv_filename = os.path.join(tmp_path, "X.csv")
        with open(csv_filename, "w") as f:
            f.write(df.to_csv(index=False))

        arrow_filename = os.path.join(tmp_path, "X.arrow")
        with open(arrow_filename, "wb") as f:
            f.write(pyarrow.ipc.serialize_pandas(df).to_pybytes())

        # act
        csv_df = StructuredInputReadUtils.read_structured_input_file_as_df(csv_filename)
        arrow_df = StructuredInputReadUtils.read_structured_input_file_as_df(arrow_filename)

        # assert
        is_nan = lambda x: isinstance(x, float) and np.isnan(x)
        is_none = lambda x: x is None

        assert_frame_equal(csv_df, arrow_df)
        # `assert_frame_equal` doesn't make a difference between None and np.nan.
        # To do an exact comparison, compare None and np.nan "masks".
        assert_frame_equal(csv_df.applymap(is_nan), arrow_df.applymap(is_nan))
        assert_frame_equal(csv_df.applymap(is_none), arrow_df.applymap(is_none))

    def test_read_structured_input_csv_unicode_error_handled(self):
        data = "a\nb\nc\nd\ne\nf\n"
        tmp_file = tempfile.NamedTemporaryFile(suffix=".csv")
        with open(tmp_file.name, "w", encoding="utf-16") as f:
            f.write(data)
        with pytest.raises(
            DrumCommonException, match="Supplied CSV input file encoding must be UTF-8."
        ):
            StructuredInputReadUtils.read_structured_input_file_as_df(tmp_file.name)
        tmp_file.close()
