"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import tempfile

import numpy as np
import pytest

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
