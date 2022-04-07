"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import io
import logging
import os

import numpy as np
import pandas as pd
from scipy.io import mmread

from datarobot_drum.drum.common import get_pyarrow_module
from datarobot_drum.drum.enum import (
    InputFormatToMimetype,
    PredictionServerMimetypes,
    LOGGER_NAME_PREFIX,
)
from datarobot_drum.drum.exceptions import DrumCommonException


logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class StructuredInputReadUtils:
    @staticmethod
    def read_structured_input_file_as_binary(filename):
        mimetype = StructuredInputReadUtils.resolve_mimetype_by_filename(filename)
        with open(filename, "rb") as file:
            binary_data = file.read()
        return binary_data, mimetype

    @staticmethod
    def read_structured_input_file_as_df(filename, sparse_column_file=None):
        binary_data, mimetype = StructuredInputReadUtils.read_structured_input_file_as_binary(
            filename
        )
        sparse_colnames = None
        if sparse_column_file:
            sparse_colnames = StructuredInputReadUtils.read_sparse_column_file_as_list(
                sparse_column_file
            )
        return StructuredInputReadUtils.read_structured_input_data_as_df(
            binary_data, mimetype, sparse_colnames
        )

    @staticmethod
    def resolve_mimetype_by_filename(filename):
        return InputFormatToMimetype.get(os.path.splitext(filename)[1])

    @staticmethod
    def read_sparse_column_data_as_list(sparse_column_data):
        sparse_columns_raw = io.BytesIO(sparse_column_data).readlines()
        return [column.strip().decode("utf-8") for column in sparse_columns_raw]

    @staticmethod
    def read_sparse_column_file_as_list(sparse_column_file):
        with open(sparse_column_file, "rb") as f:
            return StructuredInputReadUtils.read_sparse_column_data_as_list(f.read())

    @staticmethod
    def read_structured_input_data_as_df(binary_data, mimetype, sparse_colnames=None):
        try:
            if mimetype == PredictionServerMimetypes.TEXT_MTX:
                return pd.DataFrame.sparse.from_spmatrix(
                    mmread(io.BytesIO(binary_data)), columns=sparse_colnames
                )
            elif mimetype == PredictionServerMimetypes.APPLICATION_X_APACHE_ARROW_STREAM:
                df = get_pyarrow_module().ipc.deserialize_pandas(binary_data)

                # After CSV serialization+deserialization,
                # original dataframe's None and np.nan values
                # become np.nan values.
                # After Arrow serialization+deserialization,
                # original dataframe's None and np.nan values
                # become np.nan for numeric columns and None for 'object' columns.
                #
                # Since we are supporting both CSV and Arrow,
                # to be consistent with CSV serialization/deserialization,
                # it is required to replace all None with np.nan for Arrow.
                df.fillna(value=np.nan, inplace=True)

                return df
            else:  # CSV format
                try:
                    df = pd.read_csv(io.BytesIO(binary_data))
                except UnicodeDecodeError:
                    logger.error(
                        "A non UTF-8 encoding was encountered while opening the data.\nSave this using utf-8 encoding, for example with pandas to_csv('filename.csv', encoding='utf-8')."
                    )
                    raise DrumCommonException("Supplied CSV input file encoding must be UTF-8.")
                # If the DataFrame only contains a single column, treat blank lines as NANs
                if df.shape[1] == 1:
                    logger.info(
                        "Input data only contains a single column, treating blank lines as NaNs"
                    )
                    df = pd.read_csv(io.BytesIO(binary_data), skip_blank_lines=False)

                return df

        except pd.errors.ParserError as e:
            raise DrumCommonException(
                "Pandas failed to read input binary data {}".format(binary_data)
            )
