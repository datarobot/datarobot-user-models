"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
from io import BytesIO
from typing import Union

import pandas as pd

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

def is_sparse_dataframe(dataframe: pd.DataFrame) -> bool:
    return hasattr(dataframe, "sparse")


def is_sparse_series(series: pd.Series) -> bool:
    return hasattr(series, "sparse")


def read_csv(csv: Union[str, BytesIO]) -> pd.DataFrame:
    """Handle reading csv files even if they have a single str/cat column with missing values.
    Those values will show up as empty lines.  Note that equivalent code is needed in the R predictor."""
    try:
        df = pd.read_csv(csv)
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
        df = pd.read_csv(csv, skip_blank_lines=False)

    return df
