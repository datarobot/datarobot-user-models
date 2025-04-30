# Copyright 2025 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.

# -*- coding: utf-8 -*-
import json
import sys
import traceback
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import cast

from IPython.core.formatters import BaseFormatter
from IPython.core.magic import Magics
from pydantic import BaseModel
from traitlets import ObjectName
from traitlets import Unicode

is_pandas_loaded = True

try:
    from pandas import DataFrame
    from pandas import io
except ImportError:
    is_pandas_loaded = False


class Entity(BaseModel):
    """
    Base class for data transfer objects
    """

    class Config:
        allow_population_by_field_name = True


class DataframePaginationAttributes(Entity):
    limit: int
    offset: int


class DataframeAggregationParams(Entity):
    group_by: str
    aggregate_by: str
    aggregation_func: str


class DataframeFilterParams(Entity):
    filter_by: Optional[str]
    filter: str


class DataframesProcessSteps(str, Enum):
    CHART_CELL_DATAFRAME = "chart_cell_dataframe"
    AGGREGATION = "aggregation"
    PAGINATION = "pagination"
    SORTING = "sorting"
    GET_COLUMNS = "get_columns"
    DEFAULT = "get_columns"


Columns = List[Dict[str, Any]]

DEFAULT_INDEX_KEY = "index"


def _register_exception(
    e: Exception,
    step: str,
) -> Dict[str, Any]:
    exc_info = sys.exc_info()
    traceback_msg = traceback.format_exception(*exc_info)

    return {
        "step": step,
        "message": str(e),
        "traceback": traceback_msg,
    }


def _validate_columns(data: DataFrame) -> None:
    """To prevent failing some DataFrame process steps like columns extraction
    and converting to json we need ensure that columns dtypes can be converted

    Args:
        data (DataFrame): in-memory DataFrame

    Returns:
        None
    """
    convertable_types = [
        "int64",
        "float64",
        "float32",
        "bool",
        "category",
        "geometry",
        "object",
        "datetime64[ns]",
        "timedelta[ns]",
    ]
    for column in data.columns:
        dtype = data[column].dtype
        if dtype not in convertable_types:
            # Try to keep datetime dtype, remove the timezone information
            # but converting to UTC, so yielding naive UTC time
            if hasattr(data[column], "dt") and hasattr(data[column].dt, "tz_convert"):
                data[column] = data[column].dt.tz_convert(None)
            else:
                # Otherwise, keep going working with dataframe but set pandas column type to str
                data[column] = data[column].astype(str)


def _get_dataframe_columns(df: DataFrame) -> Columns:
    schema = io.json.build_table_schema(df)
    columns = cast(Columns, schema["fields"])
    return columns


# DataFrame pagination if pagination attrs exist
def _paginate_dataframe(df: DataFrame, pagination: DataframePaginationAttributes) -> DataFrame:
    start_row = pagination.offset
    end_row = start_row + pagination.limit
    return df[start_row:end_row]


def _sort_dataframe(df: DataFrame, sort_by: str) -> DataFrame:
    sorting_list = sort_by.split(",")
    sort_by_list = []
    ascending_list = []
    # sorting by default index (None or "index") raise KeyValue error
    # pandas allow to sort by columns and explicit indices
    allowed_fields = set(list(df.columns) + list(df.index.names))
    for sort_key in sorting_list:
        if sort_key not in allowed_fields:
            continue
        sort_by_list.append(sort_key.lstrip("-"))
        ascending_list.append(not sort_key.startswith("-"))

    return df.sort_values(by=sort_by_list, ascending=ascending_list, ignore_index=False)


def _aggregate_dataframe(
    df: DataFrame, aggregation_params: DataframeAggregationParams
) -> DataFrame:
    aggregated = df.groupby(aggregation_params.group_by).aggregate(
        {f"{aggregation_params.aggregate_by}": aggregation_params.aggregation_func}
    )
    return aggregated.reset_index()


def _transform_to_json(data: DataFrame) -> Any:
    if isinstance(data, list):
        return data

    if data.__class__.__name__ == "GeoDataFrame":
        return json.loads(data.to_json())["features"]
    return json.loads(data.to_json(orient="table", index=True, default_handler=str))["data"]


def _prepare_df_for_chart_cell(val: DataFrame, columns: List[str]) -> Union[DataFrame, List[str]]:
    if len(columns) == 0:
        data = []
    elif len(columns) == 1:
        # Return counts if only one column was selected or selected count of records
        data = val.groupby(columns)[columns[0]].count().reset_index(name="count").set_index("count")
    else:
        # Return only selected columns
        data = val[columns]

    return data


# This formatter can operate with data that we have received as a DataFrame
def formatter(  # noqa: C901,PLR0912
    val: "DataFrame",
    formatter: Optional[Callable[..., List[str]]] = None,
    **formatter_kwargs: Any,
) -> Dict[str, Any]:
    error = []
    dataframe_limit = 5000
    dataframe_id = id(val)
    pagination = DataframePaginationAttributes(limit=10, offset=0)
    data = val
    sort_by = ""
    selected_columns = []
    _validate_columns(data)
    try:
        columns = _get_dataframe_columns(data)
    except Exception as e:
        error.append(_register_exception(e, DataframesProcessSteps.GET_COLUMNS.value))

    index_key = data.index.name if data.index.name is not None else DEFAULT_INDEX_KEY

    # check if it's a dataframe for ChartCell then return full dataframe
    if hasattr(val, "attrs") and "returnAll" in val.attrs and val.attrs["returnAll"]:
        # Validate what to return to UI
        if hasattr(val, "attrs") and "selected_columns" in val.attrs:
            selected_columns = list(
                filter(lambda item: item is not index_key, val.attrs["selected_columns"])
            )
            try:
                data = _prepare_df_for_chart_cell(val=data, columns=selected_columns)
            except Exception as e:
                error.append(
                    _register_exception(e, DataframesProcessSteps.CHART_CELL_DATAFRAME.value)
                )
            if len(selected_columns) < 2:
                # Reset `returnAll` attribute to prevent returning a whole DF on next formatter call
                val.attrs.update({"returnAll": False})
                data = [] if len(error) > 0 else data

                return {
                    "columns": columns,
                    "data": _transform_to_json(data),
                    "referenceId": dataframe_id,
                    "error": error,
                    "indexKey": index_key,
                }

        aggregation_func = val.attrs.get("aggregation", {}).get("aggregation_func")
        if aggregation_func and aggregation_func != "no-aggregation":
            aggregation = DataframeAggregationParams(
                group_by=val.attrs["aggregation"]["group_by"],
                aggregate_by=val.attrs["aggregation"]["aggregate_by"],
                aggregation_func=val.attrs["aggregation"]["aggregation_func"],
            )
            try:
                data = _aggregate_dataframe(data, aggregation)
            except Exception as e:
                error.append(_register_exception(e, DataframesProcessSteps.AGGREGATION.value))

        if len(data.index) >= dataframe_limit:
            pagination = DataframePaginationAttributes(limit=dataframe_limit, offset=0)
            try:
                data = _paginate_dataframe(data, pagination)
            except Exception as e:
                error.append(_register_exception(e, DataframesProcessSteps.PAGINATION.value))

        # Reset `returnAll` attribute to prevent returning a whole DF on next formatter call
        val.attrs.update({"returnAll": False})

        return {
            "columns": columns,
            "data": _transform_to_json(data),
            "referenceId": dataframe_id,
            "error": error,
            "indexKey": index_key,
        }

    # Sorting step, gets attrs that have been set up in DataframeProcessor
    if hasattr(val, "attrs") and "sort_by" in val.attrs:
        try:
            data = _sort_dataframe(df=data, sort_by=val.attrs["sort_by"])
            sort_by = val.attrs["sort_by"]
        except Exception as e:
            error.append(_register_exception(e, DataframesProcessSteps.SORTING.value))

    # Pagination step, gets attrs that have been set up in DataframeProcessor
    if hasattr(val, "attrs") and "pagination" in val.attrs:
        pagination = DataframePaginationAttributes(
            limit=val.attrs["pagination"]["limit"], offset=val.attrs["pagination"]["offset"]
        )

    # If dataframe length is less than pagination limit no need to paginate it
    if len(data.index) > int(pagination.limit):
        try:
            data = _paginate_dataframe(data, pagination)
        except Exception as e:
            error.append(_register_exception(e, DataframesProcessSteps.PAGINATION.value))

    return {
        "data": _transform_to_json(data),
        "columns": columns,
        "count": len(data.index),
        "totalCount": len(val.index),
        "offset": int(pagination.offset),
        "limit": int(pagination.limit),
        "referenceId": dataframe_id,
        "sortedBy": sort_by,
        "indexKey": index_key,
        "error": error,
    }


# To Add a new data formatter we need create a new class instance based on a
# BaseFormatter from the iPython kernel
#
# Ignoring mypy error: Class cannot subclass "BaseFormatter" (has type "Any")
class DataFrameFormatter(BaseFormatter):  # type: ignore[misc]
    """A DataFrame formatter. This is basically a copy of the JSONFormatter,
    so it will return as a new mime type: application/vnd.dataframe+json in output.
    """

    format_type = Unicode("application/vnd.dataframe+json")
    _return_type = (list, dict)

    print_method = ObjectName("_repr_json_")

    def _check_return(self, r: Any, obj: Any) -> Any:
        """Check that a return value is appropriate
        Return the value if so, None otherwise, warning if invalid.
        """
        if r is None:
            return
        md = None
        if isinstance(r, tuple):
            # unpack data, metadata tuple for type checking on first element
            r, md = r

        assert not isinstance(r, str), "JSON-as-string has been deprecated since IPython < 3"

        if md is not None:
            # put the tuple back together
            r = (r, md)
        return super(DataFrameFormatter, self)._check_return(r, obj)


# Load our extension into ipython kernel
def load_ipython_extension(ipython: Magics) -> None:
    if is_pandas_loaded:
        dataframe_json_formatter = DataFrameFormatter()
        ipython.display_formatter.formatters[
            "application/vnd.dataframe+json"
        ] = dataframe_json_formatter
        dataframe_json_formatter.for_type(DataFrame, formatter)

        print("Pandas DataFrame MimeType Extension loaded")
    else:
        print("Please execute `pip install pandas` to use DataFrame extension")
