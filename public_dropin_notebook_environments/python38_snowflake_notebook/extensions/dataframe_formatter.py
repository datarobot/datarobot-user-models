# Copyright 2022 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.

# -*- coding: utf-8 -*-
import json
import typing
from pydantic import BaseModel
from traitlets import Unicode, ObjectName
from IPython.core.formatters import BaseFormatter


is_pandas_loaded = True

try:
    from pandas import DataFrame, io
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
    filter_by: typing.Optional[str]
    filter: str


def _get_dataframe_columns(df: DataFrame) -> list[dict[str, typing.Any]]:
    schema = io.json.build_table_schema(df)
    return schema["fields"]


# DataFrame pagination if pagination attrs exist
def _paginate_dataframe(df: DataFrame, pagination: DataframePaginationAttributes) -> DataFrame:
    start_row = pagination.offset
    end_row = start_row + pagination.limit
    try:
        paginated = df[start_row:end_row]
    except Exception as e:
        print(e)
        paginated = df
    return paginated


def _sort_dataframe(df: DataFrame, sort_by: str) -> DataFrame:
    sorting_list = sort_by.split(",")
    sort_by_list = []
    ascending_list = []
    for sort_key in sorting_list:
        sort_by_list.append(sort_key.lstrip("-"))
        ascending_list.append(not sort_key.startswith("-"))
    try:
        sorted_df = df.sort_values(by=sort_by_list, ascending=ascending_list, ignore_index=True)
    except Exception as e:
        print(e)
        sorted_df = df
    return sorted_df


def _aggregate_dataframe(
    df: DataFrame, aggregation_params: DataframeAggregationParams
) -> DataFrame:
    try:
        aggregated = df.groupby(aggregation_params.group_by).aggregate(
            {f"{aggregation_params.aggregate_by}": aggregation_params.aggregation_func}
        )
        return aggregated.reset_index()
    # skip aggregation if params not defined or something wrong
    except Exception as e:
        # print exception, it will be in output for chart cell
        print(e)
        return df


def _transform_to_json(data: DataFrame):
    return json.loads(data.to_json(orient="table", index=True))["data"]


# This formatter can operate with a data that we are received as a DataFrame
def formatter(
    val: "DataFrame", formatter: typing.Callable[..., list[str]] = None, **formatter_kwargs
):
    dataframe_limit = 5000
    dataframe_id = id(val)
    pagination = DataframePaginationAttributes(limit=10, offset=0)
    data = val
    sort_by = ""
    columns = _get_dataframe_columns(val)
    # check if it's a dataframe for ChartCell then return full dataframe
    if hasattr(val, "attrs") and "returnAll" in val.attrs and val.attrs["returnAll"]:
        if "aggregation" in val.attrs:
            aggregation = DataframeAggregationParams(
                group_by=val.attrs["aggregation"]["group_by"],
                aggregate_by=val.attrs["aggregation"]["aggregate_by"],
                aggregation_func=val.attrs["aggregation"]["aggregation_func"],
            )
            data = _aggregate_dataframe(val, aggregation)

        if len(data.index) >= dataframe_limit:
            pagination = DataframePaginationAttributes(limit=dataframe_limit, offset=0)
            data = _paginate_dataframe(data, pagination)

        return {
            "columns": columns,
            "data": _transform_to_json(data),
            "referenceId": dataframe_id,
        }

    # Sorting step, gets attrs that has been setuped in DataframeProcessor
    if hasattr(val, "attrs") and "sort_by" in val.attrs:
        data = _sort_dataframe(df=data, sort_by=val.attrs["sort_by"])
        sort_by = val.attrs["sort_by"]

    # Pagination step, gets attrs that has been setuped in DataframeProcessor
    if hasattr(val, "attrs") and "pagination" in val.attrs:
        pagination = DataframePaginationAttributes(
            limit=val.attrs["pagination"]["limit"], offset=val.attrs["pagination"]["offset"]
        )
    # is dataframe length is less than pagingation limit
    # no need to paginate it
    if len(data.index) > int(pagination.limit):
        data = _paginate_dataframe(data, pagination)

    return {
        "data": _transform_to_json(data),
        "columns": columns,
        "count": len(data.index),
        "totalCount": len(val.index),
        "offset": int(pagination.offset),
        "limit": int(pagination.limit),
        "referenceId": dataframe_id,
        "sortedBy": sort_by,
    }


# To Add a new data formatter we need create a new class instance based on a
# BaseFormatter from the iPython kernel
#
class DataFrameFormatter(BaseFormatter):
    """A DataFrame formatter. This is basically a copy of the JSONFormatter
    so it will return as a new mime type: application/vnd.dataframe in output.
    """

    format_type = Unicode("application/vnd.dataframe")
    _return_type = (list, dict)

    print_method = ObjectName("_repr_json_")

    def _check_return(self, r, obj):
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


# load our extension into ipython kernel
def load_ipython_extension(ipython):
    if is_pandas_loaded:
        ipython.display_formatter.formatters["application/vnd.dataframe"] = DataFrameFormatter()
        dataframe_formatter = ipython.display_formatter.formatters["application/vnd.dataframe"]
        dataframe_formatter.for_type(DataFrame, formatter)
        print("Pandas DataFrame MimeType Extension loaded")
    else:
        print("Please make `pip install pandas` to use DataFrame extension")
