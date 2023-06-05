"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd
import time

from collections import OrderedDict


class StatsOperation(object):
    SUB = "substract"
    ADD = "add"


class StatsCollectorException(Exception):
    pass


class StatsCollector(object):
    def __init__(self, iters=None, disable_instance=False):
        self._iters = iters
        self._iteration_mode = True if iters is not None else False
        self._enabled = False
        self._iter_dict = OrderedDict()
        self._stats_df = None
        self._report_cols = []
        self._disable_instance = disable_instance
        self._report_tuples = []

    def enable(self):
        if self._disable_instance:
            return
        self._iter_dict.clear()
        self._enabled = True

    def disable(self):
        if self._disable_instance:
            return

        for tup in self._report_tuples:
            if tup[2] == StatsOperation.SUB:
                self._iter_dict[tup[0]] = self._iter_dict[tup[1]] - self._iter_dict[tup[3]]
            elif tup[2] == StatsOperation.ADD:
                self._iter_dict[tup[0]] = self._iter_dict[tup[1]] + self._iter_dict[tup[3]]

        self._stats_df = pd.concat([self._stats_df, pd.DataFrame(self._iter_dict, index=[0])])

        self._iter_dict.clear()
        self._enabled = False

    def stats_reset(self):
        if self._stats_df is not None:
            self._stats_df = self._stats_df.iloc[0:0]

    def loop(self, df):
        if self._disable_instance:
            return
        for i in range(self._iters):
            self.enable()
            try:
                yield df  # with body executes here
            except Exception as e:
                print(e)
            finally:
                self.disable()

    def mark(self, name):
        if self._disable_instance:
            return
        if name is None:
            raise StatsCollectorException("name should be provided for mark")
        if not self._enabled:
            raise StatsCollectorException("call enable before setting a mark")
        if name in self._iter_dict:
            raise StatsCollectorException("mark name {} already exists".format(name))

        self._iter_dict[name] = time.time()

    def register_report(self, name, *args):
        if len(args) != 3:
            raise StatsCollectorException("register_report args len must be 3")
        self._report_tuples.append((name, args[0], args[1], args[2]))
        self._report_cols.append(name)

    def print_stats(self):
        if self._disable_instance:
            return
        print(self._stats_df)

    def to_csv(self):
        return self._stats_df.round(3).to_csv(index=False)

    def round(self):
        if self._stats_df is not None:
            self._stats_df = self._stats_df.round(3)

    def str_report(self, name, format_str=None):
        if self._disable_instance:
            return
        if name not in self._report_cols:
            raise StatsCollectorException("report {} does not exist".format(name))

        d = self.dict_report(name)

        if format_str and len(d) > 0:
            return format_str.format(name, d["min"], d["avg"], d["max"])
        elif len(d) == 0:
            return "{}:\n\tsec: min: na; avg: na; max: na"
        else:
            return "{}:\n\tsec: min: {:.2f}; avg: {:.2f}; max: {:.2f}".format(
                name, d["min"], d["avg"], d["max"]
            )

    def dict_report(self, name):
        if self._disable_instance or self._stats_df is None:
            return {"min": None, "max": None, "avg": None, "total": None}
        if name not in self._report_cols:
            raise StatsCollectorException("report {} does not exist".format(name))
        return {
            "min": self._stats_df[name].min(),
            "max": self._stats_df[name].max(),
            "avg": self._stats_df[name].mean(),
            "total": self._stats_df[name].sum(),
        }

    def print_report(self, name, format_str=None):
        print(self.str_report(name, format_str=format_str))

    def get_report_names(self):
        return self._report_cols

    def print_reports(self, format_str=None):
        if self._disable_instance:
            return
        for report in self._report_cols:
            self.print_report(report, format_str=format_str)

    def print_last(self):
        for report_name in self._report_cols:
            report_value = self._stats_df.iloc[-1][report_name]
            print("{}:\n\tsec: {}".format(report_name, report_value))
