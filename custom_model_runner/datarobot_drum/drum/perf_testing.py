import collections
import json
import os
import psutil
import pandas as pd
import shutil
from progress.bar import Bar
import requests
import subprocess
import signal
import sys
import time
from texttable import Texttable
from tempfile import mkdtemp, NamedTemporaryFile

from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation
from datarobot_drum.drum.exceptions import DrumCommonException, DrumPerfTestTimeout
from datarobot_drum.drum.utils import CMRunnerUtils
from datarobot_drum.drum.common import RunMode, ArgumentsOptions


def _get_samples_df(df, samples):
    nrows = df.shape[0]
    if nrows >= samples:
        return df.head(n=samples)
    else:
        multiplier = int(samples / nrows)
        remainder = samples % nrows
        ret_df = pd.concat([df] * multiplier, ignore_index=True)
        ret_df = ret_df.append(df.head(n=remainder))
        return ret_df


def _get_approximate_samples_in_csv_size(file, target_csv_size):
    df = pd.read_csv(file)
    file_size = os.stat(file).st_size
    lines_multiplier = target_csv_size / file_size
    return int(df.shape[0] * lines_multiplier) + 1


def _find_and_kill_cmrun_server_process(verbose=False):
    for proc in psutil.process_iter():
        if "{}".format(ArgumentsOptions.MAIN_COMMAND) in proc.name().lower():
            if "{}".format(ArgumentsOptions.SERVER) in proc.cmdline():
                if verbose:
                    print(
                        "Detected older {} process ... stopping it".format(
                            ArgumentsOptions.MAIN_COMMAND
                        )
                    )
                try:
                    proc.terminate()
                    time.sleep(0.3)
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
                break


PerfTestCase = collections.namedtuple("PerfTestCase", "name samples iterations")


class TestCaseResults:
    def __init__(self, name, iterations, samples, stats_obj):
        self.name = name
        self.iterations = iterations
        self.samples = samples
        self.stats_obj = stats_obj
        self.server_stats = None


class CMRunTests:
    REPORT_NAME = "Request time (s)"

    def __init__(self, options, run_mode):
        self.options = options
        self._input_csv = self.options.input

        self._server_addr = "localhost"
        self._server_port = CMRunnerUtils.find_free_port()
        self._url_server_address = "http://{}:{}".format(self._server_addr, self._server_port)
        self._shutdown_endpoint = "/shutdown/"
        self._predict_endpoint = "/predict/"
        self._stats_endpoint = "/stats/"
        self._timeout = 20
        self._server_process = None

        self._df_for_test = None
        self._test_cases_to_run = None
        if run_mode == RunMode.PERF_TEST:
            self._prepare_test_cases()

    def _prepare_test_cases(self):
        print("Preparing test data...")
        df = pd.read_csv(self._input_csv)
        file_size = os.stat(self._input_csv).st_size
        sample_size = int(file_size / df.shape[0])

        def _get_size_and_units(size):
            if size < 1024:
                return size, "bytes"
            else:
                return int(size / 1024), "KB"

        if self.options.iterations is None and self.options.samples is None:
            samples_in_50mb = _get_approximate_samples_in_csv_size(self._input_csv, 50 * 1048576)

            # generate 50mb data frame in the beginning,
            # because afterwards it is easier to take only part of it for every test case
            df_50mb = _get_samples_df(df, samples_in_50mb)

            _default_test_cases = [
                PerfTestCase("{} {}".format(*_get_size_and_units(sample_size)), 1, 100),
                PerfTestCase("0.1MB", int(samples_in_50mb / 500), 50),
                PerfTestCase("10MB", int(samples_in_50mb / 5), 5),
                PerfTestCase("50MB", samples_in_50mb, 1),
            ]
            self._test_cases_to_run = _default_test_cases
            self._df_for_test = df_50mb
        else:
            iters = 1 if self.options.iterations is None else self.options.iterations
            samples = df.shape[0] if self.options.samples is None else self.options.samples
            samples_size = samples * sample_size
            self._test_cases_to_run = [
                PerfTestCase("{} {}".format(*_get_size_and_units(samples_size)), samples, iters)
            ]
            self._df_for_test = df

    def _wait_for_server_to_start(self):
        while True:
            try:
                response = requests.get(self._url_server_address)
                if response.ok:
                    break
            except Exception:
                pass

            time.sleep(1)
            self._timeout = self._timeout - 1
            if self._timeout == 0:
                error_message = "Error: server failed to start while running performance testing"
                print(error_message)
                raise DrumCommonException(error_message)

    def _build_cmrun_cmd(self):
        cmd_list = [
            "{}".format(ArgumentsOptions.MAIN_COMMAND),
            "server",
            "--code-dir",
            self.options.code_dir,
        ]
        cmd_list.append("--in-perf-mode-internal")
        cmd_list.append("--address")
        cmd_list.append("{}:{}".format(self._server_addr, self._server_port))
        cmd_list.append("--logging-level")
        cmd_list.append("warning")
        cmd_list.append("--show-perf")

        if self.options.positive_class_label:
            cmd_list.append("--positive-class-label")
            cmd_list.append(self.options.positive_class_label)
        if self.options.negative_class_label:
            cmd_list.append("--negative-class-label")
            cmd_list.append(self.options.negative_class_label)

        if self.options.docker:
            cmd_list.extend(["--docker", self.options.docker])

        return cmd_list

    def _run_test_case(self, tc, results):
        print(
            "Running test case: {} - {} samples, {} iterations".format(
                tc.name, tc.samples, tc.iterations
            )
        )
        samples = tc.samples
        name = tc.name if tc.name is not None else "Test case"
        sc = StatsCollector()
        sc.register_report(CMRunTests.REPORT_NAME, "end", StatsOperation.SUB, "start")
        tc_results = TestCaseResults(
            name=name, iterations=tc.iterations, samples=samples, stats_obj=sc
        )
        results.append(tc_results)

        test_df = _get_samples_df(self._df_for_test, samples)
        test_df_nrows = test_df.shape[0]
        df_csv = test_df.to_csv(index=False)

        bar = Bar("Processing", max=tc.iterations)
        for i in range(tc.iterations):
            sc.enable()
            sc.mark("start")
            response = requests.post(
                self._url_server_address + self._predict_endpoint, files={"X": df_csv}
            )
            sc.mark("end")
            assert response.ok
            actual_num_predictions = len(json.loads(response.text)["predictions"])
            assert actual_num_predictions == test_df_nrows
            sc.disable()
            bar.next()
        bar.finish()
        response = requests.get(self._url_server_address + self._stats_endpoint)
        tc_results.server_stats = response.text

    def _generate_table_report_adv(self, results, show_mem=True, show_inside_server=True):

        tbl_report = "=" * 52 + "\n"

        table = Texttable()
        table.set_deco(Texttable.HEADER)

        header_names = ["size", "samples", "iters", "min", "avg", "max"]
        header_types = ["t", "i", "i", "f", "f", "f"]
        col_allign = ["l", "r", "r", "r", "r", "r"]

        if show_mem:
            header_names.extend(["used (MB)", "total (MB)"])
            header_types.extend(["f", "f"])
            col_allign.extend(["r", "r"])

        if show_inside_server:
            header_names.extend(["prediction"])
            header_types.extend(["f"])
            col_allign.extend(["r"])

        table.set_cols_dtype(header_types)
        table.set_cols_align(col_allign)

        try:
            terminal_size = shutil.get_terminal_size()
            table.set_max_width(terminal_size.columns)
        except Exception as e:
            pass

        rows = [header_names]

        for res in results:
            row = [res.name, res.samples, res.iterations]
            d = res.stats_obj.dict_report(CMRunTests.REPORT_NAME)
            row.extend([d["min"], d["avg"], d["max"]])
            server_stats = json.loads(res.server_stats) if res.server_stats else None

            if show_mem:
                if server_stats and "mem_info" in server_stats:
                    mem_info = server_stats["mem_info"]
                    row.extend([mem_info["predictor_rss"], mem_info["total"]])
                else:
                    row.extend([-9999, -9999])
                rows.append(row)

            if show_inside_server and server_stats:
                time_info = server_stats["time_info"]
                row.extend(
                    [time_info["run_predictor_total"]["avg"],]
                )

        table.add_rows(rows)
        tbl_report = table.draw()
        return tbl_report

    def _stop_server(self):
        try:
            requests.post(self._url_server_address + self._shutdown_endpoint, timeout=5)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            pass
        finally:
            time.sleep(0.5)
            self._server_process.kill()
            os.system("tput init")

    def performance_test(self):
        _find_and_kill_cmrun_server_process(self.options.verbose)
        if CMRunnerUtils.is_port_in_use(self._server_addr, self._server_port):
            error_message = "\nError: address: {} is in use".format(self._url_server_address)
            print(error_message)
            raise DrumCommonException(error_message)

        cmd_list = self._build_cmrun_cmd()
        self._server_process = subprocess.Popen(cmd_list, env=os.environ)
        self._wait_for_server_to_start()

        def signal_handler(sig, frame):
            print("\nCtrl+C pressed, aborting test")
            print("Sending shutdown to server")
            self._stop_server()
            os.system("tput init")
            sys.exit(0)

        def testcase_timeout(signum, frame):
            raise DrumPerfTestTimeout()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGALRM, testcase_timeout)

        results = []
        print("\n\n")
        for tc in self._test_cases_to_run:
            signal.alarm(self.options.timeout)
            try:
                self._run_test_case(tc, results)
            except DrumPerfTestTimeout:
                print("... timed out ({}s)".format(self.options.timeout))
            except Exception as e:
                print("\n...test case failed with a message: {}".format(e))

        self._stop_server()
        str_report = self._generate_table_report_adv(
            results, show_inside_server=self.options.in_server
        )
        print("\n" + str_report)
        return

    def validation_test(self):

        # TODO: create infrastructure to easily add more checks
        # NullValueImputationCheck
        test_name = "Null value imputation"
        ValidationTestResult = collections.namedtuple("ValidationTestResult", "filename retcode")

        cmd_list = sys.argv
        cmd_list[0] = ArgumentsOptions.MAIN_COMMAND
        cmd_list[1] = ArgumentsOptions.SCORE

        TMP_DIR = "/tmp"
        DIR_PREFIX = "drum_validation_checks_"

        null_datasets_dir = mkdtemp(prefix=DIR_PREFIX, dir=TMP_DIR)

        df = pd.read_csv(self._input_csv)
        column_names = list(df.iloc[[0]])

        results = {}

        for column_name in column_names:
            with NamedTemporaryFile(
                mode="w",
                dir=null_datasets_dir,
                prefix="null_value_imputation_{}_".format(column_name),
                delete=False,
            ) as temp_f:
                temp_data_name = temp_f.name
                df_tmp = df.copy()
                df_tmp[column_name] = None
                df_tmp.to_csv(temp_data_name, index=False)
                CMRunnerUtils.replace_cmd_argument_value(
                    cmd_list, ArgumentsOptions.INPUT, temp_data_name
                )

                p = subprocess.Popen(cmd_list, env=os.environ)
                retcode = p.wait()
                if retcode != 0:
                    results[column_name] = ValidationTestResult(temp_data_name, retcode)

        table = Texttable()
        table.set_deco(Texttable.HEADER)

        try:
            terminal_size = shutil.get_terminal_size()
            table.set_max_width(terminal_size.columns)
        except Exception as e:
            pass

        header_names = ["Test case", "Status"]
        col_types = ["t", "t"]
        col_align = ["l", "l"]

        rows = []

        if len(results) == 0:
            rows.append(header_names)
            rows.append([test_name, "PASSED"])
            shutil.rmtree(null_datasets_dir)
        else:
            col_types.append("t")
            col_align.append("l")
            header_names.append("Details")
            rows.append(header_names)
            for test_result in results.values():
                if not test_result.retcode:
                    os.remove(test_result.filename)

            table2 = Texttable()
            table2.set_deco(Texttable.HEADER)

            message = (
                "Null value imputation check performs check by imputing each feature with NaN value. "
                "If check fails for a feature, test dataset is saved in {}/{}. "
                "Make sure to delete those folders if it takes too much space.".format(
                    TMP_DIR, DIR_PREFIX
                )
            )
            rows.append([test_name, "FAILED", message])

            header_names2 = ["Failed feature", "Dataset filename"]

            table2.set_cols_dtype(["t", "t"])
            table2.set_cols_align(["l", "l"])

            rows2 = [header_names2]

            for key, test_result in results.items():
                if test_result.retcode:
                    rows2.append([key, test_result.filename])
                    pass

            table2.add_rows(rows2)
            table_res = table2.draw()
            rows.append(["", "", "\n{}".format(table_res)])

        table.set_cols_dtype(col_types)
        table.set_cols_align(col_align)
        table.add_rows(rows)
        tbl_report = table.draw()
        print("\n\nValidation checks results")
        print(tbl_report)
