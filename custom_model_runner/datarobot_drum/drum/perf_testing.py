"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import collections
import json
import os
import select

import psutil
import pandas as pd
import numpy as np
import shutil
from progress.bar import Bar
import requests
import subprocess
import signal
import sys
import time
from scipy.io import mmread, mmwrite
from scipy.sparse import vstack
from texttable import Texttable
from tempfile import mkdtemp, mkstemp

from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils
from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation
from datarobot_drum.drum.exceptions import (
    DrumCommonException,
    DrumPerfTestTimeout,
    DrumPredException,
    DrumSchemaValidationException,
)
from datarobot_drum.drum.utils.drum_utils import DrumUtils

from datarobot_drum.drum.enum import (
    RESPONSE_PREDICTIONS_KEY,
    SPARSE_COLNAMES,
    PERF_TEST_SERVER_LABEL,
    PredictionServerMimetypes,
    ArgumentsOptions,
    TargetType,
    InputFormatExtension,
)
from datarobot_drum.resource.drum_server_utils import DrumServerRun
from datarobot_drum.resource.transform_helpers import (
    read_csv_payload,
    read_mtx_payload,
    make_csv_payload,
    make_mtx_payload,
    parse_multi_part_response,
    filter_urllib3_logging,
    read_x_data_from_response,
)


def _get_samples_df(df, samples):
    nrows = df.shape[0]
    if nrows >= samples:
        return df.head(n=samples)
    else:
        multiplier = int(samples / nrows)
        remainder = samples % nrows
        ret_df = pd.concat([df] * multiplier, ignore_index=True)
        ret_df = pd.concat([ret_df, df.head(n=remainder)])
        return ret_df


def _get_approximate_samples_in_csv_size(file, target_csv_size):
    df = pd.read_csv(file)
    file_size = os.stat(file).st_size
    lines_multiplier = target_csv_size / file_size
    return int(df.shape[0] * lines_multiplier) + 1


def _find_drum_perf_test_server_process():
    for proc in psutil.process_iter():
        try:
            if proc.environ().get(PERF_TEST_SERVER_LABEL, False):
                return proc.pid
        except (psutil.AccessDenied, psutil.ZombieProcess, psutil.NoSuchProcess) as e:
            continue
    return None


def _kill_drum_perf_test_server_process(pid, verbose=False):
    if pid is None:
        return
    if verbose:
        print("Detected running perf test drum server process ... killing it")
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        time.sleep(0.3)
        proc.kill()
    except psutil.NoSuchProcess:
        pass


PerfTestCase = collections.namedtuple("PerfTestCase", "name samples iterations")


class TestCaseResults:
    def __init__(self, name, iterations, samples, stats_obj):
        self.name = name
        self.iterations = iterations
        self.samples = samples
        self.stats_obj = stats_obj
        self.prediction_ok = False
        self.prediction_error = None
        self.server_stats = None


class PerfTestResultsFormatter:
    def __init__(
        self, results, in_docker, memory_limit=None, show_mem=True, show_inside_server=False
    ):
        self._results = results
        self._in_docker = in_docker
        self._show_mem = show_mem
        self._memory_limit = (memory_limit,)
        self._show_inside_server = show_inside_server
        self._table = None
        self._rows = None

    def _init_table(self):
        self._table = Texttable()
        self._table.set_deco(Texttable.HEADER)
        header_names = ["size", "samples", "iters", "min", "avg", "max", "total (s)"]
        header_types = ["t", "i", "i", "f", "f", "f", "f"]
        col_allign = ["l", "r", "r", "r", "r", "r", "r"]

        if self._show_mem:
            if self._in_docker:
                header_names.extend(
                    [
                        "container\nused (MB)",
                        "container\nmax used (MB)",
                        "container\nlimit (MB)",
                        "total physical (MB)",
                    ]
                )
                header_types.extend(["f", "f", "f", "f"])
                col_allign.extend(["r", "r", "r", "r"])
            else:
                header_names.extend(["used (MB)", "total physical (MB)"])
                header_types.extend(["f", "f"])
                col_allign.extend(["r", "r"])

        if self._show_inside_server:
            header_names.extend(["prediction"])
            header_types.extend(["f"])
            col_allign.extend(["r"])

        self._table.set_cols_dtype(header_types)
        self._table.set_cols_align(col_allign)

        try:
            terminal_size = shutil.get_terminal_size()
            self._table.set_max_width(terminal_size.columns)
        except Exception as e:
            pass

        self._rows = [header_names]

    @staticmethod
    def _same_value_list(value, n):
        return [value for _ in range(n)]

    def _add_mem_info(self, row, server_stats):
        if server_stats is not None and "mem_info" in server_stats:
            mem_info = server_stats["mem_info"]
            if self._in_docker:
                if self._memory_limit is None:
                    container_limit = "no memory limit"
                else:
                    container_limit = mem_info["container_limit"]

                if "container_max_used" in mem_info:
                    max_used = mem_info["container_max_used"]
                else:
                    max_used = "N/A"

                if "container_used" in mem_info:
                    used = mem_info["container_used"]
                else:
                    used = "N/A"

                row.extend([used, max_used, container_limit, mem_info["total"]])
            else:
                row.extend([mem_info["drum_rss"], mem_info["total"]])
        else:
            row.extend(self._same_value_list(CMRunTests.NA_VALUE, 4 if self._in_docker else 2))

    def _add_inside_server_info(self, row, server_stats):
        if server_stats:
            time_info = server_stats["time_info"]
            row.extend([time_info["run_predictor_total"]["avg"]])
        else:
            row.extend([CMRunTests.NA_VALUE])

    def get_tbl_str(self):
        self._init_table()

        for res in self._results:
            row = [res.name, res.samples, res.iterations]

            server_stats = None
            if res.prediction_ok:
                d = res.stats_obj.dict_report(CMRunTests.REPORT_NAME)
                row.extend([d["min"], d["avg"], d["max"], d["total"]])
                server_stats = json.loads(res.server_stats) if res.server_stats else None
            else:
                row.extend(self._same_value_list(CMRunTests.TEST_CASE_FAIL_VALUE, 4))

            if self._show_mem:
                self._add_mem_info(row, server_stats)

            if self._show_inside_server:
                self._add_inside_server_info(row, server_stats)

            self._rows.append(row)

        self._table.add_rows(self._rows)
        tbl_report = self._table.draw()
        return tbl_report


class CMRunTests:
    REPORT_NAME = "Request time (s)"
    NA_VALUE = "NA"
    TEST_CASE_FAIL_VALUE = "Fail"

    def __init__(self, options, target_type=None, schema_validator=None):
        self.options = options
        self.target_type = target_type
        self._schema_validator = schema_validator
        self._verbose = self.options.verbose
        self._input_csv = self.options.input
        self._sparse_column_file = (
            self.options.sparse_column_file if "sparse_column_file" in self.options else None
        )
        self._input_df = StructuredInputReadUtils.read_structured_input_file_as_df(
            filename=self._input_csv, sparse_column_file=self._sparse_column_file
        )

        self._server_addr = "localhost"
        self._server_port = DrumUtils.find_free_port()
        self._url_server_address = "http://{}:{}".format(self._server_addr, self._server_port)
        self._predict_endpoint = "/predict/"
        self._stats_endpoint = "/stats/"
        self._timeout = 60
        self._server_process = None

        self._df_for_test = None
        self._test_cases_to_run = None

    @staticmethod
    def resolve_labels(target_type, options):
        if target_type == TargetType.BINARY:
            labels = [options.negative_class_label, options.positive_class_label]
        elif target_type == TargetType.MULTICLASS:
            labels = options.class_labels
        else:
            labels = None
        return labels

    @staticmethod
    def load_transform_output(response, is_sparse, request_key):
        parsed_response = parse_multi_part_response(response)
        if is_sparse:
            return pd.DataFrame(read_mtx_payload(parsed_response, request_key))
        else:
            return pd.DataFrame(read_csv_payload(parsed_response, request_key))

    def _prepare_test_cases(self):
        print("Preparing test data...")

        file_size = os.stat(self._input_csv).st_size
        sample_size = int(file_size / self._input_df.shape[0])

        def _get_size_and_units(size):
            if size < 1024:
                return size, "bytes"
            else:
                return int(size / 1024), "KB"

        if self.options.iterations is None and self.options.samples is None:
            samples_in_50mb = _get_approximate_samples_in_csv_size(self._input_csv, 50 * 1048576)

            # generate 50mb data frame in the beginning,
            # because afterwards it is easier to take only part of it for every test case
            df_50mb = _get_samples_df(self._input_df, samples_in_50mb)

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
            samples = (
                self._input_df.shape[0] if self.options.samples is None else self.options.samples
            )
            samples_size = samples * sample_size
            self._test_cases_to_run = [
                PerfTestCase("{} {}".format(*_get_size_and_units(samples_size)), samples, iters)
            ]
            self._df_for_test = self._input_df

    def _wait_for_server_to_start(self):
        while True:
            try:
                response = requests.get(self._url_server_address, timeout=1)
                if response.ok:
                    break
            except Exception:
                pass

            time.sleep(1)
            self._timeout = self._timeout - 1
            if self._timeout == 0:
                error_message = "Error: server failed to start while running performance testing."
                server_stdout = self._read_sever_stdout_non_blocking()
                if len(server_stdout):
                    error_message += "\n{}".format(server_stdout)

                raise DrumCommonException(error_message)

    def _read_sever_stdout_non_blocking(self):
        stdout = []
        while len(select.select([self._server_process_fd_read], [], [], 0)[0]) == 1:
            out = os.read(self._server_process_fd_read, 1024)
            stdout.append(out)

        return b"\n".join(stdout).decode("utf-8")

    def _build_drum_cmd(self):
        cmd_list = [
            "{}".format(ArgumentsOptions.MAIN_COMMAND),
            "server",
            "--code-dir",
            self.options.code_dir,
        ]
        cmd_list.append("--address")
        cmd_list.append("{}:{}".format(self._server_addr, self._server_port))
        cmd_list.append("--logging-level")
        cmd_list.append("warning")
        cmd_list.append("--show-perf")
        cmd_list.append("--target-type")
        cmd_list.append(self.options.target_type)
        if self.options.production:
            cmd_list.append(ArgumentsOptions.PRODUCTION)
        if self.options.max_workers:
            cmd_list.append(ArgumentsOptions.MAX_WORKERS)
            cmd_list.append(str(self.options.max_workers))

        if self.options.positive_class_label is not None:
            cmd_list.extend(
                [ArgumentsOptions.POSITIVE_CLASS_LABEL, self.options.positive_class_label]
            )
        if self.options.negative_class_label is not None:
            cmd_list.extend(
                [ArgumentsOptions.NEGATIVE_CLASS_LABEL, self.options.negative_class_label]
            )

        if self.options.class_labels:
            cmd_list.append(ArgumentsOptions.CLASS_LABELS)
            cmd_list.extend(self.options.class_labels)

        if self.options.docker:
            cmd_list.extend([ArgumentsOptions.DOCKER, self.options.docker])
        if self.options.memory:
            cmd_list.extend([ArgumentsOptions.MEMORY, self.options.memory])

        return cmd_list

    def _start_drum_server(self):
        cmd_list = self._build_drum_cmd()
        if self._verbose:
            print("Running drum using: [{}]".format(" ".join(cmd_list)))

        env_vars = os.environ
        env_vars.update({PERF_TEST_SERVER_LABEL: "1"})

        (pipe_r, pipe_w) = os.pipe()
        self._server_process_fd_read = pipe_r
        self._server_process_fd_write = pipe_w

        self._server_process = subprocess.Popen(
            cmd_list,
            env=env_vars,
            stdout=pipe_w,
            stderr=pipe_w,
            encoding="utf-8",
            preexec_fn=os.setsid,
        )

        self._wait_for_server_to_start()

    def _stop_drum_server(self):
        print("Test is done, stopping drum server")
        # cleanup pipe
        os.close(self._server_process_fd_read)
        os.close(self._server_process_fd_write)

        _kill_drum_perf_test_server_process(self._server_process.pid)
        os.system("tput init")

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

            # TODO: add try catch so no failures..
            response = requests.post(
                self._url_server_address + self._predict_endpoint, files={"X": df_csv}
            )

            sc.mark("end")
            if response.ok:
                tc_results.prediction_ok = True
            else:
                tc_results.prediction_ok = False
                tc_results.prediction_error = response.text
                if self._verbose:
                    print("Failed sending prediction request to server: {}".format(response.text))
                return

            actual_num_predictions = len(json.loads(response.text)["predictions"])
            if actual_num_predictions != test_df_nrows:
                print(
                    "Failed, number of predictions in response: {} is not as expected: {}".format(
                        actual_num_predictions, test_df_nrows
                    )
                )
                # TODO: do not throw exception here.. all should be in the tc_results.
                assert actual_num_predictions == test_df_nrows
            sc.disable()
            bar.next()
        bar.finish()

        # TODO: even if prediction request fail we should try and get server stats
        response = requests.get(self._url_server_address + self._stats_endpoint)

        if response.ok:
            tc_results.server_stats = response.text

    def _run_all_test_cases(self):
        results = []
        for tc in self._test_cases_to_run:
            print("Running test case with timeout: {}".format(self.options.timeout))
            signal.alarm(self.options.timeout)
            try:
                self._run_test_case(tc, results)
            except DrumPerfTestTimeout:
                print("... timed out ({}s)".format(self.options.timeout))
            except Exception as e:
                print("\ntest case failed with a message: {}".format(e))
        return results

    def _init_signals(self):
        def signal_handler(sig, frame):
            print("\nCtrl+C pressed, aborting test")
            self._stop_drum_server()
            os.system("tput init")
            sys.exit(0)

        def testcase_timeout(signum, frame):
            raise DrumPerfTestTimeout()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGALRM, testcase_timeout)

    def _reset_signals(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)

    def _print_perf_test_params(self):
        print("DRUM performance test")
        print("Model:      {}".format(self.options.code_dir))
        print("Data:       {}".format(self._input_csv))
        print("# Features: {}".format(len(self._input_df.columns)))
        sys.stdout.flush()

    def performance_test(self):
        self._print_perf_test_params()
        self._prepare_test_cases()

        _kill_drum_perf_test_server_process(
            _find_drum_perf_test_server_process(), self.options.verbose
        )
        if DrumUtils.is_port_in_use(self._server_addr, self._server_port):
            error_message = "\nError: address: {} is in use".format(self._url_server_address)
            print(error_message)
            raise DrumCommonException(error_message)

        self._start_drum_server()
        self._init_signals()

        print("\n\n")
        results = self._run_all_test_cases()
        self._reset_signals()
        self._stop_drum_server()
        str_report = PerfTestResultsFormatter(
            results, in_docker=bool(self.options.docker), show_inside_server=self.options.in_server
        ).get_tbl_str()

        print("\n" + str_report)
        return

    def _basic_batch_prediction_check(self):
        test_name = "Basic batch prediction"
        test_passed = True
        failure_message = ""
        cmd_list = sys.argv

        TMP_DIR = "/tmp"
        DIR_PREFIX = "drum_validation_check_"

        output_dir = mkdtemp(prefix=DIR_PREFIX, dir=TMP_DIR)
        output_filename = os.path.join(output_dir, "output")

        DrumUtils.replace_cmd_argument_value(cmd_list, ArgumentsOptions.OUTPUT, output_filename)

        p = subprocess.Popen(cmd_list, env=os.environ)
        retcode = p.wait()
        if retcode != 0:
            test_passed = False
            failure_message = "Test failed on provided dataset: {}".format(self._input_csv)

        return test_name, test_passed, failure_message

    def _null_value_imputation_check(self):
        test_name = "Null value imputation"
        test_passed = True
        failure_message = ""
        cmd_list = sys.argv

        TMP_DIR = "/tmp"
        DIR_PREFIX = "drum_validation_checks_"

        ValidationTestResult = collections.namedtuple(
            "ValidationTestResult", "filename retcode message"
        )

        null_datasets_dir = mkdtemp(prefix=DIR_PREFIX, dir=TMP_DIR)

        df = pd.read_csv(self._input_csv)
        column_names = list(df.iloc[[0]])

        results = {}
        for i, column_name in enumerate(column_names):
            output_filename = os.path.join(null_datasets_dir, "output{}".format(i))
            tmp_dataset_file_path = os.path.join(
                null_datasets_dir, "null_value_imputation_column{}".format(i)
            )
            df_tmp = df.copy()
            df_tmp[column_name] = None
            df_tmp.to_csv(tmp_dataset_file_path, index=False)
            DrumUtils.replace_cmd_argument_value(
                cmd_list, ArgumentsOptions.INPUT, tmp_dataset_file_path
            )
            DrumUtils.replace_cmd_argument_value(cmd_list, ArgumentsOptions.OUTPUT, output_filename)

            p = subprocess.Popen(cmd_list, env=os.environ)
            retcode = p.wait()
            if retcode != 0:
                test_passed = False
                results[column_name] = ValidationTestResult(tmp_dataset_file_path, retcode, "")

        # process results
        if test_passed:
            shutil.rmtree(null_datasets_dir)
        else:
            for test_result in results.values():
                if not test_result.retcode:
                    os.remove(test_result.filename)

            table = Texttable()
            table.set_deco(Texttable.HEADER)

            headers = ["Failed feature", "Message", "Dataset filename"]

            table.set_cols_dtype(["t", "t", "t"])
            table.set_cols_align(["l", "l", "l"])

            rows = [headers]

            for key, test_result in results.items():
                if test_result.retcode:
                    rows.append([key, test_result.message, test_result.filename])

            table.add_rows(rows)
            table_res = table.draw()

            message = (
                "Null value imputation check performs check by imputing each feature with NaN value. "
                "If check fails for a feature, test dataset is saved in {}/{}* "
                "Make sure to delete those folders if it takes too much space.".format(
                    TMP_DIR, DIR_PREFIX
                )
            )
            failure_message = "{}\n\n{}".format(message, table_res)

        return test_name, test_passed, failure_message

    def validation_test(self):
        # TODO: create infrastructure to easily add more checks

        cmd_list = sys.argv
        cmd_list[1] = ArgumentsOptions.SCORE

        if ArgumentsOptions.OUTPUT not in cmd_list:
            cmd_list.extend(
                [ArgumentsOptions.OUTPUT, "placeholder_to_replace_later_during_test_case"]
            )

        test_cases_results = [
            self._basic_batch_prediction_check(),
            self._null_value_imputation_check(),
        ]

        table = Texttable()
        table.set_deco(Texttable.HEADER)

        try:
            terminal_size = shutil.get_terminal_size()
            table.set_max_width(terminal_size.columns)
        except Exception as e:
            pass

        header_names = ["Test case", "Status", "Details"]
        col_types = ["t", "t", "t"]
        col_align = ["l", "l", "l"]

        rows = []
        rows.append(header_names)
        for test_name, test_passed, failure_message in test_cases_results:
            rows.append(
                [
                    test_name,
                    "PASSED" if test_passed else "FAILED",
                    "" if test_passed else failure_message,
                ]
            )

        table.set_cols_dtype(col_types)
        table.set_cols_align(col_align)
        table.add_rows(rows)
        tbl_report = table.draw()
        print("\n\nValidation checks results")
        print(tbl_report)

    def check_transform_server(self, target_temp_location=None):
        with DrumServerRun(
            self.target_type.value,
            self.resolve_labels(self.target_type, self.options),
            self.options.code_dir,
            verbose=self._verbose,
            user_secrets_mount_path=self.options.user_secrets_mount_path,
        ) as run:
            endpoint = "/transform/"
            payload = {"X": open(self.options.input)}
            if self.options.sparse_column_file:
                payload.update({SPARSE_COLNAMES: open(self.options.sparse_column_file)})

            # there is a known bug in urllib3 that needlessly gives a header warning
            # this will suppress the warning for better user experience when running performance test
            filter_urllib3_logging()
            if self.options.target:
                target_location = target_temp_location.name
                payload.update({"y": open(target_location)})
            elif self.options.target_csv:
                target_location = self.options.target_csv
                payload.update({"y": open(target_location)})

            response = requests.post(run.url_server_address + endpoint, files=payload)
            if not response.ok:
                error_msg = "Failure in {} server: {}".format(endpoint[1:-1], response.text)
                if response.status_code == 422 and response.json().get(
                    "is_schema_validation_error"
                ):
                    raise DrumSchemaValidationException(error_msg)
                else:
                    raise DrumCommonException(error_msg)
            transformed_values = read_x_data_from_response(response)
            self._schema_validator.validate_outputs(transformed_values)

    def check_prediction_side_effects(self):
        rtol = 2e-02
        atol = 1e-06
        input_extension = os.path.splitext(self.options.input)
        is_sparse = input_extension[1] == InputFormatExtension.MTX

        if is_sparse:
            columns = [
                column.strip() for column in open(self.options.sparse_column_file).readlines()
            ]
            df = pd.DataFrame.sparse.from_spmatrix(mmread(self.options.input), columns=columns)
            samplesize = min(1000, max(int(len(df) * 0.1), 10))
            data_subset = df.sample(n=samplesize, random_state=42)
            subset_payload, colnames = make_mtx_payload(data_subset)
            subset_payload = ("X.mtx", subset_payload)
            files = {
                "X": subset_payload,
                SPARSE_COLNAMES: (
                    SPARSE_COLNAMES,
                    colnames,
                    PredictionServerMimetypes.APPLICATION_OCTET_STREAM,
                ),
            }
        else:
            df = pd.read_csv(self.options.input)
            samplesize = min(1000, max(int(len(df) * 0.1), 10))
            data_subset = df.sample(n=samplesize, random_state=42)
            subset_payload = make_csv_payload(data_subset)
            files = {"X": subset_payload}

        labels = self.resolve_labels(self.target_type, self.options)

        with DrumServerRun(
            self.target_type.value,
            labels,
            self.options.code_dir,
            verbose=self._verbose,
            user_secrets_mount_path=self.options.user_secrets_mount_path,
        ) as run:
            endpoint = "/predict/"
            payload = {"X": open(self.options.input)}
            if is_sparse:
                payload.update(
                    {
                        SPARSE_COLNAMES: (
                            SPARSE_COLNAMES,
                            open(self.options.sparse_column_file),
                            PredictionServerMimetypes.APPLICATION_OCTET_STREAM,
                        )
                    }
                )

            response_full = requests.post(run.url_server_address + endpoint, files=payload)
            if not response_full.ok:
                raise DrumCommonException(
                    "Failure in {} server: {}".format(endpoint[1:-1], response_full.text)
                )

            response_sample = requests.post(run.url_server_address + endpoint, files=files)
            if not response_sample.ok:
                raise DrumCommonException(
                    "Failure in {} server: {}".format(endpoint[1:-1], response_sample.text)
                )

            preds_full = pd.DataFrame(json.loads(response_full.text)[RESPONSE_PREDICTIONS_KEY])
            preds_sample = pd.DataFrame(json.loads(response_sample.text)[RESPONSE_PREDICTIONS_KEY])

            preds_full_subset = preds_full.iloc[data_subset.index]

            self._schema_validator.validate_outputs(preds_sample)

            matches = np.isclose(preds_full_subset, preds_sample, rtol=rtol, atol=atol)
            if not np.all(matches):
                if is_sparse:
                    _, __tempfile_sample = mkstemp(suffix=InputFormatExtension.MTX)
                    sparse_mat = vstack(x[0] for x in data_subset.values)
                    mmwrite(__tempfile_sample, sparse_mat.sparse.to_coo())
                else:
                    _, __tempfile_sample = mkstemp(suffix=".csv")
                    data_subset.to_csv(__tempfile_sample, index=False)

                message = """
                            Warning: Your predictions were different when we tried to predict twice.
                            The last 10 predictions from the main predict run were: {}
                            However when we reran predictions on the same data, we got: {}.
                            The sample used to calculate prediction reruns can be found in this file: {}""".format(
                    preds_full_subset[~matches][:10].to_string(index=False),
                    preds_sample[~matches][:10].to_string(index=False),
                    __tempfile_sample,
                )
                raise DrumPredException(message)
