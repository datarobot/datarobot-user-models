"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import shutil
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.io import mmwrite

from datarobot_drum.drum.enum import ArgumentsOptions
from datarobot_drum.drum.utils.drum_utils import handle_missing_colnames
from datarobot_drum.resource.utils import (
    _cmd_add_class_labels,
    _create_custom_model_dir,
    _exec_shell_cmd,
)
from tests.constants import (
    ANOMALY,
    BINARY,
    BINARY_INT,
    BINARY_INT_TARGET,
    BINARY_TEXT,
    DOCKER_PYTHON_SKLEARN,
    R_XFORM_ESTIMATOR,
    MULTICLASS,
    PYTHON,
    R_FIT,
    RDS,
    RDS_BINARY,
    RDS_SPARSE,
    REGRESSION,
    SKLEARN_BINARY,
    SPARSE,
    SPARSE_TARGET,
    WEIGHTS_ARGS,
    WEIGHTS_CSV,
    CUSTOM_TASK_INTERFACE_REGRESSION,
    CUSTOM_TASK_INTERFACE_ANOMALY,
    CUSTOM_TASK_INTERFACE_MULTICLASS,
)


class TestFit:
    @staticmethod
    def _add_weights_cmd(weights, df, input_name, r_fit=False):
        colname = "some-weights"
        weights_data = pd.Series(np.random.randint(1, 3, len(df)))
        ext = os.path.splitext(input_name)[1]
        if weights == WEIGHTS_ARGS:
            __keep_this_around = NamedTemporaryFile("w", suffix=ext)
            df[colname] = weights_data
            if r_fit:
                df = handle_missing_colnames(df)
            if ext == ".mtx":
                with open(input_name.replace(".mtx", ".columns")) as f:
                    sparse_colnames = [col.rstrip() for col in f]
                sparse_colnames.append(colname)
                tmp_colname_file = __keep_this_around.name.replace(".mtx", ".columns")
                with open(tmp_colname_file, "w") as f:
                    f.write("\n".join(sparse_colnames))
                df[colname] = pd.arrays.SparseArray(
                    df[colname], dtype=pd.SparseDtype(np.float64, 0)
                )
                mmwrite(__keep_this_around.name, sp.csr_matrix(df.to_numpy()))
            else:
                df.to_csv(__keep_this_around.name, index=False, lineterminator="\r\n")
            return " --row-weights " + colname, __keep_this_around.name, __keep_this_around
        elif weights == WEIGHTS_CSV:
            __keep_this_around = NamedTemporaryFile("w")
            weights_data.to_csv(__keep_this_around.name, index=False, lineterminator="\r\n")
            return " --row-weights-csv " + __keep_this_around.name, input_name, __keep_this_around

        __keep_this_around = NamedTemporaryFile("w")
        return "", input_name, __keep_this_around

    @pytest.mark.parametrize(
        "framework, problem, docker",
        [
            pytest.param(
                SKLEARN_BINARY,
                BINARY_TEXT,
                DOCKER_PYTHON_SKLEARN,
                marks=pytest.mark.skip(
                    reason="RAPTOR-10673: need to implement running docker inside docker"
                ),
            ),
            pytest.param(
                CUSTOM_TASK_INTERFACE_REGRESSION,
                REGRESSION,
                DOCKER_PYTHON_SKLEARN,
                marks=pytest.mark.skip(
                    reason="RAPTOR-10673: need to implement running docker inside docker"
                ),
            ),
            pytest.param(
                CUSTOM_TASK_INTERFACE_ANOMALY,
                ANOMALY,
                DOCKER_PYTHON_SKLEARN,
                marks=pytest.mark.skip(
                    reason="RAPTOR-10673: need to implement running docker inside docker"
                ),
            ),
            pytest.param(
                CUSTOM_TASK_INTERFACE_MULTICLASS,
                MULTICLASS,
                DOCKER_PYTHON_SKLEARN,
                marks=pytest.mark.skip(
                    reason="RAPTOR-10673: need to implement running docker inside docker"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    def test_fit(
        self,
        resources,
        framework,
        problem,
        docker,
        weights,
        tmp_path,
    ):
        if framework in {RDS, RDS_BINARY, RDS_SPARSE, R_XFORM_ESTIMATOR}:
            language = R_FIT
        else:
            language = PYTHON

        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
            is_training=True,
        )

        input_dataset = resources.datasets(framework, problem)
        input_df = resources.input_data(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_df, input_dataset, r_fit=language == R_FIT
        )

        target_type = resources.target_types(problem)

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose --show-stacktrace --disable-strict-validation".format(
            ArgumentsOptions.MAIN_COMMAND, target_type, custom_model_dir, input_dataset
        )
        if problem not in {ANOMALY, SPARSE, BINARY_INT}:
            cmd += ' --target "{}"'.format(resources.targets(problem))

        if problem == SPARSE:
            input_dir = tmp_path / "input_dir"
            input_dir.mkdir(parents=True, exist_ok=True)
            target_file = os.path.join(input_dir, "y.csv")
            shutil.copyfile(resources.datasets(None, SPARSE_TARGET), target_file)
            sparse_column_file = input_dataset.replace(".mtx", ".columns")
            cmd += " --sparse-column-file {} --target-csv {}".format(
                sparse_column_file, target_file
            )
        if problem == BINARY_INT:
            # target-csv will result in target dtype int instead of str
            target_dataset = resources.datasets(None, BINARY_INT_TARGET)
            cmd += " --target-csv {}".format(target_dataset)

        if problem in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type=target_type
            )

        if docker:
            cmd += " --docker {} ".format(docker)

        cmd += weights_cmd

        _, stdout, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
        )
        assert "Starting Fit" in stdout
        assert "Starting Prediction" in stdout
