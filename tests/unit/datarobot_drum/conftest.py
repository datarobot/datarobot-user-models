"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import sys
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import List

import pytest
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pandas as pd
import yaml
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.drum import CMRunner
from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME
from datarobot_drum.drum.runtime import DrumRuntime
from scipy.io import mmwrite

from datarobot_drum.drum.utils.dataframe import is_sparse_dataframe


@pytest.fixture
def model_id():
    return "5f1f15a4d6111f01cb7f91f"


@pytest.fixture
def environment_id():
    return "5e8c889607389fe0f466c72d"


@pytest.fixture
def project_id():
    return "abc123"


@pytest.fixture
def multiclass_labels():
    return ["GALAXY", "QSO", "STAR"]


###############################################################################
# MODEL METADATA YAMLS


@pytest.fixture
def inference_metadata_yaml(environment_id):
    return dedent(
        """
        name: drumpush-regression
        type: inference
        targetType: regression
        environmentID: {environmentID}
        inferenceModel:
          targetName: Grade 2014
        validation:
          input: hello
        """
    ).format(environmentID=environment_id)


@pytest.fixture
def inference_binary_metadata_yaml_no_target_name(environment_id):
    return dedent(
        """
        name: drumpush-binary
        type: inference
        targetType: binary
        environmentID: {environmentID}
        inferenceModel:
          positiveClassLabel: yes
          negativeClassLabel: no
        validation:
          input: hello
        """
    ).format(environmentID=environment_id)


@pytest.fixture
def inference_binary_metadata_no_label():
    return dedent(
        """
        name: drumpush-binary
        type: inference
        targetType: binary
        inferenceModel:
          positiveClassLabel: yes
        """
    )


@pytest.fixture
def inference_multiclass_metadata_yaml_no_labels(environment_id):
    return dedent(
        """
        name: drumpush-multiclass
        type: inference
        targetType: multiclass
        environmentID: {}
        inferenceModel:
          targetName: class
        validation:
          input: hello
        """
    ).format(environment_id)


@pytest.fixture
def inference_multiclass_metadata_yaml(environment_id, multiclass_labels):
    return dedent(
        """
        name: drumpush-multiclass
        type: inference
        targetType: multiclass
        environmentID: {}
        inferenceModel:
          targetName: class
          classLabels:
            - {}
            - {}
            - {}
        validation:
          input: hello
        """
    ).format(environment_id, *multiclass_labels)


@pytest.fixture
def inference_multiclass_metadata_yaml_label_file(environment_id, multiclass_labels):
    with NamedTemporaryFile(mode="w+") as f:
        f.write("\n".join(multiclass_labels))
        f.flush()
        yield dedent(
            """
            name: drumpush-multiclass
            type: inference
            targetType: multiclass
            environmentID: {}
            inferenceModel:
              targetName: class
              classLabelsFile: {}
            validation:
              input: hello
            """
        ).format(environment_id, f.name)


@pytest.fixture
def inference_multiclass_metadata_yaml_labels_and_label_file(environment_id, multiclass_labels):
    with NamedTemporaryFile(mode="w+") as f:
        f.write("\n".join(multiclass_labels))
        f.flush()
        yield dedent(
            """
            name: drumpush-multiclass
            type: inference
            targetType: multiclass
            environmentID: {}
            inferenceModel:
              targetName: class
              classLabelsFile: {}
              classLabels:
                - {}
                - {}
                - {}
            validation:
              input: hello
            """
        ).format(environment_id, f.name, *multiclass_labels)


@pytest.fixture
def training_metadata_yaml(environment_id):
    return dedent(
        """
        name: drumpush-regression
        type: training
        targetType: regression
        environmentID: {environmentID}
        validation:
           input: hello 
        """
    ).format(environmentID=environment_id)


@pytest.fixture
def training_metadata_yaml_with_proj(environment_id, project_id):
    return dedent(
        """
        name: drumpush-regression
        type: training
        targetType: regression
        environmentID: {environmentID}
        trainingModel:
            trainOnProject: {projectID}
        validation:
            input: hello 
        """
    ).format(environmentID=environment_id, projectID=project_id)


@pytest.fixture
def custom_predictor_metadata_yaml():
    return dedent(
        """
        name: model-with-custom-java-predictor
        type: inference
        targetType: regression
        customPredictor:
           arbitraryField: This info is read directly by a custom predictor
        """
    )


###############################################################################
# HELPER FUNCS


@pytest.fixture
def df_to_temporary_file() -> callable:
    @contextmanager
    def _to_temporary_file(df: pd.DataFrame, header=True) -> callable:
        is_sparse = is_sparse_dataframe(df)
        suffix = ".mtx" if is_sparse else ".csv"

        target_file = NamedTemporaryFile(suffix=suffix)
        if is_sparse_dataframe(df):
            mmwrite(target_file.name, df.sparse.to_coo())
        else:
            df.to_csv(target_file.name, index=False, header=header)

        yield target_file.name

        target_file.close()

    return _to_temporary_file


##
###############################################################################
# DRUM HELPERS FUNCS


@pytest.fixture
def this_dir():
    return str(Path(__file__).absolute().parent)


@pytest.fixture
def target():
    return "some-target"


@pytest.fixture
def model_metadata_file_factory():
    with TemporaryDirectory(suffix="code-dir") as temp_dirname:

        def _inner(input_dict):
            file_path = Path(temp_dirname) / MODEL_CONFIG_FILENAME
            with file_path.open("w") as fp:
                yaml.dump(input_dict, fp)
            return temp_dirname

        yield _inner


@pytest.fixture
def temp_metadata(environment_id, model_metadata_file_factory):
    metadata = {
        "name": "joe",
        "type": "training",
        "targetType": "regression",
        "environmentID": environment_id,
        "validation": {"input": "hello"},
    }
    yield model_metadata_file_factory(metadata)


@pytest.fixture
def output_dir():
    with TemporaryDirectory(suffix="output-dir") as dir_name:
        yield dir_name


@pytest.fixture
def fit_args(temp_metadata, target, output_dir):
    return [
        "fit",
        "--code-dir",
        temp_metadata,
        "--input",
        __file__,
        "--target",
        target,
        "--target-type",
        "regression",
        "--output",
        output_dir,
    ]


@pytest.fixture
def score_args(this_dir):
    return [
        "score",
        "--code-dir",
        this_dir,
        "--input",
        __file__,
        "--target-type",
        "regression",
        "--language",
        "python",
    ]


@pytest.fixture
def server_args(this_dir):
    return [
        "server",
        "--code-dir",
        this_dir,
        "--address",
        "allthedice.com:1234",
        "--target-type",
        "regression",
        "--language",
        "python",
    ]


def set_sys_argv(cmd_line_args):
    # This is required because the sys.argv is manipulated by the 'CMRunnerArgsRegistry'
    cmd_line_args = cmd_line_args.copy()
    cmd_line_args.insert(0, sys.argv[0])
    sys.argv = cmd_line_args


def get_args_parser_options(cli_command: List[str]):
    set_sys_argv(cli_command)
    arg_parser = CMRunnerArgsRegistry.get_arg_parser()
    CMRunnerArgsRegistry.extend_sys_argv_with_env_vars()
    options = arg_parser.parse_args()
    CMRunnerArgsRegistry.verify_options(options)
    return options


@pytest.fixture
def runtime_factory():
    with DrumRuntime() as cm_runner_runtime:

        def inner(cli_args):
            options = get_args_parser_options(cli_args)
            cm_runner_runtime.options = options
            cm_runner = CMRunner(cm_runner_runtime)
            return cm_runner

        yield inner
