"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import tempfile
from contextlib import contextmanager
from textwrap import dedent

import pytest
from tempfile import NamedTemporaryFile

import pandas as pd
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

        target_file = tempfile.NamedTemporaryFile(suffix=suffix)
        if is_sparse_dataframe(df):
            mmwrite(target_file.name, df.sparse.to_coo())
        else:
            df.to_csv(target_file.name, index=False, header=header)

        yield target_file.name

        target_file.close()

    return _to_temporary_file
