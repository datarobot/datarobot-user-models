import pytest

from datarobot_drum.drum.common import ArgumentsOptions
from datarobot_drum.resource.utils import (
    _cmd_add_class_labels,
    _create_custom_model_dir,
    _exec_shell_cmd,
)
from tests.drum.test_fit import TestFit
from .constants import (
    ANOMALY,
    BINARY,
    MULTICLASS,
    PYTHON,
    R_FIT,
    REGRESSION,
    SKLEARN_TRANSFORM,
    SKLEARN_TRANSFORM_NO_HOOK,
    SKLEARN_TRANSFORM_NON_NUMERIC,
    SKLEARN_TRANSFORM_SPARSE_IN_OUT,
    SKLEARN_TRANSFORM_SPARSE_INPUT,
    SKLEARN_TRANSFORM_WITH_Y,
    SPARSE,
    SPARSE_COLUMNS,
    SPARSE_TARGET,
    TRANSFORM,
    WEIGHTS_ARGS,
    WEIGHTS_CSV,
)


class TestFitTransform(TestFit):
    @pytest.mark.parametrize(
        "framework",
        [
            SKLEARN_TRANSFORM,
            SKLEARN_TRANSFORM_WITH_Y,
            SKLEARN_TRANSFORM_NO_HOOK,
            SKLEARN_TRANSFORM_NON_NUMERIC,
        ],
    )
    @pytest.mark.parametrize("problem", [REGRESSION, BINARY, ANOMALY])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    def test_fit_transform(
        self, resources, framework, problem, weights, tmp_path,
    ):
        language = PYTHON
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language=framework,
        )

        input_dataset = resources.datasets(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_dataset, r_fit=language == R_FIT
        )

        target_type = TRANSFORM

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, target_type, custom_model_dir, input_dataset
        )
        if problem != ANOMALY:
            cmd += " --target {}".format(resources.targets(problem))

        if problem in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd, resources.class_labels(framework, problem), target_type=target_type
            )

        cmd += weights_cmd

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    @pytest.mark.parametrize(
        "framework", [SKLEARN_TRANSFORM_SPARSE_IN_OUT, SKLEARN_TRANSFORM_SPARSE_INPUT,],
    )
    def test_sparse_fit_transform(
        self, framework, resources, tmp_path,
    ):
        input_dataset = resources.datasets(None, SPARSE)
        target_dataset = resources.datasets(None, SPARSE_TARGET)

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, REGRESSION, language=framework,
        )
        columns = resources.datasets(framework, SPARSE_COLUMNS)

        cmd = "{} fit --target-type {} --code-dir {} --input {} --verbose --target-csv {} --sparse-column-file {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            TRANSFORM,
            custom_model_dir,
            input_dataset,
            target_dataset,
            columns,
        )

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
