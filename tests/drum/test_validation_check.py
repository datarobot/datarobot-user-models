import re
import pytest
import pandas as pd
from tempfile import NamedTemporaryFile

from datarobot_drum.drum.common import ArgumentsOptions

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
    _cmd_add_class_labels,
    _create_custom_model_dir,
)


from .constants import (
    SKLEARN,
    REGRESSION,
    REGRESSION_INFERENCE,
    BINARY,
    PYTHON,
    NO_CUSTOM,
    DOCKER_PYTHON_SKLEARN,
    PYTHON_NO_ARTIFACT_REGRESSION_HOOKS,
)


class TestValidationCheck:
    @pytest.mark.parametrize(
        "framework, problem, language", [(None, REGRESSION, PYTHON_NO_ARTIFACT_REGRESSION_HOOKS),],
    )
    def test_validation_check_with_bad_column_names(
        self, resources, framework, problem, language, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        column_names = [
            "column",
            "col/unm",
            "col\\unm",
            'col"umn',
            "col umn",
            "col:umn",
            'col""umn',
        ]
        d = {col: [1.0] for col in column_names}
        df = pd.DataFrame(data=d)

        with NamedTemporaryFile(mode="w") as temp_f:
            df.to_csv(temp_f.name)

            input_dataset = temp_f.name

            cmd = "{} validation --code-dir {} --input {} --target-type {}".format(
                ArgumentsOptions.MAIN_COMMAND,
                custom_model_dir,
                input_dataset,
                resources.target_types(problem),
            )

            _, stdo, _ = _exec_shell_cmd(
                cmd,
                "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
                assert_if_fail=False,
            )

            assert re.search(r"Null value imputation\s+PASSED", stdo)

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, DOCKER_PYTHON_SKLEARN),
        ],
    )
    def test_validation_check(
        self, resources, framework, problem, language, docker, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        input_dataset = resources.datasets(framework, problem)

        cmd = "{} validation --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            resources.target_types(problem),
        )
        if problem == BINARY:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, problem),
                target_type=resources.target_types(problem),
            )
        if docker:
            cmd += " --docker {}".format(docker)

        _, stdo, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        if language == NO_CUSTOM:
            assert re.search(r"Null value imputation\s+FAILED", stdo)
        else:
            assert re.search(r"Null value imputation\s+PASSED", stdo)
