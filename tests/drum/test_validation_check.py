import re
import pytest

from datarobot_drum.drum.common import ArgumentsOptions

from .utils import (
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
)


class TestValidationCheck:
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, DOCKER_PYTHON_SKLEARN),
        ],
    )
    def test_custom_models_validation_test(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_dataset = resources.datasets(framework, problem)

        cmd = "{} validation --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            resources.target_types(problem),
        )
        if problem == BINARY:
            cmd = _cmd_add_class_labels(cmd, resources.class_labels(framework, problem))
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
