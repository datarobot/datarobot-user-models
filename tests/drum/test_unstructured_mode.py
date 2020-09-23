import pytest

from datarobot_drum.drum.common import ArgumentsOptions

from .utils import (
    _exec_shell_cmd,
    _create_custom_model_dir,
)

from .constants import (
    PYTHON_UNSTRUCTURED,
    R_UNSTRUCTURED,
    UNSTRUCTURED,
    WORDS_COUNT_BASIC,
)


class TestUnstructuredMode:
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (UNSTRUCTURED, WORDS_COUNT_BASIC, PYTHON_UNSTRUCTURED, None),
            (UNSTRUCTURED, WORDS_COUNT_BASIC, R_UNSTRUCTURED, None),
        ],
    )
    def test_unstructured_models_batch(
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

        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {} --unstructured".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output
        )

        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        with open(output) as f:
            out_data = f.read()
            assert "6" in out_data
