import pytest
import re
from textwrap import dedent


from datarobot_drum.drum.common import (
    ArgumentsOptions,
)

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
)


class TestInference:
    @pytest.mark.parametrize(
        "docker_build_fails",
        [True, False],
    )
    def test_docker_image_creation(
        self,
        docker_build_fails,
        tmp_path,
    ):
        # py-slim image is used in another test container, so it is already expected to be in the registry
        dockerfile = dedent(
            """
        FROM python:3.7-slim
        VOLUME /data
        """
        )

        if docker_build_fails:
            dockerfile += "\nRUN pip install datarobot-drum==1.1.111"

        custom_model_dir = tmp_path / "custom_model"
        custom_model_dir.mkdir(parents=True, exist_ok=True)

        input_dataset = tmp_path / "fake_but_existing_input_file"
        input_dataset.touch()

        docker_context_dir_name_to_be_used_as_a_tag = "docker_image_built_by_drum"
        docker_context_path = tmp_path / docker_context_dir_name_to_be_used_as_a_tag
        docker_context_path.mkdir(parents=True, exist_ok=True)

        dockerfile_path = docker_context_path / "Dockerfile"
        with open(dockerfile_path, mode="w") as f:
            f.write(dockerfile)

        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {} --target-type regression --docker {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            docker_context_path,
        )

        _, stdo, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        if docker_build_fails:
            assert re.search(
                r"Could not find a version that satisfies the requirement datarobot-drum==1.1.111",
                stdo,
            )
        else:
            assert re.search(
                r"Image successfully built; tag: {};".format(
                    docker_context_dir_name_to_be_used_as_a_tag
                ),
                stdo,
            )
