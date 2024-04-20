"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import pytest
import re
import shutil
from textwrap import dedent

from datarobot_drum.drum.enum import ArgumentsOptions

from datarobot_drum.resource.utils import (
    _cmd_add_class_labels,
    _exec_shell_cmd,
)

from tests.constants import (
    CODEGEN,
    UNSTRUCTURED,
    PYTORCH,
    REGRESSION,
    MULTICLASS,
    MODEL_TEMPLATES_PATH,
)


@pytest.fixture
def dockerfile_content():
    # py-slim image is used in another test container, so it is already expected to be in the registry
    content = dedent(
        """
    FROM python:3.11-slim
    VOLUME /data
    """
    )
    return content


@pytest.fixture
def docker_context_factory(tmp_path):
    # py-slim image is used in another test container, so it is already expected to be in the registry

    def _inner(dockerfile_content):
        docker_context_dir_name_to_be_used_as_a_tag = "docker_image_built_by_drum"
        docker_context_path = tmp_path / docker_context_dir_name_to_be_used_as_a_tag
        tag = "{}/{}".format(
            os.path.basename(tmp_path), docker_context_dir_name_to_be_used_as_a_tag
        ).lower()
        docker_context_path.mkdir(parents=True, exist_ok=True)

        dockerfile_path = docker_context_path / "Dockerfile"
        with open(dockerfile_path, mode="w") as f:
            f.write(dockerfile_content)

        return docker_context_path, tag

    return _inner


@pytest.mark.usefixtures("skip_if_running_inside_container")
class TestDrumDocker:
    @pytest.mark.parametrize(
        "docker_build_fails",
        [True, False],
    )
    def test_docker_image_creation(
        self,
        dockerfile_content,
        docker_context_factory,
        docker_build_fails,
        tmp_path,
    ):
        dockerfile = dockerfile_content

        if docker_build_fails:
            dockerfile += "\nRUN pip install datarobot-drum==1.1.111"

        docker_context_path, expected_tag = docker_context_factory(dockerfile)

        custom_model_dir = tmp_path / "custom_model"
        custom_model_dir.mkdir(parents=True, exist_ok=True)

        output = tmp_path / "output"

        input_dataset = tmp_path / "fake_but_existing_input_file"
        input_dataset.touch()

        cmd = "{} score --code-dir {} --input {} --output {} --target-type regression --docker {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            docker_context_path,
        )

        # In docker mode we try to resolve language.
        # As code dir is a fake, explicitly provide language.
        cmd += " --language python"

        _, stdout, stderr = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        if docker_build_fails:
            assert re.search(
                r"ERROR drum:  Failed to build a docker image",
                stderr,
            )

            assert re.search(
                r"Could not find a version that satisfies the requirement datarobot-drum==1.1.111",
                stderr,
            )
        else:
            assert re.search(r"Image successfully built; tag: {};".format(expected_tag), stdout)

    @pytest.mark.parametrize(
        "framework, problem, code_dir, skip_deps_install",
        [
            (CODEGEN, REGRESSION, "java_codegen", False),
            (CODEGEN, REGRESSION, "java_codegen", True),
        ],
    )
    def test_java_docker_image_with_deps_install(
        self,
        resources,
        framework,
        problem,
        code_dir,
        skip_deps_install,
        tmp_path,
        dockerfile_content,
        docker_context_factory,
    ):
        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, code_dir)
        if framework == CODEGEN and not skip_deps_install:
            tmp_dir = tmp_path / "tmp_code_dir"
            custom_model_dir = shutil.copytree(custom_model_dir, tmp_dir)
            with open(os.path.join(custom_model_dir, "requirements.txt"), mode="w") as f:
                f.write("deps_are_not_supported_in_java")
        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {} --target-type {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )

        docker_context_path, expected_tag = docker_context_factory(dockerfile_content)
        cmd += " --docker {} --verbose ".format(docker_context_path)

        if skip_deps_install:
            cmd += " --skip-deps-install"

        _, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        if skip_deps_install:
            # DRUM score fails, as docker is not java, but we are only interested that image is built
            assert re.search(r"Image successfully built; tag: {};".format(expected_tag), stdo)
        else:
            assert re.search(
                r"WARNING drum:  Dependencies management is not supported for the 'java' language and will not be installed into an image",
                stde,
            )

    @pytest.mark.parametrize(
        "framework, problem, code_dir, skip_deps_install",
        [
            (PYTORCH, MULTICLASS, "python3_pytorch_multiclass", False),
            (PYTORCH, MULTICLASS, "python3_pytorch_multiclass", True),
        ],
    )
    def test_python_docker_image_with_deps_install(
        self,
        resources,
        framework,
        problem,
        code_dir,
        skip_deps_install,
        tmp_path,
        dockerfile_content,
        docker_context_factory,
    ):
        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, code_dir)

        tmp_dir = tmp_path / "tmp_code_dir"
        custom_model_dir = shutil.copytree(custom_model_dir, tmp_dir)
        with open(os.path.join(custom_model_dir, "requirements.txt"), mode="w") as f:
            if skip_deps_install:
                f.write("build_will_fail_if_deps_install_not_skipped")
            else:
                f.write("scipy>=1.1,<2")  # test that adding a min,max version also works correctly
        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {} --target-type {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )

        cmd = _cmd_add_class_labels(
            cmd,
            resources.class_labels(framework, problem),
            target_type=resources.target_types(problem),
            multiclass_label_file=None,
        )

        docker_context_path, expected_tag = docker_context_factory(dockerfile_content)
        cmd += " --docker {} --verbose ".format(docker_context_path)

        if skip_deps_install:
            cmd += " --skip-deps-install"

        _, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        assert re.search(r"Image successfully built; tag: {};".format(expected_tag), stdo)

    @pytest.mark.parametrize(
        "framework, problem, code_dir, skip_deps_install",
        [
            (None, UNSTRUCTURED, "r_unstructured", False),
            (None, UNSTRUCTURED, "r_unstructured", True),
        ],
    )
    def test_r_lang_docker_image_with_deps_install(
        self,
        resources,
        framework,
        problem,
        code_dir,
        skip_deps_install,
        tmp_path,
        dockerfile_content,
        docker_context_factory,
    ):
        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, code_dir)

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {} --target-type {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )

        docker_context_path, _ = docker_context_factory(dockerfile_content)
        cmd += " --docker {} --verbose ".format(docker_context_path)

        if skip_deps_install:
            cmd += " --skip-deps-install"

        _, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        if skip_deps_install:
            assert not re.search(r"withCallingHandlers", stde + stdo)
            assert not re.search(r"http://cran.rstudio.com/", stde + stdo)
        else:
            assert re.search(r"withCallingHandlers", stde)
            assert re.search(r"http://cran.rstudio.com/", stde)

    @pytest.mark.parametrize(
        "framework, problem, code_dir",
        [(PYTORCH, MULTICLASS, "python3_pytorch_multiclass")],
    )
    def test_docker_image_with_wrong_dep_install(
        self,
        resources,
        framework,
        problem,
        code_dir,
        tmp_path,
        dockerfile_content,
        docker_context_factory,
    ):
        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, code_dir)

        tmp_dir = tmp_path / "tmp_code_dir"
        custom_model_dir = shutil.copytree(custom_model_dir, tmp_dir)
        with open(os.path.join(custom_model_dir, "requirements.txt"), mode="w") as f:
            f.write("\nnon_existing_dep")

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {} --target-type {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )
        cmd = _cmd_add_class_labels(
            cmd,
            resources.class_labels(framework, problem),
            target_type=resources.target_types(problem),
            multiclass_label_file=None,
        )

        # environment may be not real as it is expected to fail.
        docker_path, _ = docker_context_factory(dockerfile_content)
        cmd += " --docker {} --verbose ".format(docker_path)

        _, _, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        assert re.search(
            r"ERROR drum:  Failed to build a docker image",
            stde,
        )

        assert re.search(
            r"Could not find a version that satisfies the requirement non_existing_dep",
            stde,
        )
