"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import pandas as pd
import pytest
import re
import shutil
from textwrap import dedent

from tempfile import NamedTemporaryFile

from datarobot_drum.drum.enum import ArgumentsOptions

from datarobot_drum.resource.utils import (
    _cmd_add_class_labels,
    _exec_shell_cmd,
)

from tests.drum.constants import (
    BINARY,
    CODEGEN,
    UNSTRUCTURED,
    PYTORCH,
    REGRESSION,
    MULTICLASS,
    MODEL_TEMPLATES_PATH,
    PUBLIC_DROPIN_ENVS_PATH,
)


class TestDrumDocker:
    @pytest.mark.parametrize(
        "docker_build_fails", [True, False],
    )
    def test_docker_image_creation(
        self, docker_build_fails, tmp_path,
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
        expected_tag = "{}/{}".format(
            os.path.basename(tmp_path), docker_context_dir_name_to_be_used_as_a_tag
        ).lower()
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

        # In docker mode we try to resolve language.
        # As code dir is a fake, explicitly provide language.
        cmd += " --language python"

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
            assert re.search(r"Image successfully built; tag: {};".format(expected_tag), stdo,)

    @pytest.mark.parametrize(
        "framework, problem, code_dir, env_dir, skip_deps_install",
        [
            (PYTORCH, MULTICLASS, "python3_pytorch_multiclass", "python3_sklearn", False),
            (PYTORCH, MULTICLASS, "python3_pytorch_multiclass", "python3_sklearn", True),
            (None, UNSTRUCTURED, "r_unstructured", "r_lang", False),
            (None, UNSTRUCTURED, "r_unstructured", "r_lang", True),
            (CODEGEN, REGRESSION, "java_codegen", "java_codegen", False),
            (CODEGEN, REGRESSION, "java_codegen", "java_codegen", True),
        ],
    )
    def test_docker_image_with_deps_install(
        self, resources, framework, problem, code_dir, env_dir, skip_deps_install, tmp_path,
    ):

        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, code_dir)
        if framework == CODEGEN and not skip_deps_install:
            tmp_dir = tmp_path / "tmp_code_dir"
            custom_model_dir = shutil.copytree(custom_model_dir, tmp_dir)
            with open(os.path.join(custom_model_dir, "requirements.txt"), mode="w") as f:
                f.write("deps_are_not_supported_in_java")
        docker_env = os.path.join(PUBLIC_DROPIN_ENVS_PATH, env_dir)
        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = '{} score --code-dir {} --input "{}" --output {} --target-type {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )
        if resources.target_types(problem) in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, problem),
                target_type=resources.target_types(problem),
                multiclass_label_file=None,
            )
        cmd += " --docker {} --verbose ".format(docker_env)

        if skip_deps_install:
            cmd += " --skip-deps-install"

        _, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        if skip_deps_install:
            # requirements.txt is not supported for java models, so test should pass
            if framework == CODEGEN:
                in_data = pd.read_csv(input_dataset)
                out_data = pd.read_csv(output)
                assert in_data.shape[0] == out_data.shape[0]
            else:
                assert re.search(r"ERROR drum:  Error from docker process:", stde,)
        else:
            if framework is None and problem == UNSTRUCTURED:
                with open(output) as f:
                    out_data = f.read()
                    assert "10" in out_data
            elif framework == PYTORCH and problem == MULTICLASS:
                in_data = pd.read_csv(input_dataset)
                out_data = pd.read_csv(output)
                assert in_data.shape[0] == out_data.shape[0]
            elif framework == CODEGEN and problem == REGRESSION:
                assert re.search(
                    r"WARNING drum:  Dependencies management is not supported for the 'java' language and will not be installed into an image",
                    stde,
                )
            else:
                assert False

    @pytest.mark.parametrize(
        "framework, problem, code_dir, env_dir",
        [(PYTORCH, MULTICLASS, "python3_pytorch_multiclass", "python3_pytorch"),],
    )
    def test_docker_image_with_wrong_dep_install(
        self, resources, framework, problem, code_dir, env_dir, tmp_path,
    ):

        custom_model_dir = os.path.join(MODEL_TEMPLATES_PATH, code_dir)

        tmp_dir = tmp_path / "tmp_code_dir"
        custom_model_dir = shutil.copytree(custom_model_dir, tmp_dir)
        with open(os.path.join(custom_model_dir, "requirements.txt"), mode="w") as f:
            f.write("\nnon_existing_dep")

        docker_env = os.path.join(PUBLIC_DROPIN_ENVS_PATH, env_dir)
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
        cmd += " --docker {} --verbose ".format(docker_env)

        _, _, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        assert re.search(r"ERROR drum:  Failed to build a docker image", stde,)

        assert re.search(
            r"Could not find a version that satisfies the requirement non_existing_dep", stde,
        )
