"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd
import pytest

from datarobot_drum.drum.enum import CustomHooks, ArgumentsOptions

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
    _create_custom_model_dir,
    _cmd_add_class_labels,
)

from .constants import (
    CODEGEN_AND_SKLEARN,
    SKLEARN,
    SKLEARN_NO_ARTIFACTS,
    RDS,
    REGRESSION,
    UNSTRUCTURED,
    BINARY,
    PYTHON,
    PYTHON_ALL_PREDICT_STRUCTURED_HOOKS,
    PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS,
    R_NO_ARTIFACTS,
    R,
    R_ALL_PREDICT_STRUCTURED_HOOKS,
    R_ALL_PREDICT_STRUCTURED_HOOKS_LOWERCASE_R,
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS,
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS_LOWERCASE_R,
    R_INT_COLNAMES_BINARY,
    R_INT_COLNAMES_MULTICLASS,
    MULTICLASS,
    NO_CUSTOM,
    R_MULTI_ARTIFACT_NEGATIVE,
)

from tests.conftest import skip_if_framework_not_in_env


class TestOtherCasesPerFramework:
    @pytest.mark.parametrize(
        "framework, problem, language", [(SKLEARN, BINARY, PYTHON), (RDS, BINARY, R)]
    )
    def test_bin_models_with_wrong_labels(
        self, resources, framework, problem, language, tmp_path, framework_env
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_dataset = resources.datasets(framework, problem)
        cmd = "{} score --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            resources.target_types(problem),
        )
        if problem == BINARY:
            cmd = cmd + " --positive-class-label yes --negative-class-label no"

        p, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        stdo_stde = str(stdo) + str(stde)

        assert "Expected predictions to have columns ['yes', 'no']" in stdo_stde

    @pytest.mark.parametrize(
        "framework, language, hooks_list, target_type",
        [
            (
                SKLEARN_NO_ARTIFACTS,
                PYTHON_ALL_PREDICT_STRUCTURED_HOOKS,
                CustomHooks.ALL_PREDICT_STRUCTURED,
                REGRESSION,
            ),
            (
                R_NO_ARTIFACTS,
                R_ALL_PREDICT_STRUCTURED_HOOKS,
                CustomHooks.ALL_PREDICT_STRUCTURED,
                REGRESSION,
            ),
            (
                R_NO_ARTIFACTS,
                R_ALL_PREDICT_STRUCTURED_HOOKS_LOWERCASE_R,
                CustomHooks.ALL_PREDICT_STRUCTURED,
                REGRESSION,
            ),
            (
                SKLEARN_NO_ARTIFACTS,
                PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS,
                CustomHooks.ALL_PREDICT_UNSTRUCTURED,
                UNSTRUCTURED,
            ),
            (
                R_NO_ARTIFACTS,
                R_ALL_PREDICT_UNSTRUCTURED_HOOKS,
                CustomHooks.ALL_PREDICT_UNSTRUCTURED,
                UNSTRUCTURED,
            ),
            (
                R_NO_ARTIFACTS,
                R_ALL_PREDICT_UNSTRUCTURED_HOOKS_LOWERCASE_R,
                CustomHooks.ALL_PREDICT_UNSTRUCTURED,
                UNSTRUCTURED,
            ),
        ],
    )
    def test_custom_model_with_all_hooks(
        self, resources, framework, language, hooks_list, target_type, tmp_path, framework_env
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            None,
            language,
        )

        input_dataset = resources.datasets(framework, REGRESSION)

        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output, target_type
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        if hooks_list == CustomHooks.ALL_PREDICT_STRUCTURED:
            preds = pd.read_csv(output)
            assert all(val for val in (preds["Predictions"] == len(hooks_list)).values), preds
        elif hooks_list == CustomHooks.ALL_PREDICT_UNSTRUCTURED:
            with open(output) as f:
                all_data = f.read()
                assert str(len(hooks_list)) in all_data

    @pytest.mark.parametrize(
        "framework, language, hooks_list, target_type",
        [
            (
                R_NO_ARTIFACTS,
                R_INT_COLNAMES_BINARY,
                CustomHooks.SCORE,
                BINARY,
            ),
            (
                R_NO_ARTIFACTS,
                R_INT_COLNAMES_MULTICLASS,
                CustomHooks.SCORE,
                MULTICLASS,
            ),
        ],
    )
    @pytest.mark.parametrize("label_type", [int, float])
    def test_custom_model_R_int_colnames_in_prediction_output(
        self,
        resources,
        framework,
        language,
        hooks_list,
        target_type,
        label_type,
        tmp_path,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)

        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            None,
            language,
        )
        input_dataset = resources.datasets(framework, REGRESSION)
        output = tmp_path / "output"

        labels = [0, 1]
        if target_type == MULTICLASS:
            labels = [0, 1, 2]

        labels = [label_type(l) for l in labels]

        cmd = "{} score --code-dir {} --input {} --output {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output, target_type
        )
        cmd = _cmd_add_class_labels(cmd, labels, target_type=target_type)

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

        preds = pd.read_csv(output)
        assert all(preds.columns == [str(l) for l in labels])

    # R version of current test cases is run in the dedicated env as rpy2 installation is required
    @pytest.mark.parametrize(
        "framework, language, target_type",
        [
            (R_NO_ARTIFACTS, R, REGRESSION),
        ],  # no artifact, custom.R without load_model
    )
    def test_detect_language(
        self, resources, framework, language, tmp_path, framework_env, target_type
    ):
        skip_if_framework_not_in_env(framework, framework_env)

        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            None,
            language,
        )

        input_dataset = resources.datasets(framework, REGRESSION)
        cmd = "{} score --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            target_type,
        )

        p, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        stdo_stde = str(stdo) + str(stde)

        case_4 = (
            str(stdo_stde).find(
                "Could not find a serialized model artifact with .rds extension, supported by default R predictor. "
                "If your artifact is not supported by default predictor, implement custom.load_model hook."
            )
            != -1
        )

        assert any([case_4])

    # R version of current test cases is run in the dedicated env as rpy2 installation is required
    @pytest.mark.parametrize(
        "framework, problem, language, set_language",
        [
            (
                CODEGEN_AND_SKLEARN,
                REGRESSION,
                NO_CUSTOM,
                "r",
            ),  # java and sklearn artifacts, no custom.py
        ],
    )
    def test_set_language(
        self, resources, framework, problem, language, set_language, tmp_path, framework_env
    ):
        skip_if_framework_not_in_env("rds", framework_env)

        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )
        input_dataset = resources.datasets(framework, problem)
        cmd = "{} score --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            resources.target_types(problem),
        )
        if set_language:
            cmd += " --language {}".format(set_language)
        if problem == BINARY:
            cmd += " --positive-class-label yes --negative-class-label no"

        p, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        if not set_language:
            stdo_stde = str(stdo) + str(stde)
            cases_4_5_6_7 = (
                str(stdo_stde).find("Can not detect language by artifacts and/or custom.py/R files")
                != -1
            )
            assert cases_4_5_6_7
        if framework == CODEGEN_AND_SKLEARN and set_language == "r":
            stdo_stde = str(stdo) + str(stde)
            case = (
                str(stdo_stde).find(
                    "Could not find a serialized model artifact with .rds extension, supported by default R predictor. "
                    "If your artifact is not supported by default predictor, implement custom.load_model hook."
                )
                != -1
            )
            assert case

    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (
                R_MULTI_ARTIFACT_NEGATIVE,
                None,
                NO_CUSTOM,
            ),
        ],
    )
    def test_multiple_r_artifacts_negative(
        self, resources, framework, problem, language, tmp_path, framework_env
    ):
        skip_if_framework_not_in_env(framework, framework_env)

        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )
        input_dataset = resources.datasets(framework, REGRESSION)
        cmd = "{} score --code-dir {} --input {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            REGRESSION,
        )

        p, stdo, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        stdo_stde = str(stdo) + str(stde)
        match = (
            str(stdo_stde).find(
                "Multiple serialized model artifacts found: [r_multi.rds r_reg.rds]"
            )
            != -1
        )
        assert match
