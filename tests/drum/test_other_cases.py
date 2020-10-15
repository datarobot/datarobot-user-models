import os
from uuid import uuid4
import pandas as pd
import pytest
import requests

from datarobot_drum.drum.common import ArgumentsOptions, CustomHooks, CUSTOM_FILE_NAME

from .drum_server_utils import DrumServerRun
from .utils import (
    _exec_shell_cmd,
    _create_custom_model_dir,
)


from .constants import (
    SKLEARN,
    RDS,
    CODEGEN,
    CODEGEN_AND_SKLEARN,
    REGRESSION,
    REGRESSION_INFERENCE,
    UNSTRUCTURED,
    BINARY,
    PYTHON,
    NO_CUSTOM,
    PYTHON_ALL_PREDICT_STRUCTURED_HOOKS,
    PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS,
    R,
    R_ALL_PREDICT_STRUCTURED_HOOKS,
    R_ALL_PREDICT_UNSTRUCTURED_HOOKS,
    DOCKER_PYTHON_SKLEARN,
)


class TestOtherCases:
    @pytest.mark.parametrize(
        "framework, problem, language", [(SKLEARN, BINARY, PYTHON), (RDS, BINARY, R)]
    )
    def test_bin_models_with_wrong_labels(
        self,
        resources,
        framework,
        problem,
        language,
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

        if framework == SKLEARN:
            assert (
                str(stdo_stde).find(
                    "Wrong class labels. Use class labels detected by sklearn model"
                )
                != -1
            )
        elif framework == RDS:
            assert (
                str(stdo_stde).find(
                    "Wrong class labels. Use class labels according to your dataset"
                )
                != -1
            )

    # testing negative cases: no artifact, no custom;
    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (None, REGRESSION, NO_CUSTOM),  # no artifact, no custom
            (SKLEARN, REGRESSION, R),  # python artifact, custom.R
            (RDS, REGRESSION, PYTHON),  # R artifact, custom.py
            (None, REGRESSION, R),  # no artifact, custom.R without load_model
            (None, REGRESSION, PYTHON),  # no artifact, custom.py without load_model
        ],
    )
    def test_detect_language(
        self,
        resources,
        framework,
        problem,
        language,
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

        cases_1_2_3 = (
            str(stdo_stde).find("Can not detect language by artifacts and/or custom.py/R files")
            != -1
        )
        case_4 = (
            str(stdo_stde).find(
                "Could not find a serialized model artifact with .rds extension, supported by default R predictor. "
                "If your artifact is not supported by default predictor, implement custom.load_model hook."
            )
            != -1
        )
        case_5 = (
            str(stdo_stde).find(
                "Could not find model artifact file in: {} supported by default predictors".format(
                    custom_model_dir
                )
            )
            != -1
        )
        assert any([cases_1_2_3, case_4, case_5])

    # testing negative cases: no artifact, no custom;
    @pytest.mark.parametrize(
        "framework, problem, language, set_language",
        [
            (SKLEARN, REGRESSION_INFERENCE, R, "python"),  # python artifact, custom.R
            (RDS, REGRESSION, PYTHON, "r"),  # R artifact, custom.py
            (CODEGEN, REGRESSION, PYTHON, "java"),  # java artifact, custom.py
            (
                CODEGEN_AND_SKLEARN,
                REGRESSION,
                NO_CUSTOM,
                "java",
            ),  # java and sklearn artifacts, no custom.py
            (
                CODEGEN_AND_SKLEARN,
                REGRESSION,
                NO_CUSTOM,
                "python",
            ),  # java and sklearn artifacts, no custom.py
            # Negative cases
            (SKLEARN, REGRESSION_INFERENCE, R, None),  # python artifact, custom.R
            (RDS, REGRESSION, PYTHON, None),  # R artifact, custom.py
            (CODEGEN, REGRESSION, PYTHON, None),  # java artifact, custom.py
            (
                CODEGEN_AND_SKLEARN,
                REGRESSION,
                NO_CUSTOM,
                None,
            ),  # java and sklearn artifacts, no custom.py
            (
                CODEGEN_AND_SKLEARN,
                REGRESSION,
                NO_CUSTOM,
                "r",
            ),  # java and sklearn artifacts, no custom.py
        ],
    )
    def test_set_language(
        self,
        resources,
        framework,
        problem,
        language,
        set_language,
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
        "framework, language, hooks_list, target_type",
        [
            (
                None,
                PYTHON_ALL_PREDICT_STRUCTURED_HOOKS,
                CustomHooks.ALL_PREDICT_STRUCTURED,
                REGRESSION,
            ),
            (None, R_ALL_PREDICT_STRUCTURED_HOOKS, CustomHooks.ALL_PREDICT_STRUCTURED, REGRESSION),
            (
                None,
                PYTHON_ALL_PREDICT_UNSTRUCTURED_HOOKS,
                CustomHooks.ALL_PREDICT_UNSTRUCTURED,
                UNSTRUCTURED,
            ),
            (
                None,
                R_ALL_PREDICT_UNSTRUCTURED_HOOKS,
                CustomHooks.ALL_PREDICT_UNSTRUCTURED,
                UNSTRUCTURED,
            ),
        ],
    )
    def test_custom_model_with_all_hooks(
        self,
        resources,
        framework,
        language,
        hooks_list,
        target_type,
        tmp_path,
    ):
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

    @pytest.mark.parametrize("language, language_suffix", [("python", ".py"), ("r", ".R")])
    def test_template_creation(self, language, language_suffix, tmp_path):
        print("Running template creation tests: {}".format(language))
        directory = tmp_path / "template_test_{}".format(uuid4())

        cmd = "{drum_prog} new model --language {language} --code-dir {directory}".format(
            drum_prog=ArgumentsOptions.MAIN_COMMAND, language=language, directory=directory
        )

        _exec_shell_cmd(cmd, "Failed creating a template for custom model, cmd={}".format(cmd))

        assert os.path.isdir(directory), "Directory {} does not exists (or not a dir)".format(
            directory
        )

        assert os.path.isfile(os.path.join(directory, "README.md"))
        custom_file = os.path.join(directory, CUSTOM_FILE_NAME + language_suffix)
        assert os.path.isfile(custom_file)

    def test_r2d2_drum_prediction_server(
        self,
        resources,
        tmp_path,
    ):
        print("current dir: {}".format(os.getcwd()))

        custom_model_dir = "tools/r2d2"

        with DrumServerRun(
            target_type=resources.target_types(REGRESSION_INFERENCE),
            labels=None,
            custom_model_dir=custom_model_dir,
            docker=DOCKER_PYTHON_SKLEARN,
            memory="500m",
            fail_on_shutdown_error=False,
        ) as run:
            print("r2d2 is running")
            cmd = "python tools/r2d2/custom.py memory 200 --server {}".format(run.server_address)
            print(cmd)

            p, stdout, stderr = _exec_shell_cmd(cmd, "Error running r2d2 main")
            print("CMD result: {}".format(p.returncode))
            print(stdout)
            print(stderr)
            assert p.returncode == 0

            data = pd.DataFrame(
                {"cmd": ["memory"], "arg": [100]},
                columns=["cmd", "arg"],
            )
            print("Sending the following data:")
            print(data)

            csv_data = data.to_csv(index=False)
            url = "{}/predict/".format(run.url_server_address)
            response = requests.post(url, files={"X": csv_data})
            print(response)
            assert response.ok

            # Sending the exception command.. should get a failed response
            data = pd.DataFrame(
                {"cmd": ["exception"], "arg": [100]},
                columns=["cmd", "arg"],
            )
            print("Sending the following data:")
            print(data)

            csv_data = data.to_csv(index=False)
            response = requests.post(url, files={"X": csv_data})
            print(response)
            assert response.status_code == 500

            # Server should be alive before we kill it with memory
            response = requests.get(run.url_server_address)
            print(response)
            assert response.ok

            # Killing the docker allocating too much memory
            data = pd.DataFrame(
                {"cmd": ["memory"], "arg": [1000]},
                columns=["cmd", "arg"],
            )

            print("Sending 1000m data:")
            print(data)
            csv_data = data.to_csv(index=False)

            try:
                response = requests.post(url, files={"X": csv_data})
                print(response)
                assert response.status_code == 500
            except Exception:
                print("Expected connection error")
