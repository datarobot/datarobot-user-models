"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
from uuid import uuid4
import pandas as pd
import pytest
import requests
from unittest.mock import patch, DEFAULT
import yaml

from datarobot_drum.drum.enum import CUSTOM_FILE_NAME, ArgumentsOptions

from datarobot_drum.resource.drum_server_utils import DrumServerRun
from datarobot_drum.resource.utils import (
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
    BINARY,
    PYTHON,
    NO_CUSTOM,
    R,
    DOCKER_PYTHON_SKLEARN,
    TESTS_DATA_PATH,
)
from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME
from datarobot_drum.drum.enum import ModelMetadataKeys
from datarobot_drum.drum.enum import ModelMetadataHyperParamTypes
from datarobot_drum.drum.enum import RunMode
from custom_model_runner.datarobot_drum.drum.drum import CMRunner
from custom_model_runner.datarobot_drum.drum.main import main
from datarobot_drum.drum.runtime import DrumRuntime
from argparse import Namespace
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.exceptions import DrumSchemaValidationException
from datarobot_drum.drum.enum import ExitCodes


class TestOtherCases:
    # testing negative cases: no artifact, no custom;
    # R version of current test cases is run in the dedicated env as rpy2 installation is required
    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (None, REGRESSION, NO_CUSTOM),  # no artifact, no custom
            (SKLEARN, REGRESSION, R),  # python artifact, custom.R
            (RDS, REGRESSION, PYTHON),  # R artifact, custom.py
            (None, REGRESSION, PYTHON),  # no artifact, custom.py without load_model
        ],
    )
    def test_detect_language(
        self, resources, framework, problem, language, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
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
                "Could not find model artifact file in: {} supported by default predictors".format(
                    custom_model_dir
                )
            )
            != -1
        )
        assert any([cases_1_2_3, case_4])

    # testing negative cases: no artifact, no custom;
    # R version of current test cases is run in the dedicated env as rpy2 installation is required
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
        ],
    )
    def test_set_language(
        self, resources, framework, problem, language, set_language, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
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
        self, resources, tmp_path,
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

            data = pd.DataFrame({"cmd": ["memory"], "arg": [100]}, columns=["cmd", "arg"],)
            print("Sending the following data:")
            print(data)

            csv_data = data.to_csv(index=False)
            url = "{}/predict/".format(run.url_server_address)
            response = requests.post(url, files={"X": csv_data})
            print(response)
            assert response.ok

            # Sending the exception command.. should get a failed response
            data = pd.DataFrame({"cmd": ["exception"], "arg": [100]}, columns=["cmd", "arg"],)
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
            data = pd.DataFrame({"cmd": ["memory"], "arg": [1000]}, columns=["cmd", "arg"],)

            print("Sending 1000m data:")
            print(data)
            csv_data = data.to_csv(index=False)

            try:
                response = requests.post(url, files={"X": csv_data})
                print(response)
                assert response.status_code == 500
            except Exception:
                print("Expected connection error")

    @pytest.mark.parametrize(
        "target_type", ["binary", "regression", "unstructured", "anomaly", "multiclass"]
    )
    def test_model_metadata_validation_fails__when_output_requirement_not_allowed_for_selected_target_types(
        self, target_type,
    ):
        """The output_requirements of model metadata defines the specs of custom task outputs. It only applies to
        transform custom task. Before testing logic in CMRunner.run_fit() is executed, the model metadata will be first
        validated. Exception will be raised when output_requirements is defined in a non-transform task.
        """
        test_data_path = os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv")
        with DrumRuntime() as runtime:
            runtime_options = Namespace(
                input=test_data_path,
                code_dir="",
                negative_class_label=None,
                positive_class_label=None,
                class_labels=None,
                target_csv=None,
                target="Species",
                row_weights=None,
                row_weights_csv=None,
                output=None,
                num_rows=0,
                sparse_column_file=None,
                parameter_file=None,
                disable_strict_validation=False,
                logging_level="warning",
                subparser_name=RunMode.FIT,
                target_type=target_type,
                verbose=False,
                content_type=None,
            )
            with patch(
                "custom_model_runner.datarobot_drum.drum.drum.read_model_metadata_yaml"
            ) as mock_model_metadata:
                mock_model_metadata.return_value = {
                    "name": "name",
                    "type": "training",
                    "targetType": target_type,
                    "typeSchema": {
                        "input_requirements": [
                            {"field": "data_types", "condition": "IN", "value": "NUM"},
                        ],
                        "output_requirements": [
                            {"field": "data_types", "condition": "IN", "value": "CAT"},
                        ],
                    },
                }
                runtime.options = runtime_options
                with pytest.raises(DrumSchemaValidationException) as ex:
                    CMRunner(runtime).run()
                assert (
                    "Specifying output_requirements in model_metadata.yaml is only valid for custom transform tasks."
                    in str(ex.value)
                )

    def test_drum_exits_code_when_custom_task_schema_validation_exception_is_raised(self):
        runtime_options = Namespace(
            code_dir="",
            disable_strict_validation=False,
            logging_level="warning",
            subparser_name=RunMode.FIT.value,
            target_type="regression",
            verbose=False,
            content_type=None,
        )
        from datarobot_drum.drum.drum import CMRunner

        with patch.multiple(
            CMRunnerArgsRegistry,
            get_arg_parser=DEFAULT,
            extend_sys_argv_with_env_vars=DEFAULT,
            verify_options=DEFAULT,
        ) as mock_cmrunner_args_registry:
            mock_arg_registry = mock_cmrunner_args_registry["get_arg_parser"].return_value
            mock_arg_registry.parse_args.return_value = runtime_options
            with patch.object(CMRunner, "run",) as mock_run:
                mock_run.side_effect = DrumSchemaValidationException()
                with pytest.raises(SystemExit) as ex:
                    main()
                assert ex.value.code == ExitCodes.SCHEMA_VALIDATION_ERROR.value

    def test_model_metadata_validation_fails__with_hyper_parameters_errors(self, tmp_path):
        with DrumRuntime() as runtime:
            runtime_options = Namespace(
                code_dir=tmp_path,
                disable_strict_validation=False,
                logging_level="warning",
                subparser_name=RunMode.FIT,
                target_type="regression",
                verbose=False,
                content_type=None,
            )
            model_templates = {
                ModelMetadataKeys.NAME: "name",
                ModelMetadataKeys.TYPE: "training",
                ModelMetadataKeys.TARGET_TYPE: "regression",
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {
                        "name": "param_int",
                        "type": ModelMetadataHyperParamTypes.INT,
                        "min": 1,
                        "max": 0,
                    },
                ],
            }
            with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
                yaml.dump(model_templates, f)
            runtime.options = runtime_options
            with pytest.raises(
                DrumCommonException,
                match="Invalid int parameter param_int: max must be greater than min",
            ) as ex:
                CMRunner(runtime)
