#!/usr/bin/env python3

import glob
import json
import logging
import os
import shutil
import subprocess
import sys

from mlpiper.pipeline.components_desc import ComponentScanner
from mlpiper.cli.mlpiper_exception import MLPiperException
from mlpiper.pipeline import json_fields
from mlpiper.common.base import Base
from mlpiper.pipeline import java_mapping
from mlpiper.pipeline.executor import Executor
from mlpiper.pipeline.component_language import ComponentLanguage


class MLPiperRunner(Base):
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    COMP_DIR = "components"
    COMPONENTS_SETUP_PY = "mcenter_components_setup.py"

    MLPIPER_SCRIPT = "mlpiper"
    DEPLOYMENT_PIPELINE = "pipeline.json"

    def __init__(self, options):
        super(MLPiperRunner, self).__init__()
        self.set_logger(logging.getLogger(self.logger_name()))
        self._comp_scanner = None
        self._comp_repo_path = None
        self._comp_root_path = None
        self._deploy_dir = None
        self._in_place = False
        self._bin_dir = None
        self._pipeline_dict = None
        self._use_color = True
        self._engine = None
        self._options = options
        self._mlpiper_jar = None

        self._skip_mlpiper_deps = False

        self._input_model_filepath = None
        self._output_model_filepath = None
        self._force = False
        self._test_mode = False

    def _print_info(self):
        self._logger.info("Deployment path: {}".format(self._deploy_dir))
        self._logger.info("Components root path: {}".format(self._comp_root_path))
        self._logger.info("bin_dir: {}".format(self._bin_dir))

    def comp_repo(self, comp_repo_path):
        self._comp_repo_path = comp_repo_path
        self._comp_scanner = ComponentScanner(comp_repo_path)
        return self

    def mlpiper_jar(self, mlpiper_jar):
        self._mlpiper_jar = mlpiper_jar
        return self

    def in_place(self, in_place):
        self._in_place = in_place
        return self

    def deployment_dir(self, deploy_dir):
        if deploy_dir:
            self._deploy_dir = os.path.abspath(deploy_dir)
            self.comp_root_path(os.path.join(self._deploy_dir, MLPiperRunner.COMP_DIR))
        return self

    def comp_root_path(self, comp_root_path):
        self._comp_root_path = comp_root_path
        return self

    def bin_dir(self, bin_dir):
        self._bin_dir = os.path.realpath(bin_dir)
        return self

    def skip_mlpiper_deps_install(self, skip_deps):
        self._skip_mlpiper_deps = skip_deps
        return self

    def use_color(self, use_color):
        self._use_color = use_color
        return self

    def pipeline(self, pipeline):
        if hasattr(pipeline, "read"):
            self._pipeline_dict = json.load(pipeline)
        elif os.path.exists(pipeline):
            self._logger.debug("Detected pipeline as a file")

            with open(pipeline, "r") as f:
                self._pipeline_dict = json.load(f)
        else:
            self._logger.debug("Trying to load pipeline as a string")
            self._pipeline_dict = json.loads(pipeline)

        self._engine = self._pipeline_dict[json_fields.PIPELINE_ENGINE_TYPE_FIELD]

        return self

    def input_model(self, model_filepath):
        model_full_filepath = os.path.abspath(model_filepath)
        if not os.path.exists(model_full_filepath):
            raise Exception("Input model file path not exists! " + model_full_filepath)

        self._input_model_filepath = model_full_filepath

    def output_model(self, model_filepath):
        self._output_model_filepath = os.path.abspath(model_filepath)

    def force(self, force):
        self._force = force
        return self

    def test_mode(self, test_mode):
        self._test_mode = test_mode
        return self

    def deploy(self):
        self._logger.info("Preparing pipeline for run")

        if not self._deploy_dir:
            raise Exception("Deployment dir was not provided")

        if os.path.exists(self._deploy_dir):
            if self._force:
                shutil.rmtree(self._deploy_dir)
            else:
                raise Exception(
                    "Deployment dir already exists: {}".format(self._deploy_dir)
                )

        if not self._pipeline_dict:
            raise Exception("Pipeline was not provided")

        self._print_info()
        self._validate_pipeline()
        self._fix_pipeline()
        self._create_components_store()
        self._create_deployment()

    def _get_comp_repo_folder_structure(self, comp_name):
        return self._comp_scanner.comp_repo_info[self._engine][comp_name]

    def _get_pipeline_components(self):
        return set([e["type"] for e in self._pipeline_dict["pipe"]])

    def _validate_pipeline(self):
        if self._engine not in self._comp_scanner.comp_repo_info:
            raise Exception(
                "No component found for '{}' pipeline engine in '{}'".format(
                    self._engine, self._comp_repo_path
                )
            )

        for comp_name in self._get_pipeline_components():
            try:
                self._get_comp_repo_folder_structure(comp_name)
            except Exception:
                raise Exception(
                    "Component {} is not found in loaded components".format(comp_name)
                )

    def _fix_pipeline(self):
        if json_fields.PIPELINE_SYSTEM_CONFIG_FIELD not in self._pipeline_dict:
            self._logger.debug(
                "Adding {} to pipeline".format(json_fields.PIPELINE_SYSTEM_CONFIG_FIELD)
            )
            system_config = {}
            self._pipeline_dict[
                json_fields.PIPELINE_SYSTEM_CONFIG_FIELD
            ] = system_config

        if self._input_model_filepath:
            self._pipeline_dict[json_fields.PIPELINE_SYSTEM_CONFIG_FIELD][
                java_mapping.MODEL_FILE_SOURCE_PATH_KEY
            ] = self._input_model_filepath

        if self._output_model_filepath:
            self._pipeline_dict[json_fields.PIPELINE_SYSTEM_CONFIG_FIELD][
                java_mapping.MODEL_FILE_SINK_PATH_KEY
            ] = self._output_model_filepath

    def _copy_components(self):
        self._logger.debug("Copying pipeline components to staging area")
        comp_copied = {}
        for comp_name in self._get_pipeline_components():
            # Copying each component only once (some pipelines can contain the same
            # component multiple times)
            if comp_name not in comp_copied:
                comp_repo_folder_struct = self._get_comp_repo_folder_structure(
                    comp_name
                )
                src_comp_root = comp_repo_folder_struct["root"]
                src_comp_files = comp_repo_folder_struct["files"]
                comp_dst_root = os.path.join(self._comp_root_path, comp_name)
                os.mkdir(comp_dst_root)
                for f in src_comp_files:
                    try:
                        src_path = os.path.join(src_comp_root, f)
                        dst_path = os.path.join(comp_dst_root, f)
                        shutil.copy(src_path, dst_path)
                    except IOError:
                        os.makedirs(os.path.dirname(dst_path))
                        shutil.copy(src_path, dst_path)
                    self._logger.debug(
                        "Copied src comp file to dst: {} ==> {}".format(
                            src_path, dst_path
                        )
                    )

                self._logger.debug(
                    "Created tmp dst component dir: {}".format(comp_dst_root)
                )

                comp_copied[comp_name] = True

    def _create_components_store(self):
        os.makedirs(self._comp_root_path)
        self._logger.debug("dist_dir: {}".format(self._comp_root_path))

        self._copy_components()

    def _copy_pkg(self, comp_pkg_dir):

        for pkg_filepath in glob.glob(comp_pkg_dir + "/" + "dist/*.egg"):
            self._logger.debug("Copying pkg {}".format(pkg_filepath))
            shutil.copy(pkg_filepath, self._deploy_dir)

        for pkg_filepath in glob.glob(comp_pkg_dir + "/" + "dist/*.whl"):
            self._logger.debug("Copying pkg {}".format(pkg_filepath))
            shutil.copy(pkg_filepath, self._deploy_dir)

    def _create_deployment(self):
        pipeline_file = os.path.join(
            self._deploy_dir, MLPiperRunner.DEPLOYMENT_PIPELINE
        )
        with open(pipeline_file, "w") as f:
            json.dump(self._pipeline_dict, f, indent=4)

    def _cleanup(self):
        self._logger.info("Cleaning up ... " + self._deploy_dir)
        shutil.rmtree(self._comp_root_path)

    def _install_deps(self, py_deps):
        self._logger.info("Installing py dependencies ... {}".format(py_deps))

        # Make sure to save the order within the dependencies
        for py_dep in py_deps:
            cmd = "yes | {} -m pip install --disable-pip-version-check {}".format(
                sys.executable, py_dep
            )

            self._logger.info("cmd: " + cmd)
            try:
                subprocess.check_call(cmd, shell=True)
            except subprocess.CalledProcessError as ex:
                print(str(ex))

    def run_deployment(self):
        if self._in_place:
            self._logger.info(
                "Running deployment in-place ... steps root path: {}".format(
                    self._comp_repo_path
                )
            )
            comp_root_path = self._comp_repo_path
            self._fix_pipeline()
        else:
            self._logger.info(
                "Running prepared deployment, {}".format(self._deploy_dir)
            )
            comp_root_path = self._comp_root_path
            self._add_deployment_eggs_to_path()
            self._load_deployment_pipeline()

        self._pipeline_dict[json_fields.PIPELINE_SYSTEM_CONFIG_FIELD][
            json_fields.PIPELINE_SYSTEM_CONFIG_TEST_MODE_PARAM
        ] = self._test_mode

        sys.path.insert(0, comp_root_path)

        pipeline_runner = (
            Executor()
            .pipeline_dict(self._pipeline_dict)
            .comp_root_path(comp_root_path)
            .use_color(self._use_color)
            .mlpiper_jar(self._mlpiper_jar)
            .standalone(True)
        )

        if not self._skip_mlpiper_deps:
            py_deps = pipeline_runner.all_py_component_dependencies()
            if py_deps:
                self._install_deps(py_deps)

        try:
            pipeline_runner.go()
        except KeyError as e:
            if str(e).find(java_mapping.MODEL_FILE_SINK_PATH_KEY) != -1:
                raise MLPiperException(
                    "Component in pipeline outputs a model, please provide"
                    " '--output-model' argument"
                )
            elif str(e).find(java_mapping.MODEL_FILE_SOURCE_PATH_KEY) != -1:
                raise MLPiperException(
                    "Component in pipeline expects to receive a model, please provide"
                    " '--input-model' argument"
                )
            else:
                raise

    def _add_deployment_eggs_to_path(self):
        py_version_major = sys.version_info[0]
        eggs = glob.glob("{}/*py{}*.egg".format(self._deploy_dir, py_version_major))
        if len(eggs) == 0:
            self._logger.warning(
                "No eggs found for py{}. Trying to find all possible eggs.".format(
                    py_version_major
                )
            )
            eggs = glob.glob("{}/*.egg".format(self._deploy_dir))
        for egg in eggs:
            sys.path.insert(0, egg)

    def _load_deployment_pipeline(self):
        pipeline_file = os.path.join(
            self._deploy_dir, MLPiperRunner.DEPLOYMENT_PIPELINE
        )
        if not os.path.exists(pipeline_file):
            raise Exception("Pipeline file not exists! path: {}".format(pipeline_file))

        with open(pipeline_file, "r") as f:
            self._pipeline_dict = json.load(f)

    def deps(self, lang, requirements_filepath):
        self._logger.info("Showing dependencies information...")

        sys.path.insert(0, self._comp_repo_path)

        pipeline_runner = (
            Executor()
            .comp_root_path(self._comp_repo_path)
            .pipeline_dict(self._pipeline_dict)
            .use_color(self._use_color)
        )

        deps = None
        if lang == ComponentLanguage.PYTHON:
            deps = pipeline_runner.all_py_component_dependencies(requirements_filepath)
        elif lang == ComponentLanguage.R:
            deps = pipeline_runner.all_r_component_dependencies(requirements_filepath)
        else:
            pass

        print("\n----- Dependencies -----")
        if deps:
            for dep in sorted(deps):
                print(dep)

            if requirements_filepath:
                print("\nRequirements saved in: {}".format(requirements_filepath))
        else:
            print(
                "No dependencies found for {} components.\nOr there are no {} components in"
                " the pipeline.".format(lang, lang)
            )
