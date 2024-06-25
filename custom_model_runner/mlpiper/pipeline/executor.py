import json
import os
import six
import sys
import traceback
import mlpiper.common.constants as MLPiperConstants

from contextlib import contextmanager
from mlpiper.common.base import Base
from mlpiper.common.verbose_printer import VerbosePrinter
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.pipeline import components_desc
from mlpiper.pipeline import json_fields
from mlpiper.pipeline.dag import Dag
from mlpiper.pipeline.data_type import EngineType
from mlpiper.pipeline.component_language import ComponentLanguage
from mlpiper.pipeline.executor_config import (
    MLPIPER_JAR_ARG,
    SPARK_JARS_ARG,
    SPARK_JARS_ENV_VAR,
)


class ExecutorException(Exception):
    pass


class Executor(Base):
    def __init__(self, args=None):
        super(Executor, self).__init__()

        self._args = args
        self._pipeline_file = None
        self._pipeline = None
        self._json_pipeline = None
        self._run_locally = False
        self._ml_engine = None
        self._mlpiper_jar = None
        self._use_color = True
        self._comp_root_path = None
        self._standalone = False
        self._uuid = None
        self._dag = None
        self._initialized = False
        self._vp = VerbosePrinter.Instance()

        if args:
            self._json_pipeline = getattr(args, "pipeline", None)
            self._pipeline_file = getattr(args, "pipeline_file", None)
            self._run_locally = getattr(args, "run_locally", False)
            self._mlpiper_jar = getattr(args, MLPIPER_JAR_ARG, None)
            self._spark_jars = getattr(args, SPARK_JARS_ARG, None)
            self._comp_root_path = getattr(args, "comp_root_path", None)
        else:
            self._spark_jars = os.environ.get(SPARK_JARS_ENV_VAR, None)

    def pipeline_file(self, pipeline_file):
        self._pipeline_file = pipeline_file
        return self

    def pipeline_dict(self, pipeline_dict):
        self._pipeline = pipeline_dict
        return self

    @property
    def pipeline_user_data(self):
        return self._ml_engine.user_data if self._ml_engine else {}

    @property
    def pipeline(self):
        if not self._pipeline:
            self._load_pipeline()
        return self._pipeline

    def mlpiper_jar(self, mlpiper_jar):
        self._mlpiper_jar = mlpiper_jar
        return self

    def use_color(self, use_color):
        self._use_color = use_color
        return self

    def comp_root_path(self, comp_root_path):
        self._comp_root_path = comp_root_path
        return self

    def standalone(self, standalone):
        self._standalone = standalone
        return self

    def set_uuid(self, uuid):
        self._uuid = uuid
        return self

    def set_verbose(self, verbose):
        self._vp.set_verbose(verbose)
        return self

    @staticmethod
    def handle(args):
        Executor(args).go()

    @staticmethod
    def handle_deps(args):
        Executor(args).print_component_deps()

    def print_component_deps(self):
        """
        Printout all the python dependencies for a given pipeline. The dependencies are taken
        from all the component's metadata description files
        """
        all_deps = self.all_component_dependencies(self._args.lang)
        if all_deps:
            for dep in sorted(all_deps):
                print(dep)
        else:
            print("No dependencies were found for '{}'!".format(self._args.lang))

    def all_py_component_dependencies(self, requirements_filepath=None):
        return self.all_component_dependencies(
            ComponentLanguage.PYTHON, requirements_filepath
        )

    def all_r_component_dependencies(self, requirements_filepath=None):
        return self.all_component_dependencies(
            ComponentLanguage.R, requirements_filepath
        )

    def all_component_dependencies(self, lang, requirements_filepath=None):
        accumulated_py_deps = []

        reqs_file_handled = []
        comps_desc_list = components_desc.ComponentsDesc(
            pipeline=self.pipeline, comp_root_path=self._comp_root_path
        ).load(extended=True)
        for comp_desc in comps_desc_list:
            if comp_desc[json_fields.PIPELINE_LANGUAGE_FIELD] == lang:
                deps = comp_desc.get(json_fields.COMPONENT_DESC_PYTHON_DEPS, None)
                if deps:
                    # Use the following lists manipulation instead of 'set' update
                    # to guarantee dependencies order
                    for d in deps:
                        if d not in accumulated_py_deps:
                            accumulated_py_deps.append(d)

                # for Python fetch deps from requirements.txt file
                if lang == ComponentLanguage.PYTHON:
                    req_file = os.path.join(
                        comp_desc[json_fields.COMPONENT_DESC_ROOT_PATH_FIELD],
                        MLPiperConstants.REQUIREMENTS_FILENAME,
                    )
                    if req_file not in reqs_file_handled:
                        reqs_file_handled.append(req_file)
                        if os.path.exists(req_file):
                            with open(req_file) as f:
                                for requirement in f:
                                    requirement = requirement.strip()
                                    if requirement not in accumulated_py_deps:
                                        accumulated_py_deps.append(requirement)

        if accumulated_py_deps and requirements_filepath:
            with open(requirements_filepath, "w") as f:
                f.write("\n".join(accumulated_py_deps))

        return accumulated_py_deps

    def _parse_exit_code(self, code):
        # in case of calls like exit("some_string")
        return code if isinstance(code, six.integer_types) else 1

    @contextmanager
    def _catch_and_finalize(self, cleanup=False):
        try:
            yield
        # This except is intended to catch exit() calls from components.
        # Do not use exit() in mlpiper code.
        except SystemExit as e:
            code = self._parse_exit_code(e.code)
            error_message = "Pipeline called exit(), with code: {}".format(e.code)
            traceback_message = traceback.format_exc()
            if code != 0:
                self._logger.error("{}\n{}".format(error_message, traceback_message))
                # For Py2 put traceback into the exception message
                if sys.version_info[0] == 2:
                    error_message = "{}\n{}".format(
                        error_message, traceback.format_exc()
                    )
                raise ExecutorException(error_message)
            else:
                self._logger.warning(error_message)
        except KeyboardInterrupt:
            # When running from mlpiper tool (standalone)
            pass
        finally:
            sys.stdout.flush()
            if cleanup:
                self._logger.info("Done running pipeline (in finally block)")
                self.cleanup_pipeline()

    def _init_pipeline(self):
        self._logger.info("Start initializing pipeline")

        self._logger.debug("Executor.go_init_and_configure()")
        self._init_ml_engine(self.pipeline)

        comps_desc_list = components_desc.ComponentsDesc(
            self._ml_engine, self.pipeline, self._comp_root_path
        ).load()
        self._logger.debug("comp_desc: {}".format(comps_desc_list))
        self._dag = Dag(self.pipeline, comps_desc_list, self._ml_engine).use_color(
            self._use_color
        )

        # Flush stdout so the logs looks a bit in order
        sys.stdout.flush()

        system_conf = (
            self.pipeline[json_fields.PIPELINE_SYSTEM_CONFIG_FIELD]
            if json_fields.PIPELINE_SYSTEM_CONFIG_FIELD in self.pipeline
            else {}
        )
        ee_conf = self.pipeline.get(json_fields.PIPELINE_EE_CONF_FIELD, dict())

        if self._dag.is_stand_alone:
            self._dag.configure_single_component_pipeline(
                system_conf, ee_conf, self._ml_engine
            )
        else:
            self._dag.configure_connected_pipeline(
                system_conf, ee_conf, self._ml_engine
            )

        self._initialized = True
        self._logger.info("Finish initializing pipeline")

    def _run_pipeline(self):
        """
        Actual execution phase
        """
        if not self._initialized:
            self._logger.debug(
                "Pipeline is not initialized. Forgot to Executor.init_pipeline()?"
            )
            print("Pipeline is not initialized. Forgot to Executor.init_pipeline()?")
            return

        self._logger.info("Start running pipeline")
        if self._dag.is_stand_alone:
            self._dag.run_single_component_pipeline(self._ml_engine)
        else:
            self._dag.run_connected_pipeline(self._ml_engine)

        self._logger.info("Finish running pipeline")

    def _terminate_pipeline_components(self):
        """Teardown pipeline components"""

        if not self._initialized:
            self._logger.debug(
                "Pipeline is not initialized. Forgot to Executor.init_pipeline()?"
            )
            print("Pipeline is not initialized. Forgot to Executor.init_pipeline()?")
            return

        self._logger.info("Cleaning up components")
        if self._dag.is_stand_alone:
            self._logger.info(
                "Pipeline component termination is skipped for a stand alone component"
            )
        else:
            self._dag.terminate_connected_pipeline()
            self._logger.info("Finish terminating pipeline components")

    def init_pipeline(self):
        with self._catch_and_finalize(cleanup=False):
            self._init_pipeline()

    def run_pipeline(self, cleanup=True):
        with self._catch_and_finalize(cleanup):
            self._run_pipeline()

    def go(self):
        """
        Actual execution phase
        """
        self._logger.debug("Executor.go()")
        self.init_pipeline()
        self.run_pipeline()
        self.cleanup_pipeline()
        self._logger.debug("End of Executor.go()")

    def _init_ml_engine(self, pipeline):
        engine_type = pipeline[json_fields.PIPELINE_ENGINE_TYPE_FIELD]
        self._logger.info("Engine type: {}".format(engine_type))
        if engine_type == EngineType.PY_SPARK:
            from mlpiper.ml_engine.py_spark_engine import PySparkEngine

            self._ml_engine = PySparkEngine(
                pipeline, self._run_locally, self._spark_jars
            )

        elif engine_type in [
            EngineType.GENERIC,
            EngineType.REST_MODEL_SERVING,
            EngineType.SAGEMAKER,
        ]:
            # All are supposed to be derived from python engine

            if engine_type == EngineType.GENERIC:
                from mlpiper.ml_engine.python_engine import PythonEngine

                self._logger.info("Using python engine")
                self._ml_engine = PythonEngine(pipeline, self._mlpiper_jar)

                if not self.is_logger_set():
                    self.set_logger(
                        self._ml_engine.get_engine_logger(self.logger_name())
                    )

            elif engine_type == EngineType.REST_MODEL_SERVING:
                from mlpiper.ml_engine.rest_model_serving_engine import (
                    RestModelServingEngine,
                )

                self._logger.info("Using REST Model Serving engine")
                self._ml_engine = RestModelServingEngine(
                    pipeline, self._mlpiper_jar, self._standalone
                )

            elif engine_type == EngineType.SAGEMAKER:
                from mlpiper.ml_engine.sagemaker_engine import SageMakerEngine

                self._logger.info("Using SageMaker engine")
                self._ml_engine = SageMakerEngine(pipeline)
        else:
            raise MLPiperException(
                "Engine type is not supported by the Python execution engine! "
                "engineType: {}".format(engine_type)
            )

    def cleanup_pipeline(self):
        if not self._initialized:
            return

        self._terminate_pipeline_components()

        if self._ml_engine:
            self._ml_engine.stop()

        if self._ml_engine:
            self._ml_engine.cleanup()

        self._initialized = False

    def _start_spark_session(self, name):
        # Doing the import here inorder not to require pyspark even if spark is not used
        if False:
            from pyspark.sql import SparkSession
        spark_session = SparkSession.builder.appName(name)
        if self._run_locally:
            spark_session.master("local[*]")

        return spark_session.getOrCreate()

    def _load_pipeline(self):
        if self._pipeline:
            return self._pipeline

        if self._json_pipeline:
            self._pipeline = json.loads(self._json_pipeline)
        elif self._pipeline_file:
            self._pipeline = json.load(self._pipeline_file)
        else:
            raise MLPiperException("Missing pipeline file!")

        # Validations
        if json_fields.PIPELINE_PIPE_FIELD not in self._pipeline:
            raise MLPiperException(
                "Pipeline does not contain any component! pipeline="
                + str(self._pipeline)
            )

        pipeline_str = str(self._pipeline)
        self._logger.debug("Pipeline: " + pipeline_str)

        return self._pipeline
