import importlib
import inspect
import time
import sys

from termcolor import colored

from mlpiper.common.base import Base
from mlpiper.common.verbose_printer import VerbosePrinter
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.pipeline import json_fields
from mlpiper.pipeline.component_language import ComponentLanguage
from mlpiper.pipeline.component_runner.external_connected_component_runner import (
    ExternalConnectedComponentRunner,
)
from mlpiper.pipeline.component_runner.external_standalone_component_runner import (
    ExternalStandaloneComponentRunner,
)
from mlpiper.pipeline.component_runner.java_connected_component_runner import (
    JavaConnectedComponentRunner,
)
from mlpiper.pipeline.component_runner.java_standalone_component_runner import (
    JavaStandaloneComponentRunner,
)
from mlpiper.pipeline.component_runner.python_connected_component_runner import (
    PythonConnectedComponentRunner,
)
from mlpiper.pipeline.component_runner.python_standalone_component_runner import (
    PythonStandaloneComponentRunner,
)
from mlpiper.pipeline.dag_node import DagNode
from mlpiper.pipeline.pipeline_utils import main_component_module
from mlpiper.pipeline.topological_sort import TopologicalSort


class Dag(Base):
    def __init__(self, pipeline, comp_desc_list, ml_engine):
        super(Dag, self).__init__(ml_engine.get_engine_logger(self.logger_name()))
        self._pipeline = pipeline
        self._comp_desc_list = comp_desc_list
        self._ml_engine = ml_engine
        self._is_stand_alone = None
        self._uploaded_comp_classes = dict()
        self._parent_data_objs_placeholder = dict()
        self._sorted_execution_graph_list = self._sorted_execution_graph()
        self._report_color = "green"
        self._use_color = True

    def use_color(self, use_color):
        self._use_color = use_color
        return self

    @property
    def is_stand_alone(self):
        return self._is_stand_alone

    def get_dag_node(self, index):
        return self._sorted_execution_graph_list[index]

    def run_single_component_pipeline(self, engine_info):
        """
        Running a single component pipeline.
        o Assemble the command line arguments
        o Run the module using runpy so as if the module is executed as a script
        :param engine_info:
        :return:
        """

        self._logger.info("running single component pipeline (stand alone)")

        if not self.is_stand_alone:
            raise Exception("Dag is not a single component pipeline")

        dag_node = self.get_dag_node(0)
        dag_node.component_runner.run(None)

    def _print_colored(self, msg):
        vp = VerbosePrinter.Instance()
        if self._use_color:
            vp.verbose_print(colored(msg, self._report_color))
        else:
            vp.verbose_print(msg)

    def _component_run_header(self, dag_node):
        self._print_colored(" ")
        self._print_colored(" ")
        self._print_colored("=" * 60)
        self._print_colored("Component: {}".format(dag_node.comp_label()))
        self._print_colored("Output:")
        self._print_colored("-" * 60)

    def _component_run_footer(self, dag_node, data_objs, runtime_in_sec):
        self._print_colored("-" * 60)
        self._print_colored("Runtime:    {:.1f} sec".format(runtime_in_sec))
        self._print_colored("NR outputs: {}".format(len(data_objs) if data_objs else 0))
        self._print_colored("=" * 60)
        self._print_colored(" ")

    def configure_single_component_pipeline(self, system_conf, ee_conf, engine_info):
        # Components configuration phase
        self._logger.info("configuring single component pipeline (stand alone)")

        if not self.is_stand_alone:
            raise Exception("Dag is not a single component pipeline")

        dag_node = self.get_dag_node(0)

        # TODO: move the preparation to be dag_node methods or runner helper
        input_args = dag_node.input_arguments(system_conf, ee_conf, comp_only_args=True)
        self._logger.info("Detected {} component".format(dag_node.comp_language()))
        dag_node.component_runner.configure(input_args)

    def configure_connected_pipeline(self, system_conf, ee_conf, engine_info):
        # Components configuration phase
        self._logger.debug("Configure connected pipeline")
        for dag_node in self._sorted_execution_graph_list:
            input_args = dag_node.input_arguments(system_conf, ee_conf)
            dag_node.component_runner.configure(input_args)

    def run_connected_pipeline(self, engine_info):
        # Components materialize phase
        self._logger.debug("Run connected pipeline")
        for dag_node in self._sorted_execution_graph_list:
            parent_data_objs = self.parent_data_objs(dag_node)

            self._logger.debug(
                "Calling dag node '{}', with args: {}".format(
                    dag_node.comp_name(), parent_data_objs
                )
            )

            self._component_run_header(dag_node)
            start = time.time()

            sys.stderr.flush()
            sys.stdout.flush()
            data_objs = dag_node.component_runner.run(parent_data_objs)
            sys.stderr.flush()
            sys.stdout.flush()

            runtime_in_sec = time.time() - start
            if data_objs and type(data_objs) is not list:
                raise MLPiperException(
                    "Invalid returned data type from component! It should be a list! "
                    "name: " + dag_node.comp_name()
                )

            self._component_run_footer(dag_node, data_objs, runtime_in_sec)

            self._logger.debug(
                "Output of dag node '{}' is: {}".format(dag_node.comp_name(), data_objs)
            )

            self.update_parent_data_objs(dag_node, data_objs)

        self._ml_engine.finalize()

    def terminate_connected_pipeline(self):
        self._logger.debug("Terminate a pipeline connected components")
        for dag_node in self._sorted_execution_graph_list:
            self._logger.debug("Terminate dag node '{}'".format(dag_node.comp_name()))
            dag_node.component_runner.terminate()

    def sorted_execution_graph(self):
        return self._sorted_execution_graph_list

    def parent_data_objs(self, dag_node):
        data_objs = []
        # dict to keep track of proper input index going to components
        data_object_dict = {}
        max_index = 0

        for parent in dag_node.parents():
            parent_id = parent[json_fields.PIPELINE_COMP_PARENTS_FIRST_FIELD]
            output_index = parent[json_fields.PIPELINE_COMP_PARENTS_SECOND_FIELD]

            input_index = parent.get(json_fields.PIPELINE_COMP_PARENTS_THIRD_FIELD, max_index)

            if parent_id in self._parent_data_objs_placeholder:
                if output_index in self._parent_data_objs_placeholder[parent_id]:
                    self._logger.debug(
                        "Concatenate parent data objs, parent_id={}, output_index={}".format(
                            parent_id, output_index
                        )
                    )
                    data_object_dict[input_index] = self._parent_data_objs_placeholder[parent_id][
                        output_index
                    ]
                else:
                    data_object_dict[input_index] = None
                    self._logger.debug(
                        "Output index not in data objs placeholder! "
                        "parent_id={}, output_index={}".format(parent_id, output_index)
                    )
            else:
                data_object_dict[input_index] = None
                self._logger.debug(
                    "Parent id not in data objs placeholder! id={}".format(parent_id)
                )

            max_index += 1
        # asserting if we really have covered all parents.
        assert len(data_object_dict) == len(dag_node.parents())

        for index in range(len(dag_node.parents())):
            data_objs.append(data_object_dict[index])

        return data_objs

    def update_parent_data_objs(self, dag_node, data_objs):
        parent_id = dag_node.pipe_id()
        if parent_id in self._parent_data_objs_placeholder:
            for output_index in self._parent_data_objs_placeholder[parent_id]:
                self._parent_data_objs_placeholder[parent_id][output_index] = (
                    data_objs[output_index] if data_objs and output_index < len(data_objs) else None
                )
                self._logger.debug(
                    "Parent entry was updated, parent_id={}, output_index={}".format(
                        parent_id, output_index
                    )
                )
        else:
            self._logger.debug("Parent id not in data objs placeholder! id={}".format(parent_id))

    def _sorted_execution_graph(self):
        execution_graph = self._execution_graph()
        sorted_list = TopologicalSort(execution_graph, self._ml_engine).sort()
        self._logger.debug("Sorted execution graph: {}".format([n.pipe_id() for n in sorted_list]))
        return sorted_list

    def _execution_graph(self):
        dag_node_dict = dict()
        for pipe_comp in self._pipeline[json_fields.PIPELINE_PIPE_FIELD]:
            self._logger.debug("Handling {}".format(pipe_comp))
            pipe_comp_id = pipe_comp[json_fields.PIPELINE_COMP_ID_FIELD]
            self._add_entry_in_parent_data_objs_placeholder(pipe_comp)
            comp_desc = self._find_comp_desc(pipe_comp)

            dag_node = DagNode(None, comp_desc, pipe_comp, self._ml_engine)

            if comp_desc[json_fields.COMPONENT_DESC_USER_STAND_ALONE]:
                if dag_node.comp_language() == ComponentLanguage.PYTHON:
                    comp_runner = PythonStandaloneComponentRunner(self._ml_engine, dag_node)
                elif dag_node.comp_language() == ComponentLanguage.JAVA:
                    comp_runner = JavaStandaloneComponentRunner(self._ml_engine, dag_node)
                elif dag_node.comp_language() == ComponentLanguage.R:
                    comp_runner = ExternalStandaloneComponentRunner(self._ml_engine, dag_node)
                else:
                    raise Exception(
                        "Language {} is not supported yet for standalone components".format(
                            dag_node.comp_language()
                        )
                    )

                self._is_stand_alone = True
            else:
                if dag_node.comp_language() == ComponentLanguage.PYTHON:
                    main_cls = self._find_main_comp_cls(comp_desc)(self._ml_engine)
                    dag_node.main_cls(main_cls)
                    comp_runner = PythonConnectedComponentRunner(self._ml_engine, dag_node)
                elif dag_node.comp_language() == ComponentLanguage.JAVA:
                    comp_runner = JavaConnectedComponentRunner(
                        self._ml_engine, dag_node, self._ml_engine.config["mlpiper_jar"]
                    )
                elif dag_node.comp_language() == ComponentLanguage.R:
                    comp_runner = ExternalConnectedComponentRunner(self._ml_engine, dag_node)
                else:
                    raise Exception(
                        "Language {} is not supported yet for connected components".format(
                            dag_node.comp_language()
                        )
                    )

            dag_node.component_runner = comp_runner
            dag_node_dict[pipe_comp_id] = dag_node

        self._logger.debug(
            "Execution graph: {}".format([dag_node_dict[k].pipe_id() for k in dag_node_dict])
        )
        self._logger.debug("parent_placeholder: {}".format(self._parent_data_objs_placeholder))
        return dag_node_dict

    def _add_entry_in_parent_data_objs_placeholder(self, pipe_comp):
        pipe_comp_parents = pipe_comp[json_fields.PIPELINE_COMP_PARENTS_FIELD]
        for connector in pipe_comp_parents:
            parent_id = connector[json_fields.PIPELINE_COMP_PARENTS_FIRST_FIELD]
            output_index = connector[json_fields.PIPELINE_COMP_PARENTS_SECOND_FIELD]

            if parent_id in self._parent_data_objs_placeholder:
                if output_index not in self._parent_data_objs_placeholder[parent_id]:
                    self._logger.debug(
                        "An entry already exists in data objs placeholder, "
                        "parend_id={}, output_id={}".format(parent_id, output_index)
                    )
                    self._parent_data_objs_placeholder[parent_id][output_index] = None
                else:
                    self._logger.debug(
                        "An entry already exists along with output index in data objs placeholder, "
                        "parend_id={}, output_id={}".format(parent_id, output_index)
                    )
            else:
                self._logger.debug(
                    "Adding an entry in data objs placeholder ... "
                    "parent_id={}, output_index={}".format(parent_id, output_index)
                )
                self._parent_data_objs_placeholder[parent_id] = {output_index: None}

    def _find_comp_desc(self, pipe_comp):
        comp_type = pipe_comp[json_fields.PIPELINE_COMP_TYPE_FIELD]
        self._logger.debug(
            "Component found in pipeline, id={}, type={}".format(
                pipe_comp[json_fields.PIPELINE_COMP_ID_FIELD], comp_type
            )
        )
        match_comp = [
            comp
            for comp in self._comp_desc_list
            if comp[json_fields.COMPONENT_DESC_NAME_FIELD] == comp_type
        ]
        if not match_comp:
            raise MLPiperException(
                "Could not find a component read from the pipeline! "
                "type=[{}] desc=[{}] pipe_comp=[{}]".format(
                    comp_type,
                    self._comp_desc_list,
                    pipe_comp[json_fields.PIPELINE_COMP_ID_FIELD],
                )
            )
        elif len(match_comp) > 1:
            raise MLPiperException("Found more then one component! type=" + comp_type)

        return match_comp[0]

    def _find_main_comp_cls(self, comp_desc):
        module = main_component_module(comp_desc)
        if module in self._uploaded_comp_classes:
            return self._uploaded_comp_classes[module]

        self._logger.debug("Importing component: {}".format(module))
        loaded_module = importlib.import_module(module)
        self._logger.debug("Component imported successfully: {}".format(module))
        cls_name = comp_desc[json_fields.COMPONENT_DESC_CLASS_FIELD]
        main_cls = inspect.getmembers(
            loaded_module, lambda o: inspect.isclass(o) and cls_name == o.__name__
        )[0][1]
        self._uploaded_comp_classes[module] = main_cls
        return main_cls
