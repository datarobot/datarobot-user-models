import runpy
import sys

from mlpiper.pipeline.pipeline_utils import main_component_module
from mlpiper.pipeline.component_runner.standalone_component_runner import (
    StandaloneComponentRunner,
)
from mlpiper.pipeline.pipeline_utils import assemble_cmdline_from_args
from mlpiper.pipeline.component_language import ComponentLanguage


class PythonStandaloneComponentRunner(StandaloneComponentRunner):
    def __init__(self, ml_engine, dag_node):
        super(PythonStandaloneComponentRunner, self).__init__(ml_engine, dag_node)

    def run(self, parent_data_objs):
        self._logger.info("Running stand alone python component")

        cmdline = assemble_cmdline_from_args(self._params)
        self._logger.debug("cmdline: {}".format(cmdline))

        orig_cmdline = sys.argv[1:]
        sys.argv[1:] = cmdline

        module_main_prog = main_component_module(self._dag_node.comp_desc())
        self._logger.info("Main program: {}".format(module_main_prog))
        self._logger.info("Dag Node {}".format(self._dag_node))

        try:
            if self._dag_node.comp_language() == ComponentLanguage.PYTHON:
                # Running the module as a script, using the __main__ as the run_name
                runpy.run_module(
                    str(module_main_prog), run_name="__main__", alter_sys=True
                )

        finally:
            sys.argv[1:] = orig_cmdline
