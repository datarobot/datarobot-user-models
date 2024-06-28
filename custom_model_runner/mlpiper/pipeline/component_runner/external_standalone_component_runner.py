from mlpiper.pipeline.pipeline_utils import assemble_cmdline_from_args
from mlpiper.pipeline.component_runner.standalone_component_runner import (
    StandaloneComponentRunner,
)
from mlpiper.pipeline.component_runner.external_program_runner import (
    ExternalProgramRunner,
)


class ExternalStandaloneComponentRunner(StandaloneComponentRunner):
    def __init__(self, ml_engine, dag_node):
        super(ExternalStandaloneComponentRunner, self).__init__(ml_engine, dag_node)

    def run(self, parent_data_objs):
        self._logger.info("Running ExternalStandaloneComponentRunner component")

        self._logger.info("Using external program runner")
        self._logger.info("Program: {}".format(self._dag_node.comp_program()))

        external_runner = ExternalProgramRunner(
            root_path=self._dag_node.comp_root_path(),
            main_program=self._dag_node.comp_program(),
        )

        cmdline = assemble_cmdline_from_args(self._params)
        ret_val = external_runner.run_standalone(cmdline)
        self._logger.info("external runner ret val: {}".format(ret_val))
        if ret_val != 0:
            msg = "External program failed: {}".format(ret_val)
            self.error(msg)
            raise Exception(msg)
