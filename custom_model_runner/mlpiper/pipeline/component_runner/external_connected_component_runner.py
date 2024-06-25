from mlpiper.pipeline.component_runner.component_runner import ComponentRunner
from mlpiper.pipeline.component_runner.external_program_runner import (
    ExternalProgramRunner,
)


class ExternalConnectedComponentRunner(ComponentRunner):
    def __init__(self, ml_engine, dag_node):
        super(ExternalConnectedComponentRunner, self).__init__(ml_engine, dag_node)

    def run(self, parent_data_objs):
        self._logger.info("Running ExternalStandaloneComponentRunner component")
        self._logger.info("Using external program runner")
        self._logger.info("Program: {}".format(self._dag_node.comp_program()))

        external_runner = ExternalProgramRunner(
            root_path=self._dag_node.comp_root_path(),
            main_program=self._dag_node.comp_program(),
        )

        ret_val, output_objs = external_runner.run_connected(
            parent_data_objs, self._params
        )
        self._logger.info("external runner ret val: {}".format(ret_val))
        if ret_val != 0:
            msg = "External program failed: {}".format(ret_val)
            self.error(msg)
            raise Exception(msg)
        return output_objs
