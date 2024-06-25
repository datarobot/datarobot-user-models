import abc
from mlpiper.pipeline.component_runner.component_runner import ComponentRunner


class StandaloneComponentRunner(ComponentRunner):
    def __init__(self, ml_engine, dag_node):
        super(StandaloneComponentRunner, self).__init__(ml_engine, dag_node)

    @abc.abstractmethod
    def run(self, parent_data_objs):
        pass
