from mlpiper.common.base import Base
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.pipeline import json_fields


class TopologicalSort(Base):
    def __init__(self, execution_graph, ml_engine):
        super(TopologicalSort, self).__init__(
            ml_engine.get_engine_logger(self.logger_name())
        )
        self._execution_graph = execution_graph
        self._sorted_execution_graph_list = []

    def sort(self):
        if not self._sorted_execution_graph_list:
            while True:
                dag_node = self._find_unmarked_node()
                if not dag_node:
                    break
                self._visit(dag_node)

        return self._sorted_execution_graph_list

    def _find_unmarked_node(self):
        # Iterete over a dict, which comply both with Python2 and Python3
        for key in self._execution_graph:
            dag_node = self._execution_graph[key]
            if not dag_node._perm_visit:
                return dag_node
        return None

    def _visit(self, dag_node):
        self._logger.debug("Visit node: {}".format(dag_node.pipe_id()))
        if dag_node._perm_visit:
            return

        if dag_node._temp_visit:
            raise MLPiperException(
                "The pipeline has invalid cyclic loop (Not a DAG)! pipe-node-id: {}".format(
                    dag_node.pipe_id()
                )
            )

        dag_node._temp_visit = True
        for parent in dag_node.parents():
            parent_id = parent[json_fields.PIPELINE_COMP_PARENTS_FIRST_FIELD]
            if parent_id not in self._execution_graph:
                raise MLPiperException(
                    "Parent id was not found in the pipeline component's list! pId: {}".format(
                        parent_id
                    )
                )

            self._visit(self._execution_graph[parent_id])

        dag_node._temp_visit = False
        dag_node._perm_visit = True
        self._sorted_execution_graph_list.append(dag_node)
