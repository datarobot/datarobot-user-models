import logging

from mlpiper.common.base import Base
from mlpiper.common.mlpiper_exception import MLPiperException


class TopologicalNode(object):
    def __init__(self, node, key, child_keys):
        self._node = node
        self._key = key
        self._child_keys = child_keys

        self.temp_visit = False
        self.perm_visit = False

    @property
    def node(self):
        return self._node

    @property
    def key(self):
        return self._key

    @property
    def child_keys(self):
        return self._child_keys

    def __str__(self):
        return "key: {}, childs: {}".format(self.key, self.child_keys)


class TopologicalSort(Base):
    """
    Generates topological sort from a list or dict, which represent graphs.
    """

    def __init__(self, graph, key_attr_name, ptr_attr_name):
        """
        :param graph:  a list or dict that contains the whole nodes in the graph
        :param key_attr_name:  a string that is literally the name of the key accessor
        :param ptr_attr_name:  a string that is literally the name of the children accessor
        """
        super(TopologicalSort, self).__init__(logging.getLogger(self.logger_name()))

        self._graph = graph
        self._graph_aux = {}

        self._key_attr_name = key_attr_name
        self._ptr_attr_name = ptr_attr_name

        self._perm_visit = {}
        self._sorted_graph = []

        self._generate_aux_graph(graph, key_attr_name, ptr_attr_name)

    def _generate_aux_graph(self, graph, key_attr_name, ptr_attr_name):
        self._logger.debug("{}".format(graph))
        if isinstance(graph, dict):
            for key, node in graph.items():
                self._add_node(node, key_attr_name, ptr_attr_name)
        elif isinstance(graph, list):
            for node in graph:
                self._add_node(node, key_attr_name, ptr_attr_name)
        else:
            raise MLPiperException(
                "Invalid graph type for topological sort! Expected 'dict' or 'list'! "
                + "type: {}".format(type(graph))
            )

        self._logger.debug(self._graph_aux)

    def _add_node(self, node, key_attr_name, ptr_attr_name):
        key_value = TopologicalSort._call_class_attr(node, key_attr_name)
        self._logger.debug("key_value: {}".format(key_value))

        if key_value not in self._graph_aux:
            ptr_value = TopologicalSort._call_class_attr(node, ptr_attr_name)
            self._logger.debug(
                "ptr_value: {}, type: {}".format(ptr_value, type(ptr_value))
            )

            child_keys = []
            try:
                for child in ptr_value:
                    if child:
                        self._add_child_key_name(child, key_attr_name, child_keys)
            except TypeError:
                if ptr_value:
                    self._add_child_key_name(child, key_attr_name, child_keys)

            self._graph_aux[key_value] = TopologicalNode(node, key_value, child_keys)

    def _add_child_key_name(self, child, key_attr_name, child_keys):
        key_name = TopologicalSort._call_class_attr(child, key_attr_name)
        self._logger.debug("child key name: {}".format(key_name))
        child_keys.append(key_name)

    @staticmethod
    def _call_class_attr(cls, attr_name):
        attr = getattr(cls, attr_name, None)
        if not attr:
            raise MLPiperException(
                "The given class does not include the given attribute name! "
                + "class: {}, attr_name: {}".format(cls, attr_name)
            )
        attr_value = attr() if callable(attr) else attr
        return attr_value

    def sort(self):
        if not self._sorted_graph:
            while True:
                t_node = self._find_unmarked_node()
                if not t_node:
                    break
                self._visit(t_node)

        return self._sorted_graph

    def _find_unmarked_node(self):
        for key, t_node in self._graph_aux.items():
            if not t_node.perm_visit:
                return t_node
        return None

    def _visit(self, t_node):
        self._logger.debug("Visiting node: {}".format(t_node.key))
        if t_node.perm_visit:
            return

        if t_node.temp_visit:
            raise MLPiperException(
                "The pipeline has invalid cyclic loop (Not a DAG)! node: {}".format(
                    t_node
                )
            )

        t_node.temp_visit = True
        for child_key in t_node.child_keys:
            if child_key not in self._graph_aux:
                raise MLPiperException(
                    "Child id was not found in the graph! key: {}".format(child_key)
                )

            self._visit(self._graph_aux[child_key])

        t_node.temp_visit = False
        t_node.perm_visit = True
        self._sorted_graph.append(t_node.node)


if __name__ == "__main__":

    class Node(object):
        def __init__(self, key, childs):
            self._key = key
            self._childs = childs

        @property
        def key(self):
            return self._key

        @property
        def childs(self):
            return self._childs

        def __str__(self):
            child_keys = [c.key for c in self.childs] if self.childs else None
            return "key: {}, childs: {}".format(self.key, child_keys)

    logging.basicConfig(
        level=logging.INFO, format="%(name)s %(levelname)s: %(message)s"
    )

    n1 = Node("a", None)
    n2 = Node("b", [n1])
    n3 = Node("c", [n1])
    n4 = Node("d", [n2, n3])
    n5 = Node("e", [n3])

    graph_list = [n3, n1, n2, n4, n5]
    sorted_graph = TopologicalSort(graph_list, "key", "childs").sort()
    print("Graph1:")
    for node in sorted_graph:
        print(node)

    graph_dict = {n3.key: n3, n1.key: n1, n2.key: n2, n4.key: n4, n5.key: n5}
    sorted_graph = TopologicalSort(graph_dict, "key", "childs").sort()
    print("\nGraph2:")
    for node in sorted_graph:
        print(node)
