from mlpiper.ml_engine.python_engine import PythonEngine
from mlpiper.pipeline.components_desc import ComponentsDesc
from mlpiper.pipeline.component_info import ComponentInfo
from mlpiper.pipeline.dag import Dag
from mlpiper.pipeline import json_fields

from .constants import PYTHON_COMPONENTS_PATH


class TestPythonIO:
    def test_correct_python_component_io(self):
        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "Test Train 1",
                    "id": 1,
                    "type": "test-python-train",
                    "parents": [],
                    "arguments": {"arg1": "arg1-value"},
                },
                {
                    "name": "Test Train 2",
                    "id": 2,
                    "type": "test-python-train",
                    "parents": [
                        {"parent": 1, "output": 1, "input": 1},
                        {"parent": 1, "output": 0, "input": 0},
                    ],
                    "arguments": {"arg1": "arg1-value"},
                },
                {
                    "name": "Test Train 3",
                    "id": 3,
                    "type": "test-python-train",
                    "parents": [
                        {"parent": 2, "output": 0, "input": 0},
                        {"parent": 2, "output": 2, "input": 2},
                        {"parent": 2, "output": 1, "input": 1},
                    ],
                    "arguments": {"arg1": "arg1-value"},
                },
                {
                    "name": "Test Train 4",
                    "id": 4,
                    "type": "test-python-train",
                    "parents": [
                        {"parent": 3, "output": 0, "input": 1},
                        {"parent": 3, "output": 1, "input": 0},
                    ],
                    "arguments": {"arg1": "arg1-value"},
                },
            ],
        }
        python_engine = PythonEngine("test-pipe")
        comps_desc_list = ComponentsDesc(
            python_engine, pipeline=pipeline, comp_root_path=PYTHON_COMPONENTS_PATH
        ).load()
        dag = Dag(pipeline, comps_desc_list, python_engine)

        dag_node_1 = dag.get_dag_node(0)
        dag_node_2 = dag.get_dag_node(1)
        dag_node_3 = dag.get_dag_node(2)
        dag_node_4 = dag.get_dag_node(3)

        # A100 means -- Type A, Node Id 1, Output 0, Goes To 0
        # pipeline is as follow

        #     OUTPUT INDEX 0 - INPUT INDEX 0      OUTPUT INDEX 0 - INPUT INDEX 0      OUTPUT INDEX 0   INPUT INDEX 0       # noqa: E501
        #    /                              \    /                              \    /              \ /             \      # noqa: E501
        # ID 1                               ID 2-OUTPUT INDEX 1 - INPUT INDEX 1-ID 3                /\              ID 4  # noqa: E501
        #    \                              /    \                              /    \              /  \            /      # noqa: E501
        #     OUTPUT INDEX 1 - INPUT INDEX 1      OUTPUT INDEX 2 - INPUT INDEX 2      OUTPUT INDEX 1    INPUT INDEX 1      # noqa: E501

        dag.update_parent_data_objs(dag_node_1, ["A100", "B111"])
        dag.update_parent_data_objs(dag_node_2, ["A200", "B211", "C222"])
        dag.update_parent_data_objs(dag_node_3, ["A301", "B310"])

        # as node 1 does not have any parents, input object should be empty
        assert dag.parent_data_objs(dag_node_1) == []
        # as node 2 have input coming but json is not correctly order, but still output should be
        # correctly indexed
        assert dag.parent_data_objs(dag_node_2) == ["A100", "B111"]
        # little complicated node 3 inputs. but same story as above
        assert dag.parent_data_objs(dag_node_3) == ["A200", "B211", "C222"]
        # node 4 gets output of node3's index 0 to its 1st input index and node3's output index 1
        # to its 0th input indexx
        assert dag.parent_data_objs(dag_node_4) == ["B310", "A301"]

    def test_component_info(self):
        comp_desc = {
            "version": "1.0.2",
            "engineType": "Generic",
            "language": "Python",
            "userStandalone": False,
            "name": "test-component-info",
            "label": "Test Component Info",
            "description": "Description for test component info",
            "program": "main.py",
            "componentClass": "TestComponentInfo",
            "modelBehavior": "ModelConsumer",
            "useMLOps": True,
            "group": "Algorithms",
            "deps": ["optparse", "reticulate"],
            "includeGlobPatterns": "__init__.py | util/*.py",
            "excludeGlobPatterns": "*.txt",
            "inputInfo": [
                {
                    "description": "Input string",
                    "label": "just-string",
                    "defaultComponent": "TestDefComp",
                    "type": "str",
                    "group": "data",
                }
            ],
            "outputInfo": [
                {
                    "description": "Output string",
                    "label": "just-output-string",
                    "defaultComponent": "OutputConnector",
                    "type": "str",
                    "group": "data",
                }
            ],
            "arguments": [
                {
                    "key": "exit-value",
                    "label": "exit-value",
                    "type": "int",
                    "description": "Exit value to use",
                    "envVar": "ENV_VAR",
                    "optional": True,
                },
                {
                    "key": "input-model",
                    "label": "Model input file",
                    "type": "str",
                    "description": "File to use for loading the model",
                    "optional": False,
                    "tag": "input_model_path",
                },
            ],
        }

        comp_info = ComponentInfo(comp_desc)
        assert comp_info.version == comp_desc[json_fields.COMPONENT_DESC_VERSION_FIELD]
        assert (
            comp_info.engine_type
            == comp_desc[json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD]
        )
        assert (
            comp_info.language == comp_desc[json_fields.COMPONENT_DESC_LANGUAGE_FIELD]
        )
        assert (
            comp_info.user_standalone
            == comp_desc[json_fields.COMPONENT_DESC_USER_STAND_ALONE]
        )
        assert comp_info.name == comp_desc[json_fields.COMPONENT_DESC_NAME_FIELD]
        assert comp_info.label == comp_desc[json_fields.COMPONENT_DESC_LABEL_FIELD]
        assert (
            comp_info.description
            == comp_desc[json_fields.COMPONENT_DESC_DESCRIPTION_FIELD]
        )
        assert comp_info.program == comp_desc[json_fields.COMPONENT_DESC_PROGRAM_FIELD]
        assert (
            comp_info.component_class
            == comp_desc[json_fields.COMPONENT_DESC_CLASS_FIELD]
        )
        assert (
            comp_info.model_behavior
            == comp_desc[json_fields.COMPONENT_DESC_MODEL_BEHAVIOR_FIELD]
        )
        assert (
            comp_info.use_mlops == comp_desc[json_fields.COMPONENT_DESC_USE_MLOPS_FIELD]
        )
        assert comp_info.group == comp_desc[json_fields.COMPONENT_DESC_GROUP_FIELD]
        assert comp_info.deps == comp_desc[json_fields.COMPONENT_DESC_PYTHON_DEPS]
        assert (
            comp_info.include_glob_patterns
            == comp_desc[json_fields.COMPONENT_DESC_INCLUDE_GLOB_PATTERNS]
        )
        assert (
            comp_info.exclude_glob_patterns
            == comp_desc[json_fields.COMPONENT_DESC_EXCLUDE_GLOB_PATTERNS]
        )

        assert len(comp_info.inputs) == 1
        assert comp_info.inputs[0].description == "Input string"
        assert comp_info.inputs[0].label == "just-string"
        assert comp_info.inputs[0].default_component == "TestDefComp"
        assert comp_info.inputs[0].type == "str"
        assert comp_info.inputs[0].group == "data"

        assert len(comp_info.outputs) == 1
        assert comp_info.outputs[0].description == "Output string"
        assert comp_info.outputs[0].label == "just-output-string"
        assert comp_info.outputs[0].default_component == "OutputConnector"
        assert comp_info.outputs[0].type == "str"
        assert comp_info.outputs[0].group == "data"

        assert len(comp_info.arguments) == 2
        assert comp_info.arguments[0].key == "exit-value"

        assert comp_info.arguments[1].description == "File to use for loading the model"
        assert comp_info.arguments[1].optional is False
        assert comp_info.get_argument("no-such-argument") is None
        assert comp_info.get_argument("exit-value").description == "Exit value to use"
