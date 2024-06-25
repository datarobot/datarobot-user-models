# flake8: noqa C901
from mlpiper.pipeline import json_fields

from mlpiper.pipeline.component_connection_info import ComponentConnectionInfo
from mlpiper.pipeline.component_argument_info import ComponentArgumentInfo
from collections import OrderedDict

import json
import six


class ComponentInfoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ComponentInfo):
            ret = OrderedDict()
            if obj.name:
                ret[json_fields.COMPONENT_DESC_NAME_FIELD] = obj.name
            if obj.engine_type:
                ret[json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD] = obj.engine_type
            if obj.language:
                ret[json_fields.COMPONENT_DESC_LANGUAGE_FIELD] = obj.language
            if obj.group:
                ret[json_fields.COMPONENT_DESC_GROUP_FIELD] = obj.group
            if obj.label:
                ret[json_fields.COMPONENT_DESC_LABEL_FIELD] = obj.label
            if obj.description:
                ret[json_fields.COMPONENT_DESC_DESCRIPTION_FIELD] = obj.description
            if obj.version:
                ret[json_fields.COMPONENT_DESC_VERSION_FIELD] = obj.version
            if obj.user_standalone:
                ret[json_fields.COMPONENT_DESC_USER_STAND_ALONE] = obj.user_standalone
            if obj.program:
                ret[json_fields.COMPONENT_DESC_PROGRAM_FIELD] = obj.program
            if obj.component_class:
                ret[json_fields.COMPONENT_DESC_CLASS_FIELD] = obj.component_class
            if obj.model_behavior:
                ret[
                    json_fields.COMPONENT_DESC_MODEL_BEHAVIOR_FIELD
                ] = obj.model_behavior
            if obj.use_mlops:
                ret[json_fields.COMPONENT_DESC_USE_MLOPS_FIELD] = obj.use_mlops
            if obj.deps:
                ret[json_fields.COMPONENT_DESC_PYTHON_DEPS] = obj.deps
            if obj.include_glob_patterns:
                ret[
                    json_fields.COMPONENT_DESC_INCLUDE_GLOB_PATTERNS
                ] = obj.include_glob_patterns
            if obj.exclude_glob_patterns:
                ret[
                    json_fields.COMPONENT_DESC_EXCLUDE_GLOB_PATTERNS
                ] = obj.exclude_glob_patterns
            if obj.inputs:
                ret[json_fields.COMPONENT_DESC_INPUT_INFO_FIELD] = obj.inputs
            if obj.outputs:
                ret[json_fields.COMPONENT_DESC_OUTPUT_INFO_FIELD] = obj.outputs
            if obj.arguments:
                ret[json_fields.COMPONENT_DESC_ARGUMENTS] = obj.arguments
            return ret
        elif isinstance(obj, ComponentConnectionInfo):
            ret = OrderedDict()
            if obj.description:
                ret[json_fields.CONNECTION_DESC_DESCRIPTION_FIELD] = obj.description
            if obj.label:
                ret[json_fields.CONNECTION_DESC_LABEL_FIELD] = obj.label
            if obj.default_component:
                ret[
                    json_fields.CONNECTION_DESC_DEFAULT_COMPONENT_FIELD
                ] = obj.default_component
            if obj.type:
                ret[json_fields.CONNECTION_DESC_TYPE_FIELD] = obj.type
            if obj.group:
                ret[json_fields.CONNECTION_DESC_GROUP_FIELD] = obj.group
            return ret
        elif isinstance(obj, ComponentArgumentInfo):
            ret = OrderedDict()
            if obj.key:
                ret[json_fields.COMPONENT_DESC_ARGUMENT_KEY] = obj.key
            if obj.type:
                ret[json_fields.COMPONENT_DESC_ARGUMENT_TYPE] = obj.type
            if obj.label:
                ret[json_fields.COMPONENT_DESC_ARGUMENT_LABEL] = obj.label
            if obj.description:
                ret[json_fields.COMPONENT_DESC_ARGUMENT_DESCRIPTION] = obj.description
            if obj.env_var:
                ret[json_fields.COMPONENT_DESC_ARGUMENT_ENV_VAR] = obj.env_var
            if obj.optional:
                ret[json_fields.COMPONENT_DESC_ARGUMENT_OPTIONAL] = obj.optional
            if obj.tag:
                ret[json_fields.COMPONENT_DESC_ARGUMENT_TAG] = obj.tag
            if obj.default_value:
                ret[json_fields.COMPONENT_DESC_ARGUMENT_DEFAULT_VAL] = obj.default_value
            return ret
        else:
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)


class ComponentInfo(object):
    def __init__(self, comp_desc_json=None):
        self._name = None
        self._engine_type = None
        self._language = None
        self._group = None
        self._label = None
        self._description = None
        self._version = None
        self._user_standalone = None
        self._program = None
        self._component_class = None
        self._model_behavior = None
        self._use_mlops = None
        self._deps = None
        self._include_glob_patterns = None
        self._exclude_glob_patterns = None
        self._inputs = None
        self._outputs = None
        self._arguments = None

        self.load_from_json(comp_desc_json)

    def load_from_json(self, comp_desc_json):
        if comp_desc_json:
            self.version = comp_desc_json.get(json_fields.COMPONENT_DESC_VERSION_FIELD)
            self.engine_type = comp_desc_json.get(
                json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD
            )
            self.language = comp_desc_json.get(
                json_fields.COMPONENT_DESC_LANGUAGE_FIELD
            )
            self.user_standalone = comp_desc_json.get(
                json_fields.COMPONENT_DESC_USER_STAND_ALONE
            )
            self.name = comp_desc_json.get(json_fields.COMPONENT_DESC_NAME_FIELD)
            self.label = comp_desc_json.get(json_fields.COMPONENT_DESC_LABEL_FIELD)
            self.description = comp_desc_json.get(
                json_fields.COMPONENT_DESC_DESCRIPTION_FIELD
            )
            self.program = comp_desc_json.get(json_fields.COMPONENT_DESC_PROGRAM_FIELD)
            self.component_class = comp_desc_json.get(
                json_fields.COMPONENT_DESC_CLASS_FIELD
            )
            self.model_behavior = comp_desc_json.get(
                json_fields.COMPONENT_DESC_MODEL_BEHAVIOR_FIELD
            )
            self.use_mlops = comp_desc_json.get(
                json_fields.COMPONENT_DESC_USE_MLOPS_FIELD
            )
            self.group = comp_desc_json.get(json_fields.COMPONENT_DESC_GROUP_FIELD)
            self._deps = comp_desc_json.get(json_fields.COMPONENT_DESC_PYTHON_DEPS)
            self._include_glob_patterns = comp_desc_json.get(
                json_fields.COMPONENT_DESC_INCLUDE_GLOB_PATTERNS
            )
            self._exclude_glob_patterns = comp_desc_json.get(
                json_fields.COMPONENT_DESC_EXCLUDE_GLOB_PATTERNS
            )

            self.version = str(self.version) if self.version is not None else None

            input_connections = comp_desc_json.get(
                json_fields.COMPONENT_DESC_INPUT_INFO_FIELD
            )
            if input_connections:
                self._inputs = []
                for input_conn_json in input_connections:
                    ic = ComponentConnectionInfo(input_conn_json)
                    self._inputs.append(ic)

            output_connections = comp_desc_json.get(
                json_fields.COMPONENT_DESC_OUTPUT_INFO_FIELD
            )
            if output_connections:
                self._outputs = []
                for output_conn_json in output_connections:
                    oc = ComponentConnectionInfo(output_conn_json)
                    self._outputs.append(oc)

            arguments = comp_desc_json.get(json_fields.COMPONENT_DESC_ARGUMENTS)
            if arguments:
                self._arguments = []
                for arg in arguments:
                    oc = ComponentArgumentInfo(arg)
                    self._arguments.append(oc)

    def get_argument(self, key):
        if key is None:
            return None
        if not isinstance(key, six.string_types):
            return None

        if self._arguments is None or len(self._arguments) == 0:
            return None

        for arg in self._arguments:
            if arg.key == key:
                return arg
        return None

    @staticmethod
    def is_valid(comp_desc):
        comp_desc_signature = [
            json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD,
            json_fields.COMPONENT_DESC_NAME_FIELD,
            json_fields.COMPONENT_DESC_LANGUAGE_FIELD,
            json_fields.COMPONENT_DESC_PROGRAM_FIELD,
        ]
        if set(comp_desc_signature) <= set(comp_desc):
            return True
        return False

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        self._version = value

    @property
    def engine_type(self):
        return self._engine_type

    @engine_type.setter
    def engine_type(self, value):
        self._engine_type = value

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value

    @property
    def user_standalone(self):
        return self._user_standalone

    @user_standalone.setter
    def user_standalone(self, value):
        self._user_standalone = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def program(self):
        return self._program

    @program.setter
    def program(self, value):
        self._program = value

    @property
    def component_class(self):
        return self._component_class

    @component_class.setter
    def component_class(self, value):
        self._component_class = value

    @property
    def model_behavior(self):
        return self._model_behavior

    @model_behavior.setter
    def model_behavior(self, value):
        self._model_behavior = value

    @property
    def use_mlops(self):
        return self._use_mlops

    @use_mlops.setter
    def use_mlops(self, value):
        self._use_mlops = value

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, value):
        self._group = value

    @property
    def deps(self):
        return self._deps

    @deps.setter
    def deps(self, value):
        self._deps = value

    @property
    def include_glob_patterns(self):
        return self._include_glob_patterns

    @include_glob_patterns.setter
    def include_glob_patterns(self, value):
        self._include_glob_patterns = value

    @property
    def exclude_glob_patterns(self):
        return self._exclude_glob_patterns

    @exclude_glob_patterns.setter
    def exclude_glob_patterns(self, value):
        self._exclude_glob_patterns = value

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    @property
    def arguments(self):
        return self._arguments

    @arguments.setter
    def arguments(self, value):
        self._arguments = value
