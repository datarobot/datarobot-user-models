from mlpiper.pipeline import json_fields


class ComponentArgumentInfo(object):
    def __init__(self, argument_desc_json=None):
        self._key = None
        self._type = None
        self._label = None
        self._description = None
        self._env_var = None
        self._optional = None
        self._tag = None
        self._default_value = None

        self.load_from_json(argument_desc_json)

    def load_from_json(self, argument_desc_json):
        if argument_desc_json:
            self.key = argument_desc_json.get(json_fields.COMPONENT_DESC_ARGUMENT_KEY)
            self.type = argument_desc_json.get(json_fields.COMPONENT_DESC_ARGUMENT_TYPE)
            self.label = argument_desc_json.get(
                json_fields.COMPONENT_DESC_ARGUMENT_LABEL
            )
            self.description = argument_desc_json.get(
                json_fields.COMPONENT_DESC_ARGUMENT_DESCRIPTION
            )
            self.env_var = argument_desc_json.get(
                json_fields.COMPONENT_DESC_ARGUMENT_ENV_VAR
            )
            self.optional = argument_desc_json.get(
                json_fields.COMPONENT_DESC_ARGUMENT_OPTIONAL
            )
            self.tag = argument_desc_json.get(json_fields.COMPONENT_DESC_ARGUMENT_TAG)
            self.default_value = argument_desc_json.get(
                json_fields.COMPONENT_DESC_ARGUMENT_DEFAULT_VAL
            )

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

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
    def env_var(self):
        return self._env_var

    @env_var.setter
    def env_var(self, value):
        self._env_var = value

    @property
    def optional(self):
        return self._optional

    @optional.setter
    def optional(self, value):
        self._optional = value

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = value

    @property
    def default_value(self):
        return self._default_value

    @default_value.setter
    def default_value(self, value):
        self._default_value = value
