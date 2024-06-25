from mlpiper.pipeline import json_fields


class ComponentConnectionInfo(object):
    def __init__(self, connection_desc_json=None):
        self._description = None
        self._label = None
        self._default_component = None
        self._type = None
        self._group = None

        self.load_from_json(connection_desc_json)

    def load_from_json(self, connection_desc_json):
        if connection_desc_json:
            self.description = connection_desc_json.get(
                json_fields.CONNECTION_DESC_DESCRIPTION_FIELD
            )
            self.label = connection_desc_json.get(
                json_fields.CONNECTION_DESC_LABEL_FIELD
            )
            self.default_component = connection_desc_json.get(
                json_fields.CONNECTION_DESC_DEFAULT_COMPONENT_FIELD
            )
            self.type = connection_desc_json.get(json_fields.CONNECTION_DESC_TYPE_FIELD)
            self.group = connection_desc_json.get(
                json_fields.CONNECTION_DESC_GROUP_FIELD
            )

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def default_component(self):
        return self._default_component

    @default_component.setter
    def default_component(self, value):
        self._default_component = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, value):
        self._group = value
