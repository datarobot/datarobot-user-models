class EeArg(object):
    def __init__(self, ee_arg):
        self._ee_arg = ee_arg if ee_arg else dict()

    @property
    def value(self):
        return self._ee_arg.get("value")

    @property
    def arg_type(self):
        return self._ee_arg.get("type")

    @property
    def optional(self):
        return self._ee_arg.get("optional")

    @property
    def label(self):
        return self._ee_arg.get("label")

    @property
    def description(self):
        return self._ee_arg.get("description")

    @property
    def editable(self):
        return self._ee_arg.get("editable")
