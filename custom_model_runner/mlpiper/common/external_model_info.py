import json
import re


word_regex_pattern = re.compile("[^A-Za-z]+")


class ExternalModelInfo:
    def __init__(self, path, format, descriptipn=None):
        self.model_path = path
        self.model_format = format
        self.description = descriptipn

    @staticmethod
    def _camel(name):
        words = word_regex_pattern.split(name)
        return "".join(w.lower() if i == 0 else w.title() for i, w in enumerate(words))

    def to_json(self, indent=0):
        members_in_camel = {}
        for k, v in self.__dict__.items():
            members_in_camel[self._camel(k)] = v

        return json.dumps(members_in_camel, indent=indent)
