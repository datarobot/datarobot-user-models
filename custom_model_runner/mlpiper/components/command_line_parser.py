import pkg_resources
import argparse
import json
from collections import namedtuple

from mlpiper.pipeline import json_fields


class Argument:
    def __init__(self, arg_description):
        self.key = arg_description["key"]
        self.label = arg_description["label"]
        self.type = arg_description["type"]
        self.optional = arg_description.get("optional", False)
        self.default_value = arg_description.get("defaultValue", None)


class CommandLineParser:
    """
    Parsing commandline parameters according to the component.json file provided to the component.
    """

    def __init__(self, pkg, component_json="component.json"):
        self._component_json = component_json
        self._pkg = pkg

    def parse_args(self):
        """
        Parse arguments and return an options dict
        :return: dict with all options
        """

        real_file_name = pkg_resources.resource_filename(
            self._pkg, self._component_json
        )

        parser = argparse.ArgumentParser()
        comp_json = json.load(open(real_file_name))
        args = comp_json[json_fields.COMPONENT_DESC_ARGUMENTS]
        for arg in args:
            parser.add_argument(
                "--" + arg[json_fields.COMPONENT_DESC_ARGUMENT_KEY],
                help=arg[json_fields.COMPONENT_DESC_DESCRIPTION_FIELD],
            )

        options = parser.parse_args()
        return options

    def params_to_obj(self, comp_desc, params_dict):
        """
        Helper to translate from parameter dict to an object
        This is like the object the argparse is returning
        :param comp_desc:
        :param params_dict:
        :return: object
        """
        args_desc = comp_desc["arguments"]
        options = {}

        for arg_desc in args_desc:
            arg_obj = Argument(arg_desc)
            if arg_obj.optional:
                options[arg_obj.key] = params_dict.get(
                    arg_obj.key, arg_obj.default_value
                )
            else:
                if arg_obj.key not in params_dict:
                    raise Exception(
                        "Argument: {} is not optional, and no value was given".format(
                            arg_obj.key
                        )
                    )
                options[arg_obj.key] = params_dict[arg_obj.key]
        return namedtuple("Options", options.keys())(*options.values())
