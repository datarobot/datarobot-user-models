import os
import re

from mlpiper.pipeline import json_fields


def main_component_module(comp_desc):
    main_script_name = os.path.splitext(
        comp_desc[json_fields.COMPONENT_DESC_PROGRAM_FIELD]
    )[0]
    return re.sub(r"[/\\]", r".", main_script_name)


def assemble_cmdline_from_args(input_args):
    cmdline_list = []
    for arg in input_args:
        cmdline_list.append("--" + arg)
        cmdline_list.append(str(input_args[arg]))
    return cmdline_list
