import os
import logging
import socket
from contextlib import closing
from jinja2 import Environment, BaseLoader, DebugUndefined
from datarobot_drum.drum.common import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class CMRunnerUtils:
    current_path = os.path.dirname(__file__)
    resource_path = os.path.abspath(os.path.join(current_path, "..", "resource"))
    assert os.path.exists(resource_path)

    def __init__(self):
        pass

    @classmethod
    def get_components_repo(cls):
        components_repo = os.path.join(cls.resource_path, "components")
        logger.debug("Components repo: {}".format(components_repo))
        return components_repo

    @classmethod
    def get_pipeline_filepath(cls, pipeline_name):
        pipeline_filepath = os.path.join(cls.resource_path, "pipelines", pipeline_name)
        logger.debug("Getting pipeline: {}".format(pipeline_filepath))
        return pipeline_filepath

    @classmethod
    def render_template_keep_undefined(cls, template_str, data):
        rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(
            template_str
        )
        return rtemplate.render(data)

    @classmethod
    def render_file(cls, filename, data):
        with open(filename) as f:
            file_str = f.read()
        return cls.render_template_keep_undefined(file_str, data)

    @classmethod
    def find_files_by_extensions(cls, dirpath, extensions):
        lst = []
        for filename in os.listdir(dirpath):
            path = os.path.join(dirpath, filename)
            if os.path.isdir(path):
                continue
            if any(filename.endswith(extension) for extension in extensions):
                lst.append(path)
        return lst

    @classmethod
    def filename_exists_and_is_file(cls, dirpath, filename):
        abs_filepath = os.path.abspath(os.path.join(dirpath, filename))
        return os.path.exists(abs_filepath) and os.path.isfile(abs_filepath)

    @classmethod
    def is_port_in_use(cls, host, port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            return s.connect_ex((host, port)) == 0

    @classmethod
    def find_free_port(cls):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @classmethod
    def replace_cmd_argument_value(cls, cmd_list, arg_name, arg_value):
        try:
            ind = cmd_list.index(arg_name)
        except ValueError:
            return
        cmd_list[ind + 1] = arg_value

    @classmethod
    def delete_cmd_argument(cls, cmd_list, arg_name):
        try:
            ind = cmd_list.index(arg_name)
            del cmd_list[ind : ind + 2]
        except ValueError:
            return
