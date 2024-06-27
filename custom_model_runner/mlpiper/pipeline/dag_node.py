import os

from mlpiper.common.base import Base
from mlpiper.pipeline import json_fields, java_mapping


class DagNode(Base):
    def __init__(self, main_cls, comp_desc, pipe_comp, ml_engine):
        super(DagNode, self).__init__(ml_engine.get_engine_logger(self.logger_name()))
        self._main_cls = main_cls
        self._component_runner = None
        self._comp_desc = comp_desc  # Taken from component.json
        self._pipe_comp = pipe_comp  # A component section from the pipeline

        self._temp_visit = False  # Used by 'TopologicalSort'
        self._perm_visit = False  # Used by 'TopologicalSort'
        self._ml_engine = ml_engine

    def __str__(self):
        return "pipe id: {}\ncomp_desc: {}".format(self.pipe_id(), self.comp_desc())

    @property
    def component_runner(self):
        return self._component_runner

    @component_runner.setter
    def component_runner(self, component_runner):
        self._component_runner = component_runner

    def comp_desc(self):
        return self._comp_desc

    def comp_name(self):
        return self.comp_desc()[json_fields.COMPONENT_DESC_NAME_FIELD]

    def comp_label(self):
        return self.comp_desc()[json_fields.COMPONENT_DESC_LABEL_FIELD]

    def comp_program(self):
        return self.comp_desc()[json_fields.COMPONENT_DESC_PROGRAM_FIELD]

    def comp_root_path(self):
        return self.comp_desc()[json_fields.COMPONENT_DESC_ROOT_PATH_FIELD]

    def comp_class(self):
        return self.comp_desc()[json_fields.COMPONENT_DESC_CLASS_FIELD]

    def comp_language(self):
        return self.comp_desc()[json_fields.COMPONENT_DESC_LANGUAGE_FIELD]

    def pipe_comp(self):
        return self._pipe_comp

    def main_cls(self, main_cls=None):
        if main_cls:
            self._main_cls = main_cls
        return self._main_cls

    def pipe_id(self):
        return self._pipe_comp[json_fields.PIPELINE_COMP_ID_FIELD]

    def parents(self):
        return self._pipe_comp[json_fields.PIPELINE_COMP_PARENTS_FIELD]

    def _get_only_component_args(self, arguments):
        arguments_copy = arguments.copy()
        comp_args_orig = self._comp_desc[json_fields.COMPONENT_DESC_ARGUMENTS]
        comp_args_by_key = {}
        for arg in comp_args_orig:
            comp_args_by_key[arg[json_fields.COMPONENT_DESC_ARGUMENT_KEY]] = arg

        for arg in arguments:
            if arg not in comp_args_by_key:
                arguments_copy.pop(arg, None)
        return arguments_copy

    def input_arguments(self, system_conf, ee_conf, comp_only_args=False):
        """
        Get the input arguments for the node
        :param system_conf: The system configuration dict
        :param comp_only_args: If True, only include the argument which are reported by
                               the component. Otherwise include all arguments.
        :return: Arguments dict
        """
        merged_arguments = self._apply_java_tags_mapping(system_conf)
        merged_arguments.update(ee_conf)
        self._update_pipeline_arguments(merged_arguments)
        self._update_default_values(merged_arguments)
        if comp_only_args:
            merged_arguments = self._get_only_component_args(merged_arguments)
        return merged_arguments

    def _update_default_values(self, input_args):
        comp_args = self._comp_desc[json_fields.COMPONENT_DESC_ARGUMENTS]
        for comp_arg in comp_args:
            arg_key = comp_arg[json_fields.COMPONENT_DESC_ARGUMENT_KEY]
            # If argument value was not provided in pipeline:
            # - try to get value from env var, if such option defined for this argument
            # - try to get default value, if env var is not defined
            if (
                arg_key not in input_args
                and json_fields.COMPONENT_DESC_ARGUMENT_ENV_VAR in comp_arg
            ):
                env_var_value = os.environ.get(
                    comp_arg[json_fields.COMPONENT_DESC_ARGUMENT_ENV_VAR]
                )
                if env_var_value:
                    input_args[arg_key] = env_var_value
            if (
                arg_key not in input_args
                and json_fields.COMPONENT_DESC_ARGUMENT_DEFAULT_VAL in comp_arg
            ):
                input_args[arg_key] = comp_arg[json_fields.COMPONENT_DESC_ARGUMENT_DEFAULT_VAL]

    def _apply_java_tags_mapping(self, system_conf):
        system_conf = system_conf.copy()
        comp_args = self._comp_desc[json_fields.COMPONENT_DESC_ARGUMENTS]
        tagged_args = [arg for arg in comp_args if json_fields.COMPONENT_DESC_ARGUMENT_TAG in arg]
        for tagged_arg in tagged_args:
            py_tag = tagged_arg[json_fields.COMPONENT_DESC_ARGUMENT_TAG]
            if py_tag in java_mapping.TAGS:
                java_key = java_mapping.TAGS[py_tag]
                py_key = tagged_arg[json_fields.COMPONENT_DESC_ARGUMENT_KEY]
                system_conf[py_key] = system_conf[java_key]
                system_conf[java_mapping.RESERVED_KEYS[py_tag]] = system_conf[java_key]
                self._logger.debug(
                    "Replaced system config key: {} => {}, value: {}".format(
                        java_key, py_key, system_conf[py_key]
                    )
                )
                del system_conf[java_key]
        return system_conf

    def _update_pipeline_arguments(self, merged_arguments):
        for k, v in self._arguments().items():
            if k in merged_arguments:
                if v:
                    merged_arguments[k] = v
            else:
                merged_arguments[k] = v

    def _arguments(self):
        return self._pipe_comp[json_fields.PIPELINE_COMP_ARGUMENTS_FIELD]
