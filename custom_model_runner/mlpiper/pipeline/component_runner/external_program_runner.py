import os
import subprocess
import sys
import logging

from mlpiper.common.base import Base
from mlpiper.components.connected_component_info import ConnectedComponentInfo
from mlpiper.components.external_component import ext_comp


class ExternalProgramRunner(Base):
    """
    Running external code (like R or maybe jar)
    """

    def __init__(self, root_path, main_program, run_command=None):
        """
        Run an external program setting its command line arguments to be like the expected
        from a component.

        :param root_path: The root path the main_program is in, this is required in order
                          to construct the full path to component's files
        :param main_program: The main program to run. E.g. main.py
        :param run_command: An optional run command. This command will be used to run
                            the main program. For example: /usr/bin/Rscript
        """
        super(ExternalProgramRunner, self).__init__(
            logging.getLogger(self.logger_name())
        )

        self._root_path = root_path
        self._main_program = main_program
        self._run_command = run_command

        if run_command and not os.path.isfile(self._run_command):
            raise Exception(
                "Run command {} does not exists or is not a file".format(
                    self._run_command
                )
            )

        self._logger.info("Creating artifact dir: {}".format(os.getcwd()))

    def run_standalone(self, cmd_args):
        """
        Running a standalone external component - all goes via command line args
        :return: exit status of the external command
        """
        cmd = []
        if self._run_command:
            cmd.append(self._run_command)
        cmd.append(os.path.join(self._root_path, self._main_program))
        cmd.extend(cmd_args)
        return self._run_external_process(cmd)

    def _run_external_process(self, cmd):
        self.info("CMD: {}".format(cmd))

        # Save env variables should be passed
        self._logger.info("================== External code start ==================")
        sys.stdout.flush()
        p = subprocess.Popen(cmd)
        p.wait()
        self._logger.info(
            "================= External code done: ret: {} =================".format(
                p.returncode
            )
        )

        sys.stdout.flush()
        if p.returncode != 0:
            self._logger.info(
                "Connector: got external program exit code: {}".format(p.returncode)
            )
        return p.returncode

    def run_connected(self, parents_objects, input_args):
        """
        Run the external program, provide input from parent components and get output
        to child components. This is required for connected external components supports
        :return: ret_val, output_obj : exit status of external command + list of output objects
        """
        self._logger.info("Running connectable external component")
        # TODO: we can support various "pickle" like serialization libraries
        self._gen_component_info(parents_objects, input_args)

        cmd = []
        if self._run_command:
            cmd.append(self._run_command)
        cmd.append(os.path.join(self._root_path, self._main_program))
        ret_val = self._run_external_process(cmd)
        output_objs = self._read_component_info()
        return ret_val, output_objs

    def _gen_component_info(self, parents_objects, input_args):
        self._logger.info("Generating ConnectedComponentInfo pickled object")

        comp_info = ConnectedComponentInfo()
        comp_info.params = input_args
        comp_info.parents_objs = parents_objects

        comp_info_file = os.path.join(self._root_path, "component_info.pkl")
        self._logger.info("Component info file: {}".format(comp_info_file))
        ext_comp.save_component_info(comp_info, comp_info_file)

    def _read_component_info(self):
        self._logger.info("Reading from ConnectedComponentInfo pickled object")
        comp_info = ext_comp.load_component_info()
        self._logger.debug("Object: {}".format(comp_info))
        return comp_info.output_objs
