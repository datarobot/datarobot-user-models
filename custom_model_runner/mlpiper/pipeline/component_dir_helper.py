import pkg_resources
import logging
import os

from mlpiper.common.base import Base


class ComponentDirHelper(Base):
    def __init__(self, pkg, main_program):
        """
        Extract component directory outside of egg, so an external command can run

        :param main_program: The main program to run. E.g. main.py
        :param pkg: The package the main_program is in, this is required in order to
                    extract the files of the componenbt outisde the egg
        """
        super(ComponentDirHelper, self).__init__(logging.getLogger(self.logger_name()))
        self._logger.debug("pkg: {}, main_program: {}".format(pkg, main_program))
        self._pkg = pkg
        self._main_program = main_program

    def extract_component_out_of_egg(self):
        """
        The artifact dir will contain all the files needed to run the R code.
        This method will copy the files outside of the egg into the artifact dir
        :return:
        """

        # TODO: check what happens if we have a directory inside the component dir
        ll = pkg_resources.resource_listdir(self._pkg, "")
        for file_name in ll:
            real_file_name = pkg_resources.resource_filename(self._pkg, file_name)
            self._logger.debug("file:      {}".format(file_name))
            self._logger.debug("real_file: {}".format(real_file_name))

        # Finding the directory we need to CD into
        base_file = os.path.basename(self._main_program)
        self._logger.debug("base_file: {}".format(base_file))
        real_file_name = pkg_resources.resource_filename(self._pkg, base_file)
        component_extracted_dir = os.path.dirname(real_file_name)
        self._logger.debug("Extraction dir: {}".format(component_extracted_dir))
        self._logger.debug("Done building artifact dir:")
        self._logger.debug("======================")
        return component_extracted_dir
