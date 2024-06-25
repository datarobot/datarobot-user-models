import os
import pickle
import logging
import sys

from mlpiper.common.base import Base


class ConnectableExternalComponent(Base):

    MLPIPER_COMPONENT_INFO_FILE_PATH_ENV = "MLPIPER_COMP_INFO_FILE"
    MLPIPER_PYTHON = "MLPIPER_PYTHON"

    def __init__(self):
        super(ConnectableExternalComponent, self).__init__(
            logging.getLogger(self.logger_name())
        )
        self._comp_info = None

    def _set_info_file(self, info_file_path):
        """
        Setting the info file to use.
        :param info_file_path:
        :return:
        """
        os.environ[
            ConnectableExternalComponent.MLPIPER_COMPONENT_INFO_FILE_PATH_ENV
        ] = info_file_path
        os.environ[ConnectableExternalComponent.MLPIPER_PYTHON] = sys.executable

    def save_component_info(self, comp_info, info_file_path=None):
        if not info_file_path:
            if (
                ConnectableExternalComponent.MLPIPER_COMPONENT_INFO_FILE_PATH_ENV
                not in os.environ
            ):
                raise Exception(
                    "Environment does not contain {}".format(
                        ConnectableExternalComponent.MLPIPER_COMPONENT_INFO_FILE_PATH_ENV
                    )
                )

            info_file_path = os.environ[
                ConnectableExternalComponent.MLPIPER_COMPONENT_INFO_FILE_PATH_ENV
            ]
        else:
            self._set_info_file(info_file_path)

        with open(info_file_path, "wb") as pickle_file:
            # [REF-5619] Force pickle protocol = 2, because it is not known which py is
            # used in external program. Requires more investigation, because in case of R
            # program, protocol 3 is used, despite of program info shows py2 is used.
            pickle.dump(comp_info, pickle_file, 2)

    def load_component_info(self, info_file_path=None):
        if not info_file_path:
            if (
                ConnectableExternalComponent.MLPIPER_COMPONENT_INFO_FILE_PATH_ENV
                not in os.environ
            ):
                raise Exception(
                    "Environment does not contain {}".format(
                        ConnectableExternalComponent.MLPIPER_COMPONENT_INFO_FILE_PATH_ENV
                    )
                )

            info_file_path = os.environ[
                ConnectableExternalComponent.MLPIPER_COMPONENT_INFO_FILE_PATH_ENV
            ]
        if not os.path.exists(info_file_path):
            raise Exception(
                "Component info file {} does not exists".format(info_file_path)
            )
        with open(info_file_path, "rb") as pickle_file:
            self._comp_info = pickle.load(pickle_file)
        return self._comp_info

    # Helper functions for the component code. So assuming the following methods are called
    # when there is a valid _comp_info

    def get_params(self):
        """
        Return the calling component parameters.
        Assumption here is that there are files with pickle objects which were
        prepared for the calling component to access
        :return: a dictionary with parameters and parameters values
        """
        if not self._comp_info:
            self.load_component_info()
        return self._comp_info.params

    def get_parents_objs(self):
        """
        Return a list of objects provided by parent components
        :return: list of objects provided by parent components
        """
        if not self._comp_info:
            self.load_component_info()

        return self._comp_info.parents_objs

    def set_output_objs(self, *output_objs):
        """
        Set output objects for calling components
        :param obj_list: The calling component list of output objects
        """

        # TODO: set the output_obj and save the comp_info object using pickle
        if not self._comp_info:
            self.load_component_info()
        self._comp_info.output_objs = list(output_objs)
        self.save_component_info(self._comp_info)
