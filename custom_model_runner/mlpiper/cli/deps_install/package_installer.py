import abc
import logging
import os
import tempfile


class PackageInstaller(object):
    def __init__(self, packages, script_prefix, script_extension):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._prefix = script_prefix
        self._extension = (
            script_extension
            if script_extension.startswith(".")
            else ".{}".format(script_extension)
        )
        self._packages = packages

    def install(self):
        if self._packages:
            _, requirements_path = tempfile.mkstemp(
                suffix=self._extension, prefix=self._prefix
            )
            self._logger.info("Requirements path: {}".format(requirements_path))
            try:
                self._generate_requirements_script(requirements_path)

                self._logger.info(
                    "Installing components dependencies ... {}".format(self._packages)
                )
                self._do_install(requirements_path)
                self._logger.info("Components dependencies installed successfully!")
            finally:
                os.remove(requirements_path)
        else:
            self._logger.info("No components dependencies found to install")

    def _generate_requirements_script(self, requirements_path):
        with open(requirements_path, "w") as f:
            f.write(self._script_content(self._packages))

    @abc.abstractmethod
    def _script_content(self, packages):
        """
        Returns the content that will be written to a temporary file, which then
        will be used by the install command
        """
        pass

    @abc.abstractmethod
    def _do_install(self, requirement_path):
        """
        Handle the installation itself.

        :param requirement_path: a full path for file/script that contains the content received from
                                 'self._script_content' (see above)
        :return: None
        :exception: Suppose to raise an exception in case on an error
        """
        pass
