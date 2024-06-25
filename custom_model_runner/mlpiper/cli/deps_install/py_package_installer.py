import logging
import subprocess
import sys

from .package_installer import PackageInstaller


class PyPackageInstaller(PackageInstaller):
    def __init__(self, packages):
        self._logger = logging.getLogger(self.__class__.__name__)
        super(PyPackageInstaller, self).__init__(
            packages, script_prefix="requirements-", script_extension=".txt"
        )

    def _script_content(self, packages):
        return "\n".join(packages)

    def _do_install(self, requirements_path):
        pip_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-color",
            "--disable-pip-version-check",
            "--requirement",
            requirements_path,
        ]

        self._logger.info(
            "Python dependencies installation, pip args: {}".format(pip_cmd)
        )
        cmd = "yes w | " + " ".join(pip_cmd)
        subprocess.call(cmd, shell=True)
