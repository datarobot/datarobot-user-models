import logging
from string import Template
import subprocess

from .package_installer import PackageInstaller


class RPackageInstaller(PackageInstaller):
    INSTALL_SCRIPT = Template(
        """
        requiredPackages = c($packages)
        for(p in requiredPackages){
          if(!require(p,character.only = TRUE)) {
            install.packages(p, repos = 'http://cran.rstudio.com/', type = 'source')
          }
        }
    """
    )

    def __init__(self, packages):
        self._logger = logging.getLogger(self.__class__.__name__)
        super(RPackageInstaller, self).__init__(
            packages, script_prefix="requirements-", script_extension=".R"
        )

    def _script_content(self, packages):
        return RPackageInstaller.INSTALL_SCRIPT.substitute(
            packages=", ".join(["'%s'" % p for p in packages])
        )

    def _do_install(self, requirements_path):
        cmd = "/usr/bin/env Rscript {}".format(requirements_path)
        self._logger.info("R dependencies installation command: {}".format(cmd))
        subprocess.check_call(cmd, shell=True)
