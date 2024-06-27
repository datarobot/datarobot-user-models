"""
For internal use only. The nginx broker class is designed to handle any 'nginx' related actions,
such as setup, configuration and execution
"""
import os
import subprocess
from pathlib import Path
import platform
import re
import shutil

from mlpiper.common.base import Base
from mlpiper.components.restful import util
from mlpiper.components.restful.nginx_conf_template import (
    NGINX_SERVER_CONF_TEMPLATE,
    NGINX_CONF_TEMPLATE_NON_ROOT,
)
from mlpiper.components.restful.constants import (
    SharedConstants,
    ComponentConstants,
    NginxConstants,
)
from mlpiper.common.mlpiper_exception import MLPiperException


class NginxBroker(Base):
    origin_nginx_conf_filepath_pattern = "{}/nginx.conf"
    new_nginx_conf_filepath_pattern = "{}/nginx.conf.new"
    bin_search_paths = [Path("/usr/bin"), Path("/sbin"), Path("/bin"), Path("/usr/sbin")]

    def __init__(self, ml_engine, dry_run=False):
        super(NginxBroker, self).__init__()
        self.set_logger(ml_engine.get_engine_logger(self.logger_name()))
        self._dry_run = dry_run
        self._conf_file = None
        self._root_user = os.getuid() == 0
        self.__debian_platform = None
        self.__redhat_platform = None

    def setup_and_run(self, shared_conf, nginx_conf):
        self._logger.info("Setup 'nginx' service ...")
        self._verify_dependencies()
        self._generate_configuration(shared_conf, nginx_conf)
        self._run(shared_conf)
        return self

    def quit(self):
        if not self._dry_run:
            self._logger.info("Stopping 'nginx' service ...")
            try:
                if self._root_user:
                    stop_cmd = NginxConstants.STOP_CMD
                else:
                    stop_cmd = "nginx -c {} -s stop".format(self._conf_file)
                subprocess.call(stop_cmd, shell=True)
            except:  # noqa: E722
                # Should catch any exception in order to avoid masking of other important errors
                # in the system
                pass

    def _verify_dependencies(self):
        util.verify_tool_installation(
            NginxConstants.VER_CMD, NginxConstants.DEV_AGAINST_VERSION, self._logger
        )

    def _generate_configuration(self, shared_conf, nginx_conf):
        access_log_off = (
            NginxConstants.ACCESS_LOG_OFF_CONFIG
            if nginx_conf[NginxConstants.DISABLE_ACCESS_LOG_KEY]
            else ""
        )

        if not self._root_user and not self._debian_platform():
            self._logger.warning(
                "Running as non root was tested only for Ubuntu platform."
                "You may need to change permissions to some of nginx folders."
            )

        conf_content = NGINX_SERVER_CONF_TEMPLATE.format(
            port=nginx_conf[ComponentConstants.PORT_KEY],
            sock_filepath=os.path.join(
                shared_conf[SharedConstants.TARGET_PATH_KEY],
                shared_conf[SharedConstants.SOCK_FILENAME_KEY],
            ),
            access_log_off=access_log_off,
            uwsgi_params_prefix=NginxConstants.NGINX_ROOT + "/",
        )

        # if user is root configuration will be written to nginx system path
        if self._root_user:
            self._conf_file = self._server_conf_filepath()
        # else it will be written to /tmp folder.
        else:
            conf_content = NGINX_CONF_TEMPLATE_NON_ROOT.format(
                nginx_server_conf_placeholder=conf_content
            )
            self._conf_file = os.path.join(
                shared_conf[SharedConstants.TARGET_PATH_KEY], NginxConstants.SERVER_CONF_FILENAME
            )

        self._logger.info("Writing nginx server configuration to ... {}".format(self._conf_file))
        with open(self._conf_file, "w") as f:
            f.write(conf_content)

        if self._root_user:
            if self._debian_platform() or self._redhat_platform():
                # Newer versions of nginx requires the folder sites-enabled in their installation
                # folder, in order to enable extended server configurations, which are configured
                # in conf.d.
                # Apparently, on 'redhat' platforms the given folder does not exits after nginx
                # installation.
                if not os.path.exists(NginxConstants.SERVER_ENABLED_DIR):
                    NginxBroker._fix_missing_sites_enabled_conf(NginxConstants.NGINX_ROOT)

                sym_link = os.path.join(
                    NginxConstants.SERVER_ENABLED_DIR, NginxConstants.SERVER_CONF_FILENAME
                )
                if not os.path.isfile(sym_link):
                    self._logger.info("Creating nginx server sym link ... {}".format(sym_link))
                    os.symlink(self._conf_file, sym_link)

        self._logger.info("Done with _generate_configuration ...")

    @staticmethod
    def _fix_missing_sites_enabled_conf(nginx_root):
        origin_nginx_conf_filepath = NginxBroker.origin_nginx_conf_filepath_pattern.format(
            nginx_root
        )
        new_nginx_conf_filepath = NginxBroker.new_nginx_conf_filepath_pattern.format(nginx_root)

        fix_configuration = True
        pattern_conf_d = re.compile(r"^\s*include\s+{}/conf\.d/\*\.conf;\s*$".format(nginx_root))
        pattern_sites_enabled = re.compile(
            r"^\s*include\s+{}/sites-enabled/.*;\s*$".format(nginx_root)
        )
        line_to_add = "    include {}/sites-enabled/*;\n".format(nginx_root)

        with open(origin_nginx_conf_filepath, "r") as fr:
            with open(new_nginx_conf_filepath, "w") as fw:
                for line in fr:
                    fw.write(line)
                    group = pattern_conf_d.match(line)
                    if group:
                        fw.write(line_to_add)
                    elif pattern_sites_enabled.match(line):
                        # sites-enabled already configured! Close and remove new file"
                        fix_configuration = False
                        break

        if fix_configuration:
            shutil.copyfile(new_nginx_conf_filepath, origin_nginx_conf_filepath)

        if os.path.exists(new_nginx_conf_filepath):
            os.remove(new_nginx_conf_filepath)

        os.mkdir(NginxConstants.SERVER_ENABLED_DIR, 0o644)

    def _server_conf_filepath(self):
        if self._debian_platform():
            d = NginxConstants.SERVER_CONF_DIR_DEBIAN
        elif self._redhat_platform():
            d = NginxConstants.SERVER_CONF_DIR_REDHAT
        elif self._macos_platform():
            if not os.path.isdir(NginxConstants.SERVER_CONF_DIR_MACOS):
                if not os.path.isdir(NginxConstants.NGINX_ROOT_MACOS):
                    raise MLPiperException(
                        "'{}' does not exist or not a directory. Is nginx installed?".format(
                            NginxConstants.NGINX_ROOT_MACOS
                        )
                    )
                os.mkdir(NginxConstants.SERVER_CONF_DIR_MACOS)
            d = NginxConstants.SERVER_CONF_DIR_MACOS
        else:
            raise MLPiperException(
                "Nginx cannot be configured! Platform is not supported: {}".format(platform.uname())
            )

        return os.path.join(d, NginxConstants.SERVER_CONF_FILENAME)

    def _redhat_platform(self):
        # There are a lot of ways to try and determine the Linux distro. The modern way is to
        # parse `/etc/os-release` but that doesn't support very old OSes (i.e. some versions of
        # CentOS/RHEL) and also seems more involved than we need for our purposes. Since all we
        # care about is how the Nginx application is packaged, we can just try and look for the
        # low-level package managers that are well-known.
        #
        # Note: looking at the kernel (e.g. platform.uname()) would be **incorrect** since we
        # support running in a container and it's possible the host OS (kernel) could be RedHat
        # but the container is Debian, for example. The most correct way to determine the distro
        # is to inspect the filesystem in some way (http://0pointer.de/blog/projects/os-release).
        if self.__redhat_platform is None:
            for path in self.bin_search_paths:
                if (path / "rpm").exists():
                    self.__redhat_platform = True
                    break
            else:
                self.__redhat_platform = False
        return self.__redhat_platform

    def _debian_platform(self):
        # All Debian derived systems will have `dpkg` installed into the path.
        if self.__debian_platform is None:
            for path in self.bin_search_paths:
                if (path / "dpkg").exists():
                    self.__debian_platform = True
                    break
            else:
                self.__debian_platform = False
        return self.__debian_platform

    def _macos_platform(self):
        # MacOS doesn't support containers so we can just look at the kernel to determine
        # the platform.
        return "Darwin" in platform.system()  # platform.system() caches its output

    def _run(self, shared_conf):
        self._logger.info("Starting 'nginx' service ... cmd: '{}'".format(NginxConstants.START_CMD))
        if self._dry_run:
            return

        if self._root_user:
            start_cmd = NginxConstants.START_CMD
        else:
            start_cmd = "nginx -c {} -p $PWD".format(self._conf_file)
        rc = subprocess.check_call(start_cmd, shell=True)
        if rc != 0:
            raise MLPiperException(
                "nginx service failed to start! It is suspected as not being installed!"
            )

        self._logger.info("'nginx' service started successfully!")


if __name__ == "__main__":
    import logging
    import tempfile

    root_dir = "/tmp/nginx-test"

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    shared_conf = {
        SharedConstants.TARGET_PATH_KEY: tempfile.mkdtemp(prefix="restful-", dir=root_dir),
        SharedConstants.SOCK_FILENAME_KEY: "restful_mlapp.sock",
    }

    nginx_conf = {
        ComponentConstants.HOST_KEY: "localhost",
        ComponentConstants.PORT_KEY: 8888,
    }

    print("Target path: {}".format(shared_conf[SharedConstants.TARGET_PATH_KEY]))

    logging.basicConfig()
    logger = logging.getLogger("NginxBroker")
    logger.setLevel(logging.INFO)
    NginxBroker(logger, dry_run=True).setup_and_run(shared_conf, nginx_conf)
