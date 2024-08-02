from configparser import ConfigParser

import tempfile

import socket

from unittest.mock import Mock
from parameterized import parameterized


from datarobot_drum.resource.components.Python.uwsgi_component.restful.uwsgi_broker import (
    UwsgiBroker,
)
from datarobot_drum.resource.components.Python.uwsgi_component.restful.constants import (
    SharedConstants,
    ComponentConstants,
    UwsgiConstants,
)


class TestUwsgiBroker:
    @parameterized.expand(
        [
            ["7", 7],
            ["1", None],
        ]
    )
    def test_generate_ini_file_(self, expected_amount_of_uwsgi_threads, uwsgi_threads):
        ml_engine = Mock()
        uwsgi_broker = UwsgiBroker(ml_engine)
        params = {}
        if uwsgi_threads:
            params[ComponentConstants.UWSGI_THREADS] = uwsgi_threads

        logging_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        entry_point_conf = {
            UwsgiConstants.PARAMS_KEY: params,
            ComponentConstants.METRICS_KEY: "total",
            ComponentConstants.SINGLE_UWSGI_WORKER_KEY: True,
            UwsgiConstants.LOGGING_UDP_SOCKET: logging_socket,
            ComponentConstants.UWSGI_DISABLE_LOGGING_KEY: True,
        }

        shared_conf = {
            SharedConstants.SOCK_FILENAME_KEY: "SOCK_FILENAME_KEY",
            SharedConstants.STATS_SOCK_FILENAME_KEY: "STATS_SOCK_FILENAME_KEY",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            uwsgi_broker._target_path = tmp_dir
            uwsgi_ini = uwsgi_broker._generate_ini_file(shared_conf, entry_point_conf)
            with open(uwsgi_ini) as f:
                print(f.read())

            config = ConfigParser(strict=False)
            config.read(uwsgi_ini)
            assert expected_amount_of_uwsgi_threads == config["uwsgi"]["threads"]
