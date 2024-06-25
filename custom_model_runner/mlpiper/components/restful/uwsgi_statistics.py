import json
import logging
import os
import socket

from mlpiper.components.restful.uwsgi_stats_snapshot import UwsiStatsSnapshot


class UwsgiStatistics(object):
    STATS_JSON_MAX_SIZE_BYTES = 64 * 2048  # max 64 cores, 2k per core

    def __init__(
        self,
        reporting_interval_sec,
        target_path,
        sock_filename,
        logger,
    ):
        self._logger = logger
        self._reporting_interval_sec = reporting_interval_sec
        self._server_address = os.path.join(target_path, sock_filename)
        self._stats_sock = None

        self._curr_stats_snapshot = None
        self._prev_stats_snapshot = None
        self._prev_metrics_snapshot = None

    def report(self):
        raw_stats = self._read_raw_statistics()
        if not raw_stats:
            return

        self._curr_stats_snapshot = UwsiStatsSnapshot(
            raw_stats, self._prev_stats_snapshot
        )

        self._logger.info(self._curr_stats_snapshot)
        self._prev_stats_snapshot = self._curr_stats_snapshot

    def _read_raw_statistics(self):
        sock = self._setup_stats_connection()
        if sock:
            try:
                data = sock.recv(UwsgiStatistics.STATS_JSON_MAX_SIZE_BYTES)
                return json.loads(data.decode("utf-8"))
            except ValueError as e:
                self._logger.error(
                    "Invalid statistics json format! {}, data:\n{}\n".format(
                        e.message, data
                    )
                )
            finally:
                if sock:
                    sock.close()
        return None

    def _setup_stats_connection(self):
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                "Connecting to uWSGI statistics server via unix socket: {}".format(
                    self._server_address
                )
            )

        stats_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            stats_sock.connect(self._server_address)
        except socket.error as ex:
            self._logger.warning(
                "Failed to open connection to uWSI statistics server! {}".format(ex)
            )
            return None
        return stats_sock
