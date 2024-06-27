"""
For internal use only. The WsgiMonitor is designed to monitor the standard output/error of the
'uWSGI'processes, as well as reading the statistics from the uwsgi master process.
"""
import select
import time
import threading
import traceback

from mlpiper.common.base import Base
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.common.buff_to_lines import BufferToLines
from mlpiper.components.restful.constants import (
    UwsgiConstants,
    ComponentConstants,
    SharedConstants,
)
from mlpiper.components.restful.uwsgi_statistics import UwsgiStatistics


class WsgiMonitor(Base):
    SUCCESS_RUN_INDICATION_MSG = "uwsgi entry point run successfully"
    STATS_JSON_MAX_SIZE_BYTES = 64 * 2048  # max 64 cores, 2k per core

    def __init__(self, ml_engine, monitor_info, shared_conf, uwsgi_entry_point_conf):
        super(self.__class__, self).__init__()
        self.set_logger(ml_engine.get_engine_logger(self.logger_name()))

        self._stats_reporting_interval_sec = uwsgi_entry_point_conf[
            ComponentConstants.STATS_REPORTING_INTERVAL_SEC
        ]

        self._logging_udp_socket = uwsgi_entry_point_conf[UwsgiConstants.LOGGING_UDP_SOCKET]

        self._stats = None
        if not shared_conf.get(SharedConstants.STANDALONE):
            self._stats = UwsgiStatistics(
                self._stats_reporting_interval_sec,
                shared_conf["target_path"],
                shared_conf["stats_sock_filename"],
                self._logger,
            )

        self._monitor_info = monitor_info
        self._shared_conf = shared_conf

    def verify_proper_startup(self):
        # Important note: the following code snippet assumes that the application's startup is not
        # demonized and consequently the logs can be read. Therefor, it is important to use
        # the 'daemonize2' option in 'uwsgi.ini' file
        self._monitor_uwsgi_proc(stop_msg=WsgiMonitor.SUCCESS_RUN_INDICATION_MSG)

    def start(self):
        # Set as daemon thread because if main thread is about to exit, we do **not** need to
        # block on this thread for a clean exit.
        th = threading.Thread(name="uWSGI Monitor", target=self._run, daemon=True)
        self._monitor_info[UwsgiConstants.MONITOR_THREAD_KEY] = th
        th.start()

    def _run(self):
        self._logger.info("Starting logging monitoring in the background ...")
        try:
            self._monitor_uwsgi_proc()
        except:  # noqa: E722
            self._monitor_info[UwsgiConstants.MONITOR_ERROR_KEY] = traceback.format_exc()
        finally:
            if self._logging_udp_socket:
                self._logging_udp_socket.close()
            self._logger.info("Exited logging monitoring!")

    def _monitor_uwsgi_proc(self, stop_msg=None):
        try:
            monitor_stats = not stop_msg
            block_size = 2048
            stdout_buff2lines = BufferToLines()

            keep_reading = True
            last_stats_read = time.time()
            while keep_reading:
                read_fs = [self._logging_udp_socket]

                # Sleep the exact time left within a 1 sec interval
                if monitor_stats:
                    sleep_time = self._stats_reporting_interval_sec - (
                        time.time() - last_stats_read
                    )
                    if sleep_time < 0:
                        sleep_time = 0
                else:
                    sleep_time = self._stats_reporting_interval_sec

                readable_fd = select.select(read_fs, [], [], sleep_time)[0]

                if monitor_stats:
                    wakeup_time = time.time()
                    if wakeup_time - last_stats_read > self._stats_reporting_interval_sec:
                        last_stats_read = wakeup_time
                        if self._stats:
                            self._stats.report()

                if readable_fd:
                    buff, _ = readable_fd[0].recvfrom(block_size)
                    stdout_buff2lines.add(buff)
                    for line in stdout_buff2lines.lines():
                        print(line)

                    if stop_msg and stop_msg.encode() in buff:
                        keep_reading = False
        except MLPiperException:
            if self._logging_udp_socket:
                self._logging_udp_socket.close()
            raise
