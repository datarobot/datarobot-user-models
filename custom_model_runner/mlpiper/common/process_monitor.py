import os
import psutil

from mlpiper.common.byte_conv import ByteConv
from mlpiper.common.bg_actor import BgActor


class ProcessMonitor(BgActor):
    POLLING_INTERVAL_SEC = 10.0

    MEMORY_INFO_TABLE_TITLE = "Physical Memory [GB]"
    TOTAL_MEMORY_LABEL = "Total"
    AVAILABLE_MEMORY_LABEL = "Available"
    FREE_MEMORY_LABEL = "Free"
    MLAPP_RSS_LABEL = "Pipeline RSS"

    def __init__(self, mlops, ml_engine, pid=None, include_childs=True):
        super(ProcessMonitor, self).__init__(
            mlops, ml_engine, ProcessMonitor.POLLING_INTERVAL_SEC
        )

        self._pid = pid if pid is not None else os.getpid()
        self._include_childs = include_childs
        self._prev_proc_rss_gb = None

    # Overloaded function
    def _do_repetitive_work(self):
        current_process = psutil.Process(os.getpid())
        proc_rss_gb = ByteConv.from_bytes(current_process.memory_info().rss).gbytes
        for child in current_process.children(recursive=True):
            proc_rss_gb += ByteConv.from_bytes(child.memory_info().rss).gbytes

        if self._prev_proc_rss_gb is None or proc_rss_gb != self._prev_proc_rss_gb:
            self._prev_proc_rss_gb = proc_rss_gb

            virtual_mem = psutil.virtual_memory()
            self._logger.debug(virtual_mem)

            total_physical_mem_gb = ByteConv.from_bytes(virtual_mem.total).gbytes
            available_mem_gb = ByteConv.from_bytes(virtual_mem.available).gbytes
            free_mem_gb = ByteConv.from_bytes(virtual_mem.free).gbytes

            self._logger.info(
                "Reporting mem info: {}: {} GB, {}:{} GB, {}:{} GB, {}:{} GB".format(
                    ProcessMonitor.TOTAL_MEMORY_LABEL,
                    total_physical_mem_gb,
                    ProcessMonitor.AVAILABLE_MEMORY_LABEL,
                    available_mem_gb,
                    ProcessMonitor.FREE_MEMORY_LABEL,
                    free_mem_gb,
                    ProcessMonitor.MLAPP_RSS_LABEL,
                    proc_rss_gb,
                )
            )

    def _finalize(self):
        self._do_repetitive_work()
