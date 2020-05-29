import os
import psutil

from mlpiper.common.byte_conv import ByteConv
import collections

MemoryInfo = collections.namedtuple("MemoryInfo", "total avail free predictor_rss")


class MemoryMonitor:

    TOTAL_MEMORY_LABEL = "Total"
    AVAILABLE_MEMORY_LABEL = "Available"
    FREE_MEMORY_LABEL = "Free"
    MLAPP_RSS_LABEL = "Pipeline RSS"

    def __init__(self, pid=None, include_childs=True):
        """

        """
        self._pid = pid if pid is not None else os.getpid()
        self._include_childs = include_childs

    def collect_memory_info(self):
        current_process = psutil.Process(os.getpid())
        proc_rss_mb = ByteConv.from_bytes(current_process.memory_info().rss).mbytes
        for child in current_process.children(recursive=True):
            proc_rss_mb += ByteConv.from_bytes(child.memory_info().rss).mbytes

        virtual_mem = psutil.virtual_memory()
        total_physical_mem_mb = ByteConv.from_bytes(virtual_mem.total).mbytes
        available_mem_mb = ByteConv.from_bytes(virtual_mem.available).mbytes
        free_mem_mb = ByteConv.from_bytes(virtual_mem.free).mbytes
        mem_info = MemoryInfo(
            total=total_physical_mem_mb,
            avail=available_mem_mb,
            free=free_mem_mb,
            predictor_rss=proc_rss_mb,
        )
        return mem_info
