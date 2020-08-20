import os
import psutil

from mlpiper.common.byte_conv import ByteConv
import collections

MemoryInfo = collections.namedtuple("MemoryInfo", "total avail free drum_rss drum_info nginx_rss")


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
        def get_proc_data(p):
            return {
                "pid": p.pid,
                "cmdline": p.cmdline(),
                "mem": ByteConv.from_bytes(p.memory_info().rss).mbytes,
            }

        drum_process = None
        nginx_rss_mb = 0
        drum_rss_mb = 0

        for proc in psutil.process_iter():
            if "drum" in proc.name().lower():
                if "server" in proc.cmdline():
                    drum_process = proc

        if drum_process:
            drum_rss_mb = ByteConv.from_bytes(drum_process.memory_info().rss).mbytes
            drum_info = []
            drum_info.append(get_proc_data(drum_process))
            for child in drum_process.children(recursive=True):
                drum_rss_mb += ByteConv.from_bytes(child.memory_info().rss).mbytes
                drum_info.append(get_proc_data(child))

        for proc in psutil.process_iter():
            if "nginx" in proc.name().lower():
                nginx_rss_mb += ByteConv.from_bytes(proc.memory_info().rss).mbytes

        virtual_mem = psutil.virtual_memory()
        total_physical_mem_mb = ByteConv.from_bytes(virtual_mem.total).mbytes
        available_mem_mb = ByteConv.from_bytes(virtual_mem.available).mbytes
        free_mem_mb = ByteConv.from_bytes(virtual_mem.free).mbytes
        mem_info = MemoryInfo(
            total=total_physical_mem_mb,
            avail=available_mem_mb,
            free=free_mem_mb,
            drum_rss=drum_rss_mb + nginx_rss_mb if drum_process else "drum process not found",
            drum_info=drum_info if drum_process else "drum process not found",
            nginx_rss=nginx_rss_mb,
        )
        return mem_info
