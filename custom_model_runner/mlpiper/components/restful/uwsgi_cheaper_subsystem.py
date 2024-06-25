"""
For internal use only. The uwsgi cheaper sub system provides the ability to dynamically
scale the number of running workers via pluggable algorithms

Reference: https://uwsgi-docs.readthedocs.io/en/latest/Cheaper.html
"""
import math
import multiprocessing


class UwsgiCheaperSubSystem:
    CPU_COUNT = multiprocessing.cpu_count()

    # workers - maximum number of workers that can be spawned
    WORKERS = "workers"

    # cheaper - minimum number of workers to keep at all times
    CHEAPER = "cheaper"

    # cheaper-initial - number of workers to spawn at startup
    CHEAPER_INITIAL = "cheaper-initial"

    # cheaper-step - how many workers should be spawned at a time
    CHEAPER_STEP = "cheaper-step"

    CONF = [
        {
            WORKERS: 2,
            CHEAPER: 1,
            CHEAPER_INITIAL: 2,
            CHEAPER_STEP: 1,
        },  # CPU_COUNT: 1 ~ 3
        {
            WORKERS: 3,
            CHEAPER: 2,
            CHEAPER_INITIAL: 2,
            CHEAPER_STEP: 1,
        },  # CPU_COUNT: 4 ~ 7
        {
            WORKERS: CPU_COUNT - 4,
            CHEAPER: 2,
            CHEAPER_INITIAL: 3,
            CHEAPER_STEP: 2,
        },  # CPU_COUNT: 8 ~ 15
        {
            WORKERS: CPU_COUNT - 4,
            CHEAPER: 5,
            CHEAPER_INITIAL: 5,
            CHEAPER_STEP: 3,
        },  # CPU_COUNT: 16 ~ 23
        {
            WORKERS: CPU_COUNT - 4,
            CHEAPER: 5,
            CHEAPER_INITIAL: 5,
            CHEAPER_STEP: 5,
        },  # CPU_COUNT: > 24
    ]

    @staticmethod
    def get_config():
        entry = int(math.log(UwsgiCheaperSubSystem.CPU_COUNT, 2)) - 1
        if entry < 0:
            entry = 0
        elif entry > (len(UwsgiCheaperSubSystem.CONF) - 1):
            entry = len(UwsgiCheaperSubSystem.CONF) - 1

        return UwsgiCheaperSubSystem.CONF[entry]
