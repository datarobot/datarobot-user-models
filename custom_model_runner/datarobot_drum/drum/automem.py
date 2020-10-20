from datarobot_drum.drum.common import RunMode
from datarobot_drum.drum.perf_testing import CMRunTests


class RunInfo:
    def __init__(self, mem_to_use, result, time):
        self.mem_to_use = mem_to_use
        self.result = result
        self.time = time


class AutoMem:
    def __init__(self, options):
        self._options = options
        self._max_mem = int(self._options.max_mem)
        self._min_mem = int(self._options.min_mem)
        self._step = int(self._options.step)
        self._run_history = []

    def _build_drum_perf_run_cmd(self):
        return "drum perf --docker {} --server"

    def _run_test_with_mem(self, mem_to_use):
        print("Trying with {} mb of memory ... ".format(mem_to_use), end='')

        # TODO: get
        CMRunTests(self._options, RunMode).performance_test()


        if self._options.verbose:
            print("Running cmd: {}".format(cmd))
        result = False
        if mem_to_use > 1000:
            result = True
        self._run_history.append(RunInfo(mem_to_use, result, 100))

        result_str = "ok" if result else "fail"
        print(result_str)
        return result

    def run(self):

        print("\n\n")
        print("Automem mode: detecting optimal memory configuration for model")
        print("Minimum memory: {}".format(self._min_mem))
        print("Maximum memory: {}".format(self._max_mem))
        print("Memory step:    {}".format(self._step))
        print("\n")
        got_good_run = False
        mem_config = -1

        # This part should be replaced by a binary search
        for mem_to_use in range(self._min_mem, self._max_mem, self._step):
            run_ok = self._run_test_with_mem(mem_to_use)
            if run_ok:
                mem_config = mem_to_use
                break

        print("\n\n")
        print("Automem History:")
        print("{:>12} {:>12} {:>20}".format("Memory (MB)", "Result", "Test Time (seconds)"))
        for run in self._run_history:
            print("{:>12} {:>12} {:>20}".format(run.mem_to_use, run.result, run.time))

        # After the search
        print("\n\n")
        if mem_config > 0:
            print("Recommended memory config: {}mb".format(mem_config))
        else:
            print("Failed. No memory configuration found")
