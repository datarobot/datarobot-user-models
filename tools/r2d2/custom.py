import pandas as pd
import sys
import time


class R2D2:
    MEGA = 1048576
    MEGA_STR = " " * MEGA

    CMD_COL = "cmd"
    ARG_COL = "arg"

    def __init__(self):
        self._mem_array = []

    def _mem_size_mb(self):
        return len(self._mem_array)

    def _clear_memory(self):
        self._mem_array = []

    def _alloc_additional_memory(self, additional_mb):
        for i in range(0, additional_mb):
            try:
                self._mem_array.append(R2D2.MEGA_STR + str(i))
                if i % 100 == 0:
                    print("Adding: {}".format(i))
            except MemoryError:
                print(
                    "Failed allocating memory, total memory allocated so far {}mb".format(
                        len(self._mem_array)
                    )
                )
                break

            i += 1

    def alloc_memory(self, arg):
        # Always taking the info to run from the first command
        memory_mb = int(arg)
        print("Found memory_mb in data: {}".format(memory_mb))
        if memory_mb <= 0:
            print("Clearing memory")
            self._clear_memory()
        elif memory_mb > 0:
            additional_mem = memory_mb - self._mem_size_mb()
            if additional_mem > 0:
                print("Allocating additional {} mb")
                self._alloc_additional_memory(additional_mem)
            else:
                print(
                    "No additional memory to allocate, current {} requested {}".format(
                        self._mem_size_mb(), memory_mb
                    )
                )

    def raise_exception(self, arg):
        print("About to raise an exception")
        raise Exception("Raising exception")

    def consume_time(self, arg):
        timeout = int(arg)
        print("About to sleep for {} seconds".format(timeout))
        time.sleep(timeout)

    def handle_prediction_request(self, data):
        if R2D2.CMD_COL not in data.columns:
            print("{} col is missing in data.. skipping".format(R2D2.CMD_COL))
            return
        if R2D2.ARG_COL not in data.columns:
            print("{} col is missing in data.. skipping".format(R2D2.ARG_COL))
            return

        cmd = data["cmd"][0]
        arg = data["arg"][0]

        if cmd == "memory":
            self.alloc_memory(arg)
        elif cmd == "exception":
            self.raise_exception(arg)
        elif cmd == "timeout":
            self.consume_time(arg)
        else:
            print("Cmd: {} is not supported".format(cmd))
            raise Exception("Bad CMD provided {}".format(cmd))


# Global instance
r2d2 = R2D2()


def load_model(input_dir):
    return "dummy"


def score(data, model, **kwargs):
    global prediction_value

    r2d2.handle_prediction_request(data)

    prediction_value = 1
    predictions = pd.DataFrame(
        [prediction_value for _ in range(data.shape[0])], columns=["Predictions"]
    )
    return predictions


# Small main for R2D2 usage.
if __name__ == "__main__":
    print("Running r2d2 main")
    print(sys.argv)

    data_mem = pd.DataFrame({"cmd": ["memory"], "arg": [1000]}, columns=["cmd", "arg"])
    data_exception = pd.DataFrame({"cmd": ["exception"], "arg": [5]}, columns=["cmd", "arg"])
    data_timeout = pd.DataFrame({"cmd": ["timeout"], "arg": [10]}, columns=["cmd", "arg"])

    print(data_mem)
    print(data_exception)
    print(data_timeout)

    init()
    score(data_mem, None)
    # This sleep give you some time to check the memory usage using tools like htop/ps
    time.sleep(30)

    score(data_timeout, None)
    score(data_exception, None)
