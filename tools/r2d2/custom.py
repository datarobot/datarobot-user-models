import pandas as pd
import sys
import time
import argparse
from enum import Enum
import requests


class R2D2Commands(Enum):
    MEMORY = "memory"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"


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
                if i % 100 == 0:
                    print("Adding: {}".format(i))
                one_mb_array = bytearray(R2D2.MEGA)
                # Touching the bytearray every 512 bytes just to make sure we touch memory
                # and memory is allocated.
                for data_idx in range(0, len(one_mb_array), 512):
                    one_mb_array[data_idx] = 55
                self._mem_array.append(one_mb_array)
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

        cmd = data[R2D2.CMD_COL][0]
        arg = data[R2D2.ARG_COL][0]
        print("cmd: [{}] arg: [{}]".format(cmd, arg))
        print(R2D2Commands.MEMORY.value)
        if cmd == R2D2Commands.MEMORY.value:
            self.alloc_memory(arg)
        elif cmd == R2D2Commands.EXCEPTION.value:
            self.raise_exception(arg)
        elif cmd == R2D2Commands.TIMEOUT.value:
            self.consume_time(arg)
        else:
            print("Cmd: [{}] is not supported".format(cmd))
            raise Exception("Bad CMD provided [{}] [{}]".format(cmd, R2D2Commands.MEMORY.value))


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


def main():
    print("Running r2d2 main")
    print(sys.argv)

    parser = argparse.ArgumentParser(description="Send actions to a running r2d2 model")
    parser.add_argument("cmd", help="command to send", choices=[e.value for e in R2D2Commands])
    parser.add_argument("arg", help="argument for the given command")
    parser.add_argument(
        "--server", default="0.0.0.0:8080", help="Server address of r2d2 model running (via drum)"
    )

    options = parser.parse_args()
    url = "http://" + options.server + "/predict/"

    print("Server: {}".format(options.server))
    print("URL:    {}".format(url))
    print("Cmd:    {}".format(options.cmd))
    print("Arg:    {}".format(options.arg))

    data = pd.DataFrame(
        {R2D2.CMD_COL: [options.cmd], R2D2.ARG_COL: [options.arg]},
        columns=[R2D2.CMD_COL, R2D2.ARG_COL],
    )
    print("Sending the following data:")
    print(data)

    csv_data = data.to_csv(index=False)
    response = requests.post(url, files={"X": csv_data})
    print(response)
    print(response.content)


# Small main for R2D2 usage.
if __name__ == "__main__":
    main()
