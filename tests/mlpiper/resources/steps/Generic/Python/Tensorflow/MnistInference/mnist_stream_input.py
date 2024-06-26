from __future__ import print_function
import argparse
import numpy as ny
from stream_input import StreamInput
import mnist_input_data


class MnistStreamInput(StreamInput):
    def __init__(self, input_dir, stop_at_record=-1, random=False):
        mnist_data = mnist_input_data.read_data_sets(input_dir, one_hot=True)

        self._samples = mnist_data.test.images
        self._labels = mnist_data.test.labels
        self._input_records = mnist_data.test.num_examples
        super(MnistStreamInput, self).__init__(
            self._input_records, stop_at_record, random
        )

    def get_next_input(self):
        index = self.get_next_input_index()
        if index < 0:
            raise (EOFError("No more records"))

        # samples are numpy float32
        return self._samples[index], self._labels[index]

    def get_total_samples(self):
        return self._input_records


def test_read_some(input_dir, records_to_read, random):
    """Test that the input stream interface works and returns the number of records requested."""

    input = MnistStreamInput(input_dir, records_to_read, random)

    num_categories = 10
    hist = []
    for i in range(0, num_categories):
        hist.append(0)

    total = 0

    while True:
        try:
            sample, label = input.get_next_input()
            hist[ny.argmax(label)] += 1
            total += 1
        except EOFError:
            break

    for i in range(0, num_categories):
        print(i, "=", hist[i])

    print("total = ", total)
    return total == records_to_read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        type=str,
        required=False,
        help="Directory for caching input data",
        default="/tmp/mnist_data",
    )
    args = parser.parse_args()

    passing = test_read_some(args.input_dir, 10000, random=False)
    print("pass is", passing)

    passing = test_read_some(args.input_dir, 40, random=False)
    print("pass is", passing)

    passing = test_read_some(args.input_dir, 20000, random=False)
    print("pass is", passing)

    passing = test_read_some(args.input_dir, 100, random=True)
    print("pass is", passing)


if __name__ == "__main__":
    main()
