from __future__ import print_function

import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--arg1", help="Test argument 1")
    options = parser.parse_args()
    return options


def main():

    options = parse_args()
    print("- inside hello-world.main.py Running main.py")
    print("arg1: {}".format(options.arg1))


if __name__ == "__main__":
    main()
