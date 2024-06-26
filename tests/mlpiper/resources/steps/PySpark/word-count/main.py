from __future__ import print_function

import argparse
import operator
import pyspark


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--words-file", help="Path of the file whose words need be counted"
    )
    options = parser.parse_args()
    return options


def main():
    """Program entry point"""

    options = parse_args()
    if options.words_file is not None:
        # Intialize a spark context
        with pyspark.SparkContext("local", "PySparkWordCount") as sc:
            # Get a RDD containing lines from this script file
            lines = sc.textFile(options.words_file)
            # Split each line into words and assign a frequency of 1 to each word
            words = lines.flatMap(lambda line: line.split(" ")).map(
                lambda word: (word, 1)
            )
            # count the frequency for words
            counts = words.reduceByKey(operator.add)
            # Sort the counts in descending order based on the word frequency
            sorted_counts = counts.sortBy(lambda x: x[1], False)
            # Get an iterator over the counts to print a word and its frequency
            for word, count in sorted_counts.toLocalIterator():
                print(u"{} --> {}".format(word, count))


if __name__ == "__main__":
    main()
