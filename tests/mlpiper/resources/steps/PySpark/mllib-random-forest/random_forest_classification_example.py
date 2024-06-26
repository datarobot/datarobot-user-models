#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Random Forest Classification Example.
"""
from __future__ import print_function

from pyspark import SparkContext

# $example on$
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

import argparse

# $example off$


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-file", help="Data file to use as input")
    parser.add_argument("--output-model", help="Path of output model to create")
    parser.add_argument("--num-trees", type=int, help="Number of trees")
    parser.add_argument("--num-classes", type=int, help="Number of classes")

    options = parser.parse_args()

    return options


def main():
    options = parse_args()

    sc = SparkContext(appName="PythonRandomForestClassificationExample")
    # $example on$
    # Load and parse the data file into an RDD of LabeledPoint.
    data = MLUtils.loadLibSVMFile(sc, options.data_file)
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    #  Note: Use larger numTrees in practice.
    #  Setting featureSubsetStrategy="auto" lets the algorithm choose.
    model = RandomForest.trainClassifier(
        trainingData,
        numClasses=2,
        categoricalFeaturesInfo={},
        numTrees=3,
        featureSubsetStrategy="auto",
        impurity="gini",
        maxDepth=4,
        maxBins=32,
    )

    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(
        testData.count()
    )
    print("Test Error = " + str(testErr))
    print("Learned classification forest model:")
    print(model.toDebugString())

    # Save and load model
    model.save(sc, options.output_model)
    RandomForestModel.load(sc, options.output_model)
    # $example off$


if __name__ == "__main__":
    main()
