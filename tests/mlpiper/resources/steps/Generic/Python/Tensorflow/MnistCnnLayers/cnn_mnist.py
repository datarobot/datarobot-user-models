#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import mnist_input_data


def cnn_model(features):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
    )

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
    )

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dense, units=10)

    return logits


def add_parameters(parser):
    parser.add_argument(
        "--step_size",
        dest="step_size",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        required=False,
        default=100,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        required=False,
        default=50,
        help="Training batch input size",
    )
    parser.add_argument(
        "--model_version",
        dest="model_version",
        type=int,
        required=False,
        default=1,
        help="Model version",
    )
    parser.add_argument(
        "--stats_interval",
        dest="stats_interval",
        type=int,
        required=False,
        default=100,
        help="Print stats after this number of iterations",
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        type=str,
        required=False,
        help="Directory for saving the trained model",
        default="/tmp/tf_model",
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        type=str,
        required=False,
        help="Directory for caching input data",
        default="/tmp/mnist_data",
    )
    parser.add_argument(
        "--tf_log",
        dest="tf_log",
        type=str,
        required=False,
        help="Tensorflow log directory",
        default="/tmp/tf_log",
    )
    parser.add_argument(
        "--text_model_format",
        dest="use_text",
        required=False,
        default=False,
        action="store_true",
        help="Whether SavedModel should be binary or text",
    )


def main(args):
    # handle parameters
    parser = argparse.ArgumentParser()
    add_parameters(parser)
    args = parser.parse_args()
    print("Training for %i iterations" % args.iterations)

    # Load training and eval data
    mnist_data = mnist_input_data.read_data_sets(args.input_dir, one_hot=True)
    serialized_tf_example = tf.placeholder(tf.string, name="tf_example")
    feature_configs = {"x": tf.FixedLenFeature(shape=[784], dtype=tf.float32)}
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example["x"], name="x")  # use tf.identity() to assign name

    y_ = tf.placeholder("float", shape=[None, 10])

    # Create the model
    model = cnn_model(x)
    y = tf.identity(model, name="y")

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=y)

    # Training
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate=args.step_size
    ).minimize(loss)

    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Track cost and accuracy for the training and test sets in TensorBoard
    tb_writer = tf.summary.FileWriter(args.tf_log, graph=tf.get_default_graph())
    train_cost_op = tf.summary.scalar("train_cost", loss)
    train_acc_op = tf.summary.scalar("train_accuracy", accuracy)
    train_stats_op = tf.summary.merge([train_cost_op, train_acc_op])

    test_cost_op = tf.summary.scalar("test_cost", loss)
    test_acc_op = tf.summary.scalar("test_accuracy", accuracy)
    test_stats_op = tf.summary.merge([test_cost_op, test_acc_op])

    # Train the model
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(args.iterations):
        batch = mnist_data.train.next_batch(args.batch_size)
        _, train_summary, train_cost, train_acc = sess.run(
            [train_step, train_stats_op, loss, accuracy],
            feed_dict={x: batch[0], y_: batch[1]},
        )

        if i % args.stats_interval == 0 or i == (args.iterations - 1):
            test_acc, test_cost, test_summary = sess.run(
                [accuracy, loss, test_stats_op],
                feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels},
            )
            print(
                "step=",
                i,
                "accuracy: train=",
                train_acc,
                "test=",
                test_acc,
                "cost: train=",
                train_cost,
                "test=",
                test_cost,
            )
            tb_writer.add_summary(train_summary, i)
            tb_writer.add_summary(test_summary, i)
            tb_writer.flush()

    # Save the model
    export_path = os.path.join(args.save_dir, str(args.model_version))
    print("Exporting trained model to", export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    values, indices = tf.nn.top_k(y, 10)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(10)])
    )
    prediction_classes = table.lookup(tf.to_int64(indices))

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(
        serialized_tf_example
    )
    output_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
    output_scores = tf.saved_model.utils.build_tensor_info(values)

    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs
        },
        outputs={
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: output_classes,
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: output_scores,
        },
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME,
    )

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"inputs": tensor_info_x},
        outputs={"outputs": tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "predict_images": prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature,
        },
        legacy_init_op=legacy_init_op,
    )

    builder.save(as_text=args.use_text)


if __name__ == "__main__":
    tf.app.run()
