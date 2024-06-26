#!/usr/bin/env python3

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Train and export a simple Softmax Regression TensorFlow model.
The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.
Usage: mnist_saved_model.py [--training_iteration=x] [--model_version=y] \
    [--save_dir export_dir]
"""

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf

import mnist_input_data

tf.app.flags.DEFINE_integer(
    "training_iteration", 1000, "number of training iterations."
)
tf.app.flags.DEFINE_integer("model_version", 1, "version number of the model.")
tf.app.flags.DEFINE_string("work_dir", "./MnistDataset", "Working directory.")
tf.app.flags.DEFINE_string("save_dir", "/tmp/tf_model", "Model save base directory.")
tf.app.flags.DEFINE_string("tf_log", "/tmp/log", "Log directory.")

FLAGS = tf.app.flags.FLAGS


def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith("-"):
        print(
            "Usage: mnist_export.py [--training_iteration=x] "
            '[--model_version=y] [--tf_log="dir"] [--save_dir export_dir]'
        )
        sys.exit(-1)
    if FLAGS.training_iteration <= 0:
        print("Please specify a positive value for training iteration.")
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print("Please specify a positive value for version number.")
        sys.exit(-1)

    # Train model
    print("Training model...")
    mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
    sess = tf.InteractiveSession()
    serialized_tf_example = tf.placeholder(tf.string, name="tf_example")
    feature_configs = {"x": tf.FixedLenFeature(shape=[784], dtype=tf.float32)}
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example["x"], name="x")  # use tf.identity() to assign name
    y_ = tf.placeholder("float", shape=[None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.global_variables_initializer())
    y = tf.nn.softmax(tf.matmul(x, w) + b, name="y")
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    values, indices = tf.nn.top_k(y, 10)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(10)])
    )
    prediction_classes = table.lookup(tf.to_int64(indices))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tb_writer = tf.summary.FileWriter(FLAGS.tf_log, graph=tf.get_default_graph())
    train_cost_op = tf.summary.scalar("train_cost", cross_entropy)
    train_acc_op = tf.summary.scalar("train_accuracy", accuracy)
    train_stats_op = tf.summary.merge([train_cost_op, train_acc_op])

    test_cost_op = tf.summary.scalar("test_cost", cross_entropy)
    test_acc_op = tf.summary.scalar("test_accuracy", accuracy)
    test_stats_op = tf.summary.merge([test_cost_op, test_acc_op])

    for i in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        _, train_summary, train_cost, train_acc = sess.run(
            [train_step, train_stats_op, cross_entropy, accuracy],
            feed_dict={x: batch[0], y_: batch[1]},
        )
        print("Training. step=", i, "accuracy=", train_acc, "cost=", train_cost)
        tb_writer.add_summary(train_summary, i)
        tb_writer.flush()
    print("Done training!")

    test_summary, test_cost, test_acc = sess.run(
        [test_stats_op, cross_entropy, accuracy],
        feed_dict={x: mnist.test.images, y_: mnist.test.labels},
    )
    print("Testing. accuracy=", test_acc, "cost=", test_cost)
    tb_writer.add_summary(test_summary, i)
    tb_writer.flush()

    # Export model
    # WARNING(break-tutorial-inline-code): The following code snippet is
    # in-lined in tutorials, please update tutorial documents accordingly
    # whenever code changes.
    export_path_base = FLAGS.save_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)),
    )
    print("Exporting trained model to", export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

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

    builder.save()

    print("Done exporting!")


if __name__ == "__main__":
    tf.app.run()
