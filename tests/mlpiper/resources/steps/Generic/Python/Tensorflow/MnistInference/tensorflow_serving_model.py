#!/usr/bin/env python2.7

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
# This is derived from
# https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py
# Removed the input, stats, and loop to work with general mnist_inference model.

"""A client that talks to tensorflow_model_server loaded with mnist model."""

from __future__ import print_function
import numpy
import threading

from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from categorical_statistics import CategoricalStatistics
from model import Model


class _CallbackHandler(object):
    """Maintains concurrency level and records results."""

    def __init__(
        self,
        concurrency,
        output,
        num_categories,
        stats_interval,
        stats_type,
        conf_thresh,
    ):
        self._stats = CategoricalStatistics(
            stats_interval, stats_type, num_categories, conf_thresh
        )
        self._concurrency = concurrency
        self._active = 0
        self._condition = threading.Condition()
        self._output = output

    def record_stats(self, sample, label, inference):
        prediction = self._stats.infer_stats(sample, label, inference)
        self._output_file.write("{}\n".format(prediction))

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(sample, label, result_counter, output_name):
    """Creates RPC callback function.
    Args:
      sample: the input sample.
      label: The correct label for the predicted example.
      result_counter: Counter for the prediction result.
    Returns:
      The callback function.
    """

    def _callback(result_future):
        """Callback function.
        Records the statistics for the prediction result.
        Args:
          result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            response = numpy.array(
                result_future.result().outputs[output_name].float_val
            )
            result_counter.record_stats(sample, label, [response])
        result_counter.dec_active()

    return _callback


class TfServingModel(Model):
    """Tests PredictionService with concurrent requests.
    Args:
      signature_def: signature_name
      host_port: Host:port address of the PredictionService.
      concurrency: Maximum number of concurrent requests.
    """

    def __init__(
        self,
        output,
        model_dir,
        signature_def,
        stats_interval,
        stats_type,
        conf_thresh,
        host_port,
        concurrency,
    ):
        super(TfServingModel, self).__init__(model_dir, signature_def)

        host, port = host_port.split(":")
        channel = implementations.insecure_channel(host, int(port))
        self._stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        self._model_name = "mnist"
        self._input_name = self.get_input_name()
        self._output_name = self.get_output_name()
        num_categories = 10

        self._result_counter = _CallbackHandler(
            concurrency, output, num_categories, stats_interval, stats_type, conf_thresh
        )

    def infer(self, sample, label):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self._model_name
        request.model_spec.signature_name = self.get_signature_name()
        request.inputs[self._input_name].CopyFrom(
            tf.contrib.util.make_tensor_proto(sample, shape=[1, sample.size])
        )
        self._result_counter.throttle()
        result_future = self._stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(sample, label, self._result_counter, self._output_name)
        )
