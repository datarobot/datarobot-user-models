import tensorflow as tf

from categorical_statistics import CategoricalStatistics
from model import Model


class SavedModel(Model):
    def __init__(
        self,
        output_file,
        model_dir,
        signature_def,
        stats_interval,
        stats_type,
        conf_thresh,
    ):
        super(SavedModel, self).__init__(model_dir, signature_def)

        self._output_file = output_file
        # loads the metagraphdef(s) into the provided session
        # restores variables, gets assets, initializes the assets into the main function
        self._sess = tf.Session()

        # For now, we only set the default tf_serving tag_set
        tag_set = "serve"
        tf.saved_model.loader.load(self._sess, [tag_set], self._model_dir)
        graph = tf.get_default_graph()

        self._input_node = graph.get_tensor_by_name(self.get_input_name())
        self._model = graph.get_tensor_by_name(self.get_output_name())
        num_categories = self.get_output_shape()

        self._stats = CategoricalStatistics(
            stats_interval, stats_type, num_categories, conf_thresh
        )

    def infer(self, sample, label):
        inference = self._sess.run(self._model, {self._input_node: [sample]})
        prediction = self._stats.infer_stats(sample, label, inference)
        self._output_file.write("{}\n".format(prediction))

    def __del__(self):
        self._sess.close()
        del self._stats
        super(SavedModel, self).__del__()
