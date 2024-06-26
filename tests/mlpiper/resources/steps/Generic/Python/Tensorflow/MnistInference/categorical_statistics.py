from inference_statistics import InferenceStatistics
import numpy as ny


class CategoricalStatistics(InferenceStatistics):
    def __init__(
        self, print_interval, stats_type, num_categories, conf_thresh, hot_label=True
    ):
        super(CategoricalStatistics, self).__init__(print_interval)
        self._num_categories = num_categories
        self._hot_label = hot_label
        self._stats_type = stats_type
        self._conf_thresh = conf_thresh / 100.0

        # These are useful for development, but should be replaced by mlops library functions
        self._label_hist = []
        self._infer_hist = []
        for i in range(0, self._num_categories):
            self._label_hist.append(0)
            self._infer_hist.append(0)

    def infer_stats(self, sample, label, inference):

        # for now, we only process 1 inference at a time
        inference = inference[0]
        prediction = ny.argmax(inference)
        confidence = inference[prediction]
        if confidence < self._conf_thresh:
            self.increment_low_conf()

        self._infer_hist[prediction] += 1

        if label is not None:
            if self._hot_label:
                label = ny.argmax(label)
            self._label_hist[label] += 1

            if prediction == label:
                self.increment_correct()

        self.increment_total()
        if self.is_time_to_report():
            self.report_stats()

        return prediction

    def report_stats(self):
        # without mlops just return
        # keep the method in order not to break anything
        return

    def __del__(self):
        super(CategoricalStatistics, self).__del__()
