from mlpiper.common.topological_sort import TopologicalSort
from mlpiper.components.restful.metric import Metric, MetricType, MetricRelation


class UwsiStatsSnapshot(object):
    def __init__(self, raw_stats, prev_stats_snapshot):
        self._raw_stats = raw_stats
        self._total_requests = None
        self._total_requests_diff = None
        self._sorted_worker_stats = None
        self._avg_rt = None
        self._uwsgi_pm_metrics = None
        self._uwsgi_pm_metrics_accumulation = {}
        self._uwsgi_pm_metrics_per_window = None
        self._metrics_execution_order = []
        self._worker_ids = None
        self._extract_relevant_stats(prev_stats_snapshot)

    def _extract_relevant_stats(self, prev_stats_snapshot):
        self._extract_workers_stats(prev_stats_snapshot)
        self._extract_metrics_stats(prev_stats_snapshot)

    def _extract_workers_stats(self, prev_stats_snapshot):
        self._avg_rt = sorted(
            [
                ("wid-{:02d}".format(w["id"]), [w["avg_rt"]])
                for w in self._raw_stats["workers"]
                if w["id"] != 0
            ]
        )

        worker_requests = {
            "wid-{:02d}".format(w["id"]): (w["requests"], w["status"].encode("utf8"))
            for w in self._raw_stats["workers"]
            if w["id"] != 0
        }
        self._worker_ids = sorted(
            [w["id"] for w in self._raw_stats["workers"] if w["id"] != 0]
        )

        self._sorted_worker_stats = sorted(
            [(k, v[0], v[1].decode()) for k, v in worker_requests.items()]
        )

        self._total_requests = sum(v for _, v, _ in self._sorted_worker_stats)

        self._total_requests_diff = (
            self._total_requests - prev_stats_snapshot.total_requests
            if prev_stats_snapshot is not None
            else self._total_requests
        )

    def _extract_metrics_stats(self, prev_stats_snapshot):
        raw_metrics = self._raw_stats.get("metrics", None)
        if not raw_metrics:
            return

        self._uwsgi_pm_metrics = self._extract_relevant_raw_metrics(raw_metrics)
        if not self._uwsgi_pm_metrics:
            return

        if not self._metrics_execution_order:
            # arrange the metrics in a topological order because it could be a DAG, where one
            # metric is dependant on a metric that depends on others. Using the topological
            # mechanism we can also allow more then one reference in a single metric definition.
            self._metrics_execution_order = TopologicalSort(
                Metric.metrics(), "metric_name", "related_metric_meta"
            ).sort()

        self._uwsgi_pm_metrics_per_window = {}

        for metric_meta in self._metrics_execution_order:
            metric_name = metric_meta.metric_name
            metric_value = self.uwsgi_pm_metrics[metric_name]

            if metric_meta.metric_type == MetricType.COUNTER_PER_TIME_WINDOW:
                if prev_stats_snapshot:
                    metric_value -= prev_stats_snapshot.uwsgi_pm_metric_by_name(
                        metric_name
                    )

                self._calculate_metric_value(
                    metric_value,
                    metric_meta,
                    self.total_requests_diff,
                    self.uwsgi_pm_metrics_per_window,
                )
            else:
                self._calculate_metric_value(
                    metric_value,
                    metric_meta,
                    self.total_requests,
                    self._uwsgi_pm_metrics,
                )
                self._uwsgi_pm_metrics_accumulation[
                    metric_name
                ] = self._uwsgi_pm_metrics[metric_name]

    def _extract_relevant_raw_metrics(self, raw_metrics):
        uwsgi_pm_metrics = {}
        # Set values according their types
        for name, body in raw_metrics.items():
            if Metric.NAME_SUFFIX in name:
                value = body["value"]
                if Metric.metric_by_name(name).value_type == float:
                    value /= Metric.FLOAT_PRECISION
                uwsgi_pm_metrics[name] = value

        return uwsgi_pm_metrics

    def _calculate_metric_value(
        self, metric_value, metric_meta, total_requests, related_metrics
    ):
        metric_name = metric_meta.metric_name

        if metric_meta.metric_relation == MetricRelation.BAR_GRAPH:
            # A graph bar is only a place holder for other metrics. It does not have a value by
            # itself.
            pass

        elif metric_meta.metric_relation == MetricRelation.AVG_PER_REQUEST:
            if total_requests:
                metric_value /= total_requests

        elif metric_meta.related_metric and metric_meta.related_metric[0]:
            related_metric_name = metric_meta.related_metric[0].metric_name
            if metric_meta.metric_relation == MetricRelation.DIVIDE_BY:
                if metric_value and related_metrics[related_metric_name]:
                    metric_value /= related_metrics[related_metric_name]
            elif metric_meta.metric_relation == MetricRelation.MULTIPLY_BY:
                metric_value *= related_metrics[related_metric_name]
            elif metric_meta.metric_relation == MetricRelation.SUM_OF:
                metric_value += related_metrics[related_metric_name]

        related_metrics[metric_name] = metric_value

    def __str__(self):
        return (
            "Total_requests: {}, diff-requests: {}, requests + status: {}, "
            "avg response time: {}, metrics: {}".format(
                self.total_requests,
                self.total_requests_diff,
                self.sorted_worker_stats,
                self.avg_workers_response_time,
                self.uwsgi_pm_metrics,
            )
        )

    @property
    def total_requests(self):
        return self._total_requests

    @property
    def total_requests_diff(self):
        return self._total_requests_diff

    @property
    def sorted_worker_stats(self):
        return self._sorted_worker_stats

    @property
    def avg_workers_response_time(self):
        return self._avg_rt

    @property
    def worker_ids(self):
        return self._worker_ids

    @property
    def uwsgi_pm_metrics(self):
        return self._uwsgi_pm_metrics

    def uwsgi_pm_metric_by_name(self, name):
        return self._uwsgi_pm_metrics[name]

    @property
    def uwsgi_pm_metrics_accumulation(self):
        return self._uwsgi_pm_metrics_accumulation

    @property
    def uwsgi_pm_metrics_per_window(self):
        return self._uwsgi_pm_metrics_per_window

    def should_report_requests_per_window_time(self, stats_snapshot):
        return stats_snapshot is None or (
            self.total_requests_diff != stats_snapshot.total_requests_diff
        )

    def should_report_average_response_time(self, stats_snapshot):
        return stats_snapshot is None or (
            self.avg_workers_response_time != stats_snapshot.avg_workers_response_time
        )

    def should_report_worker_status(self, stats_snapshot):
        return stats_snapshot is None or (
            self.sorted_worker_stats != stats_snapshot.sorted_worker_stats
        )

    def should_report_metrics_accumulation(self, stats_snapshot):
        return stats_snapshot is None or (
            self.uwsgi_pm_metrics_accumulation
            != stats_snapshot.uwsgi_pm_metrics_accumulation
        )

    def should_report_metrics_per_time_window(self, stats_snapshot):
        if self.uwsgi_pm_metrics_per_window is None:
            return False

        return stats_snapshot is None or (
            self.uwsgi_pm_metrics_per_window
            != stats_snapshot.uwsgi_pm_metrics_per_window
        )
