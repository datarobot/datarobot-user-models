import logging
import six

from mlpiper.common.base import Base
from mlpiper.common.mlpiper_exception import MLPiperException

uwsgi_loaded = False
try:
    import uwsgi

    uwsgi_loaded = True
except ImportError:
    pass


class MetricType:
    COUNTER = 1
    COUNTER_PER_TIME_WINDOW = 2


class MetricRelation:
    AVG_PER_REQUEST = 1
    DIVIDE_BY = 2
    MULTIPLY_BY = 3
    SUM_OF = 4
    BAR_GRAPH = 5


class Metric(Base):
    NAME_SUFFIX = ".__pm1234__"
    FLOAT_PRECISION = 100000.0  # 5 digits after the period

    _metrics = {}

    def __init__(
        self,
        name,
        title=None,
        hidden=False,
        metric_type=MetricType.COUNTER,
        value_type=int,
        metric_relation=None,
        related_metric=None,
    ):
        super(Metric, self).__init__(logging.getLogger(self.logger_name()))

        self._metric_name = name + Metric.NAME_SUFFIX
        self._title = title
        self._hidden = hidden
        self._metric_type = metric_type
        self._value_type = value_type
        self._metric_relation = metric_relation
        self._metric_already_displayed = False

        if not self._hidden and not self._title:
            raise MLPiperException(
                "A metric can be seen in the UI only if 'title' is provided! name: {}".format(
                    name
                )
            )

        if self.metric_relation == MetricRelation.BAR_GRAPH:
            if not isinstance(related_metric, list):
                raise MLPiperException(
                    "Bar graph metric should be provided with a list of metrics tuples. "
                    "Each tuple should contain the related metric and its bar name! "
                    "name: {}, related_metrics: {}".format(
                        self.name, self.related_metric
                    )
                )

            self._related_metric = []
            for m in related_metric:
                self.add_related_metric(m)
        else:
            self._related_metric = (
                related_metric if isinstance(related_metric, list) else [related_metric]
            )

            if (
                self._related_metric[0]
                and self._related_metric[0].metric_type != metric_type
            ):
                raise MLPiperException(
                    "Error in metrics relation! Given metric cannot relate to other metric of "
                    "different type!"
                    + " mentric: {}, type: {}, related-metric: {}, type: {}".format(
                        name,
                        metric_type,
                        self._related_metric[0].metric_name,
                        self._related_metric[0].metric_type,
                    )
                )

        if name in Metric._metrics:
            raise MLPiperException("Metric has already been defined! name: {}".name)

        self._logger.info("Add new uwsgi metric ... {}".format(self._metric_name))
        Metric._metrics[self._metric_name] = self

    def __str__(self):
        return (
            "name: {}, title: {}, hidden: {}, metric-type: {}, value-type: {}, "
            "metric-relation: {}, related_metric: {}".format(
                self.name,
                self.title,
                self.hidden,
                self.metric_type,
                self.value_type,
                self.metric_relation,
                self.related_metric,
            )
        )

    @staticmethod
    def metrics():
        return Metric._metrics

    @staticmethod
    def metric_by_name(metric_name):
        return Metric._metrics[metric_name]

    @property
    def metric_name(self):
        return self._metric_name

    @property
    def title(self):
        return self._title

    @property
    def hidden(self):
        return self._hidden

    @property
    def value_type(self):
        return self._value_type

    @property
    def metric_type(self):
        return self._metric_type

    @property
    def metric_relation(self):
        return self._metric_relation

    @property
    def related_metric(self):
        return self._related_metric

    @property
    def related_metric_meta(self):
        if isinstance(self.related_metric[0], tuple):
            return [metric_meta for metric_meta, _ in self.related_metric]
        else:
            return self.related_metric

    @property
    def metric_already_displayed(self):
        return self._metric_already_displayed

    @metric_already_displayed.setter
    def metric_already_displayed(self, value):
        self._metric_already_displayed = value

    def add_related_metric(self, bar_graph_metric):
        if self.metric_relation != MetricRelation.BAR_GRAPH:
            raise MLPiperException("Related metric can be added only to bar graph!")

        if not isinstance(bar_graph_metric, tuple) or len(bar_graph_metric) != 2:
            raise MLPiperException(
                "Related metric information should be a tuple of the metric itself and a"
                "bar column label! related_metric: {}".format(bar_graph_metric)
            )

        if not isinstance(bar_graph_metric[0], Metric):
            raise MLPiperException(
                "First element in related bar graph metric should be a Metric! "
                "provided: {}".format(bar_graph_metric[0])
            )

        if not isinstance(bar_graph_metric[1], six.string_types):
            raise MLPiperException(
                "Second element in related bar graph metric should be a string"
                "provided: {}".format(bar_graph_metric[1])
            )

        self._related_metric.append(bar_graph_metric)

    def get(self):
        value = 0
        if uwsgi_loaded:
            value = uwsgi.metric_get(self._metric_name)
            if self._value_type == float:
                value /= Metric.FLOAT_PRECISION
                return value
        return value

    def set(self, value):
        if uwsgi_loaded:
            if self._value_type == float:
                value *= Metric.FLOAT_PRECISION

            uwsgi.metric_set(self._metric_name, value)

    def set_max(self, value):
        """
        only set the metric name if the give value is greater than the one currently stored
        """
        if uwsgi_loaded:
            if self._value_type == float:
                value *= Metric.FLOAT_PRECISION

            uwsgi.metric_set_max(self._metric_name, value)

    def set_min(self, value):
        """
        only set the metric name if the give value is lower than the one currently stored
        """
        if uwsgi_loaded:
            if self._value_type == float:
                value *= Metric.FLOAT_PRECISION

            uwsgi.metric_set_min(self._metric_name, value)

    def increase(self, delta=1):
        """
        increase the metric's value by the given delta
        """
        if uwsgi_loaded:
            if self._value_type == float:
                delta *= Metric.FLOAT_PRECISION

            uwsgi.metric_inc(self._metric_name, int(delta))

    def decrease(self, delta=1):
        """
        decrease the metric's value by the given delta
        """
        if uwsgi_loaded:
            if self._value_type == float:
                delta *= Metric.FLOAT_PRECISION

            uwsgi.metric_dec(self._metric_name, int(delta))

    def multiply(self, delta):
        """
        multiply the metric's value by the given delta
        """
        if uwsgi_loaded:
            uwsgi.metric_mul(self._metric_name, delta)

    def divide(self, delta):
        """
        divide the metric's value by the given delta
        """
        if uwsgi_loaded:
            uwsgi.metric_div(self._metric_name, delta)
