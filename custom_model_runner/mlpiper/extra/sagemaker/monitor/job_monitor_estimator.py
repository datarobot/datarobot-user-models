from datetime import datetime
import pytz

from sagemaker import TrainingJobAnalytics

from mlpiper.common.cached_property import cached_property
from mlpiper.extra.sagemaker.monitor.job_monitor_base import JobMonitorBase
from mlpiper.extra.sagemaker.monitor.report import Report
from mlpiper.extra.sagemaker.monitor.sm_api_constants import SMApiConstants


class JobMonitorEstimator(JobMonitorBase):
    def __init__(self, sagemaker_client, job_name, logger):
        super(self.__class__, self).__init__(sagemaker_client, job_name, logger)

        self._metric_names = None
        self._analytics = None

    def _describe_job(self):
        return self._sagemaker_client.describe_training_job(
            TrainingJobName=self._job_name
        )

    def _job_status(self, describe_response):
        return describe_response[SMApiConstants.Estimator.JOB_STATUS]

    def _job_start_time(self, describe_response):
        return describe_response.get(SMApiConstants.Estimator.START_TIME)

    def _job_end_time(self, describe_response):
        return describe_response.get(SMApiConstants.Estimator.END_TIME)

    @cached_property
    def _host_metrics_defs(self):
        return [
            JobMonitorBase.MetricMeta(
                "cpuavg_{}",
                SMApiConstants.METRIC_CPU_UTILIZATION,
                SMApiConstants.STAT_AVG,
            ),
            JobMonitorBase.MetricMeta(
                "cpumin_{}",
                SMApiConstants.METRIC_CPU_UTILIZATION,
                SMApiConstants.STAT_MIN,
            ),
            JobMonitorBase.MetricMeta(
                "cpumax_{}",
                SMApiConstants.METRIC_CPU_UTILIZATION,
                SMApiConstants.STAT_MAX,
            ),
            JobMonitorBase.MetricMeta(
                "memavg_{}",
                SMApiConstants.METRIC_MEMORY_UTILIZATION,
                SMApiConstants.STAT_AVG,
            ),
            JobMonitorBase.MetricMeta(
                "memmin_{}",
                SMApiConstants.METRIC_MEMORY_UTILIZATION,
                SMApiConstants.STAT_MIN,
            ),
            JobMonitorBase.MetricMeta(
                "memmax_{}",
                SMApiConstants.METRIC_MEMORY_UTILIZATION,
                SMApiConstants.STAT_MAX,
            ),
            JobMonitorBase.MetricMeta(
                "diskavg_{}",
                SMApiConstants.METRIC_MEMORY_UTILIZATION,
                SMApiConstants.STAT_AVG,
            ),
            JobMonitorBase.MetricMeta(
                "diskmin_{}",
                SMApiConstants.METRIC_MEMORY_UTILIZATION,
                SMApiConstants.STAT_MIN,
            ),
            JobMonitorBase.MetricMeta(
                "diskmax_{}",
                SMApiConstants.METRIC_MEMORY_UTILIZATION,
                SMApiConstants.STAT_MAX,
            ),
        ]

    def _metrics_namespace(self):
        return SMApiConstants.Estimator.NAMESPACE

    def _report_extended_online_metrics(self, describe_response):
        self._report_secondary_transitions(describe_response)

        # No reason to start reading metrics before the job is actually starting
        start_time = self._job_start_time(describe_response)
        if start_time:
            if self._analytics is None:
                self._analytics = TrainingJobAnalytics(
                    training_job_name=self._job_name,
                    metric_names=self._metric_names_for_training_job(),
                    start_time=start_time,
                )

            metrics_df = self._analytics.dataframe(force_refresh=True)
            if not metrics_df.empty:
                for index, row in metrics_df.iterrows():
                    Report.job_metric(
                        row.get(SMApiConstants.Estimator.DF_METRIC_NAME, "Unknown"),
                        row.get(SMApiConstants.Estimator.DF_METRIC_VALUE, 0),
                    )

    def _report_secondary_transitions(self, describe_response):
        secondary_transitions = describe_response[
            SMApiConstants.Estimator.SECONDARY_TRANSITIONS
        ]

        rows = []
        for transition in secondary_transitions:
            start_time = transition["StartTime"]
            end_time = transition.get("EndTime", datetime.now(pytz.UTC))
            status = transition["Status"]
            message = transition["StatusMessage"]

            time_span = (end_time - start_time).total_seconds()

            rows.append(
                [
                    start_time.strftime("%Y-%m-%d, %H:%M:%S"),
                    end_time.strftime("%Y-%m-%d, %H:%M:%S"),
                    Report.seconds_fmt(time_span),
                    status,
                    message,
                ]
            )

        if rows:
            Report.job_secondary_transitions(rows)

    def _metric_names_for_training_job(self):
        if self._metric_names is None:
            training_description = self._sagemaker_client.describe_training_job(
                TrainingJobName=self._job_name
            )

            metric_definitions = training_description[
                SMApiConstants.Estimator.ALGO_SPEC
            ][SMApiConstants.Estimator.METRIC_DEFS]
            self._metric_names = [
                md[SMApiConstants.Estimator.METRIC_DEF_NAME]
                for md in metric_definitions
                if md[SMApiConstants.Estimator.METRIC_DEF_NAME].startswith(
                    SMApiConstants.Estimator.TRAIN_PREFIX
                )
            ]

        return self._metric_names

    def _report_extended_final_metrics(self, describe_response):
        final_metrics = describe_response.get(
            SMApiConstants.Estimator.FINAL_METRIC_DATA_LIST
        )
        if final_metrics:
            for metric in final_metrics:
                Report.job_metric(
                    metric.get(SMApiConstants.Estimator.METRIC_NAME, "Unknown"),
                    metric.get(SMApiConstants.Estimator.METRIC_VALUE, 0),
                )
