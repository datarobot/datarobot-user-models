from mlpiper.common.cached_property import cached_property
from mlpiper.extra.sagemaker.monitor.job_monitor_base import JobMonitorBase
from mlpiper.extra.sagemaker.monitor.sm_api_constants import SMApiConstants


class JobMonitorTransformer(JobMonitorBase):
    def __init__(self, sagemaker_client, job_name, logger):
        super(self.__class__, self).__init__(sagemaker_client, job_name, logger)

    def _describe_job(self):
        return self._sagemaker_client.describe_transform_job(
            TransformJobName=self._job_name
        )

    def _job_status(self, describe_response):
        return describe_response[SMApiConstants.Transformer.JOB_STATUS]

    def _job_start_time(self, describe_response):
        return describe_response.get(SMApiConstants.Transformer.START_TIME)

    def _job_end_time(self, describe_response):
        return describe_response.get(SMApiConstants.Transformer.END_TIME)

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
        ]

    def _metrics_namespace(self):
        return SMApiConstants.Transformer.NAMESPACE

    def _report_extended_online_metrics(self, describe_response):
        pass

    def _report_extended_final_metrics(self, describe_response):
        pass
