from collections import namedtuple
from datetime import timedelta
from future.utils import with_metaclass
import abc
import boto3
import logging
import pprint
import time

from mlpiper.common.cached_property import cached_property
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.common.os_util import utcnow
from mlpiper.extra.sagemaker.monitor.report import Report
from mlpiper.extra.sagemaker.monitor.sm_api_constants import SMApiConstants


class JobMonitorBase(with_metaclass(abc.ABCMeta, object)):
    MONITOR_INTERVAL_SEC = 10.0
    ONLINE_METRICS_FETCHING_NUM_RETRIES = 1
    FINAL_METRICS_FETCHING_NUM_RETRIES = 36
    SLEEP_TIME_BETWEEN_METRICS_FETCHING_RETRIES_SEC = 5.0

    MetricMeta = namedtuple("MetricMeta", ["id", "metric_name", "stat"])

    def __init__(self, sagemaker_client, job_name, logger):
        self._logger = logger
        self._sagemaker_client = sagemaker_client
        self._job_name = job_name
        self._on_complete_callback = None
        self._host_metrics_fetched_successfully = False
        self._cloudwatch_client = boto3.client("cloudwatch")

    def monitor(self):
        self._logger.info("Monitoring job ... {}".format(self._job_name))
        while True:
            response = self._describe_job()
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(pprint.pformat(response, indent=4))

            status = self._job_status(response)
            running_time_sec = self._total_running_time_sec(response)
            billing_time_sec = self._billing_time_sec(response)
            Report.job_status(
                self._job_name, running_time_sec, billing_time_sec, status
            )

            self._report_online_metrics(response)

            if status == SMApiConstants.JOB_COMPLETED:
                self._report_final_metrics(response)
                self._logger.info("Job '{}' completed!".format(self._job_name))
                if self._on_complete_callback:
                    self._on_complete_callback(response)
                break
            elif status == SMApiConstants.JOB_FAILED:
                msg = "Job '{}' failed! message: {}".format(
                    self._job_name, response[SMApiConstants.FAILURE_REASON]
                )
                self._logger.error(msg)
                raise MLPiperException(msg)
            elif status != SMApiConstants.JOB_IN_PROGRESS:
                self._logger.warning(
                    "Unexpected job status! job-name: {}, status: {}".format(
                        self._job_name, status
                    )
                )

            self._logger.info(
                "Job '{}' is still running ... {} sec".format(
                    self._job_name, running_time_sec
                )
            )
            time.sleep(JobMonitorBase.MONITOR_INTERVAL_SEC)

    def set_on_complete_callback(self, on_complete_callback):
        # The prototype of the callback is 'callback(describe_response)'
        self._on_complete_callback = on_complete_callback
        return self

    def _total_running_time_sec(self, describe_response):
        create_time = self._job_create_time(describe_response)
        if create_time is None:
            return None

        return (
            self._last_running_ref_time(describe_response) - create_time
        ).total_seconds()

    def _billing_time_sec(self, response):
        start_time = self._job_start_time(response)
        return (
            (self._last_running_ref_time(response) - start_time).total_seconds()
            if start_time
            else None
        )

    def _last_running_ref_time(self, describe_response):
        end_time = self._job_end_time(describe_response)
        return end_time if end_time else utcnow()

    def _job_create_time(self, describe_response):
        create_time = describe_response.get(SMApiConstants.CREATE_TIME)
        if create_time:
            if (
                utcnow() - create_time
            ).total_seconds() > JobMonitorBase.MONITOR_INTERVAL_SEC * 2:
                self._logger.warning(
                    "The local machine clock and AWS CloudWatch clock are not synchronized!!!"
                )

        return create_time

    def _job_is_running(self, describe_response):
        return (
            self._job_start_time(describe_response) is not None
            and self._job_end_time(describe_response) is None
        )

    def _report_online_metrics(self, describe_response):
        self._report_job_host_metrics(
            describe_response, JobMonitorBase.ONLINE_METRICS_FETCHING_NUM_RETRIES
        )
        self._report_extended_online_metrics(describe_response)

    def _report_final_metrics(self, describe_response):
        if self._host_metrics_fetched_successfully:
            self._logger.info("Skip final job host metrics fetching")
        else:
            self._logger.info(
                "Trying to fetch final host metrics ... (#attempts: {})".format(
                    JobMonitorBase.FINAL_METRICS_FETCHING_NUM_RETRIES
                )
            )
            self._report_job_host_metrics(
                describe_response, JobMonitorBase.FINAL_METRICS_FETCHING_NUM_RETRIES
            )

        self._report_extended_final_metrics(describe_response)

    def _report_job_host_metrics(self, describe_response, num_retries):
        # No reason to start reading metrics before the job is actually starting
        if self._job_start_time(describe_response):
            job_instance_ids = self._get_job_instance_ids(num_retries)
            if job_instance_ids:
                self._logger.info("Job instance ids: {}".format(job_instance_ids))
                metrics_data = self._fetch_job_host_metrics(
                    job_instance_ids, describe_response
                )
                Report.job_host_metrics(self._job_name, metrics_data)
                self._host_metrics_fetched_successfully = True
            else:
                self._logger.info("Skip transform job host metrics reporting!")

    def _get_job_instance_ids(self, num_retries):
        instance_ids = []
        for retry_index in range(num_retries):
            paginator = self._cloudwatch_client.get_paginator("list_metrics")
            response_iterator = paginator.paginate(
                Dimensions=[{"Name": SMApiConstants.HOST_KEY}],
                MetricName=SMApiConstants.METRIC_CPU_UTILIZATION,
                Namespace=self._metrics_namespace(),
            )
            for response in response_iterator:
                # if self._logger.isEnabledFor(logging.DEBUG):
                #     self._logger.debug(pprint.pformat(response, indent=4))
                for metric in response[SMApiConstants.LIST_METRICS_NAME]:
                    instance_id = metric[SMApiConstants.LIST_METRICS_DIM][0][
                        SMApiConstants.LIST_METRICS_DIM_VALUE
                    ]
                    if instance_id.startswith(self._job_name):
                        instance_ids.append(instance_id)

            if instance_ids or retry_index == num_retries - 1:
                break

            time.sleep(JobMonitorBase.SLEEP_TIME_BETWEEN_METRICS_FETCHING_RETRIES_SEC)
            self._logger.debug(
                "Another attempt to find job instance id! job name: {}, #attempt: {}".format(
                    self._job_name, retry_index
                )
            )

        if not instance_ids:
            self._logger.info(
                "Couldn't find job instance id! job name: {}".format(self._job_name)
            )

        return instance_ids

    def _fetch_job_host_metrics(self, job_instance_ids, describe_response):
        start_time = self._job_start_time(describe_response)
        # Incrementing end time by 1 min since CloudWatch drops seconds before finding the logs.
        # This results in logs being searched in the time range in which the correct log line was
        # not present.
        # Example - Log time - 2018-10-22 08:25:55
        #           Here calculated end time would also be 2018-10-22 08:25:55 (without 1 min
        #           addition). CW will consider end time as 2018-10-22 08:25 and will not be able
        #           to search the correct log.
        end_time = self._last_running_ref_time(describe_response) + timedelta(minutes=1)

        d, r = divmod((end_time - start_time).total_seconds(), 60)
        period = int(d) * 60 + 60  # must be a multiplier of 60
        self._logger.debug(
            "Start time: {}, end time: {}, period: {} sec".format(
                start_time, end_time, period
            )
        )

        metric_data_queries = self._metric_data_queries(job_instance_ids, period)

        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(pprint.pformat(metric_data_queries, indent=4))

        response = self._cloudwatch_client.get_metric_data(
            MetricDataQueries=metric_data_queries,
            StartTime=start_time,
            EndTime=end_time,
            ScanBy=SMApiConstants.TIMESTAMP_ASC,
        )

        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(pprint.pformat(response, indent=4))

        return response[SMApiConstants.METRICS_RESULTS]

    def _metric_data_queries(self, job_instance_ids, period):
        metric_data_queries = []
        for job_instance_id in job_instance_ids:
            inst_id = job_instance_id.split("-")[-1]
            for metric_meta in self._host_metrics_defs:
                query = {
                    "Id": metric_meta.id.format(inst_id),
                    "MetricStat": {
                        "Metric": {
                            "Namespace": self._metrics_namespace(),
                            "MetricName": metric_meta.metric_name,
                            "Dimensions": [
                                {
                                    "Name": SMApiConstants.HOST_KEY,
                                    "Value": job_instance_id,
                                }
                            ],
                        },
                        "Period": period,
                        "Stat": metric_meta.stat,
                    },
                }
                metric_data_queries.append(query)

        return metric_data_queries

    @abc.abstractmethod
    def _describe_job(self):
        pass

    @abc.abstractmethod
    def _job_start_time(self, describe_response):
        pass

    @abc.abstractmethod
    def _job_end_time(self, describe_response):
        pass

    @abc.abstractmethod
    def _job_status(self, describe_response):
        pass

    @abc.abstractmethod
    def _report_extended_online_metrics(self, describe_response):
        pass

    @abc.abstractmethod
    def _report_extended_final_metrics(self, describe_response):
        pass

    @cached_property
    @abc.abstractmethod
    def _host_metrics_defs(self):
        pass

    @abc.abstractmethod
    def _metrics_namespace(self):
        pass
