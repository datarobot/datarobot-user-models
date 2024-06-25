from datetime import timedelta

from parallelm.mlops import mlops
from parallelm.mlops.stats.table import Table


class Report(object):
    _last_metric_values = {}

    @staticmethod
    def job_status(job_name, running_time_sec, billing_time_sec, status=""):
        Report._last_metric_values[job_name] = status
        tbl = (
            Table()
            .name("SageMaker Job Status")
            .cols(["Job Name", "Total Running Time", "Time for Billing", "Status"])
        )
        tbl.add_row(
            [
                job_name,
                Report.seconds_fmt(running_time_sec),
                Report.seconds_fmt(billing_time_sec),
                status,
            ]
        )
        mlops.set_stat(tbl)

    @staticmethod
    def job_secondary_transitions(rows):
        tbl = (
            Table()
            .name("SageMaker Job Transitions")
            .cols(["Start Time", "End Time", "Time Span", "Status", "Description"])
        )
        for row in rows:
            tbl.add_row(row)

        mlops.set_stat(tbl)

    @staticmethod
    def job_metric(metric_name, value):
        last_value = Report._last_metric_values.get(metric_name)
        if last_value is None or last_value != value:
            Report._last_metric_values[metric_name] = value
            mlops.set_stat(metric_name, value)

    @staticmethod
    def job_host_metrics(job_name, metrics_data):
        tbl = Table().name("Job Host Metrics").cols(["Metric", "Value"])
        for metric_data in metrics_data:
            tbl.add_row(
                [
                    metric_data["Label"],
                    metric_data["Values"][0] if metric_data["Values"] else 0,
                ]
            )
        mlops.set_stat(tbl)

    @staticmethod
    def seconds_fmt(seconds):
        return str(timedelta(seconds=int(seconds))) if seconds else "~"
