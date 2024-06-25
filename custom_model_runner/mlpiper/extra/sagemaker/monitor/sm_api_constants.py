class SMApiConstants:

    # Constants from sagemaker_client.describe_training_job, sagemaker.describe_transform_job
    CREATE_TIME = "CreationTime"

    JOB_COMPLETED = "Completed"
    JOB_IN_PROGRESS = "InProgress"
    JOB_FAILED = "Failed"

    FAILURE_REASON = "FailureReason"

    METRIC_CPU_UTILIZATION = "CPUUtilization"
    METRIC_MEMORY_UTILIZATION = "MemoryUtilization"
    METRIC_DISK_UTILIZATION = "DiskUtilization"

    STAT_AVG = "Average"
    STAT_MIN = "Minimum"
    STAT_MAX = "Maximum"

    LIST_METRICS_NAME = "Metrics"
    LIST_METRICS_DIM = "Dimensions"
    LIST_METRICS_DIM_VALUE = "Value"

    HOST_KEY = "Host"
    TIMESTAMP_ASC = "TimestampAscending"

    METRICS_RESULTS = "MetricDataResults"

    class Estimator:
        JOB_STATUS = "TrainingJobStatus"
        NAMESPACE = "/aws/sagemaker/TrainingJobs"
        START_TIME = "TrainingStartTime"
        END_TIME = "TrainingEndTime"

        SECONDARY_TRANSITIONS = "SecondaryStatusTransitions"

        FINAL_METRIC_DATA_LIST = "FinalMetricDataList"
        METRIC_NAME = "MetricName"
        METRIC_VALUE = "Value"
        ALGO_SPEC = "AlgorithmSpecification"
        METRIC_DEFS = "MetricDefinitions"
        METRIC_DEF_NAME = "Name"
        DF_METRIC_NAME = "metric_name"
        DF_METRIC_VALUE = "value"
        TRAIN_PREFIX = "train:"

    class Transformer:
        JOB_STATUS = "TransformJobStatus"
        NAMESPACE = "/aws/sagemaker/TransformJobs"

        START_TIME = "TransformStartTime"
        END_TIME = "TransformEndTime"
