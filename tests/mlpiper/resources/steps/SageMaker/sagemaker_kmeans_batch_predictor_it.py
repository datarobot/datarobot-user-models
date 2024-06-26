import os
from time import gmtime, strftime

import boto3
from sagemaker.session import Session
from sagemaker.amazon.amazon_estimator import get_image_uri

from mlpiper.components.parameter import str2bool
from mlpiper.extra.sagemaker.monitor.job_monitor_transformer import (
    JobMonitorTransformer,
)
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components import ConnectableComponent

from mlpiper.extra.aws_helper import AwsHelper


class SageMakerKMeansBatchPredictorIT(ConnectableComponent):
    def __init__(self, engine):
        super(SageMakerKMeansBatchPredictorIT, self).__init__(engine)
        self._dataset_s3_url = None
        self._bucket_name = None
        self._local_model_filepath = None
        self._model_s3_filepath = None
        self._results_s3_location = None
        self._model_name = None
        self._job_name = None
        self._instance_type = None
        self._instance_count = None

        self._sagemaker_session = Session()
        self._sagemaker_client = boto3.client("sagemaker")
        self._aws_helper = AwsHelper(self._logger)
        self._job_monitor = None

    def _materialize(self, parent_data_objs, user_data):
        if not parent_data_objs:
            raise MLPiperException("Missing expected dataset S3 url from parent input!")

        if not self._init_params(parent_data_objs):
            return

        self._upload_model_to_s3()
        self._create_model()
        self._create_transformation_job()
        self._monitor_job()
        return [self._predictions_s3_url()]

    def _init_params(self, parent_data_objs):
        self._dataset_s3_url = parent_data_objs[0]

        self._local_model_filepath = self._params["local_model_filepath"]
        if not self._local_model_filepath or not os.path.isfile(
            self._local_model_filepath
        ):
            self._logger.info("Input model is empty! Skip prediction!")
            return False

        self._bucket_name = self._params.get("bucket_name")
        if not self._bucket_name:
            self._bucket_name = self._sagemaker_session.default_bucket()

        self._model_s3_filepath = self._params.get("model_s3_filepath")

        self._results_s3_location = self._params.get("results_s3_location")
        if not self._results_s3_location:
            bucket_name, input_rltv_path = AwsHelper.s3_url_parse(self._dataset_s3_url)
            self._results_s3_location = "s3://{}/prediction/results".format(bucket_name)

        self._skip_s3_model_uploading = str2bool(
            self._params.get("skip_s3_model_uploading")
        )

        self._instance_type = self._params.get("instance_type", "ml.m4.xlarge")
        self._instance_count = self._params.get("instance_count", 1)

        return True

    def _upload_model_to_s3(self):
        self._model_s3_filepath = self._aws_helper.upload_file(
            self._local_model_filepath,
            self._bucket_name,
            self._model_s3_filepath,
            self._skip_s3_model_uploading,
        )

    def _create_model(self):
        self._model_name = "Kmeans-model-{}".format(
            strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        )
        self._logger.info(
            "Creating SageMaker KMeans model ... {}".format(self._model_name)
        )

        primary_container = {
            "Image": get_image_uri(self._sagemaker_session.boto_region_name, "kmeans"),
            "ModelDataUrl": self._model_s3_filepath,
        }

        create_model_response = self._sagemaker_client.create_model(
            ModelName=self._model_name,
            ExecutionRoleArn=self._ml_engine.iam_role,
            PrimaryContainer=primary_container,
        )
        model_arn = create_model_response["ModelArn"]
        self._logger.info(
            "Model created successfully! name: {}, arn: {}".format(
                self._model_name, model_arn
            )
        )

    def _create_transformation_job(self):
        self._job_name = "kmeans-batch-prediction-" + strftime(
            "%Y-%m-%d-%H-%M-%S", gmtime()
        )
        self._logger.info(
            "Setup transform job, job-name: {}, input-dataset: {}, output-path-root:{}".format(
                self._job_name, self._dataset_s3_url, self._results_s3_location
            )
        )

        request = {
            "TransformJobName": self._job_name,
            "ModelName": self._model_name,
            "MaxConcurrentTransforms": 4,
            "MaxPayloadInMB": 6,
            "BatchStrategy": "MultiRecord",
            "TransformOutput": {"S3OutputPath": self._results_s3_location},
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": self._dataset_s3_url,
                    }
                },
                "ContentType": "text/csv;label_size=0",
                "SplitType": "Line",
                "CompressionType": "None",
            },
            "TransformResources": {
                "InstanceType": self._instance_type,
                "InstanceCount": self._instance_count,
            },
        }

        self._sagemaker_client.create_transform_job(**request)
        self._logger.info("Created transform job with name: {}".format(self._job_name))

    def _monitor_job(self):
        JobMonitorTransformer(
            self._sagemaker_client, self._job_name, self._logger
        ).monitor()

    def _predictions_s3_url(self):
        _, input_rltv_path = AwsHelper.s3_url_parse(self._dataset_s3_url)
        predictions_s3_url = "{}/{}.out".format(
            self._results_s3_location, input_rltv_path
        )
        return predictions_s3_url
