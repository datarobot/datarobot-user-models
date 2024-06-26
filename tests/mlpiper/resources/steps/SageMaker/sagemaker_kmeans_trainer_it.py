import io
from time import gmtime, strftime

import boto3
from sagemaker.session import Session

from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.amazon.common import write_numpy_to_dense_tensor

from mlpiper.components.parameter import str2bool
from mlpiper.extra.sagemaker.monitor.job_monitor_estimator import JobMonitorEstimator
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components import ConnectableComponent

from mlpiper.extra.aws_helper import AwsHelper


class SageMakerKMeansTrainerIT(ConnectableComponent):
    MONITOR_INTERVAL_SEC = 10.0

    def __init__(self, engine):
        super(SageMakerKMeansTrainerIT, self).__init__(engine)
        self._bucket_name = None
        self._train_set = None
        self._num_features = None

        self._output_model_filepath = None
        self._data_location = None
        self._data_s3_url = None
        self._output_location = None
        self._skip_s3_dataset_uploading = None
        self._instance_count = None
        self._instance_type = None
        self._volume_size_in_gb = None
        self._hyper_parameter_k = None
        self._mini_batch_size = None
        self._max_runtime_in_seconds = None

        self._model_artifact_s3_url = None
        self._kmeans = None
        self._job_name = None
        self._sagemaker_client = boto3.client("sagemaker")
        self._aws_helper = AwsHelper(self._logger)

    def _materialize(self, parent_data_objs, user_data):

        if not parent_data_objs or len(parent_data_objs) != 3:
            raise MLPiperException(
                "Expecting 3 parent inputs! got: {}, parent_data: {}".format(
                    len(parent_data_objs), parent_data_objs
                )
            )

        self._init_params(parent_data_objs)
        self._convert_and_upload()
        self._do_training()
        self._monitor_job()
        self._download_model()

    def _init_params(self, parent_data_objs):
        self._output_model_filepath = self._params["output_model_filepath"]

        self._train_set, valid_set, test_set = parent_data_objs
        self._print_statistics_info(self._train_set, valid_set, test_set)

        self._num_features = len(self._train_set[0][0])

        self._bucket_name = self._params.get("bucket_name")
        if not self._bucket_name:
            self._bucket_name = Session().default_bucket()

        self._data_location = self._params.get("data_location")
        if not self._data_location:
            self._data_location = "training/kmeans/data"

        self._output_location = self._params.get("output_location")
        if not self._output_location:
            self._output_location = "s3://{}/training/kmeans/output".format(
                self._bucket_name
            )
        else:
            self._output_location = "s3://{}/{}".format(
                self._bucket_name, self._output_location
            )

        self._skip_s3_dataset_uploading = str2bool(
            self._params.get("skip_s3_dataset_uploading")
        )

        self._instance_count = self._params.get("instance_count", 1)
        self._instance_type = self._params.get("instance_type", "ml.c4.xlarge")
        self._volume_size_in_gb = self._params.get("volume_size_in_gb", 50)
        self._hyper_parameter_k = self._params.get("hyper_parameter_k", 10)
        self._epochs = self._params.get("epochs", 1)
        self._mini_batch_size = self._params.get("mini_batch_size", 500)
        self._max_runtime_in_seconds = self._params.get("max_runtime_in_seconds", 86400)

    def _print_statistics_info(self, train_set, valid_set, test_set):
        self._logger.info("Number of features: {}".format(len(train_set[0][0])))
        self._logger.info(
            "Number of samples in training set: {}".format(len(train_set[0]))
        )
        self._logger.info(
            "Number of samples in valid set: {}".format(len(valid_set[0]))
        )
        self._logger.info("Number of samples in test set: {}".format(len(test_set[0])))
        self._logger.info(
            "First image caption in training set: '{}'".format(train_set[1][0])
        )

    def _convert_and_upload(self):
        buf = io.BytesIO()
        if not self._skip_s3_dataset_uploading:
            self._logger.info(
                "Converting the data into the format required by the SageMaker KMeans algorithm ..."
            )
            write_numpy_to_dense_tensor(buf, self._train_set[0], self._train_set[1])
            buf.seek(0)

        self._logger.info(
            "Uploading the converted data to S3, bucket: {}, location: {} ...".format(
                self._bucket_name, self._data_location
            )
        )
        self._data_s3_url = self._aws_helper.upload_file_obj(
            buf, self._bucket_name, self._data_location, self._skip_s3_dataset_uploading
        )

    def _do_training(self):
        self._logger.info("Training data is located in: {}".format(self._data_s3_url))
        self._logger.info(
            "Artifacts will be located in: {}".format(self._output_location)
        )

        self._job_name = "kmeans-batch-training-" + strftime(
            "%Y-%m-%d-%H-%M-%S", gmtime()
        )
        image = get_image_uri(boto3.Session().region_name, "kmeans")

        create_training_params = {
            "AlgorithmSpecification": {
                "TrainingImage": image,
                "TrainingInputMode": "File",
            },
            "RoleArn": self._ml_engine.iam_role,
            "OutputDataConfig": {"S3OutputPath": self._output_location},
            "ResourceConfig": {
                "InstanceCount": self._instance_count,
                "InstanceType": self._instance_type,
                "VolumeSizeInGB": self._volume_size_in_gb,
            },
            "TrainingJobName": self._job_name,
            "HyperParameters": {
                "k": str(self._hyper_parameter_k),
                "epochs": str(self._epochs),
                "feature_dim": str(self._num_features),
                "mini_batch_size": str(self._mini_batch_size),
                "force_dense": "True",
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": self._max_runtime_in_seconds},
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": self._data_s3_url,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "CompressionType": "None",
                    "RecordWrapperType": "None",
                }
            ],
        }

        self._logger.info("Creating training job ... {}".format(self._job_name))
        self._sagemaker_client.create_training_job(**create_training_params)

    def _monitor_job(self):
        job_monitor = JobMonitorEstimator(
            self._sagemaker_client, self._job_name, self._logger
        )
        job_monitor.set_on_complete_callback(self._on_complete_callback)
        job_monitor.monitor()

    def _on_complete_callback(self, describe_response):
        self._model_artifact_s3_url = describe_response["ModelArtifacts"][
            "S3ModelArtifacts"
        ]

    def _download_model(self):
        if self._output_model_filepath and self._model_artifact_s3_url:
            self._logger.info(
                "Downloading model, {} ==> {}".format(
                    self._model_artifact_s3_url, self._output_model_filepath
                )
            )
            AwsHelper(self._logger).download_file(
                self._model_artifact_s3_url, self._output_model_filepath
            )
