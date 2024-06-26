from mlpiper.components import ConnectableComponent
from sagemaker.session import Session

from mlpiper.components.parameter import str2bool
from mlpiper.extra.aws_helper import AwsHelper


class AwsS3FileUploaderIT(ConnectableComponent):
    def __init__(self, engine):
        super(AwsS3FileUploaderIT, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):

        local_filepath = self._params["local_filepath"]

        bucket_name = self._params.get("bucket_name")
        if not bucket_name:
            bucket_name = Session().default_bucket()

        remote_filepath = self._params.get("remote_filepath")

        skip_uploading = str2bool(self._params.get("skip_uploading"))
        dataset_s3_url = AwsHelper(self._logger).upload_file(
            local_filepath, bucket_name, remote_filepath, skip_uploading
        )

        return [dataset_s3_url]
