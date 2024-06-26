from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components import ConnectableComponent

from mlpiper.extra.aws_helper import AwsHelper


class AwsS3FileDownloaderIT(ConnectableComponent):
    def __init__(self, engine):
        super(AwsS3FileDownloaderIT, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):

        if len(parent_data_objs) != 1:
            raise MLPiperException("Missing a mandatory s3 url for a file as input!")

        s3_url = parent_data_objs[0]
        if s3_url:
            local_filepath = self._params["local_filepath"]
            AwsHelper(self._logger).download_file(s3_url, local_filepath)
        else:
            self._logger.info("Nothing to download from AWS S3!")
