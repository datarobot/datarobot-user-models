import boto3
import os
import time


try:  # Python3
    from urllib.parse import urlparse
except ImportError:  # Python2
    from urlparse import urlparse


class TransferMonitor(object):
    def __init__(self, total_bytes, logger):
        self._total_bytes = total_bytes
        self._logger = logger
        self._accumulated_bytes = 0
        self._last_update_time = time.time()

    def callback(self, chunk_bytes):
        self._accumulated_bytes += chunk_bytes
        curr_time = time.time()
        if curr_time - self._last_update_time >= 1:
            self._last_update_time = curr_time
            percent = 100.0 * self._accumulated_bytes / self._total_bytes
            self._logger.info("Transfer in progress ... {:.2f}%".format(percent))

    def done(self):
        self._logger.info("Transfer completed ... 100%")


class AwsHelper(object):
    def __init__(self, logger):
        self._logger = logger

    def upload_file(
        self, local_filepath, bucket_name, aws_s3_filepath, skip_upload=False
    ):
        if aws_s3_filepath:
            if aws_s3_filepath.endswith("/"):
                aws_s3_filepath += os.path.basename(local_filepath)
        else:
            aws_s3_filepath = os.path.basename(local_filepath)

        s3_url = "s3://{}/{}".format(bucket_name, aws_s3_filepath)
        self._logger.info(
            "Uploading file to S3: {} ==> {}".format(local_filepath, s3_url)
        )

        if not skip_upload:
            monitor = TransferMonitor(os.path.getsize(local_filepath), self._logger)
            boto3.client("s3").upload_file(
                local_filepath, bucket_name, aws_s3_filepath, Callback=monitor.callback
            )
            monitor.done()
            self._logger.info("File uploaded successfully!")
        else:
            self._logger.info("Skip uploading (test mode)!")

        return s3_url

    def upload_file_obj(
        self, file_obj, bucket_name, aws_s3_filepath, skip_upload=False
    ):
        s3_url = "s3://{}/{}".format(bucket_name, aws_s3_filepath)
        self._logger.info("Uploading file obj to S3 ... {}".format(s3_url))

        if not skip_upload:
            monitor = TransferMonitor(file_obj.getbuffer().nbytes, self._logger)
            boto3.resource("s3").Bucket(bucket_name).Object(
                aws_s3_filepath
            ).upload_fileobj(file_obj, Callback=monitor.callback)
            monitor.done()
            self._logger.info("File obj uploaded successfully!")
        else:
            self._logger.info("Skip uploading (test mode)!")

        return s3_url

    def download_file(self, aws_s3_url, local_filepath):
        self._logger.info(
            "Downloading file from S3: {}, to: {}".format(aws_s3_url, local_filepath)
        )
        bucket_name, model_path = AwsHelper.s3_url_parse(aws_s3_url)

        s3_bucket = boto3.resource("s3").Bucket(bucket_name)
        total_size = s3_bucket.Object(model_path).content_length
        monitor = TransferMonitor(total_size, self._logger)
        s3_bucket.download_file(model_path, local_filepath, Callback=monitor.callback)
        monitor.done()

        self._logger.info("File downloaded successfully!")

    @staticmethod
    def s3_url_parse(aws_s3_url):
        def remove_leading_slash(p):
            return p[1:] if p[0] == "/" else p

        parsed_url = urlparse(aws_s3_url)

        if parsed_url.scheme == "s3":
            bucket_name = parsed_url.netloc
            rltv_path = remove_leading_slash(parsed_url.path)
        else:
            path = remove_leading_slash(parsed_url.path)
            path_parts = path.split("/", 1)
            bucket_name = path_parts[0]
            rltv_path = path_parts[1]

        return bucket_name, rltv_path
