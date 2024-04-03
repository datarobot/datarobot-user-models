import time
import os
import urllib

import boto3
from datarobot_custom_code.storage_file_download_helper import StorageFileDownloadHelper
from datarobot_custom_code.utils import urlparse, verify_file_integrity, calculate_rate
from multiprocessing import Lock


class ProgressPercentage:
    def __init__(self, filename, file_size, update_interval_secs=10, update_interval_mb=100):
        self._filename = filename
        self._file_size = file_size
        self._seen_so_far = 0
        self._lock = Lock()
        self._last_update_time = time.time()
        self._last_update_size_bytes = 0
        self._update_interval_secs = update_interval_secs
        self._update_interval_bytes = update_interval_mb * 1024 * 1024

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            current_time = time.time()
            if (current_time - self._last_update_time >= self._update_interval_secs
                    or self._seen_so_far >= (self._last_update_size_bytes + self._update_interval_bytes)):
                self._print_progress()
                self._last_update_time = current_time
                self._last_update_size_bytes = self._seen_so_far

    def _print_progress(self):
        percentage = (self._seen_so_far / self._file_size) * 100
        seen_so_far_mb = self._seen_so_far / (1024 ** 2)
        file_size_mb = self._file_size / (1024 ** 2)
        print(f"{self._filename}  {percentage:.2f}% ({seen_so_far_mb:.1f}/{file_size_mb:.1f} MB)\n", end='', flush=True)


class S3FileDownloadHelper(StorageFileDownloadHelper):
    def __init__(self):
        self._s3 = boto3.client('s3')

    def __reduce__(self):
        # Exclude transient_data from pickling
        return (self.__class__, ())

    def __setstate__(self, state):
        # Recreate transient_data upon unpickling
        self._s3 = boto3.client('s3')

    def get_file_size(self, file_uri):
        parsed_url = urlparse(file_uri)
        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip('/')
        print("Buket: {}".format(bucket_name))
        print("Object: {}".format(object_key))

        # print("Listing the bucket")
        # response = self._s3.list_objects_v2(Bucket=bucket_name, Prefix="")
        # if 'Contents' in response:
        #     object_keys = [obj['Key'] for obj in response['Contents']]
        #     for key in object_keys:
        #         print(key)

        # Head the object to get its metadata, including content length
        response = self._s3.head_object(Bucket=bucket_name, Key=object_key)

        # Extract and return the content length (file size)
        file_size = response['ContentLength']
        return file_size

    def download_file(self, result_list, file_info, output_dir, lock, buffer_size, verify_checksum):
        print("Bucket: {}, .Object: {}, Output Dir: {}".
              format(file_info["bucket_name"], file_info["object_key"], output_dir))
        result_info = {}
        try:
            print("Downloading file: {}".format(file_info["local_file"]))
            s3 = boto3.client('s3')
            # Initialize variables for bandwidth calculation
            start_time = time.time()

            # Download the file
            with open(file_info["local_file"], 'wb') as f:
                s3.download_fileobj(file_info["bucket_name"],
                                    file_info["object_key"],
                                    f,
                                    Callback=ProgressPercentage(file_info["local_file"], file_info["file_size"]),
                                    Config=boto3.s3.transfer.TransferConfig(max_io_queue=1,
                                                                            io_chunksize=buffer_size))

            # # Calculate elapsed time and bandwidth
            end_time = time.time()
            elapsed_time = end_time - start_time

            result_info["download_ok"] = verify_file_integrity(file_info, verify_checksum)
            local_file_size = os.path.getsize(file_info["local_file"])
            print(f"Elapsed time: {elapsed_time}")
            print(f"File size: {local_file_size}")

            result_info["index"] = file_info["index"]
            result_info["elapsed_time"] = elapsed_time
            result_info["total_time_sec"] = elapsed_time
            result_info["rate_mb_sec"] = calculate_rate(local_file_size, elapsed_time)
            print("Downloaded: {}. Bandwidth: {:.1f}"
                  .format(file_info["object_key"], calculate_rate(local_file_size, elapsed_time)))
            result_list.append(result_info)
            return 0
        except Exception as e:
            print("Error downloading {}: {}".format(file_info["object_key"], e))
            return 0

    def is_uri_directory(self, file_uri):
        parsed_uri = urllib.parse.urlparse(file_uri)
        bucket_name = parsed_uri.netloc
        object_key = parsed_uri.path.lstrip('/')

        s3 = boto3.client('s3')

        # Use head_object to check if the S3 URI points to an object or a directory
        try:
            s3.head_object(Bucket=bucket_name, Key=object_key)
            return False  # File exists, it's not a directory
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return True  # Directory does not exist, it's a directory
            else:
                raise  # Other errors should be raised

    def list_uris_in_directory(self, dir_uri):
        # Parse the S3 directory URI to extract bucket name and prefix (directory path)
        parsed_uri = urlparse(dir_uri)
        bucket_name = parsed_uri.netloc
        prefix = parsed_uri.path.lstrip('/')

        # Create an S3 client
        s3 = boto3.client('s3')

        # List all objects in the specified S3 directory
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        # Extract the URIs of all files from the response
        file_uris = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'] != prefix:
                    file_uris.append({"uri": f"s3://{bucket_name}/{obj['Key']}",
                                      "path": obj['Key'],
                                      "parent_dir": prefix
                                      })

        return file_uris
