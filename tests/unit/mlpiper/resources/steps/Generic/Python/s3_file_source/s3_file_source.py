from __future__ import print_function

import argparse
import os
import boto3
import uuid

from mlpiper.components import ConnectableComponent
from mlpiper.ml_engine.python_engine import PythonEngine


class S3FileSource(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        file_path = self._fetch_file()
        return [file_path]

    def _fetch_file(self):
        self._logger.info(" *** import_model.. params:{}".format(self._params))

        client = boto3.client(
            "s3",
            aws_access_key_id=self._params["aws_access_key_id"],
            aws_secret_access_key=self._params["aws_secret_access_key"],
        )

        file_path = os.path.join(self._params["parent_directory"], "s3_file_" + str(uuid.uuid4()))
        client.download_file(self._params["bucket"], self._params["key"], file_path)

        return file_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws-access-key-id", default=None, help="Access key ID")
    parser.add_argument("--aws-secret-access-key", default=None, help="Secret key")
    parser.add_argument("--region", default=None, help="AWS region name")
    parser.add_argument("--bucket", default=None, help="S3 bucket name")
    parser.add_argument("--key", default=None, help="S3 key name")
    parser.add_argument(
        "--parent-directory", default="/tmp", help="Parent directory where to save file"
    )
    options = parser.parse_args()
    return options


def main():
    options = parse_args()

    python_engine = PythonEngine("s3_file_source")
    s3_source = S3FileSource(python_engine)
    s3_source.configure(vars(options))
    output_list = s3_source.materialize([])

    print("File downloaded to: {}".format(output_list[0]))


if __name__ == "__main__":
    main()
