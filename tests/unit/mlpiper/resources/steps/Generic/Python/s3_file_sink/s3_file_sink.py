from __future__ import print_function

import argparse
import boto3

from mlpiper.components import ConnectableComponent
from mlpiper.ml_engine.python_engine import PythonEngine


class S3FileSink(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        file_path = str(parent_data_objs[0])
        self._save_file(file_path)
        return []

    def _save_file(self, file_path):
        self._logger.info(" *** save file .. params:{}".format(self._params))

        client = boto3.client(
            "s3",
            aws_access_key_id=self._params["aws_access_key_id"],
            aws_secret_access_key=self._params["aws_secret_access_key"],
        )
        data = open(file_path, "rb")
        client.put_object(Bucket=self._params["bucket"], Key=self._params["key"], Body=data)

        return file_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-file", default=None, help="File to save in S3")

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

    params = vars(options)
    input_file = params["input_file"]
    params.pop("input_file")

    python_engine = PythonEngine("s3_file_source")
    s3_source = S3FileSink(python_engine)
    s3_source.configure(vars(options))

    parent_objects = [input_file]
    s3_source.materialize(parent_objects)


if __name__ == "__main__":
    main()
