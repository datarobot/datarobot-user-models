import os
import boto3
from mlpiper.extra.aws_helper import AwsHelper
from datarobot_drum import RuntimeParameters


# this path is required by NeMo Inference Server
NEMO_MODEL_STORE_DIR = "/model-store"


def load_model(code_dir: str):
    s3_url = RuntimeParameters.get("s3Url")

    s3_credential = RuntimeParameters.get("s3Credential")
    s3_client = S3Client(s3_url, s3_credential)

    keys, dirs = s3_client.list_objects()
    s3_client.download_files(NEMO_MODEL_STORE_DIR, keys, dirs)
    return "succeeded"  # a non-empty response is required to signal that load_model succeeded


class S3Client:
    def __init__(self, s3_url, credential):
        parsed_url = AwsHelper.s3_url_parse(s3_url)
        self.bucket_name = parsed_url[0]
        self.prefix = parsed_url[1]

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=credential["awsAccessKeyId"],
            aws_secret_access_key=credential["awsSecretAccessKey"],
            aws_session_token=credential["awsSessionToken"]
            if "awsSessionToken" in credential
            else None,
        )

    def list_objects(self):
        keys = []
        dirs = []
        next_token = ""
        base_kwargs = {
            "Bucket": self.bucket_name,
            "Prefix": self.prefix,
        }
        while next_token is not None:
            kwargs = base_kwargs.copy()
            if next_token != "":
                kwargs.update({"ContinuationToken": next_token})
            results = self.s3_client.list_objects_v2(**kwargs)
            contents = results.get("Contents")
            for i in contents:
                k = i.get("Key")
                if k[-1] != "/":
                    keys.append(k)
                else:
                    dirs.append(k)
            next_token = results.get("NextContinuationToken")

        return keys, dirs

    def download_files(self, destination_dir, keys, dirs):
        for d in dirs:
            dest_pathname = os.path.join(destination_dir, str(d).replace(self.prefix, ""))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        for k in keys:
            dest_pathname = os.path.join(destination_dir, str(k).replace(self.prefix, ""))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            self.s3_client.download_file(self.bucket_name, k, dest_pathname)
