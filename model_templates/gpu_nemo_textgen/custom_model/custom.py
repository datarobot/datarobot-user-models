import boto3
from pathlib import Path
from mlpiper.extra.aws_helper import AwsHelper
from datarobot_drum import RuntimeParameters


# this path is required by NeMo Inference Server
NEMO_MODEL_STORE_DIR = "/model-store"


def load_model(code_dir: str):
    print("Downloading model from S3...")

    s3_url = RuntimeParameters.get("s3Url")
    s3_credential = RuntimeParameters.get("s3Credential")
    s3_client = S3Client(s3_url, s3_credential)

    keys, prefixes = s3_client.list_objects()
    s3_client.download_files(NEMO_MODEL_STORE_DIR, keys, prefixes)

    print("Download complete")
    return "DONE"  # a non empty response is required


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
        prefixes = []
        next_token = ""
        while next_token is not None:
            list_object_params = {"Bucket": self.bucket_name, "Prefix": self.prefix}
            if next_token != "":
                list_object_params["ContinuationToken"] = next_token

            response = self.s3_client.list_objects_v2(**list_object_params)
            contents = response.get("Contents")

            for result in contents:
                key = result.get("Key")
                if key[-1] == "/":
                    prefixes.append(key)
                else:
                    keys.append(key)

            next_token = response.get("NextContinuationToken")

        return keys, prefixes

    def download_files(self, destination_dir, keys, prefixes):
        destination_dir = Path(destination_dir)

        for prefix in prefixes:
            dir_path = Path.joinpath(destination_dir, prefix)
            dir_path.mkdir(parents=True, exist_ok=True)

        for key in keys:
            file_path = Path.joinpath(destination_dir, key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, key, str(file_path))
