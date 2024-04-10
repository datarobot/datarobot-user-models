import os
import boto3
from urllib.parse import urlparse
from datarobot_drum import RuntimeParameters


s3_client = boto3.client('s3')


def download_dir(bucket, prefix, local, client=s3_client):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, str(d).replace(prefix, ''))
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, str(k).replace(prefix, ''))
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)


def load_model(code_dir: str):
    print("\nDownloading model from S3...\n")
    src = RuntimeParameters.get("s3Url")
    credential = RuntimeParameters.get("s3Credential")

    model_store_path = '/model-store'
    s3_url = urlparse(src, allow_fragments=False)
    bucket = s3_url.netloc
    prefix = s3_url.path[1:]  # strip /

    s3_client = boto3.client(
        's3',
        aws_access_key_id=credential["awsAccessKeyId"],
        aws_secret_access_key=credential["awsSecretAccessKey"],
        aws_session_token=credential["awsSessionToken"] if "awsSessionToken" in credential else None
    )

    download_dir(bucket, prefix, model_store_path, s3_client)
    print("Download complete")

    return "DONE"
