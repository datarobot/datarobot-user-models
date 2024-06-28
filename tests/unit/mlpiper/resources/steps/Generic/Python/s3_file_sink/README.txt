#!/bin/bash
# How to get S3: Access Key / Secret Access Key
# 1. Open the IAM console, Select your IAM user name
# 2. Click User Actions, and then click Manage Access Keys.
# 3. Click Create Access Key.
# 4. Copy the keys, alternatively download the file `accessKeys.csv`

ACCESS_KEY="<fill-in>"
SECRET_KEY="<fill-in>"
REGION=us-west-1
BUCKET=mlhub
KEY=h2o-demo/predictions.csv

python ./s3_file_sink.py --aws-access-key-id $ACCESS_KEY \
                 --aws-secret-access-key $SECRET_KEY \
                 --region $REGION \
                 --bucket $BUCKET \
                 --key $KEY \
                 --input-file /tmp/predictions.csv

