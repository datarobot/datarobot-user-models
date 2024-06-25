import datetime
import json
import logging
import os
import pprint
import time
import uuid

from mlpiper.common import constants
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.ml_engine.ee_arg import EeArg
from mlpiper.ml_engine.python_engine import PythonEngine


class SageMakerEngine(PythonEngine):
    TYPE = "sagemaker"

    SERVICE_ROLE_PATH = "/service-role/"
    ROLE_RESPONSE_GROUP = "Role"
    ROLE_RESPONSE_NAME_KEY = "RoleName"
    ROLE_RESPONSE_ARN_KEY = "Arn"
    FULL_ACCESS_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"

    AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
    AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"

    def __init__(self, pipeline):
        super(SageMakerEngine, self).__init__(pipeline, None)
        self._iam_role = None
        self._iam_role_name = None

        eng_args_config = self._read_execution_env_params()

        self._tag_key = EeArg(eng_args_config.get("tag-key")).value
        self._tag_value = EeArg(eng_args_config.get("tag-value")).value

        self._setup_env(eng_args_config)

        # Note: the 'boto3' module should be imported only after we setup the environment variables,
        # which setup the shared credentials and region configuration
        global boto3, ClientError

        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise MLPiperException(
                "'sagemaker' python package is missing! "
                "You may install it using: 'pip install sagemaker'!"
            )

        self._setup_logger(eng_args_config)
        self._setup_iam_role(eng_args_config)

    @property
    def iam_role(self):
        return self._iam_role

    def _setup_iam_role(self, eng_args_config):
        self._iam_role = EeArg(eng_args_config.get("iam_role")).value
        if not self._iam_role:
            self._create_role()

    def _create_role(self):
        """
        Creates an IAM role for SageMaker service (
        https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#\
            sagemaker-roles-amazonsagemakerfullaccess-policy)

        Please note that in order to be able to execute this function, proper permissions
        should be set upfront for the given user. To set permissions, do the following:
        1. Enter to AWS console ==> "My Security Credentials" ==> Users ==> <relevant-user>
            ==> "Add inline policy"
        2. set the following write permissions:
            "iam:DetachRolePolicy", "iam:CreateRole", "iam:DeleteRole", "iam:AttachRolePolicy"
        3. Provides a name to the new attached permissions and save
        4. Make sure there are no any other contradicting permission policy that block one
            of the permissions above.
        """
        now = datetime.datetime.now()
        role_name = "SageMaker-ExecutionRole-{y}{month}{d}T{h:02}{minute:02}{s:02}-{unique}".format(
            y=now.year,
            month=now.month,
            d=now.day,
            h=now.hour,
            minute=now.minute,
            s=now.second,
            unique=uuid.uuid4().hex[:10],
        )

        tags = (
            [{"Key": self._tag_key, "Value": self._tag_value}] if self._tag_key else []
        )

        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

        client = boto3.client("iam")

        try:
            self._logger.info("Creating sagemaker role, name: {}".format(role_name))
            response = client.create_role(
                Path=SageMakerEngine.SERVICE_ROLE_PATH,
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Auto generated sagemaker aim role by ParallelM",
                Tags=tags,
            )
            self._logger.debug(pprint.pformat(response))

            self._iam_role_name = response[SageMakerEngine.ROLE_RESPONSE_GROUP][
                SageMakerEngine.ROLE_RESPONSE_NAME_KEY
            ]
            self._iam_role = response[SageMakerEngine.ROLE_RESPONSE_GROUP][
                SageMakerEngine.ROLE_RESPONSE_ARN_KEY
            ]

            self._logger.info(
                "Attaching sagemaker role to full access policy, name: {}, policy: {}".format(
                    role_name, SageMakerEngine.FULL_ACCESS_POLICY
                )
            )
            response = client.attach_role_policy(
                PolicyArn=SageMakerEngine.FULL_ACCESS_POLICY, RoleName=role_name
            )
            self._logger.debug(pprint.pformat(response))

            # Give the AWS IAM service enough time to setup the role properly. Trying to
            # query the IAM role did not help, so using an artificial delay here.
            time.sleep(3.0)

        except ClientError as e:
            self._logger.error(
                "Failed to create a an IAM role for sagemaker service!\n{}".format(e)
            )
            raise e

    def _read_execution_env_params(self):
        ee_config = self._pipeline.get("executionEnvironment", dict()).get("configs")
        if not ee_config:
            raise MLPiperException(
                "Missing execution environment section in pipeline json!"
            )

        eng_config = ee_config.get("engConfig")
        if not eng_config:
            raise MLPiperException(
                "Missing execution environment engine section in pipeline json!"
            )

        if eng_config["type"] != SageMakerEngine.TYPE:
            raise MLPiperException(
                "Unexpected engine type in execution environment! expected: '{}', got: {}".format(
                    SageMakerEngine.TYPE, eng_config["type"]
                )
            )

        return eng_config["arguments"]

    def _setup_env(self, eng_args_config):
        region = EeArg(eng_args_config.get("region")).value

        aws_access_key_id = EeArg(eng_args_config.get("aws_access_key_id")).value
        if not aws_access_key_id:
            raise MLPiperException(
                "Empty 'aws_access_key_id' parameter in execution environment!"
            )

        aws_secret_access_key = EeArg(
            eng_args_config.get("aws_secret_access_key")
        ).value
        if not aws_secret_access_key:
            raise MLPiperException(
                "Missing 'aws_secret_access_key' parameter in execution environment!"
            )

        os.environ[SageMakerEngine.AWS_DEFAULT_REGION] = region
        os.environ[SageMakerEngine.AWS_ACCESS_KEY_ID] = aws_access_key_id
        os.environ[SageMakerEngine.AWS_SECRET_ACCESS_KEY] = aws_secret_access_key

    def _setup_logger(self, eng_args_config):
        modules_logging_level = EeArg(eng_args_config.get("boto3_logging_level")).value
        if modules_logging_level:
            for module, level in modules_logging_level.items():
                logging_level = constants.LOG_LEVELS.get(
                    "info" if not level else level.lower(), logging.INFO
                )
                self._logger.info(
                    "Set logging level, '{}' ==> {}".format(
                        module, logging.getLevelName(logging_level)
                    )
                )
                boto3.set_stream_logger(module, logging_level)

    def cleanup(self):
        # TODO: Cleanup all resources that were created during this session.
        # TODO: Add this option to execution environment.
        # TODO: Cleanup according to a given tag, by listing all resources for that tag. The tags
        # TODO: should be assigned when a resource is created.
        if self._iam_role_name:
            client = boto3.client("iam")
            try:
                self._logger.info(
                    "Cleaning up sagemaker iam role: {}".format(self._iam_role_name)
                )
                client.detach_role_policy(
                    RoleName=self._iam_role_name,
                    PolicyArn=SageMakerEngine.FULL_ACCESS_POLICY,
                )
                response = client.delete_role(RoleName=self._iam_role_name)
                self._logger.debug(pprint.pformat(response))
            except ClientError as e:
                self._logger.error(
                    "Failed to delete the generated SageMaker iam role!\n{}".format(e)
                )
