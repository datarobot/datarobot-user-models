#!/usr/bin/env python
#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import os
import posixpath
from pathlib import Path

import click
import requests
from bson import ObjectId


def check_response_code(response):
    if response.status_code // 100 != 2:
        msg = f"Unexpected response code {response.status_code}\n" + response.text
        raise Exception(msg)


def find_base_env_by_name(session, endpoint, base_env_name):
    url = posixpath.join(endpoint, "executionEnvironments/")
    result = session.get(url, params={"searchFor": base_env_name})
    check_response_code(result)
    data = result.json()
    for env in data["data"]:
        if env["name"] == base_env_name:
            return env

    msg = f"Can't find environment with name {base_env_name}"
    raise Exception(msg)


def find_base_env_by_id(session, endpoint, base_env_id):
    url = posixpath.join(endpoint, f"executionEnvironments/{base_env_id}/")
    result = session.get(url)
    check_response_code(result)
    return result.json()


def get_base_env_last_version(session, endpoint, base_env):
    if ObjectId.is_valid(base_env):
        env = find_base_env_by_id(session, endpoint, base_env)
    else:
        env = find_base_env_by_name(session, endpoint, base_env)

    use_cases = env.get("useCases", [])
    if "customApplication" not in use_cases:
        msg = f"Environment {base_env} can't be used for custom application"
        raise Exception(msg)

    if not env.get("latestVersion"):
        msg = f"Can't find last version for environment {base_env}"
        raise Exception(msg)

    return env["latestVersion"]["id"]


def get_or_create_app_image(session, endpoint, name):
    url = posixpath.join(endpoint, "customApplicationImages/")
    result = session.get(url)
    check_response_code(result)
    data = result.json()

    for image in data["data"]:
        if image["name"] == name:
            return image

    result = session.post(url, json={"name": name})
    check_response_code(result)
    return result.json()


def form_files_data(file_folder):
    files_in_folder = [file for file in Path(file_folder).rglob("*") if file.is_file()]
    data = {"file": [], "filePath": []}
    for file in files_in_folder:
        data["file"].append(open(file, "rb"))
        data["filePath"].append(str(file.relative_to(file_folder)))
    return data


def create_image_version(session, endpoint, image_id, env_version, file_folder):
    url = posixpath.join(endpoint, f"customApplicationImages/{image_id}/versions/")

    result = session.get(url)
    check_response_code(result)
    existing_versions = result.json()

    version_label = f'script_generated_{existing_versions["count"]}'
    result = session.post(url, json={"label": version_label})
    check_response_code(result)
    new_version_id = result.json()["id"]

    patch_url = posixpath.join(url, f"{new_version_id}/")
    files_data = form_files_data(file_folder)

    payload = {"baseEnvironmentVersionId": env_version, "filePath": files_data["filePath"]}
    files = [("file", file) for file in files_data["file"]]

    result = session.patch(patch_url, files=files, data=payload)
    check_response_code(result)

    return new_version_id


def create_custom_app(session, endpoint, name, image_id):
    url = posixpath.join(endpoint, "customApplications/")

    payload = {"name": name, "applicationImageId": image_id}

    result = session.post(url, json=payload)
    check_response_code(result)
    return result.json()["id"]


def prepare_token(parameter_token):
    if parameter_token:
        return parameter_token
    token = os.environ.get("DATAROBOT_API_TOKEN")
    if not token:
        raise ValueError(
            "You need to set DR API token through parameters or DATAROBOT_API_TOKEN env variable"
        )
    return token


def prepare_endpoint(parameter_endpoint):
    if parameter_endpoint:
        return parameter_endpoint

    # using app.datarobot.com/api/v2 as default
    dr_host = os.environ.get("DATAROBOT_HOST", "https://app.datarobot.com")
    return os.environ.get("DATAROBOT_ENDPOINT", posixpath.join(dr_host, "api/v2"))


@click.command()
@click.option(
    "-e", "--base-env", required=True, type=str, help="Name or ID for execution environment"
)
@click.option(
    "-p",
    "--path",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to folder with files that should be uploaded",
)
@click.option(
    "-n",
    "--name",
    default="CustomApp",
    type=str,
    help="Name for new custom application. Default: CustomApp",
)
@click.option("-t", "--token", type=str, help="Pubic API access token.")
@click.option("-E", "--endpoint", type=str, help="Data Robot Public API endpoint")
def upload(token, base_env, path, name, endpoint):
    """App that uses local file for create new custom application"""

    token = prepare_token(token)
    endpoint = prepare_endpoint(endpoint)

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    env_version = get_base_env_last_version(session, endpoint, base_env)
    click.echo(f"Getting base environment version: {env_version}")

    image_name = f"{name}Image"
    image = get_or_create_app_image(session, endpoint, image_name)
    click.echo(f'Preparing custom application image: {image["id"]}')

    version_id = create_image_version(session, endpoint, image["id"], env_version, path)
    click.echo(f"Creating new image version: {version_id}")

    app_id = create_custom_app(session, endpoint, name, image["id"])
    click.echo(f"Starting new application version: {app_id}")

    click.echo("Custom application successfully created")


if __name__ == "__main__":
    upload()
