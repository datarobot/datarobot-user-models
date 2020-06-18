import argparse
import os
import json

from bson import ObjectId

"""
This script helps to update drop in environments version IDs.
If IDs are not updated, environment will not be installed.

How to install environments on staging/production.
TODO: add link to instructions here.
"""

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
PUBLIC_ENVS_DIR_NAME = "public_dropin_environments"
ENV_INFO_JSON = "env_info.json"


def main(dir_to_scan):
    """
    Iterate over directories in dir_to_scan, load json from env._info.json,
    adn replace environment version id

    Parameters
    ----------
    dir_to_scan: str
        folder with drop in environments
    """
    for item in os.listdir(dir_to_scan):
        item_abs_path = os.path.abspath(os.path.join(dir_to_scan, item))
        if os.path.isdir(item_abs_path):
            env_info_json = os.path.join(item_abs_path, ENV_INFO_JSON)
            with open(env_info_json) as json_file:
                metadata = json.load(json_file)
                metadata["environmentVersionId"] = str(ObjectId())
            with open(env_info_json, "w") as json_file:
                json.dump(metadata, json_file, indent=2)
                json_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update drop in environment version ids")
    parser.add_argument(
        "-d",
        "--dir",
        default=os.path.join(ROOT_DIR, PUBLIC_ENVS_DIR_NAME),
        help="Path to public drop-in envs",
    )

    args = parser.parse_args()
    envs_dir = args.dir
    main(envs_dir)
