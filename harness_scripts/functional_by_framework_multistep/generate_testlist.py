import json
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def load_tests_list(tests_list_path):
    try:
        logger.info(f"Loading tests list from {tests_list_path}")
        with open(tests_list_path, 'r') as f:
            data = json.load(f)
        return data.get("environments", [])
    except FileNotFoundError:
        logger.error(f"Tests list file not found: {tests_list_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in tests list.")
        sys.exit(1)

def load_env_info(env_info_path):
    try:
        logger.info(f"Reading env_info from {env_info_path}")
        with open(env_info_path, 'r') as f:
            data = json.load(f)
        return {
            "repo": data["imageRepository"],
            "tag": data["environmentVersionId"]
        }
    except FileNotFoundError:
        logger.error(f"env_info.json not found at path: {env_info_path}")
        return None
    except KeyError as e:
        logger.error(f"Missing key in env_info.json: {e}")
        return None
    except json.JSONDecodeError:
        logger.error("Invalid JSON in env_info.json.")
        return None

def build_environments(env_list, root_path):
    output_envs = []

    for env in env_list:
        env_folder = env["env_folder"]
        framework = env["framework"]
        with_local = env.get("with_local", False)

        env_info_path = os.path.join(root_path, env_folder, framework, "env_info.json")
        env_info = load_env_info(env_info_path)

        if not env_info:
            logger.warning(f"Skipping environment due to invalid env_info: {env}")
            continue

        base_record = {
            "env_folder": env_folder,
            "framework": framework,
            "repo": env_info["repo"],
            "tag": env_info["tag"]
        }

        output_envs.append(base_record)

        if with_local:
            local_record = base_record.copy()
            local_record["tag"] = f"{env_info['tag']}.local"
            output_envs.append(local_record)
            logger.info(f"Added local version for environment: {env_folder}/{framework}")

    return output_envs

def main():
    if len(sys.argv) != 3:
        logger.error("Usage: python build_env_output.py <tests-list.json> <root-path>")
        sys.exit(1)

    tests_list_path = sys.argv[1]
    root_path = sys.argv[2]

    env_list = load_tests_list(tests_list_path)
    final_envs = build_environments(env_list, root_path)

    output = {"environments": final_envs}
    print(json.dumps(output, separators=(',', ':')))  # Minified JSON

if __name__ == "__main__":
    main()
