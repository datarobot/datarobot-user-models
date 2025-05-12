import json
import yaml
import argparse
import logging
from pathlib import Path
from collections import OrderedDict, namedtuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define field spec as named tuple
FieldSpec = namedtuple("FieldSpec", ["yaml_key", "default", "json_key", "include_in_yaml"])

# Define field order with include_in_yaml = False for internal-use fields
FIELD_ORDER = [
    FieldSpec("name", "", "imageRepository", True),
    FieldSpec("image", "", None, True),
    FieldSpec("tags", [], None, True),
    FieldSpec("environment_name", "", "name", True),
    FieldSpec("environment_id", "", "id", True),
    FieldSpec("environment_version_id", "", "environmentVersionId", True),
    FieldSpec("image_repository", "", "imageRepository", False),
    FieldSpec("environment_version_label", "", "label", True),
    FieldSpec("environment_description", "", "description", True),
    FieldSpec("environment_version_description", "", "environmentVersionDescription", True),
    FieldSpec("programming_language", "", "programmingLanguage", True),
    FieldSpec("is_public", False, "isPublic", True),
    FieldSpec("use_cases", [], "useCases", True),
    FieldSpec("include", True, "include", False),
]

# Fields that should be quoted in YAML
QUOTE_FIELDS = {
    "image_repository",
    "description",
    "environment_description",
    "environment_version_description",
    "environment_name",
    "environment_version_label",
}

class QuotedString(str):
    pass

def quoted_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

class QuotingDumper(yaml.SafeDumper):
    pass

# Register custom representers
QuotingDumper.add_representer(QuotedString, quoted_presenter)
QuotingDumper.add_representer(OrderedDict, yaml.SafeDumper.represent_dict)

def parse_value(value, field_name):
    if field_name in {"is_public", "include"}:
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"
    elif field_name in {"tags", "use_cases"}:
        if isinstance(value, list):
            return value
        return value.split(",") if value else []
    elif field_name in QUOTE_FIELDS:
        return QuotedString(value)
    else:
        return value

def reorder_and_fill_fields(source_data):
    result = OrderedDict()
    env_version_id = parse_value(
        source_data.get("environmentVersionId", ""), "environment_version_id"
    )

    for field in FIELD_ORDER:
        if not field.include_in_yaml:
            continue

        if field.json_key is None:
            if field.yaml_key == "tags":
                value = [env_version_id]
            elif field.yaml_key == "image":
                img_repo = source_data.get("imageRepository", "")
                value = f"datarobot/{img_repo}:{env_version_id}" if img_repo and env_version_id else ""
            else:
                value = field.default
        else:
            raw_value = source_data.get(field.json_key, field.default)
            value = parse_value(raw_value, field.yaml_key)

        if field.yaml_key in QUOTE_FIELDS and not isinstance(value, QuotedString):
            value = QuotedString(value)

        result[field.yaml_key] = value

    return result

def read_json_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def write_yaml_data(yaml_file, data):
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=QuotingDumper)

def convert_json_to_yaml(json_path, yaml_path):
    json_data = read_json_data(json_path)
    final_data = reorder_and_fill_fields(json_data)
    write_yaml_data(yaml_path, final_data)
    logging.info(f"YAML written to: {yaml_path}")

def set_values_in_yaml(key_value_pairs, values_file):
    yaml_path = Path(values_file)

    if not yaml_path.exists():
        logging.error(f"The file {yaml_path} does not exist.")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        values_data = yaml.safe_load(f) or {}

    valid_fields = {f.yaml_key for f in FIELD_ORDER}

    for key, value in key_value_pairs.items():
        if key in valid_fields:
            parsed_value = parse_value(value, key)
            if key == "image" and isinstance(parsed_value, str):
                values_data[key] = QuotedString(parsed_value)
            else:
                values_data[key] = parsed_value
        else:
            logging.warning(f"'{key}' is not a valid field. Skipped.")

    write_yaml_data(yaml_path, values_data)

def make_chart(path, subfolders, chart_file, regenerate=False):
    path = Path(path)
    subfolder_set = set()
    for item in subfolders:
        subfolder_set.update(item.split(","))

    env_files = [
        p for p in path.rglob("env_info.json") if any(sf in str(p.parent) for sf in subfolder_set)
    ]

    logging.info(f"Found {len(env_files)} env_info.json files matching subfolders")

    with open(chart_file, "r", encoding="utf-8") as f:
        chart_data = yaml.safe_load(f) or {}

    existing_envs = chart_data.get("environments", [])
    env_by_id = {env["id"]: env for env in existing_envs if "id" in env}

    if regenerate:
        env_by_id.clear()
        logging.info("Regenerating environments from scratch")

    for json_file in env_files:
        json_data = read_json_data(json_file)

        if "imageRepository" not in json_data:
            logging.warning(f"Skipping {json_file} as it lacks 'imageRepository'")
            continue

        env_yaml = reorder_and_fill_fields(json_data)
        env_id = env_yaml["environment_id"]
        include = env_yaml.get("include", True)

        if not include:
            if env_id in env_by_id:
                logging.info(f"Excluding environment with id {env_id}")
                del env_by_id[env_id]
            continue

        if regenerate or env_id not in env_by_id:
            logging.info(f"Adding new environment with id {env_id}")
            env_by_id[env_id] = env_yaml
        else:
            logging.info(f"Updating existing environment with id {env_id}")
            for k, v in env_yaml.items():
                env_by_id[env_id][k] = v

    chart_data["environments"] = list(env_by_id.values())
    write_yaml_data(chart_file, chart_data)
    logging.info(f"Chart.yaml updated with {len(env_by_id)} environments.")

def main():
    parser = argparse.ArgumentParser(
        prog="build_values_yaml", description="Environment YAML builder"
    )
    subparsers = parser.add_subparsers(dest="command")

    from_json_parser = subparsers.add_parser("from-json", help="Convert JSON to values.yaml")
    from_json_parser.add_argument("--json", required=True, help="Path to the input JSON file")
    from_json_parser.add_argument("--yaml", required=True, help="Path to the output YAML file")

    set_values_parser = subparsers.add_parser(
        "set-values", help="Set specific values in values.yaml"
    )
    set_values_parser.add_argument(
        "key_value_pairs", nargs="+", help="Key-value pairs to set (key=value)"
    )
    set_values_parser.add_argument("--yaml", required=True, help="Path to the YAML file")

    make_chart_parser = subparsers.add_parser(
        "make-chart", help="Update Chart.yaml with environments"
    )
    make_chart_parser.add_argument("--path", required=True, help="Root directory to search")
    make_chart_parser.add_argument(
        "--subfolders", nargs="+", required=True, help="Subfolders to include (comma or space separated)"
    )
    make_chart_parser.add_argument("--chart", required=True, help="Path to Chart.yaml")
    make_chart_parser.add_argument(
        "--regenerate", action="store_true", help="Regenerate environments list"
    )

    args = parser.parse_args()

    if args.command == "from-json":
        convert_json_to_yaml(args.json, args.yaml)
    elif args.command == "set-values":
        key_value_pairs = dict(pair.split("=", 1) for pair in args.key_value_pairs)
        set_values_in_yaml(key_value_pairs, values_file=args.yaml)
    elif args.command == "make-chart":
        make_chart(args.path, args.subfolders, args.chart, args.regenerate)

if __name__ == "__main__":
    main()
