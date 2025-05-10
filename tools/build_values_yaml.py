import json
import yaml
import argparse
from pathlib import Path

# Define field order with default values
FIELD_ORDER = [
    ("id", ""),
    ("name", ""),
    ("label", ""),
    ("description", ""),
    ("programmingLanguage", ""),
    ("environmentVersionId", ""),
    ("environmentVersionDescription", ""),
    ("imageRepository", ""),  # New field added here
    ("image", ""),
    ("tags", []),
    ("isPublic", False),
    ("useCases", []),
]

# Track fields that should always be quoted
QUOTE_FIELDS = {"name", "description"}


class QuotedString(str):
    pass


def quoted_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


class QuotingDumper(yaml.SafeDumper):
    pass


QuotingDumper.add_representer(QuotedString, quoted_presenter)


def reorder_and_fill_fields(data, field_order):
    """Reorder dictionary fields and fill missing ones with defaults."""
    result = {}
    for key, default in field_order:
        value = data.get(key, default)
        if key in QUOTE_FIELDS:
            value = QuotedString(value)
        result[key] = value
    return result


def parse_value(value, field_name):
    """Parse the input value to the appropriate type based on field type."""
    if field_name == "isPublic":
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"
    elif field_name in ["tags", "useCases"]:
        if isinstance(value, list):
            return value
        return value.split(",") if value else []
    elif field_name in QUOTE_FIELDS:
        return QuotedString(value)
    else:
        return value


def set_values_in_yaml(key_value_pairs, values_file):
    yaml_path = Path(values_file)

    if not yaml_path.exists():
        print(f"Error: The file {yaml_path} does not exist.")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        values_data = yaml.safe_load(f) or {}

    for key, value in key_value_pairs.items():
        if key in dict(FIELD_ORDER):
            values_data[key] = parse_value(value, key)
        else:
            print(f"Warning: '{key}' is not a valid field. Skipped.")

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(values_data, f, sort_keys=False, allow_unicode=True, Dumper=QuotingDumper)


def json_to_yaml(json_path, yaml_path):
    json_path = Path(json_path)
    yaml_path = Path(yaml_path)

    # If the YAML file exists, read it. Otherwise, start with an empty dict
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            existing_data = yaml.safe_load(f) or {}
    else:
        existing_data = {}

    # Read the JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Update existing YAML fields with JSON values, but don't overwrite existing values
    for key, _ in FIELD_ORDER:
        json_value = json_data.get(key)
        if json_value is not None:
            parsed_value = parse_value(json_value, key)
        else:
            # Retain existing value if key is missing from JSON
            parsed_value = existing_data.get(key, parse_value(dict(FIELD_ORDER).get(key), key))

        existing_data[key] = parsed_value

    # Reorder fields and add defaults where necessary
    final_data = reorder_and_fill_fields(existing_data, FIELD_ORDER)

    # Write the updated YAML file
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(final_data, f, sort_keys=False, allow_unicode=True, Dumper=QuotingDumper)

    print(f"YAML written to: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="build_values_yaml", description="Build and modify values.yaml from environment JSON"
    )

    subparsers = parser.add_subparsers(dest="command")

    # from-json subcommand
    from_json_parser = subparsers.add_parser("from-json", help="Convert JSON to values.yaml")
    from_json_parser.add_argument("--json", required=True, help="Path to the input JSON file")
    from_json_parser.add_argument("--yaml", required=True, help="Path to the output YAML file")

    # set-values subcommand
    set_values_parser = subparsers.add_parser(
        "set-values", help="Set specific values in values.yaml"
    )
    set_values_parser.add_argument(
        "key_value_pairs", nargs="+", help="Key-value pairs to set (key=value)"
    )
    set_values_parser.add_argument("--yaml", required=True, help="Path to the YAML file")

    args = parser.parse_args()

    if args.command == "from-json":
        json_to_yaml(args.json, args.yaml)
    elif args.command == "set-values":
        key_value_pairs = dict(pair.split("=", 1) for pair in args.key_value_pairs)
        set_values_in_yaml(key_value_pairs, values_file=args.yaml)


if __name__ == "__main__":
    main()
