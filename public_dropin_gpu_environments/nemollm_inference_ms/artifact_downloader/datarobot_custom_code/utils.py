import os
import sys
import tarfile
import zipfile
from urllib.parse import urlparse

import yaml
import psutil
import hashlib


def calculate_sha256(file_path):
    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        # Initialize the hash object with SHA256
        sha256_hash = hashlib.sha256()

        # Read the file in chunks to avoid loading the entire file into memory
        chunk_size = 4096  # You can adjust the chunk size as needed
        while True:
            # Read a chunk of data from the file
            data = f.read(chunk_size)
            if not data:
                break  # End of file

            # Update the hash object with the data read from the file
            sha256_hash.update(data)

    # Get the hexadecimal digest of the hash
    hex_digest = sha256_hash.hexdigest()
    return hex_digest

def get_disk_space(dir_to_check):
    # Get disk usage statistics
    disk_usage = psutil.disk_usage(dir_to_check)
    # Convert bytes to megabytes
    total_mb = disk_usage.total / (1024 * 1024)
    used_mb = disk_usage.used / (1024 * 1024)
    free_mb = disk_usage.free / (1024 * 1024)
    # Return disk space size in MB
    return total_mb, used_mb, free_mb


def bytes_to_mb_str(bytes_value):
    mb_value = bytes_value / (1024 * 1024)  # Convert bytes to megabytes
    return f"{mb_value:.2f} MB"


def calculate_rate(size_bytes, time_seconds):
    if time_seconds == 0:
        return "Infinity MB/sec"
    rate_mb_sec = size_bytes / (time_seconds * 1024 * 1024)  # Convert bytes to MB and divide by time in seconds
    return rate_mb_sec


def calculate_rate_str(size_bytes, time_seconds):
    if time_seconds == 0:
        return "Infinity MB/sec"
    return "{:.2f} MB/sec".format(calculate_rate(size_bytes, time_seconds))


def parse_s3_uri(s3_uri):
    parsed_uri = urlparse(s3_uri)
    if parsed_uri.scheme != 's3':
        raise ValueError("Not an S3 URI")

    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip('/')
    return bucket_name, object_key


def list_zip_contents(zip_file):
    try:
        # Open the zip file for reading
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract all contents to the current working directory
            # zip_ref.extractall()
            # Alternatively, you can extract specific files by passing their names to extract():
            # zip_ref.extract('file_name.txt')
            # zip_ref.extract('directory_name')

            # Get a list of the names of all contents in the zip file
            zip_contents = zip_ref.namelist()
            return zip_contents
    except zipfile.BadZipFile as e:
        print(f"Error: {e}")
        return None


def extract_zip_content(zip_file, file_to_extract=None, dest_dir=None):
    try:
        # Open the zip file for reading
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract all contents to the current working directory
            if file_to_extract:
                print(f"Extracting {file_to_extract}")
                zip_ref.extract(file_to_extract)
            else:
                print(f"Extracting {zip_file} to {dest_dir}")
                zip_ref.extractall(dest_dir)
            # Alternatively, you can extract specific files by passing their names to extract():
            # zip_ref.extract('file_name.txt')
            # zip_ref.extract('directory_name')

            # Get a list of the names of all contents in the zip file
            zip_contents = zip_ref.namelist()
            return zip_contents
    except zipfile.BadZipFile as e:
        print(f"Error: {e}")
        return None


def list_tar_contents(tar_file):
    try:
        # Open the tar file for reading
        with tarfile.open(tar_file, 'r') as tar:
            # List the contents of the tar file
            tar_contents = tar.getnames()
            return tar_contents
    except tarfile.TarError as e:
        print(f"Error: {e}")
        return None


def load_yaml_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error loading YAML from file: {e}")
        return None


def sum_file_sizes(list_of_dicts):
    total_size = 0
    for d in list_of_dicts:
        total_size += d.get("file_size", 0)
    return total_size


def verify_file_checksum(file_path, expected_sha256):
    sha256 = calculate_sha256(file_path)
    if sha256 != expected_sha256:
        print("Error checksum does not match expected checksum", file=sys.stderr)
        return False
    else:
        print("Success checksum matches expected checksum")
        return True


def merge_lists(list1, list2):
    merged_list = []
    index_dict = {item['index']: item for item in list2}
    for item1 in list1:
        index = item1.get("index")
        if index is not None and index in index_dict:
            item2 = index_dict[index]
            merged_item = {**item1, **item2}
            merged_list.append(merged_item)
    return merged_list


def verify_file_integrity(file_info, verify_checksum):
    local_file_size = os.path.getsize(file_info["local_file"])

    if local_file_size == file_info["file_size"]:
        print("All is good file sizes match")
        if verify_checksum:
            if file_info["checksum"] is None:
                print("Checksum not found in file info - skipping")
            else:
                print("Verifying checksum")
                return verify_file_checksum(file_info["local_file"], file_info["checksum"])

        return True
    else:
        print("Error local_file_size != file_size {} {}".format(local_file_size, file_info["file_size"]))
        return False
