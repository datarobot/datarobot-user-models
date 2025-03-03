# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# You probably don't want to modify this file
import os
import sys
import stat


def is_text_file(filepath):
    """Check if a file is a text file by reading the first 512 bytes."""
    try:
        with open(filepath, "rb") as f:
            # Read a small chunk to check for text
            chunk = f.read(512)
            # If it contains null bytes, it's likely binary
            return b"\0" not in chunk and chunk.strip() != b""
    except Exception:
        return False


def is_executable(filepath):
    """Check if a file has executable permissions."""
    st = os.stat(filepath)
    return bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def add_shebang_if_needed(filepath):
    """Add '#!/bin/sh' to the file if it's executable and missing the shebang."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Check if the file is empty or already has the shebang
        if not lines or lines[0].strip() != "#!/bin/sh":
            # Prepend the shebang
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\n")
                f.writelines(lines)
            print(f"Updated: {filepath} - Added '#!/bin/sh'")
        else:
            print(f"Skipped: {filepath} - Already has '#!/bin/sh'")

    except UnicodeDecodeError:
        print(f"Skipped: {filepath} - Not a valid text file")
    except PermissionError:
        print(f"Error: {filepath} - Permission denied")
    except Exception as e:
        print(f"Error: {filepath} - {str(e)}")


def process_directory(directory):
    """Iterate over files in the directory and process text files."""
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Check if it's a text file and executable
            if is_text_file(filepath) and is_executable(filepath):
                add_shebang_if_needed(filepath)


def main():
    """Main function to handle command-line input."""
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    process_directory(directory)


if __name__ == "__main__":
    main()
