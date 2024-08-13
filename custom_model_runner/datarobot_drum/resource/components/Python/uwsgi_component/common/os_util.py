from datetime import datetime
import os
import pytz
import subprocess
import tempfile


def tmp_filepath():
    tmp_file = tempfile.NamedTemporaryFile()
    filepath = tmp_file.name
    tmp_file.close()
    return filepath


def remove_file_safely(filepath):
    if os.path.isfile(filepath):
        os.unlink(filepath)


def service_installed(service_name):
    try:
        subprocess.check_output(
            "service {} status".format(service_name),
            shell=True,
            stderr=subprocess.STDOUT,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def utcnow():
    return datetime.now(pytz.UTC)
