"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import glob
import logging
import os
import tempfile
from pathlib import Path

from unittest.mock import Mock

import pytest

from datarobot_drum.drum.drum import create_custom_inference_model_folder, output_in_code_dir
from datarobot_drum.drum.utils.stacktraces import capture_R_traceback_if_errors
from datarobot_drum.drum.utils.drum_utils import DrumUtils

logger = logging.getLogger(__name__)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.embedded import RRuntimeError
except ImportError:
    error_message = (
        "rpy2 package is not installed."
        "Install datarobot-drum using 'pip install datarobot-drum[R]'"
        "Available for Python>=3.6"
    )
    logger.error(error_message)
    exit(1)


def test_R_traceback_captured():
    r_handler = ro.r
    mock_logger = Mock()
    with pytest.raises(RRuntimeError):
        with capture_R_traceback_if_errors(r_handler, mock_logger):
            r_handler('stop("capture this")')

    assert mock_logger.error.call_count == 1
    assert 'R Traceback:\n3: stop("capture this")\n2:' in mock_logger.error.call_args[0][0]


def test_endswith_extension_ignore_case():
    assert DrumUtils.endswith_extension_ignore_case("f.ExT", ".eXt")
    assert DrumUtils.endswith_extension_ignore_case("f.pY", [".Py", ".Java"])
    assert not DrumUtils.endswith_extension_ignore_case("f.py", [".R"])


def test_find_files_by_extension(tmp_path):
    exts = [".ext", ".Rds", ".py"]
    Path(f"{tmp_path}/file.ext").touch()
    Path(f"{tmp_path}/file.RDS").touch()
    Path(f"{tmp_path}/file.PY").touch()
    Path(f"{tmp_path}/file.pY").touch()
    assert 4 == len(DrumUtils.find_files_by_extensions(tmp_path, exts))


def test_filename_exists_and_is_file(tmp_path, caplog):
    Path(f"{tmp_path}/custom.py").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.py")
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    Path(f"{tmp_path}/custom.r").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.r", "custom.R")
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    Path(f"{tmp_path}/custom.R").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.r", "custom.R")
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    Path(f"{tmp_path}/custom.r").touch()
    Path(f"{tmp_path}/custom.R").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.r", "custom.R")
    assert "Found filenames that case-insensitively match each other" in caplog.text
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    caplog.clear()

    Path(f"{tmp_path}/custom.jl").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.jl")
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    Path(f"{tmp_path}/custom.PY").touch()
    assert not DrumUtils.filename_exists_and_is_file(tmp_path, "custom.py")
    assert "Found filenames that case-insensitively match expected filenames" in caplog.text
    assert "Found: ['custom.PY']" in caplog.text
    assert "Expected one of: ['custom.py']" in caplog.text
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    caplog.clear()


def test_output_dir_copy():
    with tempfile.TemporaryDirectory() as tempdir:
        # setup
        file = Path(tempdir, "test.py")
        file.touch()
        Path(tempdir, "__pycache__").mkdir()
        out_dir = Path(tempdir, "out")
        out_dir.mkdir()

        # test
        create_custom_inference_model_folder(tempdir, str(out_dir))
        assert Path(out_dir, "test.py").exists()
        assert not Path(out_dir, "__pycache__").exists()
        assert not Path(out_dir, "out").exists()


def test_output_in_code_dir():
    code_dir = "/test/code/is/here"
    output_other = "/test/not/code"
    output_code_dir = "/test/code/is/here/output"
    assert not output_in_code_dir(code_dir, output_other)
    assert output_in_code_dir(code_dir, output_code_dir)
