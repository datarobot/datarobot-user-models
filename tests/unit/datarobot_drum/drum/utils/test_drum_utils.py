#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import glob
import logging
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from datarobot_drum.drum.utils.drum_utils import DrumUtils
from datarobot_drum.drum.utils.stacktraces import capture_R_traceback_if_errors

logger = logging.getLogger(__name__)

try:
    import rpy2.robjects as ro
    from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects import pandas2ri

    r_supported = True
except ImportError:
    error_message = (
        "rpy2 package is not installed."
        "Install datarobot-drum using 'pip install datarobot-drum[R]'"
        "Available for Python>=3.6"
    )
    logger.error(error_message)
    r_supported = False


@pytest.mark.skipif(not r_supported, reason="requires R framework to be installed")
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
    exts = [".ext", ".Rds", ".py", ".pbtxt"]
    Path(f"{tmp_path}/file.ext").touch()
    Path(f"{tmp_path}/file.RDS").touch()
    Path(f"{tmp_path}/file.PY").touch()
    Path(f"{tmp_path}/file.pY").touch()

    # Triton model artifacts are expected to be located in the subdirectories
    Path(f"{tmp_path}/vllm/").mkdir()
    Path(f"{tmp_path}/vllm/config.PBTXT").touch()
    Path(f"{tmp_path}/model_repository/vllm/").mkdir(parents=True)
    Path(f"{tmp_path}/model_repository/vllm/config.pbtxt").touch()
    Path(f"{tmp_path}/config.PbTxT").touch()
    assert 7 == len(DrumUtils.find_files_by_extensions(tmp_path, exts))


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
