from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import Mock

import pytest

from datarobot_drum.drum.utils import extract_class_order, capture_R_traceback_if_errors

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


@dataclass
class FakeFitClass:
    negative_class_label: Any = None
    positive_class_label: Any = None
    class_labels: Optional[list] = None


@pytest.mark.parametrize(
    "fit_class, expected",
    [
        (FakeFitClass("a", "B"), ["a", "B"]),
        (FakeFitClass(0, 1), [0, 1]),
        (FakeFitClass(class_labels=["a", "b", "c"]), ["a", "b", "c"]),
        (FakeFitClass(class_labels=[0, 1, 2]), [0, 1, 2]),
        (FakeFitClass(), None),
    ],
)
def test_extract_classorder(fit_class, expected):
    result = extract_class_order(fit_class)
    assert result == expected


def test_R_traceback_captured():
    r_handler = ro.r
    mock_logger = Mock()
    with pytest.raises(RRuntimeError):
        with capture_R_traceback_if_errors(r_handler, mock_logger):
            r_handler('stop("capture this")')

    assert mock_logger.error.call_count == 1
    assert 'R Traceback:\n3: stop("capture this")\n2:' in mock_logger.error.call_args[0][0]
