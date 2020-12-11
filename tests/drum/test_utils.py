from dataclasses import dataclass
from typing import Any, Optional

import pytest

from datarobot_drum.drum.utils import extract_class_order


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
