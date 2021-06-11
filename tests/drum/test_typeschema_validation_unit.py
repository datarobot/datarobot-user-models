from datarobot_drum.drum.typeschema_validation import NumColumns, Conditions

import pytest


@pytest.mark.parametrize(
    "condition, value, fails",
    [
        (Conditions.IN, [1, 2], False),
        (Conditions.IN, [0, 3], True),
        (Conditions.NOT_IN, [0, 1, 2], False),
        (Conditions.NOT_IN, [-2, 3], True),
        (Conditions.NOT_LESS_THAN, [0], False),
        (Conditions.NOT_LESS_THAN, [-5], True),
        (Conditions.NOT_GREATER_THAN, [5], False),
        (Conditions.NOT_GREATER_THAN, [0], True),
        (Conditions.GREATER_THAN, [0], False),
        (Conditions.GREATER_THAN, [-10], True),
        (Conditions.LESS_THAN, [10], False),
        (Conditions.LESS_THAN, [0], True),
        (Conditions.EQUALS, [1], False),
        (Conditions.EQUALS, [0], True),
        (Conditions.NOT_EQUALS, [0], False),
        (Conditions.NOT_EQUALS, [-9], True),
    ],
)
def test_num_col_values(condition, value, fails):
    if fails:
        with pytest.raises(ValueError):
            NumColumns(condition, value)
    else:
        NumColumns(condition, value)
