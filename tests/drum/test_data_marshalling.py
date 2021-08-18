from datarobot_drum.drum.utils import _order_by_float, _can_be_converted_to_float, marshal_labels


def test_marshal_labels():
    assert marshal_labels(expected_labels=["True", "False"], actual_labels=[False, True]) == [
        "False",
        "True",
    ]


def test__order_by_float():
    assert _order_by_float(["0", "01"], ["1.0", ".0"]) == ["01", "0"]
    assert _order_by_float(["0", "1"], [1.0, 0.0]) == ["1", "0"]
    assert _order_by_float(["0", "1"], ["1.0", "0.0"]) == ["1", "0"]
    assert _order_by_float(["0.0", "1"], ["1", ".0"]) == ["1", "0.0"]
    assert _order_by_float(["1.0", "2.4", "0.4", "1.4"], [2.4, 1.0, 0.4, 1.4]) == [
        "2.4",
        "1.0",
        "0.4",
        "1.4",
    ]


def test_can_be_converted():
    assert _can_be_converted_to_float(["05.99999999", "0.2", "-.13"])
    assert not _can_be_converted_to_float(["1.0_", "1", "2"])
