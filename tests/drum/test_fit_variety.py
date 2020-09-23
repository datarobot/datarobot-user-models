import pytest

from .constants import SKLEARN, BINARY


@pytest.mark.parametrize("framework", [SKLEARN])
def test_fit_variety(variety_resources, variety_data_names, framework):
    df = variety_data_names
    df_path = variety_resources.dataset(df)
    problem = variety_resources.problem(df)
    target = variety_resources.target(df)
    if problem == BINARY:
        class_labels = variety_resources.class_labels(df)


