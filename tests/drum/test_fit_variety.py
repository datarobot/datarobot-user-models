import pytest

from datarobot_drum.drum.common import ArgumentsOptions
from .constants import SKLEARN, BINARY, RDS, R_FIT, PYTHON, ANOMALY, PYTORCH, XGB, KERAS
from .utils import _create_custom_model_dir, _cmd_add_class_labels, _exec_shell_cmd


@pytest.mark.skip(reason="currently fails on DR_Demo_Listing_Interest and DR_Demo_Telecomms_Churn")
@pytest.mark.parametrize("framework", [SKLEARN, PYTORCH, KERAS, XGB, RDS])
def test_fit_variety(framework, variety_resources, resources, variety_data_names, tmp_path):

    # get data info from fixtures
    df = variety_data_names
    df_path = variety_resources.dataset(df)
    problem = variety_resources.problem(df)
    target = variety_resources.target(df)
    if problem == BINARY:
        class_labels = variety_resources.class_labels(df)
        if framework == RDS:
            # there's one annoying dataset where R needs 0 and 1 and python wants 1.0 and 0.0
            class_labels = [int(x) if type(x) is float else x for x in class_labels]
    # figure out language
    if framework == RDS:
        language = R_FIT
    else:
        language = PYTHON

    custom_model_dir = _create_custom_model_dir(
        resources,
        tmp_path,
        framework,
        problem,
        language,
        is_training=True,
        nested=False,
    )

    output = tmp_path / "output"
    output.mkdir()

    cmd = "{} fit --code-dir {} --input {} --verbose ".format(
        ArgumentsOptions.MAIN_COMMAND, custom_model_dir, df_path
    )
    if problem == ANOMALY:
        cmd += " --unsupervised"
    else:
        cmd += " --target {}".format(target)

    if problem == BINARY:
        cmd = _cmd_add_class_labels(cmd, class_labels)

    p, _, err = _exec_shell_cmd(
        cmd,
        "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
        assert_if_fail=False,
    )

    if p.returncode != 0:
        raise AssertionError(err)
