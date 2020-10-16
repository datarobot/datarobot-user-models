import os
import glob
import shutil
import subprocess


from .constants import PYTHON, R, R_ALL_PREDICT_STRUCTURED_HOOKS, R_FIT


def _create_custom_model_dir(
    resources,
    tmp_dir,
    framework,
    problem,
    language,
    is_training=False,
    nested=False,
):
    custom_model_dir = tmp_dir / "custom_model"
    if nested:
        custom_model_dir = custom_model_dir.joinpath("nested_dir")
    custom_model_dir.mkdir(parents=True, exist_ok=True)
    if is_training:
        model_template_dir = resources.training_models(language, framework)

        if language == PYTHON:
            files = glob.glob(r"{}/*.py".format(model_template_dir))
        elif language in [R, R_ALL_PREDICT_STRUCTURED_HOOKS, R_FIT]:
            files = glob.glob(r"{}/*.r".format(model_template_dir)) + glob.glob(
                r"{}/*.R".format(model_template_dir)
            )

        for filename in files:
            shutil.copy2(filename, custom_model_dir)
    else:
        artifact_filenames = resources.artifacts(framework, problem)
        if artifact_filenames is not None:
            if not isinstance(artifact_filenames, list):
                artifact_filenames = [artifact_filenames]
            for filename in artifact_filenames:
                shutil.copy2(filename, custom_model_dir)

        fixture_filename, rename = resources.custom(language)
        if fixture_filename:
            shutil.copy2(fixture_filename, os.path.join(custom_model_dir, rename))
    return custom_model_dir


def _exec_shell_cmd(cmd, err_msg, assert_if_fail=True, process_obj_holder=None, env=os.environ):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=env,
        universal_newlines=True,
    )
    if process_obj_holder is not None:
        process_obj_holder.process = p

    (stdout, stderr) = p.communicate()

    if process_obj_holder is not None:
        process_obj_holder.out_stream = stdout
        process_obj_holder.err_stream = stderr

    if p.returncode != 0:
        print("stdout: {}".format(stdout))
        print("stderr: {}".format(stderr))
        if assert_if_fail:
            assert p.returncode == 0, err_msg

    return p, stdout, stderr


def _cmd_add_class_labels(cmd, labels):
    if not labels or len(labels) == 2:
        pos = labels[1] if labels else "yes"
        neg = labels[0] if labels else "no"
        cmd = cmd + " --positive-class-label {} --negative-class-label {}".format(pos, neg)
    elif labels and len(labels) > 2:
        wrapped_labels = ["\"{}\"".format(label) for label in labels]
        cmd += " --class-labels {}".format(" ".join(wrapped_labels))
    return cmd
