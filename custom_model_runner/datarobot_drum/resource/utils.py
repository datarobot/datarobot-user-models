"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import glob
import shutil
import shlex
import subprocess


from datarobot_drum.drum.enum import (
    ArgumentOptionsEnvVars,
    PythonArtifacts,
    RArtifacts,
    JavaArtifacts,
    JuliaArtifacts,
)

PYTHON = "python3"
JULIA = "julia"
R = "R"
R_ALL_PREDICT_STRUCTURED_HOOKS = "R_all_predict_structured_hooks"
R_FIT = "R_fit"
BINARY = "binary"
MULTICLASS = "multiclass"


def _create_custom_model_dir(
    resources,
    tmp_dir,
    framework,
    problem,
    language,
    is_training=False,
    nested=False,
    include_metadata=False,
    capitalize_artifact_extension=False,
):
    """
    Helper function for tests and validation to create temp custom model directory
    with relevant files and/or artifacts
    """
    custom_model_dir = tmp_dir / "custom_model"
    if nested:
        custom_model_dir = custom_model_dir.joinpath("nested_dir")
    custom_model_dir.mkdir(parents=True, exist_ok=True)
    if is_training:
        model_template_dir = resources.training_models(language, framework)
        if language == PYTHON:
            files = glob.glob(r"{}/*.py".format(model_template_dir))
        elif language == JULIA:
            files = glob.glob(r"{}/*.jl".format(model_template_dir))
        elif language in [R, R_ALL_PREDICT_STRUCTURED_HOOKS, R_FIT]:
            files = glob.glob(r"{}/*.r".format(model_template_dir)) + glob.glob(
                r"{}/*.R".format(model_template_dir)
            )
        if include_metadata:
            files.extend(glob.glob(r"{}/model-metadata.yaml".format(model_template_dir)))
        for filename in files:
            shutil.copy2(filename, custom_model_dir)
    else:
        artifact_filenames = resources.artifacts(framework, problem)
        if artifact_filenames is not None:
            if not isinstance(artifact_filenames, list):
                artifact_filenames = [artifact_filenames]
            for filename in artifact_filenames:
                if capitalize_artifact_extension:
                    name, ext = os.path.splitext(os.path.basename(filename))
                    if (
                        ext
                        in PythonArtifacts.ALL
                        + RArtifacts.ALL
                        + JavaArtifacts.ALL
                        + JuliaArtifacts.ALL
                    ):
                        ext = ext.upper()
                    dst = os.path.join(custom_model_dir, f"{name}{ext}")
                else:
                    dst = custom_model_dir
                shutil.copy2(filename, dst)

        fixture_filename, rename = resources.custom(language)
        if fixture_filename:
            shutil.copy2(fixture_filename, os.path.join(custom_model_dir, rename))
    return custom_model_dir


def _exec_shell_cmd(
    cmd,
    err_msg,
    assert_if_fail=True,
    process_obj_holder=None,
    env=os.environ,
    verbose=True,
    capture_output=True,
):
    """
    Wrapper used by tests and validation to run shell command.
    Can assert that the command does not fail (usually used for tests)
    or return process, stdout and stderr
    """
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        env=env,
        universal_newlines=True,
        encoding="utf-8",
        preexec_fn=os.setsid,
    )
    if process_obj_holder is not None:
        process_obj_holder.process = p

    if capture_output:
        (stdout, stderr) = p.communicate()
    else:
        stdout, stderr = None, None

    if process_obj_holder is not None:
        process_obj_holder.out_stream = stdout
        process_obj_holder.err_stream = stderr

    if verbose:
        if capture_output:
            if len(stdout):
                print("stdout: {}".format(stdout))
            if len(stderr):
                print("stderr: {}".format(stderr))
    if assert_if_fail:
        assert p.returncode == 0, err_msg

    return p, stdout, stderr


def _cmd_add_class_labels(
    cmd, labels, target_type, multiclass_label_file=None, pass_args_as_env_vars=False
):
    """
    utility used by tests and validation to add class label information to a drum command
    for binary or multiclass cases
    """
    if not labels or target_type == BINARY:
        pos = labels[1] if labels else "yes"
        neg = labels[0] if labels else "no"
        if pass_args_as_env_vars:
            os.environ[ArgumentOptionsEnvVars.POSITIVE_CLASS_LABEL] = pos
            os.environ[ArgumentOptionsEnvVars.NEGATIVE_CLASS_LABEL] = neg
        else:
            cmd = cmd + " --positive-class-label '{}' --negative-class-label '{}'".format(pos, neg)
    elif labels and target_type == MULTICLASS:
        if multiclass_label_file:
            multiclass_label_file.truncate(0)
            for label in labels:
                multiclass_label_file.write(label.encode("utf-8"))
                multiclass_label_file.write("\n".encode("utf-8"))
            multiclass_label_file.flush()
            if pass_args_as_env_vars:
                os.environ[ArgumentOptionsEnvVars.CLASS_LABELS_FILE] = multiclass_label_file.name
            else:
                cmd += " --class-labels-file {}".format(multiclass_label_file.name)
        else:
            if pass_args_as_env_vars:
                # stringify(for numeric) and join labels
                labels_str = " ".join(["{}".format(label) for label in labels])
                os.environ[ArgumentOptionsEnvVars.CLASS_LABELS] = labels_str
            else:
                # stringify(for numeric), quote (for spaces) and join labels
                labels_str = " ".join(['"{}"'.format(label) for label in labels])
                cmd += " --class-labels {}".format(labels_str)
    return cmd
