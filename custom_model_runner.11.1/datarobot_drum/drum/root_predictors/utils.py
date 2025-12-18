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
import time
from queue import Queue, Empty
from threading import Thread

from datarobot_drum.drum.common import get_drum_logger
from datarobot_drum.drum.enum import (
    ArgumentOptionsEnvVars,
    PythonArtifacts,
    RArtifacts,
    JavaArtifacts,
    JuliaArtifacts,
)

logger = get_drum_logger(__name__)

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
        # An artifact can be:
        # * a single artifact path
        # * a tuple(path, Optional (target file name)),
        # * list of artifact paths/tuples
        artifact_filenames_or_tuples = resources.artifacts(framework, problem)
        if artifact_filenames_or_tuples is not None:
            if not isinstance(artifact_filenames_or_tuples, list):
                artifact_filenames_or_tuples = [artifact_filenames_or_tuples]
            for filename_or_tuple in artifact_filenames_or_tuples:
                source_filepath, target_name = (
                    (filename_or_tuple[0], filename_or_tuple[1])
                    if isinstance(filename_or_tuple, tuple)
                    else (filename_or_tuple, None)
                )
                source_filename = os.path.basename(source_filepath)
                target_name = target_name or source_filename
                dst = os.path.join(custom_model_dir, f"{target_name}")

                if capitalize_artifact_extension:
                    name, ext = os.path.splitext(source_filename)
                    if (
                        ext
                        in PythonArtifacts.ALL
                        + RArtifacts.ALL
                        + JavaArtifacts.ALL
                        + JuliaArtifacts.ALL
                    ):
                        ext = ext.upper()
                    dst = os.path.join(custom_model_dir, f"{name}{ext}")
                shutil.copy2(source_filepath, dst)

        fixture_filename, rename = resources.custom(language)
        if fixture_filename:
            shutil.copy2(fixture_filename, os.path.join(custom_model_dir, rename))
    return custom_model_dir


def _queue_output(stdout, stderr, queue):
    """Helper function to stream output from subprocess to a queue."""
    for line in iter(stdout.readline, b""):
        queue.put(line)
    for line in iter(stderr.readline, b""):
        queue.put(line)
    stdout.close()
    stderr.close()


def _stream_p_open(subprocess_popen: subprocess.Popen):
    """Wraps the Popen object to stream output in a separate thread.
    This realtime output of the stdout and stderr of the process
    is streamed to the terminal.

    This logic was added because there is not a direct python execution flow for DRUM.
    SEE: https://datarobot.atlassian.net/browse/RAPTOR-12510
    """
    logger_queue = Queue()
    logger_thread = Thread(
        target=_queue_output,
        daemon=True,
        args=(subprocess_popen.stdout, subprocess_popen.stderr, logger_queue),
    )
    logger_thread.start()
    while True:
        try:
            # Stream output if available
            line = logger_queue.get_nowait()
            logger.info(line.strip()) if len(line.strip()) > 0 else None
        except Empty:
            # Check if the process has terminated
            if subprocess_popen.poll() is not None:
                break
            time.sleep(1)
        except Exception:
            break
    # Output has already been displayed
    return "", ""


def _exec_shell_cmd(
    cmd,
    err_msg,
    assert_if_fail=True,
    process_obj_holder=None,
    env=os.environ,
    verbose=True,
    capture_output=True,
    stream_output=False,
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
        if stream_output:
            (stdout, stderr) = _stream_p_open(p)
        else:
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
