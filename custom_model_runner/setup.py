"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import importlib.util
from pathlib import Path
from setuptools import setup, find_packages

def direct_import(path):
    # Direct imports are needed because datarobot_drum/__init__.py imports other modules
    # that depend on 3rd party libraries and `setup.py` **must** be able to run in a blank
    # virutalenv. This code snippet was adapted from and simply loads the module directly
    # without loading all its parents:
    #   https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    module_path = path.resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# The directory containing this file
root = Path(__file__).absolute().parent
description = direct_import(root / 'datarobot_drum' / 'drum' / 'description.py')
enum = direct_import(root / 'datarobot_drum' / 'drum' / 'enum.py')


with open(root / "requirements.txt") as f:
    requirements = f.read().splitlines()

with open(root / "README.md") as f:
    long_desc = f.read()

extras_require = {
    "scikit-learn": enum.extra_deps[enum.SupportedFrameworks.SKLEARN],
    "torch": enum.extra_deps[enum.SupportedFrameworks.TORCH],
    "keras": enum.extra_deps[enum.SupportedFrameworks.KERAS],
    "xgboost": enum.extra_deps[enum.SupportedFrameworks.XGBOOST],
    "R": ["rpy2==3.5.2;python_version>='3.6'"],
    "pypmml": enum.extra_deps[enum.SupportedFrameworks.PYPMML],
    "uwsgi": ["uwsgi"],
}

setup(
    name=description.project_name,
    version=description.version,
    description="DRUM - develop, test and deploy custom models",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="http://datarobot.com",
    author="DataRobot",
    author_email="support@datarobot.com",
    license="Apache License, Version 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Unix",
    ],
    zip_safe=False,
    include_package_data=True,
    packages=find_packages("."),
    package_data={
        "": ["*.json", "*.jar", "*.R", "*.j2", "*.jl", "*.toml"],
        "datarobot_drum.resource.pipelines": ["*"],
        "datarobot_drum.resource.default_typeschema": ["*.yaml"],
    },
    scripts=["bin/drum"],
    install_requires=requirements,
    extras_require=extras_require,
    python_requires=">=3.4,<3.10",
)
