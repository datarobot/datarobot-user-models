"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from setuptools import setup, find_packages
import os

from datarobot_drum.drum.description import version, project_name
from datarobot_drum.drum.enum import SupportedFrameworks, extra_deps

# The directory containing this file
root = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(root, "requirements.txt")) as f:
    requirements = f.read().splitlines()

with open(os.path.join(root, "README.md")) as f:
    long_desc = f.read()

extras_require = {
    "scikit-learn": extra_deps[SupportedFrameworks.SKLEARN],
    "torch": extra_deps[SupportedFrameworks.TORCH],
    "keras": extra_deps[SupportedFrameworks.KERAS],
    "xgboost": extra_deps[SupportedFrameworks.XGBOOST],
    "R": ["rpy2;python_version>='3.6'"],
    "pypmml": extra_deps[SupportedFrameworks.PYPMML],
    "trainingModels": ["datarobot>=2.26.0"],
    "uwsgi": ["uwsgi"],
}

setup(
    name=project_name,
    version=version,
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
        "License :: OSI Approved :: Apache Software License",
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
