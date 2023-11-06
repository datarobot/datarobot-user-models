"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from pathlib import Path
from setuptools import setup, find_packages


# The directory containing this file
root = Path(__file__).absolute().parent

# It is not safe to import modules from the package we are building so this is an
# idiomatic way to keep things DRY.
meta = {}
with open(root / "datarobot_drum" / "drum" / "enum.py", mode="r", encoding="utf-8") as f:
    exec(f.read(), meta)
with open(root / "datarobot_drum" / "drum" / "description.py", mode="r", encoding="utf-8") as f:
    exec(f.read(), meta)
SupportedFrameworks = meta["SupportedFrameworks"]
extra_deps = meta["extra_deps"]

with open(root / "requirements.txt") as f:
    requirements = f.read().splitlines()

with open(root / "README.md") as f:
    long_desc = f.read()

extras_require = {framework: extra_deps[framework] for framework in SupportedFrameworks.ALL}
extras_require["R"] = ["rpy2==3.5.8;python_version>='3.6'"]
extras_require["uwsgi"] = ["uwsgi"]

setup(
    name=meta["project_name"],
    version=meta["__version__"],
    description="DRUM - develop, test and deploy custom models",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="http://datarobot.com",
    project_urls={
        "Source": "https://github.com/datarobot/datarobot-user-models",
        "Changelog": "https://github.com/datarobot/datarobot-user-models/blob/master/custom_model_runner/CHANGELOG.md",
    },
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
    scripts=["bin/drum", "../drapps/drapps.py"],
    install_requires=requirements,
    extras_require=extras_require,
    python_requires=">=3.4,<3.12",
)
