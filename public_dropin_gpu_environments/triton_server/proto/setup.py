import setuptools
from setuptools import setup

setup(
    name="triton_model_config",
    version="24.02",
    install_requires=[
        "protobuf==5.26.0",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
)
