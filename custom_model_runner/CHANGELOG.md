# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] - 2020-06-12
### Added
- PMML support

## [1.1.1] - 2020-06-10
### Added
- language detection by artifact and custom.py/R

### Changed
- now version check is called as `drum --version`

## [1.1.0] - 2020-05-29
### Changed
- public envs are pinned to drum==1.1.0

## [1.0.20] - 2020-05-28
### Changed
- the name to **drum**

## [1.0.19] - 2020-05-18
### Changed
- build only Python 3 wheel

## [1.0.18] - 2020-05-12
### Added
- cmrunner for custom model fit
- fit code to python3_sklearn model template

### Fixed
- printing result if `--output` is not provided

## [1.0.17] - 2020-05-07
### Changed
- change predict() hook to score()
- change signature to `score(data, mode, **kwargs)`
- change command `cmrun predict` to `cmrun score`

### Added
- `new` subcommand for model templates creation 

## [1.0.16] - 2020-05-05
### Changed
- unpin rpy2 dependency version

## [1.0.15] - 2020-05-04
### Changed
- require to use sub-command, e.g. `cmrun predict`  

## [1.0.14] - 2020-04-30
### Changed
- refactored command/subcommand usage

## [1.0.13] - 2020-04-27
### Added
- CMRUNNER_JAVA_XMX env var to pass max heap memory value to JVM
- autocompletion support
- performance tests feature

## [1.0.12] - 2020-04-23
### Changed
- optimized py4j usage for java predictor

## [1.0.11] - 2020-04-21
### Changed
- improved class labels handling in R binary classification

## [1.0.10] - 2020-04-20
### Added
- --threaded flag which allows to start cmrunner prediction server(Flask) in threaded mode
### Changes
- replaced custom hooks for R with
    - init() -> NULL
    - load_model(input_dir: character) -> Any
    - transform(data: data.frame, model: Any) -> data.frame
    - predict(data: data.frame, model: Any, positive_class_label: character, negative_class_label: character) -> data.frame
    - post_process(predictions: data.frame, model: Any) -> data.frame
- unify argument names in file reading/writing components

## [1.0.9] - 2020-04-16
### Changes
- improved prediction server response serialization

## [1.0.8] - 2020-04-10
### Added
- support for --docker command line option. Running the model inside a docker image

## [1.0.7] - 2020-04-08
### Added
- support Java Codegen models in batch and server modes

## [1.0.6] - 2020-04-03
### Changed
- replaced custom hooks for python with
    - init() -> None
    - load_model(input_dir: str) -> Any
    - transform(data: DataFrame, model: Any) -> DataFrame
    - predict(data: DataFrame, model: Any, positive_class_label: str, negative_class_label: str) -> DataFrame
    - post_process(predictions: DataFrame, model: Any) -> DataFrame

## [1.0.5] - 2020-03-30
### Added
- handle class labels for scikit-learn and R binary classification

## [1.0.4] - 2020-03-20
### Added
- make rpy2 dependency optional

## [1.0.3] - 2020-03-20
### Added
- R models support

## [1.0.2] - 2020-03-19
### Changed
- Change dep: mlpiper==2.2.0
- Bump version to push to test PyPI

## [1.0.1] - 2020-03-09
### Added
- Added wheelhouse for distributing "mlpiper"
- Added README
