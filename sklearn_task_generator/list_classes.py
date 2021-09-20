import glob
import importlib
import inspect
import os
from pprint import pprint
from typing import List, Tuple, Type, TypedDict

import sklearn
from sklearn.base import ClassifierMixin, OutlierMixin, RegressorMixin, TransformerMixin

# this code is specific to Scikit Learn v0.23.1.
# Other versions may have added or removed some classes
SKLEARN_VERSION = "0.23.1"

# path to sklearn files
SKLEARN_PATH = sklearn.__file__.replace("__init__.py", "")

# list of classes to skip that can't be automatically filtered out by inspection
SKIP_TASKS = {
    # Wrapper classes
    "sklearn.base.MultiOutputMixin",
    "sklearn.ensemble._stacking._BaseStacking",
    "sklearn.ensemble._stacking.StackingClassifier",
    "sklearn.ensemble._stacking.StackingRegressor",
    "sklearn.ensemble._voting._BaseVoting",
    "sklearn.ensemble._voting.VotingClassifier",
    "sklearn.ensemble._voting.VotingRegressor",
    "sklearn.feature_selection._from_model.SelectFromModel",
    "sklearn.feature_selection._rfe.RFE",
    "sklearn.feature_selection._rfe.RFECV",
    "sklearn.feature_selection._sequential.SequentialFeatureSelector",
    "sklearn.multiclass.OneVsOneClassifier",
    "sklearn.multiclass.OneVsRestClassifier",
    "sklearn.multiclass.OutputCodeClassifier",
    "sklearn.multiclass.OutputCodeClassifier",
    "sklearn.multioutput._BaseChain",
    "sklearn.multioutput.ClassifierChain",
    "sklearn.multioutput.MultiOutputClassifier",
    "sklearn.multioutput._MultiOutputEstimator",
    "sklearn.multioutput.MultiOutputRegressor",
    "sklearn.multioutput.RegressorChain",
    # Not for tabular data
    "sklearn.feature_extraction._dict_vectorizer.DictVectorizer",
    # Requires callable parameter
    "sklearn.compose._target.TransformedTargetRegressor",
    # Testing class
    "sklearn.utils._mocking.CheckingClassifier",
    # Multi-label only
    "sklearn.linear_model._coordinate_descent.MultiTaskElasticNet",
    "sklearn.linear_model._coordinate_descent.MultiTaskElasticNetCV",
    "sklearn.linear_model._coordinate_descent.MultiTaskLasso",
    "sklearn.linear_model._coordinate_descent.MultiTaskLassoCV",
    # Only works for a single feature
    "sklearn.isotonic.IsotonicRegression",
}


class TargetTasks(TypedDict):
    target_type: str
    task_classes: List[Type]


def get_sklearn_classes() -> List[Type]:
    """Get a list of Scikit Learn classes"""

    assert (
        sklearn.__version__ == SKLEARN_VERSION
    ), f"Version {SKLEARN_VERSION} of scikit-learn required, found {sklearn.__version__}."

    seen_class = set()
    sklearn_classes = []
    # loop over all sklearn modules
    for source_file in glob.glob(os.path.join(SKLEARN_PATH, "**", "*.py"), recursive=True):
        # skip tests
        if "/tests/" in source_file:
            continue
        module_name = source_file.replace(SKLEARN_PATH, "").replace("/", ".").replace(".py", "")
        module = importlib.import_module(f"sklearn.{module_name}")
        # list all classes in module
        for _, obj in inspect.getmembers(module, inspect.isclass):
            class_name = f"{obj.__module__}.{obj.__name__}"

            if class_name in SKIP_TASKS:
                continue

            # skip non-sklearn classes
            if not class_name.startswith("sklearn"):
                continue

            # skip duplicates
            if class_name in seen_class:
                continue
            seen_class.add(class_name)
            sklearn_classes.append(obj)
    return sklearn_classes


def filter_classes_by_type(
    sklearn_classes: List[Type],
) -> Tuple[List[Type], List[Type], List[Type], List[Type]]:
    """Select classes which can be used in DR blueprints and sort them by type"""
    anomaly_classes = []
    classifier_classes = []
    regressor_classes = []
    transformer_classes = []
    for obj in sklearn_classes:
        has_fit = hasattr(obj, "fit")
        has_transform = hasattr(obj, "transform")
        has_predict = hasattr(obj, "predict")
        has_predict_proba = hasattr(obj, "predict_proba")

        is_transformer = issubclass(obj, TransformerMixin)
        is_classifier = issubclass(obj, ClassifierMixin)
        is_regressor = issubclass(obj, RegressorMixin)
        is_anomaly = issubclass(obj, OutlierMixin)

        is_estimator = (
            has_fit
            and (has_transform or has_predict)
            and (is_transformer or is_classifier or is_regressor or is_anomaly)
        )
        is_public = not obj.__name__.startswith("_")
        is_concrete = not (obj.__name__.startswith("Base") or inspect.isabstract(obj))

        # only add public non-abstract estimators
        if is_estimator and is_public and is_concrete:
            if is_anomaly:
                anomaly_classes.append(obj)
            if is_transformer:
                transformer_classes.append(obj)
            if is_classifier and has_predict_proba:
                classifier_classes.append(obj)
            if is_regressor:
                regressor_classes.append(obj)
    return classifier_classes, regressor_classes, anomaly_classes, transformer_classes


def get_valid_tasks() -> List[TargetTasks]:
    """Get a list of Scikit Learn classes appropriate for use in DataRobot blueprints"""
    sklearn_classes = get_sklearn_classes()
    (
        classifier_classes,
        regressor_classes,
        anomaly_classes,
        transformer_classes,
    ) = filter_classes_by_type(sklearn_classes)
    return [
        {"target_type": "Anomaly", "task_classes": anomaly_classes},
        {"target_type": "Binary", "task_classes": classifier_classes},
        {"target_type": "Multiclass", "task_classes": classifier_classes},
        {"target_type": "Regression", "task_classes": regressor_classes},
        {"target_type": "Transform", "task_classes": transformer_classes},
    ]


if __name__ == "__main__":
    pprint(get_valid_tasks())
