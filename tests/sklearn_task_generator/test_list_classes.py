import pytest

from sklearn_task_generator import list_classes


class AnomalyEstimator(list_classes.OutlierMixin):
    def fit(self):
        pass

    def predict(self):
        pass


class RegressionEstimator(list_classes.RegressorMixin):
    def fit(self):
        pass

    def predict(self):
        pass


class ClassificationEstimator(list_classes.ClassifierMixin):
    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass


class TransformerTask(list_classes.TransformerMixin):
    def fit(self):
        pass

    def transform(self):
        pass


def copy_class(cls):
    """Create a copy such that delattr doesn't affect the original"""
    return type("ClassCopy", cls.__bases__, dict(cls.__dict__))


def test_filter_one_of_each_type():
    """Check that classes are sorted by type"""
    class_list = [
        AnomalyEstimator,
        RegressionEstimator,
        ClassificationEstimator,
        TransformerTask,
    ]
    (
        classifier_classes,
        regressor_classes,
        anomaly_classes,
        transformer_classes,
    ) = list_classes.filter_classes_by_type(class_list)
    assert anomaly_classes == [AnomalyEstimator]
    assert regressor_classes == [RegressionEstimator]
    assert classifier_classes == [ClassificationEstimator]
    assert transformer_classes == [TransformerTask]


@pytest.mark.parametrize(
    "task_class", [AnomalyEstimator, RegressionEstimator, ClassificationEstimator, TransformerTask]
)
def test_filter_no_fit(task_class):
    """Check that classes with no fit are rejected"""
    task_class = copy_class(task_class)
    assert list_classes.filter_classes_by_type([task_class]) != ([], [], [], [])
    delattr(task_class, "fit")
    assert list_classes.filter_classes_by_type([task_class]) == ([], [], [], [])


@pytest.mark.parametrize(
    "task_class", [AnomalyEstimator, RegressionEstimator, ClassificationEstimator]
)
def test_filter_no_predict(task_class):
    """Check that classes with no predict are rejected"""
    task_class = copy_class(task_class)
    assert list_classes.filter_classes_by_type([task_class]) != ([], [], [], [])
    delattr(task_class, "predict")
    assert list_classes.filter_classes_by_type([task_class]) == ([], [], [], [])


def test_filter_no_transform():
    """Check that transformers with no transform are rejected"""
    task_class = copy_class(TransformerTask)
    assert list_classes.filter_classes_by_type([task_class]) != ([], [], [], [])
    delattr(task_class, "transform")
    assert list_classes.filter_classes_by_type([task_class]) == ([], [], [], [])


@pytest.mark.parametrize(
    "task_class", [AnomalyEstimator, RegressionEstimator, ClassificationEstimator, TransformerTask]
)
def test_filter_private(task_class):
    """Check that private classes are excluded"""
    task_class = copy_class(task_class)
    assert list_classes.filter_classes_by_type([task_class]) != ([], [], [], [])
    task_class.__name__ = "_Private"
    assert list_classes.filter_classes_by_type([task_class]) == ([], [], [], [])


@pytest.mark.parametrize(
    "task_class", [AnomalyEstimator, RegressionEstimator, ClassificationEstimator, TransformerTask]
)
def test_filter_abstract(task_class):
    """Check that base classes are excluded"""
    task_class = copy_class(task_class)
    assert list_classes.filter_classes_by_type([task_class]) != ([], [], [], [])
    task_class.__name__ = "BaseClass"
    assert list_classes.filter_classes_by_type([task_class]) == ([], [], [], [])


def test_get_sklearn_classes():
    """Check that correct number of sklearn classes are retrieved"""
    # this is correct for v0.23.1
    expected_count = 412
    classes = list_classes.get_sklearn_classes()
    assert len(classes) == expected_count
    assert all("tests" not in str(cls) for cls in classes)
    assert all(cls.__module__.startswith("sklearn") for cls in classes)
    assert len(classes) == len(set(classes))


def test_get_valid_tasks():
    tasks_by_type = list_classes.get_valid_tasks()
    assert {task["target_type"] for task in tasks_by_type} == {
        "Anomaly",
        "Binary",
        "Regression",
        "Multiclass",
        "Transform",
    }

    # these are correct for v0.23.1
    expected_anomaly_count = 4
    expected_binary_count = 28
    expected_multiclass_count = 28
    expected_regression_count = 45
    expected_transform_count = 67

    anomaly_tasks = [t["task_classes"] for t in tasks_by_type if t["target_type"] == "Anomaly"][0]
    binary_tasks = [t["task_classes"] for t in tasks_by_type if t["target_type"] == "Binary"][0]
    multiclass_tasks = [
        t["task_classes"] for t in tasks_by_type if t["target_type"] == "Multiclass"
    ][0]
    regression_tasks = [
        t["task_classes"] for t in tasks_by_type if t["target_type"] == "Regression"
    ][0]
    transform_tasks = [t["task_classes"] for t in tasks_by_type if t["target_type"] == "Transform"][
        0
    ]

    assert len(anomaly_tasks) == expected_anomaly_count
    assert len(binary_tasks) == expected_binary_count
    assert len(multiclass_tasks) == expected_multiclass_count
    assert len(regression_tasks) == expected_regression_count
    assert len(transform_tasks) == expected_transform_count
