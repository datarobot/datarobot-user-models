MAGIC_MARKER = "__drum_auto_fit__"


def drum_autofit(estimator):
    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        raise ValueError("You don't have scikit-learn installed, so you can't use this hook")
    if not isinstance(estimator, BaseEstimator):
        raise ValueError("The object passed in does not inherit from BaseEstimator")
    setattr(estimator, MAGIC_MARKER, True)
    return estimator
