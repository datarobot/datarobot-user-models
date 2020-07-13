def custom(estimator):
    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        raise ValueError("You don't have scikit-learn installed, so you can't use this hook")
    if not isinstance(estimator, BaseEstimator):
        raise ValueError("The object passed in does not inherit from BaseEstimator")
    estimator.is_custom = True
    return estimator
