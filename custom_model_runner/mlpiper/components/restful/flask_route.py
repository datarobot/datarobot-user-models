"""
For internal use only. A decorator that is used to indicate a REST endpoint function.
Reference: http://flask.pocoo.org/docs/0.12/api/#flask.Flask.route
"""
from functools import wraps


class FlaskRoute(object):
    METHODS_KEY = "methods"
    RAW_KEY = "raw"
    _routes = []

    def __init__(self, rule, **options):
        self._rule = rule
        self._options = options
        self._raw = self._options.pop(FlaskRoute.RAW_KEY, False)

        if FlaskRoute.METHODS_KEY not in self._options:
            self._options[FlaskRoute.METHODS_KEY] = ["GET", "POST"]

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        endpoint = self._options.pop("endpoint", None)
        if not any(self._rule == e[0] for e in FlaskRoute._routes):
            FlaskRoute._routes.append(
                (self._rule, endpoint, f.__name__, self._options, self._raw)
            )

        return wrapper

    @staticmethod
    def routes():
        return FlaskRoute._routes
