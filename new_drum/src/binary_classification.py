from .estimator import EstimatorInterface


class BinaryClassificationInterface(EstimatorInterface):
    def __init__(self):
        self.prediction_columns = ["Predictions"]
