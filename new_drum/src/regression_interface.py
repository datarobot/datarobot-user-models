from estimator import EstimatorInterface


class RegressionInterface(EstimatorInterface):
    def __init__(self):
        self.prediction_columns = ["Predictions"]
