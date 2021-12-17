from datarobot_drum.custom_task_interfaces.estimator_interface import EstimatorInterface


class RegressionInterface(EstimatorInterface):
    def __init__(self):
        self.prediction_columns = ["Predictions"]
