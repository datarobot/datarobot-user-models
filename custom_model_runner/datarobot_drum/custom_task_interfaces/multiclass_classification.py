from datarobot_drum.custom_task_interfaces.estimator_interface import EstimatorInterface


class MultiClassificationInterface(EstimatorInterface):
    def __init__(self):
        self.prediction_columns = ["Predictions"]
