"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import os
from locust import HttpUser, task, between

DATSET_ENV_VAR_NAME = "LOCUST_DRUM_DATASET"


class MyUser(HttpUser):
    wait_time = between(0.1, 0.5)
    dataset = os.environ.get(DATSET_ENV_VAR_NAME)
    if dataset is None or not os.path.exists(dataset):
        print("Dataset doesn't exist: {}".format(dataset))
        print("Please provide dataset path using env var: {}".format(DATSET_ENV_VAR_NAME))
        exit(1)

    @task
    def predict(self):
        self.client.post("/predict/", files={"X": open(self.dataset)})
