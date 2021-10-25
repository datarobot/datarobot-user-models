"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from locust import HttpUser, task, between


class MyUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict_unstructured(self):
        # unstructured model allows to send empty data
        self.client.post("/predictUnstructured/", data=None)
