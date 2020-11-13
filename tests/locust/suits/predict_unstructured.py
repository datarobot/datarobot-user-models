from locust import HttpUser, task, between


class MyUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict_unstructured(self):
        # unstructured model allows to send empty data
        self.client.post("/predictUnstructured/", data=None)
