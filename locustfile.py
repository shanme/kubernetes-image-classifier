from pathlib import Path

from locust import HttpUser, between, task

IMAGE_CONTENT = Path("image.png").read_bytes()


class ImageClassifierUser(HttpUser):
    """Simulates a virtual user performing image classification requests.

    This class defines the behavior of a user that sends periodic image upload
    requests to the service. It is used to generate synthetic load and monitor
    system metrics, response latency, and pod scaling behavior.
    """

    wait_time = between(1, 5)

    @task
    def upload_image(self):
        """Sends a sample image file to the /predict endpoint.

        The task uploads a pre-loaded image from memory as multipart/form-data
        to measure the API's throughput and response time under load.
        """
        self.client.post(
            "/predict",
            files={
                "image": ("image.png", IMAGE_CONTENT, "image/png"),
            },
            headers={"Connection": "close"},
        )