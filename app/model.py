import io
from typing import cast

import torch
from PIL import Image
from torch import Tensor
from torchvision.models import get_model, get_model_weights


class ImageClassifier:
    """A high-level interface for image classification using PyTorch.

    This class handles model initialization, image preprocessing, and inference
    execution. It uses pre-trained weights from torchvision to categorize
    images into predefined labels.
    """

    def __init__(self, settings):
        self.model = get_model(settings.model_name, weights=settings.model_weights)
        self.model_weights = get_model_weights(settings.model_name)[
            settings.model_weights
        ]
        self.model.eval()

        self.transform = self.model_weights.transforms()
        self.categories = self.model_weights.meta["categories"]

    def _preprocess(self, image_bytes: bytes) -> Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = cast(Tensor, self.transform(image))
        tensor = tensor.unsqueeze(0)

        return tensor

    def predict(self, image_bytes: bytes) -> dict[str, float]:
        """Performs inference on the image and returns classification results.

        The method executes the forward pass in evaluation mode without gradient
        tracking. It calculates class probabilities using Softmax and extracts
        the top-1 prediction.

        Args:
            image_bytes: The raw content of the image file.

        Returns:
            A dictionary containing the predicted 'label' and 'confidence' score.
        """
        tensor = self._preprocess(image_bytes=image_bytes)

        with torch.no_grad():
            output: Tensor = self.model(tensor)

            probabilities: Tensor = torch.nn.functional.softmax(output, dim=1)

            max_confidence, index = torch.max(probabilities, dim=1)

            predicted_label = self.categories[index.item()]

            return {"label": predicted_label, "confidence": max_confidence.item()}
