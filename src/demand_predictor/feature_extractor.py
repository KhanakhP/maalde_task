import os

from .config import TORCH_HOME

os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class FeatureExtractor:
    def __init__(self):
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model(image)

        return features.squeeze().numpy()
