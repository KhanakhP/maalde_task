import joblib
import numpy as np

from .config import MODEL_INFO_PATH, MODEL_PATH
from .feature_extractor import FeatureExtractor


class Predictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)

        self.model = joblib.load(MODEL_PATH)
        self.model_info = joblib.load(MODEL_INFO_PATH)
        self.extractor = FeatureExtractor()

    def predict(self, image_path, rate):
        image_features = self.extractor.extract(image_path)
        rate = float(rate)
        rate_mean = float(self.model_info["rate_mean"])
        rate_features = [rate, np.log1p(rate), rate / rate_mean]
        features_with_rate = [[*image_features.reshape(1, -1)[0], *rate_features]]

        prediction = self.model.predict(features_with_rate)
        return max(0, round(float(prediction[0])))
