import numpy as np
import pandas as pd

from .config import (
    IMAGE_DIR,
    IMAGE_EXTENSIONS,
    MODEL_DATASET_PATH,
    PROCESSED_DATA_PATH,
    X_PATH,
    Y_PATH,
)
from .feature_extractor import FeatureExtractor


def normalize_code(value):
    return str(int(float(value))).zfill(8)


def find_image_path(code):
    for ext in IMAGE_EXTENSIONS:
        image_path = IMAGE_DIR / f"{code}{ext}"
        if image_path.exists():
            return image_path
    return None


def build_dataset():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    extractor = FeatureExtractor()
    rate_mean = float(df["avg_rate"].mean())

    X = []
    y = []
    rows = []

    for _, row in df.iterrows():
        code = normalize_code(row["code"])
        image_path = find_image_path(code)

        if image_path:
            image_features = extractor.extract(image_path)
            rate = float(row["avg_rate"])
            rate_features = np.array([rate, np.log1p(rate), rate / rate_mean])
            features_with_rate = np.append(image_features, rate_features)

            X.append(features_with_rate)
            y.append(row["total_qty"])
            rows.append(
                {
                    "code": code,
                    "image_path": str(image_path.relative_to(IMAGE_DIR.parent)),
                    "avg_rate": row["avg_rate"],
                    "total_qty": row["total_qty"],
                }
            )
        else:
            print(f"Missing image for code: {code}")

    if not X:
        raise ValueError("No matching images found. Check processed_data.csv and renamed_images/.")

    np.save(X_PATH, np.array(X))
    np.save(Y_PATH, np.array(y))
    pd.DataFrame(rows).to_csv(MODEL_DATASET_PATH, index=False)

    print(f"Dataset built with {len(X)} samples")
