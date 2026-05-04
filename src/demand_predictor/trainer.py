import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR, MODEL_INFO_PATH, MODEL_PATH, X_PATH, Y_PATH


def train():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    image_feature_count = X.shape[1] - 3
    rate_mean = float(X[:, image_feature_count].mean())

    preprocessor = ColumnTransformer(
        [
            (
                "image",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=24, random_state=42)),
                    ]
                ),
                list(range(image_feature_count)),
            ),
            ("rate", "passthrough", list(range(image_feature_count, X.shape[1]))),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    max_features="sqrt",
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"R2 score: {r2:.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(
        {
            "feature_count": X.shape[1],
            "image_feature_count": image_feature_count,
            "rate_mean": rate_mean,
            "target": "total_qty",
            "features": ["image", "rate", "log_rate", "normalized_rate"],
        },
        MODEL_INFO_PATH,
    )

    print(f"Model saved to {MODEL_PATH}")
