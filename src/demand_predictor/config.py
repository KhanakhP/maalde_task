from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_EXCEL_PATH = ROOT_DIR / "AI ML Task Sheet.xlsx"
SHEET_NAME = "sales data"

DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = ROOT_DIR / "renamed_images"
MODEL_DIR = ROOT_DIR / "models"
TORCH_HOME = ROOT_DIR / ".torch"

PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
MODEL_DATASET_PATH = DATA_DIR / "model_dataset.csv"
X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"

MODEL_PATH = MODEL_DIR / "model.pkl"
MODEL_INFO_PATH = MODEL_DIR / "model_info.pkl"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
