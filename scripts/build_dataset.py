import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.demand_predictor.dataset import build_dataset


if __name__ == "__main__":
    build_dataset()
