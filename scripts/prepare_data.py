import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.demand_predictor.data_prep import prepare_data


if __name__ == "__main__":
    prepare_data()
