from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    data_csv: Path = Path("data/accepted/accepted_2007_to_2018Q4.csv")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")

@dataclass(frozen=True)
class SplitConfig:
    # Time split by issue_d (inclusive)
    train_end: str = "2016-09-01"
    valid_end: str = "2017-09-01"
    # test = > valid_end

@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42

    # XGBoost params (reasonable defaults for tabular credit risk)
    n_estimators: int = 800
    learning_rate: float = 0.03
    max_depth: int = 4
    subsample: float = 0.9
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: float = 2.0
