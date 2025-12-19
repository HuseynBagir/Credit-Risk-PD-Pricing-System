from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PricingConfig:
    lgd: float = 0.60
    funding_cost_apr: float = 0.05
    opex_rate: float = 0.01

def expected_loss(pd: np.ndarray, ead: np.ndarray, lgd: float) -> np.ndarray:
    return pd * lgd * ead

def decide_and_price(pd: np.ndarray):
    # 1=approve, 0=decline
    decision = (pd <= 0.10).astype(int)
    apr = np.full_like(pd, np.nan, dtype=float)

    apr[pd <= 0.02] = 0.10
    apr[(pd > 0.02) & (pd <= 0.05)] = 0.14
    apr[(pd > 0.05) & (pd <= 0.10)] = 0.18

    return decision, apr

def expected_profit(pd: np.ndarray, ead: np.ndarray, apr: np.ndarray, cfg: PricingConfig):
    """
    Very simple proxy:
      interest_income ≈ ead * apr
      costs ≈ ead * funding_cost_apr + ead * opex_rate + EL
    """
    el = expected_loss(pd, ead, cfg.lgd)
    interest_income = ead * np.nan_to_num(apr, nan=0.0)
    costs = ead * (cfg.funding_cost_apr + cfg.opex_rate) + el
    return interest_income - costs
