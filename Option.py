from __future__ import annotations
import datetime as dt
from enum import Enum
import numpy as np

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OptionExerciceType(Enum):
    EUROPEEN = "europeen"
    AMERICAINE = "americaine"

class Option:
    def __init__(self, K: float, maturity: dt.date, optionType: OptionType | str,
                 optionExerciceType: OptionExerciceType | str):
        if K <= 0.0:
            raise ValueError("Le strike doit être strictement positif.")
        self.K: float = float(K)
        self.maturity = maturity
        self.call_put: str = OptionType(optionType).value        # "call" ou "put"
        self.eur_am = OptionExerciceType(optionExerciceType)

    def is_american(self) -> bool:
        return self.eur_am is OptionExerciceType.AMERICAINE

    def payoff(self, und_price: float) -> float:
        if self.call_put == "call":
            return max(und_price - self.K, 0.0)
        return max(self.K - und_price, 0.0)

    def payoff_array(self, und_price: np.ndarray) -> np.ndarray:
        """Version vectorisée du payoff."""
        S = np.asarray(und_price, dtype=float)
        if self.call_put == "call":
            return np.maximum(S - self.K, 0.0)
        return np.maximum(self.K - S, 0.0)
