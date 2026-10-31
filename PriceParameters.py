"""Paramètres propres au pricer : méthode d'arbre, nombre de pas, pruning, moteur."""
from __future__ import annotations
import datetime as dt
import math
from enum import Enum

class TreeMethod(Enum):
    BINOMIAL = "binomial"          # arbre binomial (pas trinomial au dividende)
    TRINOMIAL = "trinomial"
    @property
    def nb_nodes(self) -> int:
        return 3 if self is TreeMethod.TRINOMIAL else 2
    @property
    def label(self) -> str:
        return "Trinomial" if self is TreeMethod.TRINOMIAL else "Binomial"

class PriceParameters:
    DAYS_IN_YEAR = 365.0

    def __init__(self, pricing_date: dt.date, nb_steps: int, tree_method: TreeMethod = TreeMethod.TRINOMIAL,
                 pruning: bool = True, p_min: float = 1e-8, engine: str = "vector",
                 pricing_method: str = "backward") -> None:
        self.pricing_date = pricing_date
        self.nb_steps = nb_steps
        self.tree_method = tree_method
        self.pruning = pruning
        self.p_min = p_min
        self.engine = engine                    # "vector" (numpy) ou "objects" (nœuds Node)
        self.pricing_method = pricing_method    # "backward" ou "recursive" (moteur objets)

    def max_unpruned_steps(self, engine: str | None = None) -> int:
        """Taille maximale d'un arbre sans pruning (mémoire)."""
        if (engine or self.engine) == "vector":
            return 3000 if self.tree_method.nb_nodes == 2 else 2000
        return 1000 if self.tree_method.nb_nodes == 2 else 700

    @staticmethod
    def nb_steps_from_error(max_error: float, s0: float, vol: float, T: float) -> int:
        """Nombre de pas minimal pour que l'erreur de l'arbre reste sous max_error.
        N = sigma²T / ln(1 + 8·√(2π)·max_error·√(e^(sigma²T) - 1) / (3·S0)).
        """
        if max_error <= 0.0:
            raise ValueError("L'erreur maximale doit être strictement positive.")
        if vol <= 0.0 or T <= 0.0:
            return 1                                       # aucune incertitude : un pas suffit
        down = math.log(1.0 + 8.0 * math.sqrt(2.0 * math.pi) * max_error * math.sqrt(math.expm1(vol ** 2 * T))
                        / (3.0 * s0))
        return max(1, round(vol ** 2 * T / down))

    @staticmethod
    def year_fraction(d0: dt.date, d1: dt.date) -> float:
        """Fraction d'année ACT/365 entre deux dates."""
        return (d1 - d0).days / PriceParameters.DAYS_IN_YEAR
