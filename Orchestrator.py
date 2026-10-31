from __future__ import annotations
import gc
import time
import tracemalloc
import numpy as np
from Black_Scholes import Black_Scholes, GreeksBS
from Market import Market
from Option import Option
from OptionPricer import OptionPricerTree
from PriceParameters import PriceParameters
from Sensibilities import GREEKS, GreeksTree
from Tree import Tree, TreeNode, TreeVectorized

class Orchestrator:
    """Construit, valorise et libère les arbres ; relie pricing, Black-Scholes et grecques."""
    @staticmethod
    def build(market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> Tree:
        """Arbre du moteur demandé : TreeNode (objets Node) ou TreeVectorized (tableaux numpy)."""
        tree_class = TreeNode if pricer.engine == "objects" else TreeVectorized
        return tree_class(market, option, pricer, pre_steps)

    @staticmethod
    def price(market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> float:
        """Prix de l'arbre (backward, ou récursif si demandé en moteur objets) ; l'arbre est libéré aussitôt."""
        tree = Orchestrator.build(market, option, pricer, pre_steps)
        try:
            return OptionPricerTree(tree).price(pricer.pricing_method if pricer.engine == "objects" else "backward")
        finally:
            tree.release()

    @staticmethod
    def pricing_report(market: Market, option: Option, pricer: PriceParameters) -> dict:
        """Pricing principal (bouton « Lancer le pricer ») : prix, temps, nombre de nœuds, et l'arbre vectorisé
        valorisé gardé pour les analyses (en moteur objets, l'arbre de Node est libéré et l'arbre vectorisé,
        identique, est construit pour l'affichage)."""
        t0 = time.perf_counter()
        tree = Orchestrator.build(market, option, pricer)
        t1 = time.perf_counter()
        pricer_tree = OptionPricerTree(tree)
        price = pricer_tree.price(pricer.pricing_method if pricer.engine == "objects" else "backward")
        t2 = time.perf_counter()
        if isinstance(tree, TreeNode):
            tree.release()
            tree = Orchestrator.vector_tree(market, option, pricer)
        return {"price": price, "nodes": pricer_tree.nodes_priced, "build_s": t1 - t0, "price_s": t2 - t1,
                "tree": tree, "bs": Orchestrator.black_scholes(market, option, pricer)}

    @staticmethod
    def vector_tree(market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> TreeVectorized:
        """Arbre vectorisé valorisé, gardé pour les analyses (distribution, dessin de l'arbre, Delta / Gamma).

        C'est le même arbre qu'en moteur objets ; seule la représentation change. L'appelant le libère.
        """
        tree = TreeVectorized(market, option, PriceParameters(**{**vars(pricer), "engine": "vector"}), pre_steps)
        OptionPricerTree(tree).backward_pricing()
        return tree

    @staticmethod
    def price_many(market: Market, option: Option, pricer: PriceParameters, strikes, is_call,
                   is_american) -> np.ndarray:
        """Plusieurs calls / puts valorisés sur une seule grille vectorisée (une colonne par option)."""
        tree = TreeVectorized(market, option, PriceParameters(**{**vars(pricer), "engine": "vector"}))
        try:
            return OptionPricerTree(tree).price_options_vectorized(strikes, is_call, is_american)
        finally:
            tree.release()

    @staticmethod
    def black_scholes(market: Market, option: Option, pricer: PriceParameters) -> float | None:
        """Prix Black-Scholes, ou None si la formule ne s'applique pas (américaine ou dividende)."""
        if not Black_Scholes.is_available(market, option):
            return None
        return Black_Scholes(pricer, market, option).pricing_BS()

    @staticmethod
    def greeks_tree(market: Market, option: Option, pricer: PriceParameters, greeks=GREEKS,
                    spot_shift: float = 0.01, vol_shift: float = 0.01, rate_shift: float = 0.01,
                    theta_days: int = 1) -> dict:
        """Grecques de l'arbre par différences finies : GreeksTree rappelle Orchestrator.price sur des paramètres
        choqués (arbres vectorisés, plus rapides, même résultat)."""
        vector_pricer = PriceParameters(**{**vars(pricer), "engine": "vector"})
        gt = GreeksTree(vector_pricer, market, option, Orchestrator.price, spot_shift, vol_shift, rate_shift,
                        theta_days)
        return {g: float(getattr(gt, g.lower())()) for g in greeks}

    @staticmethod
    def greeks_bs(market: Market, option: Option, pricer: PriceParameters, greeks=GREEKS) -> dict | None:
        """Grecques Black-Scholes, ou None si la formule ne s'applique pas."""
        if not Black_Scholes.is_available(market, option):
            return None
        return GreeksBS(pricer, market, option).compute(greeks)

    DELTA_METHODS = ("Choc du spot", "Nœuds de l'arbre", "Nœuds autour de la racine")

    @staticmethod
    def delta_gamma(market: Market, option: Option, pricer: PriceParameters, method: str) -> tuple[float, float]:
        """Delta et Gamma par les trois méthodes du cours.

        * Choc du spot : GreeksTree (arbre revalorisé à S0 ± 1 %).
        * Nœuds de l'arbre : 3 nœuds du premier niveau qui en compte 3, sur un seul arbre.
        * Nœuds autour de la racine : arbre démarré 1 pas (trinomial) ou 2 pas (binomial) avant la date de pricing,
          qui a donc 3 nœuds à la date de pricing, S0 au milieu.
        Delta et Gamma sont lus sur 3 nœuds voisins (pas inégaux), Delta ramené en S0 par Taylor.
        """
        if method == "Choc du spot":
            g = Orchestrator.greeks_tree(market, option, pricer, ("Delta", "Gamma"))
            return g["Delta"], g["Gamma"]
        pre = (1 if pricer.tree_method.nb_nodes == 3 else 2) if method == "Nœuds autour de la racine" else 0
        tree = Orchestrator.vector_tree(market, option, pricer, pre)
        levels = tree.levels
        tree.release()
        level = levels[pre] if pre else next(lv for lv in levels if len(lv["S"]) >= 3)
        order = np.argsort(level["S"])
        S, V = level["S"][order], level["V"][order]
        j = int(np.clip(np.argmin(np.abs(S - market.s0)), 1, len(S) - 2))
        (s0, s1, s2), (v0, v1, v2) = S[j - 1:j + 2], V[j - 1:j + 2]
        gamma = 2.0 * ((v2 - v1) / (s2 - s1) - (v1 - v0) / (s1 - s0)) / (s2 - s0)
        return float((v2 - v0) / (s2 - s0) + gamma * (market.s0 - s1)), float(gamma)

    @staticmethod
    def benchmark(market: Market, option: Option, pricer: PriceParameters, method: str = "backward",
                  memory: bool = False, with_recursive: bool = False) -> dict:
        """Un pricing mesuré : temps de construction, de valorisation, nombre de nœuds (et pic mémoire Python)."""
        gc.collect()
        if memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        tree = Orchestrator.build(market, option, pricer)
        t1 = time.perf_counter()
        pricer_tree = OptionPricerTree(tree)
        price = pricer_tree.price(method)
        t2 = time.perf_counter()
        out = {"N": pricer.nb_steps, "Construction (s)": t1 - t0, "Valorisation (s)": t2 - t1,
               "Total (s)": t2 - t0, "Nœuds": pricer_tree.nodes_priced, "Prix": price}
        if memory:
            out["Pic mémoire (Mo)"] = tracemalloc.get_traced_memory()[1] / 2 ** 20
            tracemalloc.stop()
        if with_recursive and isinstance(tree, TreeNode):        # même arbre, revalorisé en récursif
            tree.reset_values()
            t3 = time.perf_counter()
            pricer_tree.recursive_pricing()
            out["Récursif (s)"] = time.perf_counter() - t3
        tree.release()
        gc.collect()
        return out
