from __future__ import annotations
import gc
import math
from contextlib import contextmanager
import numpy as np
from Level import LevelNode, LevelVectorized
from Market import Market
from Node import Node, Trunk
from Option import Option
from PriceParameters import PriceParameters

class Tree:
    def __init__(self, market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> None:
        self.market, self.option, self.pricer = market, option, pricer
        T = PriceParameters.year_fraction(pricer.pricing_date, option.maturity)
        self.dt_ = T / pricer.nb_steps
        self.n_steps = pricer.nb_steps + pre_steps
        self.nb_nodes = pricer.tree_method.nb_nodes
        self.discount_value = math.exp(-market.rate * self.dt_)
        self.spot = market.s0 * math.exp(-market.rate * pre_steps * self.dt_)     # spot de la racine

        # pas qui contient le dividende (détaché à la fin de ce pas) ; 0 = pas de dividende dans ]pricing, T]
        self.div_step = 0
        if market.dividend > 0.0 and market.ex_div_date is not None:
            t_div = PriceParameters.year_fraction(pricer.pricing_date, market.ex_div_date)
            if 0.0 < t_div <= T + 1e-12:
                self.div_step = min(max(round(t_div / self.dt_), 1), pricer.nb_steps) + pre_steps

        # facteurs de grille : tous calculés, le bon est choisi à chaque pas
        x = math.expm1(market.vol ** 2 * self.dt_)
        self.sqrt_alpha_bin = 1.0 + x / 2.0 + math.sqrt(x * (1.0 + x / 4.0))     # pas binomial
        self.alpha_bin = self.sqrt_alpha_bin ** 2                                  # pas du dividende en binomial
        self.alpha_tri = math.exp(market.vol * math.sqrt(3.0 * self.dt_))         # pas trinomial

        # pour chaque pas : forward du tronc F (formule du cours via Market), grille et type de transition
        self._steps = [None]
        forward = self.spot
        for i in range(1, self.n_steps + 1):
            dividend = market.dividend if i == self.div_step else 0.0
            forward = Market.esperance(forward, (i - 1) * self.dt_, market.rate, self.dt_, dividend, i * self.dt_)
            if forward <= 0.0:
                raise ValueError(f"Forward négatif au pas {i} : le dividende est trop élevé par rapport au spot.")
            if self.nb_nodes == 3:
                alpha, is_tri = self.alpha_tri, True
            elif dividend > 0.0:                       # binomial : seul le pas du dividende devient trinomial
                alpha, is_tri = self.alpha_bin, True
            else:
                alpha, is_tri = self.sqrt_alpha_bin, False
            self._steps.append({"F": forward, "alpha": alpha, "tri": is_tri, "D": dividend,
                                "log_alpha": math.log(alpha) if alpha > 1.0 + 1e-12 else 0.0,
                                "flat": alpha <= 1.0 + 1e-12})

    @staticmethod
    @contextmanager
    def gc_paused():
        """Suspend le garbage collector pendant une construction (aucun nœud n'y est libérable)."""
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            yield
        finally:
            if was_enabled:
                gc.enable()

class TreeNode(Tree):
    """Arbre d'objets Node chaînés, construit niveau par niveau à partir du tronc (jamais de tableaux numpy)."""
    def __init__(self, market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> None:
        super().__init__(market, option, pricer, pre_steps)          # données communes
        with Tree.gc_paused():
            self.root: Node | None = Node(self.spot, 0, 0, 1.0, self)
            self.last_trunk: Trunk | None = Trunk(node_ref=self.root, prec_trunk=None)
            for i in range(1, self.n_steps + 1):
                self.last_trunk = LevelNode(self, i).generate_level(self.last_trunk)

    def reset_values(self) -> None:
        """Efface les valeurs d'option de tous les nœuds (pour revaloriser sur la même grille)."""
        trunk = self.last_trunk
        while trunk is not None:
            node = trunk.node_ref
            while node.down is not None:
                node = node.down
            while node is not None:
                node.option_value, node.is_exerced = None, False
                node = node.up
            trunk = trunk.prec

    def release(self) -> None:
        """Casse les liens entre nœuds (cycles up/down, référence vers l'arbre) : la mémoire est rendue tout de suite."""
        trunk = self.last_trunk
        while trunk is not None:
            node = trunk.node_ref
            while node is not None and node.down is not None:
                node = node.down
            while node is not None:
                nxt = node.up
                node.up = node.down = node.next_up = node.next_middle = node.next_down = node.tree = None
                node = nxt
            trunk.node_ref = None
            trunk = trunk.prec
        self.root = self.last_trunk = None


class TreeVectorized(Tree):
    """Arbre dont chaque niveau est un dictionnaire de tableaux {"S", "P", "pos"} (+ "V", "ex" après valorisation)."""

    def __init__(self, market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> None:
        super().__init__(market, option, pricer, pre_steps)          # données communes
        self.level_builder = LevelVectorized(self)
        levels = [{"S": np.array([self.spot]), "P": np.array([1.0]), "pos": np.array([0], dtype=np.int64)}]
        with Tree.gc_paused():
            for i in range(1, self.n_steps + 1):
                prev = levels[-1]
                tr = self.level_builder.transitions(i, prev["S"], prev["P"], prev["pos"])
                k, has, prob = tr["k"], tr["has"], tr["p"]
                # positions des enfants qui existent, puis probabilités cumulées (somme sur les parents)
                lo = int(k.min()) - 1
                present = np.zeros(int(k.max()) + 2 - lo, dtype=bool)
                for shift, h in zip((1, 0, -1), has):
                    present[k[h] + (shift - lo)] = True
                child_pos = np.flatnonzero(present) + lo
                P = np.zeros(len(child_pos))
                for idx, h, p in zip(LevelVectorized.child_index(child_pos, k, has), has, prob):
                    P += np.bincount(idx, weights=np.where(h, prev["P"] * p, 0.0), minlength=len(child_pos))
                c = self._steps[i]
                levels.append({"S": c["F"] * np.power(c["alpha"], child_pos.astype(float)), "P": P, "pos": child_pos})
        self.levels: list[dict] | None = levels

    def transitions(self, i: int) -> dict:
        """Transitions du niveau i − 1 vers le niveau i de cet arbre (valorisation, visualisation)."""
        lv = self.levels[i - 1]
        return self.level_builder.transitions(i, lv["S"], lv["P"], lv["pos"])

    def reset_values(self) -> None:
        for lv in self.levels:
            lv.pop("V", None)
            lv.pop("ex", None)

    def release(self) -> None:
        self.levels = None
