from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from Market import Market
from Node import Node, Trunk

if TYPE_CHECKING:  # pragma: no cover
    from Tree import Tree

class LevelNode:
    def __init__(self, tree: Tree, time_index: int) -> None:
        step = tree._steps[time_index]
        m = tree.market
        self.tree, self.time_index = tree, time_index
        self.forward, self.alpha, self.dividend = step["F"], step["alpha"], step["D"]
        self.is_tri, self.flat, self.log_alpha = step["tri"], step["flat"], step["log_alpha"]
        self.p_min = tree.pricer.p_min if tree.pricer.pruning else 0.0
        self.er = math.exp(m.rate * tree.dt_)                                     # facteur forward d'un pas
        self.var_factor = math.exp(2.0 * m.rate * tree.dt_) * math.expm1(m.vol ** 2 * tree.dt_)
        b = tree.sqrt_alpha_bin
        self.p_up_bin, self.p_down_bin = 1.0 / (b + 1.0), b / (b + 1.0)           # probabilités binomiales
        self.next_repere: Node | None = None                                     # dernier enfant créé ou trouvé
    def generate_level(self, trunk_previous: Trunk) -> Trunk:
        """Branche le tronc, puis les parents au-dessus (.up), puis en dessous (.down) ; renvoie le nouveau tronc."""
        trunk_parent = trunk_previous.node_ref
        self.branch(trunk_parent, True)
        for direction in ("up", "down"):
            parent = getattr(trunk_parent, direction)
            while parent is not None:
                self.branch(parent, False)
                parent = getattr(parent, direction)
        # nouveau tronc : enfant le plus proche de la position 0 (égalité résolue vers le haut)
        node = self.next_repere
        while node.down is not None:
            node = node.down
        best = node
        while node is not None:
            if abs(node.position) < abs(best.position) or (abs(node.position) == abs(best.position) and node.position >= 0):
                best = node
            node = node.up
        return Trunk(node_ref=best, prec_trunk=trunk_previous)

    def branch(self, parent: Node, is_trunk: bool) -> None:
        s = parent.und_price
        expected = s * self.er
        if self.dividend > 0.0:                          # dividende plafonné à 99 % de S·e^(rΔt)
            expected -= min(self.dividend, 0.99 * expected)
        if expected <= 0.0:
            raise ValueError(f"Espérance négative au pas {self.time_index} : dividende trop élevé.")
        k = round(math.log(expected / self.forward) / self.log_alpha) if self.log_alpha else 0
        pruned = not is_trunk and parent.probas_cum < self.p_min
        f, a = self.forward, self.alpha                  # enfant en position j : prix f·a^j
        parent.next_up = parent.next_middle = parent.next_down = None
        parent.p_up = parent.p_middle = parent.p_down = 0.0
        if not self.is_tri:                                                       # pas binomial
            if pruned:
                if k > 0:
                    parent.p_down = 1.0
                    parent.next_down = self.find_or_create_child(f * a ** (k - 1), parent, k - 1, 1.0)
                else:
                    parent.p_up = 1.0
                    parent.next_up = self.find_or_create_child(f * a ** (k + 1), parent, k + 1, 1.0)
            elif self.flat:
                parent.p_middle = 1.0
                parent.next_middle = self.find_or_create_child(f * a ** k, parent, k, 1.0)
            else:
                parent.p_up, parent.p_down = self.p_up_bin, self.p_down_bin
                parent.next_up = self.find_or_create_child(f * a ** (k + 1), parent, k + 1, self.p_up_bin)
                parent.next_down = self.find_or_create_child(f * a ** (k - 1), parent, k - 1, self.p_down_bin)
            return
        if pruned or self.flat:                                                   # trinomial, un seul enfant
            parent.p_middle = 1.0
            parent.next_middle = self.find_or_create_child(f * a ** k, parent, k, 1.0)
            return
        m = f * a ** k                                                            # trinomial complet
        variance = s * s * self.var_factor
        x = expected / m - 1.0
        p_down = ((variance + expected ** 2) / m ** 2 - 1.0 - (a + 1.0) * x) / ((1.0 - a) * (a ** -2 - 1.0))
        p_up = (x - (1.0 / a - 1.0) * p_down) / (a - 1.0)
        probs = [max(p_up, 0.0), max(1.0 - p_up - p_down, 0.0), max(p_down, 0.0)]
        total = sum(probs)
        parent.p_up, parent.p_middle, parent.p_down = (p / total for p in probs) if total > 0.0 else (0.0, 1.0, 0.0)
        parent.next_down = self.find_or_create_child(f * a ** (k - 1), parent, k - 1, parent.p_down)
        parent.next_middle = self.find_or_create_child(f * a ** k, parent, k, parent.p_middle)
        parent.next_up = self.find_or_create_child(f * a ** (k + 1), parent, k + 1, parent.p_up)

    def find_or_create_child(self, price: float, parent: Node, position: int, transition_probability: float) -> Node:
        probability = parent.probas_cum * transition_probability
        current = self.next_repere
        if current is None:
            self.next_repere = Node(price, self.time_index, position, probability, self.tree)
            return self.next_repere
        while current.position < position and current.up is not None:
            current = current.up
        while current.position > position and current.down is not None:
            current = current.down
        if current.position == position:
            current.probas_cum += probability
            self.next_repere = current
            return current
        child = Node(price, self.time_index, position, probability, self.tree)
        if current.position < position:                  # insertion juste au-dessus de current
            child.down, child.up = current, current.up
            if current.up is not None:
                current.up.down = child
            current.up = child
        else:                                            # insertion juste en dessous de current
            child.up, child.down = current, current.down
            if current.down is not None:
                current.down.up = child
            current.down = child
        self.next_repere = child
        return child


class LevelVectorized:
    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def transitions(self, i: int, S: np.ndarray, P: np.ndarray, pos_prev: np.ndarray) -> dict:
        tree, m = self.tree, self.tree.market
        c = tree._steps[i]
        n = len(S)
        # moments du cours (Market) ; dividende plafonné à 99 % de S·e^(rΔt)
        expected = Market.esperance(S, (i - 1) * tree.dt_, m.rate, tree.dt_)
        if c["D"] > 0.0:
            expected = expected - np.minimum(c["D"], 0.99 * expected)
        variance = Market.variance(S, (i - 1) * tree.dt_, m.rate, m.vol, tree.dt_)
        if not np.all(expected > 0.0):
            raise ValueError(f"Espérance négative au pas {i} : dividende trop élevé.")
        k = (np.rint(np.log(expected / c["F"]) / c["log_alpha"]).astype(np.int64) if c["log_alpha"]
             else np.zeros(n, dtype=np.int64))
        trunk = int(np.argmin(2 * np.abs(pos_prev) + (pos_prev < 0)))        # tronc : jamais élagué
        pruned = (P < tree.pricer.p_min) if tree.pricer.pruning else np.zeros(n, dtype=bool)
        pruned[trunk] = False
        p_up, p_mid, p_down = np.zeros(n), np.zeros(n), np.zeros(n)
        if not c["tri"]:                                                     # pas binomial
            has_up, has_mid, has_down = np.ones(n, dtype=bool), np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
            if c["flat"]:
                has_up[:] = has_down[:] = False
                has_mid[:] = True
                p_mid[:] = 1.0
            else:
                b = tree.sqrt_alpha_bin
                p_up[:], p_down[:] = 1.0 / (b + 1.0), b / (b + 1.0)
            go_down, go_up = pruned & (k > 0), pruned & (k <= 0)             # élagué : un enfant, côté centre
            has_mid[pruned], p_mid[pruned] = False, 0.0
            has_up[go_down], p_up[go_down], has_down[go_down], p_down[go_down] = False, 0.0, True, 1.0
            has_down[go_up], p_down[go_up], has_up[go_up], p_up[go_up] = False, 0.0, True, 1.0
        else:                                                                # pas trinomial
            single = pruned | c["flat"]
            full = ~single
            has_up, has_mid, has_down = full.copy(), np.ones(n, dtype=bool), full.copy()
            p_mid[single] = 1.0
            if full.any():
                a = c["alpha"]
                inv = 1.0 / (c["F"] * np.power(a, k[full].astype(float)))
                e = expected[full]
                x = inv * e - 1.0
                pd_ = (inv * inv * (variance[full] + e * e) - 1.0 - (a + 1.0) * x) / ((1.0 - a) * (a ** -2 - 1.0))
                pu_ = (x - (a ** -1 - 1.0) * pd_) / (a - 1.0)
                pu_, pm_, pd_ = np.maximum(pu_, 0.0), np.maximum(1.0 - pu_ - pd_, 0.0), np.maximum(pd_, 0.0)
                tot = pu_ + pm_ + pd_
                bad = ~(tot > 0.0)
                tot[bad] = 1.0
                pu_, pm_, pd_ = pu_ / tot, pm_ / tot, pd_ / tot
                pu_[bad], pm_[bad], pd_[bad] = 0.0, 1.0, 0.0
                p_up[full], p_mid[full], p_down[full] = pu_, pm_, pd_
        return {"k": k, "has": (has_up, has_mid, has_down), "p": (p_up, p_mid, p_down), "E": expected}

    @staticmethod
    def child_index(child_pos: np.ndarray, k: np.ndarray, has: tuple) -> tuple:
        """Indices des enfants (k + 1, k, k − 1) dans le niveau suivant ; 0 si la branche n'existe pas."""
        lo = min(int(k.min()) - 1, int(child_pos[0]))
        lookup = np.zeros(max(int(k.max()) + 1, int(child_pos[-1])) - lo + 1, dtype=np.int64)
        lookup[child_pos - lo] = np.arange(len(child_pos))
        out = []
        for shift, h in zip((1, 0, -1), has):
            idx = lookup[k - lo + shift]
            idx[~h] = 0
            out.append(idx)
        return tuple(out)