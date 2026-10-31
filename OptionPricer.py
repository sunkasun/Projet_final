from __future__ import annotations
import sys
import threading
import numpy as np
from Level import LevelVectorized
from Tree import Tree, TreeNode, TreeVectorized

class OptionPricerTree:
    """Valorise l'option portée par un arbre (TreeNode ou TreeVectorized)."""

    def __init__(self, tree: Tree):
        self.tree = tree
        self.nodes_priced = 0

    def price(self, method: str = "backward") -> float:
        return self.recursive_pricing() if method == "recursive" else self.backward_pricing()

    def backward_pricing(self) -> float:
        """Induction rétrograde : de la maturité vers la racine.

        TreeNode : pour chaque nœud, valeur actualisée des enfants et exercice anticipé si américaine (formule de
        Node.price_with_child). TreeVectorized : la même formule sur un niveau entier ; remplit V et ex.
        """
        tree = self.tree
        if isinstance(tree, TreeNode):
            # Même formule que Node.price_with_child, écrite directement dans la boucle : les constantes (K, call /
            # put, américaine, actualisation) sont lues une seule fois au lieu d'un appel de méthode par nœud,
            # ce qui divise le temps par 2,5 (≈ 0,7 s pour 3,5 millions de nœuds).
            option, disc, last_step = tree.option, tree.discount_value, tree.n_steps
            K, is_call, american = option.K, option.call_put == "call", option.is_american()
            count = 0
            trunk = tree.last_trunk
            while trunk is not None:                     # niveaux de la maturité vers la racine
                node = trunk.node_ref
                while node.down is not None:
                    node = node.down
                maturity = node.time == last_step
                while node is not None:                  # chaque niveau du bas vers le haut
                    payoff = node.und_price - K if is_call else K - node.und_price
                    if payoff < 0.0:
                        payoff = 0.0
                    if maturity:
                        value, exercised = payoff, american and payoff > 0.0
                    else:                                # espérance actualisée des enfants
                        value = 0.0
                        child = node.next_up
                        if child is not None:
                            value += child.option_value * node.p_up
                        child = node.next_middle
                        if child is not None:
                            value += child.option_value * node.p_middle
                        child = node.next_down
                        if child is not None:
                            value += child.option_value * node.p_down
                        value *= disc
                        exercised = american and payoff > value
                        if exercised:                    # exercice anticipé
                            value = payoff
                    node.option_value = value
                    node.is_exerced = exercised
                    count += 1
                    node = node.up
                trunk = trunk.prec
            self.nodes_priced = count
            return float(tree.root.option_value)

        option, levels = tree.option, tree.levels
        american = option.is_american()
        last = levels[-1]
        last["V"] = option.payoff_array(last["S"])
        last["ex"] = (last["V"] > 0.0) if american else np.zeros(len(last["S"]), dtype=bool)
        for i in range(tree.n_steps, 0, -1):
            lv, child = levels[i - 1], levels[i]
            tr = tree.transitions(i)
            iu, im, idn = LevelVectorized.child_index(child["pos"], tr["k"], tr["has"])
            pu, pm, pd_ = tr["p"]
            value = (child["V"][iu] * pu + child["V"][im] * pm + child["V"][idn] * pd_) * tree.discount_value
            ex = np.zeros(len(value), dtype=bool)
            if american:
                payoff = option.payoff_array(lv["S"])
                ex = payoff > value
                value = np.where(ex, payoff, value)
            lv["V"], lv["ex"] = value, ex
        self.nodes_priced = int(sum(len(lv["S"]) for lv in levels))
        return float(levels[0]["V"][0])

    def recursive_pricing(self) -> float:
        """Récursion depuis la racine : Node.price_with_child_recursive valorise d'abord les enfants, puis le nœud
        avec la même formule que le backward.

        N appels imbriqués : on relève la limite de récursion de Python (1 000 par défaut) et on lance le calcul
        dans un thread à grande pile (256 Mo), sinon la pile déborde dès quelques milliers de pas.
        """
        tree = self.tree
        if not isinstance(tree, TreeNode):
            raise ValueError("Le pricing récursif parcourt des objets Node : utilisez le moteur « objets ».")
        result = {}
        old_limit, old_stack = sys.getrecursionlimit(), threading.stack_size()
        sys.setrecursionlimit(max(old_limit, 4 * tree.n_steps + 1000))
        threading.stack_size(256 * 2 ** 20)
        try:
            worker = threading.Thread(target=lambda: result.update(value=tree.root.price_with_child_recursive()))
            worker.start()
            worker.join()
        finally:
            threading.stack_size(old_stack)
            sys.setrecursionlimit(old_limit)
        if "value" not in result:
            raise RuntimeError("Le pricing récursif a échoué (détail de l'erreur dans la console).")
        self.nodes_priced = 0
        trunk = tree.last_trunk
        while trunk is not None:
            node = trunk.node_ref
            while node.down is not None:
                node = node.down
            while node is not None:
                self.nodes_priced += 1
                node = node.up
            trunk = trunk.prec
        return float(result["value"])

    def price_options_vectorized(self, strikes, is_call, is_american) -> np.ndarray:
        """Plusieurs calls / puts valorisés d'un coup sur la même grille (une colonne de V par option)."""
        tree = self.tree
        if not isinstance(tree, TreeVectorized):
            raise ValueError("Le pricing de plusieurs options demande un arbre vectorisé.")
        levels = tree.levels
        K = np.asarray(strikes, float)[None, :]
        sign = np.where(np.asarray(is_call, bool), 1.0, -1.0)[None, :]
        am = np.asarray(is_american, bool)[None, :]
        V = np.maximum(sign * (levels[-1]["S"][:, None] - K), 0.0)          # payoffs à maturité
        for i in range(tree.n_steps, 0, -1):
            tr = tree.transitions(i)
            iu, im, idn = LevelVectorized.child_index(levels[i]["pos"], tr["k"], tr["has"])
            pu, pm, pd_ = (p[:, None] for p in tr["p"])
            V = (V[iu] * pu + V[im] * pm + V[idn] * pd_) * tree.discount_value
            if am.any():
                exercise = np.maximum(sign * (levels[i - 1]["S"][:, None] - K), 0.0)
                V = np.where(am & (exercise > V), exercise, V)
        return V[0]
