from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from Level import LevelVectorized
from Tree import TreeVectorized

class TreeVisualizer:
    Y_OPTIONS = {"Prix du sous-jacent": "S", "Prix de l'option": "V"}
    COLOR_OPTIONS = {"Probabilité cumulée": "mass", "Marge d'exercice": "margin",
                     "Écart d'espérance": "dmean", "Écart de variance": "dvar"}
    MAX_EDGES_NODES = 15_000      # au-delà, les branches ne sont pas tracées (navigateur)
    MAX_NODES = 300_000           # au-delà, un niveau sur k est affiché

    @staticmethod
    def build(tree: TreeVectorized, y_mode: str = "S", color_mode: str = "mass", t_range: tuple | None = None,
              log_y: bool = False, height: int = 620) -> dict:
        """Figure de l'arbre et phrase de légende.

        Couleurs : probabilité cumulée (log10), marge d'exercice (payoff − continuation), écart entre la moyenne
        (ou la variance) des enfants et la valeur théorique du modèle (vérification du moment matching).
        Pour une option américaine : nœuds cerclés = exercice optimal ; ligne noire = frontière d'exercice (à
        chaque pas, prix critique : plus bas nœud exercé pour un call, plus haut pour un put ; nœuds élagués ignorés).
        """
        levels, n = tree.levels, len(tree.levels) - 1
        t0, t1 = t_range if t_range is not None else (0, n)
        size = sum(len(levels[t]["S"]) for t in range(t0, t1 + 1))
        stride = max(1, int(np.ceil(size / TreeVisualizer.MAX_NODES)))
        times = list(range(t0, t1 + 1, stride))
        draw_edges = stride == 1 and size <= TreeVisualizer.MAX_EDGES_NODES
        american, is_call = tree.option.is_american(), tree.option.call_put == "call"
        p_min = tree.pricer.p_min if tree.pricer.pruning else 0.0
        x, y, color, hover, ex_x, ex_y, front_x, front_y, edge_x, edge_y = ([] for _ in range(10))
        for t in times:
            lv = levels[t]
            S, V, P = lv["S"], lv["V"], lv["P"]
            yv = S if y_mode == "S" else V
            last = t == n
            if not last and (color_mode != "mass" or draw_edges):     # transitions vers le niveau t + 1
                tr = tree.transitions(t + 1)
                child = levels[t + 1]
                iu, im, idn = LevelVectorized.child_index(child["pos"], tr["k"], tr["has"])
                pu, pm, pd_ = tr["p"]
            if color_mode == "mass":
                c = np.log10(np.maximum(P, 1e-300))
            elif last:
                c = np.zeros(len(S))
            elif color_mode == "margin":
                cont = (child["V"][iu] * pu + child["V"][im] * pm + child["V"][idn] * pd_) * tree.discount_value
                c = tree.option.payoff_array(S) - cont
            else:
                mean = child["S"][iu] * pu + child["S"][im] * pm + child["S"][idn] * pd_
                if color_mode == "dmean":
                    c = mean - tr["E"]
                else:
                    second = child["S"][iu] ** 2 * pu + child["S"][im] ** 2 * pm + child["S"][idn] ** 2 * pd_
                    theory = (S ** 2 * np.exp(2 * tree.market.rate * tree.dt_)
                              * np.expm1(tree.market.vol ** 2 * tree.dt_))
                    c = (second - mean ** 2) - theory
            x.append(np.full(len(S), t))
            y.append(yv)
            color.append(c)
            hover.append([f"Pas {t}<br>S = {s:.4f}<br>V = {v:.4f}<br>P = {p:.3e}" for s, v, p in zip(S, V, P)])
            if draw_edges and not last:                                    # segments parent → enfant
                yc = child["S"] if y_mode == "S" else child["V"]
                for h, ix in zip(tr["has"], (iu, im, idn)):
                    sel = np.flatnonzero(h)
                    seg = np.full((len(sel), 3), np.nan)
                    seg[:, 0], seg[:, 1] = t, t + 1
                    edge_x.append(seg.ravel())
                    seg_y = np.full((len(sel), 3), np.nan)
                    seg_y[:, 0], seg_y[:, 1] = yv[sel], yc[ix[sel]]
                    edge_y.append(seg_y.ravel())
            if american:                                                   # exercice optimal et frontière
                ex = lv["ex"] & (P >= p_min)
                ex_x.append(np.full(ex.sum(), t))
                ex_y.append(yv[ex])
                if ex.any():
                    j = np.flatnonzero(ex)[np.argmin(S[ex]) if is_call else np.argmax(S[ex])]
                    if front_x and front_x[-1] is not None and front_x[-1] != t - stride:
                        front_x.append(None)                               # pas sans exercice : ligne coupée
                        front_y.append(None)
                    front_x.append(t)
                    front_y.append(float(yv[j]))

        fig = go.Figure()
        if edge_x:
            fig.add_trace(go.Scattergl(x=np.concatenate(edge_x), y=np.concatenate(edge_y), mode="lines",
                                       line=dict(color="rgba(110,122,140,0.35)", width=1), hoverinfo="skip",
                                       name="Branches"))
        fig.add_trace(go.Scattergl(
            x=np.concatenate(x), y=np.concatenate(y), mode="markers", name="Nœuds",
            text=[h for level in hover for h in level], hovertemplate="%{text}<extra></extra>",
            marker=dict(size=6 if size < 2_000 else 4 if size < 50_000 else 2, color=np.concatenate(color),
                        colorscale="Turbo", showscale=True,
                        colorbar=dict(title="log10(P)" if color_mode == "mass" else "écart", thickness=12))))
        if american and ex_x:
            fig.add_trace(go.Scattergl(x=np.concatenate(ex_x), y=np.concatenate(ex_y), mode="markers",
                                       name="Exercice optimal", hoverinfo="skip",
                                       marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color="black", width=1))))
            fig.add_trace(go.Scatter(x=front_x, y=front_y, mode="lines+markers", name="Frontière d'exercice",
                                     line=dict(color="black", width=2), hoverinfo="skip"))
        fig.update_layout(height=height, template="plotly_white", margin=dict(l=25, r=25, t=40, b=45),
                          xaxis_title="Pas de temps",
                          yaxis_title=next(k for k, v in TreeVisualizer.Y_OPTIONS.items() if v == y_mode),
                          hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        fig.update_yaxes(type="log" if log_y else "linear")
        caption = (f"{size:,} nœuds dans la fenêtre (pas {t0} → {t1})"
                   + (f", un niveau sur {stride} affiché" if stride > 1 else "")
                   + ("" if draw_edges else " ; branches masquées au-delà de "
                      f"{TreeVisualizer.MAX_EDGES_NODES:,} nœuds (réduisez la fenêtre pour les voir)") + ".")
        return {"fig": fig, "caption": caption}
