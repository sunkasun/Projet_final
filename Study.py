from __future__ import annotations
import datetime as dt
import gc
import math
import re
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from Black_Scholes import Black_Scholes
from Market import Market
from Option import Option
from Orchestrator import Orchestrator
from PriceParameters import PriceParameters, TreeMethod
from Sensibilities import GREEKS
from TreeVisualizer import TreeVisualizer


# ====================================================================================
# Mise en forme commune aux trois onglets
# ====================================================================================
class Affichage:
    """Mise en forme commune : graphiques, tableaux, blocs, colonnes, messages d'erreur, résumé des paramètres."""

    NEW_WIDTH_API = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2]) >= (1, 50)
    COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    METHOD_STYLE = {TreeMethod.BINOMIAL: (COLORS[0], "solid"), TreeMethod.TRINOMIAL: (COLORS[1], "dash")}

    @staticmethod
    def show_chart(fig: go.Figure, key: str) -> None:
        width = {"width": "stretch"} if Affichage.NEW_WIDTH_API else {"use_container_width": True}
        st.plotly_chart(fig, key=key, config={"displaylogo": False, "scrollZoom": True}, **width)

    @staticmethod
    def show_df(df: pd.DataFrame, **kw) -> None:
        width = {"width": "stretch"} if Affichage.NEW_WIDTH_API else {"use_container_width": True}
        st.dataframe(df, hide_index=True, **width, **kw)

    @staticmethod
    def new_fig(title: str, x_title: str, y_title: str, height: int = 430) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)), template="plotly_white",
                          height=height, margin=dict(l=70, r=30, t=60, b=40), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
        fig.update_xaxes(title_text=x_title, showline=True, linecolor="#bbb")
        fig.update_yaxes(title_text=y_title, showline=True, linecolor="#bbb", tickformat="~g")
        return fig

    @staticmethod
    def lines(series: dict, title: str, y_title: str, key: str, log_x: bool = False, log_y: bool = False) -> None:
        """Courbes « N → valeur » : series = {libellé: (DataFrame, colonne, style de trait)}."""
        fig = Affichage.new_fig(title, "Nombre de pas N", y_title)
        for i, (label, (frame, col, dash)) in enumerate(series.items()):
            if not frame.empty:
                fig.add_trace(go.Scatter(x=frame["N"], y=frame[col], mode="lines+markers", name=label,
                                         line=dict(color=Affichage.COLORS[i % 7], dash=dash, width=2)))
        fig.update_xaxes(type="log" if log_x else "linear")
        fig.update_yaxes(type="log" if log_y else "linear")
        Affichage.show_chart(fig, key)

    @staticmethod
    def error_heatmap(df: pd.DataFrame, value: str, title: str, key: str) -> None:
        """Heatmap p_min x N d'une erreur, couleur en log10."""
        pivot = df.pivot(index="p_min", columns="N", values=value).sort_index(ascending=False)
        values = pivot.to_numpy(dtype=float)
        fig = Affichage.new_fig(title, "Nombre de pas N", "Seuil de pruning p_min")
        fig.add_trace(go.Heatmap(x=[str(c) for c in pivot.columns], y=[f"{p:.0e}" for p in pivot.index],
                                 z=np.log10(np.maximum(values, 1e-16)), colorscale="Viridis",
                                 text=[[f"{v:.1e}" for v in row] for row in values], texttemplate="%{text}",
                                 colorbar=dict(title="log10(erreur)", thickness=14)))
        Affichage.show_chart(fig, key)

    @staticmethod
    @contextmanager
    def section(title: str, caption: str | None = None):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if caption:
                st.caption(caption)
            yield

    @staticmethod
    @contextmanager
    def safe(label: str):
        """Affiche une erreur lisible au lieu d'interrompre toute la page."""
        try:
            yield
        except Exception as exc:          # noqa: BLE001 - message utilisateur
            st.error(f"{label} : {exc}")

    @staticmethod
    def params_expander(market: Market, option: Option, pricer: PriceParameters, extra: dict | None = None) -> None:
        """Liste déroulante des paramètres utilisés dans l'analyse."""
        day = lambda d: d.strftime("%d/%m/%Y") if d is not None else "—"
        T = PriceParameters.year_fraction(pricer.pricing_date, option.maturity)
        rows = {"Spot S0": f"{market.s0:,.2f}", "Taux sans risque r": f"{market.rate:.2%}",
                "Volatilité": f"{market.vol:.2%}", "Dividende": f"{market.dividend:,.2f}",
                "Date ex-dividende": day(market.ex_div_date) if market.dividend > 0 else "—",
                "Strike K": f"{option.K:,.2f}",
                "Option": f"{option.call_put.capitalize()} {'américain' if option.is_american() else 'européen'}",
                "Date de pricing": day(pricer.pricing_date), "Maturité": f"{day(option.maturity)} (T = {T:.4f} an)",
                "Méthode d'arbre": pricer.tree_method.label, "Nombre de pas N": f"{pricer.nb_steps:,}",
                "Moteur": "Vectorisé (numpy)" if pricer.engine == "vector" else f"Objets Node ({pricer.pricing_method})",
                "Pruning": f"Activé (p_min = {pricer.p_min:.0e})" if pricer.pruning else "Désactivé", **(extra or {})}
        with st.expander("Paramètres utilisés dans l'analyse", expanded=False):
            Affichage.show_df(pd.DataFrame({"Paramètre": list(rows), "Valeur": [str(v) for v in rows.values()]}))


# ====================================================================================
# Onglet 1 : pricer et analyse d'un arbre
# ====================================================================================
class EtudePricing:
    """Résultats, arbre, distribution, grecques."""

    @classmethod
    def render(cls, market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        Affichage.params_expander(market, option, pricer, {"Temps de construction (s)": f"{state['build_s']:.3f}","Temps de valorisation (s)": f"{state['price_s']:.3f}"})
        with Affichage.safe("Résultat du pricing"):
            cls.resultats(market, option, pricer, state)
        with Affichage.safe("Grecques"):
            cls.grecques(market, option, pricer, state)
        with Affichage.safe("Arbre"):
            cls.arbre(pricer, state)
        with Affichage.safe("Distribution"):
            cls.distribution(market, option, pricer, state)

    @staticmethod
    def resultats(market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        """Cartes : prix, valeur intrinsèque, valeur temps, Black-Scholes, taille, probabilités, temps."""
        price, bs = state["price"], state["bs"]
        intrinsic = option.payoff(market.s0)
        last = state["tree"].levels[-1]
        S, P = last["S"], last["P"]
        tol = 1e-9 * option.K
        label = f"{option.call_put.capitalize()} {'américain' if option.is_american() else 'européen'}"
        with Affichage.section(f"Résultat du pricing - {label}, {pricer.tree_method.label}, {pricer.nb_steps:,} pas"):
            c = st.columns(5)
            c[0].metric("Prix de l'option", f"{price:,.4f}")
            c[1].metric("Valeur intrinsèque", f"{intrinsic:,.4f}")
            c[2].metric("Valeur temps", f"{price - intrinsic:,.4f}")
            c[3].metric("Black-Scholes", "N/A" if bs is None else f"{bs:,.4f}",
                        delta=None if bs is None else f"arbre - BS : {price - bs:+.5f}", delta_color="off",
                        help="Disponible seulement pour une option européenne sans dividende.")
            c[4].metric("Taille de l'arbre", f"{state['nodes']:,} nœuds")
            c = st.columns(5)
            c[0].metric("P(S_T > K)", f"{P[S > option.K + tol].sum():.2%}")
            c[1].metric("P(S_T < K)", f"{P[S < option.K - tol].sum():.2%}")
            c[2].metric("P(S_T = K)", f"{P[np.abs(S - option.K) <= tol].sum():.2%}")
            c[3].metric("Temps de calcul", f"{state['build_s'] + state['price_s']:.2f} s",
                        help=f"Construction {state['build_s']:.2f} s + valorisation {state['price_s']:.2f} s.")

    @staticmethod
    def arbre(pricer: PriceParameters, state: dict) -> None:
        """Arbre de valorisation : ordonnée, couleur des nœuds, fenêtre de pas, frontière d'exercice."""
        n = pricer.nb_steps
        with Affichage.section(f"Arbre de valorisation"):
            c_y, c_color, c_log = st.columns([1, 1, 0.6], vertical_alignment="bottom")
            y_label = c_y.selectbox("Ordonnée", list(TreeVisualizer.Y_OPTIONS), key="tree_y")
            color_label = c_color.selectbox("Couleur des nœuds", list(TreeVisualizer.COLOR_OPTIONS), key="tree_color")
            log_y = c_log.checkbox("Échelle log", key="tree_log")
            t_range = st.slider("Fenêtre de pas affichée", 0, n, (0, min(n, 60)), key="tree_window") if n > 1 else (0, n)
            cache_key = (id(state["tree"]), y_label, color_label, tuple(t_range), log_y)
            if st.session_state.get("tree_view_key") != cache_key:
                st.session_state["tree_view"] = TreeVisualizer.build(
                    state["tree"], TreeVisualizer.Y_OPTIONS[y_label],
                    TreeVisualizer.COLOR_OPTIONS[color_label], tuple(t_range), log_y)
                st.session_state["tree_view_key"] = cache_key
            view = st.session_state["tree_view"]
            Affichage.show_chart(view["fig"], "tree_plot")
            st.caption(view["caption"])

    @staticmethod
    def distribution(market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        """Histogramme et fonction de répartition du sous-jacent à un pas choisi (axe cadré sur 99,9 % de la masse)."""
        levels = state["tree"].levels
        n = len(levels) - 1
        with Affichage.section(f"Distribution du sous-jacent",
                           "Loi risque-neutre du sous-jacent lue sur l'arbre au pas choisi."):
            t = st.slider("Pas de temps t étudié", 0, n, n, key="dist_t")
            cache_key = (id(state["tree"]), t)
            if st.session_state.get("dist_key") != cache_key:
                order = np.argsort(levels[t]["S"])
                S, P = levels[t]["S"][order], levels[t]["P"][order] / levels[t]["P"].sum()
                cdf = np.cumsum(P)
                mean = float((S * P).sum())
                lo = S[min(int(np.searchsorted(cdf, 5e-4)), len(S) - 1)]
                hi = S[min(int(np.searchsorted(cdf, 1 - 5e-4)), len(S) - 1)]
                lo, hi = (lo - 1.0, hi + 1.0) if hi <= lo else (lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
                inside = (S >= lo) & (S <= hi)
                edges = np.linspace(lo, hi, 61)
                mass = np.bincount(np.clip(np.searchsorted(edges, S[inside], side="right") - 1, 0, 59),
                                   weights=P[inside], minlength=60)
                st.session_state["dist"] = (S, P, cdf, mean, lo, hi, edges, mass)
                st.session_state["dist_key"] = cache_key
            S, P, cdf, mean, lo, hi, edges, mass = st.session_state["dist"]
            st.caption(f"Espérance de l'arbre {mean:,.4f} · écart-type {math.sqrt(max((P * (S - mean) ** 2).sum(), 0)):,.4f}"
                       f" · {len(S):,} nœuds")
            left, right = st.columns(2)
            with left:
                fig = Affichage.new_fig(f"Distribution au pas {t}", "Sous-jacent S", "Probabilité par intervalle")
                fig.add_trace(go.Bar(x=0.5 * (edges[:-1] + edges[1:]), y=mass, width=0.92 * (edges[1] - edges[0]),
                                     marker_color=Affichage.COLORS[0]))
                fig.add_vline(x=option.K, line_dash="dash", line_color="grey", annotation_text="K")
                fig.update_layout(hovermode="closest", showlegend=False)
                fig.update_xaxes(range=[lo, hi])
                Affichage.show_chart(fig, "dist_plot")
            with right:
                fig = Affichage.new_fig(f"Fonction de répartition au pas {t}", "Sous-jacent S", "P(S ≤ x)")
                fig.add_trace(go.Scatter(x=S, y=cdf, mode="lines", line_shape="hv",
                                         line=dict(color=Affichage.COLORS[1], width=2)))
                fig.add_vline(x=option.K, line_dash="dash", line_color="grey", annotation_text="K")
                fig.update_layout(hovermode="closest", showlegend=False)
                fig.update_xaxes(range=[lo, hi])
                fig.update_yaxes(range=[0, 1.03], tickformat=".0%")
                Affichage.show_chart(fig, "cdf_plot")

    @staticmethod
    def grecques(market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        """Grecques de l'arbre (différences finies, GreeksTree) et Black-Scholes ; Delta / Gamma par 3 méthodes."""
        with Affichage.section("Grecques",
                               "GreeksTree (méthode de l'énoncé) : l'arbre complet est revalorisé avec un paramètre "
                               "choqué (S0 ± 1 %, sigma ± 1 %, r ± 1 %, maturité - 1 jour). Delta et Gamma peuvent aussi "
                               "être lus sur les nœuds (autres méthodes du cours)."):
            c_sel, c_meth = st.columns([3, 2], vertical_alignment="bottom")
            selected = c_sel.multiselect("Grecques", GREEKS, default=["Delta", "Gamma", "Theta"], key="greeks_sel")
            dg_method = c_meth.selectbox("Méthode pour Delta / Gamma", Orchestrator.DELTA_METHODS, key="greeks_dg")
            if not selected:
                return
            store = st.session_state.setdefault("res_greeks", {})
            missing = [g for g in selected if g not in store]
            if missing:
                with st.spinner("Calcul des grecques…"):
                    store.update(Orchestrator.greeks_tree(market, option, pricer, missing))
            values = {g: store[g] for g in selected}
            if dg_method != Orchestrator.DELTA_METHODS[0] and {"Delta", "Gamma"} & set(selected):
                dg_store = st.session_state.setdefault("res_dg_values", {})
                if dg_method not in dg_store:
                    with st.spinner(f"Delta et Gamma via « {dg_method} »…"):
                        dg_store[dg_method] = Orchestrator.delta_gamma(market, option, pricer, dg_method)
                values["Delta"], values["Gamma"] = dg_store[dg_method]
            bs = Orchestrator.greeks_bs(market, option, pricer, selected) or {}
            df = pd.DataFrame([{"Grecque": g, "Arbre": values[g], "Black-Scholes": bs.get(g, np.nan),
                                "Écart (arbre - BS)": values[g] - bs.get(g, np.nan)} for g in selected])
            Affichage.show_df(df if bs else df[["Grecque", "Arbre"]],
                              column_config={c: st.column_config.NumberColumn(format="%.6f") for c in df.columns[1:]})
            st.caption("Theta par jour de bourse (/252) ; Vega, Rho, Vanna pour 1 % ; Vomma pour 1 %². Gamma par choc "
                       "du spot : bruité à faible N (la grille se déplace avec S0) ; les méthodes « nœuds » le donnent "
                       "avec une grille fixe.")


# ====================================================================================
# Onglet 2 : analyse comparative
# ====================================================================================
class EtudeComparative:
    """Convergence (prix, grecques), prix selon le strike, 4 types d'options, grecques selon un paramètre."""

    METHODS = (TreeMethod.BINOMIAL, TreeMethod.TRINOMIAL)
    CONV_VIEWS = ("Erreur (arbre - référence)", "Erreur x N")

    @classmethod
    def render(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        Affichage.params_expander(market, option, pricer)
        if st.button("Lancer l'analyse comparative", type="primary", key="run_cmp"):
            st.session_state["cmp_run"] = key
        if st.session_state.get("cmp_run") != key:
            st.info("Cliquez sur « Lancer l'analyse comparative » pour générer les graphiques (résultats gardés en mémoire ensuite).")
            return
        with Affichage.safe("Convergence du prix"):
            cls.convergence_prix(market, option, pricer, key)
        with Affichage.safe("Convergence des grecques"):
            cls.convergence_grecques(market, option, pricer, key)
        with Affichage.safe("Prix et pente selon le strike"):
            cls.prix_selon_strike(market, option, pricer, key)
        with Affichage.safe("Types d'options"):
            cls.types_options(market, option, pricer, key)
        with Affichage.safe("Grecques selon un paramètre"):
            cls.grecques_selon_parametre(market, option, pricer, key)
        gc.collect()

    @classmethod
    def convergence_prix(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Prix binomial et trinomial selon N (1 à 100, puis 40 valeurs jusqu'au N maximal) : niveau, erreur, x N."""
        n_max = max(500, pricer.nb_steps)
        with Affichage.section(f"Convergence du prix : binomial vs trinomial – 1 à {n_max:,} pas",
                           "Trait plein : binomial ; tirets : "
                           "trinomial. Référence : Black-Scholes, sinon trinomial à 2 x N."):
            view = st.radio("Graphique", cls.CONV_VIEWS, horizontal=True, key=f"cmp_conv_view_{key}")
            saved = st.session_state.get("res_conv_price")
            if saved is None or saved[0] != key:
                with st.spinner("Convergence du prix…"):
                    grid = sorted(set(range(1, 101)) | {int(round(x)) for x in np.geomspace(100, n_max, 41)})
                    df = pd.DataFrame([{"N": n, **{m.label: Orchestrator.price(market, option, PriceParameters(
                        **{**vars(pricer), "nb_steps": n, "tree_method": m, "engine": "vector"}))
                        for m in cls.METHODS}} for n in grid])
                    ref, ref_label = Orchestrator.black_scholes(market, option, pricer), "Black-Scholes"
                    if ref is None:
                        n_ref = min(10_000, 2 * n_max)
                        ref = Orchestrator.price(market, option, PriceParameters(
                            **{**vars(pricer), "nb_steps": n_ref, "tree_method": TreeMethod.TRINOMIAL,
                               "engine": "vector"}))
                        ref_label = f"trinomial N = {n_ref:,}"
                st.session_state["res_conv_price"] = (key, df, ref, ref_label)
            _, df, ref, ref_label = st.session_state["res_conv_price"]
            scale = df["N"] if view == "Erreur x N" else 1
            y_title = "Erreur × N (arbre - référence)" if view == "Erreur x N" else "Erreur (arbre - référence)"
            fig = Affichage.new_fig(f"Convergence du prix vs {ref_label}", "Nombre de pas N", y_title, height=470)
            for m in cls.METHODS:
                color, dash = Affichage.METHOD_STYLE[m]
                fig.add_trace(go.Scatter(x=df["N"], y=(df[m.label] - ref) * scale, mode="lines+markers",
                                         name=m.label, marker=dict(size=3),
                                         line=dict(color=color, width=1.6, dash=dash)))
            fig.add_hline(y=0, line_dash="dot", line_color="black")
            fig.update_xaxes(type="log")
            Affichage.show_chart(fig, f"cmp_conv_{key}")

    @classmethod
    def convergence_grecques(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Une grecque (GreeksTree) selon N, binomial et trinomial ; référence Black-Scholes si disponible."""
        bs = Orchestrator.greeks_bs(market, option, pricer)
        n_max = min(max(100, pricer.nb_steps),
                    2000 if pricer.pruning else pricer.max_unpruned_steps("vector"))
        with Affichage.section(f"Convergence des grecques : binomial vs trinomial – 3 à {n_max:,} pas",
                               "Erreur vs Black-Scholes ; « Erreur x N » pour l'ordre de convergence. "
                               "Si BS n'est pas disponible (américaine, dividende) : valeurs brutes."):
            c_g, c_v = st.columns([2, 3], vertical_alignment="bottom")
            greek = c_g.selectbox("Grecque", GREEKS, key=f"cmp_greek_conv_{key}")
            if bs:
                view = c_v.radio("Graphique", cls.CONV_VIEWS, horizontal=True, key=f"cmp_gconv_view_{key}")
            else:
                view = None
                c_v.caption("Black-Scholes indisponible : valeurs brutes affichées (pas de bascule erreur).")
            store = st.session_state.setdefault("res_conv_greeks", {})
            if store.get("key") != key:
                store.clear()
                store["key"] = key
            if greek not in store:
                with st.spinner(f"Convergence de {greek}…"):
                    grid = sorted(set(range(3, min(100, n_max) + 1)) |
                                  ({int(round(x)) for x in np.geomspace(100, n_max, 31)} if n_max > 100 else set()))
                    store[greek] = {m: pd.DataFrame([{"N": n, greek: Orchestrator.greeks_tree(
                        market, option, PriceParameters(**{**vars(pricer), "nb_steps": n, "tree_method": m}),
                        (greek,))[greek]} for n in grid]) for m in cls.METHODS}
            if bs:
                ref = bs[greek]
                scale = next(iter(store[greek].values()))["N"] if view == "Erreur x N" else 1
                y_title = f"{greek} : erreur × N vs BS" if view == "Erreur x N" else f"{greek} : erreur vs BS"
                fig = Affichage.new_fig(f"Convergence de {greek} vs Black-Scholes", "Nombre de pas N", y_title, height=470)
                for m, df in store[greek].items():
                    color, dash = Affichage.METHOD_STYLE[m]
                    fig.add_trace(go.Scatter(x=df["N"], y=(df[greek] - ref) * scale, mode="lines+markers",
                                             name=m.label, marker=dict(size=3),
                                             line=dict(color=color, width=1.6, dash=dash)))
                fig.add_hline(y=0, line_dash="dot", line_color="black")
            else:
                fig = Affichage.new_fig(f"Convergence de {greek} (valeurs brutes)", "Nombre de pas N", greek, height=470)
                for m, df in store[greek].items():
                    color, dash = Affichage.METHOD_STYLE[m]
                    fig.add_trace(go.Scatter(x=df["N"], y=df[greek], mode="lines+markers",
                                             name=m.label, marker=dict(size=3),
                                             line=dict(color=color, width=1.6, dash=dash)))
            fig.update_xaxes(type="log")
            Affichage.show_chart(fig, f"cmp_gconv_{key}")
            st.caption("Les dérivées secondes (Gamma, Vomma) oscillent davantage : elles dérivent deux fois un prix déjà "
                       "discrétisé, et la grille se déplace avec le paramètre choqué.")

    @classmethod
    def prix_selon_strike(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Prix, pente dV/dK et écart à Black-Scholes pour 101 strikes valorisés sur la même grille."""
        bs_ok = Black_Scholes.is_available(market, option)
        n_used = min(pricer.nb_steps, 5000)
        method = pricer.tree_method
        with Affichage.section(
                f"Prix et pente selon le strike – {n_used:,} pas ({method.label.lower()})",
                "Pente d'un call : -e^(-rT)·P(S_T > K) ; d'un put : e^(-rT)·P(S_T < K). "
                "Axe de droite : pente · axe extérieur : écart arbre - BS."):
            strikes = np.linspace(0.5 * option.K, 1.5 * option.K, 101)
            saved = st.session_state.get("res_strike")
            if saved is None or saved[0] != key:
                with st.spinner("Prix selon le strike…"):
                    prices = Orchestrator.price_many(
                        market, option,
                        PriceParameters(**{**vars(pricer), "nb_steps": n_used, "tree_method": method}),
                        strikes, [option.call_put == "call"] * 101, [option.is_american()] * 101)
                    curve = pd.DataFrame({"K": strikes, "Prix": prices,
                                          "Pente": np.gradient(prices, strikes)})
                st.session_state["res_strike"] = (key, curve)
            curve = st.session_state["res_strike"][1]
    
            fig = Affichage.new_fig(f"Prix et pente selon le strike – {method.label}", "Strike K",
                                    "Prix de l'option", height=480)
            color, _ = Affichage.METHOD_STYLE[method]
            fig.add_trace(go.Scatter(x=curve["K"], y=curve["Prix"], name=f"Prix – {method.label}",
                                     line=dict(color=color)))
            fig.add_trace(go.Scatter(x=curve["K"], y=curve["Pente"], name=f"Pente – {method.label}",
                                     yaxis="y2", line=dict(color=color, dash="dash")))
    
            if bs_ok:                                                    # BS : prix fermé et pente analytique
                T = PriceParameters.year_fraction(pricer.pricing_date, option.maturity)
                bs_price = np.array([Black_Scholes.price_from(market.s0, k, T, market.rate, market.vol,
                                                              option.call_put) for k in strikes])
                d2 = (np.log(market.s0 / strikes) + (market.rate - 0.5 * market.vol ** 2) * T) / (
                        market.vol * math.sqrt(T))
                disc = math.exp(-market.rate * T)
                slope = -disc * norm.cdf(d2) if option.call_put == "call" else disc * norm.cdf(-d2)
                fig.add_trace(go.Scatter(x=strikes, y=bs_price, name="Prix – BS",
                                         line=dict(color="black", width=1)))
                fig.add_trace(go.Scatter(x=strikes, y=slope, name="Pente – BS", yaxis="y2",
                                         line=dict(color="black", width=1, dash="dash")))
                fig.add_trace(go.Scatter(x=strikes, y=curve["Prix"] - bs_price,
                                         name=f"Écart arbre – BS – {method.label}", yaxis="y3",
                                         line=dict(color=color, dash="dot", width=2)))
            if bs_ok:
                fig.update_layout(
                    xaxis=dict(domain=[0, 0.72]),
                    yaxis2=dict(title="Pente", overlaying="y", side="right", showgrid=False),
                    yaxis3=dict(title="Écart arbre - BS", overlaying="y", side="right",
                                anchor="free", position=1.0, showgrid=False, tickformat=".2e"),
                    margin=dict(l=70, r=170, t=60, b=40),
                )
            else:
                fig.update_layout(yaxis2=dict(title="Pente", overlaying="y", side="right", showgrid=False))
    
            fig.add_vline(x=market.s0, line_dash="dash", line_color="grey", annotation_text="S0")
            Affichage.show_chart(fig, f"cmp_strike_{key}")
                
    @staticmethod
    def types_options(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Call / put x européen / américain quand le dividende, sa date, la maturité ou le taux varie."""
        n_used = min(pricer.nb_steps, 1000)
        with Affichage.section(f"Call / put, américain / européen – {n_used:,} pas",
                               "Les quatre options sur la même grille. L'écart "
                               "américaine - européenne mesure la valeur de l'exercice anticipé."):
            c1, c2, c3 = st.columns([3, 2, 1.4], vertical_alignment="bottom")
            which = c1.radio("Paramètre qui varie", ["Dividende", "Date ex-dividende", "Maturité", "Taux"], horizontal=True, key=f"cmp_types_{key}")
            view = c2.radio("Afficher", ["Prix", "Écart américain - européen"], horizontal=True, key=f"cmp_types_view_{key}")
            zero_rate = c3.checkbox("Taux r = 0", value=which != "Taux", disabled=which == "Taux", key=f"cmp_types_r0_{key}_{which}",
                                    help="Isole l'effet du dividende de celui du taux.")
            m = Market(**{**vars(market), "rate": 0.0}) if zero_rate and which != "Taux" else market
            saved = st.session_state.get("res_types")
            if saved is None or saved[0] != (key, which, zero_rate):
                with st.spinner("Calcul des 4 types d'options…"):
                    days = (option.maturity - pricer.pricing_date).days
                    div_date = (m.ex_div_date if m.ex_div_date and 0 < (m.ex_div_date - pricer.pricing_date).days <= days
                                else pricer.pricing_date + dt.timedelta(days=days // 2))
                    div = m.dividend if m.dividend > 0 else round(0.03 * m.s0, 2)
                    if which == "Dividende":
                        xs = list(np.linspace(0.0, 0.15 * m.s0, 13))
                        cases = [(Market(m.s0, m.rate, m.vol, x, div_date), option) for x in xs]
                    elif which == "Date ex-dividende":
                        xs = [pricer.pricing_date + dt.timedelta(days=int(d)) for d in np.linspace(1, days, 13)]
                        cases = [(Market(m.s0, m.rate, m.vol, div, x), option) for x in xs]
                    elif which == "Maturité":
                        xs = [pricer.pricing_date + dt.timedelta(days=max(1, int(d))) for d in np.linspace(0.1 * days, 2 * days, 13)]
                        cases = [(m, Option(option.K, x, option.call_put, option.eur_am)) for x in xs]
                    else:
                        xs = list(np.linspace(-0.01, 0.10, 12))
                        cases = [(Market(m.s0, x, m.vol, m.dividend, m.ex_div_date), option) for x in xs]
                    p = PriceParameters(**{**vars(pricer), "nb_steps": n_used})
                    labels = ["Call européen", "Call américain", "Put européen", "Put américain"]
                    df = pd.DataFrame([{"x": x, **dict(zip(labels, Orchestrator.price_many(
                        mk, op, p, [op.K] * 4, [True, True, False, False], [False, True, False, True])))}
                        for x, (mk, op) in zip(xs, cases)])
                st.session_state["res_types"] = ((key, which, zero_rate), df)
            df = st.session_state["res_types"][1]
            fig = Affichage.new_fig(f"{view} selon : {which.lower()}" + (" (r = 0)" if m.rate == 0 and which != "Taux" else ""),
                                    which, "Prix" if view == "Prix" else "Américaine - européenne")
            if view == "Prix":
                for col in df.columns[1:]:
                    fig.add_trace(go.Scatter(x=df["x"], y=df[col], mode="lines+markers", name=col,
                                             line=dict(color=Affichage.COLORS[0 if col.startswith("Call") else 1],
                                                       dash="solid" if "américain" in col else "dot")))
            else:
                for i, kind in enumerate(("Call", "Put")):
                    fig.add_trace(go.Scatter(x=df["x"], y=df[f"{kind} américain"] - df[f"{kind} européen"],
                                             mode="lines+markers", name=f"{kind} : US − EU",
                                             line=dict(color=Affichage.COLORS[i])))
            Affichage.show_chart(fig, f"cmp_types_plot_{key}")

    @staticmethod
    def grecques_selon_parametre(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Grecques de l'arbre (trait plein) et Black-Scholes (pointillés) quand S0, r, T ou σ varie."""
        n_used = min(pricer.nb_steps, 300)
        with Affichage.section(f"Grecques selon un paramètre, comparées à Black-Scholes – {n_used:,} pas", ""):
            c_p, c_g, c_n = st.columns([2, 3, 1.3], vertical_alignment="bottom")
            which = c_p.radio("Paramètre qui varie", ["S0", "r", "T", "σ"], horizontal=True, key=f"cmp_gparam_{key}")
            greeks = c_g.multiselect("Grecques", GREEKS, default=["Delta", "Gamma", "Vega"], key=f"cmp_gparam_sel_{key}")
            normalize = c_n.checkbox("Normaliser", value=True, key=f"cmp_norm_{key}",
                                     help="Divise chaque grecque par son max |arbre| : rend comparables des grecques d'ordres de grandeur différents.")
            saved = st.session_state.get("res_gparam")
            if saved is None or saved[0] != (key, which):
                with st.spinner("Grecques générées par l'arbre…"):
                    days = (option.maturity - pricer.pricing_date).days
                    if which == "S0":
                        xs = list(np.linspace(0.6 * market.s0, 1.4 * market.s0, 25))
                        cases = [(Market(**{**vars(market), "s0": x}), option) for x in xs]
                    elif which == "r":
                        xs = list(np.linspace(-0.01, 0.10, 25))
                        cases = [(Market(**{**vars(market), "rate": x}), option) for x in xs]
                    elif which == "T":
                        ds = sorted({max(2, int(d)) for d in np.linspace(0.1 * days, 2 * days, 25)})
                        xs = [d / 365.0 for d in ds]
                        cases = [(market, Option(option.K, pricer.pricing_date + dt.timedelta(days=d), option.call_put,
                                                 option.eur_am)) for d in ds]
                    else:
                        xs = list(np.linspace(0.05, 0.60, 25))
                        cases = [(Market(**{**vars(market), "vol": x}), option) for x in xs]
                    p = PriceParameters(**{**vars(pricer), "nb_steps": n_used})
                    rows = []
                    for x, (mk, op) in zip(xs, cases):
                        bs = Orchestrator.greeks_bs(mk, op, p) or {}
                        rows.append({"x": x, **Orchestrator.greeks_tree(mk, op, p), **{f"BS {g}": v for g, v in bs.items()}})
                st.session_state["res_gparam"] = ((key, which), pd.DataFrame(rows))
            df = st.session_state["res_gparam"][1]
            fig = Affichage.new_fig(f"Grecques selon {which}", which, "Grecque normalisée" if normalize else "Valeur")
            for g in greeks:
                color = Affichage.COLORS[GREEKS.index(g) % 7]
                scale = (float(np.nanmax(np.abs(df[g]))) or 1.0) if normalize else 1.0
                fig.add_trace(go.Scatter(x=df["x"], y=df[g] / scale, mode="lines+markers", name=f"{g} – arbre",
                                         line=dict(color=color)))
                if f"BS {g}" in df:
                    fig.add_trace(go.Scatter(x=df["x"], y=df[f"BS {g}"] / scale, name=f"{g} – BS",
                                             line=dict(color=color, dash="dot")))
            Affichage.show_chart(fig, f"cmp_gparam_plot_{key}")


# ====================================================================================
# Onglet 3 : analyse des performances
# ====================================================================================
class EtudePerformances:
    """Temps, mémoire, backward / récursif / vectorisé, méthodes Delta / Gamma, erreurs dues au pruning."""

    PMINS = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]

    @classmethod
    def render(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        Affichage.params_expander(market, option, pricer)
        if st.button("Lancer l'analyse des performances", type="primary", key="run_perf"):
            st.session_state["perf_run"] = key
        if st.session_state.get("perf_run") != key:
            st.info("Cliquez sur « Lancer l'analyse des performances » (de quelques secondes à quelques minutes).")
            return
        with Affichage.safe("Temps d'exécution"):
            cls.temps_execution(market, option, pricer, key)
        with Affichage.safe("Mémoire"):
            cls.memoire(market, option, pricer, key)
        with Affichage.safe("Backward, récursif et vectorisé"):
            cls.backward_recursif_vectorise(market, option, pricer, key)
        with Affichage.safe("Méthodes Delta / Gamma"):
            cls.methodes_delta_gamma(market, option, pricer, key)
        with Affichage.safe("Erreurs dues au pruning"):
            cls.erreurs_pruning(market, option, pricer, key)
        gc.collect()

    @staticmethod
    def temps_execution(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Temps total (construction + valorisation) : moteur vectorisé vs objets, pruning ON / OFF."""
        n_max = min(max(1000, pricer.nb_steps), 10_000)
        grid = sorted({n for n in (50, 100, 200, 500, 1000, 2000, 5000, 10000) if n <= n_max} | {pricer.nb_steps})
        with Affichage.section(f"Temps d'exécution : vectorisé vs objets, avec et sans pruning – 50 à {grid[-1]:,} pas",
                               "Trait plein : pruning ON ; pointillés : OFF. Les deux moteurs construisent le même arbre."):
            saved = st.session_state.get("res_time")
            if saved is None or saved[0] != key:
                n_max = min(max(1000, pricer.nb_steps), 10_000)
                grid = sorted({n for n in (50, 100, 200, 500, 1000, 2000, 5000, 10000) if n <= n_max} | {pricer.nb_steps})
                frames = {}
                with st.spinner("Mesure des temps (les deux moteurs)…"):
                    for engine, name in (("vector", "Vectorisé"), ("objects", "Objets")):
                        for prune in (True, False):
                            p = PriceParameters(**{**vars(pricer), "engine": engine, "pruning": prune})
                            frames[f"{name} – {'ON' if prune else 'OFF'}"] = pd.DataFrame([
                                Orchestrator.benchmark(market, option, PriceParameters(**{**vars(p), "nb_steps": n}))
                                for n in grid if prune or n <= p.max_unpruned_steps()])
                st.session_state["res_time"] = (key, frames)
            frames = st.session_state["res_time"][1]
            Affichage.lines({k: (v, "Total (s)", "solid" if k.endswith("ON") else "dot") for k, v in frames.items()},
                            "Temps total selon N", "Temps (s)", "perf_time", log_x=True, log_y=True)

    @staticmethod
    def memoire(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Pic de mémoire Python (tracemalloc) et nombre de nœuds : pruning ON vs OFF, N = 10 → 100."""
        with Affichage.section("Mémoire et nœuds conservés : pruning ON vs OFF – 10 à 100 pas", ""):
            saved = st.session_state.get("res_memory")
            if saved is None or saved[0] != key:
                with st.spinner("Mesures mémoire…"):
                    frames = {prune: pd.DataFrame([Orchestrator.benchmark(market, option, PriceParameters(
                        **{**vars(pricer), "nb_steps": n, "pruning": prune}), memory=True) for n in range(10, 101, 10)])
                        for prune in (True, False)}
                st.session_state["res_memory"] = (key, frames)
            on, off = st.session_state["res_memory"][1][True], st.session_state["res_memory"][1][False]
            c1, c2 = st.columns(2)
            with c1:
                Affichage.lines({"Pruning ON": (on, "Pic mémoire (Mo)", "solid"), "Pruning OFF": (off, "Pic mémoire (Mo)", "dot")},
                                "Pic de mémoire Python", "Mo", "perf_mem")
            with c2:
                Affichage.lines({"Pruning ON": (on, "Nœuds", "solid"), "Pruning OFF": (off, "Nœuds", "dot")},
                                "Nœuds conservés", "Nombre de nœuds", "perf_nodes")

    @staticmethod
    def backward_recursif_vectorise(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Valorisation seule : backward et récursif (même arbre de Node) contre vectorisé (numpy)."""
        n_max = min(max(1000, pricer.nb_steps), 5000 if pricer.pruning else pricer.max_unpruned_steps("objects"))
        grid = [n for n in (50, 100, 250, 500, 1000, 2000, 5000) if n <= n_max]
        grid_hi = max(grid) if grid else n_max
        with Affichage.section(f"Valorisation : backward (objets) vs récursif (objets) vs vectorisé (numpy) – 50 à {grid_hi:,} pas", ""):
            saved = st.session_state.get("res_br")
            if saved is None or saved[0] != key:
                n_max = min(max(1000, pricer.nb_steps), 5000 if pricer.pruning else pricer.max_unpruned_steps("objects"))
                rows = []
                with st.spinner("Backward, récursif et vectorisé…"):
                    for n in (n for n in (50, 100, 250, 500, 1000, 2000, 5000) if n <= n_max):
                        obj = Orchestrator.benchmark(market, option, PriceParameters(
                            **{**vars(pricer), "nb_steps": n, "engine": "objects"}), with_recursive=True)
                        vec = Orchestrator.benchmark(market, option, PriceParameters(
                            **{**vars(pricer), "nb_steps": n, "engine": "vector"}))
                        rows.append({"N": n, "Backward (s)": obj["Valorisation (s)"], "Récursif (s)": obj["Récursif (s)"],
                                     "Vectorisé (s)": vec["Valorisation (s)"]})
                st.session_state["res_br"] = (key, pd.DataFrame(rows))
            br = st.session_state["res_br"][1]
            Affichage.lines({"Backward (objets)": (br, "Backward (s)", "solid"), "Récursif (objets)": (br, "Récursif (s)", "dot"),
                             "Vectorisé (numpy)": (br, "Vectorisé (s)", "dash")},
                            "Temps de valorisation selon N", "Temps (s)", "perf_br", log_x=True, log_y=True)

    @staticmethod
    def methodes_delta_gamma(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Delta et Gamma par les trois méthodes du cours : temps de calcul et écart à Black-Scholes."""
        bs = Orchestrator.greeks_bs(market, option, pricer, ("Delta", "Gamma"))
        with Affichage.section("Delta et Gamma : trois méthodes de calcul (slides du cours) – 25 à 1 000 pas", ""):
            saved = st.session_state.get("res_dg")
            if saved is None or saved[0] != key:
                rows = []
                with st.spinner("Delta et Gamma par les trois méthodes…"):
                    for n in (25, 50, 100, 200, 400, 700, 1000):
                        for m in Orchestrator.DELTA_METHODS:
                            t0 = time.perf_counter()
                            delta, gamma = Orchestrator.delta_gamma(market, option, PriceParameters(
                                **{**vars(pricer), "nb_steps": n}), m)
                            rows.append({"N": n, "Méthode": m, "Delta": delta, "Gamma": gamma,
                                         "Temps (s)": time.perf_counter() - t0})
                st.session_state["res_dg"] = (key, pd.DataFrame(rows))
            gm = st.session_state["res_dg"][1]
            views = ["Temps de calcul", "Delta selon N", "Gamma selon N"]
            view = st.radio("Graphique", views, horizontal=True, key="perf_dg_view")
            col = {"Temps de calcul": "Temps (s)", "Delta selon N": "Delta", "Gamma selon N": "Gamma"}[view]
            Affichage.lines({m: (gm[gm["Méthode"] == m], col, d) for m, d in zip(Orchestrator.DELTA_METHODS, ("solid", "dot", "dash"))},
                            view, col, f"perf_dg_{key}", log_x=True, log_y=view == "Temps de calcul")
            if bs:
                st.caption(f"Black-Scholes : Delta {bs['Delta']:.6f}, Gamma {bs['Gamma']:.6f}. Le choc du spot déplace la "
                           "grille, ce qui rend son Gamma bruité ; les nœuds autour de la racine gardent une grille fixe.")

    @classmethod
    def erreurs_pruning(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Heatmaps p_min × N : erreur de prix et erreur de distribution terminale (Kolmogorov-Smirnov)."""
        with Affichage.section("Erreur due au pruning selon p_min et N – 50 à 400 pas",
                               "Erreur = |avec pruning − sans pruning| à N identique (couleur en log10)."):
            saved = st.session_state.get("res_prune")
            if saved is None or saved[0] != key:
                rows = []
                with st.spinner("Erreurs dues au pruning…"):
                    for n in (50, 100, 200, 400):
                        ref = Orchestrator.vector_tree(market, option, PriceParameters(
                            **{**vars(pricer), "nb_steps": n, "pruning": False})).levels
                        for p_min in cls.PMINS:
                            lv = Orchestrator.vector_tree(market, option, PriceParameters(
                                **{**vars(pricer), "nb_steps": n, "pruning": True, "p_min": p_min})).levels
                            grid = np.union1d(lv[-1]["S"], ref[-1]["S"])          # distance KS des lois terminales
                            cdfs = []
                            for level in (lv[-1], ref[-1]):
                                order = np.argsort(level["S"])
                                idx = np.searchsorted(level["S"][order], grid, side="right") - 1
                                cum = np.cumsum(level["P"][order])
                                cdfs.append(np.where(idx >= 0, cum[np.clip(idx, 0, None)], 0.0))
                            rows.append({"N": n, "p_min": p_min, "Erreur de prix": abs(lv[0]["V"][0] - ref[0]["V"][0]),
                                         "Erreur de distribution": float(np.max(np.abs(cdfs[0] - cdfs[1])))})
                st.session_state["res_prune"] = (key, pd.DataFrame(rows))
            err = st.session_state["res_prune"][1]
            c1, c2 = st.columns(2)
            with c1:
                Affichage.error_heatmap(err, "Erreur de prix", "Erreur de prix", f"perf_prune_price_{key}")
            with c2:
                Affichage.error_heatmap(err, "Erreur de distribution", "Erreur de distribution terminale (KS)", f"perf_prune_dist_{key}")
