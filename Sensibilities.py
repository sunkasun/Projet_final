from __future__ import annotations
import copy
import datetime as dt
from typing import Callable
from Market import Market
from Option import Option
from PriceParameters import PriceParameters

GREEKS = ("Delta", "Gamma", "Theta", "Vega", "Rho", "Vanna", "Vomma")
TRADING_DAYS = 252.0      # Theta exprimé par jour de bourse (convention de marché, comme le Theta Black-Scholes)

class Parameters:
    """Copie « à plat » et modifiable de toutes les données de pricing -> c'est l'objet que l'on choque."""
    def __init__(self, pricer_parameters: PriceParameters, market_data: Market, option_data: Option) -> None:
        self.s0, self.rate, self.vol = market_data.s0, market_data.rate, market_data.vol
        self.dividend, self.ex_div_date = market_data.dividend, market_data.ex_div_date
        self.K, self.maturity = option_data.K, option_data.maturity
        self.call_put, self.eur_am = option_data.call_put, option_data.eur_am
        self.pricer = pricer_parameters                    # pas, méthode, pruning : jamais choqués

    def to_objects(self) -> tuple[Market, Option, PriceParameters]:
        """Objets du pricer correspondant à cet état (choqué ou non)."""
        return (Market(self.s0, self.rate, self.vol, self.dividend, self.ex_div_date),
                Option(self.K, self.maturity, self.call_put, self.eur_am), self.pricer)


class OneDimDerivative:
    """Dérivée empirique par rapport à un seul paramètre (copies choquées de ± h)."""
    def __init__(self, pricing_function: Callable[[Parameters], float], parameters: Parameters) -> None:
        self.pricing_function = pricing_function
        self.parameters = parameters

    def first_derivative(self, name: str, h: float, is_gamma: bool = False):
        """(V(x + h) − V(x − h)) / 2h ; avec is_gamma, les deux pentes décentrées (pour la dérivée seconde)."""
        up, down = copy.deepcopy(self.parameters), copy.deepcopy(self.parameters)
        setattr(up, name, getattr(up, name) + h)
        setattr(down, name, getattr(down, name) - h)
        v_up, v_down = self.pricing_function(up), self.pricing_function(down)
        if is_gamma:
            v_mid = self.pricing_function(self.parameters)
            return (v_up - v_mid) / h, (v_mid - v_down) / h
        return (v_up - v_down) / (2 * h)

    def second_derivative(self, name: str, h: float) -> float:
        slope_up, slope_down = self.first_derivative(name, h, is_gamma=True)
        return (slope_up - slope_down) / h


class GreeksTree:
    # Pas de garbage collector ici : chaque pricing passe par Orchestrator.price, qui libère son arbre dès le prix
    # obtenu (liens entre nœuds cassés, tableaux oubliés) ; la mémoire est rendue sans attendre le GC. Seuls des
    # nombres (les prix mémorisés) restent dans GreeksTree. Un gc.collect() par prix ralentirait le calcul.
    """Grecques de l'arbre : l'arbre complet est revalorisé avec un paramètre choqué, même nombre de pas.

    pricing_function(market, option, pricer) -> prix (fournie par Orchestrator). Les chocs sont réglables ;
    les prix déjà calculés sont mémorisés (V(S0) sert à Gamma, Theta et Vomma).
    """

    def __init__(self, pricer_parameters: PriceParameters, market_data: Market, option_data: Option,
                 pricing_function: Callable[[Market, Option, PriceParameters], float], spot_shift: float = 0.01,
                 vol_shift: float = 0.01, rate_shift: float = 0.01, theta_days: int = 1) -> None:
        self.params = Parameters(pricer_parameters, market_data, option_data)
        self.pricing_function = pricing_function
        self.h_spot = spot_shift * market_data.s0                      # choc relatif du spot (1 % par défaut)
        self.h_vol = min(vol_shift, 0.5 * market_data.vol) if market_data.vol > 0 else vol_shift
        self.h_rate, self.theta_days = rate_shift, theta_days
        self._prices: dict = {}

    def price_tree_one_dim(self, params: Parameters) -> float:
        """Prix de l'arbre pour un état des paramètres (mémorisé : clé = valeurs de tous les attributs)."""
        key = tuple(vars(params).values())
        if key not in self._prices:
            self._prices[key] = self.pricing_function(*params.to_objects())
        return self._prices[key]

    def delta(self) -> float:
        return OneDimDerivative(self.price_tree_one_dim, self.params).first_derivative("s0", self.h_spot)

    def gamma(self) -> float:
        return OneDimDerivative(self.price_tree_one_dim, self.params).second_derivative("s0", self.h_spot)

    def vega(self) -> float:
        return OneDimDerivative(self.price_tree_one_dim, self.params).first_derivative("vol", self.h_vol) / 100

    def rho(self) -> float:
        return OneDimDerivative(self.price_tree_one_dim, self.params).first_derivative("rate", self.h_rate) / 100

    def vomma(self) -> float:
        return OneDimDerivative(self.price_tree_one_dim, self.params).second_derivative("vol", self.h_vol) / 100 ** 2

    def vanna(self) -> float:
        """dDelta/dσ : Delta (choc de S0) dérivé par rapport à σ."""
        delta_at = lambda p: OneDimDerivative(self.price_tree_one_dim, p).first_derivative("s0", self.h_spot)
        return OneDimDerivative(delta_at, self.params).first_derivative("vol", self.h_vol) / 100

    def theta(self) -> float:
        """Maturité raccourcie de theta_days jours calendaires, ramenée à un jour de bourse (× 365/252)."""
        shocked = copy.deepcopy(self.params)
        shocked.maturity -= dt.timedelta(days=self.theta_days)
        v_now, v_later = self.price_tree_one_dim(self.params), self.price_tree_one_dim(shocked)
        return (v_later - v_now) / self.theta_days * (365.0 / TRADING_DAYS)
