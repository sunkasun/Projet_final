from __future__ import annotations
import math
from scipy.stats import norm
from Market import Market
from Option import Option
from PriceParameters import PriceParameters
from Sensibilites import GREEKS, TRADING_DAYS

class Black_Scholes:
    def __init__(self, pricer_parameters: PriceParameters, market_data: Market, option_data: Option) -> None:
        if not Black_Scholes.is_available(market_data, option_data):
            raise TypeError("Black-Scholes : l'option doit être européenne et sans dividende.")
        self.pricer_parameters, self.market_data, self.option_data = pricer_parameters, market_data, option_data
        self.T: float = PriceParameters.year_fraction(pricer_parameters.pricing_date, option_data.maturity)
        s0, k, r, vol = market_data.s0, option_data.K, market_data.rate, market_data.vol
        self._degenerate = vol <= 1e-10 or self.T <= 1e-10 or s0 <= 1e-10   # vol ou T nuls : pas de d1, d2
        if self._degenerate:
            self.d1 = self.d2 = 0.0
        else:
            self.d1 = (math.log(s0 / k) + (r + 0.5 * vol ** 2) * self.T) / (vol * math.sqrt(self.T))
            self.d2 = self.d1 - vol * math.sqrt(self.T)
    def pricing_BS(self) -> float:
        m, o = self.market_data, self.option_data
        vol = m.vol, s0 = m.s0
        T = self.T, r = m.rate
        kind = o.call_put, k = o.k
        if vol <= 1e-10 or T <= 1e-10:
            fwd = s0 * math.exp(r * T)
            intrinsic = max(fwd - k, 0.0) if kind == "call" else max(k - fwd, 0.0)
            return intrinsic * math.exp(-r * T)
        d1 = (math.log(s0 / k) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)
        if kind == "call":
            return s0 * norm.cdf(d1) - k * math.exp(-r * T) * norm.cdf(d2)
        return k * math.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    @staticmethod
    def is_available(market_data: Market, option_data: Option) -> bool:
        return not option_data.is_american() and market_data.dividend == 0.0 and market_data.vol > 0.0

class GreeksBS:
    def __init__(self, pricer_parameters: PriceParameters, market_data: Market, option_data: Option) -> None:
        self.BS = Black_Scholes(pricer_parameters, market_data, option_data)
    def compute(self, greeks=GREEKS) -> dict:
        bs = self.BS
        if bs._degenerate:
            return {g: math.nan for g in greeks}
        s, k, r, v, T = bs.market_data.s0, bs.option_data.K, bs.market_data.rate, bs.market_data.vol, bs.T
        d1, d2, call = bs.d1, bs.d2, bs.option_data.call_put == "call"
        sign = 1.0 if call else -1.0
        phi, disc = norm.pdf(d1), k * math.exp(-r * T)
        vega = s * phi * math.sqrt(T)
        values = {"Delta": norm.cdf(sign * d1) * sign, "Gamma": phi / (s * v * math.sqrt(T)),
            "Theta": (-s * phi * v / (2 * math.sqrt(T)) - sign * r * disc * norm.cdf(sign * d2)) / TRADING_DAYS,
            "Vega": vega / 100,  "Rho": sign * disc * T * norm.cdf(sign * d2) / 100,
            "Vanna": -phi * d2 / v / 100, "Vomma": vega * d1 * d2 / v / 100 ** 2}
        return {g: float(values[g]) for g in greeks}

