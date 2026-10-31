from __future__ import annotations
import datetime as dt
import math

class Market:
    def __init__(self, s0: float, rate: float, vol: float, dividend: float = 0.0, ex_div_date: dt.date | None = None) -> None:
        self.s0 : float = s0
        self.rate : float = rate
        self.vol : float = vol
        self.dividend : float = dividend
        self.ex_div_date = ex_div_date
    @staticmethod
    def esperance(und_price: float, time: float, r: float, dt_: float,
                  dividende: float = 0.0, t_div: float | None = 0.0) -> float:
        result = und_price * math.exp(r * dt_)
        if dividende > 0.0:
            result -= dividende * math.exp(r * (time + dt_ - t_div))
        return result
    @staticmethod
    def variance(und_price: float, time: float, r: float, sigma: float, dt_: float,
                 dividende: float = 0.0, t_div: float | None = 0.0) -> float:
        dividend_fraction = 0.0
        if dividende > 0.0:
            dividend_fraction = (time + dt_ - t_div) / dt_
        adjusted_price = und_price - dividend_fraction * dividende
        result = adjusted_price ** 2 * math.exp(2.0 * r * dt_) * math.expm1(dt_ * sigma ** 2)
        return result
