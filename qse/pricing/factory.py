"""Factory for option pricers."""

from __future__ import annotations

from qse.config.factories import FactoryBase
from qse.exceptions import DependencyError
from qse.pricing.black_scholes import BlackScholesPricer
from qse.pricing.py_vollib import PyVollibPricer
from qse.pricing.quantlib_stub import QuantLibPricer

_PRICERS = {
    "black_scholes": BlackScholesPricer,
    "black-scholes": BlackScholesPricer,
    "bs": BlackScholesPricer,
    "py_vollib": PyVollibPricer,
    "py-vollib": PyVollibPricer,
    "quantlib": QuantLibPricer,
}


def get_pricer(name: str):
    normalized = name.lower()
    pricer_cls = _PRICERS.get(normalized)
    if pricer_cls is None:
        raise DependencyError(f"Unknown pricer: {name}")
    return pricer_cls()


def pricer_factory(name: str) -> FactoryBase:
    normalized = name.lower()
    return FactoryBase(name=normalized, builder=lambda: get_pricer(normalized))
