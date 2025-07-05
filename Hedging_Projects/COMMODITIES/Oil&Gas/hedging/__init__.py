"""
Hedging package for Oil & Gas commodity risk management.

This package provides tools for analyzing hedging strategies on oil & gas commodities,
including futures and options hedging, risk metrics calculation, and Monte Carlo simulation.
"""

__version__ = "1.0.0"
__author__ = "Oil & Gas Hedging Simulator"

# Make key functions easily accessible
from .data import get_prices, get_current_price, get_available_commodities
from .strategies import compute_payoff_diagram, compute_futures_hedge, compute_options_hedge
from .simulation import simulate_hedged_vs_unhedged, simulate_pnl
from .risk import calculate_risk_metrics, calculate_delta_exposure

__all__ = [
    'get_prices',
    'get_current_price', 
    'get_available_commodities',
    'compute_payoff_diagram',
    'compute_futures_hedge',
    'compute_options_hedge',
    'simulate_hedged_vs_unhedged',
    'simulate_pnl',
    'calculate_risk_metrics',
    'calculate_delta_exposure'
]
