"""
hedging/greeks.py

Advanced options Greeks calculations for sophisticated hedging analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Optional

def calculate_black_scholes_greeks(spot_price: float, strike_price: float, 
                                 time_to_expiry: float, risk_free_rate: float,
                                 volatility: float, option_type: str) -> Dict[str, float]:
    """
    Calculate all Black-Scholes Greeks for an option.
    
    Returns:
        Dict with Delta, Gamma, Theta, Vega, Rho
    """
    
    # Black-Scholes intermediate calculations
    d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    # Standard normal PDF and CDF
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    npdf_d1 = norm.pdf(d1)
    
    if option_type.lower() == "call":
        # Call option Greeks
        delta = nd1
        theta = (-spot_price * npdf_d1 * volatility / (2 * np.sqrt(time_to_expiry)) 
                - risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * nd2) / 365
        rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * nd2 / 100
        
    else:  # put option
        # Put option Greeks  
        delta = nd1 - 1
        theta = (-spot_price * npdf_d1 * volatility / (2 * np.sqrt(time_to_expiry)) 
                + risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * (1 - nd2)) / 365
        rho = -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * (1 - nd2) / 100
    
    # Greeks that are the same for calls and puts
    gamma = npdf_d1 / (spot_price * volatility * np.sqrt(time_to_expiry))
    vega = spot_price * npdf_d1 * np.sqrt(time_to_expiry) / 100
    
    return {
        'delta': float(delta),
        'gamma': float(gamma), 
        'theta': float(theta),
        'vega': float(vega),
        'rho': float(rho)
    }

def calculate_portfolio_greeks(positions: list, spot_price: float) -> Dict[str, float]:
    """
    Calculate net Greeks for a portfolio of options positions.
    
    Args:
        positions: List of dicts with option details and quantities
        spot_price: Current underlying price
    
    Returns:
        Net portfolio Greeks
    """
    
    net_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    for position in positions:
        greeks = calculate_black_scholes_greeks(
            spot_price, position['strike'], position['expiry'],
            position['rate'], position['vol'], position['type']
        )
        
        # Multiply by position size and add to portfolio
        for greek in net_greeks:
            net_greeks[greek] += greeks[greek] * position['quantity']
    
    return net_greeks

def simulate_greeks_pnl(spot_prices: np.ndarray, greeks: Dict[str, float],
                       price_change_pct: float = 0.01) -> Dict[str, np.ndarray]:
    """
    Simulate P&L attribution from different Greeks.
    
    Returns:
        Dict with P&L from Delta, Gamma, Theta, Vega effects
    """
    
    initial_price = spot_prices[0]
    price_changes = spot_prices - initial_price
    
    # P&L attribution (simplified)
    delta_pnl = greeks['delta'] * price_changes
    gamma_pnl = 0.5 * greeks['gamma'] * price_changes**2  
    theta_pnl = greeks['theta'] * np.ones_like(price_changes)  # Per day
    
    return {
        'delta_pnl': delta_pnl,
        'gamma_pnl': gamma_pnl, 
        'theta_pnl': theta_pnl,
        'total_pnl': delta_pnl + gamma_pnl + theta_pnl
    }
