"""
Module for hedge payoff calculations for different hedging strategies.
Supports Futures and Options hedging with payoff diagram generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import norm


def compute_futures_hedge(prices: pd.Series, position: float, hedge_ratio: float) -> pd.Series:
    """
    Calculate P&L for futures hedge strategy.
    """
  
    price_changes = prices.diff().dropna()
    
    # Futures hedge P&L = ΔP × position × hedge_ratio
    # For a long position, we sell futures (short hedge)
    # So hedge P&L = -ΔP × position × hedge_ratio
    hedge_pnl = -price_changes * position * hedge_ratio
    
    return hedge_pnl


def compute_options_hedge(prices: pd.Series, position: float, hedge_ratio: float, 
                         strike_price: float, option_type: str = "put",
                         time_to_expiry: float = 0.25, risk_free_rate: float = 0.05,
                         volatility: Optional[float] = None) -> pd.Series:
    """
    Calculate P&L for options hedge strategy (simplified Black-Scholes approach).
    """
    
    if volatility is None:
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    deltas = []
    for price in prices:
        delta = calculate_option_delta(price, strike_price, time_to_expiry, 
                                     risk_free_rate, volatility, option_type)
        deltas.append(delta)
    
    deltas = pd.Series(deltas, index=prices.index)
    
    price_changes = prices.diff().dropna()
    deltas_aligned = deltas[price_changes.index]
    
    # For a long position, we buy puts (protective hedge)
    hedge_pnl = price_changes * position * hedge_ratio * deltas_aligned
    
    return hedge_pnl


def calculate_option_delta(spot_price: float, strike_price: float, time_to_expiry: float,
                          risk_free_rate: float, volatility: float, option_type: str) -> float:
    """
    Calculate option delta using Black-Scholes formula.
    """
    
    if time_to_expiry <= 0:
        if option_type.lower() == "call":
            return 1.0 if spot_price > strike_price else 0.0
        else:  # put
            return -1.0 if spot_price < strike_price else 0.0
    
    d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
    else:  # put
        delta = norm.cdf(d1) - 1
    
    return delta


def compute_payoff_diagram(current_price: float, position: float, hedge_ratio: float,
                          strategy: str, strike_price: Optional[float] = None,
                          price_range_pct: float = 0.3, num_points: int = 100) -> Dict[str, pd.Series]:
    """
    Compute payoff diagram data for hedged vs unhedged positions.
    """
    
    if strike_price is None:
        strike_price = current_price
    
    price_min = current_price * (1 - price_range_pct)
    price_max = current_price * (1 + price_range_pct)
    spot_prices = np.linspace(price_min, price_max, num_points)
    
    # Calculate underlying position P&L
    # P&L = (S - S0) × position
    underlying_pnl = (spot_prices - current_price) * position
    
    # Calculate hedge P&L based on strategy
    if strategy.lower() == "futures":
        # Futures hedge: linear payoff
        # For long position, we sell futures (short hedge)
        # Hedge P&L = -(S - S0) × position × hedge_ratio
        hedge_pnl = -(spot_prices - current_price) * position * hedge_ratio
        
    elif strategy.lower() == "options":
        # Options hedge: non-linear payoff
        hedge_pnl = np.zeros_like(spot_prices)
        
        for i, spot_price in enumerate(spot_prices):
            if position > 0:  # Long underlying position - buy puts
                # Put option payoff: max(K - S, 0)
                option_payoff = max(strike_price - spot_price, 0)
            else:  # Short underlying position - buy calls
                # Call option payoff: max(S - K, 0)
                option_payoff = max(spot_price - strike_price, 0)
            
            # Scale by position and hedge ratio
            hedge_pnl[i] = option_payoff * abs(position) * hedge_ratio
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'Futures' or 'Options'")
    
    net_pnl = underlying_pnl + hedge_pnl
    breakeven_prices = find_breakeven_prices(spot_prices, net_pnl)
    
    result = {
        'spot_prices': pd.Series(spot_prices, name='Spot Price'),
        'underlying_pnl': pd.Series(underlying_pnl, name='Underlying P&L'),
        'hedge_pnl': pd.Series(hedge_pnl, name='Hedge P&L'),
        'net_pnl': pd.Series(net_pnl, name='Net P&L'),
        'breakeven_prices': breakeven_prices
    }
    
    return result


def find_breakeven_prices(prices: np.ndarray, pnl: np.ndarray, tolerance: float = 0.01) -> list:
    """
    Find breakeven prices where P&L crosses zero.
    """
    
    breakeven_prices = []
    
    for i in range(len(pnl) - 1):
        # Check if P&L crosses zero between consecutive points
        if (pnl[i] * pnl[i + 1] <= 0) and (abs(pnl[i]) > tolerance or abs(pnl[i + 1]) > tolerance):
            # Linear interpolation to find exact breakeven price
            if pnl[i + 1] != pnl[i]:
                breakeven_price = prices[i] - pnl[i] * (prices[i + 1] - prices[i]) / (pnl[i + 1] - pnl[i])
                breakeven_prices.append(breakeven_price)
    
    return breakeven_prices


def get_hedge_summary(current_price: float, position: float, hedge_ratio: float,
                     strategy: str, strike_price: Optional[float] = None) -> Dict[str, float]:
    """
    Get summary statistics for a hedging strategy.
    """
    
    payoff_data = compute_payoff_diagram(current_price, position, hedge_ratio, 
                                       strategy, strike_price)
    
    net_pnl = payoff_data['net_pnl']
    
    summary = {
        'hedge_ratio': hedge_ratio,
        'position_size': position,
        'current_price': current_price,
        'max_profit': float(net_pnl.max()),
        'max_loss': float(net_pnl.min()),
        'profit_range': float(net_pnl.max() - net_pnl.min()),
        'num_breakevens': len(payoff_data['breakeven_prices'])
    }
    
    if strategy.lower() == "options" and strike_price is not None:
        summary['strike_price'] = strike_price
        summary['moneyness'] = current_price / strike_price
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    # Test the strategies module
    print("Testing hedging/strategies.py module...")
    
    current_price = 75.0  # Example WTI price
    position = 1000.0     # 1000 barrels
    hedge_ratio = 0.8     # 80% hedge
    strike_price = 75.0   # ATM strike
    
    print("\n=== Futures Hedge Test ===")
    futures_payoff = compute_payoff_diagram(current_price, position, hedge_ratio, "Futures")
    print(f"Price range: ${futures_payoff['spot_prices'].min():.2f} - ${futures_payoff['spot_prices'].max():.2f}")
    print(f"Max profit: ${futures_payoff['net_pnl'].max():.2f}")
    print(f"Max loss: ${futures_payoff['net_pnl'].min():.2f}")
    print(f"Breakeven prices: {[f'${p:.2f}' for p in futures_payoff['breakeven_prices']]}")
    
    print("\n=== Options Hedge Test ===")
    options_payoff = compute_payoff_diagram(current_price, position, hedge_ratio, "Options", strike_price)
    print(f"Price range: ${options_payoff['spot_prices'].min():.2f} - ${options_payoff['spot_prices'].max():.2f}")
    print(f"Max profit: ${options_payoff['net_pnl'].max():.2f}")
    print(f"Max loss: ${options_payoff['net_pnl'].min():.2f}")
    print(f"Breakeven prices: {[f'${p:.2f}' for p in options_payoff['breakeven_prices']]}")
    
    print("\n=== Hedge Summaries ===")
    futures_summary = get_hedge_summary(current_price, position, hedge_ratio, "Futures")
    options_summary = get_hedge_summary(current_price, position, hedge_ratio, "Options", strike_price)
    
    print(f"Futures hedge summary: {futures_summary}")
    print(f"Options hedge summary: {options_summary}")
