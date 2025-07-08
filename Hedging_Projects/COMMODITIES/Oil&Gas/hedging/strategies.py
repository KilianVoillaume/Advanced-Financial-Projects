"""
hedging/strategies.py

Module for hedge payoff calculations for different hedging strategies.
Supports Futures, Options, and Crack Spread hedging with payoff diagram generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import norm


def compute_futures_hedge(prices: pd.Series, position: float, hedge_ratio: float) -> pd.Series:
    """ Calculate P&L for futures hedge strategy """
    
    # Convert inputs to float to ensure compatibility
    position = float(position)
    hedge_ratio = float(hedge_ratio)
    
    price_changes = prices.diff().dropna()
    hedge_pnl = -price_changes * position * hedge_ratio
    
    return hedge_pnl


def compute_options_hedge(prices: pd.Series, position: float, hedge_ratio: float, strike_price: float, option_type: str = "put",
                         time_to_expiry: float = 0.25, risk_free_rate: float = 0.05, volatility: Optional[float] = None) -> pd.Series:

    
    # Convert inputs to float to ensure compatibility
    position = float(position)
    hedge_ratio = float(hedge_ratio)
    strike_price = float(strike_price)
    time_to_expiry = float(time_to_expiry)
    risk_free_rate = float(risk_free_rate)
    
    if volatility is None:
        returns = prices.pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252))  # Annualized volatility
    else:
        volatility = float(volatility)
    
    deltas = []
    for price in prices:
        delta = calculate_option_delta(float(price), strike_price, time_to_expiry, 
                                     risk_free_rate, volatility, option_type)
        deltas.append(delta)
    deltas = pd.Series(deltas, index=prices.index)
    
    price_changes = prices.diff().dropna()
    deltas_aligned = deltas[price_changes.index]
    hedge_pnl = price_changes * position * hedge_ratio * deltas_aligned
    
    return hedge_pnl


def calculate_option_delta(spot_price: float, strike_price: float, time_to_expiry: float,
                          risk_free_rate: float, volatility: float, option_type: str) -> float:
    
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


def compute_payoff_diagram(current_price: float, position: float, hedge_ratio: float, strategy: str, strike_price: Optional[float] = None,
                          price_range_pct: float = 0.3, num_points: int = 100) -> Dict[str, pd.Series]:

    # Set default strike price to current price (ATM)
    if strike_price is None:
        strike_price = float(current_price)
    else:
        strike_price = float(strike_price)
    
    current_price = float(current_price)
    
    price_min = current_price * (1 - price_range_pct)
    price_max = current_price * (1 + price_range_pct)
    spot_prices = np.linspace(price_min, price_max, num_points)
    underlying_pnl = (spot_prices - current_price) * float(position)
    
    if strategy.lower() == "futures":
        hedge_pnl = -(spot_prices - current_price) * float(position) * float(hedge_ratio)
        
    elif strategy.lower() == "options":
        hedge_pnl = np.zeros_like(spot_prices)
        
        for i, spot_price in enumerate(spot_prices):
            if float(position) > 0:  # Long underlying position - buy puts
                option_payoff = max(strike_price - spot_price, 0)
            else:  # Short underlying position - buy calls
                option_payoff = max(spot_price - strike_price, 0)
            
            # Scale by position and hedge ratio
            hedge_pnl[i] = option_payoff * abs(float(position)) * float(hedge_ratio)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Supported: 'Futures', 'Options'")
    
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
    
    payoff_data = compute_payoff_diagram(current_price, position, hedge_ratio, strategy, strike_price)
                         
    net_pnl = payoff_data['net_pnl']
    
    summary = {
        'hedge_ratio': float(hedge_ratio),
        'position_size': float(position),
        'current_price': float(current_price),
        'max_profit': float(net_pnl.max()),
        'max_loss': float(net_pnl.min()),
        'profit_range': float(net_pnl.max() - net_pnl.min()),
        'num_breakevens': len(payoff_data['breakeven_prices'])
    }
    
    if strategy.lower() == "options" and strike_price is not None:
        summary['strike_price'] = float(strike_price)
        summary['moneyness'] = float(current_price) / float(strike_price)
    
    return summary


def compute_crack_spread_simulation(refinery_capacity: float, hedge_ratio: float) -> Dict[str, pd.Series]:
    """ Simplified crack spread simulation for hedging analysis """

    # Generate synthetic crack spread data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    
    # Simulate crack spread changes (typically $10-25/barrel with high volatility)
    base_spread = 15.0  # $15/barrel base crack spread
    spread_changes = np.random.normal(0, 2.0, 252)  # $2/barrel daily std dev
    crack_spreads = base_spread + np.cumsum(spread_changes * 0.1)  # Mean reverting
    
    crack_spread_series = pd.Series(crack_spreads, index=dates)
    spread_changes_series = crack_spread_series.diff().dropna()
    
    # Calculate refinery P&L (3-2-1 crack spread)
    daily_capacity = refinery_capacity
    unhedged_pnl = spread_changes_series * (daily_capacity / 3)  # 3 barrels crude per unit
    
    hedge_pnl = -spread_changes_series * (daily_capacity / 3) * hedge_ratio
    hedged_pnl = unhedged_pnl + hedge_pnl
    
    return {
        'unhedged_pnl': unhedged_pnl,
        'hedge_pnl': hedge_pnl,
        'hedged_pnl': hedged_pnl,
        'crack_spreads': crack_spread_series
    }


if __name__ == "__main__":
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
