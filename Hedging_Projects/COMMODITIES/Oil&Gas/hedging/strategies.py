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
    """
    Calculate P&L for futures hedge strategy.
    
    Args:
        prices (pd.Series): Historical commodity prices
        position (float): Position size (positive for long, negative for short)
        hedge_ratio (float): Hedge ratio between 0.0 and 1.0
    
    Returns:
        pd.Series: P&L series for futures hedge
    """
    
    # Convert inputs to float to ensure compatibility
    position = float(position)
    hedge_ratio = float(hedge_ratio)
    
    # Calculate price changes
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
    
    Args:
        prices (pd.Series): Historical commodity prices
        position (float): Position size (positive for long, negative for short)
        hedge_ratio (float): Hedge ratio between 0.0 and 1.0
        strike_price (float): Strike price of the option
        option_type (str): "put" or "call" (default: "put" for long position hedge)
        time_to_expiry (float): Time to expiry in years (default: 0.25 = 3 months)
        risk_free_rate (float): Risk-free rate (default: 0.05 = 5%)
        volatility (Optional[float]): Implied volatility (if None, estimated from price data)
    
    Returns:
        pd.Series: P&L series for options hedge
    """
    
    # Convert inputs to float to ensure compatibility
    position = float(position)
    hedge_ratio = float(hedge_ratio)
    strike_price = float(strike_price)
    time_to_expiry = float(time_to_expiry)
    risk_free_rate = float(risk_free_rate)
    
    # Estimate volatility if not provided
    if volatility is None:
        returns = prices.pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252))  # Annualized volatility
    else:
        volatility = float(volatility)
    
    # Calculate option delta for each price point
    deltas = []
    for price in prices:
        delta = calculate_option_delta(float(price), strike_price, time_to_expiry, 
                                     risk_free_rate, volatility, option_type)
        deltas.append(delta)
    
    deltas = pd.Series(deltas, index=prices.index)
    
    # Calculate price changes
    price_changes = prices.diff().dropna()
    deltas_aligned = deltas[price_changes.index]
    
    # Options hedge P&L = ΔP × position × hedge_ratio × delta
    # For a long position, we buy puts (protective hedge)
    hedge_pnl = price_changes * position * hedge_ratio * deltas_aligned
    
    return hedge_pnl


def calculate_option_delta(spot_price: float, strike_price: float, time_to_expiry: float,
                          risk_free_rate: float, volatility: float, option_type: str) -> float:
    """
    Calculate option delta using Black-Scholes formula.
    
    Args:
        spot_price (float): Current spot price
        strike_price (float): Strike price
        time_to_expiry (float): Time to expiry in years
        risk_free_rate (float): Risk-free rate
        volatility (float): Volatility
        option_type (str): "call" or "put"
    
    Returns:
        float: Option delta
    """
    
    if time_to_expiry <= 0:
        # At expiry, delta is 0 or 1
        if option_type.lower() == "call":
            return 1.0 if spot_price > strike_price else 0.0
        else:  # put
            return -1.0 if spot_price < strike_price else 0.0
    
    # Black-Scholes delta calculation
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
    
    Args:
        current_price (float): Current spot price
        position (float): Position size
        hedge_ratio (float): Hedge ratio between 0.0 and 1.0
        strategy (str): "Futures" or "Options"
        strike_price (Optional[float]): Strike price for options (default: current_price)
        price_range_pct (float): Price range as percentage of current price (default: 0.3 = ±30%)
        num_points (int): Number of price points to calculate (default: 100)
    
    Returns:
        Dict[str, pd.Series]: Dictionary containing:
            - 'spot_prices': Array of spot prices
            - 'underlying_pnl': Underlying position P&L
            - 'hedge_pnl': Hedge position P&L
            - 'net_pnl': Net P&L (underlying + hedge)
            - 'breakeven_prices': Breakeven price points
    """
    
    # Set default strike price to current price (ATM)
    if strike_price is None:
        strike_price = float(current_price)
    else:
        strike_price = float(strike_price)
    
    # Ensure current_price is float
    current_price = float(current_price)
    
    # Generate price range (±price_range_pct around current price)
    price_min = current_price * (1 - price_range_pct)
    price_max = current_price * (1 + price_range_pct)
    spot_prices = np.linspace(price_min, price_max, num_points)
    
    # Calculate underlying position P&L
    # P&L = (S - S0) × position
    underlying_pnl = (spot_prices - current_price) * float(position)
    
    # Calculate hedge P&L based on strategy
    if strategy.lower() == "futures":
        # Futures hedge: linear payoff
        # For long position, we sell futures (short hedge)
        # Hedge P&L = -(S - S0) × position × hedge_ratio
        hedge_pnl = -(spot_prices - current_price) * float(position) * float(hedge_ratio)
        
    elif strategy.lower() == "options":
        # Options hedge: non-linear payoff
        hedge_pnl = np.zeros_like(spot_prices)
        
        for i, spot_price in enumerate(spot_prices):
            if float(position) > 0:  # Long underlying position - buy puts
                # Put option payoff: max(K - S, 0)
                option_payoff = max(strike_price - spot_price, 0)
            else:  # Short underlying position - buy calls
                # Call option payoff: max(S - K, 0)
                option_payoff = max(spot_price - strike_price, 0)
            
            # Scale by position and hedge ratio
            hedge_pnl[i] = option_payoff * abs(float(position)) * float(hedge_ratio)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Supported: 'Futures', 'Options'")
    
    # Calculate net P&L
    net_pnl = underlying_pnl + hedge_pnl
    
    # Find breakeven prices (where net P&L = 0)
    breakeven_prices = find_breakeven_prices(spot_prices, net_pnl)
    
    # Create result dictionary
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
    
    Args:
        prices (np.ndarray): Array of prices
        pnl (np.ndarray): Array of P&L values
        tolerance (float): Tolerance for zero crossing detection
    
    Returns:
        list: List of breakeven prices
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
    
    Args:
        current_price (float): Current spot price
        position (float): Position size
        hedge_ratio (float): Hedge ratio
        strategy (str): "Futures" or "Options"
        strike_price (Optional[float]): Strike price for options
    
    Returns:
        Dict[str, float]: Summary statistics
    """
    
    # Calculate payoff diagram
    payoff_data = compute_payoff_diagram(current_price, position, hedge_ratio, 
                                       strategy, strike_price)
    
    # Calculate summary statistics
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
    """
    Simplified crack spread simulation for hedging analysis.
    
    Args:
        refinery_capacity: Refinery capacity in barrels per day
        hedge_ratio: Hedge ratio between 0.0 and 1.0
    
    Returns:
        Dict with simulated crack spread P&L
    """
    
    # Simplified crack spread simulation (placeholder for now)
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
    
    # Hedge P&L (selling product futures, buying crude futures)
    hedge_pnl = -spread_changes_series * (daily_capacity / 3) * hedge_ratio
    
    # Net hedged P&L
    hedged_pnl = unhedged_pnl + hedge_pnl
    
    return {
        'unhedged_pnl': unhedged_pnl,
        'hedge_pnl': hedge_pnl,
        'hedged_pnl': hedged_pnl,
        'crack_spreads': crack_spread_series
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the strategies module
    print("Testing hedging/strategies.py module...")
    
    # Test parameters
    current_price = 75.0  # Example WTI price
    position = 1000.0     # 1000 barrels
    hedge_ratio = 0.8     # 80% hedge
    strike_price = 75.0   # ATM strike
    
    # Test futures hedge payoff diagram
    print("\n=== Futures Hedge Test ===")
    futures_payoff = compute_payoff_diagram(current_price, position, hedge_ratio, "Futures")
    print(f"Price range: ${futures_payoff['spot_prices'].min():.2f} - ${futures_payoff['spot_prices'].max():.2f}")
    print(f"Max profit: ${futures_payoff['net_pnl'].max():.2f}")
    print(f"Max loss: ${futures_payoff['net_pnl'].min():.2f}")
    print(f"Breakeven prices: {[f'${p:.2f}' for p in futures_payoff['breakeven_prices']]}")
    
    # Test options hedge payoff diagram
    print("\n=== Options Hedge Test ===")
    options_payoff = compute_payoff_diagram(current_price, position, hedge_ratio, "Options", strike_price)
    print(f"Price range: ${options_payoff['spot_prices'].min():.2f} - ${options_payoff['spot_prices'].max():.2f}")
    print(f"Max profit: ${options_payoff['net_pnl'].max():.2f}")
    print(f"Max loss: ${options_payoff['net_pnl'].min():.2f}")
    print(f"Breakeven prices: {[f'${p:.2f}' for p in options_payoff['breakeven_prices']]}")
    
    # Test hedge summaries
    print("\n=== Hedge Summaries ===")
    futures_summary = get_hedge_summary(current_price, position, hedge_ratio, "Futures")
    options_summary = get_hedge_summary(current_price, position, hedge_ratio, "Options", strike_price)
    
    print(f"Futures hedge summary: {futures_summary}")
    print(f"Options hedge summary: {options_summary}")
