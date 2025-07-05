"""
hedging/simulation.py

P&L simulation engine for hedged and unhedged positions.
Generates Monte Carlo simulations based on historical price data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from .strategies import compute_futures_hedge, compute_options_hedge


def simulate_pnl(pnl_series: pd.Series, n_sim: int = 1000, simulation_method: str = "normal") -> np.ndarray:
    """
    Simulate P&L outcomes using historical P&L series.
    
    Args:
        pnl_series (pd.Series): Historical P&L series
        n_sim (int): Number of simulations to run (default: 1000)
        simulation_method (str): Method for simulation ("normal", "bootstrap", "t_dist")
    
    Returns:
        np.ndarray: Array of simulated P&L outcomes
    """
    
    if pnl_series.empty:
        raise ValueError("P&L series cannot be empty")
    
    # Remove any NaN values
    pnl_clean = pnl_series.dropna()
    
    if len(pnl_clean) < 2:
        raise ValueError("Need at least 2 data points for simulation")
    
    # Generate simulations based on method
    if simulation_method.lower() == "normal":
        simulated_pnl = _simulate_normal_distribution(pnl_clean, n_sim)
    elif simulation_method.lower() == "bootstrap":
        simulated_pnl = _simulate_bootstrap(pnl_clean, n_sim)
    elif simulation_method.lower() == "t_dist":
        simulated_pnl = _simulate_t_distribution(pnl_clean, n_sim)
    else:
        raise ValueError(f"Unknown simulation method: {simulation_method}")
    
    return simulated_pnl


def _simulate_normal_distribution(pnl_series: pd.Series, n_sim: int) -> np.ndarray:
    """
    Simulate P&L using normal distribution fitted to historical data.
    
    Args:
        pnl_series (pd.Series): Historical P&L series
        n_sim (int): Number of simulations
    
    Returns:
        np.ndarray: Simulated P&L outcomes
    """
    
    # Calculate mean and standard deviation
    pnl_mean = pnl_series.mean()
    pnl_std = pnl_series.std()
    
    # Generate random samples from normal distribution
    simulated_pnl = np.random.normal(pnl_mean, pnl_std, n_sim)
    
    return simulated_pnl


def _simulate_bootstrap(pnl_series: pd.Series, n_sim: int) -> np.ndarray:
    """
    Simulate P&L using bootstrap resampling of historical data.
    
    Args:
        pnl_series (pd.Series): Historical P&L series
        n_sim (int): Number of simulations
    
    Returns:
        np.ndarray: Simulated P&L outcomes
    """
    
    # Bootstrap resampling with replacement
    simulated_pnl = np.random.choice(pnl_series.values, size=n_sim, replace=True)
    
    return simulated_pnl


def _simulate_t_distribution(pnl_series: pd.Series, n_sim: int) -> np.ndarray:
    """
    Simulate P&L using t-distribution fitted to historical data.
    
    Args:
        pnl_series (pd.Series): Historical P&L series
        n_sim (int): Number of simulations
    
    Returns:
        np.ndarray: Simulated P&L outcomes
    """
    
    # Fit t-distribution to data
    params = stats.t.fit(pnl_series.values)
    df, loc, scale = params
    
    # Generate random samples from fitted t-distribution
    simulated_pnl = stats.t.rvs(df=df, loc=loc, scale=scale, size=n_sim)
    
    return simulated_pnl


def simulate_hedged_vs_unhedged(prices: pd.Series, position: float, hedge_ratio: float,
                               strategy: str, strike_price: Optional[float] = None,
                               n_sim: int = 1000, simulation_method: str = "normal") -> Dict[str, np.ndarray]:
    """
    Simulate P&L for both hedged and unhedged positions.
    
    Args:
        prices (pd.Series): Historical commodity prices
        position (float): Position size
        hedge_ratio (float): Hedge ratio between 0.0 and 1.0
        strategy (str): "Futures", "Options", or "Crack Spread (3-2-1)"
        strike_price (Optional[float]): Strike price for options
        n_sim (int): Number of simulations (default: 1000)
        simulation_method (str): Simulation method (default: "normal")
    
    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'unhedged_pnl': Unhedged position simulations
            - 'hedged_pnl': Hedged position simulations
            - 'hedge_benefit': Difference between hedged and unhedged
    """
    
    # Convert inputs to float to ensure compatibility
    position = float(position)
    hedge_ratio = float(hedge_ratio)
    if strike_price is not None:
        strike_price = float(strike_price)
    
    # Handle crack spread separately since it uses different simulation logic
    if strategy.lower() == "crack spread (3-2-1)":
        # For crack spreads, we need refinery capacity
        # Use position as a proxy for daily capacity (in barrels)
        refinery_capacity = abs(position) * 100  # Scale position to reasonable refinery size
        
        # Use the crack spread simulation function
        crack_results = compute_crack_spread_simulation(refinery_capacity, hedge_ratio)
        
        # Convert to simulation format expected by the rest of the app
        unhedged_pnl_series = crack_results['unhedged_pnl']
        hedged_pnl_series = crack_results['hedged_pnl']
        
        # Generate simulations from the historical crack spread data
        unhedged_sim = simulate_pnl(unhedged_pnl_series, n_sim, simulation_method)
        hedged_sim = simulate_pnl(hedged_pnl_series, n_sim, simulation_method)
        
        # Calculate hedge benefit
        hedge_benefit = hedged_sim - unhedged_sim
        
        return {
            'unhedged_pnl': unhedged_sim,
            'hedged_pnl': hedged_sim,
            'hedge_benefit': hedge_benefit
        }
    
    # Original logic for Futures and Options
    # Calculate historical unhedged P&L
    price_changes = prices.diff().dropna()
    unhedged_pnl = price_changes * position
    
    # Calculate historical hedge P&L based on strategy
    if strategy.lower() == "futures":
        hedge_pnl = compute_futures_hedge(prices, position, hedge_ratio)
    elif strategy.lower() == "options":
        if strike_price is None:
            strike_price = float(prices.iloc[-1])  # Use last price as default
        hedge_pnl = compute_options_hedge(prices, position, hedge_ratio, strike_price)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Supported strategies: 'Futures', 'Options', 'Crack Spread (3-2-1)'")
    
    # Align hedge P&L with unhedged P&L (same time periods)
    hedge_pnl_aligned = hedge_pnl.reindex(unhedged_pnl.index, fill_value=0)
    
    # Calculate net hedged P&L
    hedged_pnl = unhedged_pnl + hedge_pnl_aligned
    
    # Simulate both hedged and unhedged scenarios
    unhedged_sim = simulate_pnl(unhedged_pnl, n_sim, simulation_method)
    hedged_sim = simulate_pnl(hedged_pnl, n_sim, simulation_method)
    
    # Calculate hedge benefit
    hedge_benefit = hedged_sim - unhedged_sim
    
    return {
        'unhedged_pnl': unhedged_sim,
        'hedged_pnl': hedged_sim,
        'hedge_benefit': hedge_benefit
    }


def simulate_price_scenarios(current_price: float, volatility: float, time_horizon: float,
                           n_sim: int = 1000, n_steps: int = 252) -> np.ndarray:
    """
    Simulate future price paths using Geometric Brownian Motion.
    
    Args:
        current_price (float): Current spot price
        volatility (float): Annualized volatility
        time_horizon (float): Time horizon in years
        n_sim (int): Number of price path simulations (default: 1000)
        n_steps (int): Number of time steps (default: 252 trading days)
    
    Returns:
        np.ndarray: Array of simulated final prices (shape: n_sim)
    """
    
    # Time step
    dt = time_horizon / n_steps
    
    # Drift (assume zero for simplicity in commodity markets)
    mu = 0.0
    
    # Generate random shocks
    random_shocks = np.random.normal(0, 1, (n_sim, n_steps))
    
    # Initialize price paths
    price_paths = np.zeros((n_sim, n_steps + 1))
    price_paths[:, 0] = current_price
    
    # Simulate price paths using GBM
    for t in range(n_steps):
        price_paths[:, t + 1] = price_paths[:, t] * np.exp(
            (mu - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shocks[:, t]
        )
    
    # Return final prices
    final_prices = price_paths[:, -1]
    
    return final_prices


def calculate_simulation_statistics(simulated_pnl: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for simulated P&L outcomes.
    
    Args:
        simulated_pnl (np.ndarray): Array of simulated P&L values
    
    Returns:
        Dict[str, float]: Dictionary of statistics
    """
    
    if len(simulated_pnl) == 0:
        raise ValueError("Simulated P&L array cannot be empty")
    
    # Calculate basic statistics
    stats_dict = {
        'mean': float(np.mean(simulated_pnl)),
        'std': float(np.std(simulated_pnl)),
        'min': float(np.min(simulated_pnl)),
        'max': float(np.max(simulated_pnl)),
        'median': float(np.median(simulated_pnl)),
        'skewness': float(stats.skew(simulated_pnl)),
        'kurtosis': float(stats.kurtosis(simulated_pnl)),
        'percentile_5': float(np.percentile(simulated_pnl, 5)),
        'percentile_25': float(np.percentile(simulated_pnl, 25)),
        'percentile_75': float(np.percentile(simulated_pnl, 75)),
        'percentile_95': float(np.percentile(simulated_pnl, 95))
    }
    
    # Calculate probability of profit/loss
    stats_dict['prob_profit'] = float(np.sum(simulated_pnl > 0) / len(simulated_pnl))
    stats_dict['prob_loss'] = float(np.sum(simulated_pnl < 0) / len(simulated_pnl))
    
    return stats_dict


def compare_hedging_effectiveness(hedged_sim: np.ndarray, unhedged_sim: np.ndarray) -> Dict[str, float]:
    """
    Compare effectiveness of hedging strategy vs unhedged position.
    
    Args:
        hedged_sim (np.ndarray): Simulated hedged P&L
        unhedged_sim (np.ndarray): Simulated unhedged P&L
    
    Returns:
        Dict[str, float]: Comparison metrics
    """
    
    # Calculate statistics for both
    hedged_stats = calculate_simulation_statistics(hedged_sim)
    unhedged_stats = calculate_simulation_statistics(unhedged_sim)
    
    # Calculate Sharpe ratios
    hedged_sharpe = hedged_stats['mean'] / hedged_stats['std'] if hedged_stats['std'] > 0 else 0
    unhedged_sharpe = unhedged_stats['mean'] / unhedged_stats['std'] if unhedged_stats['std'] > 0 else 0
    
    # Calculate comparison metrics
    comparison = {
        'hedged_mean': hedged_stats['mean'],
        'unhedged_mean': unhedged_stats['mean'],
        'mean_difference': hedged_stats['mean'] - unhedged_stats['mean'],
        'hedged_std': hedged_stats['std'],
        'unhedged_std': unhedged_stats['std'],
        'volatility_reduction': (unhedged_stats['std'] - hedged_stats['std']) / unhedged_stats['std'],
        'hedged_sharpe': hedged_sharpe,
        'unhedged_sharpe': unhedged_sharpe,
        'sharpe_improvement': hedged_sharpe - unhedged_sharpe,
        'hedged_prob_loss': hedged_stats['prob_loss'],
        'unhedged_prob_loss': unhedged_stats['prob_loss'],
        'loss_prob_reduction': unhedged_stats['prob_loss'] - hedged_stats['prob_loss']
    }
    
    return comparison


# Example usage and testing
if __name__ == "__main__":
    # Test the simulation module
    print("Testing hedging/simulation.py module...")
    
    # Create sample price data
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    initial_price = 75.0
    returns = np.random.normal(0.0002, 0.02, 252)  # Small daily returns with 2% daily vol
    prices = pd.Series(initial_price * np.exp(np.cumsum(returns)), index=dates)
    
    # Test parameters
    position = 1000.0     # 1000 barrels
    hedge_ratio = 0.8     # 80% hedge
    n_sim = 1000         # 1000 simulations
    
    print(f"Sample price data: {len(prices)} days")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"Final price: ${prices.iloc[-1]:.2f}")
    
    # Test basic P&L simulation
    print("\n=== Basic P&L Simulation Test ===")
    price_changes = prices.diff().dropna()
    unhedged_pnl = price_changes * position
    
    simulated_unhedged = simulate_pnl(unhedged_pnl, n_sim)
    unhedged_stats = calculate_simulation_statistics(simulated_unhedged)
    
    print(f"Unhedged simulation stats:")
    print(f"  Mean P&L: ${unhedged_stats['mean']:.2f}")
    print(f"  Std Dev: ${unhedged_stats['std']:.2f}")
    print(f"  Prob of Loss: {unhedged_stats['prob_loss']:.2%}")
    
    # Test hedged vs unhedged comparison
    print("\n=== Hedged vs Unhedged Comparison ===")
    hedge_comparison = simulate_hedged_vs_unhedged(prices, position, hedge_ratio, "Futures", n_sim=n_sim)
    
    effectiveness = compare_hedging_effectiveness(hedge_comparison['hedged_pnl'], 
                                                hedge_comparison['unhedged_pnl'])
    
    print(f"Hedging effectiveness:")
    print(f"  Volatility reduction: {effectiveness['volatility_reduction']:.2%}")
    print(f"  Loss probability reduction: {effectiveness['loss_prob_reduction']:.2%}")
    print(f"  Mean P&L change: ${effectiveness['mean_difference']:.2f}")
    
    # Test price scenario simulation
    print("\n=== Price Scenario Simulation ===")
    current_price = prices.iloc[-1]
    volatility = prices.pct_change().std() * np.sqrt(252)  # Annualized vol
    
    future_prices = simulate_price_scenarios(current_price, volatility, 0.25, n_sim=n_sim)  # 3 months
    
    print(f"Future price simulation (3 months):")
    print(f"  Current price: ${current_price:.2f}")
    print(f"  Volatility: {volatility:.2%}")
    print(f"  Mean future price: ${np.mean(future_prices):.2f}")
    print(f"  Price range (95% conf): ${np.percentile(future_prices, 2.5):.2f} - ${np.percentile(future_prices, 97.5):.2f}")
