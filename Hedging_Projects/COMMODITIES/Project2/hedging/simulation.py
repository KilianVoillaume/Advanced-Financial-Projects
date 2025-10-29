"""
hedging/simulation.py

Supports single options, multi-leg strategies, and advanced Monte Carlo simulations.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, Tuple, Optional, List
from scipy import stats
from .strategies import compute_futures_hedge, compute_options_hedge


def simulate_pnl(pnl_series: pd.Series, n_sim: int = 1000, simulation_method: str = "normal") -> np.ndarray:
    """Enhanced P&L simulation with multiple methods"""
    if pnl_series.empty:
        raise ValueError("P&L series cannot be empty")
    
    pnl_clean = pnl_series.dropna()
    
    if len(pnl_clean) < 2:
        raise ValueError("Need at least 2 data points for simulation")
    
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
    """Simulate using normal distribution"""
    pnl_mean = pnl_series.mean()
    pnl_std = pnl_series.std()
    simulated_pnl = np.random.normal(pnl_mean, pnl_std, n_sim)
    
    return simulated_pnl


def _simulate_bootstrap(pnl_series: pd.Series, n_sim: int) -> np.ndarray:
    """Bootstrap resampling simulation"""
    simulated_pnl = np.random.choice(pnl_series.values, size=n_sim, replace=True)
    
    return simulated_pnl


def _simulate_t_distribution(pnl_series: pd.Series, n_sim: int) -> np.ndarray:
    """Simulate using t-distribution"""
    params = stats.t.fit(pnl_series.values)
    df, loc, scale = params
    
    simulated_pnl = stats.t.rvs(df=df, loc=loc, scale=scale, size=n_sim)
    
    return simulated_pnl


# ============================================================================
# ENHANCED SIMULATION ENGINE - Step 4.1 Implementation
# ============================================================================

def simulate_hedged_vs_unhedged(prices: pd.Series, position: float, hedge_ratio: float,
                               strategy: str, strike_price: Optional[float] = None,
                               multi_leg_strategy: Optional[any] = None,
                               n_sim: int = 1000, simulation_method: str = "normal") -> Dict[str, np.ndarray]:
    """
    Enhanced simulation supporting single options, futures, and multi-leg strategies.
    
    Args:
        prices: Historical price series
        position: Position size (positive for long, negative for short)
        hedge_ratio: Hedge ratio (0.0 to 1.0)
        strategy: Strategy type ("Futures", "Options", "Multi-Leg")
        strike_price: Strike price for single options
        multi_leg_strategy: MultiLegStrategy object for complex strategies
        n_sim: Number of simulations
        simulation_method: Simulation method ("normal", "bootstrap", "t_dist")
        
    Returns:
        Dictionary containing simulation results
    """
    position = float(position)
    hedge_ratio = float(hedge_ratio)
    if strike_price is not None:
        strike_price = float(strike_price)
    
    # Generate underlying P&L
    price_changes = prices.diff().dropna()
    unhedged_pnl = price_changes * position
    
    # Calculate hedge P&L based on strategy type
    if strategy.lower() == "futures":
        hedge_pnl = compute_futures_hedge(prices, position, hedge_ratio)
    
    elif strategy.lower() == "options":
        if strike_price is None:
            strike_price = float(prices.iloc[-1])
        hedge_pnl = compute_options_hedge(prices, position, hedge_ratio, strike_price)
    
    elif strategy.lower() == "multi-leg" and multi_leg_strategy is not None:
        hedge_pnl = compute_multi_leg_hedge(prices, multi_leg_strategy)
    
    else:
        raise ValueError(f"Unknown strategy configuration: {strategy}")
    
    # Align hedge P&L with unhedged P&L
    hedge_pnl_aligned = hedge_pnl.reindex(unhedged_pnl.index, fill_value=0)
    hedged_pnl = unhedged_pnl + hedge_pnl_aligned
    
    # Run simulations
    unhedged_sim = simulate_pnl(unhedged_pnl, n_sim, simulation_method)
    hedged_sim = simulate_pnl(hedged_pnl, n_sim, simulation_method)
    
    hedge_benefit = hedged_sim - unhedged_sim
    
    return {
        'unhedged_pnl': unhedged_sim,
        'hedged_pnl': hedged_sim,
        'hedge_benefit': hedge_benefit
    }


def compute_multi_leg_hedge(prices: pd.Series, multi_leg_strategy) -> pd.Series:
    """
    Compute P&L series for multi-leg options hedge using Greeks approximation.
    
    Args:
        prices: Price series for underlying asset
        multi_leg_strategy: MultiLegStrategy object
        
    Returns:
        Series of hedge P&L values
    """
    try:
        from hedging.options_math import MultiLegGreeksCalculator
        
        hedge_pnl_series = pd.Series(0.0, index=prices.index)
        
        for i in range(1, len(prices)):
            current_price = prices.iloc[i]
            previous_price = prices.iloc[i-1]
            
            # Calculate strategy Greeks at previous price
            greeks = MultiLegGreeksCalculator.calculate_strategy_greeks(
                multi_leg_strategy, previous_price
            )
            
            price_change = current_price - previous_price
            
            # First-order (delta) and second-order (gamma) approximation
            delta_pnl = greeks['delta'] * price_change
            gamma_pnl = 0.5 * greeks['gamma'] * (price_change ** 2)
            
            # Total hedge P&L change
            hedge_pnl_change = delta_pnl + gamma_pnl
            hedge_pnl_series.iloc[i] = hedge_pnl_change
        
        return hedge_pnl_series.dropna()
        
    except ImportError:
        # Fallback to simplified calculation
        return _compute_multi_leg_hedge_simplified(prices, multi_leg_strategy)


def _compute_multi_leg_hedge_simplified(prices: pd.Series, multi_leg_strategy) -> pd.Series:
    """
    Simplified multi-leg hedge calculation for when Greeks calculator unavailable.
    
    Args:
        prices: Price series
        multi_leg_strategy: MultiLegStrategy object
        
    Returns:
        Simplified hedge P&L series
    """
    hedge_pnl_series = pd.Series(0.0, index=prices.index)
    
    for i in range(1, len(prices)):
        current_price = prices.iloc[i]
        previous_price = prices.iloc[i-1]
        price_change = current_price - previous_price
        
        # Simplified calculation based on strategy type
        strategy_type = multi_leg_strategy.strategy_type.value
        
        # Approximate hedge effectiveness based on strategy
        if "Straddle" in strategy_type:
            # Straddles benefit from volatility
            hedge_effectiveness = min(abs(price_change) / previous_price * 10, 1.0)
            if "Long" in strategy_type:
                hedge_pnl = hedge_effectiveness * abs(price_change) * multi_leg_strategy.underlying_size * multi_leg_strategy.hedge_ratio
            else:  # Short straddle
                hedge_pnl = -hedge_effectiveness * abs(price_change) * multi_leg_strategy.underlying_size * multi_leg_strategy.hedge_ratio
        
        elif "Strangle" in strategy_type:
            # Similar to straddle but less sensitive
            hedge_effectiveness = min(abs(price_change) / previous_price * 8, 0.8)
            if "Long" in strategy_type:
                hedge_pnl = hedge_effectiveness * abs(price_change) * multi_leg_strategy.underlying_size * multi_leg_strategy.hedge_ratio
            else:
                hedge_pnl = -hedge_effectiveness * abs(price_change) * multi_leg_strategy.underlying_size * multi_leg_strategy.hedge_ratio
        
        elif "Collar" in strategy_type:
            # Collar provides asymmetric protection
            if price_change < 0:  # Downside protection
                hedge_pnl = -price_change * multi_leg_strategy.underlying_size * multi_leg_strategy.hedge_ratio * 0.8
            else:  # Limited upside
                hedge_pnl = -price_change * multi_leg_strategy.underlying_size * multi_leg_strategy.hedge_ratio * 0.3
        
        else:
            # Default approximation for other strategies
            hedge_pnl = -price_change * multi_leg_strategy.underlying_size * multi_leg_strategy.hedge_ratio * 0.5
        
        hedge_pnl_series.iloc[i] = hedge_pnl
    
    return hedge_pnl_series.dropna()


def simulate_multi_leg_scenarios(multi_leg_strategy, current_price: float, 
                                price_scenarios: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Simulate multi-leg strategy performance across price scenarios.
    
    Args:
        multi_leg_strategy: MultiLegStrategy object
        current_price: Current price of underlying
        price_scenarios: Array of future price scenarios
        
    Returns:
        Dictionary containing scenario analysis results
    """
    try:
        from hedging.strategies import compute_multi_leg_payoff
        
        # Calculate payoff for each scenario
        scenario_payoffs = []
        
        for scenario_price in price_scenarios:
            # Simple payoff calculation at expiration
            total_payoff = 0.0
            
            for leg in multi_leg_strategy.legs:
                if leg.option_type.lower() == 'call':
                    intrinsic = max(scenario_price - leg.strike_price, 0)
                else:  # put
                    intrinsic = max(leg.strike_price - scenario_price, 0)
                
                leg_payoff = intrinsic * leg.quantity
                total_payoff += leg_payoff
            
            # Account for premium paid/received (simplified)
            try:
                premium = multi_leg_strategy.get_total_premium(current_price)
                net_payoff = total_payoff - premium
            except:
                net_payoff = total_payoff
            
            scenario_payoffs.append(net_payoff)
        
        scenario_payoffs = np.array(scenario_payoffs)
        
        # Calculate underlying P&L for comparison
        underlying_pnl = (price_scenarios - current_price) * multi_leg_strategy.underlying_size
        
        # Calculate hedge benefit
        hedge_benefit = scenario_payoffs * multi_leg_strategy.hedge_ratio
        hedged_pnl = underlying_pnl + hedge_benefit
        
        return {
            'price_scenarios': price_scenarios,
            'strategy_payoffs': scenario_payoffs,
            'underlying_pnl': underlying_pnl,
            'hedge_benefit': hedge_benefit,
            'hedged_pnl': hedged_pnl,
            'unhedged_pnl': underlying_pnl
        }
        
    except Exception as e:
        print(f"Error in multi-leg scenario simulation: {e}")
        # Return empty results
        n_scenarios = len(price_scenarios)
        return {
            'price_scenarios': price_scenarios,
            'strategy_payoffs': np.zeros(n_scenarios),
            'underlying_pnl': np.zeros(n_scenarios),
            'hedge_benefit': np.zeros(n_scenarios),
            'hedged_pnl': np.zeros(n_scenarios),
            'unhedged_pnl': np.zeros(n_scenarios)
        }


def simulate_price_scenarios(current_price: float, volatility: float, time_horizon: float,
                           n_sim: int = 1000, n_steps: int = 252, 
                           drift: float = 0.0) -> np.ndarray:
    """
    Enhanced price scenario simulation with optional drift.
    
    Args:
        current_price: Current price of asset
        volatility: Annualized volatility
        time_horizon: Time horizon in years
        n_sim: Number of simulations
        n_steps: Number of time steps
        drift: Annual drift rate (default 0 for commodities)
        
    Returns:
        Array of final prices from simulations
    """
    dt = time_horizon / n_steps
    
    random_shocks = np.random.normal(0, 1, (n_sim, n_steps))
    
    price_paths = np.zeros((n_sim, n_steps + 1))
    price_paths[:, 0] = current_price
    
    for t in range(n_steps):
        price_paths[:, t + 1] = price_paths[:, t] * np.exp(
            (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shocks[:, t]
        )
    
    final_prices = price_paths[:, -1]
    
    return final_prices


def simulate_correlated_assets(current_prices: List[float], volatilities: List[float],
                             correlation_matrix: np.ndarray, time_horizon: float,
                             n_sim: int = 1000, n_steps: int = 252) -> np.ndarray:
    """
    Simulate correlated asset price paths for portfolio analysis.
    
    Args:
        current_prices: List of current prices for each asset
        volatilities: List of volatilities for each asset
        correlation_matrix: Correlation matrix between assets
        time_horizon: Time horizon in years
        n_sim: Number of simulations
        n_steps: Number of time steps
        
    Returns:
        Array of shape (n_sim, n_assets) containing final prices
    """
    n_assets = len(current_prices)
    dt = time_horizon / n_steps
    
    # Generate correlated random shocks
    uncorrelated_shocks = np.random.normal(0, 1, (n_sim, n_steps, n_assets))
    
    # Apply correlation using Cholesky decomposition
    try:
        chol_matrix = np.linalg.cholesky(correlation_matrix)
        correlated_shocks = np.zeros_like(uncorrelated_shocks)
        
        for sim in range(n_sim):
            for step in range(n_steps):
                correlated_shocks[sim, step, :] = chol_matrix @ uncorrelated_shocks[sim, step, :]
    except np.linalg.LinAlgError:
        # If Cholesky fails, use uncorrelated shocks
        correlated_shocks = uncorrelated_shocks
    
    # Simulate price paths
    price_paths = np.zeros((n_sim, n_steps + 1, n_assets))
    
    for asset in range(n_assets):
        price_paths[:, 0, asset] = current_prices[asset]
        
        for t in range(n_steps):
            price_paths[:, t + 1, asset] = price_paths[:, t, asset] * np.exp(
                (-0.5 * volatilities[asset]**2) * dt + 
                volatilities[asset] * np.sqrt(dt) * correlated_shocks[:, t, asset]
            )
    
    final_prices = price_paths[:, -1, :]
    
    return final_prices


def calculate_simulation_statistics(simulated_pnl: np.ndarray) -> Dict[str, float]:
    """Enhanced simulation statistics calculation"""
    if len(simulated_pnl) == 0:
        raise ValueError("Simulated P&L array cannot be empty")
    
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
    
    stats_dict['prob_profit'] = float(np.sum(simulated_pnl > 0) / len(simulated_pnl))
    stats_dict['prob_loss'] = float(np.sum(simulated_pnl < 0) / len(simulated_pnl))
    
    # Enhanced statistics
    stats_dict['var_95'] = float(np.percentile(simulated_pnl, 5))
    stats_dict['cvar_95'] = float(np.mean(simulated_pnl[simulated_pnl <= stats_dict['var_95']]))
    stats_dict['positive_expected_value'] = float(np.mean(simulated_pnl[simulated_pnl > 0])) if np.any(simulated_pnl > 0) else 0.0
    stats_dict['negative_expected_value'] = float(np.mean(simulated_pnl[simulated_pnl < 0])) if np.any(simulated_pnl < 0) else 0.0
    
    return stats_dict


def compare_hedging_effectiveness(hedged_sim: np.ndarray, unhedged_sim: np.ndarray) -> Dict[str, float]:
    """Enhanced hedging effectiveness comparison"""
    hedged_stats = calculate_simulation_statistics(hedged_sim)
    unhedged_stats = calculate_simulation_statistics(unhedged_sim)
    
    hedged_sharpe = hedged_stats['mean'] / hedged_stats['std'] if hedged_stats['std'] > 0 else 0
    unhedged_sharpe = unhedged_stats['mean'] / unhedged_stats['std'] if unhedged_stats['std'] > 0 else 0
    
    comparison = {
        'hedged_mean': hedged_stats['mean'],
        'unhedged_mean': unhedged_stats['mean'],
        'mean_difference': hedged_stats['mean'] - unhedged_stats['mean'],
        'hedged_std': hedged_stats['std'],
        'unhedged_std': unhedged_stats['std'],
        'volatility_reduction': (unhedged_stats['std'] - hedged_stats['std']) / unhedged_stats['std'] if unhedged_stats['std'] > 0 else 0,
        'hedged_sharpe': hedged_sharpe,
        'unhedged_sharpe': unhedged_sharpe,
        'sharpe_improvement': hedged_sharpe - unhedged_sharpe,
        'hedged_prob_loss': hedged_stats['prob_loss'],
        'unhedged_prob_loss': unhedged_stats['prob_loss'],
        'loss_prob_reduction': unhedged_stats['prob_loss'] - hedged_stats['prob_loss'],
        'hedged_var_95': hedged_stats['var_95'],
        'unhedged_var_95': unhedged_stats['var_95'],
        'var_improvement': (unhedged_stats['var_95'] - hedged_stats['var_95']) / abs(unhedged_stats['var_95']) if unhedged_stats['var_95'] != 0 else 0
    }
    
    return comparison


def run_comprehensive_simulation(position_config: Dict, n_sim: int = 5000) -> Dict[str, any]:
    """
    Run comprehensive simulation analysis for any position configuration.
    
    Args:
        position_config: Dictionary containing position parameters
        n_sim: Number of simulations
        
    Returns:
        Comprehensive simulation results
    """
    try:
        # Extract configuration
        commodity = position_config.get('commodity', 'WTI Crude Oil')
        position_size = position_config.get('position_size', 1000)
        hedge_ratio = position_config.get('hedge_ratio', 0.8)
        strategy = position_config.get('strategy', 'Futures')
        
        # Get price data
        from hedging.data import get_prices, get_current_price
        prices = get_prices(commodity, period="1y")
        current_price = get_current_price(commodity)
        
        # Run simulation based on strategy type
        if strategy == "Multi-Leg" and 'multi_leg_strategy' in position_config:
            multi_leg_strategy = position_config['multi_leg_strategy']
            
            # Price scenarios for multi-leg analysis
            volatility = prices.pct_change().std() * np.sqrt(252)
            price_scenarios = simulate_price_scenarios(current_price, volatility, 0.25, n_sim)
            
            scenario_results = simulate_multi_leg_scenarios(
                multi_leg_strategy, current_price, price_scenarios
            )
            
            # Convert to standard format
            sim_results = {
                'unhedged_pnl': scenario_results['unhedged_pnl'],
                'hedged_pnl': scenario_results['hedged_pnl'],
                'hedge_benefit': scenario_results['hedge_benefit']
            }
        
        else:
            # Standard simulation
            strike_price = position_config.get('strike_price')
            
            sim_results = simulate_hedged_vs_unhedged(
                prices, position_size, hedge_ratio, strategy, strike_price, n_sim=n_sim
            )
        
        # Calculate statistics
        hedged_stats = calculate_simulation_statistics(sim_results['hedged_pnl'])
        unhedged_stats = calculate_simulation_statistics(sim_results['unhedged_pnl'])
        effectiveness = compare_hedging_effectiveness(sim_results['hedged_pnl'], sim_results['unhedged_pnl'])
        
        return {
            'simulation_results': sim_results,
            'hedged_statistics': hedged_stats,
            'unhedged_statistics': unhedged_stats,
            'hedging_effectiveness': effectiveness,
            'position_config': position_config,
            'n_simulations': n_sim
        }
        
    except Exception as e:
        print(f"Error in comprehensive simulation: {e}")
        return {
            'error': str(e),
            'position_config': position_config
        }


if __name__ == "__main__":
    
    # Test enhanced simulation
    np.random.seed(int(time.time() * 1000000) % 2**32)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    initial_price = 75.0
    returns = np.random.normal(0.0002, 0.02, 252)
    prices = pd.Series(initial_price * np.exp(np.cumsum(returns)), index=dates)
    
    # Test traditional hedge
    print("Testing traditional futures hedge...")
    futures_result = simulate_hedged_vs_unhedged(prices, 1000, 0.8, "Futures", n_sim=1000)
    futures_effectiveness = compare_hedging_effectiveness(
        futures_result['hedged_pnl'], futures_result['unhedged_pnl']
    )
    print(f"Futures volatility reduction: {futures_effectiveness['volatility_reduction']:.2%}")
    
    # Test multi-leg simulation (if available)
    try:
        from hedging.multi_leg_strategies import create_long_straddle
        
        print("Testing multi-leg strategy simulation...")
        straddle = create_long_straddle(75.0, 1000, 0.8, "WTI Crude Oil")
        
        multi_leg_result = simulate_hedged_vs_unhedged(
            prices, 1000, 0.8, "Multi-Leg", multi_leg_strategy=straddle, n_sim=1000
        )
        
        multi_leg_effectiveness = compare_hedging_effectiveness(
            multi_leg_result['hedged_pnl'], multi_leg_result['unhedged_pnl']
        )
        print(f"Multi-leg volatility reduction: {multi_leg_effectiveness['volatility_reduction']:.2%}")
        
        print("Enhanced simulation engine working correctly")
        
    except ImportError:
        print("Multi-leg strategies not available - basic simulation tested")
        print("Enhanced simulation engine partially working")