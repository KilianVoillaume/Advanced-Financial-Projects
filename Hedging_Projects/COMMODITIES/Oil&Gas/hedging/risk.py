"""
hedging/risk.py

Risk metrics computations for hedged and unhedged positions.
Calculates Expected P&L, Value-at-Risk (VaR), Conditional VaR (CVaR), and delta exposure.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats


def calculate_risk_metrics(pnl_sim: np.ndarray, confidence: float, risk_free_rate: float = 0.05) -> pd.DataFrame:
    
    if len(pnl_sim) == 0:
        raise ValueError("P&L simulation array cannot be empty")
    
    if not (0.90 <= confidence <= 0.99):
        raise ValueError("Confidence level must be between 0.90 and 0.99")
    
    expected_pnl = float(np.mean(pnl_sim))
    
    volatility = float(np.std(pnl_sim))
    
    if volatility > 0:
        sharpe_ratio = expected_pnl / volatility
    else:
        sharpe_ratio = 0.0
    
    var_percentile = (1 - confidence) * 100
    var = float(np.percentile(pnl_sim, var_percentile))
    
    losses_worse_than_var = pnl_sim[pnl_sim <= var]
    if len(losses_worse_than_var) > 0:
        cvar = float(np.mean(losses_worse_than_var))
    else:
        cvar = var  # If no losses worse than VaR, CVaR equals VaR
    
    downside_deviation = calculate_downside_deviation(pnl_sim)
    max_drawdown = calculate_max_drawdown_estimate(pnl_sim)
    
    # Results DataFrame
    metrics_data = {
        'Metric': [
            'Expected P&L',
            f'VaR ({confidence:.0%})',
            f'CVaR ({confidence:.0%})',
            'Volatility',
            'Sharpe Ratio',
            'Downside Deviation',
            'Max Drawdown (Est.)',
            'Skewness',
            'Kurtosis'
        ],
        'Value': [
            expected_pnl,
            var,
            cvar,
            volatility,
            sharpe_ratio,
            downside_deviation,
            max_drawdown,
            float(stats.skew(pnl_sim)),
            float(stats.kurtosis(pnl_sim))
        ]
    }
    
    risk_metrics_df = pd.DataFrame(metrics_data)
    
    return risk_metrics_df


def calculate_downside_deviation(pnl_sim: np.ndarray, target_return: float = 0.0) -> float:

    # Get returns below target
    downside_returns = pnl_sim[pnl_sim < target_return]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_deviation = float(np.sqrt(np.mean((downside_returns - target_return) ** 2)))
    
    return downside_deviation


def calculate_max_drawdown_estimate(pnl_sim: np.ndarray) -> float:

    sorted_pnl = np.sort(pnl_sim)[::-1]
    cumulative_pnl = np.cumsum(sorted_pnl)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = cumulative_pnl - running_max
    max_drawdown = float(np.min(drawdowns))
    
    return max_drawdown


def calculate_delta_exposure(prices: pd.Series, position: float, hedge_ratio: float, strategy: str, strike_price: Optional[float] = None) -> float:
   
    if strategy.lower() == "futures":
        net_delta = 1.0 - (1.0 * hedge_ratio)
        
    elif strategy.lower() == "options":
        # Options delta varies with price and time
        current_price = float(prices.iloc[-1])
        
        if strike_price is None:
            strike_price = current_price  # ATM
        
        # Delta approximation for ATM options
        if abs(current_price - strike_price) / strike_price < 0.05:  # Within 5% of ATM
            option_delta = 0.5  
        else:
            moneyness = current_price / strike_price
            if position > 0:  # Long position, buying puts
                option_delta = max(0.0, min(1.0, 1.5 - moneyness))
            else:  # Short position, buying calls
                option_delta = max(0.0, min(1.0, moneyness - 0.5))
        
        # Net delta for hedged position
        # Long position + long puts: net_delta = 1.0 - put_delta * hedge_ratio
        if position > 0:
            net_delta = 1.0 - (option_delta * hedge_ratio)
        else:
            net_delta = -1.0 + (option_delta * hedge_ratio)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Supported strategies: 'Futures', 'Options'")
    
    delta_exposure = net_delta * abs(position)
    
    return float(delta_exposure)

def calculate_hedge_efficiency(hedged_pnl: np.ndarray, unhedged_pnl: np.ndarray) -> Dict[str, float]:

    hedged_var = np.var(hedged_pnl)
    unhedged_var = np.var(unhedged_pnl)
    variance_reduction = (unhedged_var - hedged_var) / unhedged_var
    
    correlation = np.corrcoef(hedged_pnl, unhedged_pnl)[0, 1]
    
    tracking_error = np.std(hedged_pnl - unhedged_pnl)
    
    hedge_efficiency = 1 - (hedged_var / unhedged_var)
    
    hedged_mean = np.mean(hedged_pnl)
    unhedged_mean = np.mean(unhedged_pnl)
    hedged_std = np.std(hedged_pnl)
    unhedged_std = np.std(unhedged_pnl)
    
    hedged_sharpe = hedged_mean / hedged_std if hedged_std > 0 else 0
    unhedged_sharpe = unhedged_mean / unhedged_std if unhedged_std > 0 else 0
    sharpe_improvement = hedged_sharpe - unhedged_sharpe
    
    efficiency_metrics = {
        'variance_reduction': float(variance_reduction),
        'hedge_efficiency': float(hedge_efficiency),
        'correlation': float(correlation),
        'tracking_error': float(tracking_error),
        'hedged_sharpe': float(hedged_sharpe),
        'unhedged_sharpe': float(unhedged_sharpe),
        'sharpe_improvement': float(sharpe_improvement)
    }
    
    return efficiency_metrics


def calculate_scenario_analysis(current_price: float, position: float, hedge_ratio: float,
                                strategy: str, strike_price: Optional[float] = None,
                                price_shocks: Optional[list] = None) -> pd.DataFrame:
    
    if price_shocks is None:
        price_shocks = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
    
    if strike_price is None:
        strike_price = current_price
    
    scenarios = []
    
    for shock in price_shocks:
        shocked_price = current_price * (1 + shock)
        
        underlying_pnl = (shocked_price - current_price) * position
        
        if strategy.lower() == "futures":
            hedge_pnl = -(shocked_price - current_price) * position * hedge_ratio
        
        elif strategy.lower() == "options":
            # Options hedge P&L (at expiry)
            if position > 0:  # Long position, buying puts
                option_payoff = max(strike_price - shocked_price, 0)
            else:  # Short position, buying calls
                option_payoff = max(shocked_price - strike_price, 0)
            
            hedge_pnl = option_payoff * abs(position) * hedge_ratio
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Calculate net P&L
        net_pnl = underlying_pnl + hedge_pnl
        
        scenarios.append({
            'Price_Shock': f"{shock:.1%}",
            'New_Price': shocked_price,
            'Underlying_PnL': underlying_pnl,
            'Hedge_PnL': hedge_pnl,
            'Net_PnL': net_pnl,
            'Hedge_Effectiveness': abs(hedge_pnl / underlying_pnl) if underlying_pnl != 0 else 0
        })
    
    scenario_df = pd.DataFrame(scenarios)
    
    return scenario_df


def summarize_risk_comparison(hedged_metrics: pd.DataFrame, unhedged_metrics: pd.DataFrame) -> pd.DataFrame:

    # Merge the metrics
    comparison = pd.merge(hedged_metrics, unhedged_metrics, on='Metric', suffixes=('_Hedged', '_Unhedged'))
    
    comparison['Difference'] = comparison['Value_Hedged'] - comparison['Value_Unhedged']
    comparison['Improvement'] = comparison['Difference'] / comparison['Value_Unhedged'].abs()
    
    comparison = comparison[['Metric', 'Value_Unhedged', 'Value_Hedged', 'Difference', 'Improvement']]
    
    return comparison


if __name__ == "__main__":
    print("Testing hedging/risk.py module...")
    
    np.random.seed(42) 
    n_sim = 10000
    
    unhedged_pnl = np.random.normal(100, 500, n_sim)  # Mean $100, Std $500
    hedged_pnl = np.random.normal(80, 200, n_sim)   # Mean $80, Std $200
    
    confidence_level = 0.95
    
    print(f"\n=== Risk Metrics Test (Confidence: {confidence_level:.0%}) ===")
    
    unhedged_risk = calculate_risk_metrics(unhedged_pnl, confidence_level)
    hedged_risk = calculate_risk_metrics(hedged_pnl, confidence_level)
    
    print("Unhedged Position Risk Metrics:")
    print(unhedged_risk.to_string(index=False, float_format='%.2f'))
    
    print("\nHedged Position Risk Metrics:")
    print(hedged_risk.to_string(index=False, float_format='%.2f'))
    
    print("\n=== Risk Comparison ===")
    risk_comparison = summarize_risk_comparison(hedged_risk, unhedged_risk)
    print(risk_comparison.to_string(index=False, float_format='%.2f'))
    
    print("\n=== Hedge Efficiency Metrics ===")
    efficiency = calculate_hedge_efficiency(hedged_pnl, unhedged_pnl)
    for metric, value in efficiency.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    print("\n=== Delta Exposure Test ===")
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.Series(75.0 + np.cumsum(np.random.normal(0, 1, 100)), index=dates)
    
    futures_delta = calculate_delta_exposure(prices, 1000, 0.8, "Futures")
    options_delta = calculate_delta_exposure(prices, 1000, 0.8, "Options", 75.0)
    
    print(f"Futures hedge delta exposure: {futures_delta:.2f}")
    print(f"Options hedge delta exposure: {options_delta:.2f}")
    
    print("\n=== Scenario Analysis Test ===")
    scenarios = calculate_scenario_analysis(75.0, 1000, 0.8, "Futures")
    print("Futures Hedge Scenarios:")
    print(scenarios.to_string(index=False, float_format='%.2f'))
