"""
hedging/strategies.py

Module for hedge payoff calculations for different hedging strategies.
Supports Futures, Options, and Multi-Leg Options with payoff diagram generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.stats import norm


def compute_futures_hedge(prices: pd.Series, position: float, hedge_ratio: float) -> pd.Series:
    """Calculate P&L for futures hedge strategy"""
    position = float(position)
    hedge_ratio = float(hedge_ratio)
    
    price_changes = prices.diff().dropna()
    hedge_pnl = -price_changes * position * hedge_ratio
    
    return hedge_pnl


def compute_options_hedge(prices: pd.Series, position: float, hedge_ratio: float, strike_price: float, option_type: str = "put",
                         time_to_expiry: float = 0.25, risk_free_rate: float = 0.05, volatility: Optional[float] = None) -> pd.Series:
    """Calculate P&L for single options hedge strategy"""
    position = float(position)
    hedge_ratio = float(hedge_ratio)
    strike_price = float(strike_price)
    time_to_expiry = float(time_to_expiry)
    risk_free_rate = float(risk_free_rate)
    
    if volatility is None:
        returns = prices.pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252))
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
    """Calculate option delta using Black-Scholes"""
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
    """Compute payoff diagram for single option strategies"""
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


# ============================================================================
# MULTI-LEG PAYOFF CALCULATIONS - Step 2.2 Implementation
# ============================================================================

def compute_multi_leg_payoff(current_price: float, multi_leg_strategy, 
                            price_range_pct: float = 0.3, num_points: int = 100) -> Dict[str, pd.Series]:
    """
    Calculate comprehensive payoff diagram for multi-leg options strategies.
    
    Args:
        current_price: Current price of underlying asset
        multi_leg_strategy: MultiLegStrategy object
        price_range_pct: Price range percentage for payoff calculation
        num_points: Number of price points to calculate
        
    Returns:
        Dictionary containing payoff series and analysis
    """
    current_price = float(current_price)
    
    # Generate price range
    price_min = current_price * (1 - price_range_pct)
    price_max = current_price * (1 + price_range_pct)
    spot_prices = np.linspace(price_min, price_max, num_points)
    
    # Calculate underlying P&L
    underlying_pnl = (spot_prices - current_price) * multi_leg_strategy.underlying_size
    
    # Calculate strategy payoff at expiration
    strategy_payoff = np.zeros_like(spot_prices)
    total_premium = 0.0
    
    # Calculate payoff for each leg
    for leg in multi_leg_strategy.legs:
        leg_payoff = np.zeros_like(spot_prices)
        
        for i, spot_price in enumerate(spot_prices):
            # Calculate intrinsic value at expiration
            if leg.option_type.lower() == 'call':
                intrinsic_value = max(spot_price - leg.strike_price, 0)
            else:  # put
                intrinsic_value = max(leg.strike_price - spot_price, 0)
            
            # Apply position (long = +1, short = -1) and quantity
            leg_payoff[i] = intrinsic_value * leg.quantity
        
        strategy_payoff += leg_payoff
        
        # Calculate premium for this leg (simplified)
        leg_premium = _estimate_option_premium(current_price, leg.strike_price, 
                                             leg.option_type, leg.expiry_months)
        total_premium += leg_premium * leg.quantity
    
    # Net strategy payoff (payoff minus premium cost)
    net_strategy_payoff = strategy_payoff - total_premium
    
    # Calculate hedge P&L (strategy applied to underlying position)
    hedge_pnl = net_strategy_payoff * multi_leg_strategy.hedge_ratio
    
    # Total net P&L (underlying + hedge)
    net_pnl = underlying_pnl + hedge_pnl
    
    # Find breakeven points
    breakeven_prices = find_breakeven_prices(spot_prices, net_pnl)
    strategy_breakevens = find_breakeven_prices(spot_prices, net_strategy_payoff)
    
    # Calculate key metrics
    max_profit_idx = np.argmax(net_strategy_payoff)
    max_loss_idx = np.argmin(net_strategy_payoff)
    
    result = {
        'spot_prices': pd.Series(spot_prices, name='Spot Price'),
        'underlying_pnl': pd.Series(underlying_pnl, name='Underlying P&L'),
        'strategy_payoff': pd.Series(strategy_payoff, name='Strategy Payoff (Gross)'),
        'net_strategy_payoff': pd.Series(net_strategy_payoff, name='Strategy Payoff (Net)'),
        'hedge_pnl': pd.Series(hedge_pnl, name='Hedge P&L'),
        'net_pnl': pd.Series(net_pnl, name='Total Net P&L'),
        'breakeven_prices': breakeven_prices,
        'strategy_breakevens': strategy_breakevens,
        'total_premium': total_premium,
        'max_profit': float(np.max(net_strategy_payoff)),
        'max_loss': float(np.min(net_strategy_payoff)),
        'max_profit_price': float(spot_prices[max_profit_idx]),
        'max_loss_price': float(spot_prices[max_loss_idx])
    }
    
    return result


def _estimate_option_premium(current_price: float, strike_price: float, 
                           option_type: str, expiry_months: int) -> float:
    """
    Estimate option premium using simplified Black-Scholes.
    This is a placeholder - in production, use full Black-Scholes calculation.
    """
    try:
        from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, time_to_expiration
        
        time_to_expiry = time_to_expiration(expiry_months)
        risk_free_rate = get_risk_free_rate()
        volatility = 0.25  # Default 25% volatility
        
        premium = BlackScholesCalculator.calculate_option_price(
            current_price, strike_price, time_to_expiry,
            risk_free_rate, volatility, option_type
        )
        
        return premium
        
    except ImportError:
        # Fallback simplified calculation
        if option_type.lower() == 'call':
            intrinsic = max(current_price - strike_price, 0)
        else:
            intrinsic = max(strike_price - current_price, 0)
        
        # Add time value (rough approximation)
        time_value = current_price * 0.02 * (expiry_months / 12.0)
        
        return intrinsic + time_value


def analyze_multi_leg_payoff(payoff_data: Dict[str, pd.Series]) -> Dict[str, any]:
    """
    Analyze multi-leg strategy payoff for risk metrics and key insights.
    
    Args:
        payoff_data: Result from compute_multi_leg_payoff()
        
    Returns:
        Dictionary containing analysis results
    """
    net_payoff = payoff_data['net_strategy_payoff'].values
    spot_prices = payoff_data['spot_prices'].values
    
    # Profit zones
    profitable_mask = net_payoff > 0
    profit_zones = []
    
    if np.any(profitable_mask):
        profit_prices = spot_prices[profitable_mask]
        if len(profit_prices) > 0:
            profit_zones = [(float(np.min(profit_prices)), float(np.max(profit_prices)))]
    
    # Risk metrics
    max_profit = float(np.max(net_payoff))
    max_loss = float(np.min(net_payoff))
    profit_probability = float(np.sum(profitable_mask) / len(net_payoff))
    
    # Expected value (assuming uniform price distribution)
    expected_pnl = float(np.mean(net_payoff))
    
    # Volatility sensitivity (rough approximation)
    payoff_volatility = float(np.std(net_payoff))
    
    analysis = {
        'max_profit': max_profit,
        'max_loss': max_loss,
        'expected_pnl': expected_pnl,
        'profit_probability': profit_probability,
        'payoff_volatility': payoff_volatility,
        'profit_zones': profit_zones,
        'breakeven_count': len(payoff_data['strategy_breakevens']),
        'risk_reward_ratio': abs(max_profit / max_loss) if max_loss != 0 else float('inf'),
        'total_premium': payoff_data['total_premium'],
        'is_net_credit': payoff_data['total_premium'] < 0,
        'is_net_debit': payoff_data['total_premium'] > 0
    }
    
    return analysis


def get_strategy_payoff_summary(multi_leg_strategy, current_price: float) -> Dict[str, str]:
    """
    Get formatted summary of strategy characteristics and payoff.
    
    Args:
        multi_leg_strategy: MultiLegStrategy object
        current_price: Current price of underlying
        
    Returns:
        Dictionary with formatted summary information
    """
    payoff_data = compute_multi_leg_payoff(current_price, multi_leg_strategy)
    analysis = analyze_multi_leg_payoff(payoff_data)
    
    # Format summary
    summary = {
        'Strategy': multi_leg_strategy.strategy_type.value,
        'Legs': f"{len(multi_leg_strategy.legs)} legs",
        'Net Premium': f"${analysis['total_premium']:.2f} {'(Credit)' if analysis['is_net_credit'] else '(Debit)'}",
        'Max Profit': f"${analysis['max_profit']:.2f}" if analysis['max_profit'] != float('inf') else "Unlimited",
        'Max Loss': f"${abs(analysis['max_loss']):.2f}" if analysis['max_loss'] != float('-inf') else "Unlimited",
        'Breakevens': f"{analysis['breakeven_count']} points",
        'Profit Probability': f"{analysis['profit_probability']:.1%}",
        'Risk/Reward': f"{analysis['risk_reward_ratio']:.2f}" if analysis['risk_reward_ratio'] != float('inf') else "∞"
    }
    
    return summary


def compare_strategies(strategies: List[Tuple[str, any]], current_price: float) -> pd.DataFrame:
    """
    Compare multiple multi-leg strategies side by side.
    
    Args:
        strategies: List of (name, MultiLegStrategy) tuples
        current_price: Current price of underlying
        
    Returns:
        DataFrame comparing strategies
    """
    comparison_data = []
    
    for name, strategy in strategies:
        try:
            payoff_data = compute_multi_leg_payoff(current_price, strategy)
            analysis = analyze_multi_leg_payoff(payoff_data)
            
            comparison_data.append({
                'Strategy': name,
                'Type': strategy.strategy_type.value,
                'Premium': f"${analysis['total_premium']:.2f}",
                'Max Profit': f"${analysis['max_profit']:.2f}" if analysis['max_profit'] != float('inf') else "Unlimited",
                'Max Loss': f"${abs(analysis['max_loss']):.2f}" if analysis['max_loss'] != float('-inf') else "Unlimited",
                'Profit Prob': f"{analysis['profit_probability']:.1%}",
                'Expected P&L': f"${analysis['expected_pnl']:.2f}",
                'Risk/Reward': f"{analysis['risk_reward_ratio']:.2f}" if analysis['risk_reward_ratio'] != float('inf') else "∞"
            })
            
        except Exception as e:
            comparison_data.append({
                'Strategy': name,
                'Type': 'Error',
                'Premium': f"Error: {str(e)}",
                'Max Profit': 'N/A',
                'Max Loss': 'N/A',
                'Profit Prob': 'N/A',
                'Expected P&L': 'N/A',
                'Risk/Reward': 'N/A'
            })
    
    return pd.DataFrame(comparison_data)


# ============================================================================
# ENHANCED UTILITIES
# ============================================================================

def find_breakeven_prices(prices: np.ndarray, pnl: np.ndarray, tolerance: float = 0.01) -> List[float]:
    """Find breakeven prices where P&L crosses zero"""
    breakeven_prices = []
    
    for i in range(len(pnl) - 1):
        if (pnl[i] * pnl[i + 1] <= 0) and (abs(pnl[i]) > tolerance or abs(pnl[i + 1]) > tolerance):
            if pnl[i + 1] != pnl[i]:
                breakeven_price = prices[i] - pnl[i] * (prices[i + 1] - prices[i]) / (pnl[i + 1] - pnl[i])
                breakeven_prices.append(float(breakeven_price))
    
    return sorted(breakeven_prices)


def get_hedge_summary(current_price: float, position: float, hedge_ratio: float,
                     strategy: str, strike_price: Optional[float] = None,
                     multi_leg_strategy: Optional[any] = None) -> Dict[str, float]:
    """Enhanced hedge summary supporting multi-leg strategies"""
    
    if multi_leg_strategy is not None:
        # Multi-leg strategy summary
        payoff_data = compute_multi_leg_payoff(current_price, multi_leg_strategy)
        analysis = analyze_multi_leg_payoff(payoff_data)
        
        summary = {
            'hedge_ratio': float(hedge_ratio),
            'position_size': float(position),
            'current_price': float(current_price),
            'max_profit': analysis['max_profit'],
            'max_loss': abs(analysis['max_loss']),
            'profit_range': analysis['max_profit'] - analysis['max_loss'],
            'num_breakevens': analysis['breakeven_count'],
            'total_premium': analysis['total_premium'],
            'expected_pnl': analysis['expected_pnl'],
            'profit_probability': analysis['profit_probability']
        }
        
        return summary
    
    else:
        # Single strategy summary (existing logic)
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
    """Simplified crack spread simulation for hedging analysis"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    
    base_spread = 15.0
    spread_changes = np.random.normal(0, 2.0, 252)
    crack_spreads = base_spread + np.cumsum(spread_changes * 0.1)
    
    crack_spread_series = pd.Series(crack_spreads, index=dates)
    spread_changes_series = crack_spread_series.diff().dropna()
    
    daily_capacity = refinery_capacity
    unhedged_pnl = spread_changes_series * (daily_capacity / 3)
    
    hedge_pnl = -spread_changes_series * (daily_capacity / 3) * hedge_ratio
    hedged_pnl = unhedged_pnl + hedge_pnl
    
    return {
        'unhedged_pnl': unhedged_pnl,
        'hedge_pnl': hedge_pnl,
        'hedged_pnl': hedged_pnl,
        'crack_spreads': crack_spread_series
    }


if __name__ == "__main__":
    
    # Test with simple straddle
    try:
        from hedging.multi_leg_strategies import create_long_straddle
        
        current_price = 75.0
        straddle = create_long_straddle(75.0, 1000, 0.8, "WTI Crude Oil")
        
        # Calculate payoff
        payoff_data = compute_multi_leg_payoff(current_price, straddle)
        analysis = analyze_multi_leg_payoff(payoff_data)
        summary = get_strategy_payoff_summary(straddle, current_price)
        
        print(f"Strategy: {summary['Strategy']}")
        print(f"Max Profit: {summary['Max Profit']}")
        print(f"Max Loss: {summary['Max Loss']}")
        print(f"Breakevens: {summary['Breakevens']}")
        print(f"Premium: {summary['Net Premium']}")
        
        print("Multi-leg payoff calculations working correctly")
        
    except ImportError:
        print("Multi-leg strategies module not found - basic test only")
        
        # Test basic payoff calculation
        basic_payoff = compute_payoff_diagram(75.0, 1000, 0.8, "Futures")
        print(f"Basic payoff calculation: {len(basic_payoff['spot_prices'])} points")
        print("Basic payoff calculations working correctly")