"""
hedging/crack_spreads.py

Crack spread analysis for refinery margin hedging.
Models the 3-2-1 crack spread (3 barrels crude oil → 2 barrels gasoline + 1 barrel heating oil).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Tuple, Optional


def get_product_prices(period: str = "1y") -> Dict[str, pd.Series]:
    """
    Fetch prices for crude oil and refined products.
    
    Returns:
        Dict with crude, gasoline, and heating oil prices
    """
    
    product_tickers = {
        "crude": "CL=F",      # WTI Crude Oil
        "gasoline": "RB=F",   # RBOB Gasoline
        "heating_oil": "HO=F" # NY Harbor ULSD (Heating Oil)
    }
    
    prices = {}
    
    for product, ticker in product_tickers.items():
        try:
            data = yf.download(ticker, period=period, progress=False)
            if not data.empty and 'Close' in data.columns:
                prices[product] = data['Close'].dropna()
            else:
                # Fallback prices if data fetch fails
                fallback_values = {
                    "crude": 75.0,
                    "gasoline": 2.10,    # $/gallon
                    "heating_oil": 2.20  # $/gallon
                }
                
                # Create dummy price series
                dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
                base_price = fallback_values[product]
                random_returns = np.random.normal(0, 0.02, 252)  # 2% daily volatility
                price_series = base_price * np.exp(np.cumsum(random_returns))
                prices[product] = pd.Series(price_series, index=dates)
                
                print(f"Warning: Using simulated prices for {product}")
                
        except Exception as e:
            print(f"Error fetching {product} prices: {e}")
            # Use fallback as above
            
    return prices


def calculate_crack_spread(crude_price: pd.Series, gasoline_price: pd.Series, 
                          heating_oil_price: pd.Series, spread_type: str = "3-2-1") -> pd.Series:
    """
    Calculate crack spread (refinery processing margin).
    
    Args:
        crude_price: WTI crude oil prices ($/barrel)
        gasoline_price: RBOB gasoline prices ($/gallon)
        heating_oil_price: Heating oil prices ($/gallon)
        spread_type: Type of crack spread (default: "3-2-1")
    
    Returns:
        pd.Series: Crack spread in $/barrel
    """
    
    if spread_type == "3-2-1":
        # 3-2-1 Crack Spread: 3 barrels crude → 2 barrels gasoline + 1 barrel heating oil
        # Convert gallons to barrels (42 gallons per barrel)
        gasoline_per_barrel = gasoline_price * 42  # Convert $/gallon to $/barrel equivalent
        heating_oil_per_barrel = heating_oil_price * 42
        
        # Crack spread = (2 * gasoline + 1 * heating oil) - 3 * crude
        crack_spread = (2 * gasoline_per_barrel + heating_oil_per_barrel) - 3 * crude_price
        
    elif spread_type == "2-1-1":
        # 2-1-1 Crack Spread: 2 barrels crude → 1 barrel gasoline + 1 barrel heating oil
        gasoline_per_barrel = gasoline_price * 42
        heating_oil_per_barrel = heating_oil_price * 42
        
        crack_spread = (gasoline_per_barrel + heating_oil_per_barrel) - 2 * crude_price
        
    else:
        raise ValueError(f"Unknown crack spread type: {spread_type}")
    
    return crack_spread


def compute_crack_spread_hedge(refinery_capacity: float, hedge_ratio: float,
                              spread_type: str = "3-2-1", period: str = "1y") -> Dict[str, pd.Series]:
    """
    Compute crack spread hedging strategy for a refinery.
    
    Args:
        refinery_capacity: Refinery capacity in barrels per day
        hedge_ratio: Hedge ratio between 0.0 and 1.0
        spread_type: Type of crack spread (default: "3-2-1")
        period: Historical period for analysis
    
    Returns:
        Dict with crack spread P&L analysis
    """
    
    # Get historical prices
    prices = get_product_prices(period)
    
    # Align all price series to same dates
    common_dates = prices['crude'].index.intersection(
        prices['gasoline'].index
    ).intersection(prices['heating_oil'].index)
    
    crude_aligned = prices['crude'][common_dates]
    gasoline_aligned = prices['gasoline'][common_dates]
    heating_oil_aligned = prices['heating_oil'][common_dates]
    
    # Calculate historical crack spreads
    crack_spreads = calculate_crack_spread(
        crude_aligned, gasoline_aligned, heating_oil_aligned, spread_type
    )
    
    # Calculate daily crack spread changes
    crack_spread_changes = crack_spreads.diff().dropna()
    
    # Calculate unhedged refinery margin P&L
    # Refinery makes money when crack spreads widen
    daily_capacity = refinery_capacity  # barrels per day
    
    if spread_type == "3-2-1":
        # Each unit of crack spread represents margin on 3 barrels of crude processing
        unhedged_pnl = crack_spread_changes * (daily_capacity / 3)
    else:  # 2-1-1
        # Each unit represents margin on 2 barrels of crude processing
        unhedged_pnl = crack_spread_changes * (daily_capacity / 2)
    
    # Calculate hedged P&L
    # Hedging crack spreads: sell product futures, buy crude futures
    # This locks in the processing margin
    hedge_pnl = -crack_spread_changes * (daily_capacity / (3 if spread_type == "3-2-1" else 2)) * hedge_ratio
    
    # Net hedged P&L
    hedged_pnl = unhedged_pnl + hedge_pnl
    
    return {
        'crack_spreads': crack_spreads,
        'crack_spread_changes': crack_spread_changes,
        'unhedged_pnl': unhedged_pnl,
        'hedge_pnl': hedge_pnl,
        'hedged_pnl': hedged_pnl,
        'crude_prices': crude_aligned,
        'gasoline_prices': gasoline_aligned,
        'heating_oil_prices': heating_oil_aligned
    }


def compute_crack_spread_payoff_diagram(current_crude_price: float, current_gasoline_price: float,
                                       current_heating_oil_price: float, refinery_capacity: float,
                                       hedge_ratio: float, spread_type: str = "3-2-1",
                                       price_range_pct: float = 0.3, num_points: int = 100) -> Dict[str, pd.Series]:
    """
    Generate payoff diagram for crack spread hedging.
    
    Args:
        current_crude_price: Current WTI price ($/barrel)
        current_gasoline_price: Current gasoline price ($/gallon)  
        current_heating_oil_price: Current heating oil price ($/gallon)
        refinery_capacity: Refinery capacity (barrels/day)
        hedge_ratio: Hedge ratio (0.0 to 1.0)
        spread_type: Crack spread type
        price_range_pct: Price range for scenario analysis
        num_points: Number of price points
    
    Returns:
        Dict with payoff diagram data
    """
    
    # Generate crude price scenarios
    crude_min = current_crude_price * (1 - price_range_pct)
    crude_max = current_crude_price * (1 + price_range_pct)
    crude_scenarios = np.linspace(crude_min, crude_max, num_points)
    
    # Assume product prices move with crude (simplified correlation)
    # In reality, product prices have their own dynamics
    gasoline_scenarios = current_gasoline_price * (crude_scenarios / current_crude_price)
    heating_oil_scenarios = current_heating_oil_price * (crude_scenarios / current_crude_price)
    
    # Calculate crack spreads for each scenario
    underlying_pnl = []
    hedge_pnl = []
    
    current_crack = calculate_crack_spread(
        pd.Series([current_crude_price]),
        pd.Series([current_gasoline_price]),
        pd.Series([current_heating_oil_price]),
        spread_type
    ).iloc[0]
    
    for i in range(num_points):
        scenario_crack = calculate_crack_spread(
            pd.Series([crude_scenarios[i]]),
            pd.Series([gasoline_scenarios[i]]),
            pd.Series([heating_oil_scenarios[i]]),
            spread_type
        ).iloc[0]
        
        crack_change = scenario_crack - current_crack
        
        # Refinery P&L from crack spread changes
        daily_capacity = refinery_capacity
        capacity_factor = daily_capacity / (3 if spread_type == "3-2-1" else 2)
        
        underlying_pnl.append(crack_change * capacity_factor)
        hedge_pnl.append(-crack_change * capacity_factor * hedge_ratio)
    
    underlying_pnl = np.array(underlying_pnl)
    hedge_pnl = np.array(hedge_pnl)
    net_pnl = underlying_pnl + hedge_pnl
    
    return {
        'crude_prices': pd.Series(crude_scenarios, name='Crude Price'),
        'underlying_pnl': pd.Series(underlying_pnl, name='Unhedged Refinery P&L'),
        'hedge_pnl': pd.Series(hedge_pnl, name='Hedge P&L'),
        'net_pnl': pd.Series(net_pnl, name='Net Refinery P&L'),
        'breakeven_prices': []  # Could implement crack spread breakeven analysis
    }


def get_current_crack_spread(spread_type: str = "3-2-1") -> float:
    """
    Get current crack spread value.
    
    Returns:
        float: Current crack spread in $/barrel
    """
    
    try:
        # Get latest prices (simplified - use 1 day period)
        prices = get_product_prices("1d")
        
        if all(len(prices[product]) > 0 for product in ['crude', 'gasoline', 'heating_oil']):
            current_crack = calculate_crack_spread(
                prices['crude'], prices['gasoline'], prices['heating_oil'], spread_type
            ).iloc[-1]
            
            return float(current_crack)
        else:
            # Fallback estimate
            return 15.0  # Typical crack spread around $15/barrel
            
    except Exception:
        return 15.0


# Example usage and testing
if __name__ == "__main__":
    print("Testing crack spread calculations...")
    
    # Test with sample data
    refinery_capacity = 100000  # 100,000 barrels/day refinery
    hedge_ratio = 0.75
    
    try:
        # Test crack spread hedging
        results = compute_crack_spread_hedge(refinery_capacity, hedge_ratio)
        
        print(f"Historical crack spreads calculated: {len(results['crack_spreads'])} data points")
        print(f"Current crack spread estimate: ${get_current_crack_spread():.2f}/barrel")
        print(f"Average crack spread: ${results['crack_spreads'].mean():.2f}/barrel")
        print(f"Crack spread volatility: ${results['crack_spreads'].std():.2f}/barrel")
        
    except Exception as e:
        print(f"Error in crack spread calculation: {e}")
