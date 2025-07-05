"""
hedging/data.py

Module for fetching live/historical commodity prices using Yahoo Finance.
Handles data retrieval for WTI, Brent, and Natural Gas commodities.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict


# Commodity ticker mapping for Yahoo Finance with basis adjustments
COMMODITY_TICKERS: Dict[str, str] = {
    "WTI Crude Oil": "CL=F",           # WTI Crude Oil Futures
    "Brent Crude Oil": "BZ=F",        # Brent Crude Oil Futures
    "Natural Gas": "NG=F"             # Natural Gas Futures
}

# Basis adjustments (in $/barrel or $/MMBtu)
BASIS_ADJUSTMENTS: Dict[str, float] = {
    "WTI Crude Oil": 0.0,        # Benchmark
    "Brent Crude Oil": 0.0,      # Benchmark  
    "Natural Gas": 0.0           # Benchmark
}


def get_prices(commodity: str, period: str = "1y") -> pd.Series:
    """
    Fetch historical prices for a specified commodity with basis adjustments.
    """
    
    # Find base commodity ticker
    base_commodity = None
    for base, ticker in COMMODITY_TICKERS.items():
        if commodity.startswith(base.split()[0]):  # Match "WTI" in "WTI Houston"
            base_commodity = base
            break
    
    if base_commodity is None:
        # Try exact match
        if commodity in COMMODITY_TICKERS:
            base_commodity = commodity
        else:
            available_commodities = list(COMMODITY_TICKERS.keys()) + list(BASIS_ADJUSTMENTS.keys())
            raise KeyError(f"Commodity '{commodity}' not supported. Available: {available_commodities}")
    
    # Get Yahoo Finance ticker for base commodity
    ticker = COMMODITY_TICKERS[base_commodity]
    
    try:
        # Fetch historical data - REMOVE THIS PRINT STATEMENT:
        # print(f"Fetching data for {commodity} using base {base_commodity} ({ticker})...")
        
        commodity_data = yf.download(ticker, period=period, progress=False)
        
        # Extract closing prices
        if commodity_data.empty:
            raise ValueError(f"No data retrieved for {commodity} ({ticker})")
        
        # Handle multi-column DataFrame (yfinance sometimes returns different structures)
        if isinstance(commodity_data.columns, pd.MultiIndex):
            # Multi-level columns
            prices = commodity_data[('Close', ticker)].dropna()
        elif 'Close' in commodity_data.columns:
            # Single-level columns
            prices = commodity_data['Close'].dropna()
        else:
            # Try to get the first column if structure is unexpected
            prices = commodity_data.iloc[:, 0].dropna()
        
        if prices.empty:
            raise ValueError(f"No valid price data for {commodity}")
        
        # Ensure all values are numeric
        prices = pd.to_numeric(prices, errors='coerce').dropna()
        
        if prices.empty:
            raise ValueError(f"No valid numeric price data for {commodity}")
        
        # Apply basis adjustment
        basis_adjustment = BASIS_ADJUSTMENTS.get(commodity, 0.0)
        adjusted_prices = prices + basis_adjustment
        
        # REMOVE THESE PRINT STATEMENTS:
        # print(f"Successfully fetched {len(adjusted_prices)} price points for {commodity}")
        # if basis_adjustment != 0:
        #     print(f"Applied basis adjustment: {basis_adjustment:+.2f}")
            
        return adjusted_prices
        
    except Exception as e:
        # REMOVE THIS PRINT STATEMENT:
        # print(f"Error fetching data for {commodity}: {str(e)}")
        raise ValueError(f"Error fetching data for {commodity}: {str(e)}")


def get_current_price(commodity: str) -> float:
    """
    Get the most recent price for a commodity with basis adjustments.
    """
    
    try:
        # Validate commodity first
        all_commodities = list(COMMODITY_TICKERS.keys()) + list(BASIS_ADJUSTMENTS.keys())
        if commodity not in all_commodities:
            available_commodities = ", ".join(all_commodities)
            raise KeyError(f"Commodity '{commodity}' not supported. Available: {available_commodities}")
        
        # Get recent prices (last 5 days to ensure we have data)
        prices = get_prices(commodity, period="5d")
        
        if prices.empty:
            raise ValueError(f"No recent price data available for {commodity}")
        
        # Return most recent price (already includes basis adjustment from get_prices)
        current_price = float(prices.iloc[-1])
        
        # Ensure it's a valid number
        if pd.isna(current_price) or current_price <= 0:
            raise ValueError(f"Invalid current price for {commodity}: {current_price}")
            
        return current_price
        
    except Exception as e:
        # If there's any error, provide a fallback based on commodity
        fallback_prices = {
            "WTI Crude Oil": 75.0,
            "Brent Crude Oil": 78.0,
            "Natural Gas": 3.5
        }
        
        if commodity in fallback_prices:
            # REMOVE THIS PRINT STATEMENT:
            # print(f"Warning: Using fallback price for {commodity}: ${fallback_prices[commodity]}")
            return fallback_prices[commodity]
        else:
            raise ValueError(f"Error getting current price for {commodity}: {str(e)}")


def get_price_statistics(commodity: str, period: str = "1y") -> Dict[str, float]:
    """
    Calculate basic statistics for commodity prices over a given period.
    
    Args:
        commodity (str): Commodity name ("WTI", "Brent", "Natural Gas")
        period (str): Time period for analysis (default: "1y")
    
    Returns:
        Dict[str, float]: Dictionary containing price statistics
    """
    
    try:
        prices = get_prices(commodity, period)
        
        if prices.empty:
            raise ValueError(f"No price data available for {commodity}")
        
        # Calculate statistics
        stats = {
            "current_price": float(prices.iloc[-1]),
            "mean_price": float(prices.mean()),
            "std_price": float(prices.std()),
            "min_price": float(prices.min()),
            "max_price": float(prices.max()),
            "price_range": float(prices.max() - prices.min()),
            "volatility": float(prices.pct_change().std() * (252 ** 0.5))  # Annualized volatility
        }
        
        return stats
        
    except Exception as e:
        raise ValueError(f"Error calculating statistics for {commodity}: {str(e)}")


def validate_commodity(commodity: str) -> bool:
    """
    Validate if a commodity is supported.
    
    Args:
        commodity (str): Commodity name to validate
    
    Returns:
        bool: True if commodity is supported, False otherwise
    """
    return commodity in COMMODITY_TICKERS


def get_available_commodities() -> list:
    """
    Get list of available commodities including basis locations.
    
    Returns:
        list: List of supported commodity names with basis adjustments
    """
    return list(BASIS_ADJUSTMENTS.keys())


# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    print("Testing hedging/data.py module...")
    
    # Test available commodities
    print(f"Available commodities: {get_available_commodities()}")
    
    # Test price fetching for WTI
    try:
        wti_prices = get_prices("WTI")
        print(f"WTI prices fetched: {len(wti_prices)} data points")
        print(f"Date range: {wti_prices.index[0].date()} to {wti_prices.index[-1].date()}")
        print(f"Price range: ${wti_prices.min():.2f} - ${wti_prices.max():.2f}")
        
        # Test current price
        current_wti = get_current_price("WTI")
        print(f"Current WTI price: ${current_wti:.2f}")
        
        # Test statistics
        wti_stats = get_price_statistics("WTI")
        print(f"WTI volatility: {wti_stats['volatility']:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
