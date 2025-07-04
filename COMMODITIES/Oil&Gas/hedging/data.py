"""
hedging/data.py

Module for fetching live/historical commodity prices using Yahoo Finance.
Handles data retrieval for WTI, Brent, and Natural Gas commodities.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict


# Commodity ticker mapping for Yahoo Finance
COMMODITY_TICKERS: Dict[str, str] = {
    "WTI": "CL=F",
    "Brent": "BZ=F", 
    "Natural Gas": "NG=F"
}


def get_prices(commodity: str, period: str = "1y") -> pd.Series:
    """
    Fetch historical prices for a specified commodity.
    
    Args:
        commodity (str): Commodity name ("WTI", "Brent", "Natural Gas")
        period (str): Time period for historical data (default: "1y")
                     Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    Returns:
        pd.Series: Time series of daily closing prices
        
    Raises:
        KeyError: If commodity is not supported
        ValueError: If no data is retrieved
    """
    
    # Validate commodity input
    if commodity not in COMMODITY_TICKERS:
        available_commodities = ", ".join(COMMODITY_TICKERS.keys())
        raise KeyError(f"Commodity '{commodity}' not supported. Available: {available_commodities}")
    
    # Get Yahoo Finance ticker
    ticker = COMMODITY_TICKERS[commodity]
    
    try:
        # Fetch historical data
        commodity_data = yf.download(ticker, period=period, progress=False)
        
        # Extract closing prices
        if commodity_data.empty:
            raise ValueError(f"No data retrieved for {commodity} ({ticker})")
        
        # Return closing prices as Series
        prices = commodity_data['Close'].dropna()
        
        if prices.empty:
            raise ValueError(f"No valid price data for {commodity}")
            
        return prices
        
    except Exception as e:
        raise ValueError(f"Error fetching data for {commodity}: {str(e)}")


def get_current_price(commodity: str) -> float:
    """
    Get the most recent price for a commodity.
    
    Args:
        commodity (str): Commodity name ("WTI", "Brent", "Natural Gas")
    
    Returns:
        float: Most recent closing price
        
    Raises:
        KeyError: If commodity is not supported
        ValueError: If no current price is available
    """
    
    try:
        # Get recent prices (last 5 days to ensure we have data)
        prices = get_prices(commodity, period="5d")
        
        if prices.empty:
            raise ValueError(f"No recent price data available for {commodity}")
        
        # Return most recent price
        current_price = float(prices.iloc[-1])
        
        # Ensure it's a valid number
        if pd.isna(current_price) or current_price <= 0:
            raise ValueError(f"Invalid current price for {commodity}: {current_price}")
            
        return current_price
        
    except Exception as e:
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
    Get list of available commodities.
    
    Returns:
        list: List of supported commodity names
    """
    return list(COMMODITY_TICKERS.keys())


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
