"""
Module for fetching live/historical commodity prices using Yahoo Finance.
Handles data retrieval for WTI, Brent, and Natural Gas commodities.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict


COMMODITY_TICKERS: Dict[str, str] = {
    "WTI": "CL=F",
    "Brent": "BZ=F", 
    "Natural Gas": "NG=F"
}


def get_prices(commodity: str, period: str = "1y") -> pd.Series:
    """
    Fetch historical prices for a specified commodity.
    """
    
    if commodity not in COMMODITY_TICKERS:
        available_commodities = ", ".join(COMMODITY_TICKERS.keys())
        raise KeyError(f"Commodity '{commodity}' not supported. Available: {available_commodities}")
    
    ticker = COMMODITY_TICKERS[commodity]
    
    try:
        commodity_data = yf.download(ticker, period=period, progress=False)
        
        if commodity_data.empty:
            raise ValueError(f"No data retrieved for {commodity} ({ticker})")
        
        prices = commodity_data['Close'].dropna()
        
        if prices.empty:
            raise ValueError(f"No valid price data for {commodity}")
            
        return prices
        
    except Exception as e:
        raise ValueError(f"Error fetching data for {commodity}: {str(e)}")


def get_current_price(commodity: str) -> float:
    """
    Get the most recent price for a commodity.
    """
    
    try:
        # Get recent prices (last 5 days to ensure we have data)
        prices = get_prices(commodity, period="5d")
        
        if prices.empty:
            raise ValueError(f"No recent price data available for {commodity}")
        
        # Return most recent price
        current_price = float(prices.iloc[-1])
        return current_price
        
    except Exception as e:
        raise ValueError(f"Error getting current price for {commodity}: {str(e)}")


def get_price_statistics(commodity: str, period: str = "1y") -> Dict[str, float]:
    """
    Calculate basic statistics for commodity prices over a given period.
    """
    
    try:
        prices = get_prices(commodity, period)
        
        if prices.empty:
            raise ValueError(f"No price data available for {commodity}")
        
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
    return commodity in COMMODITY_TICKERS

def get_available_commodities() -> list:
    return list(COMMODITY_TICKERS.keys())

if __name__ == "__main__":
    print("Testing hedging/data.py module...")
    
    print(f"Available commodities: {get_available_commodities()}")
    
    try:
        wti_prices = get_prices("WTI")
        print(f"WTI prices fetched: {len(wti_prices)} data points")
        print(f"Date range: {wti_prices.index[0].date()} to {wti_prices.index[-1].date()}")
        print(f"Price range: ${wti_prices.min():.2f} - ${wti_prices.max():.2f}")
        
        current_wti = get_current_price("WTI")
        print(f"Current WTI price: ${current_wti:.2f}")
        
        wti_stats = get_price_statistics("WTI")
        print(f"WTI volatility: {wti_stats['volatility']:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
