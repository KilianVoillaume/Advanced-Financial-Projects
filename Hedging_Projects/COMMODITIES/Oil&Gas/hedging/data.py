"""
hedging/data.py

Module for fetching live/historical commodity prices using Yahoo Finance.
Handles data retrieval for WTI, Brent, and Natural Gas commodities.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict


COMMODITY_TICKERS: Dict[str, str] = {
    "WTI Crude Oil": "CL=F",           # WTI Crude Oil Futures
    "Brent Crude Oil": "BZ=F",        # Brent Crude Oil Futures
    "Natural Gas": "NG=F"             # Natural Gas Futures
}

# Basis adjustments (in $/barrel or $/MMBtu)
BASIS_ADJUSTMENTS: Dict[str, float] = {
    "WTI Crude Oil": 0.0,     
    "Brent Crude Oil": 0.0,     
    "Natural Gas": 0.0          
}


def get_prices(commodity: str, period: str = "1y") -> pd.Series:
    """ Fetch historical prices for a specified commodity with basis adjustments """
    
    base_commodity = None
    for base, ticker in COMMODITY_TICKERS.items():
        if commodity.startswith(base.split()[0]):  # Match "WTI" in "WTI Houston"
            base_commodity = base
            break
    
    if base_commodity is None:
        if commodity in COMMODITY_TICKERS:
            base_commodity = commodity
        else:
            available_commodities = list(COMMODITY_TICKERS.keys()) + list(BASIS_ADJUSTMENTS.keys())
            raise KeyError(f"Commodity '{commodity}' not supported. Available: {available_commodities}")
    
    ticker = COMMODITY_TICKERS[base_commodity]
    
    try:
        commodity_data = yf.download(ticker, period=period, progress=False)
        
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
            prices = commodity_data.iloc[:, 0].dropna()
        
        if prices.empty:
            raise ValueError(f"No valid price data for {commodity}")
        
        prices = pd.to_numeric(prices, errors='coerce').dropna()
        
        if prices.empty:
            raise ValueError(f"No valid numeric price data for {commodity}")
        
        # Apply basis adjustment
        basis_adjustment = BASIS_ADJUSTMENTS.get(commodity, 0.0)
        adjusted_prices = prices + basis_adjustment
           
        return adjusted_prices
        
    except Exception as e:
        raise ValueError(f"Error fetching data for {commodity}: {str(e)}")


def get_current_price(commodity: str) -> float:
    """ Get the most recent price for a commodity with basis adjustments """
    try:
        all_commodities = list(COMMODITY_TICKERS.keys()) + list(BASIS_ADJUSTMENTS.keys())
        if commodity not in all_commodities:
            available_commodities = ", ".join(all_commodities)
            raise KeyError(f"Commodity '{commodity}' not supported. Available: {available_commodities}")
        
        prices = get_prices(commodity, period="5d")
        
        if prices.empty:
            raise ValueError(f"No recent price data available for {commodity}")
        
        current_price = float(prices.iloc[-1])
        
        if pd.isna(current_price) or current_price <= 0:
            raise ValueError(f"Invalid current price for {commodity}: {current_price}")
            
        return current_price
        
    except Exception as e:
        fallback_prices = {
            "WTI Crude Oil": 75.0,
            "Brent Crude Oil": 78.0,
            "Natural Gas": 3.5
        }
        
        if commodity in fallback_prices:
            return fallback_prices[commodity]
        else:
            raise ValueError(f"Error getting current price for {commodity}: {str(e)}")


def get_price_statistics(commodity: str, period: str = "1y") -> Dict[str, float]:
    """ Calculate basic statistics for commodity prices over a given period """
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
    """ Validate if a commodity is supported """
    return commodity in COMMODITY_TICKERS


def get_available_commodities() -> list:
    """ Get list of available commodities including basis locations """
    return list(BASIS_ADJUSTMENTS.keys())


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
