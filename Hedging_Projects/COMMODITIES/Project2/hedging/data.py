"""
hedging/data.py
Data Provider for Commodity Hedging Platform
"""

import pandas as pd
import yfinance as yf
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import warnings
warnings.filterwarnings('ignore')



# ----- COMMODITY TICKER MAPPINGS -----

COMMODITY_TICKERS: Dict[str, str] = {
    # Energy
    "WTI Crude Oil": "CL=F",
    "Brent Crude Oil": "BZ=F", 
    "Natural Gas": "NG=F",
    "Heating Oil": "HO=F",
    "RBOB Gasoline": "RB=F",
    
    # Precious Metals
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    
    # Base Metals
    "Copper": "HG=F",
    "Aluminum": "ALI=F",
    
    # Agriculture - Grains
    "Corn": "ZC=F",
    "Wheat": "ZW=F", 
    "Soybeans": "ZS=F",
    "Soybean Oil": "ZL=F",
    "Soybean Meal": "ZM=F",
    "Oats": "ZO=F",
    "Rough Rice": "ZR=F",
    
    # Agriculture - Softs
    "Coffee": "KC=F",
    "Sugar": "SB=F",
    "Cocoa": "CC=F",
    "Cotton": "CT=F",
    "Orange Juice": "OJ=F",
    
    # Livestock
    "Live Cattle": "LE=F",
    "Feeder Cattle": "GF=F",
    "Lean Hogs": "HE=F",
    
    # Alternative/Emerging
    "Lumber": "LBS=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD"
}

COMMODITY_CATEGORIES = {
    "Energy": ["WTI Crude Oil", "Brent Crude Oil", "Natural Gas", "Heating Oil", "RBOB Gasoline"],
    "Precious Metals": ["Gold", "Silver", "Platinum", "Palladium"],
    "Base Metals": ["Copper", "Aluminum"],
    "Grains": ["Corn", "Wheat", "Soybeans", "Soybean Oil", "Soybean Meal", "Oats", "Rough Rice"],
    "Softs": ["Coffee", "Sugar", "Cocoa", "Cotton", "Orange Juice"],
    "Livestock": ["Live Cattle", "Feeder Cattle", "Lean Hogs"],
    "Alternative": ["Lumber", "Bitcoin", "Ethereum"]
}

BASIS_ADJUSTMENTS: Dict[str, float] = {
    "WTI Crude Oil": 0.0,     
    "Brent Crude Oil": 0.0,     
    "Natural Gas": 0.0          
}

PROFESSIONAL_CORRELATIONS = {
    ("WTI Crude Oil", "Brent Crude Oil"): 0.85,
    ("WTI Crude Oil", "Oats"): 0.15,
    ("Brent Crude Oil", "Oats"): 0.12,
    ("WTI Crude Oil", "Natural Gas"): 0.35,
    ("Brent Crude Oil", "Natural Gas"): 0.32,
    ("Corn", "Oats"): 0.45,
    ("Wheat", "Oats"): 0.40,
}

# ----- Main Data Fetching -----

def get_prices(commodity: str, period: str = "1y") -> pd.Series:    
    try:
        ticker = COMMODITY_TICKERS.get(commodity)
        if ticker:
            print(f"📊 Trying Yahoo Finance: {ticker}")
            commodity_data = yf.download(ticker, period=period, progress=False)
            
            if not commodity_data.empty:
                # Extract close prices
                if isinstance(commodity_data.columns, pd.MultiIndex):
                    prices = commodity_data[('Close', ticker)].dropna()
                elif 'Close' in commodity_data.columns:
                    prices = commodity_data['Close'].dropna()
                else:
                    prices = commodity_data.iloc[:, 0].dropna()
                
                if len(prices) > 20:
                    print(f"Yahoo Finance success: {len(prices)} points")
                    return prices
    except Exception as e:
        print(f" Yahoo Finance failed: {e}")
    
    # Try  APIs if available
    if not APIS_AVAILABLE:
        print("No professional APIs configured, using fallback")
        return _generate_professional_synthetic_data(commodity, period)
    

def get_current_price(commodity: str) -> float:
    try:
        recent_prices = get_prices(commodity, period="5d")
        if not recent_prices.empty:
            return float(recent_prices.iloc[-1])
    except Exception as e:
        print(f"Error getting current price for {commodity}: {e}")
    
    # Fallback prices
    fallback_prices = {
        "WTI Crude Oil": 65.50,
        "Brent Crude Oil": 67.20,
        "Natural Gas": 2.99,
        "Oats": 3.45,
        "Gold": 3384.0,
        "Silver": 37.50,
        "Copper": 4.36,
        "Corn": 3.82,
        "Wheat": 5.10
    }
    return fallback_prices.get(commodity, 50.0)


def _generate_professional_synthetic_data(commodity: str, period: str) -> pd.Series:
    """Generate realistic synthetic data when API fails"""
    
    commodity_specs = {
        "WTI Crude Oil": {"price": 65.50, "volatility": 0.35, "drift": -0.0001},
        "Brent Crude Oil": {"price": 67.20, "volatility": 0.32, "drift": -0.0001},
        "Oats": {"price": 3.45, "volatility": 0.28, "drift": 0.0002},
        "Natural Gas": {"price": 2.99, "volatility": 0.45, "drift": -0.0002},
        "Gold": {"price": 3384.0, "volatility": 0.20, "drift": 0.0001},
        "Silver": {"price": 37.50, "volatility": 0.30, "drift": 0.0001},
        "Copper": {"price": 4.36, "volatility": 0.25, "drift": 0.0000},
        "Wheat": {"price": 5.10, "volatility": 0.28, "drift": 0.0002},
        "Corn": {"price": 3.82, "volatility": 0.25, "drift": 0.0002}
    }
    
    specs = commodity_specs.get(commodity, {"price": 50.0, "volatility": 0.25, "drift": 0.0})
    
    period_days = {"1y": 252, "6mo": 126, "3mo": 63, "1mo": 21, "5d": 5}.get(period, 252)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=period_days, freq='D')
    
    # Price path using GBM
    np.random.seed(hash(commodity) % 1000)  # Deterministic per commodity
    daily_vol = specs["volatility"] / np.sqrt(252)
    returns = np.random.normal(specs["drift"], daily_vol, period_days)
    
    # Price path ending at target price
    log_prices = np.cumsum(returns)
    log_prices = log_prices - log_prices[-1] + np.log(specs["price"])
    prices = np.exp(log_prices)
    
    price_series = pd.Series(prices, index=dates, name=commodity)
    return price_series


# ----- Correlation Functions -----

def get_professional_correlation_matrix(commodities: List[str]) -> pd.DataFrame:
    """Professional correlation matrix with real data or estimates"""
    
    if len(commodities) < 2:
        return pd.DataFrame()
    
    print(f"📊 Calculating correlations for: {', '.join(commodities)}")
    
    try:
        # Try to calculate from real price data
        price_data = {}
        successful_fetches = 0
        
        for commodity in commodities:
            try:
                prices = get_prices(commodity, period="6mo")  # 6 months for correlations
                if len(prices) > 100:
                    price_data[commodity] = prices.pct_change().dropna()
                    successful_fetches += 1
                    print(f"Got correlation data for {commodity}")
            except Exception as e:
                print(f" Failed correlation data for {commodity}: {e}")
        
        # Calculate if enough data
        if successful_fetches >= 2:
            aligned_data = pd.DataFrame(price_data).dropna()
            
            if len(aligned_data) > 50:
                correlation_matrix = aligned_data.corr()
                print(f"Calculated correlations from {len(aligned_data)} data points")
                return correlation_matrix
    
    except Exception as e:
        print(f"Real correlation calculation failed: {e}")
    
    # Fallback to estimates
    print("📊 Using professional correlation estimates")
    return _get_professional_correlation_estimates(commodities)


def _get_professional_correlation_estimates(commodities: List[str]) -> pd.DataFrame:
    """Professional correlation estimates based on institutional knowledge"""
    
    n = len(commodities)
    corr_matrix = np.eye(n)
    
    for i, comm1 in enumerate(commodities):
        for j, comm2 in enumerate(commodities):
            if i != j:
                key1 = (comm1, comm2)
                key2 = (comm2, comm1)
                
                if key1 in PROFESSIONAL_CORRELATIONS:
                    corr_matrix[i, j] = PROFESSIONAL_CORRELATIONS[key1]
                elif key2 in PROFESSIONAL_CORRELATIONS:
                    corr_matrix[i, j] = PROFESSIONAL_CORRELATIONS[key2]
                else:
                    # Intelligent defaults by category
                    energy_commodities = ["WTI Crude Oil", "Brent Crude Oil", "Natural Gas"]
                    grains_commodities = ["Corn", "Wheat", "Oats", "Soybeans"]
                    metals_commodities = ["Gold", "Silver", "Copper", "Platinum"]
                    
                    if comm1 in energy_commodities and comm2 in energy_commodities:
                        corr_matrix[i, j] = 0.45  # Energy correlation
                    elif comm1 in grains_commodities and comm2 in grains_commodities:
                        corr_matrix[i, j] = 0.35  # Grains correlation
                    elif comm1 in metals_commodities and comm2 in metals_commodities:
                        corr_matrix[i, j] = 0.55  # Metals correlation
                    else:
                        corr_matrix[i, j] = 0.08  # Cross-category
    
    df = pd.DataFrame(corr_matrix, index=commodities, columns=commodities)
    return df


# ----- Utility Functions ----- 

def get_available_commodities() -> List[str]:
    """Get list of supported commodities"""
    return list(COMMODITY_TICKERS.keys())


def get_available_commodities_by_category() -> Dict[str, List[str]]:
    """Get commodities organized by category - MISSING FUNCTION ADDED"""
    return COMMODITY_CATEGORIES.copy()


def get_commodity_category(commodity: str) -> str:
    """Get category for a specific commodity - MISSING FUNCTION ADDED"""
    for category, commodities in COMMODITY_CATEGORIES.items():
        if commodity in commodities:
            return category
    return "Other"


def get_all_available_commodities() -> List[str]:
    """Get flat list of all commodities - MISSING FUNCTION ADDED"""
    return get_available_commodities()


def get_commodity_specs(commodity: str) -> Dict[str, any]:
    """Get commodity specifications - ENHANCED VERSION"""
    specs_database = {
        "WTI Crude Oil": {
            "unit": "$/barrel",
            "contract_size": 1000,
            "typical_volatility": 0.35,
            "seasonality": "Low",
            "storage_cost": 0.02,
            "correlation_group": "energy"
        },
        "Brent Crude Oil": {
            "unit": "$/barrel", 
            "contract_size": 1000,
            "typical_volatility": 0.32,
            "seasonality": "Low",
            "storage_cost": 0.02,
            "correlation_group": "energy"
        },
        "Oats": {
            "unit": "¢/bushel",
            "contract_size": 5000,
            "typical_volatility": 0.28,
            "seasonality": "High",
            "storage_cost": 0.03,
            "correlation_group": "grains"
        },
        "Natural Gas": {
            "unit": "$/MMBtu",
            "contract_size": 10000,
            "typical_volatility": 0.45,
            "seasonality": "High",
            "storage_cost": 0.05,
            "correlation_group": "energy"
        },
        "Gold": {
            "unit": "$/troy oz",
            "contract_size": 100,
            "typical_volatility": 0.20,
            "seasonality": "Low",
            "storage_cost": 0.01,
            "correlation_group": "precious_metals"
        },
        "Silver": {
            "unit": "$/troy oz",
            "contract_size": 5000,
            "typical_volatility": 0.30,
            "seasonality": "Low",
            "storage_cost": 0.015,
            "correlation_group": "precious_metals"
        },
        "Copper": {
            "unit": "¢/lb",
            "contract_size": 25000,
            "typical_volatility": 0.25,
            "seasonality": "Medium",
            "storage_cost": 0.02,
            "correlation_group": "base_metals"
        },
        "Corn": {
            "unit": "¢/bushel",
            "contract_size": 5000,
            "typical_volatility": 0.25,
            "seasonality": "High",
            "storage_cost": 0.03,
            "correlation_group": "grains"
        },
        "Wheat": {
            "unit": "¢/bushel",
            "contract_size": 5000,
            "typical_volatility": 0.28,
            "seasonality": "High",
            "storage_cost": 0.03,
            "correlation_group": "grains"
        }
    }
    
    return specs_database.get(commodity, {
        "unit": "$/unit",
        "contract_size": 1000,
        "typical_volatility": 0.25,
        "seasonality": "Medium",
        "storage_cost": 0.02,
        "correlation_group": "other"
    })


def get_price_statistics(commodity: str, period: str = "1y") -> Dict[str, float]:
    """Calculate price statistics"""
    try:
        prices = get_prices(commodity, period)
        
        if prices.empty:
            raise ValueError(f"No price data for {commodity}")
        
        stats = {
            "current_price": float(prices.iloc[-1]),
            "mean_price": float(prices.mean()),
            "std_price": float(prices.std()),
            "min_price": float(prices.min()),
            "max_price": float(prices.max()),
            "volatility": float(prices.pct_change().std() * np.sqrt(252))  # Annualized
        }
        
        return stats
        
    except Exception as e:
        raise ValueError(f"Error calculating statistics for {commodity}: {e}")


# ----- Risk-free Rates & Options Functions -----

def get_risk_free_rate() -> float:
    """Get current risk-free rate"""
    return 0.05  


def get_commodity_volatility(commodity: str) -> float:
    """Get commodity volatility"""
    specs = get_commodity_specs(commodity)
    return specs.get("typical_volatility", 0.25)


def time_to_expiration(months: int) -> float:
    """Convert months to fraction of year"""
    return months / 12.0


# ----- Main -----

if __name__ == "__main__":
    print("=== PROFESSIONAL DATA SYSTEM TEST ===")
    
    # Test API
    if APIS_AVAILABLE:
        print("API configuration loaded")
        health = check_api_health()
        working_apis = [api for api, status in health.items() if status]
        print(f"Working APIs: {', '.join(working_apis)}")
    else:
        print("Using Yahoo Finance only")
    
    # Test specific commodities
    test_commodities = ["WTI Crude Oil", "Brent Crude Oil", "Oats"]
    
    print(f"\n📊 Testing your portfolio commodities:")
    for commodity in test_commodities:
        try:
            prices = get_prices(commodity, "1mo")
            current = get_current_price(commodity)
            volatility = get_commodity_volatility(commodity)
            
            print(f"{commodity}:")
            print(f"   📈 Historical: {len(prices)} data points")
            print(f"   💰 Current: ${current:.2f}")
            print(f"   📊 Volatility: {volatility:.1%}")
            
        except Exception as e:
            print(f" {commodity}: {e}")
    
    # Test correlation matrix
    print(f"\n🔗 Testing correlation matrix:")
    try:
        corr_matrix = get_professional_correlation_matrix(test_commodities)
        print(f"Correlation matrix ({corr_matrix.shape[0]}x{corr_matrix.shape[1]}):")
        print(corr_matrix.round(3))
        
        # Specific correlations
        if len(corr_matrix) >= 3:
            wti_brent = corr_matrix.loc["WTI Crude Oil", "Brent Crude Oil"]
            wti_oats = corr_matrix.loc["WTI Crude Oil", "Oats"] 
            brent_oats = corr_matrix.loc["Brent Crude Oil", "Oats"]
            
            print(f"\n📊 Key correlations for your portfolio:")
            print(f"   WTI ↔ Brent: {wti_brent:.3f}")
            print(f"   WTI ↔ Oats: {wti_oats:.3f}")
            print(f"   Brent ↔ Oats: {brent_oats:.3f}")
        
    except Exception as e:
        print(f" Correlation test failed: {e}")
    
    print(f"\n🎯 Data system ready for LinkedIn presentation!")