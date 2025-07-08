"""
hedging/options_math.py - Advanced Options Mathematics Module

This module provides comprehensive options pricing and Greeks calculations
for commodity hedging strategies.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class BlackScholesCalculator:
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma)
        return d1_val - sigma * np.sqrt(T)
    
    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """ Calculate European option price using Black-Scholes formula. """
        if T <= 0:
            # At expiration
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        if sigma <= 0:
            return 0.0
            
        try:
            d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma)
            d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
            
            return max(price, 0.0)  # Ensure non-negative price
            
        except Exception as e:
            print(f"Error calculating option price: {e}")
            return 0.0
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """ Calculate all Greeks for an option. """
        if T <= 0:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        if sigma <= 0:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        try:
            d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma)
            d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma)
            
            sqrt_T = np.sqrt(T)
            exp_rT = np.exp(-r * T)
            
            # DELTA
            if option_type.lower() == 'call':
                delta = norm.cdf(d1_val)
            else:  # put
                delta = norm.cdf(d1_val) - 1.0
            
            # GAMMA 
            gamma = norm.pdf(d1_val) / (S * sigma * sqrt_T)
            
            # THETA
            theta_common = (-S * norm.pdf(d1_val) * sigma / (2 * sqrt_T) - 
                           r * K * exp_rT * norm.cdf(d2_val if option_type.lower() == 'call' else -d2_val))
            
            if option_type.lower() == 'call':
                theta = theta_common
            else:  # put
                theta = theta_common + r * K * exp_rT
             
            theta = theta / 365.0
            
            # VEGA 
            vega = S * norm.pdf(d1_val) * sqrt_T / 100.0  # Per 1% change in volatility
            
            # RHO
            if option_type.lower() == 'call':
                rho = K * T * exp_rT * norm.cdf(d2_val) / 100.0  # Per 1% change in rate
            else:  # put
                rho = -K * T * exp_rT * norm.cdf(-d2_val) / 100.0
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            print(f"Error calculating Greeks: {e}")
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
    
    @staticmethod
    def calculate_implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str, max_iterations: int = 100) -> float:
        """ Calculate implied volatility from market price using Brent's method. """
        if T <= 0 or market_price <= 0:
            return 0.0
        
        # Check if option is in-the-money at expiration
        if option_type.lower() == 'call':
            intrinsic_value = max(S - K * np.exp(-r * T), 0)
        else:
            intrinsic_value = max(K * np.exp(-r * T) - S, 0)
        
        if market_price <= intrinsic_value:
            return 0.0
        
        def objective(sigma):
            try:
                theoretical_price = BlackScholesCalculator.calculate_option_price(
                    S, K, T, r, sigma, option_type
                )
                return theoretical_price - market_price
            except:
                return float('inf')
        
        try:
            # Use Brent's method to find implied volatility
            implied_vol = brentq(objective, 0.001, 5.0, maxiter=max_iterations)
            return max(implied_vol, 0.0)
        except:
            # If Brent's method fails, try a simple search
            try:
                vol_range = np.linspace(0.01, 3.0, 100)
                errors = []
                
                for vol in vol_range:
                    price = BlackScholesCalculator.calculate_option_price(
                        S, K, T, r, vol, option_type
                    )
                    errors.append(abs(price - market_price))
                
                best_idx = np.argmin(errors)
                return vol_range[best_idx]
            except:
                return 0.25  # Default volatility if all methods fail


class AsianOptionCalculator:
    """
    Calculator for Asian (average price) options using Monte Carlo simulation.
    """
    
    @staticmethod
    def monte_carlo_asian_price(S0: float, K: float, T: float, r: float, sigma: float, n_steps: int, n_simulations: int,
                              option_type: str, average_type: str = "arithmetic") -> Dict[str, float]:

        """ Price Asian option using Monte Carlo simulation. """
        if T <= 0 or n_steps <= 0 or n_simulations <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'confidence_interval': (0.0, 0.0)}
        
        dt = T / n_steps
        discount_factor = np.exp(-r * T)
        
        # Pre-calculate random numbers for efficiency
        np.random.seed(42)  # For reproducibility
        random_numbers = np.random.standard_normal((n_simulations, n_steps))
        
        payoffs = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            # Simulate price path
            prices = np.zeros(n_steps + 1)
            prices[0] = S0
            
            for j in range(n_steps):
                prices[j + 1] = prices[j] * np.exp(
                    (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_numbers[i, j]
                )
            
            # Calculate average price (excluding initial price)
            if average_type == "arithmetic":
                avg_price = np.mean(prices[1:])
            else:  # geometric
                avg_price = np.exp(np.mean(np.log(prices[1:])))
            
            # Calculate payoff
            if option_type.lower() == 'call':
                payoffs[i] = max(avg_price - K, 0)
            else:  # put
                payoffs[i] = max(K - avg_price, 0)
        
        # Calculate statistics
        discounted_payoffs = payoffs * discount_factor
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        # 95% confidence interval
        z_score = 1.96
        confidence_interval = (
            option_price - z_score * std_error,
            option_price + z_score * std_error
        )
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': confidence_interval
        }
    
    @staticmethod
    def calculate_asian_greeks(S0: float, K: float, T: float, r: float, sigma: float, n_steps: int, n_simulations: int,
                             option_type: str, average_type: str = "arithmetic") -> Dict[str, float]:
        """ Calculate Greeks for Asian options using finite difference method. """
        # Base price
        base_result = AsianOptionCalculator.monte_carlo_asian_price(
            S0, K, T, r, sigma, n_steps, n_simulations, option_type, average_type
        )
        base_price = base_result['price']
        
        # Delta (finite difference)
        h_S = 0.01 * S0
        up_price = AsianOptionCalculator.monte_carlo_asian_price(
            S0 + h_S, K, T, r, sigma, n_steps, n_simulations, option_type, average_type
        )['price']
        down_price = AsianOptionCalculator.monte_carlo_asian_price(
            S0 - h_S, K, T, r, sigma, n_steps, n_simulations, option_type, average_type
        )['price']
        delta = (up_price - down_price) / (2 * h_S)
        
        # Gamma (second derivative)
        gamma = (up_price - 2 * base_price + down_price) / (h_S**2)
        
        # Vega (sensitivity to volatility)
        h_sigma = 0.01
        vega_price = AsianOptionCalculator.monte_carlo_asian_price(
            S0, K, T, r, sigma + h_sigma, n_steps, n_simulations, option_type, average_type
        )['price']
        vega = (vega_price - base_price) / h_sigma
        
        # Theta (time decay)
        if T > 0.01:  # Avoid division by zero
            h_T = 0.01
            theta_price = AsianOptionCalculator.monte_carlo_asian_price(
                S0, K, T - h_T, r, sigma, n_steps, n_simulations, option_type, average_type
            )['price']
            theta = (theta_price - base_price) / h_T
        else:
            theta = 0.0
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365.0,  # Per day
            'vega': vega,
            'rho': 0.0  # Not implemented for Asian options
        }


class OptionsPortfolioAnalyzer:
    """
    Analyzer for portfolios containing multiple options positions.
    """
    
    @staticmethod
    def calculate_portfolio_greeks(positions: List[Dict]) -> Dict[str, float]:
        """ Calculate net Greeks for a portfolio of options. """

        net_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for position in positions:
            try:
                greeks = BlackScholesCalculator.calculate_greeks(
                    position['S'], position['K'], position['T'],
                    position['r'], position['sigma'], position['option_type']
                )
                
                quantity = position.get('quantity', 1)
                
                for greek_name, greek_value in greeks.items():
                    net_greeks[greek_name] += greek_value * quantity
                    
            except Exception as e:
                print(f"Error calculating Greeks for position: {e}")
                continue
        
        return net_greeks
    
    @staticmethod
    def calculate_portfolio_var(positions: List[Dict], confidence_level: float = 0.95, time_horizon: float = 1.0) -> Dict[str, float]:
        """ Calculate Value at Risk for options portfolio using delta-gamma approximation. """
        # Calculate portfolio Greeks
        portfolio_greeks = OptionsPortfolioAnalyzer.calculate_portfolio_greeks(positions)
        
        # Assume average volatility and correlation for portfolio
        avg_volatility = np.mean([pos.get('sigma', 0.25) for pos in positions])
        avg_spot = np.mean([pos.get('S', 100) for pos in positions])
        
        # Delta-normal VaR
        delta_var = abs(portfolio_greeks['delta']) * avg_spot * avg_volatility * np.sqrt(time_horizon)
        
        # Gamma adjustment
        gamma_adjustment = 0.5 * portfolio_greeks['gamma'] * (avg_spot * avg_volatility * np.sqrt(time_horizon))**2
        
        # Total VaR with gamma adjustment
        total_var = delta_var + gamma_adjustment
        
        # Apply confidence level
        z_score = norm.ppf(confidence_level)
        var_estimate = total_var * z_score
        
        return {
            'delta_var': delta_var,
            'gamma_adjustment': gamma_adjustment,
            'total_var': var_estimate,
            'portfolio_delta': portfolio_greeks['delta'],
            'portfolio_gamma': portfolio_greeks['gamma']
        }


class VolatilityEstimator:
    """
    Volatility estimation utilities for options pricing.
    """
    
    @staticmethod
    def historical_volatility(prices: pd.Series, window: int = 30) -> float:
        """ Calculate historical volatility from price series. """
        if len(prices) < window + 1:
            return 0.25  # Default volatility
        
        try:
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            # Calculate rolling volatility
            rolling_vol = log_returns.rolling(window=window).std()
            
            # Annualize (assuming 252 trading days per year)
            annualized_vol = rolling_vol.iloc[-1] * np.sqrt(252)
            
            return annualized_vol if not np.isnan(annualized_vol) else 0.25
            
        except Exception as e:
            print(f"Error calculating historical volatility: {e}")
            return 0.25
    
    @staticmethod
    def garch_volatility(prices: pd.Series, forecast_periods: int = 1) -> float:
        """ Simple GARCH(1,1) volatility forecast. """
        try:
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            if len(log_returns) < 50:
                return VolatilityEstimator.historical_volatility(prices)
            
            # Simple GARCH(1,1) parameters (you can make these more sophisticated)
            alpha = 0.1
            beta = 0.85
            omega = 0.000001
            
            # Calculate conditional variance
            returns_squared = log_returns**2
            conditional_variance = np.zeros(len(returns_squared))
            conditional_variance[0] = returns_squared.var()
            
            for i in range(1, len(returns_squared)):
                conditional_variance[i] = (omega + 
                                         alpha * returns_squared.iloc[i-1] + 
                                         beta * conditional_variance[i-1])
            
            # Forecast volatility
            last_variance = conditional_variance[-1]
            last_return_squared = returns_squared.iloc[-1]
            
            forecasted_variance = omega + alpha * last_return_squared + beta * last_variance
            
            # Annualize
            forecasted_volatility = np.sqrt(forecasted_variance * 252)
            
            return forecasted_volatility if not np.isnan(forecasted_volatility) else 0.25
            
        except Exception as e:
            print(f"Error calculating GARCH volatility: {e}")
            return VolatilityEstimator.historical_volatility(prices)


# Utility functions for common operations
def get_risk_free_rate() -> float:
    """Get current risk-free rate (simplified - in practice, you'd fetch from API)."""
    return 0.03  # 3% default rate

def get_commodity_volatility(commodity: str, prices: pd.Series = None) -> float:
    """Get volatility estimate for commodity."""
    if prices is not None and len(prices) > 30:
        return VolatilityEstimator.historical_volatility(prices)
    
    # Default volatilities by commodity
    volatility_map = {
        'WTI Crude Oil': 0.35,
        'Brent Crude Oil': 0.32,
        'Natural Gas': 0.45,
        'Gold': 0.20,
        'Silver': 0.30,
        'Copper': 0.25
    }
    
    return volatility_map.get(commodity, 0.25)

def time_to_expiration(expiry_months: int) -> float:
    """Convert months to years for options calculations."""
    return expiry_months / 12.0


# Example usage and testing
if __name__ == "__main__":
    print("=== Options Math Module Test ===")
    
    S = 75.0  
    K = 80.0 
    T = 0.25  
    r = 0.03  
    sigma = 0.35  
    
    call_price = BlackScholesCalculator.calculate_option_price(S, K, T, r, sigma, 'call')
    call_greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, 'call')
    
    print(f"Call Option Price: ${call_price:.2f}")
    print(f"Call Greeks: {call_greeks}")
    
    # Test Asian option pricing
    asian_result = AsianOptionCalculator.monte_carlo_asian_price(
        S, K, T, r, sigma, 50, 10000, 'call', 'arithmetic'
    )
    
    print(f"\nAsian Call Option Price: ${asian_result['price']:.2f}")
    print(f"Standard Error: ${asian_result['std_error']:.2f}")
    print(f"95% Confidence Interval: ${asian_result['confidence_interval'][0]:.2f} - ${asian_result['confidence_interval'][1]:.2f}")
    
    # Test implied volatility
    market_price = 2.50
    implied_vol = BlackScholesCalculator.calculate_implied_volatility(
        market_price, S, K, T, r, 'call'
    )
    
    print(f"\nImplied Volatility: {implied_vol:.2%}")
    print("=== Test Complete ===")
