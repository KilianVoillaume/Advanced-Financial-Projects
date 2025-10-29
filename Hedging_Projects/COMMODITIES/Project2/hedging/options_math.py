"""
hedging/options_math.py - Advanced Options Mathematics Module

This module provides comprehensive options pricing and Greeks calculations
for single options and multi-leg commodity hedging strategies.
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
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma)
        return d1_val - sigma * np.sqrt(T)
    
    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if T <= 0:
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
            
            return max(price, 0.0)
            
        except Exception as e:
            print(f"Error calculating option price: {e}")
            return 0.0
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        if T <= 0:
            # Handle expiration day properly
            if option_type.lower() == 'call':
                delta = 1.0 if S > K else 0.0
                gamma = 0.0
                theta = 0.0
                vega = 0.0
                rho = 0.0
            else:  # put
                delta = -1.0 if S < K else 0.0
                gamma = 0.0
                theta = 0.0
                vega = 0.0
                rho = 0.0
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }

        # For very short time to expiration, use minimum time
        if T < 0.00274:  # Less than 1 day
            T = 0.00274
        
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
            vega = S * norm.pdf(d1_val) * sqrt_T / 100.0
            
            # RHO
            if option_type.lower() == 'call':
                rho = K * T * exp_rT * norm.cdf(d2_val) / 100.0
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
        if T <= 0 or market_price <= 0:
            return 0.0
        
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
            implied_vol = brentq(objective, 0.001, 5.0, maxiter=max_iterations)
            return max(implied_vol, 0.0)
        except:
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
                return 0.25


# ============================================================================
# MULTI-LEG GREEKS CALCULATOR - Step 3.1 Implementation
# ============================================================================

class MultiLegGreeksCalculator:
    """
    Advanced Greeks calculator for multi-leg options strategies.
    Provides comprehensive risk sensitivities for complex strategies.
    """
    
    @staticmethod
    def calculate_strategy_greeks(multi_leg_strategy, current_price: float) -> Dict[str, float]:
        """
        Calculate net Greeks for multi-leg strategy.
        
        Args:
            multi_leg_strategy: MultiLegStrategy object
            current_price: Current price of underlying asset
            
        Returns:
            Dictionary containing net delta, gamma, theta, vega, rho
        """
        try:
            net_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
            
            risk_free_rate = get_risk_free_rate()
            volatility = get_commodity_volatility(multi_leg_strategy.commodity)
            
            for leg in multi_leg_strategy.legs:
                time_to_expiry = time_to_expiration(leg.expiry_months)
                
                # Calculate Greeks for this leg
                leg_greeks = BlackScholesCalculator.calculate_greeks(
                    current_price, leg.strike_price, time_to_expiry,
                    risk_free_rate, volatility, leg.option_type
                )
                
                # Scale by leg quantity and add to net
                for greek_name in net_greeks:
                    net_greeks[greek_name] += leg_greeks[greek_name] * leg.quantity
            
            # Scale by underlying size and hedge ratio
            multiplier = abs(multi_leg_strategy.underlying_size) * multi_leg_strategy.hedge_ratio
            
            for greek_name in net_greeks:
                net_greeks[greek_name] *= multiplier
            
            return net_greeks
            
        except Exception as e:
            print(f"Error calculating multi-leg Greeks: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    
    @staticmethod
    def calculate_detailed_greeks(multi_leg_strategy, current_price: float) -> Dict[str, any]:
        """
        Calculate detailed Greeks analysis including leg-by-leg breakdown.
        
        Args:
            multi_leg_strategy: MultiLegStrategy object
            current_price: Current price of underlying asset
            
        Returns:
            Dictionary containing detailed Greeks analysis
        """
        risk_free_rate = get_risk_free_rate()
        volatility = get_commodity_volatility(multi_leg_strategy.commodity)
        
        leg_details = []
        net_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        
        for i, leg in enumerate(multi_leg_strategy.legs):
            time_to_expiry = time_to_expiration(leg.expiry_months)
            
            # Calculate individual leg Greeks
            leg_greeks = BlackScholesCalculator.calculate_greeks(
                current_price, leg.strike_price, time_to_expiry,
                risk_free_rate, volatility, leg.option_type
            )
            
            # Scale by quantity
            scaled_greeks = {k: v * leg.quantity for k, v in leg_greeks.items()}
            
            # Add to net
            for greek_name in net_greeks:
                net_greeks[greek_name] += scaled_greeks[greek_name]
            
            # Store leg details
            leg_details.append({
                'leg_number': i + 1,
                'option_type': leg.option_type.title(),
                'strike_price': leg.strike_price,
                'quantity': leg.quantity,
                'position': 'Long' if leg.quantity > 0 else 'Short',
                'delta': scaled_greeks['delta'],
                'gamma': scaled_greeks['gamma'],
                'theta': scaled_greeks['theta'],
                'vega': scaled_greeks['vega'],
                'rho': scaled_greeks['rho'],
                'moneyness': current_price / leg.strike_price
            })
        
        # Scale by underlying size and hedge ratio
        multiplier = abs(multi_leg_strategy.underlying_size) * multi_leg_strategy.hedge_ratio
        
        portfolio_greeks = {k: v * multiplier for k, v in net_greeks.items()}
        
        return {
            'strategy_name': multi_leg_strategy.strategy_type.value,
            'current_price': current_price,
            'underlying_size': multi_leg_strategy.underlying_size,
            'hedge_ratio': multi_leg_strategy.hedge_ratio,
            'net_strategy_greeks': net_greeks,
            'portfolio_greeks': portfolio_greeks,
            'leg_details': leg_details,
            'total_legs': len(multi_leg_strategy.legs)
        }
    
    @staticmethod
    def calculate_greeks_sensitivity(multi_leg_strategy, current_price: float,
                                   price_range: float = 0.1, vol_range: float = 0.05) -> Dict[str, any]:
        """
        Calculate Greeks sensitivity to price and volatility changes.
        
        Args:
            multi_leg_strategy: MultiLegStrategy object
            current_price: Current price of underlying
            price_range: Price change range (± percentage)
            vol_range: Volatility change range (± absolute)
            
        Returns:
            Dictionary containing sensitivity analysis
        """
        # Price sensitivity
        price_shocks = np.linspace(-price_range, price_range, 11)
        price_sensitivities = []
        
        for shock in price_shocks:
            shocked_price = current_price * (1 + shock)
            greeks = MultiLegGreeksCalculator.calculate_strategy_greeks(
                multi_leg_strategy, shocked_price
            )
            price_sensitivities.append({
                'price_shock': shock,
                'price': shocked_price,
                **greeks
            })
        
        # Volatility sensitivity (simplified - would need to modify vol in strategy)
        vol_base = get_commodity_volatility(multi_leg_strategy.commodity)
        vol_shocks = np.linspace(-vol_range, vol_range, 11)
        vol_sensitivities = []
        
        base_greeks = MultiLegGreeksCalculator.calculate_strategy_greeks(
            multi_leg_strategy, current_price
        )
        
        for vol_shock in vol_shocks:
            # Approximate vega impact
            vega_impact = base_greeks['vega'] * vol_shock * 100  # Convert to percentage
            vol_sensitivities.append({
                'vol_shock': vol_shock,
                'volatility': vol_base + vol_shock,
                'vega_impact': vega_impact
            })
        
        return {
            'price_sensitivities': price_sensitivities,
            'vol_sensitivities': vol_sensitivities,
            'base_greeks': base_greeks
        }
    
    @staticmethod
    def calculate_time_decay_profile(multi_leg_strategy, current_price: float,
                                   days_ahead: int = 30) -> Dict[str, any]:
        """
        Calculate how Greeks evolve over time (theta decay profile).
        
        Args:
            multi_leg_strategy: MultiLegStrategy object
            current_price: Current price of underlying
            days_ahead: Number of days to project forward
            
        Returns:
            Dictionary containing time decay analysis
        """
        days = np.arange(0, days_ahead + 1)
        time_profile = []
        
        risk_free_rate = get_risk_free_rate()
        volatility = get_commodity_volatility(multi_leg_strategy.commodity)
        
        for day in days:
            day_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
            
            for leg in multi_leg_strategy.legs:
                # Calculate remaining time
                original_time = time_to_expiration(leg.expiry_months)
                remaining_time = max(original_time - (day / 365.0), 0.001)
                
                # Calculate Greeks for this time point
                leg_greeks = BlackScholesCalculator.calculate_greeks(
                    current_price, leg.strike_price, remaining_time,
                    risk_free_rate, volatility, leg.option_type
                )
                
                # Scale and add
                for greek_name in day_greeks:
                    day_greeks[greek_name] += leg_greeks[greek_name] * leg.quantity
            
            # Scale by position
            multiplier = abs(multi_leg_strategy.underlying_size) * multi_leg_strategy.hedge_ratio
            portfolio_greeks = {k: v * multiplier for k, v in day_greeks.items()}
            
            time_profile.append({
                'day': day,
                'days_to_expiry': int(original_time * 365 - day),
                **portfolio_greeks
            })
        
        return {
            'time_profile': time_profile,
            'total_theta_decay': sum(point['theta'] for point in time_profile),
            'daily_theta_average': sum(point['theta'] for point in time_profile) / len(time_profile)
        }


class GreeksRiskAnalyzer:
    """
    Advanced risk analysis based on Greeks for multi-leg strategies.
    """
    
    @staticmethod
    def assess_greek_risks(greeks: Dict[str, float], position_size: float) -> Dict[str, str]:
        """
        Assess risk levels based on Greeks magnitudes.
        
        Args:
            greeks: Dictionary of Greek values
            position_size: Size of underlying position
            
        Returns:
            Dictionary of risk assessments
        """
        # Define risk thresholds (scaled by position size)
        scale_factor = abs(position_size) / 1000.0  # Normalize to 1000 units
        
        delta_threshold = {'low': 0.1 * scale_factor, 'medium': 0.3 * scale_factor}
        gamma_threshold = {'low': 0.01 * scale_factor, 'medium': 0.05 * scale_factor}
        theta_threshold = {'low': -10 * scale_factor, 'medium': -50 * scale_factor}
        vega_threshold = {'low': 20 * scale_factor, 'medium': 100 * scale_factor}
        
        assessments = {}
        
        # Delta risk
        abs_delta = abs(greeks.get('delta', 0))
        if abs_delta < delta_threshold['low']:
            assessments['delta_risk'] = 'Low'
        elif abs_delta < delta_threshold['medium']:
            assessments['delta_risk'] = 'Medium'
        else:
            assessments['delta_risk'] = 'High'
        
        # Gamma risk
        abs_gamma = abs(greeks.get('gamma', 0))
        if abs_gamma < gamma_threshold['low']:
            assessments['gamma_risk'] = 'Low'
        elif abs_gamma < gamma_threshold['medium']:
            assessments['gamma_risk'] = 'Medium'
        else:
            assessments['gamma_risk'] = 'High'
        
        # Theta risk
        theta = greeks.get('theta', 0)
        if theta > theta_threshold['low']:
            assessments['theta_risk'] = 'Low'
        elif theta > theta_threshold['medium']:
            assessments['theta_risk'] = 'Medium'
        else:
            assessments['theta_risk'] = 'High'
        
        # Vega risk
        abs_vega = abs(greeks.get('vega', 0))
        if abs_vega < vega_threshold['low']:
            assessments['vega_risk'] = 'Low'
        elif abs_vega < vega_threshold['medium']:
            assessments['vega_risk'] = 'Medium'
        else:
            assessments['vega_risk'] = 'High'
        
        return assessments
    
    @staticmethod
    def generate_risk_alerts(greeks: Dict[str, float], position_size: float) -> List[Dict[str, str]]:
        """
        Generate specific risk alerts based on Greeks.
        
        Args:
            greeks: Dictionary of Greek values
            position_size: Size of underlying position
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        scale_factor = abs(position_size) / 1000.0
        
        # Delta alerts
        delta = greeks.get('delta', 0)
        if abs(delta) > 0.5 * scale_factor:
            alerts.append({
                'type': 'warning' if abs(delta) < 1.0 * scale_factor else 'critical',
                'greek': 'Delta',
                'message': f"High directional exposure: {delta:.3f}",
                'recommendation': 'Consider delta hedging or position adjustment'
            })
        
        # Gamma alerts
        gamma = greeks.get('gamma', 0)
        if abs(gamma) > 0.1 * scale_factor:
            alerts.append({
                'type': 'warning' if abs(gamma) < 0.2 * scale_factor else 'critical',
                'greek': 'Gamma',
                'message': f"High convexity risk: {gamma:.4f}",
                'recommendation': 'Monitor delta changes closely'
            })
        
        # Theta alerts
        theta = greeks.get('theta', 0)
        if theta < -100 * scale_factor:
            alerts.append({
                'type': 'warning' if theta > -200 * scale_factor else 'critical',
                'greek': 'Theta',
                'message': f"High time decay: ${theta:.2f}/day",
                'recommendation': 'Consider closing or rolling positions'
            })
        
        # Vega alerts
        vega = greeks.get('vega', 0)
        if abs(vega) > 200 * scale_factor:
            alerts.append({
                'type': 'warning' if abs(vega) < 500 * scale_factor else 'critical',
                'greek': 'Vega',
                'message': f"High volatility sensitivity: ${vega:.2f}",
                'recommendation': 'Monitor implied volatility changes'
            })
        
        return alerts
    
    @staticmethod
    def calculate_portfolio_var_greeks(greeks: Dict[str, float], price_volatility: float,
                                     vol_of_vol: float = 0.1, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate VaR using Greeks approximation.
        
        Args:
            greeks: Portfolio Greeks
            price_volatility: Underlying price volatility
            vol_of_vol: Volatility of volatility
            confidence: Confidence level
            
        Returns:
            Dictionary containing VaR estimates
        """
        # 1-day VaR calculation using Greeks
        delta = greeks.get('delta', 0)
        gamma = greeks.get('gamma', 0)
        vega = greeks.get('vega', 0)
        
        # Price change VaR (delta + gamma terms)
        price_std = price_volatility / np.sqrt(252)  # Daily volatility
        z_score = norm.ppf(confidence)
        
        # Linear (delta) VaR
        delta_var = abs(delta) * price_std * z_score
        
        # Quadratic (gamma) adjustment
        gamma_var = 0.5 * abs(gamma) * (price_std * z_score) ** 2
        
        # Volatility VaR (vega)
        vol_std = vol_of_vol / np.sqrt(252)  # Daily vol of vol
        vega_var = abs(vega) * vol_std * z_score
        
        # Combined VaR (simplified - assumes independence)
        total_var = np.sqrt(delta_var**2 + gamma_var**2 + vega_var**2)
        
        return {
            'delta_var': delta_var,
            'gamma_var': gamma_var,
            'vega_var': vega_var,
            'total_var': total_var,
            'confidence_level': confidence
        }


# ============================================================================
# ENHANCED UTILITIES AND HELPERS
# ============================================================================

class AsianOptionCalculator: 
    @staticmethod
    def monte_carlo_asian_price(S0: float, K: float, T: float, r: float, sigma: float, n_steps: int, n_simulations: int,
                              option_type: str, average_type: str = "arithmetic") -> Dict[str, float]:
        if T <= 0 or n_steps <= 0 or n_simulations <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'confidence_interval': (0.0, 0.0)}
        
        dt = T / n_steps
        discount_factor = np.exp(-r * T)
        
        np.random.seed(42)
        random_numbers = np.random.standard_normal((n_simulations, n_steps))
        
        payoffs = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            prices = np.zeros(n_steps + 1)
            prices[0] = S0
            
            for j in range(n_steps):
                prices[j + 1] = prices[j] * np.exp(
                    (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_numbers[i, j]
                )
            
            if average_type == "arithmetic":
                avg_price = np.mean(prices[1:])
            else:
                avg_price = np.exp(np.mean(np.log(prices[1:])))
            
            if option_type.lower() == 'call':
                payoffs[i] = max(avg_price - K, 0)
            else:
                payoffs[i] = max(K - avg_price, 0)
        
        discounted_payoffs = payoffs * discount_factor
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
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


class OptionsPortfolioAnalyzer:    
    @staticmethod
    def calculate_portfolio_greeks(positions: List[Dict]) -> Dict[str, float]:
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
        portfolio_greeks = OptionsPortfolioAnalyzer.calculate_portfolio_greeks(positions)
        
        avg_volatility = np.mean([pos.get('sigma', 0.25) for pos in positions])
        avg_spot = np.mean([pos.get('S', 100) for pos in positions])
        
        delta_var = abs(portfolio_greeks['delta']) * avg_spot * avg_volatility * np.sqrt(time_horizon)
        gamma_adjustment = 0.5 * portfolio_greeks['gamma'] * (avg_spot * avg_volatility * np.sqrt(time_horizon))**2
        total_var = delta_var + gamma_adjustment
        
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
    @staticmethod
    def historical_volatility(prices: pd.Series, window: int = 30) -> float:
        if len(prices) < window + 1:
            return 0.25
        
        try:
            log_returns = np.log(prices / prices.shift(1)).dropna()
            rolling_vol = log_returns.rolling(window=window).std()
            annualized_vol = rolling_vol.iloc[-1] * np.sqrt(252)
            
            return annualized_vol if not np.isnan(annualized_vol) else 0.25
            
        except Exception as e:
            print(f"Error calculating historical volatility: {e}")
            return 0.25
    
    @staticmethod
    def garch_volatility(prices: pd.Series, forecast_periods: int = 1) -> float:
        try:
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            if len(log_returns) < 50:
                return VolatilityEstimator.historical_volatility(prices)
            
            alpha = 0.1
            beta = 0.85
            omega = 0.000001
            
            returns_squared = log_returns**2
            conditional_variance = np.zeros(len(returns_squared))
            conditional_variance[0] = returns_squared.var()
            
            for i in range(1, len(returns_squared)):
                conditional_variance[i] = (omega + 
                                         alpha * returns_squared.iloc[i-1] + 
                                         beta * conditional_variance[i-1])
            
            last_variance = conditional_variance[-1]
            last_return_squared = returns_squared.iloc[-1]
            
            forecasted_variance = omega + alpha * last_return_squared + beta * last_variance
            forecasted_volatility = np.sqrt(forecasted_variance * 252)
            
            return forecasted_volatility if not np.isnan(forecasted_volatility) else 0.25
            
        except Exception as e:
            print(f"Error calculating GARCH volatility: {e}")
            return VolatilityEstimator.historical_volatility(prices)


def get_risk_free_rate() -> float:
    """Get current risk-free rate"""
    return 0.03

def get_commodity_volatility(commodity: str, prices: pd.Series = None) -> float:
    """Get volatility estimate for commodity"""
    if prices is not None and len(prices) > 30:
        return VolatilityEstimator.historical_volatility(prices)
    
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
    return expiry_months / 12.0


if __name__ == "__main__":
    print("=== Multi-Leg Greeks Calculator Test ===")
    
    # Test with mock multi-leg strategy
    try:
        from hedging.multi_leg_strategies import create_long_straddle
        
        current_price = 75.0
        straddle = create_long_straddle(75.0, 1000, 0.8, "WTI Crude Oil")
        
        # Test basic Greeks calculation
        greeks = MultiLegGreeksCalculator.calculate_strategy_greeks(straddle, current_price)
        print(f"Straddle Greeks: {greeks}")
        
        # Test detailed analysis
        detailed = MultiLegGreeksCalculator.calculate_detailed_greeks(straddle, current_price)
        print(f"Strategy: {detailed['strategy_name']}")
        print(f"Total legs: {detailed['total_legs']}")
        print(f"Net portfolio delta: {detailed['portfolio_greeks']['delta']:.3f}")
        
        # Test risk assessment
        risk_assessment = GreeksRiskAnalyzer.assess_greek_risks(greeks, 1000)
        print(f"Risk Assessment: {risk_assessment}")
        
        # Test alerts
        alerts = GreeksRiskAnalyzer.generate_risk_alerts(greeks, 1000)
        if alerts:
            print(f"Risk Alerts: {len(alerts)} alerts generated")
            for alert in alerts:
                print(f"  {alert['type'].upper()}: {alert['message']}")
        else:
            print("No risk alerts generated")
        
        # Test VaR calculation
        var_analysis = GreeksRiskAnalyzer.calculate_portfolio_var_greeks(
            greeks, 0.35, 0.1, 0.95
        )
        print(f"Portfolio VaR (95%): ${var_analysis['total_var']:.2f}")
        
        print("Multi-leg Greeks calculations working")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Testing basic Black-Scholes functionality...")
        
        # Test basic functionality
        S, K, T, r, sigma = 75.0, 75.0, 0.25, 0.03, 0.35
        
        call_price = BlackScholesCalculator.calculate_option_price(S, K, T, r, sigma, 'call')
        call_greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, 'call')
        
        print(f"Call Price: ${call_price:.2f}")
        print(f"Call Delta: {call_greeks['delta']:.3f}")
        print(f"Call Gamma: {call_greeks['gamma']:.4f}")
        print(f"Call Theta: ${call_greeks['theta']:.2f}")
        print(f"Call Vega: ${call_greeks['vega']:.2f}")
        
        print("Basic options math working")
    
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Basic framework loaded")