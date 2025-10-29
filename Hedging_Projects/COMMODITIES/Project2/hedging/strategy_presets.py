"""
hedging/strategy_presets.py

Advanced strategy presets and intelligent defaults for multi-leg options strategies.
Provides educational information, risk profiles, and optimal configurations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class StrategyProfile:
    """Comprehensive strategy profile with risk characteristics"""
    name: str
    description: str
    best_for: str
    market_outlook: str
    max_loss: str
    max_gain: str
    breakeven_count: int
    complexity_level: str
    volatility_view: str
    time_decay_impact: str
    typical_dte: str
    profit_probability: float
    risk_reward_ratio: float
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'Description': self.description,
            'Best For': self.best_for,
            'Market Outlook': self.market_outlook,
            'Max Loss': self.max_loss,
            'Max Gain': self.max_gain,
            'Breakevens': f"{self.breakeven_count} points",
            'Complexity': self.complexity_level,
            'Volatility View': self.volatility_view,
            'Time Decay': self.time_decay_impact,
            'Typical DTE': self.typical_dte,
            'Profit Probability': f"{self.profit_probability:.0%}",
            'Risk/Reward': f"{self.risk_reward_ratio:.2f}"
        }


class StrategyPresets:
    """Advanced strategy presets with intelligent defaults"""
    
    @staticmethod
    def get_strategy_profiles() -> Dict[str, StrategyProfile]:
        """Get comprehensive strategy profiles"""
        
        profiles = {
            "Long Straddle": StrategyProfile(
                name="Long Straddle",
                description="Buy call and put at same strike - profit from high volatility",
                best_for="Expecting major price movement but uncertain of direction",
                market_outlook="Neutral direction, bullish volatility",
                max_loss="Net premium paid",
                max_gain="Unlimited",
                breakeven_count=2,
                complexity_level="Intermediate",
                volatility_view="Bullish volatility",
                time_decay_impact="Negative (theta decay hurts)",
                typical_dte="30-60 days",
                profit_probability=0.35,
                risk_reward_ratio=2.5
            ),
            
            "Short Straddle": StrategyProfile(
                name="Short Straddle",
                description="Sell call and put at same strike - profit from low volatility",
                best_for="Expecting minimal price movement within range",
                market_outlook="Neutral with low volatility expectation",
                max_loss="Unlimited",
                max_gain="Net premium received",
                breakeven_count=2,
                complexity_level="Advanced",
                volatility_view="Bearish volatility",
                time_decay_impact="Positive (theta decay helps)",
                typical_dte="30-45 days",
                profit_probability=0.55,
                risk_reward_ratio=0.4
            ),
            
            "Long Strangle": StrategyProfile(
                name="Long Strangle",
                description="Buy OTM call and put - lower cost than straddle",
                best_for="Expecting significant movement, lower cost than straddle",
                market_outlook="Neutral direction, bullish volatility",
                max_loss="Net premium paid",
                max_gain="Unlimited",
                breakeven_count=2,
                complexity_level="Intermediate",
                volatility_view="Bullish volatility",
                time_decay_impact="Negative (theta decay hurts)",
                typical_dte="30-60 days",
                profit_probability=0.30,
                risk_reward_ratio=3.0
            ),
            
            "Short Strangle": StrategyProfile(
                name="Short Strangle",
                description="Sell OTM call and put - profit from range-bound market",
                best_for="Expecting price to stay within wide range",
                market_outlook="Neutral with controlled volatility",
                max_loss="Unlimited",
                max_gain="Net premium received",
                breakeven_count=2,
                complexity_level="Advanced",
                volatility_view="Bearish volatility",
                time_decay_impact="Positive (theta decay helps)",
                typical_dte="30-45 days",
                profit_probability=0.60,
                risk_reward_ratio=0.5
            ),
            
            "Collar": StrategyProfile(
                name="Collar",
                description="Protective put + covered call - limited risk and reward",
                best_for="Hedging long positions with limited upside participation",
                market_outlook="Slightly bullish with protection",
                max_loss="Limited to put strike",
                max_gain="Limited to call strike",
                breakeven_count=1,
                complexity_level="Intermediate",
                volatility_view="Neutral",
                time_decay_impact="Neutral (offsetting effects)",
                typical_dte="60-90 days",
                profit_probability=0.50,
                risk_reward_ratio=1.0
            ),
            
            "Butterfly Spread": StrategyProfile(
                name="Butterfly Spread",
                description="Long wings + short body - profit from minimal movement",
                best_for="Expecting price to stay near middle strike",
                market_outlook="Neutral with precision timing",
                max_loss="Net premium paid",
                max_gain="Strike spacing minus premium",
                breakeven_count=2,
                complexity_level="Advanced",
                volatility_view="Bearish volatility",
                time_decay_impact="Mixed (depends on position)",
                typical_dte="30-60 days",
                profit_probability=0.25,
                risk_reward_ratio=3.0
            ),
            
            "Iron Condor": StrategyProfile(
                name="Iron Condor",
                description="Short strangle + long protection - profit from range-bound market",
                best_for="Expecting price to stay within wide range with limited risk",
                market_outlook="Neutral with defined risk",
                max_loss="Strike spacing minus premium",
                max_gain="Net premium received",
                breakeven_count=2,
                complexity_level="Advanced",
                volatility_view="Bearish volatility",
                time_decay_impact="Positive (theta decay helps)",
                typical_dte="30-60 days",
                profit_probability=0.65,
                risk_reward_ratio=0.6
            )
        }
        
        return profiles
    
    @staticmethod
    def get_strategy_defaults(strategy_type: str, current_price: float, 
                            commodity: str = "WTI Crude Oil") -> Dict:
        """Get intelligent defaults for strategy configuration"""
        
        # Market-specific adjustments
        volatility_factor = StrategyPresets._get_commodity_volatility_factor(commodity)
        
        defaults = {
            "Long Straddle": {
                "strikes": {"strike": current_price},
                "expiry_months": 3,
                "hedge_ratio": 0.8,
                "strike_spacing": 0.0,
                "estimated_cost": current_price * 0.04 * volatility_factor,
                "optimal_vol_range": (0.25, 0.45),
                "min_move_required": current_price * 0.06,
                "risk_level": "Medium-High"
            },
            
            "Short Straddle": {
                "strikes": {"strike": current_price},
                "expiry_months": 2,
                "hedge_ratio": 0.6,
                "strike_spacing": 0.0,
                "estimated_credit": current_price * 0.03 * volatility_factor,
                "optimal_vol_range": (0.15, 0.25),
                "max_safe_move": current_price * 0.04,
                "risk_level": "High"
            },
            
            "Long Strangle": {
                "strikes": {
                    "call_strike": current_price * 1.05,
                    "put_strike": current_price * 0.95
                },
                "expiry_months": 3,
                "hedge_ratio": 0.8,
                "strike_spacing": current_price * 0.10,
                "estimated_cost": current_price * 0.025 * volatility_factor,
                "optimal_vol_range": (0.30, 0.50),
                "min_move_required": current_price * 0.08,
                "risk_level": "Medium"
            },
            
            "Short Strangle": {
                "strikes": {
                    "call_strike": current_price * 1.10,
                    "put_strike": current_price * 0.90
                },
                "expiry_months": 2,
                "hedge_ratio": 0.5,
                "strike_spacing": current_price * 0.20,
                "estimated_credit": current_price * 0.02 * volatility_factor,
                "optimal_vol_range": (0.15, 0.25),
                "max_safe_move": current_price * 0.08,
                "risk_level": "High"
            },
            
            "Collar": {
                "strikes": {
                    "call_strike": current_price * 1.10,
                    "put_strike": current_price * 0.90
                },
                "expiry_months": 6,
                "hedge_ratio": 0.9,
                "strike_spacing": current_price * 0.20,
                "estimated_cost": current_price * 0.005,  # Often near zero cost
                "protection_level": 0.10,  # 10% downside protection
                "upside_cap": 0.10,  # 10% upside cap
                "risk_level": "Low"
            },
            
            "Butterfly Spread": {
                "strikes": {
                    "lower_strike": current_price * 0.95,
                    "middle_strike": current_price,
                    "upper_strike": current_price * 1.05
                },
                "expiry_months": 2,
                "hedge_ratio": 0.6,
                "strike_spacing": current_price * 0.05,
                "estimated_cost": current_price * 0.01,
                "optimal_vol_range": (0.15, 0.25),
                "target_zone": (current_price * 0.98, current_price * 1.02),
                "risk_level": "Medium"
            },
            
            "Iron Condor": {
                "strikes": {
                    "put_strike_low": current_price * 0.85,
                    "put_strike_high": current_price * 0.95,
                    "call_strike_low": current_price * 1.05,
                    "call_strike_high": current_price * 1.15
                },
                "expiry_months": 2,
                "hedge_ratio": 0.7,
                "strike_spacing": current_price * 0.10,
                "estimated_credit": current_price * 0.015,
                "optimal_vol_range": (0.15, 0.25),
                "profit_zone": (current_price * 0.95, current_price * 1.05),
                "risk_level": "Medium-High"
            }
        }
        
        return defaults.get(strategy_type, {})
    
    @staticmethod
    def _get_commodity_volatility_factor(commodity: str) -> float:
        """Adjust defaults based on commodity volatility characteristics"""
        
        volatility_factors = {
            "WTI Crude Oil": 1.0,
            "Brent Crude Oil": 0.95,
            "Natural Gas": 1.3,
            "Gold": 0.7,
            "Silver": 1.1,
            "Copper": 0.9
        }
        
        return volatility_factors.get(commodity, 1.0)
    
    @staticmethod
    def get_strategy_recommendations(market_conditions: Dict[str, float]) -> List[Dict[str, str]]:
        """Provide strategy recommendations based on market conditions"""
        
        current_vol = market_conditions.get('volatility', 0.25)
        trend_strength = market_conditions.get('trend_strength', 0.0)
        time_to_event = market_conditions.get('days_to_event', 30)
        
        recommendations = []
        
        # High volatility environment
        if current_vol > 0.35:
            if trend_strength < 0.3:  # Low trend, high vol
                recommendations.append({
                    'strategy': 'Short Straddle',
                    'confidence': 'High',
                    'reason': 'High volatility likely to revert to mean',
                    'risk_warning': 'Unlimited risk - use stops'
                })
            else:  # High trend, high vol
                recommendations.append({
                    'strategy': 'Long Strangle',
                    'confidence': 'Medium',
                    'reason': 'Directional movement with high volatility',
                    'risk_warning': 'Needs large move to profit'
                })
        
        # Low volatility environment
        elif current_vol < 0.20:
            recommendations.append({
                'strategy': 'Long Straddle',
                'confidence': 'High',
                'reason': 'Low volatility likely to increase',
                'risk_warning': 'Time decay is enemy'
            })
            
            recommendations.append({
                'strategy': 'Iron Condor',
                'confidence': 'Medium',
                'reason': 'Profit from continued low volatility',
                'risk_warning': 'Needs range-bound market'
            })
        
        # Moderate volatility
        else:
            if time_to_event < 15:  # Event approaching
                recommendations.append({
                    'strategy': 'Long Strangle',
                    'confidence': 'Medium',
                    'reason': 'Potential volatility expansion near event',
                    'risk_warning': 'High time decay risk'
                })
            else:  # Normal conditions
                recommendations.append({
                    'strategy': 'Collar',
                    'confidence': 'High',
                    'reason': 'Balanced risk/reward for hedging',
                    'risk_warning': 'Limited upside participation'
                })
        
        return recommendations
    
    @staticmethod
    def get_strike_recommendations(strategy_type: str, current_price: float,
                                 volatility: float, days_to_expiry: int) -> Dict[str, float]:
        """Get recommended strike prices based on strategy and market conditions"""
        
        # Volatility-adjusted strike spacing
        vol_adjustment = volatility / 0.25  # Normalize to 25% vol
        time_adjustment = np.sqrt(days_to_expiry / 30)  # Normalize to 30 days
        
        base_spacing = current_price * 0.05 * vol_adjustment * time_adjustment
        
        recommendations = {}
        
        if strategy_type == "Long Straddle":
            # ATM for maximum gamma
            recommendations['strike'] = current_price
            
        elif strategy_type == "Short Straddle":
            # Slightly ITM for better probability
            recommendations['strike'] = current_price * 1.01
            
        elif strategy_type == "Long Strangle":
            # Wider strikes for lower cost
            recommendations['put_strike'] = current_price - (base_spacing * 1.5)
            recommendations['call_strike'] = current_price + (base_spacing * 1.5)
            
        elif strategy_type == "Short Strangle":
            # Further OTM for better probability
            recommendations['put_strike'] = current_price - (base_spacing * 2.0)
            recommendations['call_strike'] = current_price + (base_spacing * 2.0)
            
        elif strategy_type == "Collar":
            # Balanced protection vs. cost
            recommendations['put_strike'] = current_price - (base_spacing * 2.0)
            recommendations['call_strike'] = current_price + (base_spacing * 2.2)
            
        elif strategy_type == "Butterfly Spread":
            # Tight spacing around current price
            recommendations['lower_strike'] = current_price - base_spacing
            recommendations['middle_strike'] = current_price
            recommendations['upper_strike'] = current_price + base_spacing
            
        elif strategy_type == "Iron Condor":
            # Wide profit zone
            recommendations['put_strike_low'] = current_price - (base_spacing * 3.0)
            recommendations['put_strike_high'] = current_price - (base_spacing * 1.5)
            recommendations['call_strike_low'] = current_price + (base_spacing * 1.5)
            recommendations['call_strike_high'] = current_price + (base_spacing * 3.0)
        
        return recommendations
    
    @staticmethod
    def validate_strategy_configuration(strategy_type: str, strikes: Dict[str, float],
                                      current_price: float) -> List[str]:
        """Validate strategy configuration and return warnings"""
        
        warnings = []
        
        if strategy_type in ["Long Straddle", "Short Straddle"]:
            strike = strikes.get('strike', current_price)
            moneyness = current_price / strike
            
            if abs(moneyness - 1.0) > 0.10:
                warnings.append(f"⚠️ Strike is {abs(moneyness - 1.0):.1%} away from ATM")
                
        elif strategy_type in ["Long Strangle", "Short Strangle"]:
            call_strike = strikes.get('call_strike', current_price * 1.05)
            put_strike = strikes.get('put_strike', current_price * 0.95)
            
            if call_strike <= put_strike:
                warnings.append("❌ Call strike must be higher than put strike")
            
            width = call_strike - put_strike
            if width < current_price * 0.05:
                warnings.append("⚠️ Strike width is very narrow - consider wider strikes")
            elif width > current_price * 0.30:
                warnings.append("⚠️ Strike width is very wide - may need large moves")
                
        elif strategy_type == "Collar":
            call_strike = strikes.get('call_strike', current_price * 1.10)
            put_strike = strikes.get('put_strike', current_price * 0.90)
            
            if call_strike <= current_price:
                warnings.append("❌ Call strike should be above current price")
            if put_strike >= current_price:
                warnings.append("❌ Put strike should be below current price")
                
            protection = (current_price - put_strike) / current_price
            if protection < 0.05:
                warnings.append("⚠️ Very little downside protection")
            elif protection > 0.20:
                warnings.append("⚠️ Expensive protection - high cost")
                
        elif strategy_type == "Butterfly Spread":
            lower = strikes.get('lower_strike', current_price * 0.95)
            middle = strikes.get('middle_strike', current_price)
            upper = strikes.get('upper_strike', current_price * 1.05)
            
            if not (lower < middle < upper):
                warnings.append("❌ Strikes must be ordered: lower < middle < upper")
            
            lower_width = middle - lower
            upper_width = upper - middle
            
            if abs(lower_width - upper_width) / middle > 0.02:
                warnings.append("⚠️ Uneven strike spacing - consider equal spacing")
                
        elif strategy_type == "Iron Condor":
            put_low = strikes.get('put_strike_low', current_price * 0.85)
            put_high = strikes.get('put_strike_high', current_price * 0.95)
            call_low = strikes.get('call_strike_low', current_price * 1.05)
            call_high = strikes.get('call_strike_high', current_price * 1.15)
            
            if not (put_low < put_high < call_low < call_high):
                warnings.append("❌ Invalid strike ordering")
            
            put_width = put_high - put_low
            call_width = call_high - call_low
            
            if abs(put_width - call_width) / current_price > 0.02:
                warnings.append("⚠️ Uneven wing widths - consider equal widths")
        
        return warnings
    
    @staticmethod
    def get_educational_content(strategy_type: str) -> Dict[str, str]:
        """Get educational content for strategy learning"""
        
        content = {
            "Long Straddle": {
                "when_to_use": "Before earnings announcements, FDA approvals, or other binary events",
                "profit_mechanism": "Profits when actual move exceeds implied volatility expectations",
                "risk_factors": "Time decay accelerates as expiration approaches",
                "management_tips": "Consider closing at 50% profit or 25% loss",
                "example_scenario": "Oil at $75, buy $75 calls and puts before OPEC meeting"
            },
            
            "Short Straddle": {
                "when_to_use": "After volatility spikes, when expecting consolidation",
                "profit_mechanism": "Profits from volatility contraction and time decay",
                "risk_factors": "Unlimited risk if market moves significantly",
                "management_tips": "Use stop-loss orders and position sizing",
                "example_scenario": "After oil spikes to $90, sell $90 calls and puts expecting range"
            },
            
            "Collar": {
                "when_to_use": "Protecting long positions while generating income",
                "profit_mechanism": "Put provides downside protection, call generates income",
                "risk_factors": "Limited upside participation above call strike",
                "management_tips": "Roll strikes higher in bull markets",
                "example_scenario": "Own oil at $70, buy $65 puts, sell $80 calls"
            },
            
            "Iron Condor": {
                "when_to_use": "High implied volatility with expectation of range-bound trading",
                "profit_mechanism": "Profits from volatility contraction and time decay",
                "risk_factors": "Loses money if price moves outside the wings",
                "management_tips": "Close early if reaching 25% of max profit",
                "example_scenario": "Oil at $75, expect trading between $70-$80"
            }
        }
        
        return content.get(strategy_type, {})


class StrategyOptimizer:
    """Optimize strategy parameters based on market conditions"""
    
    @staticmethod
    def optimize_hedge_ratio(strategy_type: str, portfolio_size: float,
                           risk_tolerance: str, market_volatility: float) -> float:
        """Optimize hedge ratio based on portfolio and market conditions"""
        
        base_ratios = {
            "Long Straddle": 0.8,
            "Short Straddle": 0.4,
            "Long Strangle": 0.7,
            "Short Strangle": 0.5,
            "Collar": 0.9,
            "Butterfly Spread": 0.6,
            "Iron Condor": 0.7
        }
        
        base_ratio = base_ratios.get(strategy_type, 0.6)
        
        # Adjust for risk tolerance
        risk_multipliers = {
            "Conservative": 0.8,
            "Moderate": 1.0,
            "Aggressive": 1.2
        }
        
        risk_multiplier = risk_multipliers.get(risk_tolerance, 1.0)
        
        # Adjust for market volatility
        if market_volatility > 0.35:
            vol_adjustment = 0.9  # Reduce exposure in high vol
        elif market_volatility < 0.20:
            vol_adjustment = 1.1  # Increase exposure in low vol
        else:
            vol_adjustment = 1.0
        
        # Adjust for portfolio size (larger portfolios can take more risk)
        if portfolio_size > 10000000:  # $10M+
            size_adjustment = 1.1
        elif portfolio_size < 1000000:  # <$1M
            size_adjustment = 0.9
        else:
            size_adjustment = 1.0
        
        optimized_ratio = base_ratio * risk_multiplier * vol_adjustment * size_adjustment
        
        return max(0.1, min(1.0, optimized_ratio))
    
    @staticmethod
    def optimize_expiry_selection(strategy_type: str, volatility: float,
                                time_to_event: int) -> int:
        """Optimize expiration selection based on strategy and market conditions"""
        
        base_expiries = {
            "Long Straddle": 45,
            "Short Straddle": 30,
            "Long Strangle": 60,
            "Short Strangle": 30,
            "Collar": 90,
            "Butterfly Spread": 30,
            "Iron Condor": 45
        }
        
        base_expiry = base_expiries.get(strategy_type, 45)
        
        # Adjust for volatility
        if volatility > 0.35:
            vol_adjustment = 0.8  # Shorter expiry in high vol
        elif volatility < 0.20:
            vol_adjustment = 1.2  # Longer expiry in low vol
        else:
            vol_adjustment = 1.0
        
        # Adjust for upcoming events
        if time_to_event > 0 and time_to_event < base_expiry:
            # Ensure expiry is after the event
            return max(time_to_event + 7, int(base_expiry * 0.6))
        
        optimized_expiry = int(base_expiry * vol_adjustment)
        
        return max(7, min(365, optimized_expiry))


if __name__ == "__main__":
    presets = StrategyPresets()
    
    print("=== Strategy Profiles ===")
    profiles = presets.get_strategy_profiles()
    for name, profile in profiles.items():
        print(f"\n{name}:")
        print(f"  Description: {profile.description}")
        print(f"  Profit Probability: {profile.profit_probability:.1%}")
        print(f"  Risk Level: {profile.complexity_level}")
    
    print("\n=== Strategy Defaults ===")
    current_price = 75.0
    defaults = presets.get_strategy_defaults("Long Straddle", current_price)
    print(f"Long Straddle defaults: {defaults}")
    
    print("\n=== Strategy Recommendations ===")
    market_conditions = {
        'volatility': 0.15,  # Low volatility
        'trend_strength': 0.2,
        'days_to_event': 45
    }
    recommendations = presets.get_strategy_recommendations(market_conditions)
    for rec in recommendations:
        print(f"  {rec['strategy']}: {rec['reason']} ({rec['confidence']} confidence)")
    
    print("\n=== Strike Recommendations ===")
    strike_recs = presets.get_strike_recommendations("Long Strangle", current_price, 0.25, 30)
    print(f"Long Strangle strikes: {strike_recs}")
    
    print("\n=== Configuration Validation ===")
    test_strikes = {'call_strike': 80.0, 'put_strike': 70.0}
    warnings = presets.validate_strategy_configuration("Long Strangle", test_strikes, current_price)
    print(f"Validation warnings: {warnings}")
    
    print("\nStrategy presets module working correctly!")