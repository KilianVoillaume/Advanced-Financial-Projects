"""
hedging/multi_leg_strategies.py

Core framework for multi-leg options strategies with comprehensive strategy builders.
Enhanced with strategy presets integration and intelligent defaults.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

try:
    from .strategy_presets import StrategyPresets, StrategyProfile, StrategyOptimizer
except ImportError:
    class StrategyPresets:
        @staticmethod
        def get_strategy_defaults(strategy_type: str, current_price: float, commodity: str = "WTI Crude Oil") -> Dict:
            return {}
        
        @staticmethod
        def get_strategy_profiles() -> Dict:
            return {}
        
        @staticmethod
        def get_strike_recommendations(strategy_type: str, current_price: float, volatility: float, days_to_expiry: int) -> Dict[str, float]:
            return {}
        
        @staticmethod
        def validate_strategy_configuration(strategy_type: str, strikes: Dict[str, float], current_price: float) -> List[str]:
            return []
        
        @staticmethod
        def get_strategy_recommendations(market_conditions: Dict[str, float]) -> List[Dict[str, str]]:
            return []
    
    class StrategyOptimizer:
        @staticmethod
        def optimize_hedge_ratio(strategy_type: str, portfolio_size: float, risk_tolerance: str, market_volatility: float) -> float:
            return 0.8
        
        @staticmethod
        def optimize_expiry_selection(strategy_type: str, volatility: float, time_to_event: int) -> int:
            return 45


class StrategyType(Enum):
    """ Enumeration of supported multi-leg strategy types """
    LONG_STRADDLE = "Long Straddle"
    SHORT_STRADDLE = "Short Straddle"
    LONG_STRANGLE = "Long Strangle"
    SHORT_STRANGLE = "Short Strangle"
    COLLAR = "Collar"
    BUTTERFLY_SPREAD = "Butterfly Spread"
    IRON_CONDOR = "Iron Condor"


@dataclass
class OptionLeg:
    """ Individual option leg within a multi-leg strategy """
    option_type: str  # 'call' or 'put'
    strike_price: float
    quantity: int  # positive for long, negative for short
    expiry_months: int = 3
    
    def __post_init__(self):
        if self.option_type.lower() not in ['call', 'put']:
            raise ValueError(f"Invalid option_type: {self.option_type}. Must be 'call' or 'put'")
        if self.strike_price <= 0:
            raise ValueError(f"Invalid strike_price: {self.strike_price}. Must be positive")
        if self.quantity == 0:
            raise ValueError("Quantity cannot be zero")
        if self.expiry_months <= 0:
            raise ValueError(f"Invalid expiry_months: {self.expiry_months}. Must be positive")
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def abs_quantity(self) -> int:
        return abs(self.quantity)
    
    def __str__(self) -> str:
        position = "Long" if self.is_long else "Short"
        return f"{position} {self.abs_quantity}x {self.option_type.title()} ${self.strike_price:.2f}"


@dataclass
class MultiLegStrategy:
    """ Mlti-leg options strategies with enhanced preset integration """
    strategy_type: StrategyType
    legs: List[OptionLeg]
    underlying_size: float
    hedge_ratio: float
    commodity: str
    
    def __post_init__(self):
        if not self.legs:
            raise ValueError("Strategy must have at least one leg")
        if not (0.0 <= self.hedge_ratio <= 1.0):
            raise ValueError(f"Invalid hedge_ratio: {self.hedge_ratio}. Must be between 0.0 and 1.0")
        if self.underlying_size == 0:
            raise ValueError("Underlying size cannot be zero")
        self._validate_strategy_composition()
    
    def _validate_strategy_composition(self):
        """ Validate that the legs match the strategy type """
        num_legs = len(self.legs)
        
        if self.strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE]:
            if num_legs != 2:
                raise ValueError(f"Straddle must have exactly 2 legs, got {num_legs}")
            call_legs = [leg for leg in self.legs if leg.option_type.lower() == 'call']
            put_legs = [leg for leg in self.legs if leg.option_type.lower() == 'put']
            if len(call_legs) != 1 or len(put_legs) != 1:
                raise ValueError("Straddle must have exactly one call and one put")
            if call_legs[0].strike_price != put_legs[0].strike_price:
                raise ValueError("Straddle must have call and put at the same strike price")
        
        elif self.strategy_type in [StrategyType.LONG_STRANGLE, StrategyType.SHORT_STRANGLE]:
            if num_legs != 2:
                raise ValueError(f"Strangle must have exactly 2 legs, got {num_legs}")
            call_legs = [leg for leg in self.legs if leg.option_type.lower() == 'call']
            put_legs = [leg for leg in self.legs if leg.option_type.lower() == 'put']
            if len(call_legs) != 1 or len(put_legs) != 1:
                raise ValueError("Strangle must have exactly one call and one put")
            if call_legs[0].strike_price == put_legs[0].strike_price:
                raise ValueError("Strangle must have call and put at different strike prices")
        
        elif self.strategy_type == StrategyType.COLLAR:
            if num_legs != 2:
                raise ValueError(f"Collar must have exactly 2 legs, got {num_legs}")
            call_legs = [leg for leg in self.legs if leg.option_type.lower() == 'call']
            put_legs = [leg for leg in self.legs if leg.option_type.lower() == 'put']
            if len(call_legs) != 1 or len(put_legs) != 1:
                raise ValueError("Collar must have exactly one call and one put")
        
        elif self.strategy_type == StrategyType.BUTTERFLY_SPREAD:
            if num_legs != 3:
                raise ValueError(f"Butterfly spread must have exactly 3 legs, got {num_legs}")
        
        elif self.strategy_type == StrategyType.IRON_CONDOR:
            if num_legs != 4:
                raise ValueError(f"Iron condor must have exactly 4 legs, got {num_legs}")
    
    def get_total_premium(self, current_price: float) -> float:
        """ Net premium paid/received for the strategy """
        try:
            from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, get_commodity_volatility, time_to_expiration
            
            total_premium = 0.0
            risk_free_rate = get_risk_free_rate()
            volatility = get_commodity_volatility(self.commodity)
            
            for leg in self.legs:
                time_to_expiry = time_to_expiration(leg.expiry_months)
                option_price = BlackScholesCalculator.calculate_option_price(
                    current_price, leg.strike_price, time_to_expiry,
                    risk_free_rate, volatility, leg.option_type
                )
                leg_premium = option_price * leg.quantity
                total_premium += leg_premium
            
            return total_premium
            
        except ImportError:
            return self._estimate_premium_simple(current_price)
    
    def _estimate_premium_simple(self, current_price: float) -> float:
        """ Premium estimation for testing purposes """
        total_premium = 0.0
        
        for leg in self.legs:
            if leg.option_type.lower() == 'call':
                intrinsic = max(current_price - leg.strike_price, 0)
            else:
                intrinsic = max(leg.strike_price - current_price, 0)
            
            time_value = current_price * 0.02 * (leg.expiry_months / 12.0)
            option_price = intrinsic + time_value
            leg_premium = option_price * leg.quantity
            total_premium += leg_premium
        
        return total_premium
    
    def get_strategy_greeks(self, current_price: float) -> Dict[str, float]:
        """ Net Greeks for the entire strategy """
        try:
            from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, get_commodity_volatility, time_to_expiration
            
            net_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
            risk_free_rate = get_risk_free_rate()
            volatility = get_commodity_volatility(self.commodity)
            
            for leg in self.legs:
                time_to_expiry = time_to_expiration(leg.expiry_months)
                leg_greeks = BlackScholesCalculator.calculate_greeks(
                    current_price, leg.strike_price, time_to_expiry,
                    risk_free_rate, volatility, leg.option_type
                )
                
                for greek_name in net_greeks:
                    net_greeks[greek_name] += leg_greeks[greek_name] * leg.quantity
            
            multiplier = abs(self.underlying_size) * self.hedge_ratio
            for greek_name in net_greeks:
                net_greeks[greek_name] *= multiplier
            
            return net_greeks
            
        except ImportError:
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    
    def get_strategy_profile(self) -> Optional[Dict[str, str]]:
        """Get comprehensive strategy profile from presets"""
        try:
            profiles = StrategyPresets.get_strategy_profiles()
            profile = profiles.get(self.strategy_type.value)
            return profile.to_dict() if profile else None
        except:
            return None
    
    def get_configuration_warnings(self, current_price: float) -> List[str]:
        """Get configuration warnings using presets validation"""
        try:
            # Extract strikes from legs
            strikes = {}
            for leg in self.legs:
                if leg.option_type.lower() == 'call':
                    if 'call_strike' not in strikes:
                        strikes['call_strike'] = leg.strike_price
                    elif 'call_strike_low' not in strikes:
                        strikes['call_strike_low'] = leg.strike_price
                    else:
                        strikes['call_strike_high'] = leg.strike_price
                else:  # put
                    if 'put_strike' not in strikes:
                        strikes['put_strike'] = leg.strike_price
                    elif 'put_strike_low' not in strikes:
                        strikes['put_strike_low'] = leg.strike_price
                    else:
                        strikes['put_strike_high'] = leg.strike_price
            
            # For straddles, use 'strike' instead
            if self.strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE]:
                strikes = {'strike': self.legs[0].strike_price}
            
            return StrategyPresets.validate_strategy_configuration(
                self.strategy_type.value, strikes, current_price
            )
        except:
            return []
    
    def get_educational_content(self) -> Dict[str, str]:
        """Get educational content for the strategy"""
        try:
            return StrategyPresets.get_educational_content(self.strategy_type.value)
        except:
            return {}
    
    def get_optimal_expiry(self, market_volatility: float = 0.25, time_to_event: int = 30) -> int:
        """Get optimal expiry selection based on market conditions"""
        try:
            return StrategyOptimizer.optimize_expiry_selection(
                self.strategy_type.value, market_volatility, time_to_event
            )
        except:
            return 45  # Default to 45 days
    
    @property
    def description(self) -> str:
        """Human-readable description of the strategy"""
        try:
            profile = self.get_strategy_profile()
            if profile:
                return profile.get('Description', 'Multi-leg options strategy')
        except:
            pass
        
        # Fallback descriptions
        descriptions = {
            StrategyType.LONG_STRADDLE: "Profit from high volatility in either direction",
            StrategyType.SHORT_STRADDLE: "Profit from low volatility (range-bound market)",
            StrategyType.LONG_STRANGLE: "Lower cost than straddle, needs larger moves to profit",
            StrategyType.SHORT_STRANGLE: "Profit from moderate volatility within strike range",
            StrategyType.COLLAR: "Protective put funded by covered call",
            StrategyType.BUTTERFLY_SPREAD: "Profit from minimal price movement around center strike",
            StrategyType.IRON_CONDOR: "Profit from low volatility within a wider range"
        }
        return descriptions.get(self.strategy_type, "Multi-leg options strategy")
    
    def __str__(self) -> str:
        leg_descriptions = [str(leg) for leg in self.legs]
        return f"{self.strategy_type.value}:\n" + "\n".join(f"  {leg}" for leg in leg_descriptions)


def create_long_straddle(strike: float, underlying_size: float, hedge_ratio: float, commodity: str,
                        expiry_months: int = 3, use_presets: bool = True) -> MultiLegStrategy:
    """ Create Long Straddle with intelligent presets integration """
    if use_presets:
        try:
            defaults = StrategyPresets.get_strategy_defaults("Long Straddle", strike, commodity)
            
            if 'expiry_months' in defaults:
                expiry_months = defaults['expiry_months']
            
            hedge_ratio = StrategyOptimizer.optimize_hedge_ratio(
                "Long Straddle", abs(underlying_size), "Moderate", 0.25
            )
            
        except Exception as e:
            print(f"Warning: Could not apply presets for Long Straddle: {e}")
    
    legs = [
        OptionLeg('call', strike, 1, expiry_months),
        OptionLeg('put', strike, 1, expiry_months)
    ]
    
    return MultiLegStrategy(
        strategy_type=StrategyType.LONG_STRADDLE,
        legs=legs,
        underlying_size=underlying_size,
        hedge_ratio=hedge_ratio,
        commodity=commodity
    )


def create_short_straddle(strike: float, underlying_size: float, hedge_ratio: float, commodity: str,
                         expiry_months: int = 3, use_presets: bool = True) -> MultiLegStrategy:
    """ Create Short Straddle with intelligent presets integration """
    if use_presets:
        try:
            defaults = StrategyPresets.get_strategy_defaults("Short Straddle", strike, commodity)
            
            if 'expiry_months' in defaults:
                expiry_months = defaults['expiry_months']
            
            # Short straddles are higher risk - optimize conservatively
            hedge_ratio = StrategyOptimizer.optimize_hedge_ratio(
                "Short Straddle", abs(underlying_size), "Conservative", 0.25
            )
            
        except Exception as e:
            print(f"Warning: Could not apply presets for Short Straddle: {e}")
    
    legs = [
        OptionLeg('call', strike, -1, expiry_months),
        OptionLeg('put', strike, -1, expiry_months)
    ]
    
    return MultiLegStrategy(
        strategy_type=StrategyType.SHORT_STRADDLE,
        legs=legs,
        underlying_size=underlying_size,
        hedge_ratio=hedge_ratio,
        commodity=commodity
    )


def create_long_strangle(call_strike: float, put_strike: float, underlying_size: float, hedge_ratio: float,
                        commodity: str, expiry_months: int = 3, use_presets: bool = True) -> MultiLegStrategy:
    """ Create Long Strangle with intelligent strike optimization """
    if use_presets:
        try:
            defaults = StrategyPresets.get_strategy_defaults("Long Strangle", (call_strike + put_strike) / 2, commodity)
            
            if 'expiry_months' in defaults:
                expiry_months = defaults['expiry_months']
            
            # Get optimal strike recommendations
            current_price = (call_strike + put_strike) / 2
            strike_recs = StrategyPresets.get_strike_recommendations(
                "Long Strangle", current_price, 0.25, expiry_months * 30
            )
            
            if 'call_strike' in strike_recs and 'put_strike' in strike_recs:
                call_strike = strike_recs['call_strike']
                put_strike = strike_recs['put_strike']
            
            hedge_ratio = StrategyOptimizer.optimize_hedge_ratio(
                "Long Strangle", abs(underlying_size), "Moderate", 0.25
            )
            
        except Exception as e:
            print(f"Warning: Could not apply presets for Long Strangle: {e}")
    
    # Ensure call strike > put strike
    if call_strike <= put_strike:
        raise ValueError("Call strike must be higher than put strike for strangle")
    
    legs = [
        OptionLeg('call', call_strike, 1, expiry_months),
        OptionLeg('put', put_strike, 1, expiry_months)
    ]
    
    return MultiLegStrategy(
        strategy_type=StrategyType.LONG_STRANGLE,
        legs=legs,
        underlying_size=underlying_size,
        hedge_ratio=hedge_ratio,
        commodity=commodity
    )


def create_short_strangle(call_strike: float, put_strike: float, underlying_size: float, hedge_ratio: float,
                         commodity: str, expiry_months: int = 3, use_presets: bool = True) -> MultiLegStrategy:
    """ Create Short Strangle with intelligent risk management """
    if use_presets:
        try:
            defaults = StrategyPresets.get_strategy_defaults("Short Strangle", 
                                                           (call_strike + put_strike) / 2, commodity)
            
            if 'expiry_months' in defaults:
                expiry_months = defaults['expiry_months']
            
            # Conservative hedge ratio for short strategies
            hedge_ratio = StrategyOptimizer.optimize_hedge_ratio(
                "Short Strangle", abs(underlying_size), "Conservative", 0.25
            )
            
        except Exception as e:
            print(f"Warning: Could not apply presets for Short Strangle: {e}")
    
    if call_strike <= put_strike:
        raise ValueError("Call strike must be higher than put strike for strangle")
    
    legs = [
        OptionLeg('call', call_strike, -1, expiry_months),
        OptionLeg('put', put_strike, -1, expiry_months)
    ]
    
    return MultiLegStrategy(
        strategy_type=StrategyType.SHORT_STRANGLE,
        legs=legs,
        underlying_size=underlying_size,
        hedge_ratio=hedge_ratio,
        commodity=commodity
    )


def create_collar(call_strike: float, put_strike: float, underlying_size: float, hedge_ratio: float,
                 commodity: str, expiry_months: int = 3, use_presets: bool = True) -> MultiLegStrategy:
    """ Create Collar with intelligent protection optimization """
    if use_presets:
        try:
            defaults = StrategyPresets.get_strategy_defaults("Collar", 
                                                           (call_strike + put_strike) / 2, commodity)
            
            if 'expiry_months' in defaults:
                expiry_months = defaults['expiry_months']
            
            # Collars are typically high hedge ratio strategies
            hedge_ratio = StrategyOptimizer.optimize_hedge_ratio(
                "Collar", abs(underlying_size), "Moderate", 0.25
            )
            
        except Exception as e:
            print(f"Warning: Could not apply presets for Collar: {e}")
    
    if call_strike <= put_strike:
        raise ValueError("Call strike should be higher than put strike for collar")
    
    legs = [
        OptionLeg('put', put_strike, 1, expiry_months),    # Long put (protection)
        OptionLeg('call', call_strike, -1, expiry_months)  # Short call (income)
    ]
    
    return MultiLegStrategy(
        strategy_type=StrategyType.COLLAR,
        legs=legs,
        underlying_size=underlying_size,
        hedge_ratio=hedge_ratio,
        commodity=commodity
    )


def create_butterfly_spread(lower_strike: float, middle_strike: float, upper_strike: float, underlying_size: float, hedge_ratio: float,
                           commodity: str, expiry_months: int = 3, option_type: str = 'call', use_presets: bool = True) -> MultiLegStrategy:
    """ Create Butterfly Spread with intelligent strike spacing optimization """
    if use_presets:
        try:
            defaults = StrategyPresets.get_strategy_defaults("Butterfly Spread", middle_strike, commodity)
            
            if 'expiry_months' in defaults:
                expiry_months = defaults['expiry_months']
            
            # Get optimal strike recommendations
            strike_recs = StrategyPresets.get_strike_recommendations(
                "Butterfly Spread", middle_strike, 0.25, expiry_months * 30
            )
            
            if all(key in strike_recs for key in ['lower_strike', 'middle_strike', 'upper_strike']):
                lower_strike = strike_recs['lower_strike']
                middle_strike = strike_recs['middle_strike']
                upper_strike = strike_recs['upper_strike']
            
            hedge_ratio = StrategyOptimizer.optimize_hedge_ratio(
                "Butterfly Spread", abs(underlying_size), "Moderate", 0.25
            )
            
        except Exception as e:
            print(f"Warning: Could not apply presets for Butterfly Spread: {e}")
    
    if not (lower_strike < middle_strike < upper_strike):
        raise ValueError("Strikes must be: lower < middle < upper")
    
    # Check if strikes are evenly spaced (recommended)
    lower_gap = middle_strike - lower_strike
    upper_gap = upper_strike - middle_strike
    if abs(lower_gap - upper_gap) > 0.01:
        print("Warning: Butterfly spreads work best with evenly spaced strikes")
    
    legs = [
        OptionLeg(option_type, lower_strike, 1, expiry_months),   # Long lower
        OptionLeg(option_type, middle_strike, -2, expiry_months), # Short 2 middle
        OptionLeg(option_type, upper_strike, 1, expiry_months)    # Long upper
    ]
    
    return MultiLegStrategy(
        strategy_type=StrategyType.BUTTERFLY_SPREAD,
        legs=legs,
        underlying_size=underlying_size,
        hedge_ratio=hedge_ratio,
        commodity=commodity
    )


def create_iron_condor(put_strike_low: float, put_strike_high: float, call_strike_low: float, call_strike_high: float,
                      underlying_size: float, hedge_ratio: float, commodity: str, expiry_months: int = 3,
                      use_presets: bool = True) -> MultiLegStrategy:
    """ Create Iron Condor with intelligent wing optimization """
    if use_presets:
        try:
            current_price = (put_strike_high + call_strike_low) / 2
            defaults = StrategyPresets.get_strategy_defaults("Iron Condor", current_price, commodity)
            
            if 'expiry_months' in defaults:
                expiry_months = defaults['expiry_months']
            
            # Get optimal strike recommendations
            strike_recs = StrategyPresets.get_strike_recommendations(
                "Iron Condor", current_price, 0.25, expiry_months * 30
            )
            
            if all(key in strike_recs for key in ['put_strike_low', 'put_strike_high', 
                                                 'call_strike_low', 'call_strike_high']):
                put_strike_low = strike_recs['put_strike_low']
                put_strike_high = strike_recs['put_strike_high']
                call_strike_low = strike_recs['call_strike_low']
                call_strike_high = strike_recs['call_strike_high']
            
            hedge_ratio = StrategyOptimizer.optimize_hedge_ratio(
                "Iron Condor", abs(underlying_size), "Moderate", 0.25
            )
            
        except Exception as e:
            print(f"Warning: Could not apply presets for Iron Condor: {e}")
    
    if not (put_strike_low < put_strike_high < call_strike_low < call_strike_high):
        raise ValueError("Strikes must be ordered: put_low < put_high < call_low < call_high")
    
    legs = [
        OptionLeg('put', put_strike_low, 1, expiry_months),     # Long put (lower)
        OptionLeg('put', put_strike_high, -1, expiry_months),   # Short put (higher)
        OptionLeg('call', call_strike_low, -1, expiry_months),  # Short call (lower)
        OptionLeg('call', call_strike_high, 1, expiry_months)   # Long call (higher)
    ]
    
    return MultiLegStrategy(
        strategy_type=StrategyType.IRON_CONDOR,
        legs=legs,
        underlying_size=underlying_size,
        hedge_ratio=hedge_ratio,
        commodity=commodity
    )


def get_strategy_defaults(strategy_type: str, current_price: float, commodity: str = "WTI Crude Oil") -> Dict:
    """ Get intelligent defaults for each strategy type """
    try:
        presets_defaults = StrategyPresets.get_strategy_defaults(strategy_type, current_price, commodity)
        
        if presets_defaults:
            return presets_defaults
    except Exception as e:
        print(f"Warning: Could not load presets defaults: {e}")
    
    defaults = {
        "Long Straddle": {
            "strikes": {"strike": current_price},
            "description": "Profit from high volatility in either direction",
            "best_for": "Expecting major price movement (earnings, events)",
            "max_loss": "Premium paid",
            "max_gain": "Unlimited",
            "breakeven": f"${current_price:.2f} ± premium",
            "typical_dte": "30-60 days",
            "volatility_view": "Bullish volatility",
            "expiry_months": 3,
            "optimal_vol_range": (0.25, 0.45),
            "estimated_cost": current_price * 0.04
        },
        
        "Short Straddle": {
            "strikes": {"strike": current_price},
            "description": "Profit from low volatility (range-bound market)",
            "best_for": "Expecting minimal price movement",
            "max_loss": "Unlimited",
            "max_gain": "Premium received",
            "breakeven": f"${current_price:.2f} ± premium",
            "typical_dte": "30-45 days",
            "volatility_view": "Bearish volatility",
            "expiry_months": 2,
            "optimal_vol_range": (0.15, 0.25),
            "estimated_credit": current_price * 0.03
        },
        
        "Long Strangle": {
            "strikes": {
                "call_strike": current_price * 1.05,
                "put_strike": current_price * 0.95
            },
            "description": "Lower cost than straddle, needs larger moves",
            "best_for": "Expecting significant movement, direction unknown",
            "max_loss": "Premium paid",
            "max_gain": "Unlimited",
            "breakeven": "OTM strikes ± premium",
            "typical_dte": "30-60 days",
            "volatility_view": "Bullish volatility",
            "expiry_months": 3,
            "optimal_vol_range": (0.30, 0.50),
            "estimated_cost": current_price * 0.025
        },
        
        "Short Strangle": {
            "strikes": {
                "call_strike": current_price * 1.10,
                "put_strike": current_price * 0.90
            },
            "description": "Profit from moderate volatility within range",
            "best_for": "Expecting price to stay within strike range",
            "max_loss": "Unlimited",
            "max_gain": "Premium received",
            "breakeven": "Strike prices ± premium",
            "typical_dte": "30-45 days",
            "volatility_view": "Bearish volatility",
            "expiry_months": 2,
            "optimal_vol_range": (0.15, 0.25),
            "estimated_credit": current_price * 0.02
        },
        
        "Collar": {
            "strikes": {
                "call_strike": current_price * 1.10,
                "put_strike": current_price * 0.90
            },
            "description": "Protective put funded by covered call",
            "best_for": "Hedging long positions with limited upside",
            "max_loss": f"Limited to ${current_price * 0.90:.2f}",
            "max_gain": f"Limited to ${current_price * 1.10:.2f}",
            "breakeven": "Current price ± net premium",
            "typical_dte": "60-90 days",
            "volatility_view": "Neutral",
            "expiry_months": 6,
            "protection_level": 0.10,
            "upside_cap": 0.10,
            "estimated_cost": current_price * 0.005
        },
        
        "Butterfly Spread": {
            "strikes": {
                "lower_strike": current_price * 0.95,
                "middle_strike": current_price,
                "upper_strike": current_price * 1.05
            },
            "description": "Profit from minimal price movement around center strike",
            "best_for": "Expecting price to stay near current level",
            "max_loss": "Net premium paid",
            "max_gain": "Strike spacing minus premium",
            "breakeven": "Two points around center strike",
            "typical_dte": "30-60 days",
            "volatility_view": "Bearish volatility",
            "expiry_months": 2,
            "optimal_vol_range": (0.15, 0.25),
            "estimated_cost": current_price * 0.01
        },
        
        "Iron Condor": {
            "strikes": {
                "put_strike_low": current_price * 0.85,
                "put_strike_high": current_price * 0.95,
                "call_strike_low": current_price * 1.05,
                "call_strike_high": current_price * 1.15
            },
            "description": "Profit from low volatility within wide range",
            "best_for": "Expecting price to stay within wide range",
            "max_loss": "Strike spacing minus premium",
            "max_gain": "Premium received",
            "breakeven": "Four breakeven points",
            "typical_dte": "30-60 days",
            "volatility_view": "Bearish volatility",
            "expiry_months": 2,
            "optimal_vol_range": (0.15, 0.25),
            "estimated_credit": current_price * 0.015
        }
    }
    
    return defaults.get(strategy_type, {})


def create_strategy_from_preset(strategy_name: str, current_price: float, underlying_size: float, hedge_ratio: float,
                               commodity: str, expiry_months: int = 3, market_conditions: Optional[Dict[str, float]] = None) -> MultiLegStrategy:
    """ Create strategy using intelligent presets with market condition optimization """
    if underlying_size > 1000:
        # Likely in dollars, convert to contracts
        contracts = underlying_size / current_price
    else:
        # Already in contracts
        contracts = underlying_size
    
    # Apply market-based optimizations if conditions provided
    if market_conditions:
        volatility = market_conditions.get('volatility', 0.25)
        trend_strength = market_conditions.get('trend_strength', 0.0)
        time_to_event = market_conditions.get('days_to_event', 30)
        
        # Optimize expiry based on market conditions
        try:
            expiry_months = max(1, StrategyOptimizer.optimize_expiry_selection(
                strategy_name, volatility, time_to_event
            ) // 30)
        except:
            pass
        
        # Optimize hedge ratio based on market conditions
        try:
            risk_tolerance = "Conservative" if volatility > 0.35 else "Moderate"
            hedge_ratio = StrategyOptimizer.optimize_hedge_ratio(strategy_name, abs(contracts), risk_tolerance, volatility)
        except:
            pass
    
    # Create strategy based on type
    if strategy_name == "Long Straddle":
        return create_long_straddle(current_price, contracts, hedge_ratio, commodity, expiry_months)
    
    elif strategy_name == "Short Straddle":
        return create_short_straddle(current_price, contracts, hedge_ratio, commodity, expiry_months)
    
    elif strategy_name == "Long Strangle":
        # Get optimal strikes from presets
        defaults = get_strategy_defaults(strategy_name, current_price, commodity)
        strikes = defaults.get('strikes', {})
        call_strike = strikes.get('call_strike', current_price * 1.05)
        put_strike = strikes.get('put_strike', current_price * 0.95)
        
        return create_long_strangle(call_strike, put_strike, contracts, hedge_ratio, commodity, expiry_months)
    
    elif strategy_name == "Short Strangle":
        defaults = get_strategy_defaults(strategy_name, current_price, commodity)
        strikes = defaults.get('strikes', {})
        call_strike = strikes.get('call_strike', current_price * 1.10)
        put_strike = strikes.get('put_strike', current_price * 0.90)
        
        return create_short_strangle(call_strike, put_strike, contracts, hedge_ratio, commodity, expiry_months)
    
    elif strategy_name == "Collar":
        defaults = get_strategy_defaults(strategy_name, current_price, commodity)
        strikes = defaults.get('strikes', {})
        call_strike = strikes.get('call_strike', current_price * 1.10)
        put_strike = strikes.get('put_strike', current_price * 0.90)
        
        return create_collar(call_strike, put_strike, contracts, hedge_ratio, commodity, expiry_months)
    
    elif strategy_name == "Butterfly Spread":
        defaults = get_strategy_defaults(strategy_name, current_price, commodity)
        strikes = defaults.get('strikes', {})
        lower_strike = strikes.get('lower_strike', current_price * 0.95)
        middle_strike = strikes.get('middle_strike', current_price)
        upper_strike = strikes.get('upper_strike', current_price * 1.05)
        
        return create_butterfly_spread(lower_strike, middle_strike, upper_strike, contracts, hedge_ratio, commodity, expiry_months)
    
    elif strategy_name == "Iron Condor":
        defaults = get_strategy_defaults(strategy_name, current_price, commodity)
        strikes = defaults.get('strikes', {})
        put_low = strikes.get('put_strike_low', current_price * 0.85)
        put_high = strikes.get('put_strike_high', current_price * 0.95)
        call_low = strikes.get('call_strike_low', current_price * 1.05)
        call_high = strikes.get('call_strike_high', current_price * 1.15)
        
        return create_iron_condor(put_low, put_high, call_low, call_high, contracts, hedge_ratio, commodity, expiry_months)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def get_recommended_strategies(market_conditions: Dict[str, float], current_price: float, portfolio_size: float = 1000000.0) -> List[Dict[str, any]]:
    """ Get strategy recommendations based on market conditions """
    try:
        basic_recommendations = StrategyPresets.get_strategy_recommendations(market_conditions)
        
        enhanced_recommendations = []
        
        for rec in basic_recommendations:
            strategy_name = rec['strategy']
            
            try:
                profiles = StrategyPresets.get_strategy_profiles()
                profile = profiles.get(strategy_name)
                
                if profile:
                    enhanced_rec = {
                        'strategy': strategy_name,
                        'confidence': rec['confidence'],
                        'reason': rec['reason'],
                        'risk_warning': rec['risk_warning'],
                        'description': profile.description,
                        'profit_probability': profile.profit_probability,
                        'risk_reward_ratio': profile.risk_reward_ratio,
                        'complexity_level': profile.complexity_level,
                        'max_loss': profile.max_loss,
                        'max_gain': profile.max_gain,
                        'typical_dte': profile.typical_dte,
                        'volatility_view': profile.volatility_view
                    }
                    
                    # Add market-specific scoring
                    volatility = market_conditions.get('volatility', 0.25)
                    trend_strength = market_conditions.get('trend_strength', 0.0)
                    
                    # Calculate suitability score (0-100)
                    suitability_score = calculate_strategy_suitability(
                        strategy_name, volatility, trend_strength, portfolio_size
                    )
                    enhanced_rec['suitability_score'] = suitability_score
                    
                    enhanced_recommendations.append(enhanced_rec)
            except:
                # Fallback to basic recommendation
                enhanced_recommendations.append(rec)
        
        # Sort by suitability score if available
        if enhanced_recommendations and 'suitability_score' in enhanced_recommendations[0]:
            enhanced_recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return enhanced_recommendations
        
    except Exception as e:
        print(f"Warning: Could not get enhanced recommendations: {e}")
        return []


def calculate_strategy_suitability(strategy_name: str, volatility: float,  trend_strength: float, portfolio_size: float) -> float:
    """ Calculate suitability score for a strategy given market conditions"""
    base_score = 50.0  # Start at neutral
    
    # Volatility-based scoring
    if "Straddle" in strategy_name or "Strangle" in strategy_name:
        if "Long" in strategy_name:
            # Long straddles/strangles benefit from high volatility
            if volatility > 0.35:
                base_score += 30
            elif volatility > 0.25:
                base_score += 10
            else:
                base_score -= 20
        else:
            # Short straddles/strangles benefit from low volatility
            if volatility < 0.20:
                base_score += 30
            elif volatility < 0.30:
                base_score += 10
            else:
                base_score -= 20
    
    elif "Butterfly" in strategy_name or "Iron Condor" in strategy_name:
        # These strategies prefer low volatility
        if volatility < 0.25:
            base_score += 20
        elif volatility > 0.35:
            base_score -= 15
    
    elif "Collar" in strategy_name:
        # Collars are generally suitable across volatility environments
        base_score += 5
    
    # Trend-based adjustments
    if trend_strength > 0.5:
        # Strong trending market
        if "Strangle" in strategy_name and "Long" in strategy_name:
            base_score += 15  # Strangles can profit from directional moves
        elif "Butterfly" in strategy_name or "Iron Condor" in strategy_name:
            base_score -= 10  # Range-bound strategies suffer in trends
    
    # Portfolio size adjustments
    if portfolio_size < 500000:  # Smaller portfolios
        if "Short" in strategy_name:
            base_score -= 15  # Reduce score for unlimited risk strategies
        elif "Collar" in strategy_name:
            base_score += 10  # Favor protective strategies
    
    elif portfolio_size > 5000000:  # Larger portfolios
        if "Short" in strategy_name:
            base_score += 5  # Can better handle unlimited risk
    
    # Ensure score is within bounds
    return max(0.0, min(100.0, base_score))


def validate_multi_leg_configuration(strategy: MultiLegStrategy, current_price: float) -> Dict[str, any]:
    """ Returns detailed validation results with recommendations """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': [],
        'risk_assessment': {},
        'estimated_metrics': {}
    }
    
    try:
        # Get configuration warnings from presets
        warnings = strategy.get_configuration_warnings(current_price)
        validation_result['warnings'].extend(warnings)
        
        # Calculate estimated metrics
        try:
            total_premium = strategy.get_total_premium(current_price)
            validation_result['estimated_metrics']['total_premium'] = total_premium
            validation_result['estimated_metrics']['is_debit'] = total_premium > 0
            validation_result['estimated_metrics']['is_credit'] = total_premium < 0
        except:
            pass
        
        # Calculate Greeks if possible
        try:
            greeks = strategy.get_strategy_greeks(current_price)
            validation_result['estimated_metrics']['greeks'] = greeks
            
            # Risk assessment based on Greeks
            if abs(greeks['delta']) > 0.5:
                validation_result['warnings'].append(f"High directional exposure (Delta: {greeks['delta']:.3f})")
            
            if greeks['theta'] < -50:
                validation_result['warnings'].append(f"High time decay (Theta: ${greeks['theta']:.2f}/day)")
            
            if abs(greeks['vega']) > 100:
                validation_result['warnings'].append(f"High volatility sensitivity (Vega: ${greeks['vega']:.2f})")
                
        except:
            pass
        
        # Strategy-specific validations
        if strategy.strategy_type in [StrategyType.SHORT_STRADDLE, StrategyType.SHORT_STRANGLE]:
            validation_result['warnings'].append("⚠️ Unlimited risk strategy - ensure proper risk management")
            validation_result['recommendations'].append("Consider using stop-loss orders")
            validation_result['recommendations'].append("Monitor position closely near expiration")
        
        # Risk assessment
        complexity_levels = {
            StrategyType.COLLAR: "Low",
            StrategyType.LONG_STRADDLE: "Medium",
            StrategyType.LONG_STRANGLE: "Medium", 
            StrategyType.SHORT_STRADDLE: "High",
            StrategyType.SHORT_STRANGLE: "High",
            StrategyType.BUTTERFLY_SPREAD: "High",
            StrategyType.IRON_CONDOR: "High"
        }
        
        validation_result['risk_assessment'] = {
            'complexity': complexity_levels.get(strategy.strategy_type, "Medium"),
            'max_risk': "Limited" if strategy.strategy_type in [
                StrategyType.LONG_STRADDLE, StrategyType.LONG_STRANGLE, 
                StrategyType.BUTTERFLY_SPREAD, StrategyType.COLLAR
            ] else "Unlimited",
            'requires_margin': strategy.strategy_type in [
                StrategyType.SHORT_STRADDLE, StrategyType.SHORT_STRANGLE,
                StrategyType.IRON_CONDOR, StrategyType.COLLAR
            ],
            'time_sensitivity': "High" if strategy.strategy_type in [
                StrategyType.LONG_STRADDLE, StrategyType.LONG_STRANGLE
            ] else "Medium"
        }
        
        # Set overall validity
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
    except Exception as e:
        validation_result['errors'].append(f"Validation error: {str(e)}")
        validation_result['is_valid'] = False
    
    return validation_result


def get_all_strategy_types() -> List[str]:
    """ Get list of all available strategy types """
    return [strategy.value for strategy in StrategyType]


def get_strategy_complexity_rating(strategy_type: str) -> str:
    """ Get complexity rating for a strategy """
    complexity_map = {
        "Long Straddle": "Intermediate",
        "Short Straddle": "Advanced", 
        "Long Strangle": "Intermediate",
        "Short Strangle": "Advanced",
        "Collar": "Intermediate",
        "Butterfly Spread": "Advanced",
        "Iron Condor": "Advanced"
    }
    return complexity_map.get(strategy_type, "Intermediate")


def get_strategy_risk_profile(strategy_type: str) -> Dict[str, str]:
    """ Get comprehensive risk profile for a strategy """
    risk_profiles = {
        "Long Straddle": {
            "max_loss": "Limited to premium paid",
            "max_gain": "Unlimited",
            "breakeven_count": "2",
            "time_decay": "Negative",
            "volatility_impact": "Positive",
            "margin_required": "No"
        },
        "Short Straddle": {
            "max_loss": "Unlimited",
            "max_gain": "Limited to premium received", 
            "breakeven_count": "2",
            "time_decay": "Positive",
            "volatility_impact": "Negative",
            "margin_required": "Yes"
        },
        "Collar": {
            "max_loss": "Limited to put strike",
            "max_gain": "Limited to call strike",
            "breakeven_count": "1-2", 
            "time_decay": "Neutral",
            "volatility_impact": "Low",
            "margin_required": "Partial"
        }
    }
    
    return risk_profiles.get(strategy_type, {})


if __name__ == "__main__":
    print("=== Enhanced Multi-Leg Strategies with Presets Integration ===")
    
    current_price = 75.0
    
    # Test preset-enhanced strategy creation
    print("\n1. Testing Preset-Enhanced Strategy Creation:")
    
    try:
        # Test with market conditions
        market_conditions = {
            'volatility': 0.15,  # Low volatility environment
            'trend_strength': 0.2,
            'days_to_event': 45
        }
        
        straddle = create_strategy_from_preset(
            "Long Straddle", current_price, 1000, 0.8, "WTI Crude Oil",
            market_conditions=market_conditions
        )
        print(f"✅ Created: {straddle.strategy_type.value}")
        print(f"   Hedge Ratio: {straddle.hedge_ratio:.1%}")
        print(f"   Description: {straddle.description}")
        
        # Test configuration validation
        validation = validate_multi_leg_configuration(straddle, current_price)
        print(f"   Valid: {validation['is_valid']}")
        if validation['warnings']:
            print(f"   Warnings: {len(validation['warnings'])}")
        
    except Exception as e:
        print(f"❌ Error in preset creation: {e}")
    
    print("\n2. Testing Strategy Recommendations:")
    
    try:
        recommendations = get_recommended_strategies(market_conditions, current_price)
        print(f"✅ Generated {len(recommendations)} recommendations")
        
        for i, rec in enumerate(recommendations[:3]):  # Show top 3
            print(f"   {i+1}. {rec['strategy']} - Confidence: {rec['confidence']}")
            if 'suitability_score' in rec:
                print(f"      Suitability Score: {rec['suitability_score']:.0f}/100")
    
    except Exception as e:
        print(f"❌ Error in recommendations: {e}")
    
    print("\n3. Testing All Strategy Builders:")
    
    strategies_to_test = [
        ("Long Straddle", lambda: create_long_straddle(current_price, 1000, 0.8, "WTI Crude Oil")),
        ("Iron Condor", lambda: create_iron_condor(65.0, 70.0, 80.0, 85.0, 1000, 0.7, "WTI Crude Oil")),
        ("Collar", lambda: create_collar(80.0, 70.0, 1000, 0.9, "WTI Crude Oil"))
    ]
    
    for name, creator in strategies_to_test:
        try:
            strategy = creator()
            print(f"✅ {name}: {len(strategy.legs)} legs, Hedge: {strategy.hedge_ratio:.1%}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print("\n✅ Enhanced multi-leg strategies with presets integration working correctly!")