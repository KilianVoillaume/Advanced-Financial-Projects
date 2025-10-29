"""
hedging/portfolio.py

Core multi-commodity portfolio management with multi-leg options support.
Handles portfolio construction, risk analytics, and optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import warnings

from .data import get_prices, get_current_price
from .simulation import simulate_hedged_vs_unhedged
from .risk import calculate_risk_metrics
from .multi_leg_strategies import MultiLegStrategy


@dataclass(frozen=True)
class Position:
    commodity: str
    size: float
    hedge_ratio: float
    strategy: str = "Futures"
    
    # Single option fields
    strike_price: Optional[float] = None
    option_type: Optional[str] = None
    
    # Multi-leg strategy support
    multi_leg_strategy: Optional[MultiLegStrategy] = None
    
    current_price: Optional[float] = None
    
    def __post_init__(self):
        if self.current_price is None:
            try:
                price = get_current_price(self.commodity)
                object.__setattr__(self, 'current_price', price)
            except Exception:
                fallback_prices = {
                    "WTI Crude Oil": 75.0,
                    "Brent Crude Oil": 78.0,
                    "Natural Gas": 3.5
                }
                fallback_price = fallback_prices.get(self.commodity, 75.0)
                object.__setattr__(self, 'current_price', fallback_price)
        
        # Validate strategy consistency
        if self.strategy == "Multi-Leg" and self.multi_leg_strategy is None:
            raise ValueError("Multi-Leg strategy requires multi_leg_strategy parameter")
        
        if self.multi_leg_strategy is not None and self.strategy != "Multi-Leg":
            object.__setattr__(self, 'strategy', "Multi-Leg")
    
    @property
    def direction(self) -> str:
        return "Long" if self.size > 0 else "Short"
    
    @property
    def abs_size(self) -> float:
        return abs(self.size)
    
    @property
    def notional_value(self) -> float:
        return self.abs_size * self.current_price
    
    @property
    def is_hedged(self) -> bool:
        return self.hedge_ratio > 0
    
    @property
    def is_multi_leg(self) -> bool:
        return self.multi_leg_strategy is not None
    
    @property
    def strategy_description(self) -> str:
        """Get human-readable strategy description"""
        if self.is_multi_leg:
            return self.multi_leg_strategy.strategy_type.value
        return self.strategy
    
    def with_hedge_ratio(self, new_ratio: float) -> 'Position':
        return replace(self, hedge_ratio=new_ratio)
    
    def with_size(self, new_size: float) -> 'Position':
        return replace(self, size=new_size)
    
    def get_position_greeks(self) -> Dict[str, float]:
        """Calculate Greeks for this position"""
        if self.is_multi_leg:
            return self.multi_leg_strategy.get_strategy_greeks(self.current_price)
        
        elif self.strategy == "Options" and self.strike_price:
            try:
                from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, get_commodity_volatility, time_to_expiration
                
                current_price = self.current_price
                strike = self.strike_price
                time_to_exp = time_to_expiration(3)
                risk_free_rate = get_risk_free_rate()
                volatility = get_commodity_volatility(self.commodity)
                
                option_type = self.option_type.lower() if self.option_type else 'put'
                
                greeks = BlackScholesCalculator.calculate_greeks(
                    current_price, strike, time_to_exp, risk_free_rate, volatility, option_type
                )
                
                multiplier = self.size * self.hedge_ratio
                
                return {
                    'delta': greeks['delta'] * multiplier,
                    'gamma': greeks['gamma'] * multiplier,
                    'theta': greeks['theta'] * multiplier,
                    'vega': greeks['vega'] * multiplier,
                    'rho': greeks['rho'] * multiplier
                }
                
            except Exception as e:
                print(f"Error calculating Greeks: {e}")
                return self._zero_greeks()
        
        return self._zero_greeks()
    
    def _zero_greeks(self) -> Dict[str, float]:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}


@dataclass
class PortfolioRiskMetrics:
    expected_pnl: float
    var_95: float
    cvar_95: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    num_positions: int
    total_notional: float
    time_horizon: str = "1-Day"  
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {
            'Expected P&L': f"${self.expected_pnl:,.0f}",
            'Expected P&L (Daily)': f"${self.expected_pnl:,.0f}/day",  
            'Expected P&L (Monthly)': f"${self.expected_pnl * 21:,.0f}/month",  
            'Expected P&L (Annual)': f"${self.expected_pnl * 252:,.0f}/year",  
            'VaR (95%)': f"${self.var_95:,.0f}",
            'VaR (95% Daily)': f"${self.var_95:,.0f}/day", 
            'CVaR (95%)': f"${self.cvar_95:,.0f}",
            'CVaR (95% Daily)': f"${self.cvar_95:,.0f}/day",  
            'Volatility': f"${self.volatility:,.0f}",
            'Daily Volatility': f"${self.volatility:,.0f}/day",  
            'Sharpe Ratio': f"{self.sharpe_ratio:.3f}",
            'Sharpe Ratio (Daily)': f"{self.sharpe_ratio:.3f} (daily)",  
            'Max Drawdown': f"${self.max_drawdown:,.0f}",
            'Positions': str(self.num_positions),
            'Total Notional': f"${self.total_notional:,.0f}",
            'Time Horizon': self.time_horizon  
        }


class PortfolioManager:
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.price_data: Dict[str, pd.Series] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.portfolio_risk: Optional[PortfolioRiskMetrics] = None
        self._simulation_cache = None  
        self._last_simulation_hash = None
        
        self.config = {
            'correlation_window': 252,
            'simulation_runs': 5000,
            'confidence_level': 0.95,
            'risk_free_rate': 0.05
        }
        
        self._cache = {
            'correlations': None,
            'portfolio_simulation': None,
            'risk_metrics': None,
            'last_update': None
        }
        
        self._portfolio_hash = None
    
    def add_position(self, name: str, position: Position) -> 'PortfolioManager':
        self.positions[name] = position
        self._invalidate_cache()
        self._load_price_data_async(position.commodity)
        self.clear_simulation_cache()
        return self
    
    def remove_position(self, name: str) -> 'PortfolioManager':
        if name in self.positions:
            del self.positions[name]
            self._invalidate_cache()
        self.clear_simulation_cache()
        return self
    
    def update_position(self, name: str, **kwargs) -> 'PortfolioManager':
        if name in self.positions:
            current_pos = self.positions[name]
            self.positions[name] = replace(current_pos, **kwargs)
            self._invalidate_cache()
        return self
    
    def set_config(self, **config_updates) -> 'PortfolioManager':
        self.config.update(config_updates)
        self._invalidate_cache()
        return self
    
    def _get_portfolio_hash(self) -> str:
        if not self.positions:
            return "empty"
        
        position_strings = []
        for name, pos in sorted(self.positions.items()):
            pos_str = f"{name}:{pos.commodity}:{pos.size}:{pos.hedge_ratio}:{pos.strategy}"
            if pos.multi_leg_strategy:
                pos_str += f":{pos.multi_leg_strategy.strategy_type.value}"
            position_strings.append(pos_str)
        
        return str(hash("|".join(position_strings)))

    def get_simulation_data(self, force_recalculate: bool = False) -> np.ndarray:
        return self._simulate_portfolio_pnl(force_recalculate)

    def clear_simulation_cache(self):
        self._simulation_cache = None
        self._last_simulation_hash = None
        self._cache['risk_metrics'] = None

    def calculate_correlations(self, force_recalculate: bool = False) -> 'PortfolioManager':
        if not force_recalculate and self._cache['correlations'] is not None:
            self.correlation_matrix = self._cache['correlations']
            return self
        
        commodities = list(set(pos.commodity for pos in self.positions.values()))
        
        if len(commodities) < 2:
            self.correlation_matrix = pd.DataFrame()
            return self
        
        try:
            aligned_returns = {}
            min_length = float('inf')
            
            for commodity in commodities:
                if commodity in self.price_data and not self.price_data[commodity].empty:
                    returns = self.price_data[commodity].pct_change().dropna()
                    returns = returns.tail(self.config['correlation_window'])
                    aligned_returns[commodity] = returns
                    min_length = min(min_length, len(returns))
            
            if len(aligned_returns) >= 2 and min_length > 10:
                for commodity in aligned_returns:
                    aligned_returns[commodity] = aligned_returns[commodity].tail(int(min_length))
                
                returns_df = pd.DataFrame(aligned_returns).dropna()
                if not returns_df.empty:
                    self.correlation_matrix = returns_df.corr()
                    self._cache['correlations'] = self.correlation_matrix
                else:
                    self.correlation_matrix = pd.DataFrame()
            else:
                self.correlation_matrix = pd.DataFrame()
                
        except Exception as e:
            print(f"Warning: Could not calculate correlations: {e}")
            self.correlation_matrix = pd.DataFrame()
        
        return self
    
    def calculate_portfolio_risk(self, force_recalculate: bool = False) -> 'PortfolioManager':
        current_hash = self._get_portfolio_hash()
        
        if (not force_recalculate and 
            self._cache['risk_metrics'] is not None and
            self._last_simulation_hash == current_hash):
            self.portfolio_risk = self._cache['risk_metrics']
            return self
        
        if len(self.positions) == 0:
            return self
        
        try:
            portfolio_pnl = self._simulate_portfolio_pnl(force_recalculate)
            
            if len(portfolio_pnl) == 0:
                return self
            
            confidence = self.config['confidence_level']
            var_percentile = (1 - confidence) * 100
            
            expected_pnl = float(np.mean(portfolio_pnl))
            var_95 = float(np.percentile(portfolio_pnl, var_percentile))
            
            losses_worse_than_var = portfolio_pnl[portfolio_pnl <= var_95]
            cvar_95 = float(np.mean(losses_worse_than_var)) if len(losses_worse_than_var) > 0 else var_95
            
            volatility = float(np.std(portfolio_pnl))
            sharpe_ratio = expected_pnl / volatility if volatility > 0 else 0
            
            cumulative_pnl = np.cumsum(portfolio_pnl)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdowns = cumulative_pnl - running_max
            max_drawdown = float(np.min(drawdowns))
            
            total_notional = sum(pos.notional_value for pos in self.positions.values())
            
            self.portfolio_risk = PortfolioRiskMetrics(
                expected_pnl=expected_pnl,
                var_95=var_95,
                cvar_95=cvar_95,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                num_positions=len(self.positions),
                total_notional=total_notional
            )
            
            # Cache the results
            self._cache['risk_metrics'] = self.portfolio_risk
            
        except Exception as e:
            print(f"Warning: Could not calculate portfolio risk: {e}")
        
        return self
    
    def get_portfolio_summary(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        
        summary_data = []
        total_notional = sum(pos.notional_value for pos in self.positions.values())
        
        for name, position in self.positions.items():
            weight = (position.notional_value / total_notional * 100) if total_notional > 0 else 0
            
            # Enhanced strategy description for multi-leg
            if position.is_multi_leg:
                strategy_desc = position.strategy_description
                legs_count = len(position.multi_leg_strategy.legs)
                strategy_detail = f"{strategy_desc} ({legs_count} legs)"
            else:
                strategy_detail = position.strategy
            
            summary_data.append({
                'Position Name': name,
                'Commodity': position.commodity,
                'Strategy': strategy_detail,
                'Direction': position.direction,
                'Size': f"{position.abs_size:,.0f}",
                'Current Price': f"${position.current_price:.2f}",
                'Notional Value': f"${position.notional_value:,.0f}",
                'Weight': f"{weight:.1f}%",
                'Hedge Ratio': f"{position.hedge_ratio:.1%}",
                'Strike Price': self._get_strike_display(position)
            })
        
        return pd.DataFrame(summary_data)
    
    def _get_strike_display(self, position: Position) -> str:
        """Get strike price display for position"""
        if position.is_multi_leg:
            strikes = [leg.strike_price for leg in position.multi_leg_strategy.legs]
            unique_strikes = sorted(set(strikes))
            if len(unique_strikes) == 1:
                return f"${unique_strikes[0]:.2f}"
            else:
                return f"${min(unique_strikes):.2f}-${max(unique_strikes):.2f}"
        elif position.strike_price:
            return f"${position.strike_price:.2f}"
        else:
            return "N/A"
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        if not self.positions:
            return {}
        
        total_notional = sum(pos.notional_value for pos in self.positions.values())
        
        if total_notional == 0:
            return {name: 0 for name in self.positions.keys()}
        
        return {
            name: pos.notional_value / total_notional 
            for name, pos in self.positions.items()
        }
    
    def get_commodity_exposure(self) -> pd.DataFrame:
        """Enhanced commodity exposure with category grouping and better formatting"""
        if not self.positions:
            return pd.DataFrame()
        
        # Import the new functions
        from .data import get_commodity_category, get_commodity_specs
        
        commodity_exposure = {}
        category_exposure = {}
        
        for position in self.positions.values():
            commodity = position.commodity
            category = get_commodity_category(commodity)
            specs = get_commodity_specs(commodity)
            
            exposure = position.size * position.current_price
            
            # Commodity-level exposure
            if commodity in commodity_exposure:
                commodity_exposure[commodity] += exposure
            else:
                commodity_exposure[commodity] = exposure
            
            # Category-level exposure  
            if category in category_exposure:
                category_exposure[category] += exposure
            else:
                category_exposure[category] = exposure
        
        # Create enhanced exposure data
        exposure_data = []
        
        for commodity, exposure in commodity_exposure.items():
            category = get_commodity_category(commodity)
            specs = get_commodity_specs(commodity)
            
            # Determine direction and color coding
            if exposure > 0:
                direction = "Long"
                direction_emoji = "📈"
                color_class = "positive"
            elif exposure < 0:
                direction = "Short" 
                direction_emoji = "📉"
                color_class = "negative"
            else:
                direction = "Neutral"
                direction_emoji = "⚖️"
                color_class = "neutral"
            
            exposure_data.append({
                'Category': category,
                'Commodity': commodity,
                'Position Type': self._get_position_types_for_commodity(commodity),  
                'Net Exposure': f"${exposure:,.0f}",
                'Direction': f"{direction_emoji} {direction}",
                'Unit': specs['unit'],
                'Abs Exposure': abs(exposure),  # For sorting
                'Raw Exposure': exposure,  # For calculations
                'Color Class': color_class
            })
        
        exposure_data.sort(key=lambda x: (x['Category'], -x['Abs Exposure']))
        exposure_df = pd.DataFrame(exposure_data)
        
        category_totals = []
        for category, total_exposure in category_exposure.items():
            direction = "Long" if total_exposure > 0 else "Short" if total_exposure < 0 else "Neutral"
            direction_emoji = "📈" if total_exposure > 0 else "📉" if total_exposure < 0 else "⚖️"
            
            category_totals.append({
                'Category': f"📊 {category} Total",
                'Commodity': f"All {category}",
                'Net Exposure': f"${total_exposure:,.0f}",
                'Direction': f"{direction_emoji} {direction}",
                'Unit': "Mixed",
                'Abs Exposure': abs(total_exposure),
                'Raw Exposure': total_exposure,
                'Color Class': 'category_total'
            })
        
        display_columns = ['Category', 'Commodity', 'Position Type', 'Net Exposure', 'Direction', 'Unit']
        
        if not exposure_df.empty:
            return exposure_df[display_columns]
        else:
            return pd.DataFrame(columns=display_columns)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        if self.correlation_matrix is None:
            self.calculate_correlations()
        return self.correlation_matrix if self.correlation_matrix is not None else pd.DataFrame()
    
    def get_portfolio_risk_summary(self) -> Dict[str, Union[str, float]]:
        if self.portfolio_risk is None:
            self.calculate_portfolio_risk()
        
        if self.portfolio_risk is None:
            return {}
        
        return self.portfolio_risk.to_dict()
    
    def _load_price_data_async(self, commodity: str) -> None:
        if commodity not in self.price_data:
            try:
                self.price_data[commodity] = get_prices(commodity, period="1y")
            except Exception as e:
                print(f"Warning: Could not load price data for {commodity}: {e}")
                dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
                dummy_prices = pd.Series(
                    np.random.normal(75, 5, 252), 
                    index=dates,
                    name=commodity
                )
                self.price_data[commodity] = dummy_prices
    
    def _simulate_portfolio_pnl(self, force_recalculate: bool = False) -> np.ndarray:
        current_hash = self._get_portfolio_hash()
        
        # Return cached simulation if portfolio hasn't changed
        if (not force_recalculate and 
            self._simulation_cache is not None and 
            self._last_simulation_hash == current_hash):
            print("Using cached simulation results")
            return self._simulation_cache
        
        print(f"Running new portfolio simulation for {len(self.positions)} positions...")
        
        if not self.positions:
            self._simulation_cache = np.array([])
            return self._simulation_cache
        
        n_sim = self.config['simulation_runs']
        portfolio_pnl = np.zeros(n_sim)
        
        for name, position in self.positions.items():
            print(f"Simulating position: {name} ({position.strategy})")
            
            # Ensure price data exists
            if position.commodity not in self.price_data:
                self._load_price_data_async(position.commodity)
            
            if position.commodity not in self.price_data:
                print(f"No price data for {position.commodity}, skipping...")
                continue
            
            prices = self.price_data[position.commodity]
            if len(prices) < 10:
                print(f"Insufficient price data for {position.commodity}, skipping...")
                continue
            
            try:
                from hedging.simulation import simulate_hedged_vs_unhedged
                
                # Run simulation based on strategy type
                if position.is_multi_leg:
                    sim_result = simulate_hedged_vs_unhedged(
                        prices, position.size, position.hedge_ratio, "Multi-Leg",
                        multi_leg_strategy=position.multi_leg_strategy,
                        n_sim=n_sim
                    )
                elif position.strategy == "Options":
                    sim_result = simulate_hedged_vs_unhedged(
                        prices, position.size, position.hedge_ratio, "Options",
                        strike_price=position.strike_price,
                        n_sim=n_sim
                    )
                elif position.strategy == "Futures":
                    sim_result = simulate_hedged_vs_unhedged(
                        prices, position.size, position.hedge_ratio, "Futures",
                        n_sim=n_sim
                    )
                else:
                    print(f"Unknown strategy for {name}: {position.strategy}")
                    continue
                
                # Add to portfolio P&L
                if 'hedged_pnl' in sim_result and len(sim_result['hedged_pnl']) == n_sim:
                    portfolio_pnl += sim_result['hedged_pnl']
                    print(f"Added {name} P&L: mean={np.mean(sim_result['hedged_pnl']):.0f}")
                else:
                    print(f"Invalid simulation result for {name}")
                    
            except Exception as e:
                print(f"Error simulating position {name}: {e}")
                continue
        
        # Cache the results
        self._simulation_cache = portfolio_pnl
        self._last_simulation_hash = current_hash
        
        print(f"Simulation complete: mean={np.mean(portfolio_pnl):.0f}, std={np.std(portfolio_pnl):.0f}")
        return portfolio_pnl
    
    def _simulate_multi_leg_position(self, position: Position) -> Dict[str, np.ndarray]:
        """Simulate P&L for multi-leg strategy position"""
        try:
            # For now, use simplified simulation
            # This will be enhanced when we implement proper multi-leg simulation
            n_sim = self.config['simulation_runs']
            
            # Generate price scenarios
            prices = self.price_data[position.commodity]
            price_returns = prices.pct_change().dropna()
            
            # Simulate future returns
            simulated_returns = np.random.choice(price_returns.values, size=n_sim, replace=True)
            current_price = position.current_price
            simulated_prices = current_price * (1 + simulated_returns)
            
            # Calculate strategy P&L using simplified Greeks approximation
            strategy_greeks = position.multi_leg_strategy.get_strategy_greeks(current_price)
            
            unhedged_pnl = (simulated_prices - current_price) * position.size
            
            # Approximate hedge P&L using delta
            hedge_pnl = strategy_greeks['delta'] * (simulated_prices - current_price)
            
            hedged_pnl = unhedged_pnl + hedge_pnl
            
            return {
                'unhedged_pnl': unhedged_pnl,
                'hedged_pnl': hedged_pnl,
                'hedge_benefit': hedge_pnl
            }
            
        except Exception as e:
            print(f"Error simulating multi-leg position: {e}")
            n_sim = self.config['simulation_runs']
            return {
                'unhedged_pnl': np.zeros(n_sim),
                'hedged_pnl': np.zeros(n_sim),
                'hedge_benefit': np.zeros(n_sim)
            }
    
    def _get_position_types_for_commodity(self, commodity: str) -> str:
        """Get position types for a specific commodity"""
        position_types = []
        
        for position in self.positions.values():
            if position.commodity == commodity:
                if position.is_multi_leg:
                    strategy_name = position.multi_leg_strategy.strategy_type.value
                    position_types.append(f"Multi-Leg ({strategy_name})")
                elif position.strategy == "Options":
                    option_type = position.option_type.title() if hasattr(position, 'option_type') and position.option_type else "Option"
                    position_types.append(f"{option_type} Option")
                elif position.strategy == "Futures":
                    position_types.append("Futures")
                else:
                    position_types.append(position.strategy)
        
        unique_types = list(dict.fromkeys(position_types))  
        
        if len(unique_types) == 1:
            return unique_types[0]
        elif len(unique_types) <= 3:
            return ", ".join(unique_types)
        else:
            return f"{unique_types[0]}, +{len(unique_types)-1} more"


    def _invalidate_cache(self) -> None:
        self._cache = {key: None for key in self._cache.keys()}
        self._cache['last_update'] = datetime.now()
    
    def copy(self) -> 'PortfolioManager':
        new_portfolio = PortfolioManager()
        new_portfolio.positions = self.positions.copy()
        new_portfolio.price_data = self.price_data.copy()
        new_portfolio.config = self.config.copy()
        return new_portfolio
    
    def clear(self) -> 'PortfolioManager':
        self.positions.clear()
        self.price_data.clear()
        self._invalidate_cache()
        self.clear_simulation_cache()
        return self
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __contains__(self, position_name: str) -> bool:
        return position_name in self.positions
    
    def __str__(self) -> str:
        if not self.positions:
            return "Empty Portfolio"
        
        total_notional = sum(pos.notional_value for pos in self.positions.values())
        return f"Portfolio: {len(self.positions)} positions, ${total_notional:,.0f} notional"
    
    def __repr__(self) -> str:
        return f"PortfolioManager(positions={len(self.positions)}, total_notional=${sum(pos.notional_value for pos in self.positions.values()):,.0f})"


# Factory functions (keep existing ones + add multi-leg support)
def create_oil_position(size: float, hedge_ratio: float = 0.0, strategy: str = "Futures", 
                       strike_price: Optional[float] = None, option_type: Optional[str] = None,
                       multi_leg_strategy: Optional[MultiLegStrategy] = None) -> Position:
    return Position(
        commodity="WTI Crude Oil",
        size=size,
        hedge_ratio=hedge_ratio,
        strategy=strategy,
        strike_price=strike_price,
        option_type=option_type,
        multi_leg_strategy=multi_leg_strategy
    )


def create_gas_position(size: float, hedge_ratio: float = 0.0, strategy: str = "Futures", 
                       strike_price: Optional[float] = None, option_type: Optional[str] = None,
                       multi_leg_strategy: Optional[MultiLegStrategy] = None) -> Position:
    return Position(
        commodity="Natural Gas",
        size=size,
        hedge_ratio=hedge_ratio,
        strategy=strategy,
        strike_price=strike_price,
        option_type=option_type,
        multi_leg_strategy=multi_leg_strategy
    )


def create_brent_position(size: float, hedge_ratio: float = 0.0, strategy: str = "Futures", 
                         strike_price: Optional[float] = None, option_type: Optional[str] = None,
                         multi_leg_strategy: Optional[MultiLegStrategy] = None) -> Position:
    return Position(
        commodity="Brent Crude Oil",
        size=size,
        hedge_ratio=hedge_ratio,
        strategy=strategy,
        strike_price=strike_price,
        option_type=option_type,
        multi_leg_strategy=multi_leg_strategy
    )


def create_sample_portfolio() -> PortfolioManager:
    return (PortfolioManager()
        .add_position("wti_main", create_oil_position(10000, 0.8))
        .add_position("gas_hedge", create_gas_position(-5000, 0.6))
        .add_position("brent_arb", create_brent_position(3000, 0.7))
        .calculate_correlations()
        .calculate_portfolio_risk())


# Multi-leg portfolio creation helper
def create_sample_options_portfolio() -> PortfolioManager:
    """Create sample portfolio with multi-leg options strategies"""
    from .multi_leg_strategies import create_long_straddle, create_collar
    
    # Create multi-leg strategies
    straddle_strategy = create_long_straddle(75.0, 1000, 0.8, "WTI Crude Oil")
    collar_strategy = create_collar(80.0, 70.0, 2000, 0.9, "Brent Crude Oil")
    
    return (PortfolioManager()
        .add_position("oil_main", create_oil_position(5000, 0.7))
        .add_position("volatility_hedge", create_oil_position(1000, 0.8, "Multi-Leg", 
                                                             multi_leg_strategy=straddle_strategy))
        .add_position("collar_protection", create_brent_position(2000, 0.9, "Multi-Leg",
                                                               multi_leg_strategy=collar_strategy))
        .calculate_correlations()
        .calculate_portfolio_risk())


if __name__ == "__main__":
    portfolio = create_sample_options_portfolio()
    print(f"\n{portfolio}")
    print("\nPortfolio Summary:")
    print(portfolio.get_portfolio_summary().to_string(index=False))