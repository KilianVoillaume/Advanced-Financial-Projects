"""
hedging/portfolio.py

Core multi-commodity portfolio management with fluent interface.
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


@dataclass(frozen=True)
class Position:
    commodity: str
    size: float
    hedge_ratio: float
    strategy: str = "Futures"
    strike_price: Optional[float] = None
    option_type: Optional[str] = None
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
    
    @property
    def direction(self) -> str:
        """ Position direction: Long/Short """
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

    
    def with_hedge_ratio(self, new_ratio: float) -> 'Position':
        """ Create new position with different hedge ratio """
        return replace(self, hedge_ratio=new_ratio)

    
    def with_size(self, new_size: float) -> 'Position':
        """ Create new position with different size """
        return replace(self, size=new_size)

    
    def get_position_greeks(self) -> Dict[str, float]:
        if self.strategy != "Options" or not self.strike_price:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        try:
            from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, get_commodity_volatility, time_to_expiration
            
            # TO DEBUG, TO DELETE AFTER EVEYTHING IS CLEAN -------------------------------------------------------------------------
            print(f"DEBUG: Position {self.size}, option_type: {self.option_type}")


            current_price = self.current_price
            strike = self.strike_price
            time_to_exp = time_to_expiration(3)  # 3 months 
            risk_free_rate = get_risk_free_rate()
            volatility = get_commodity_volatility(self.commodity)
            
            option_type = self.option_type.lower() if self.option_type else 'put'
            
            # TO DEBUG, TO DELETE AFTER EVEYTHING IS CLEAN -------------------------------------------------------------------------
            print(f"DEBUG: Using option_type: {option_type}")

            greeks = BlackScholesCalculator.calculate_greeks(
                current_price, strike, time_to_exp, risk_free_rate, volatility, option_type
            )
            
            # Scale by position size and hedge ratio
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
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }


@dataclass
class PortfolioRiskMetrics:
    """ Container for portfolio risk metrics """
    expected_pnl: float
    var_95: float
    cvar_95: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    num_positions: int
    total_notional: float

    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """ Convert to formatted dictionary for display """
        return {
            'Expected P&L': f"${self.expected_pnl:,.0f}",
            'VaR (95%)': f"${self.var_95:,.0f}",
            'CVaR (95%)': f"${self.cvar_95:,.0f}",
            'Volatility': f"${self.volatility:,.0f}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.3f}",
            'Max Drawdown': f"${self.max_drawdown:,.0f}",
            'Positions': str(self.num_positions),
            'Total Notional': f"${self.total_notional:,.0f}"
        }


class PortfolioManager:
    """ Multi-commodity portfolio manager with fluent interface """
    
    def __init__(self):
        """ Empty portfolio """
        self.positions: Dict[str, Position] = {}
        self.price_data: Dict[str, pd.Series] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.portfolio_risk: Optional[PortfolioRiskMetrics] = None
        
        self.config = {
            'correlation_window': 252,  # 1 year
            'simulation_runs': 5000,
            'confidence_level': 0.95,
            'risk_free_rate': 0.05
        }
        
        # Performance cache
        self._cache = {
            'correlations': None,
            'portfolio_simulation': None,
            'risk_metrics': None,
            'last_update': None
        }
        
        # Track if portfolio has changed
        self._portfolio_hash = None

    
    def add_position(self, name: str, position: Position) -> 'PortfolioManager':
       
        self.positions[name] = position
        self._invalidate_cache()
        self._load_price_data_async(position.commodity)
        return self

    
    def remove_position(self, name: str) -> 'PortfolioManager':

        if name in self.positions:
            del self.positions[name]
            self._invalidate_cache()
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

    
    def calculate_correlations(self, force_recalculate: bool = False) -> 'PortfolioManager':

        if not force_recalculate and self._cache['correlations'] is not None:
            self.correlation_matrix = self._cache['correlations']
            return self
        
        # Get unique commodities
        commodities = list(set(pos.commodity for pos in self.positions.values()))
        
        if len(commodities) < 2:
            self.correlation_matrix = pd.DataFrame()
            return self
        
        try:
            # Align price data for correlation calculation
            aligned_returns = {}
            min_length = float('inf')
            
            for commodity in commodities:
                if commodity in self.price_data and not self.price_data[commodity].empty:
                    returns = self.price_data[commodity].pct_change().dropna()
                    returns = returns.tail(self.config['correlation_window'])
                    aligned_returns[commodity] = returns
                    min_length = min(min_length, len(returns))
            
            if len(aligned_returns) >= 2 and min_length > 10:
                # Truncate all series to same length
                for commodity in aligned_returns:
                    aligned_returns[commodity] = aligned_returns[commodity].tail(int(min_length))
                
                # Create correlation matrix
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

        if not force_recalculate and self._cache['risk_metrics'] is not None:
            self.portfolio_risk = self._cache['risk_metrics']
            return self
        
        if len(self.positions) == 0:
            return self
        
        try:
            portfolio_pnl = self._simulate_portfolio_pnl()
            
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
            
            self._cache['risk_metrics'] = self.portfolio_risk
            
        except Exception as e:
            print(f"Warning: Could not calculate portfolio risk: {e}")
        
        return self

    
    def get_portfolio_summary(self) -> pd.DataFrame:
        """ Get detailed portfolio summary table """
        if not self.positions:
            return pd.DataFrame()
        
        summary_data = []
        total_notional = sum(pos.notional_value for pos in self.positions.values())
        
        for name, position in self.positions.items():
            weight = (position.notional_value / total_notional * 100) if total_notional > 0 else 0
            
            summary_data.append({
                'Position Name': name,
                'Commodity': position.commodity,
                'Direction': position.direction,
                'Size': f"{position.abs_size:,.0f}",
                'Current Price': f"${position.current_price:.2f}",
                'Notional Value': f"${position.notional_value:,.0f}",
                'Weight': f"{weight:.1f}%",
                'Hedge Ratio': f"{position.hedge_ratio:.1%}",
                'Strategy': position.strategy,
                'Strike Price': f"${position.strike_price:.2f}" if position.strike_price else "N/A"
            })
        
        return pd.DataFrame(summary_data)

    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """ Get portfolio weights by notional value """
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
        """ Get net exposure by commodity """
        if not self.positions:
            return pd.DataFrame()
        
        commodity_exposure = {}
        
        for position in self.positions.values():
            commodity = position.commodity
            exposure = position.size * position.current_price
            
            if commodity in commodity_exposure:
                commodity_exposure[commodity] += exposure
            else:
                commodity_exposure[commodity] = exposure
        
        exposure_data = [
            {
                'Commodity': commodity,
                'Net Exposure': f"${exposure:,.0f}",
                'Direction': 'Long' if exposure > 0 else 'Short' if exposure < 0 else 'Neutral'
            }
            for commodity, exposure in commodity_exposure.items()
        ]
        
        return pd.DataFrame(exposure_data)

    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """ Get correlation matrix (calculate if needed) """
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
                # Create dummy data to prevent errors
                dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
                dummy_prices = pd.Series(
                    np.random.normal(75, 5, 252), 
                    index=dates,
                    name=commodity
                )
                self.price_data[commodity] = dummy_prices

    
    def _simulate_portfolio_pnl(self) -> np.ndarray:
        if not self.positions:
            return np.array([])
        
        portfolio_pnl = np.zeros(self.config['simulation_runs'])
        
        for name, position in self.positions.items():
            if position.commodity not in self.price_data:
                continue
            
            try:
                sim_result = simulate_hedged_vs_unhedged(
                    self.price_data[position.commodity],
                    position.size,
                    position.hedge_ratio,
                    position.strategy,
                    position.strike_price,
                    self.config['simulation_runs']
                )
                
                # Add hedged P&L to portfolio total
                portfolio_pnl += sim_result['hedged_pnl']
                
            except Exception as e:
                print(f"Warning: Could not simulate position {name}: {e}")
                continue
        
        return portfolio_pnl

    
    def _invalidate_cache(self) -> None:
        """ Invalidate all cached calculations """
        self._cache = {key: None for key in self._cache.keys()}
        self._cache['last_update'] = datetime.now()
    
    def _get_portfolio_hash(self) -> str:
        """ Generate hash of current portfolio state """
        position_strings = []
        for name, pos in sorted(self.positions.items()):
            pos_str = f"{name}:{pos.commodity}:{pos.size}:{pos.hedge_ratio}:{pos.strategy}"
            position_strings.append(pos_str)
        
        return hash("|".join(position_strings))
    

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
        return self
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __contains__(self, position_name: str) -> bool:
        """ Check if position exists """
        return position_name in self.positions
    
    def __str__(self) -> str:
        """ String representation of portfolio """
        if not self.positions:
            return "Empty Portfolio"
        
        total_notional = sum(pos.notional_value for pos in self.positions.values())
        return f"Portfolio: {len(self.positions)} positions, ${total_notional:,.0f} notional"
    
    def __repr__(self) -> str:
        """ Detailed representation of portfolio """
        return f"PortfolioManager(positions={len(self.positions)}, total_notional=${sum(pos.notional_value for pos in self.positions.values()):,.0f})"


def create_oil_position(size: float, hedge_ratio: float = 0.0, strategy: str = "Futures", strike_price: Optional[float] = None, option_type: Optional[str] = None) -> Position:
    """ Create WTI Crude Oil position """
    return Position(
        commodity="WTI Crude Oil",
        size=size,
        hedge_ratio=hedge_ratio,
        strategy=strategy,
        strike_price=strike_price,
        option_type=option_type
    )


def create_gas_position(size: float, hedge_ratio: float = 0.0, strategy: str = "Futures", strike_price: Optional[float] = None, option_type: Optional[str] = None) -> Position:
    """ Create Natural Gas position """
    return Position(
        commodity="Natural Gas",
        size=size,
        hedge_ratio=hedge_ratio,
        strategy=strategy,
        strike_price=strike_price,
        option_type=option_type
    )


def create_brent_position(size: float, hedge_ratio: float = 0.0, strategy: str = "Futures", strike_price: Optional[float] = None, option_type: Optional[str] = None) -> Position:
    """ Create Brent Crude Oil position """
    return Position(
        commodity="Brent Crude Oil",
        size=size,
        hedge_ratio=hedge_ratio,
        strategy=strategy,
        strike_price=strike_price,
        option_type=option_type
    )


def create_sample_portfolio() -> PortfolioManager:
    """ Create a sample diversified portfolio for testing """
    return (PortfolioManager()
        .add_position("wti_main", create_oil_position(10000, 0.8))
        .add_position("gas_hedge", create_gas_position(-5000, 0.6))
        .add_position("brent_arb", create_brent_position(3000, 0.7))
        .calculate_correlations()
        .calculate_portfolio_risk())


if __name__ == "__main__":
    print("Creating sample portfolio...")
    
    portfolio = (PortfolioManager()
        .add_position("oil_main", create_oil_position(15000, 0.8))
        .add_position("gas_hedge", create_gas_position(-8000, 0.6))
        .calculate_correlations()
        .calculate_portfolio_risk())
    
    print(f"\n{portfolio}")
    print("\nPortfolio Summary:")
    print(portfolio.get_portfolio_summary().to_string(index=False))
    
    print("\nRisk Summary:")
    risk_summary = portfolio.get_portfolio_risk_summary()
    for metric, value in risk_summary.items():
        print(f"  {metric}: {value}")
    
    print("\nCorrelation Matrix:")
    corr_matrix = portfolio.get_correlation_matrix()
    if not corr_matrix.empty:
        print(corr_matrix.round(3))
    else:
        print("  Insufficient data for correlation calculation")
    
    print("\nCommodity Exposure:")
    print(portfolio.get_commodity_exposure().to_string(index=False))
