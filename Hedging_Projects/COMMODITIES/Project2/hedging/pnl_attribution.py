"""
hedging/pnl_attribution.py

Real-Time P&L Attribution System for Professional Trading
Breaks down daily P&L into component sources (Delta, Gamma, Theta, Vega, etc.)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .data import get_current_price, get_prices
from .portfolio import PortfolioManager, Position

class PnLAttributionEngine:
    """Professional P&L Attribution Engine for Trading Desks"""
    
    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio = portfolio_manager
        self.attribution_cache = {}
        self.last_prices = {}
        self.last_greeks = {}
        
    def calculate_daily_pnl_attribution(self, target_date: datetime = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate detailed P&L attribution for each position and portfolio level.
        
        Returns:
            Dict with position-level and portfolio-level attribution
        """
        if target_date is None:
            target_date = datetime.now()
        
        attribution_results = {
            'portfolio_total': {},
            'positions': {},
            'metadata': {
                'calculation_time': target_date,
                'total_pnl': 0.0,
                'attribution_complete': True
            }
        }
        
        portfolio_attribution = {
            'delta_pnl': 0.0,
            'gamma_pnl': 0.0,
            'theta_pnl': 0.0,
            'vega_pnl': 0.0,
            'rho_pnl': 0.0,
            'carry_pnl': 0.0,
            'other_pnl': 0.0,
            'total_pnl': 0.0
        }
        
        # Calculate attribution for each position
        for position_name, position in self.portfolio.positions.items():
            try:
                pos_attribution = self._calculate_position_attribution(position, target_date)
                attribution_results['positions'][position_name] = pos_attribution
                
                # Aggregate to portfolio level
                for key in portfolio_attribution.keys():
                    if key != 'total_pnl':
                        portfolio_attribution[key] += pos_attribution.get(key, 0.0)
                
                portfolio_attribution['total_pnl'] += pos_attribution.get('total_pnl', 0.0)
                
            except Exception as e:
                st.error(f"Error calculating attribution for {position_name}: {e}")
                continue
        
        attribution_results['portfolio_total'] = portfolio_attribution
        attribution_results['metadata']['total_pnl'] = portfolio_attribution['total_pnl']
        
        return attribution_results
    
    def _calculate_position_attribution(self, position: Position, target_date: datetime) -> Dict[str, float]:
        """Calculate P&L attribution for a single position"""
        
        attribution = {
            'delta_pnl': 0.0,
            'gamma_pnl': 0.0,
            'theta_pnl': 0.0,
            'vega_pnl': 0.0,
            'rho_pnl': 0.0,
            'carry_pnl': 0.0,
            'other_pnl': 0.0,
            'total_pnl': 0.0
        }
        
        try:
            # Get current and previous prices
            current_price = get_current_price(position.commodity)
            previous_price = self._get_previous_price(position.commodity, target_date)
            
            if previous_price is None:
                previous_price = current_price * 0.99  # Fallback: assume 1% move
            
            price_change = current_price - previous_price
            
            # Get current and previous Greeks
            current_greeks = self._get_position_greeks(position, current_price)
            previous_greeks = self._get_position_greeks(position, previous_price)
            
            # Calculate P&L components based on strategy type
            if position.strategy == "Futures":
                attribution = self._calculate_futures_attribution(
                    position, price_change, current_price, previous_price
                )
            elif position.strategy == "Options":
                attribution = self._calculate_options_attribution(
                    position, price_change, current_price, previous_price,
                    current_greeks, previous_greeks
                )
            elif position.is_multi_leg:
                attribution = self._calculate_multi_leg_attribution(
                    position, price_change, current_price, previous_price,
                    current_greeks, previous_greeks
                )
            
            # Add metadata
            attribution['position_name'] = getattr(position, 'name', 'Unknown')
            attribution['commodity'] = position.commodity
            attribution['strategy'] = position.strategy
            attribution['position_size'] = position.size
            attribution['current_price'] = current_price
            attribution['previous_price'] = previous_price
            attribution['price_change'] = price_change
            attribution['price_change_pct'] = (price_change / previous_price * 100) if previous_price != 0 else 0
            
        except Exception as e:
            print(f"Error in position attribution: {e}")
            attribution['error'] = str(e)
        
        return attribution
    
    def _calculate_futures_attribution(self, position: Position, price_change: float, 
                                     current_price: float, previous_price: float) -> Dict[str, float]:
        """Calculate attribution for futures positions"""
        
        # Futures are pure delta plays
        delta_pnl = price_change * position.size * position.hedge_ratio
        
        # Small carry component from financing
        carry_pnl = position.notional_value * 0.0001  # Minimal carry for futures
        
        total_pnl = delta_pnl + carry_pnl
        
        return {
            'delta_pnl': delta_pnl,
            'gamma_pnl': 0.0,  # Futures have no gamma
            'theta_pnl': 0.0,  # No time decay
            'vega_pnl': 0.0,   # No vol sensitivity
            'rho_pnl': 0.0,    # Minimal rate sensitivity
            'carry_pnl': carry_pnl,
            'other_pnl': 0.0,
            'total_pnl': total_pnl
        }
    
    def _calculate_options_attribution(self, position: Position, price_change: float,
                                     current_price: float, previous_price: float,
                                     current_greeks: Dict, previous_greeks: Dict) -> Dict[str, float]:
        """Calculate attribution for single options positions"""
        
        # Get average Greeks for P&L calculation
        avg_delta = (current_greeks.get('delta', 0) + previous_greeks.get('delta', 0)) / 2
        avg_gamma = (current_greeks.get('gamma', 0) + previous_greeks.get('gamma', 0)) / 2
        avg_theta = (current_greeks.get('theta', 0) + previous_greeks.get('theta', 0)) / 2
        avg_vega = (current_greeks.get('vega', 0) + previous_greeks.get('vega', 0)) / 2
        
        # Calculate P&L components
        delta_pnl = avg_delta * price_change
        gamma_pnl = 0.5 * avg_gamma * (price_change ** 2)

        # Add charm effect (gamma vs time)
        if 'charm' in current_greeks:
            charm_pnl = current_greeks.get('charm', 0) * 1.0  # 1-day time effect on gamma
            gamma_pnl += charm_pnl

        # Add vanna effect (delta vs volatility)
        if 'vanna' in current_greeks:
            vol_change = self._estimate_vol_change(price_change, previous_price)
            vanna_pnl = current_greeks.get('vanna', 0) * vol_change
            gamma_pnl += vanna_pnl
            
        theta_pnl = avg_theta * 1.0  # One day of theta decay
        
        # Vega P&L (simplified - would need vol change data)
        # For now, estimate vol change based on price move
        vol_change = self._estimate_vol_change(price_change, previous_price)
        vega_pnl = avg_vega * vol_change
        
        # Rho P&L (minimal for commodities)
        rho_pnl = current_greeks.get('rho', 0) * 0.0001  # Assume minimal rate change
        
        total_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
        
        return {
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'rho_pnl': rho_pnl,
            'carry_pnl': 0.0,
            'other_pnl': 0.0,
            'total_pnl': total_pnl
        }
    
    def _calculate_multi_leg_attribution(self, position: Position, price_change: float,
                                       current_price: float, previous_price: float,
                                       current_greeks: Dict, previous_greeks: Dict) -> Dict[str, float]:
        """Calculate attribution for multi-leg strategies"""
        
        # Similar to options but with more complex Greeks interactions
        avg_delta = (current_greeks.get('delta', 0) + previous_greeks.get('delta', 0)) / 2
        avg_gamma = (current_greeks.get('gamma', 0) + previous_greeks.get('gamma', 0)) / 2
        avg_theta = (current_greeks.get('theta', 0) + previous_greeks.get('theta', 0)) / 2
        avg_vega = (current_greeks.get('vega', 0) + previous_greeks.get('vega', 0)) / 2
        
        # Multi-leg strategies often have enhanced gamma effects
        delta_pnl = avg_delta * price_change
        gamma_pnl = 0.5 * avg_gamma * (price_change ** 2)
        theta_pnl = avg_theta * 1.0  # Daily theta decay
        
        # Vega is often the main driver for multi-leg strategies
        vol_change = self._estimate_vol_change(price_change, previous_price)
        vega_pnl = avg_vega * vol_change
        
        # Cross-gamma effects (simplified)
        cross_gamma_pnl = 0.1 * gamma_pnl  # Approximate cross-gamma effects
        
        total_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + cross_gamma_pnl
        
        return {
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl + cross_gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'rho_pnl': 0.0,
            'carry_pnl': 0.0,
            'other_pnl': 0.0,
            'total_pnl': total_pnl
        }
    
    def _get_position_greeks(self, position: Position, price: float) -> Dict[str, float]:
        """Get Greeks for a position at a specific price"""
        try:
            if position.strategy == "Futures":
                direction = 1.0 if position.size > 0 else -1.0
                delta = direction * abs(position.size) * position.hedge_ratio / 1000  # Normalize
                return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
            
            elif position.strategy == "Options" and hasattr(position, 'strike_price'):
                from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, get_commodity_volatility, time_to_expiration
                
                strike = position.strike_price
                time_to_exp = time_to_expiration(3)  # 3 months default
                risk_free_rate = get_risk_free_rate()
                volatility = get_commodity_volatility(position.commodity)
                option_type = getattr(position, 'option_type', 'put').lower()
                
                greeks = BlackScholesCalculator.calculate_greeks(
                    price, strike, time_to_exp, risk_free_rate, volatility, option_type
                )
                
                multiplier = position.size * position.hedge_ratio
                return {k: v * multiplier for k, v in greeks.items()}
            
            elif position.is_multi_leg:
                from hedging.options_math import MultiLegGreeksCalculator
                return MultiLegGreeksCalculator.calculate_strategy_greeks(
                    position.multi_leg_strategy, price
                )
            
            else:
                return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
                
        except Exception as e:
            print(f"Error calculating Greeks: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    
    def _get_previous_price(self, commodity: str, target_date: datetime) -> Optional[float]:
        """Get previous day's closing price"""
        try:
            # Get recent price history
            prices = get_prices(commodity, period="5d")
            if len(prices) >= 2:
                return float(prices.iloc[-2])  # Previous day's price
            return None
        except Exception:
            return None
    
    def _estimate_vol_change(self, price_change: float, previous_price: float) -> float:
        """Estimate volatility change based on price movement (simplified)"""
        # In practice, you'd use actual implied vol data
        # For now, assume vol increases with large moves
        price_move_pct = abs(price_change / previous_price) if previous_price != 0 else 0
        
        if price_move_pct > 0.05:  # Large move (>5%)
            return 0.02  # Vol increases by 2%
        elif price_move_pct > 0.02:  # Medium move (2-5%)
            return 0.005  # Vol increases by 0.5%
        else:
            return -0.001  # Vol decreases slightly on quiet days
    
    def get_historical_attribution(self, days: int = 30) -> pd.DataFrame:
        """Get historical P&L attribution for analysis"""
        attribution_history = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            try:
                daily_attribution = self.calculate_daily_pnl_attribution(date)
                portfolio_pnl = daily_attribution['portfolio_total']
                
                # FIXED: Create proper dictionary instead of modifying the original
                historical_entry = {
                    'date': date,
                    'delta_pnl': portfolio_pnl.get('delta_pnl', 0),
                    'gamma_pnl': portfolio_pnl.get('gamma_pnl', 0),
                    'theta_pnl': portfolio_pnl.get('theta_pnl', 0),
                    'vega_pnl': portfolio_pnl.get('vega_pnl', 0),
                    'rho_pnl': portfolio_pnl.get('rho_pnl', 0),
                    'total_pnl': portfolio_pnl.get('total_pnl', 0)
                }
                attribution_history.append(historical_entry)
                
            except Exception as e:
                print(f"Error calculating attribution for {date}: {e}")
                # Add placeholder data to avoid empty dataframe
                historical_entry = {
                    'date': date,
                    'delta_pnl': 0,
                    'gamma_pnl': 0,
                    'theta_pnl': 0,
                    'vega_pnl': 0,
                    'rho_pnl': 0,
                    'total_pnl': 0
                }
                attribution_history.append(historical_entry)
                continue
        
        if not attribution_history:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=['date', 'delta_pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl', 'rho_pnl', 'total_pnl'])
        
        return pd.DataFrame(attribution_history)


class PnLAttributionUI:
    """UI Components for P&L Attribution Display"""
    
    @staticmethod
    def render_pnl_attribution_dashboard(attribution_engine: PnLAttributionEngine):
        """Render the main P&L attribution dashboard"""
        
        st.markdown("## 💰 Real-Time P&L Attribution")
        
        # Calculate today's attribution
        with st.spinner("🔄 Calculating P&L attribution..."):
            attribution_results = attribution_engine.calculate_daily_pnl_attribution()
        
        portfolio_pnl = attribution_results['portfolio_total']
        positions_pnl = attribution_results['positions']
        
        # Summary cards
        PnLAttributionUI._render_pnl_summary_cards(portfolio_pnl)
        
        # Main attribution chart
        PnLAttributionUI._render_pnl_attribution_chart(portfolio_pnl)
        
        # Position-level breakdown
        PnLAttributionUI._render_position_breakdown(positions_pnl)
    
    @staticmethod
    def _render_pnl_summary_cards(portfolio_pnl: Dict[str, float]):
        """Render P&L summary cards"""
        
        st.markdown("### 📊 Today's P&L Breakdown")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            delta_pnl = portfolio_pnl.get('delta_pnl', 0)
            delta_color = "#4ECDC4" if delta_pnl >= 0 else "#FF6B6B"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {delta_color};">Delta P&L</div>
                <div class="metric-value-premium">${delta_pnl:,.0f}</div>
                <div class="metric-subtitle-premium">Price Moves</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            gamma_pnl = portfolio_pnl.get('gamma_pnl', 0)
            gamma_color = "#48bb78" if gamma_pnl >= 0 else "#e74c3c"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {gamma_color};">Gamma P&L</div>
                <div class="metric-value-premium">${gamma_pnl:,.0f}</div>
                <div class="metric-subtitle-premium">Convexity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            theta_pnl = portfolio_pnl.get('theta_pnl', 0)
            theta_color = "#FF6B6B" if theta_pnl < 0 else "#48bb78"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {theta_color};">Theta P&L</div>
                <div class="metric-value-premium">${theta_pnl:,.0f}</div>
                <div class="metric-subtitle-premium">Time Decay</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            vega_pnl = portfolio_pnl.get('vega_pnl', 0)
            vega_color = "#667eea" if vega_pnl >= 0 else "#e67e22"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {vega_color};">Vega P&L</div>
                <div class="metric-value-premium">${vega_pnl:,.0f}</div>
                <div class="metric-subtitle-premium">Vol Changes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            total_pnl = portfolio_pnl.get('total_pnl', 0)
            total_color = "#2ECC71" if total_pnl >= 0 else "#E74C3C"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {total_color}; font-weight: bold;">Total P&L</div>
                <div class="metric-value-premium" style="font-size: 2.5rem;">${total_pnl:,.0f}</div>
                <div class="metric-subtitle-premium">Net Result</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def _render_pnl_attribution_chart(portfolio_pnl: Dict[str, float]):
        """Render P&L attribution waterfall chart"""
        st.markdown("---")

        st.markdown("### 📊 P&L Attribution Waterfall")
        
        # Prepare data for waterfall chart
        categories = ['Delta', 'Gamma', 'Theta', 'Vega', 'Other']
        values = [
            portfolio_pnl.get('delta_pnl', 0),
            portfolio_pnl.get('gamma_pnl', 0),
            portfolio_pnl.get('theta_pnl', 0),
            portfolio_pnl.get('vega_pnl', 0),
            portfolio_pnl.get('carry_pnl', 0) + portfolio_pnl.get('rho_pnl', 0) + portfolio_pnl.get('other_pnl', 0)
        ]
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Add bars with colors based on positive/negative
        colors = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in values]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'${v:,.0f}' for v in values],
            textposition='auto',
            name='P&L Components'
        ))
        
        # Add total line
        total_pnl = sum(values)
        fig.add_hline(y=total_pnl, line_dash="dash", line_color="black", 
                     annotation_text=f"Total: ${total_pnl:,.0f}")
        
        fig.update_layout(
            title="Daily P&L Attribution Breakdown",
            xaxis_title="P&L Components",
            yaxis_title="P&L ($)",
            height=400,
            showlegend=False,
            font=dict(family="Inter", size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_position_breakdown(positions_pnl: Dict[str, Dict[str, float]]):
        """Render position-level P&L breakdown"""
        
        st.markdown("### 📋 Position-Level Attribution")
        
        if not positions_pnl:
            st.info("No position data available")
            return
        
        # Create DataFrame for display
        breakdown_data = []
        for pos_name, pos_pnl in positions_pnl.items():
            breakdown_data.append({
                'Position': pos_name,
                'Strategy': pos_pnl.get('strategy', 'Unknown'),
                'Commodity': pos_pnl.get('commodity', 'Unknown'),
                'Delta P&L': f"${pos_pnl.get('delta_pnl', 0):,.0f}",
                'Gamma P&L': f"${pos_pnl.get('gamma_pnl', 0):,.0f}",
                'Theta P&L': f"${pos_pnl.get('theta_pnl', 0):,.0f}",
                'Vega P&L': f"${pos_pnl.get('vega_pnl', 0):,.0f}",
                'Total P&L': f"${pos_pnl.get('total_pnl', 0):,.0f}",
                'Price Change': f"{pos_pnl.get('price_change_pct', 0):+.2f}%"
            })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)


def render_pnl_attribution_tab(portfolio_manager: PortfolioManager):
    """Main function to render P&L attribution tab"""
    
    if len(portfolio_manager.positions) == 0:
        st.info("📊 Add positions to see P&L attribution analysis")
        return
    
    # Initialize attribution engine
    attribution_engine = PnLAttributionEngine(portfolio_manager)
    
    # Render dashboard
    PnLAttributionUI.render_pnl_attribution_dashboard(attribution_engine)