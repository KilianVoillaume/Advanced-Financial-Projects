"""
hedging/greeks_dashboard.py - Enhanced Real-Time Greeks Dashboard
This module provides comprehensive Greeks visualization and monitoring for single options and multi-leg strategies in commodity hedging.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, get_commodity_volatility, time_to_expiration
try:
    from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, get_commodity_volatility, time_to_expiration
except ImportError:
    # Fallback if options_math module isn't available
    def get_risk_free_rate():
        return 0.03
    def get_commodity_volatility(commodity):
        return 0.25
    def time_to_expiration(months):
        return months / 12.0

class GreeksDashboard:
    """
    Enhanced real-time Greeks dashboard supporting both single and multi-leg strategies.
    """
    
    @staticmethod
    def _calculate_position_greeks(position) -> Dict[str, float]:
        try:
            # Multi-leg strategy
            if position.is_multi_leg:
                from hedging.options_math import MultiLegGreeksCalculator
                return MultiLegGreeksCalculator.calculate_strategy_greeks(
                    position.multi_leg_strategy, position.current_price
                )
            
            # Single option strategy
            elif position.strategy == "Options" and hasattr(position, 'strike_price') and position.strike_price:
                current_price = position.current_price
                strike = position.strike_price
                time_to_exp = time_to_expiration(3)
                risk_free_rate = get_risk_free_rate()
                volatility = get_commodity_volatility(position.commodity)
                
                option_type = position.option_type.lower() if hasattr(position, 'option_type') and position.option_type else 'put'
                
                greeks = BlackScholesCalculator.calculate_greeks(
                    current_price, strike, time_to_exp, risk_free_rate, volatility, option_type
                )
                
                position_multiplier = position.size * position.hedge_ratio
                
                return {
                    'delta': greeks['delta'] * position_multiplier,
                    'gamma': greeks['gamma'] * position_multiplier,
                    'theta': greeks['theta'] * position_multiplier,
                    'vega': greeks['vega'] * position_multiplier,
                    'rho': greeks['rho'] * position_multiplier
                }
            
            # FIXED: Futures position - they DO have delta!
            elif position.strategy == "Futures":
                direction = 1.0 if position.size > 0 else -1.0
                delta = direction * abs(position.size / 1000) * position.hedge_ratio
                
                return {
                    'delta': delta,
                    'gamma': 0.0,  # Futures have no gamma
                    'theta': 0.0,  # No time decay
                    'vega': 0.0,   # No volatility sensitivity
                    'rho': 0.0     # Minimal rate sensitivity
                }
            
            # Non-options position
            else:
                return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
                
        except Exception as e:
            print(f"Error calculating Greeks for position: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    
    @staticmethod
    def create_enhanced_greeks_heatmap(portfolio_positions: Dict) -> go.Figure:
        """Create enhanced heatmap showing Greeks across all positions"""
        
        if not portfolio_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Portfolio Greeks Heatmap", height=400)
            return fig
        
        greeks_data = []
        position_names = []
        position_types = []
        
        for name, position in portfolio_positions.items():
            if position.strategy in ["Options", "Multi-Leg"]:
                greeks = GreeksDashboard._calculate_position_greeks(position)
                greeks_data.append([
                    greeks['delta'],
                    greeks['gamma'],
                    greeks['theta'],
                    greeks['vega'],
                    greeks['rho']
                ])
                position_names.append(name)
                
                # Enhanced position type labeling
                if position.is_multi_leg:
                    strategy_type = position.multi_leg_strategy.strategy_type.value
                    position_types.append(f"{strategy_type}")
                else:
                    position_types.append(f"Single {position.option_type.title()}")
        
        if not greeks_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Portfolio Greeks Heatmap", height=400)
            return fig
        
        greeks_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
        greeks_array = np.array(greeks_data)
        
        # Enhanced normalization with better scaling
        normalized_data = np.zeros_like(greeks_array)
        for i in range(greeks_array.shape[1]):
            col = greeks_array[:, i]
            if np.std(col) > 1e-6:  # Avoid division by zero
                normalized_data[:, i] = (col - np.mean(col)) / (np.std(col) + 1e-6)
            else:
                normalized_data[:, i] = col
        
        # Create custom hover text
        hover_text = []
        for i, row in enumerate(greeks_array):
            row_text = []
            for j, val in enumerate(row):
                row_text.append(f"{greeks_names[j]}: {val:.4f}<br>Type: {position_types[i]}")
            hover_text.append(row_text)
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized_data,
            x=greeks_names,
            y=position_names,
            colorscale='RdBu',
            zmid=0,
            text=greeks_array,
            texttemplate='%{text:.3f}',
            textfont={"size": 11, "color": "white", "family": "Inter"},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>%{text}<extra></extra>',
            colorbar=dict(
                title="Normalized Greeks",
                title_font=dict(size=12),
                tickfont=dict(size=10)
            )
        ))
        
        fig.update_layout(
            title={
                'text': "Portfolio Greeks Heatmap",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Inter'}
            },
            xaxis_title="Greeks",
            yaxis_title="Positions",
            height=400,
            font=dict(family="Inter", size=12),
            margin=dict(l=100, r=100, t=80, b=80)
        )
        
        return fig
    
    @staticmethod
    def create_multi_leg_breakdown_chart(portfolio_positions: Dict) -> go.Figure:
        """Create detailed breakdown chart for multi-leg strategies"""
        
        multi_leg_positions = {name: pos for name, pos in portfolio_positions.items() 
                             if pos.is_multi_leg}
        
        if not multi_leg_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No multi-leg strategies found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Multi-Leg Strategy Breakdown", height=400)
            return fig
        
        # Create subplots for each multi-leg strategy
        n_strategies = len(multi_leg_positions)
        fig = make_subplots(
            rows=min(n_strategies, 3), 
            cols=1,
            subplot_titles=[f"{name}: {pos.strategy_description}" 
                          for name, pos in list(multi_leg_positions.items())[:3]],
            vertical_spacing=0.15
        )
        
        colors = ['#667eea', '#764ba2', '#4ECDC4', '#FF6B6B', '#48bb78']
        
        for idx, (name, position) in enumerate(list(multi_leg_positions.items())[:3]):
            try:
                from hedging.options_math import MultiLegGreeksCalculator
                
                detailed_greeks = MultiLegGreeksCalculator.calculate_detailed_greeks(
                    position.multi_leg_strategy, position.current_price
                )
                
                leg_details = detailed_greeks['leg_details']
                
                # Extract data for visualization
                leg_names = [f"Leg {leg['leg_number']}: {leg['position']} {leg['option_type']}" 
                           for leg in leg_details]
                deltas = [leg['delta'] for leg in leg_details]
                
                fig.add_trace(
                    go.Bar(
                        x=leg_names,
                        y=deltas,
                        name=f"{name} Delta",
                        marker_color=colors[idx % len(colors)],
                        text=[f'{delta:.3f}' for delta in deltas],
                        textposition='auto',
                        showlegend=True
                    ),
                    row=idx + 1, col=1
                )
                
            except Exception as e:
                st.error(f"Error creating breakdown for {name}: {e}")
                continue
        
        fig.update_layout(
            title={
                'text': "Multi-Leg Strategy Greeks Breakdown",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            height=min(600, 200 * n_strategies),
            font=dict(family="Inter", size=10),
            showlegend=True
        )
        
        # Add horizontal reference lines
        for i in range(min(n_strategies, 3)):
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i + 1, col=1)
        
        return fig
    
    @staticmethod
    def create_enhanced_delta_exposure_chart(portfolio_positions: Dict) -> go.Figure:
        """Create enhanced delta exposure visualization with multi-leg support"""
        
        if not portfolio_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Delta Exposure Analysis", height=400)
            return fig
        
        delta_by_commodity = {}
        position_data = []
        
        for name, position in portfolio_positions.items():
            if position.strategy in ["Options", "Multi-Leg"]:
                greeks = GreeksDashboard._calculate_position_greeks(position)
                delta = greeks['delta']
                
                commodity = position.commodity
                if commodity not in delta_by_commodity:
                    delta_by_commodity[commodity] = 0
                delta_by_commodity[commodity] += delta
                
                # Enhanced position information
                strategy_info = ""
                if position.is_multi_leg:
                    strategy_info = f"{position.multi_leg_strategy.strategy_type.value}"
                    legs_count = len(position.multi_leg_strategy.legs)
                    strategy_info += f" ({legs_count} legs)"
                else:
                    strategy_info = f"Single {position.option_type.title()}"
                
                position_data.append({
                    'name': name,
                    'delta': delta,
                    'strategy': strategy_info,
                    'commodity': commodity,
                    'hedge_ratio': position.hedge_ratio
                })
        
        if not position_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Delta Exposure Analysis", height=400)
            return fig
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Delta by Position", "Delta by Commodity"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Chart 1: Delta by position with enhanced information
        position_names = [pos['name'] for pos in position_data]
        position_deltas = [pos['delta'] for pos in position_data]
        colors = ['#FF6B6B' if delta < 0 else '#4ECDC4' for delta in position_deltas]
        
        hover_text = [f"Strategy: {pos['strategy']}<br>Hedge Ratio: {pos['hedge_ratio']:.1%}<br>Delta: {pos['delta']:.3f}" 
                     for pos in position_data]
        
        fig.add_trace(
            go.Bar(
                x=position_names,
                y=position_deltas,
                name="Position Delta",
                marker_color=colors,
                text=[f'{delta:.3f}' for delta in position_deltas],
                textposition='auto',
                showlegend=False,
                hovertext=hover_text,
                hovertemplate='<b>%{x}</b><br>%{hovertext}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Chart 2: Delta by commodity
        if delta_by_commodity:
            commodities = list(delta_by_commodity.keys())
            commodity_deltas = list(delta_by_commodity.values())
            commodity_colors = ['#FF6B6B' if delta < 0 else '#48bb78' for delta in commodity_deltas]
            
            fig.add_trace(
                go.Bar(
                    x=commodities,
                    y=commodity_deltas,
                    name="Commodity Delta",
                    marker_color=commodity_colors,
                    text=[f'{delta:.3f}' for delta in commodity_deltas],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title={
                'text': "Enhanced Delta Exposure Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            height=400,
            showlegend=False,
            font=dict(family="Inter", size=10)
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
        
        return fig
    
    @staticmethod
    def create_time_decay_evolution_chart(portfolio_positions: Dict, days_ahead: int = 30) -> go.Figure:
        """Create enhanced time decay chart showing Greeks evolution"""
        
        options_positions = {name: pos for name, pos in portfolio_positions.items() 
                           if pos.strategy in ["Options", "Multi-Leg"]}
        
        if not options_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Time Decay Evolution", height=400)
            return fig
        
        days = np.arange(0, days_ahead + 1)
        
        # Track evolution for each position
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Portfolio Delta Evolution", "Portfolio Gamma Evolution", 
                          "Portfolio Theta Evolution", "Portfolio Vega Evolution"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Calculate evolution for portfolio
        portfolio_evolution = {
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': []
        }
        
        for day in days:
            daily_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
            
            for position in options_positions.values():
                try:
                    if position.is_multi_leg:
                        from hedging.options_math import MultiLegGreeksCalculator
                        
                        # Calculate time decay for multi-leg strategy
                        decay_profile = MultiLegGreeksCalculator.calculate_time_decay_profile(
                            position.multi_leg_strategy, position.current_price, days_ahead
                        )
                        
                        if day < len(decay_profile['time_profile']):
                            day_data = decay_profile['time_profile'][day]
                            for greek in daily_greeks:
                                daily_greeks[greek] += day_data.get(greek, 0)
                    
                    else:
                        # Single option calculation
                        current_price = position.current_price
                        strike = position.strike_price
                        original_time = time_to_expiration(3)
                        remaining_time = max(original_time - (day / 365.0), 0.01)
                        risk_free_rate = get_risk_free_rate()
                        volatility = get_commodity_volatility(position.commodity)
                        
                        option_type = position.option_type.lower() if position.option_type else 'put'
                        
                        greeks = BlackScholesCalculator.calculate_greeks(
                            current_price, strike, remaining_time, risk_free_rate, volatility, option_type
                        )
                        
                        multiplier = position.size * position.hedge_ratio
                        for greek in daily_greeks:
                            daily_greeks[greek] += greeks[greek] * multiplier
                
                except Exception as e:
                    continue
            
            for greek in portfolio_evolution:
                portfolio_evolution[greek].append(daily_greeks[greek])
        
        # Plot evolution
        colors = {'delta': '#4ECDC4', 'gamma': '#48bb78', 'theta': '#FF6B6B', 'vega': '#667eea'}
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, (greek, values) in enumerate(portfolio_evolution.items()):
            row, col = positions[idx]
            
            fig.add_trace(
                go.Scatter(
                    x=days, 
                    y=values, 
                    name=f"Portfolio {greek.title()}", 
                    line=dict(color=colors[greek], width=3),
                    mode='lines+markers',
                    marker=dict(size=4)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title={
                'text': "Greeks Evolution Over Time",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            height=600,
            showlegend=False,
            font=dict(family="Inter", size=10)
        )
        
        # Add reference lines
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=row, col=col)
        
        fig.update_xaxes(title_text="Days", row=2, col=1)
        fig.update_xaxes(title_text="Days", row=2, col=2)
        
        return fig
    
    @staticmethod
    def render_enhanced_greeks_summary_cards(portfolio_positions: Dict):
        """Enhanced Greeks display with separate portfolio and normalized cards"""
        
        net_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        
        options_count = 0
        multi_leg_count = 0
        single_options_count = 0
        total_notional = 0.0
        
        # Calculate portfolio Greeks and totals
        for position in portfolio_positions.values():
            if position.strategy in ["Options", "Multi-Leg", "Futures"]:
                options_count += 1
                
                if position.is_multi_leg:
                    multi_leg_count += 1
                elif position.strategy == "Options":
                    single_options_count += 1
                
                greeks = GreeksDashboard._calculate_position_greeks(position)
                for greek_name, greek_value in greeks.items():
                    net_greeks[greek_name] += greek_value
                
                total_notional += position.notional_value
        
        if options_count == 0:
            st.info("📊 No positions found. Add positions to see Greeks analysis.")
            return
        
        normalized_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

        if total_notional > 0:
            for name, position in portfolio_positions.items():
                if position.strategy in ["Options", "Multi-Leg", "Futures"]:
                    position_weight = position.notional_value / total_notional
                    
                    if position.strategy == "Futures":
                        # Futures: normalized delta = ±1 * hedge_ratio
                        position_normalized_delta = (1.0 if position.size > 0 else -1.0) * position.hedge_ratio
                        
                    elif position.strategy == "Options" and hasattr(position, 'strike_price') and position.strike_price:
                        # Options: estimate delta based on moneyness
                        current_price = position.current_price
                        strike_price = position.strike_price
                        moneyness = current_price / strike_price
                        
                        if hasattr(position, 'option_type') and position.option_type:
                            if position.option_type.lower() == 'call':
                                estimated_delta = max(0.0, min(1.0, (moneyness - 0.8) * 2.5))
                            else:  # put
                                estimated_delta = min(0.0, max(-1.0, (0.8 - moneyness) * 2.5))
                        else:
                            estimated_delta = 0.5  # Default ATM
                        
                        position_normalized_delta = estimated_delta * position.hedge_ratio
                        
                    elif position.is_multi_leg:
                        # Multi-leg: approximate from strategy type
                        try:
                            strategy_type = position.multi_leg_strategy.strategy_type.value
                            if "Straddle" in strategy_type:
                                position_normalized_delta = 0.0  # Delta neutral
                            elif "Collar" in strategy_type:
                                position_normalized_delta = 0.3 * position.hedge_ratio
                            else:
                                position_normalized_delta = 0.1 * position.hedge_ratio
                        except:
                            position_normalized_delta = 0.1 * position.hedge_ratio
                    
                    else:
                        position_normalized_delta = 0.0
                    
                    normalized_greeks['delta'] += position_normalized_delta * position_weight
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; font-size: 1.3rem;">Portfolio Greeks Analysis</h3>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                        {options_count} positions • ${total_notional:,.0f} notional • {multi_leg_count} multi-leg • {single_options_count} single options
                    </p>
                </div>
                <div style="font-size: 2rem;">📊</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # PORTFOLIO GREEKS (First Row)
        st.markdown("### 📊 Portfolio Greeks (Dollar Impact)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            delta_color = "#4ECDC4" if net_greeks['delta'] >= 0 else "#FF6B6B"
            delta_status = "Long Bias" if net_greeks['delta'] > 10 else "Short Bias" if net_greeks['delta'] < -10 else "Neutral"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {delta_color};">Portfolio Delta</div>
                <div class="metric-value-premium">{net_greeks['delta']:.1f}</div>
                <div class="metric-subtitle-premium">{delta_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            gamma_color = "#48bb78" if net_greeks['gamma'] >= 0 else "#e74c3c"
            gamma_status = "High Convexity" if abs(net_greeks['gamma']) > 50 else "Low Convexity"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {gamma_color};">Portfolio Gamma</div>
                <div class="metric-value-premium">{net_greeks['gamma']:.1f}</div>
                <div class="metric-subtitle-premium">{gamma_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            theta_color = "#FF6B6B" if net_greeks['theta'] < 0 else "#48bb78"
            theta_status = "High Decay" if net_greeks['theta'] < -100 else "Moderate Decay" if net_greeks['theta'] < -20 else "Low Decay"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {theta_color};">Portfolio Theta</div>
                <div class="metric-value-premium">${net_greeks['theta']:.0f}</div>
                <div class="metric-subtitle-premium">{theta_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            vega_color = "#667eea"
            vega_status = "High Vol Risk" if abs(net_greeks['vega']) > 500 else "Moderate Vol Risk" if abs(net_greeks['vega']) > 100 else "Low Vol Risk"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {vega_color};">Portfolio Vega</div>
                <div class="metric-value-premium">${net_greeks['vega']:.0f}</div>
                <div class="metric-subtitle-premium">{vega_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            rho_color = "#8e44ad"
            rho_status = "High Rate Risk" if abs(net_greeks['rho']) > 100 else "Low Rate Risk"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {rho_color};">Portfolio Rho</div>
                <div class="metric-value-premium">${net_greeks['rho']:.0f}</div>
                <div class="metric-subtitle-premium">{rho_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚖️ Normalized Greeks (Per $1M Exposure)")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            norm_delta_color = "#4ECDC4" if normalized_greeks['delta'] >= 0 else "#FF6B6B"
            norm_delta_status = "Long Bias" if normalized_greeks['delta'] > 0.1 else "Short Bias" if normalized_greeks['delta'] < -0.1 else "Neutral"
            
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {norm_delta_color};">Normalized Delta</div>
                <div class="metric-value-premium">{normalized_greeks['delta']:.3f}</div>
                <div class="metric-subtitle-premium">{norm_delta_status}</div>
            </div>
            """, unsafe_allow_html=True)

        if total_notional > 0:
            # Normalize to $1M exposure (industry standard)
            normalization_factor = 1000000 / total_notional
            
            # These show: "What would my Greeks be if portfolio was exactly $1M?"
            norm_gamma = net_greeks['gamma'] * normalization_factor
            norm_theta = net_greeks['theta'] * normalization_factor
            norm_vega = net_greeks['vega'] * normalization_factor
            norm_rho = net_greeks['rho'] * normalization_factor
        else:
            norm_gamma = norm_theta = norm_vega = norm_rho = 0.0

        with col2:
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: #48bb78;">Normalized Gamma</div>
                <div class="metric-value-premium">{norm_gamma:.4f}</div>
                <div class="metric-subtitle-premium">Per $1M Exposure</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: #FF6B6B;">Normalized Theta</div>
                <div class="metric-value-premium">{norm_theta:.4f}</div>
                <div class="metric-subtitle-premium">Per $1M Exposure</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: #667eea;">Normalized Vega</div>
                <div class="metric-value-premium">{norm_vega:.4f}</div>
                <div class="metric-subtitle-premium">Per $1M Exposure</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: #8e44ad;">Normalized Rho</div>
                <div class="metric-value-premium">{norm_rho:.4f}</div>
                <div class="metric-subtitle-premium">Per $1M Exposure</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📖 Greeks Explanation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **📊 Portfolio Values (Top Row):**
            - **Delta {net_greeks['delta']:.1f}**: Portfolio gains $ ${abs(net_greeks['delta']):.1f} per $1 price increase
            - **Gamma {net_greeks['gamma']:.1f}**: Delta changes by {net_greeks['gamma']:.1f} per $1 price move  
            - **Theta {net_greeks['theta']:.0f}**: Portfolio loses $ ${abs(net_greeks['theta']):.0f} per day from time decay
            - **Vega {net_greeks['vega']:.0f}**: Portfolio gains $ ${net_greeks['vega']:.0f} per 1% vol increase
            """)
        
        with col2:
            st.markdown(f"""
            **⚖️ Normalized Values (Bottom Row):**
            - **Delta {normalized_greeks['delta']:.3f}**: Position-weighted average delta (-1 to +1)
            - **Range**: Delta between -1.0 and +1.0 (like individual options)
            - **Interpretation**: How directional your overall portfolio is
            - **Usage**: Compare risk across different portfolio sizes
            """)
        
        st.markdown("---")
        
        st.markdown("### 🚨 Risk Interpretation")
        
        risk_alerts = []
        
        if abs(net_greeks['delta']) > 100:
            risk_alerts.append(f"🔴 **High Delta Risk**: {net_greeks['delta']:.1f} - Portfolio is highly directional")
        elif abs(net_greeks['delta']) > 50:
            risk_alerts.append(f"🟡 **Moderate Delta Risk**: {net_greeks['delta']:.1f} - Some directional exposure")
        else:
            risk_alerts.append(f"🟢 **Low Delta Risk**: {net_greeks['delta']:.1f} - Relatively delta neutral")
        
        if net_greeks['theta'] < -100:
            risk_alerts.append(f"🔴 **High Time Decay**: ${net_greeks['theta']:.0f}/day - Significant theta risk")
        elif net_greeks['theta'] < -20:
            risk_alerts.append(f"🟡 **Moderate Time Decay**: ${net_greeks['theta']:.0f}/day - Monitor closely")
        else:
            risk_alerts.append(f"🟢 **Low Time Decay**: ${net_greeks['theta']:.0f}/day - Minimal theta risk")
        
        if abs(net_greeks['vega']) > 500:
            risk_alerts.append(f"🔴 **High Volatility Risk**: ${net_greeks['vega']:.0f} - Very vol sensitive")
        elif abs(net_greeks['vega']) > 100:
            risk_alerts.append(f"🟡 **Moderate Volatility Risk**: ${net_greeks['vega']:.0f} - Some vol exposure")
        else:
            risk_alerts.append(f"🟢 **Low Volatility Risk**: ${net_greeks['vega']:.0f} - Limited vol impact")
        
        for alert in risk_alerts:
            st.markdown(alert)


    @staticmethod
    def render_detailed_greeks_breakdown(portfolio_positions: Dict):
        """Show detailed Greeks breakdown by position"""
        
        st.markdown("### 📋 Detailed Greeks Breakdown")
        
        breakdown_data = []
        
        for name, position in portfolio_positions.items():
            if position.strategy in ["Options", "Multi-Leg", "Futures"]:
                greeks = GreeksDashboard._calculate_position_greeks(position)
                
                # Calculate normalized Greeks for this position
                pos_notional = position.notional_value
                
                breakdown_data.append({
                    'Position': name,
                    'Strategy': position.strategy_description if position.is_multi_leg else position.strategy,
                    'Notional': f"${pos_notional:,.0f}",
                    'Delta': f"{greeks['delta']:.1f}",
                    'Gamma': f"{greeks['gamma']:.1f}",
                    'Theta': f"${greeks['theta']:.0f}",
                    'Vega': f"${greeks['vega']:.0f}",
                    'Rho': f"${greeks['rho']:.0f}"
                })
        
        if breakdown_data:
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        else:
            st.info("No positions with Greeks found")
    
    @staticmethod
    def render_greeks_summary_stats(portfolio_positions: Dict):
        """Show Greeks summary statistics"""
        
        st.markdown("### 📊 Greeks Summary Statistics")
        
        all_greeks = {'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}
        
        for position in portfolio_positions.values():
            if position.strategy in ["Options", "Multi-Leg", "Futures"]:
                greeks = GreeksDashboard._calculate_position_greeks(position)
                
                for greek_name, greek_value in greeks.items():
                    all_greeks[greek_name].append(greek_value)
        
        if any(all_greeks.values()):
            stats_data = []
            
            for greek_name, values in all_greeks.items():
                if values:
                    stats_data.append({
                        'Greek': greek_name.title(),
                        'Min': f"{min(values):.2f}",
                        'Max': f"{max(values):.2f}",
                        'Mean': f"{np.mean(values):.2f}",
                        'Std Dev': f"{np.std(values):.2f}",
                        'Sum': f"{sum(values):.2f}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No Greeks data available for statistics")


class GreeksMonitor:
    """Enhanced Greeks monitoring with multi-leg strategy support"""
    
    @staticmethod
    def check_enhanced_risk_limits(portfolio_positions: Dict) -> Dict[str, List[str]]:
        """Enhanced risk checking supporting multi-leg strategies"""
        alerts = {'critical': [], 'warning': [], 'info': []}
        
        net_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        total_notional = 0.0
        
        for position in portfolio_positions.values():
            if position.strategy in ["Options", "Multi-Leg"]:
                greeks = GreeksDashboard._calculate_position_greeks(position)
                for greek_name, greek_value in greeks.items():
                    net_greeks[greek_name] += greek_value
                
                total_notional += position.notional_value
        
        # Dynamic limits based on portfolio size
        scale_factor = max(total_notional / 100000.0, 1.0)  # Scale to $100k base
        
        limits = {
            'delta': {'critical': 2.0 * scale_factor, 'warning': 1.0 * scale_factor},
            'gamma': {'critical': 0.2 * scale_factor, 'warning': 0.1 * scale_factor},
            'theta': {'critical': -200 * scale_factor, 'warning': -100 * scale_factor},
            'vega': {'critical': 1000 * scale_factor, 'warning': 500 * scale_factor}
        }
        
        # Check limits and generate alerts
        for greek_name, greek_value in net_greeks.items():
            if greek_name in limits:
                limit = limits[greek_name]
                
                if greek_name == 'theta':  # Negative values for theta
                    if greek_value < limit['critical']:
                        alerts['critical'].append(f"Excessive time decay: ${greek_value:.2f}/day")
                    elif greek_value < limit['warning']:
                        alerts['warning'].append(f"High time decay: ${greek_value:.2f}/day")
                else:  # Positive values for other Greeks
                    if abs(greek_value) > limit['critical']:
                        alerts['critical'].append(f"High {greek_name} exposure: {greek_value:.3f}")
                    elif abs(greek_value) > limit['warning']:
                        alerts['warning'].append(f"Moderate {greek_name} exposure: {greek_value:.3f}")
        
        # Multi-leg specific alerts
        multi_leg_positions = [pos for pos in portfolio_positions.values() if pos.is_multi_leg]
        if len(multi_leg_positions) > 3:
            alerts['warning'].append(f"Complex portfolio: {len(multi_leg_positions)} multi-leg strategies")
        
        # Portfolio concentration alerts
        if len(portfolio_positions) > 0:
            options_ratio = len([pos for pos in portfolio_positions.values() 
                               if pos.strategy in ["Options", "Multi-Leg"]]) / len(portfolio_positions)
            if options_ratio > 0.8:
                alerts['warning'].append(f"High options concentration: {options_ratio:.1%}")
        
        if len(alerts['critical']) == 0 and len(alerts['warning']) == 0:
            alerts['info'].append("All Greeks within acceptable risk limits")
        
        return alerts


def render_enhanced_greeks_tab(portfolio, analysis_ready):
    """Enhanced Greeks monitoring tab supporting multi-leg strategies"""
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to see Greeks")
        return
    
    if len(portfolio.positions) == 0:
        st.info("📊 Add positions to start Greeks monitoring")
        return
    
    # Check if any options/multi-leg positions exist
    options_positions = {name: pos for name, pos in portfolio.positions.items() 
                        if pos.strategy in ["Options", "Multi-Leg"]}
    
    if not options_positions:
        st.info("📊 Add options or multi-leg strategies to see Greeks analysis")
        return
            
    GreeksDashboard.render_enhanced_greeks_summary_cards(portfolio.positions)
    
    st.markdown("---")
    st.markdown("### 📊 Comprehensive Greeks Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Current Greeks", 
        "🔍 Multi-Leg Breakdown", 
        "⏰ Time Evolution", 
        "🎯 Advanced Analytics"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            delta_chart = GreeksDashboard.create_enhanced_delta_exposure_chart(portfolio.positions)
            st.plotly_chart(delta_chart, use_container_width=True)
        
        with col2:
            heatmap = GreeksDashboard.create_enhanced_greeks_heatmap(portfolio.positions)
            st.plotly_chart(heatmap, use_container_width=True)
        
        # Strategy comparison table
        st.markdown("### 📋 Strategy Comparison")
        render_strategy_comparison_table(portfolio.positions)
    
    with tab2:
        # Multi-leg strategy breakdown
        breakdown_chart = GreeksDashboard.create_multi_leg_breakdown_chart(portfolio.positions)
        st.plotly_chart(breakdown_chart, use_container_width=True)
        
        # Detailed multi-leg analysis
        render_detailed_multi_leg_analysis(portfolio.positions)
    
    with tab3:
        # Time evolution analysis
        evolution_chart = GreeksDashboard.create_time_decay_evolution_chart(portfolio.positions)
        st.plotly_chart(evolution_chart, use_container_width=True)
        
        # Time decay insights
        render_time_decay_insights(portfolio.positions)
    
    with tab4:
        # Advanced analytics
        render_advanced_greeks_analytics(portfolio.positions)


def render_strategy_comparison_table(portfolio_positions: Dict):
    """Render enhanced strategy comparison table"""
    
    comparison_data = []
    
    for name, position in portfolio_positions.items():
        if position.strategy in ["Options", "Multi-Leg"]:
            greeks = GreeksDashboard._calculate_position_greeks(position)
            
            if position.is_multi_leg:
                strategy_desc = position.multi_leg_strategy.strategy_type.value
                legs_info = f"{len(position.multi_leg_strategy.legs)} legs"
                complexity = "High" if len(position.multi_leg_strategy.legs) > 2 else "Medium"
            else:
                strategy_desc = f"Single {position.option_type.title()}"
                legs_info = "1 leg"
                complexity = "Low"
            
            comparison_data.append({
                'Position': name,
                'Strategy': strategy_desc,
                'Legs': legs_info,
                'Complexity': complexity,
                'Delta': f"{greeks['delta']:.3f}",
                'Gamma': f"{greeks['gamma']:.4f}",
                'Theta': f"${greeks['theta']:.2f}",
                'Vega': f"${greeks['vega']:.2f}",
                'Hedge Ratio': f"{position.hedge_ratio:.1%}"
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            height=min(400, len(comparison_data) * 40 + 100)
        )
    else:
        st.info("No options positions to compare")


def render_detailed_multi_leg_analysis(portfolio_positions: Dict):
    """Render detailed analysis for multi-leg strategies"""
    
    multi_leg_positions = {name: pos for name, pos in portfolio_positions.items() 
                          if pos.is_multi_leg}
    
    if not multi_leg_positions:
        st.info("No multi-leg strategies found for detailed analysis")
        return
    
    st.markdown("### 🔍 Multi-Leg Strategy Details")
    
    for name, position in multi_leg_positions.items():
        with st.expander(f"📊 {name}: {position.strategy_description}", expanded=False):
            try:
                from hedging.options_math import MultiLegGreeksCalculator
                
                detailed_greeks = MultiLegGreeksCalculator.calculate_detailed_greeks(
                    position.multi_leg_strategy, position.current_price
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Strategy Information:**")
                    st.write(f"• Current Price: ${position.current_price:.2f}")
                    st.write(f"• Underlying Size: {position.size:,.0f}")
                    st.write(f"• Hedge Ratio: {position.hedge_ratio:.1%}")
                    st.write(f"• Total Legs: {detailed_greeks['total_legs']}")
                
                with col2:
                    st.markdown("**Net Strategy Greeks:**")
                    net_greeks = detailed_greeks['net_strategy_greeks']
                    st.write(f"• Delta: {net_greeks['delta']:.4f}")
                    st.write(f"• Gamma: {net_greeks['gamma']:.6f}")
                    st.write(f"• Theta: ${net_greeks['theta']:.2f}")
                    st.write(f"• Vega: ${net_greeks['vega']:.2f}")
                
                # Leg breakdown table
                st.markdown("**Individual Legs:**")
                leg_df = pd.DataFrame(detailed_greeks['leg_details'])
                leg_display_df = leg_df[['option_type', 'strike_price', 'position', 'delta', 'gamma', 'theta']].copy()
                leg_display_df.columns = ['Type', 'Strike', 'Position', 'Delta', 'Gamma', 'Theta']
                
                st.dataframe(leg_display_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error analyzing {name}: {e}")


def render_time_decay_insights(portfolio_positions: Dict):
    """Render time decay insights and recommendations"""
    
    st.markdown("### ⏰ Time Decay Insights")
    
    total_theta = 0.0
    theta_positions = []
    
    for name, position in portfolio_positions.items():
        if position.strategy in ["Options", "Multi-Leg"]:
            greeks = GreeksDashboard._calculate_position_greeks(position)
            theta = greeks['theta']
            total_theta += theta
            
            if abs(theta) > 1.0:  # Only show significant theta positions
                theta_positions.append({
                    'position': name,
                    'theta': theta,
                    'strategy': position.strategy_description if position.is_multi_leg else f"Single {position.option_type}"
                })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Time Decay Summary:**")
        st.metric("Daily Theta", f"${total_theta:.2f}")
        st.metric("Weekly Theta", f"${total_theta * 7:.2f}")
        st.metric("Monthly Theta", f"${total_theta * 30:.2f}")
        
        if total_theta < -50:
            st.warning("⚠️ High time decay - consider position management")
        elif total_theta < -10:
            st.info("ℹ️ Moderate time decay - monitor closely")
        else:
            st.success("✅ Low time decay risk")
    
    with col2:
        if theta_positions:
            st.markdown("**🔍 High Theta Positions:**")
            for pos in sorted(theta_positions, key=lambda x: abs(x['theta']), reverse=True)[:5]:
                color = "🔴" if pos['theta'] < -25 else "🟡" if pos['theta'] < -10 else "🟢"
                st.write(f"{color} {pos['position']}: ${pos['theta']:.2f}/day ({pos['strategy']})")


def render_advanced_greeks_analytics(portfolio_positions: Dict):
    """Render advanced Greeks analytics and insights"""
    
    st.markdown("### 🎯 Advanced Greeks Analytics")
    
    # Portfolio Greeks VaR analysis
    try:
        from hedging.options_math import GreeksRiskAnalyzer
        
        # Calculate net portfolio Greeks
        net_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        total_notional = 0.0
        
        for position in portfolio_positions.values():
            if position.strategy in ["Options", "Multi-Leg"]:
                greeks = GreeksDashboard._calculate_position_greeks(position)
                for greek_name, greek_value in greeks.items():
                    net_greeks[greek_name] += greek_value
                total_notional += position.notional_value
        
        if total_notional > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Risk Assessment:**")
                risk_assessment = GreeksRiskAnalyzer.assess_greek_risks(net_greeks, total_notional)
                
                for greek, risk_level in risk_assessment.items():
                    color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                    greek_name = greek.replace('_risk', '').title()
                    st.write(f"{color} {greek_name} Risk: {risk_level}")
            
            with col2:
                st.markdown("**📊 Greeks VaR Analysis:**")
                var_analysis = GreeksRiskAnalyzer.calculate_portfolio_var_greeks(
                    net_greeks, 0.35, 0.1, 0.95
                )
                
                st.metric("Delta VaR", f"${var_analysis['delta_var']:.2f}")
                st.metric("Gamma VaR", f"${var_analysis['gamma_var']:.2f}")
                st.metric("Vega VaR", f"${var_analysis['vega_var']:.2f}")
                st.metric("Total VaR", f"${var_analysis['total_var']:.2f}")
            
            # Generate specific alerts
            st.markdown("**🚨 Risk Alerts:**")
            alerts = GreeksRiskAnalyzer.generate_risk_alerts(net_greeks, total_notional)
            
            if alerts:
                for alert in alerts:
                    alert_color = "danger" if alert['type'] == 'critical' else "warning"
                    st.error(f"**{alert['greek']}**: {alert['message']}")
                    st.info(f"💡 **Recommendation**: {alert['recommendation']}")
            else:
                st.success("✅ No risk alerts generated - portfolio within acceptable limits")
        
        # Strategy efficiency analysis
        st.markdown("### 📈 Strategy Efficiency Analysis")
        
        multi_leg_count = len([pos for pos in portfolio_positions.values() if pos.is_multi_leg])
        single_options_count = len([pos for pos in portfolio_positions.values() 
                                  if pos.strategy == "Options" and not pos.is_multi_leg])
        
        if multi_leg_count > 0 or single_options_count > 0:
            efficiency_data = []
            
            for name, position in portfolio_positions.items():
                if position.strategy in ["Options", "Multi-Leg"]:
                    greeks = GreeksDashboard._calculate_position_greeks(position)
                    
                    # Calculate efficiency metrics
                    abs_delta = abs(greeks['delta'])
                    abs_theta = abs(greeks['theta'])
                    abs_vega = abs(greeks['vega'])
                    
                    # Simple efficiency score (this could be enhanced)
                    if position.is_multi_leg:
                        complexity_penalty = len(position.multi_leg_strategy.legs) * 0.1
                    else:
                        complexity_penalty = 0
                    
                    efficiency_score = max(0, 10 - abs_delta - abs_theta/10 - abs_vega/50 - complexity_penalty)
                    
                    efficiency_data.append({
                        'Position': name,
                        'Strategy Type': 'Multi-Leg' if position.is_multi_leg else 'Single Option',
                        'Efficiency Score': f"{efficiency_score:.1f}/10",
                        'Delta Risk': f"{abs_delta:.3f}",
                        'Theta Decay': f"${abs_theta:.2f}",
                        'Vega Risk': f"${abs_vega:.2f}"
                    })
            
            if efficiency_data:
                efficiency_df = pd.DataFrame(efficiency_data)
                st.dataframe(efficiency_df, use_container_width=True, hide_index=True)
                
                st.markdown("""
                **📝 Efficiency Notes:**
                - Higher efficiency scores indicate better risk-adjusted positioning
                - Multi-leg strategies have complexity penalties but may offer better risk profiles
                - Consider consolidating positions with low efficiency scores
                """)
        
    except ImportError:
        st.warning("⚠️ Advanced analytics require updated options_math module")
    except Exception as e:
        st.error(f"Error in advanced analytics: {e}")


if __name__ == "__main__":
    print("Greeks Dashboard module initialized")